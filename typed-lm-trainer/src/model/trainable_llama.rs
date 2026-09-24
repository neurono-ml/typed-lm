//! Differentiable full-sequence Llama forward with LoRA adapters.
//!
//! Unlike the serving forward (which reads a single decision token behind a KV
//! cache), training needs a **full-sequence, differentiable** pass: every
//! position is computed, a causal mask keeps the attention honest, and the loss
//! at the decision position back-propagates into the LoRA adapters. The base
//! projections are frozen dense weights; only the adapters are `Var`s.
//!
//! The module is built directly from a `VarMap` so the adapter variables appear
//! in [`VarMap::all_vars`] for the optimizer. Qwen2 (which biases q/k/v) lives
//! in [`crate::model::trainable_qwen2`]; this module covers Llama and, through
//! the shared `has_query_key_value_bias` flag, is also usable for architectures
//! without attention biases.

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{rotary_emb, VarBuilder, VarMap};
use typed_lm_common::model_config::ParallelModelConfig;

use crate::model::lora::LoRALinear;

/// LoRA hyper-parameters for one trainable model.
#[derive(Debug, Clone, Copy)]
pub struct LoRAConfiguration {
    /// Adapter rank.
    pub rank: usize,
    /// Adapter scaling numerator (paired with the rank).
    pub alpha: f64,
}

impl LoRAConfiguration {
    pub fn new(rank: usize, alpha: f64) -> Self {
        Self { rank, alpha }
    }
}

/// Frozen RMSNorm built from a plain tensor (not a variable).
#[derive(Debug, Clone)]
struct FrozenRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl FrozenRmsNorm {
    fn from_base(base: &Tensor, eps: f64) -> Self {
        Self {
            weight: base.clone(),
            eps,
        }
    }

    /// Differentiable RMSNorm built from primitive ops.
    ///
    /// The fused `candle_nn::ops::rms_norm` kernel has no backward pass on the
    /// CPU in candle 0.11, so the trainable forward reimplements the same
    /// formula with differentiable primitives (`sqr`, `sum_keepdim`, `sqrt`).
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden_size = input.dim(D::Minus1)?;
        let mean_square = (input.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let normalized = input.broadcast_div(&(mean_square + self.eps)?.sqrt()?)?;
        normalized.broadcast_mul(&self.weight)
    }
}

/// Numerically stable, differentiable softmax over the last dimension.
///
/// `candle_nn::ops::softmax_last_dim` is not differentiable on the CPU in candle
/// 0.11; this primitive implementation keeps the training graph alive.
fn differentiable_softmax(input: &Tensor) -> Result<Tensor> {
    let maximum = input.max_keepdim(D::Minus1)?;
    let shifted = input.broadcast_sub(&maximum)?;
    let exponentials = shifted.exp()?;
    let total = exponentials.sum_keepdim(D::Minus1)?;
    exponentials.broadcast_div(&total)
}

/// Causal mask filled with `NEG_INFINITY` above the diagonal.
fn causal_mask(sequence_length: usize, device: &Device) -> Result<Tensor> {
    let mask = candle_transformers::utils::build_causal_mask(sequence_length, 0, device)?;
    mask.to_dtype(DType::F32)
}

/// Attention block with LoRA-wrapped q/k/v/o projections.
#[derive(Debug, Clone)]
struct TrainableAttention {
    query_projection: LoRALinear,
    key_projection: LoRALinear,
    value_projection: LoRALinear,
    output_projection: LoRALinear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dimension: usize,
    cos: Tensor,
    sin: Tensor,
}

impl TrainableAttention {
    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch_size, sequence_length, hidden_size) = hidden.dims3()?;
        let query = self.query_projection.forward(hidden)?;
        let key = self.key_projection.forward(hidden)?;
        let value = self.value_projection.forward(hidden)?;

        let query = query
            .reshape((
                batch_size,
                sequence_length,
                self.num_attention_heads,
                self.head_dimension,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let key = key
            .reshape((
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dimension,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let value = value
            .reshape((
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dimension,
            ))?
            .transpose(1, 2)?
            .contiguous()?;

        let query = rotary_emb::rope(&query, &self.cos, &self.sin)?;
        let key = rotary_emb::rope(&key, &self.cos, &self.sin)?;

        let key = repeat_key_value(&key, self.num_attention_heads / self.num_key_value_heads)?;
        let value = repeat_key_value(&value, self.num_attention_heads / self.num_key_value_heads)?;

        let scale = 1.0 / (self.head_dimension as f64).sqrt();
        let attention = (query.matmul(&key.t()?)? * scale)?;
        let attention = attention.broadcast_add(mask)?;
        let attention = differentiable_softmax(&attention)?;
        let context = attention.matmul(&value.contiguous()?)?;
        let context =
            context
                .transpose(1, 2)?
                .reshape(&[batch_size, sequence_length, hidden_size])?;
        self.output_projection.forward(&context)
    }
}

/// Replicates grouped key/value heads to the query head count.
fn repeat_key_value(hidden: &Tensor, groups: usize) -> Result<Tensor> {
    if groups <= 1 {
        return Ok(hidden.clone());
    }
    let (batch_size, key_value_heads, sequence_length, head_dimension) = hidden.dims4()?;
    hidden
        .unsqueeze(2)?
        .expand((
            batch_size,
            key_value_heads,
            groups,
            sequence_length,
            head_dimension,
        ))?
        .reshape((
            batch_size,
            key_value_heads * groups,
            sequence_length,
            head_dimension,
        ))
}

/// SwiGLU MLP with LoRA-wrapped gate/up/down projections.
#[derive(Debug, Clone)]
struct TrainableMlp {
    gate_projection: LoRALinear,
    up_projection: LoRALinear,
    down_projection: LoRALinear,
}

impl TrainableMlp {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_projection.forward(hidden)?)?;
        let up = self.up_projection.forward(hidden)?;
        self.down_projection.forward(&(gate * up)?)
    }
}

/// One decoder block: input norm, attention, post-attention norm, MLP.
#[derive(Debug, Clone)]
struct TrainableBlock {
    input_norm: FrozenRmsNorm,
    attention: TrainableAttention,
    post_attention_norm: FrozenRmsNorm,
    mlp: TrainableMlp,
}

impl TrainableBlock {
    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = hidden;
        let normalized = self.input_norm.forward(hidden)?;
        let attended = self.attention.forward(&normalized, mask)?;
        let hidden = (attended + residual)?;
        let residual = &hidden;
        let normalized = self.post_attention_norm.forward(&hidden)?;
        self.mlp.forward(&normalized)? + residual
    }
}

/// A trainable, LoRA-augmented Llama model over a frozen base.
#[derive(Debug, Clone)]
pub struct TrainableLlama {
    token_embedding: Tensor,
    blocks: Vec<TrainableBlock>,
    final_norm: FrozenRmsNorm,
    language_model_head: Tensor,
    tie_word_embeddings: bool,
    causal_mask: Tensor,
    sequence_capacity: usize,
}

impl TrainableLlama {
    /// Builds the model from a base tensor map and a `VarMap` for the adapters.
    ///
    /// Every base weight is read from `base`; the adapter variables are created
    /// in `variable_map` under `model.layers.{index}...lora_a/lora_b`.
    pub fn load(
        base: &HashMap<String, Tensor>,
        config: &ParallelModelConfig,
        lora: LoRAConfiguration,
        variable_map: &mut VarMap,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let base_builder = VarBuilder::from_tensors(base.clone(), DType::F32, device);
        let adapter_builder = VarBuilder::from_varmap(variable_map, DType::F32, device);

        let token_embedding = base_builder
            .get(
                (config.vocab_size, config.hidden_size),
                "model.embed_tokens.weight",
            )?
            .to_dtype(DType::F32)?;
        let language_model_head = if config.tie_word_embeddings {
            token_embedding.clone()
        } else {
            base_builder
                .get((config.vocab_size, config.hidden_size), "lm_head.weight")?
                .to_dtype(DType::F32)?
        };
        let final_norm = FrozenRmsNorm::from_base(
            &base_builder.get(config.hidden_size, "model.norm.weight")?,
            config.rms_norm_eps,
        );

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer_prefix = format!("model.layers.{index}");
            let block = Self::load_block(
                &base_builder,
                &adapter_builder,
                config,
                lora,
                &layer_prefix,
                device,
            )?;
            blocks.push(block);
        }

        let (cos, sin) = build_rotary_tables(config, device)?;
        for block in &mut blocks {
            block.attention.cos = cos.clone();
            block.attention.sin = sin.clone();
        }

        Ok(Self {
            token_embedding,
            blocks,
            final_norm,
            language_model_head,
            tie_word_embeddings: config.tie_word_embeddings,
            causal_mask: causal_mask(config.max_position_embeddings.min(2048), device)?,
            sequence_capacity: config.max_position_embeddings.min(2048),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn load_block(
        base_builder: &VarBuilder,
        adapter_builder: &VarBuilder,
        config: &ParallelModelConfig,
        lora: LoRAConfiguration,
        layer_prefix: &str,
        device: &Device,
    ) -> anyhow::Result<TrainableBlock> {
        let input_norm = FrozenRmsNorm::from_base(
            &base_builder.get(
                config.hidden_size,
                &format!("{layer_prefix}.input_layernorm.weight"),
            )?,
            config.rms_norm_eps,
        );
        let post_attention_norm = FrozenRmsNorm::from_base(
            &base_builder.get(
                config.hidden_size,
                &format!("{layer_prefix}.post_attention_layernorm.weight"),
            )?,
            config.rms_norm_eps,
        );
        let head_dimension = config.head_dimension();
        let query_size = head_dimension * config.num_attention_heads;
        let key_value_size = head_dimension * config.num_key_value_heads;

        let attention_prefix = format!("{layer_prefix}.self_attn");
        let attention = TrainableAttention {
            query_projection: build_lora(
                base_builder,
                adapter_builder,
                &format!("{attention_prefix}.q_proj"),
                query_size,
                config.hidden_size,
                config.has_query_key_value_bias(),
                lora,
            )?,
            key_projection: build_lora(
                base_builder,
                adapter_builder,
                &format!("{attention_prefix}.k_proj"),
                key_value_size,
                config.hidden_size,
                config.has_query_key_value_bias(),
                lora,
            )?,
            value_projection: build_lora(
                base_builder,
                adapter_builder,
                &format!("{attention_prefix}.v_proj"),
                key_value_size,
                config.hidden_size,
                config.has_query_key_value_bias(),
                lora,
            )?,
            output_projection: build_lora(
                base_builder,
                adapter_builder,
                &format!("{attention_prefix}.o_proj"),
                config.hidden_size,
                query_size,
                false,
                lora,
            )?,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dimension,
            cos: Tensor::zeros((1, 1), DType::F32, device)?,
            sin: Tensor::zeros((1, 1), DType::F32, device)?,
        };

        let mlp_prefix = format!("{layer_prefix}.mlp");
        let mlp = TrainableMlp {
            gate_projection: build_lora(
                base_builder,
                adapter_builder,
                &format!("{mlp_prefix}.gate_proj"),
                config.intermediate_size,
                config.hidden_size,
                false,
                lora,
            )?,
            up_projection: build_lora(
                base_builder,
                adapter_builder,
                &format!("{mlp_prefix}.up_proj"),
                config.intermediate_size,
                config.hidden_size,
                false,
                lora,
            )?,
            down_projection: build_lora(
                base_builder,
                adapter_builder,
                &format!("{mlp_prefix}.down_proj"),
                config.hidden_size,
                config.intermediate_size,
                false,
                lora,
            )?,
        };

        Ok(TrainableBlock {
            input_norm,
            attention,
            post_attention_norm,
            mlp,
        })
    }

    /// Full-sequence forward returning logits `(batch, sequence, vocabulary)`.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, sequence_length) = token_ids.dims2()?;
        if sequence_length > self.sequence_capacity {
            return Err(candle_core::Error::Msg(format!(
                "sequence length {sequence_length} exceeds the capacity {}",
                self.sequence_capacity
            )));
        }
        let mask = self
            .causal_mask
            .narrow(0, 0, sequence_length)?
            .narrow(1, 0, sequence_length)?
            .broadcast_as((batch_size, 1, sequence_length, sequence_length))?;

        let hidden_size = self.token_embedding.dim(1)?;
        let flat_tokens = token_ids.flatten_all()?;
        let mut hidden = self
            .token_embedding
            .index_select(&flat_tokens, 0)?
            .to_dtype(DType::F32)?
            .reshape((batch_size, sequence_length, hidden_size))?;

        for block in &self.blocks {
            hidden = block.forward(&hidden, &mask)?;
        }
        let hidden = self.final_norm.forward(&hidden)?;
        let (batch_size, sequence_length, hidden_size) = hidden.dims3()?;
        let head_weight = if self.tie_word_embeddings {
            &self.token_embedding
        } else {
            &self.language_model_head
        };
        let logits = hidden
            .reshape((batch_size * sequence_length, hidden_size))?
            .matmul(&head_weight.t()?)?
            .reshape((batch_size, sequence_length, head_weight.dim(0)?))?;
        logits.to_dtype(DType::F32)
    }

    /// Logits of one row at its decision position, shaped `(vocabulary,)`.
    pub fn decision_logits(
        &self,
        token_ids: &Tensor,
        decision_positions: &[usize],
    ) -> Result<Tensor> {
        let logits = self.forward(token_ids)?;
        let mut rows = Vec::with_capacity(decision_positions.len());
        for (row, position) in decision_positions.iter().enumerate() {
            rows.push(logits.i((row, *position, ..))?);
        }
        Tensor::stack(&rows, 0)
    }

    /// The vocabulary dimension of the output logits.
    pub fn vocabulary_size(&self) -> usize {
        self.token_embedding.dim(0).unwrap_or(0)
    }

    /// Every trainable adapter variable, in a deterministic order.
    pub fn adapter_variables(&self) -> Vec<candle_core::Var> {
        let mut variables: Vec<candle_core::Var> = Vec::new();
        for block in &self.blocks {
            for projection in [
                &block.attention.query_projection,
                &block.attention.key_projection,
                &block.attention.value_projection,
                &block.attention.output_projection,
                &block.mlp.gate_projection,
                &block.mlp.up_projection,
                &block.mlp.down_projection,
            ] {
                variables.push(projection.down_projection().clone());
                variables.push(projection.up_projection().clone());
            }
        }
        variables
    }
}

impl crate::training::r#loop::TrainableModel for TrainableLlama {
    fn variables(&self) -> Vec<candle_core::Var> {
        self.adapter_variables()
    }

    fn forward(&self, batch: &crate::dataset::collate::TrainingBatch) -> anyhow::Result<Tensor> {
        Ok(TrainableLlama::forward(self, &batch.input_ids)?)
    }
}

/// Builds a LoRA-wrapped projection from the frozen base weight.
#[allow(clippy::too_many_arguments)]
fn build_lora(
    base_builder: &VarBuilder,
    adapter_builder: &VarBuilder,
    prefix: &str,
    output_features: usize,
    input_features: usize,
    with_bias: bool,
    lora: LoRAConfiguration,
) -> anyhow::Result<LoRALinear> {
    let base_weight = base_builder
        .get(
            (output_features, input_features),
            &format!("{prefix}.weight"),
        )?
        .to_dtype(DType::F32)?;
    let base_bias = if with_bias {
        Some(
            base_builder
                .get(output_features, &format!("{prefix}.bias"))?
                .to_dtype(DType::F32)?,
        )
    } else {
        None
    };
    LoRALinear::new(
        base_weight,
        base_bias,
        lora.rank,
        lora.alpha,
        adapter_builder,
        prefix,
    )
}

/// Builds the RoPE cosine/sine tables for the configured positions.
fn build_rotary_tables(
    config: &ParallelModelConfig,
    device: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let head_dimension = config.head_dimension();
    let positions = config.max_position_embeddings.min(2048);
    let inverse_frequencies: Vec<f32> = (0..head_dimension)
        .step_by(2)
        .map(|index| 1f32 / config.rope_theta.powf(index as f32 / head_dimension as f32))
        .collect();
    let theta = Tensor::new(inverse_frequencies.as_slice(), device)?;
    let index_theta = Tensor::arange(0, positions as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((positions, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((index_theta.cos()?, index_theta.sin()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_config() -> ParallelModelConfig {
        ParallelModelConfig {
            architecture: typed_lm_common::checkpoint::ModelArchitecture::Llama,
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            rope_scaling: None,
            attention_bias: false,
            explicit_head_dimension: None,
            sliding_window: None,
            max_window_layers: 0,
            logit_softcapping: None,
            attention_logit_softcapping: None,
            query_pre_attention_scalar: None,
            rms_norm_unit_offset: false,
            embedding_scale: None,
            rope_local_base_frequency: None,
        }
    }

    fn deterministic_tensor(shape: (usize, usize), scale: f32) -> anyhow::Result<Tensor> {
        let device = Device::Cpu;
        let count = shape.0 * shape.1;
        let mut seed = 0x1234_5678_u64;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            values.push((((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * scale);
        }
        Ok(Tensor::from_vec(values, shape, &device)?)
    }

    fn deterministic_vector(size: usize) -> anyhow::Result<Tensor> {
        let device = Device::Cpu;
        let mut seed = 0x9abc_def0_u64;
        let mut values = Vec::with_capacity(size);
        for _ in 0..size {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            values.push((((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 0.1 + 1.0);
        }
        Ok(Tensor::from_vec(values, size, &device)?)
    }

    fn base_tensors(config: &ParallelModelConfig) -> anyhow::Result<HashMap<String, Tensor>> {
        let head_dimension = config.head_dimension();
        let query_size = head_dimension * config.num_attention_heads;
        let key_value_size = head_dimension * config.num_key_value_heads;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            deterministic_tensor((config.vocab_size, config.hidden_size), 0.1)?,
        );
        tensors.insert(
            "lm_head.weight".to_string(),
            deterministic_tensor((config.vocab_size, config.hidden_size), 0.1)?,
        );
        tensors.insert(
            "model.norm.weight".to_string(),
            deterministic_vector(config.hidden_size)?,
        );
        for index in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{index}");
            tensors.insert(
                format!("{prefix}.input_layernorm.weight"),
                deterministic_vector(config.hidden_size)?,
            );
            tensors.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                deterministic_vector(config.hidden_size)?,
            );
            for (projection, output) in [
                ("self_attn.q_proj", query_size),
                ("self_attn.k_proj", key_value_size),
                ("self_attn.v_proj", key_value_size),
                ("self_attn.o_proj", config.hidden_size),
                ("mlp.gate_proj", config.intermediate_size),
                ("mlp.up_proj", config.intermediate_size),
                ("mlp.down_proj", config.hidden_size),
            ] {
                let input = if projection == "self_attn.o_proj" {
                    query_size
                } else if projection.starts_with("mlp.") && projection != "mlp.down_proj" {
                    config.hidden_size
                } else if projection == "mlp.down_proj" {
                    config.intermediate_size
                } else {
                    config.hidden_size
                };
                tensors.insert(
                    format!("{prefix}.{projection}.weight"),
                    deterministic_tensor((output, input), 0.1)?,
                );
            }
        }
        Ok(tensors)
    }

    #[test]
    fn forward_produces_finite_logits() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let config = tiny_config();
        let base = base_tensors(&config)?;
        let mut variable_map = VarMap::new();
        let model = TrainableLlama::load(
            &base,
            &config,
            LoRAConfiguration::new(2, 4.0),
            &mut variable_map,
            &device,
        )?;
        let tokens = Tensor::new(&[[1_u32, 2, 3, 4]], &device)?;
        let logits = model.forward(&tokens)?;
        assert_eq!(logits.dims(), &[1, 4, config.vocab_size]);
        for value in logits.flatten_all()?.to_vec1::<f32>()? {
            assert!(value.is_finite(), "logits must be finite, got {value}");
        }
        Ok(())
    }

    #[test]
    fn backward_produces_adapter_gradients() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let config = tiny_config();
        let base = base_tensors(&config)?;
        let mut variable_map = VarMap::new();
        let model = TrainableLlama::load(
            &base,
            &config,
            LoRAConfiguration::new(2, 4.0),
            &mut variable_map,
            &device,
        )?;
        let tokens = Tensor::new(&[[1_u32, 2, 3]], &device)?;
        let decision_positions = vec![2_usize];
        let logits = model.decision_logits(&tokens, &decision_positions)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &Tensor::new(&[5_u32], &device)?)?;
        let gradients = loss.backward()?;
        let variables = variable_map.all_vars();
        assert!(!variables.is_empty(), "the adapter must expose variables");
        let mut found_gradient = false;
        for variable in &variables {
            if let Some(gradient) = gradients.get(variable) {
                found_gradient = true;
                let values = gradient.flatten_all()?.to_vec1::<f32>()?;
                assert!(values.iter().all(|value| value.is_finite()));
            }
        }
        assert!(found_gradient, "at least one adapter gradient is expected");
        Ok(())
    }

    #[test]
    fn frozen_base_weights_are_not_variables() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let config = tiny_config();
        let base = base_tensors(&config)?;
        let mut variable_map = VarMap::new();
        let model = TrainableLlama::load(
            &base,
            &config,
            LoRAConfiguration::new(2, 4.0),
            &mut variable_map,
            &device,
        )?;
        assert!(!model.token_embedding.is_variable());
        assert!(!model.language_model_head.is_variable());
        for variable in variable_map.all_vars() {
            assert!(variable.dtype().is_float());
        }
        Ok(())
    }

    #[test]
    fn a_sequence_longer_than_capacity_is_an_error() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut config = tiny_config();
        config.max_position_embeddings = 4;
        let base = base_tensors(&config)?;
        let mut variable_map = VarMap::new();
        let model = TrainableLlama::load(
            &base,
            &config,
            LoRAConfiguration::new(2, 4.0),
            &mut variable_map,
            &device,
        )?;
        let tokens = Tensor::new(&[[1_u32, 2, 3, 4, 5, 6]], &device)?;
        assert!(model.forward(&tokens).is_err());
        Ok(())
    }

    /// Wave 4 acceptance: a tiny dummy model overfits a single example.
    ///
    /// No weights are downloaded; the base is generated deterministically. A
    /// plain SGD update on the adapters must reduce the decision-position loss
    /// within a handful of steps, proving the full-sequence forward is
    /// differentiable end to end.
    #[test]
    fn overfits_a_single_example_with_plain_sgd() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let config = tiny_config();
        let base = base_tensors(&config)?;
        let mut variable_map = VarMap::new();
        let model = TrainableLlama::load(
            &base,
            &config,
            LoRAConfiguration::new(4, 8.0),
            &mut variable_map,
            &device,
        )?;
        let tokens = Tensor::new(&[[1_u32, 2, 3, 4]], &device)?;
        let decision_positions = vec![3_usize];
        let target = Tensor::new(&[7_u32], &device)?;
        let variables = variable_map.all_vars();
        assert!(!variables.is_empty());

        let first_loss = {
            let logits = model.decision_logits(&tokens, &decision_positions)?;
            candle_nn::loss::cross_entropy(&logits, &target)?.to_vec0::<f32>()?
        };

        let learning_rate = 0.5_f64;
        let mut last_loss = first_loss;
        for _ in 0..20 {
            let logits = model.decision_logits(&tokens, &decision_positions)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &target)?;
            last_loss = loss.to_vec0::<f32>()?;
            let gradients = loss.backward()?;
            for variable in &variables {
                if let Some(gradient) = gradients.get(variable) {
                    let update = (gradient * learning_rate)?;
                    variable.set(&(variable.as_tensor() - update)?)?;
                }
            }
        }

        assert!(
            last_loss < first_loss,
            "the loss must fall: first {first_loss}, last {last_loss}"
        );
        Ok(())
    }

    /// Wave 5 end-to-end: the generic training loop overfits a dummy batch.
    #[test]
    fn training_loop_reduces_the_loss_on_a_real_trainable_model() -> anyhow::Result<()> {
        use crate::dataset::collate::TrainingBatch;
        use crate::training::r#loop::{train, TrainableModel, TrainingLoopConfiguration};

        let device = Device::Cpu;
        let config = tiny_config();
        let base = base_tensors(&config)?;
        let mut variable_map = VarMap::new();
        let model = TrainableLlama::load(
            &base,
            &config,
            LoRAConfiguration::new(4, 8.0),
            &mut variable_map,
            &device,
        )?;
        assert!(!model.variables().is_empty());

        let batch = TrainingBatch {
            input_ids: Tensor::new(&[[1_u32, 2, 3, 4]], &device)?,
            decision_mask: Tensor::new(&[[0_u32, 0, 0, 1]], &device)?,
            decision_positions: Tensor::new(&[3_u32], &device)?,
            label_token_ids: Tensor::new(&[7_u32], &device)?,
            sequence_length: 4,
        };
        let configuration = TrainingLoopConfiguration {
            epochs: 40,
            learning_rate: 0.5,
            warmup_steps: 2,
            maximum_gradient_norm: 10.0,
            ..TrainingLoopConfiguration::default()
        };
        let outcome = train(&model, &[batch], &configuration, &device)?;
        let first = outcome
            .epochs
            .first()
            .ok_or_else(|| anyhow::anyhow!("no metrics recorded"))?;
        let last = outcome
            .epochs
            .last()
            .ok_or_else(|| anyhow::anyhow!("no metrics recorded"))?;
        assert!(
            last.mean_loss < first.mean_loss,
            "loop loss must fall: first {}, last {}",
            first.mean_loss,
            last.mean_loss
        );
        Ok(())
    }
}
