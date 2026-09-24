//! Fully-trainable dense model for `--method full` and `--method from-scratch`.
//!
//! Unlike [`crate::model::trainable_llama::TrainableLlama`] (which freezes the
//! base and trains only a LoRA adapter), this model holds **every** weight as a
//! `Var`: embeddings, norms and all projections. It is the training target of
//! `full` (initialized from a checkpoint) and `from-scratch` (initialized
//! randomly). The forward pass mirrors the vendored dense forward so the
//! from-scratch artifact is served identically.
//!
//! The per-family variations (attention biases, explicit head dimension,
//! sliding window, logit soft-capping, RMSNorm unit offset, embedding scale and
//! local RoPE) are driven by [`ParallelModelConfig`], exactly like the serving
//! path.

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, Var, D};
use candle_nn::{rotary_emb, VarBuilder, VarMap};
use typed_lm_common::model_config::ParallelModelConfig;

use crate::dataset::collate::TrainingBatch;
use crate::error::TrainerError;
use crate::model::trainable_linear::TrainableLinear;
use crate::model::trainable_rms_norm::TrainableRmsNorm;

/// Numerically stable, differentiable softmax over the last dimension.
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

/// Fully-trainable attention block.
#[derive(Debug, Clone)]
struct FullAttention {
    query_projection: TrainableLinear,
    key_projection: TrainableLinear,
    value_projection: TrainableLinear,
    output_projection: TrainableLinear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dimension: usize,
    attention_scale: f64,
    attention_logit_softcapping: Option<f64>,
    cos: Tensor,
    sin: Tensor,
}

impl FullAttention {
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

        let mut attention = (query.matmul(&key.t()?)? * self.attention_scale)?;
        if let Some(cap) = self.attention_logit_softcapping {
            attention = ((attention / cap)?.tanh()? * cap)?;
        }
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

/// Fully-trainable SwiGLU MLP.
#[derive(Debug, Clone)]
struct FullMlp {
    gate_projection: TrainableLinear,
    up_projection: TrainableLinear,
    down_projection: TrainableLinear,
}

impl FullMlp {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_projection.forward(hidden)?)?;
        let up = self.up_projection.forward(hidden)?;
        self.down_projection.forward(&(gate * up)?)
    }
}

/// One fully-trainable decoder block.
#[derive(Debug, Clone)]
struct FullBlock {
    input_norm: TrainableRmsNorm,
    attention: FullAttention,
    post_attention_norm: TrainableRmsNorm,
    mlp: FullMlp,
}

impl FullBlock {
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

/// A dense model whose weights are all trainable `Var`s.
#[derive(Debug, Clone)]
pub struct TrainableFull {
    token_embedding: Var,
    blocks: Vec<FullBlock>,
    final_norm: TrainableRmsNorm,
    language_model_head: Var,
    tie_word_embeddings: bool,
    embedding_scale: Option<f64>,
    logit_softcapping: Option<f64>,
    causal_mask: Tensor,
    sequence_capacity: usize,
}

impl TrainableFull {
    /// Builds a fully-trainable model, creating every weight in `variable_map`.
    pub fn load(
        config: &ParallelModelConfig,
        variable_map: &mut VarMap,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let builder = VarBuilder::from_varmap(variable_map, DType::F32, device);
        Self::build_base(&builder, config, device)
    }

    /// Builds a fully-trainable model from an existing base tensor map.
    ///
    /// This is the `--method full` path: every base weight becomes a `Var`
    /// initialized from the checkpoint, so training can update all parameters.
    pub fn from_base(
        base: &HashMap<String, Tensor>,
        config: &ParallelModelConfig,
        variable_map: &mut VarMap,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let builder = VarBuilder::from_varmap(variable_map, DType::F32, device);
        Self::build_base_with_source(Some(base), &builder, config, device)
    }

    /// Builds a fully-trainable model from randomly initialized weights.
    ///
    /// This is the `--method from-scratch` path: the caller supplies the
    /// canonical tensor map produced by
    /// [`crate::model::initialization::initialize_model_tensors`].
    pub fn from_initialized(
        tensors: &HashMap<String, Tensor>,
        config: &ParallelModelConfig,
        variable_map: &mut VarMap,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let builder = VarBuilder::from_varmap(variable_map, DType::F32, device);
        Self::build_base_with_source(Some(tensors), &builder, config, device)
    }

    fn build_base(
        builder: &VarBuilder,
        config: &ParallelModelConfig,
        device: &Device,
    ) -> anyhow::Result<Self> {
        Self::build_base_with_source(None, builder, config, device)
    }

    #[allow(clippy::too_many_lines)]
    fn build_base_with_source(
        source: Option<&HashMap<String, Tensor>>,
        builder: &VarBuilder,
        config: &ParallelModelConfig,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let head_dimension = config.head_dimension();
        let query_size = head_dimension * config.num_attention_heads;
        let key_value_size = head_dimension * config.num_key_value_heads;
        let with_bias = config.has_query_key_value_bias();

        let token_embedding = create_variable(
            source,
            builder,
            "model.embed_tokens.weight",
            (config.vocab_size, config.hidden_size),
            config,
        )?;
        let language_model_head = if config.tie_word_embeddings {
            token_embedding.clone()
        } else {
            create_variable(
                source,
                builder,
                "lm_head.weight",
                (config.vocab_size, config.hidden_size),
                config,
            )?
        };
        let final_norm = build_norm(
            source,
            builder,
            "model.norm",
            config.hidden_size,
            config,
        )?;

        let attention_scale = match config.query_pre_attention_scalar {
            Some(scalar) => 1.0 / (scalar as f64).sqrt(),
            None => 1.0 / (head_dimension as f64).sqrt(),
        };
        let (cos, sin) = build_rotary_tables(config, device)?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer_prefix = format!("model.layers.{index}");
            let input_norm = build_norm(
                source,
                builder,
                &format!("{layer_prefix}.input_layernorm"),
                config.hidden_size,
                config,
            )?;
            let post_attention_norm = build_norm(
                source,
                builder,
                &format!("{layer_prefix}.post_attention_layernorm"),
                config.hidden_size,
                config,
            )?;
            let attention_prefix = format!("{layer_prefix}.self_attn");
            let attention = FullAttention {
                query_projection: build_linear(
                    source,
                    builder,
                    &format!("{attention_prefix}.q_proj"),
                    query_size,
                    config.hidden_size,
                    with_bias,
                    config,
                )?,
                key_projection: build_linear(
                    source,
                    builder,
                    &format!("{attention_prefix}.k_proj"),
                    key_value_size,
                    config.hidden_size,
                    with_bias,
                    config,
                )?,
                value_projection: build_linear(
                    source,
                    builder,
                    &format!("{attention_prefix}.v_proj"),
                    key_value_size,
                    config.hidden_size,
                    with_bias,
                    config,
                )?,
                output_projection: build_linear(
                    source,
                    builder,
                    &format!("{attention_prefix}.o_proj"),
                    config.hidden_size,
                    query_size,
                    false,
                    config,
                )?,
                num_attention_heads: config.num_attention_heads,
                num_key_value_heads: config.num_key_value_heads,
                head_dimension,
                attention_scale,
                attention_logit_softcapping: config.attention_logit_softcapping,
                cos: cos.clone(),
                sin: sin.clone(),
            };
            let mlp_prefix = format!("{layer_prefix}.mlp");
            let mlp = FullMlp {
                gate_projection: build_linear(
                    source,
                    builder,
                    &format!("{mlp_prefix}.gate_proj"),
                    config.intermediate_size,
                    config.hidden_size,
                    false,
                    config,
                )?,
                up_projection: build_linear(
                    source,
                    builder,
                    &format!("{mlp_prefix}.up_proj"),
                    config.intermediate_size,
                    config.hidden_size,
                    false,
                    config,
                )?,
                down_projection: build_linear(
                    source,
                    builder,
                    &format!("{mlp_prefix}.down_proj"),
                    config.hidden_size,
                    config.intermediate_size,
                    false,
                    config,
                )?,
            };
            blocks.push(FullBlock {
                input_norm,
                attention,
                post_attention_norm,
                mlp,
            });
        }

        let sequence_capacity = config.max_position_embeddings.min(2048);
        Ok(Self {
            token_embedding,
            blocks,
            final_norm,
            language_model_head,
            tie_word_embeddings: config.tie_word_embeddings,
            embedding_scale: config.embedding_scale,
            logit_softcapping: config.logit_softcapping,
            causal_mask: causal_mask(sequence_capacity, device)?,
            sequence_capacity,
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
            .as_tensor()
            .index_select(&flat_tokens, 0)?
            .to_dtype(DType::F32)?
            .reshape((batch_size, sequence_length, hidden_size))?;
        if let Some(scale) = self.embedding_scale {
            hidden = (hidden * scale)?;
        }

        for block in &self.blocks {
            hidden = block.forward(&hidden, &mask)?;
        }
        let hidden = self.final_norm.forward(&hidden)?;
        let (batch_size, sequence_length, hidden_size) = hidden.dims3()?;
        let head_weight = if self.tie_word_embeddings {
            self.token_embedding.as_tensor().clone()
        } else {
            self.language_model_head.as_tensor().clone()
        };
        let mut logits = hidden
            .reshape((batch_size * sequence_length, hidden_size))?
            .matmul(&head_weight.t()?)?
            .reshape((batch_size, sequence_length, head_weight.dim(0)?))?
            .to_dtype(DType::F32)?;
        if let Some(cap) = self.logit_softcapping {
            logits = ((logits / cap)?.tanh()? * cap)?;
        }
        Ok(logits)
    }

    /// Logits of one row at its decision position, shaped `(rows, vocabulary)`.
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

    /// Every trainable variable, in a deterministic order.
    pub fn all_trainable_variables(&self) -> Vec<Var> {
        let mut variables: Vec<Var> = Vec::new();
        variables.push(self.token_embedding.clone());
        for block in &self.blocks {
            variables.extend(block.input_norm.variables());
            variables.extend(block.attention.query_projection.variables());
            variables.extend(block.attention.key_projection.variables());
            variables.extend(block.attention.value_projection.variables());
            variables.extend(block.attention.output_projection.variables());
            variables.extend(block.post_attention_norm.variables());
            variables.extend(block.mlp.gate_projection.variables());
            variables.extend(block.mlp.up_projection.variables());
            variables.extend(block.mlp.down_projection.variables());
        }
        variables.extend(self.final_norm.variables());
        if !self.tie_word_embeddings {
            variables.push(self.language_model_head.clone());
        }
        variables
    }

    /// Collects every trainable parameter into a canonical-name tensor map.
    ///
    /// Used to persist a complete checkpoint after training; `lora`-style
    /// adapters are absent because every weight is a full parameter here.
    pub fn state_dict(&self) -> anyhow::Result<HashMap<String, Tensor>> {
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            self.token_embedding.as_tensor().clone(),
        );
        tensors.insert(
            "model.norm.weight".to_string(),
            self.final_norm.weight().as_tensor().clone(),
        );
        if !self.tie_word_embeddings {
            tensors.insert(
                "lm_head.weight".to_string(),
                self.language_model_head.as_tensor().clone(),
            );
        }
        for (index, block) in self.blocks.iter().enumerate() {
            let prefix = format!("model.layers.{index}");
            insert_variable(
                &mut tensors,
                &format!("{prefix}.input_layernorm.weight"),
                block.input_norm.weight(),
            );
            insert_variable(
                &mut tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
                block.post_attention_norm.weight(),
            );
            insert_linear(&mut tensors, &format!("{prefix}.self_attn.q_proj"), &block.attention.query_projection);
            insert_linear(&mut tensors, &format!("{prefix}.self_attn.k_proj"), &block.attention.key_projection);
            insert_linear(&mut tensors, &format!("{prefix}.self_attn.v_proj"), &block.attention.value_projection);
            insert_linear(&mut tensors, &format!("{prefix}.self_attn.o_proj"), &block.attention.output_projection);
            insert_linear(&mut tensors, &format!("{prefix}.mlp.gate_proj"), &block.mlp.gate_projection);
            insert_linear(&mut tensors, &format!("{prefix}.mlp.up_proj"), &block.mlp.up_projection);
            insert_linear(&mut tensors, &format!("{prefix}.mlp.down_proj"), &block.mlp.down_projection);
        }
        Ok(tensors)
    }
}

impl crate::training::r#loop::TrainableModel for TrainableFull {
    fn variables(&self) -> Vec<Var> {
        self.all_trainable_variables()
    }

    fn forward(&self, batch: &TrainingBatch) -> anyhow::Result<Tensor> {
        Ok(TrainableFull::forward(self, &batch.input_ids)?)
    }
}

/// Inserts one variable's tensor under a canonical name.
fn insert_variable(tensors: &mut HashMap<String, Tensor>, name: &str, variable: &Var) {
    tensors.insert(name.to_string(), variable.as_tensor().clone());
}

/// Inserts a linear layer's weight (and optional bias) under canonical names.
fn insert_linear(tensors: &mut HashMap<String, Tensor>, prefix: &str, linear: &TrainableLinear) {
    insert_variable(tensors, &format!("{prefix}.weight"), linear.weight());
    if let Some(bias) = linear.bias() {
        insert_variable(tensors, &format!("{prefix}.bias"), bias);
    }
}

/// Builds a trainable linear, copying the source weight/bias when provided.
#[allow(clippy::too_many_arguments)]
fn build_linear(
    source: Option<&HashMap<String, Tensor>>,
    builder: &VarBuilder,
    prefix: &str,
    output_features: usize,
    input_features: usize,
    with_bias: bool,
    config: &ParallelModelConfig,
) -> anyhow::Result<TrainableLinear> {
    if let Some(source) = source {
        let weight = source
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| {
                TrainerError::Model(format!("missing base tensor '{prefix}.weight'"))
            })?
            .to_dtype(DType::F32)?;
        let bias = if with_bias {
            Some(
                source
                    .get(&format!("{prefix}.bias"))
                    .ok_or_else(|| {
                        TrainerError::Model(format!("missing base tensor '{prefix}.bias'"))
                    })?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };
        TrainableLinear::from_tensors(weight, bias, builder, prefix)
    } else {
        let _ = config;
        TrainableLinear::new(output_features, input_features, with_bias, builder, prefix)
    }
}

/// Builds a trainable RMSNorm, copying the source weight when provided.
fn build_norm(
    source: Option<&HashMap<String, Tensor>>,
    builder: &VarBuilder,
    prefix: &str,
    hidden_size: usize,
    config: &ParallelModelConfig,
) -> anyhow::Result<TrainableRmsNorm> {
    if let Some(source) = source {
        let weight = source
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| TrainerError::Model(format!("missing base tensor '{prefix}.weight'")))?
            .to_dtype(DType::F32)?;
        TrainableRmsNorm::from_tensor(
            weight,
            config.rms_norm_eps,
            config.rms_norm_unit_offset,
            builder,
            prefix,
        )
    } else {
        TrainableRmsNorm::new(
            hidden_size,
            config.rms_norm_eps,
            config.rms_norm_unit_offset,
            builder,
            prefix,
        )
    }
}

/// Creates a variable from the source map when provided, else randomly.
fn create_variable(
    source: Option<&HashMap<String, Tensor>>,
    builder: &VarBuilder,
    name: &str,
    shape: (usize, usize),
    config: &ParallelModelConfig,
) -> anyhow::Result<Var> {
    let _ = config;
    if let Some(source) = source {
        let tensor = source
            .get(name)
            .ok_or_else(|| TrainerError::Model(format!("missing base tensor '{name}'")))?
            .to_dtype(DType::F32)?;
        let created = builder.get_with_hints(shape, name, candle_nn::Init::Const(0.0))?;
        let variable = Var::from_tensor(&created)?;
        variable.set(&tensor)?;
        Ok(variable)
    } else {
        Ok(Var::from_tensor(&builder.get_with_hints(
            shape,
            name,
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?)?)
    }
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
    use crate::model::initialization::{
        initialize_model_tensors, InitializationConfiguration,
    };
    use typed_lm_common::architecture_traits::DenseArchitectureTraits;
    use typed_lm_common::checkpoint::ModelArchitecture;

    fn tiny_config(architecture: ModelArchitecture) -> ParallelModelConfig {
        ParallelModelConfig {
            architecture,
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
            attention_bias: architecture.has_query_key_value_bias(),
            explicit_head_dimension: DenseArchitectureTraits::for_architecture(architecture)
                .explicit_head_dimension
                .then_some(4),
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

    #[test]
    fn forward_produces_finite_logits() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let config = tiny_config(ModelArchitecture::Llama);
        let mut variable_map = VarMap::new();
        let model = TrainableFull::load(&config, &mut variable_map, &device)?;
        let tokens = Tensor::new(&[[1_u32, 2, 3, 4]], &device)?;
        let logits = model.forward(&tokens)?;
        assert_eq!(logits.dims(), &[1, 4, config.vocab_size]);
        for value in logits.flatten_all()?.to_vec1::<f32>()? {
            assert!(value.is_finite());
        }
        Ok(())
    }

    #[test]
    fn every_parameter_is_trainable() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let config = tiny_config(ModelArchitecture::Llama);
        let mut variable_map = VarMap::new();
        let model = TrainableFull::load(&config, &mut variable_map, &device)?;
        // 2 layers x (2 norms + 4 attention weights + 3 mlp weights) + embed + final norm + lm_head.
        let expected = 2 * (2 + 4 + 3) + 2 + 1;
        assert_eq!(model.all_trainable_variables().len(), expected);
        assert_eq!(variable_map.all_vars().len(), expected);
        Ok(())
    }

    #[test]
    fn from_initialized_copies_the_tensors() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let config = tiny_config(ModelArchitecture::Llama);
        let tensors = initialize_model_tensors(
            &config,
            &InitializationConfiguration::default(),
            7,
            &device,
        )?;
        let mut variable_map = VarMap::new();
        let model = TrainableFull::from_initialized(&tensors, &config, &mut variable_map, &device)?;
        let state = model.state_dict()?;
        assert_eq!(state.len(), tensors.len());
        let source = tensors
            .get("model.embed_tokens.weight")
            .ok_or_else(|| anyhow::anyhow!("missing embedding"))?;
        let copied = state
            .get("model.embed_tokens.weight")
            .ok_or_else(|| anyhow::anyhow!("missing copied embedding"))?;
        let source_values = source.flatten_all()?.to_vec1::<f32>()?;
        let copied_values = copied.flatten_all()?.to_vec1::<f32>()?;
        for (left, right) in source_values.iter().zip(copied_values.iter()) {
            assert!((left - right).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn state_dict_round_trips_every_canonical_name() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let config = tiny_config(ModelArchitecture::Qwen2);
        let mut variable_map = VarMap::new();
        let model = TrainableFull::load(&config, &mut variable_map, &device)?;
        let state = model.state_dict()?;
        // Qwen2 biases q/k/v, so each layer contributes three extra bias tensors.
        assert!(state.contains_key("model.embed_tokens.weight"));
        assert!(state.contains_key("model.layers.0.self_attn.q_proj.bias"));
        assert!(!state.contains_key("model.layers.0.self_attn.o_proj.bias"));
        assert!(state.contains_key("model.layers.1.mlp.down_proj.weight"));
        Ok(())
    }

    #[test]
    fn gradients_reach_every_parameter() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let config = tiny_config(ModelArchitecture::Llama);
        let mut variable_map = VarMap::new();
        let model = TrainableFull::load(&config, &mut variable_map, &device)?;
        let tokens = Tensor::new(&[[1_u32, 2, 3]], &device)?;
        let logits = model.decision_logits(&tokens, &[2_usize])?;
        let loss = candle_nn::loss::cross_entropy(&logits, &Tensor::new(&[5_u32], &device)?)?;
        let gradients = loss.backward()?;
        let variables = model.all_trainable_variables();
        let mut with_gradient = 0_usize;
        for variable in &variables {
            if let Some(gradient) = gradients.get(variable) {
                with_gradient += 1;
                assert!(gradient.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| v.is_finite()));
            }
        }
        assert!(with_gradient > 0);
        Ok(())
    }

    #[test]
    fn from_base_rejects_a_missing_tensor() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let config = tiny_config(ModelArchitecture::Llama);
        let mut variable_map = VarMap::new();
        let result = TrainableFull::from_base(&HashMap::new(), &config, &mut variable_map, &device);
        assert!(result.is_err());
        Ok(())
    }
}
