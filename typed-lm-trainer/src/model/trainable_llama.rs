//! Differentiable full-sequence Llama forward with LoRA adapters.
//!
//! Unlike the serving forward (which reads a single decision token behind a KV
//! cache), training needs a **full-sequence, differentiable** pass: every
//! position is computed, a causal mask keeps the attention honest, and the loss
//! at the decision position back-propagates into the LoRA adapters. The base
//! projections are frozen dense weights; only the adapters are `Var`s.
//!
//! The module is built directly from a `VarMap` so the adapter variables appear
//! in [`VarMap::all_vars`] for the optimizer. This module covers every dense
//! family: the per-family variations (attention biases, explicit head dimension,
//! sliding window, logit soft-capping, RMSNorm unit offset, embedding scale and a
//! local RoPE base frequency) are driven by [`ParallelModelConfig`].
//! [`crate::model::trainable_dense`] is the supported entry point that validates
//! the configuration against the detected architecture first.

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{rotary_emb, VarBuilder, VarMap};
use typed_lm_common::checkpoint::ModelArchitecture;
use typed_lm_common::model_config::ParallelModelConfig;

use crate::model::lora::LoRALinear;

/// Gemma3 default `sliding_window_pattern`.
///
/// Gemma3 alternates a full-attention (global) layer every
/// `sliding_window_pattern` layers; the checkpoint's `config.json` may override
/// it, but [`ParallelModelConfig`] does not surface the pattern, so the
/// architecture default (6) is used. Every other family with a window applies
/// it to all of its layers (Mistral, Gemma2) or to the layers at or above
/// `max_window_layers` (Qwen3).
const GEMMA3_SLIDING_WINDOW_PATTERN: usize = 6;

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
///
/// The effective multiplicative weight is precomputed once at construction:
/// `weight` normally, or `(1 + weight)` for the Gemma* families that offset the
/// norm weight (`rms_norm_unit_offset`). Precomputing keeps [`Self::forward`] a
/// pure multiplication while staying differentiable; the norm weight itself is
/// frozen, so the offset does not need to be differentiated.
#[derive(Debug, Clone)]
struct FrozenRmsNorm {
    effective_weight: Tensor,
    eps: f64,
}

impl FrozenRmsNorm {
    fn from_base(base: &Tensor, eps: f64, unit_offset: bool) -> Result<Self> {
        let effective_weight = if unit_offset {
            (base + 1.0)?
        } else {
            base.clone()
        };
        Ok(Self {
            effective_weight,
            eps,
        })
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
        normalized.broadcast_mul(&self.effective_weight)
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

/// Causal mask that additionally masks keys more than `window` positions behind
/// the query (sliding-window attention).
///
/// A key at position `key` is visible to the query at position `query` when
/// `key <= query` **and** `key + window >= query`, matching the upstream
/// families' `if query < key || key + window < query` rule. The window includes
/// the query position itself, so every query attends to at most `window + 1`
/// keys. `window == 0` degenerates to the plain causal mask.
fn causal_window_mask(sequence_length: usize, window: usize, device: &Device) -> Result<Tensor> {
    let mut values = vec![0f32; sequence_length * sequence_length];
    for query in 0..sequence_length {
        for key in 0..sequence_length {
            if query < key || key + window < query {
                values[query * sequence_length + key] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(values, (sequence_length, sequence_length), device)
}

/// Per-layer sliding-window size (`None` = full attention).
///
/// The choice is derived from the family, because
/// [`ParallelModelConfig`] carries the window size but not the per-layer
/// pattern:
///
/// - Qwen3: the first `max_window_layers` layers use full attention and the
///   remaining layers use the window;
/// - Gemma3: a full-attention layer every `sliding_window_pattern` layers
///   (reusing the architecture default, since the pattern is not surfaced);
/// - Mistral and Gemma2: the window applies to every layer. Gemma2 alternates
///   local/global attention upstream; [`ParallelModelConfig`] does not surface
///   that alternation, so the training forward applies the window uniformly to
///   its layers. The serving path stays authoritative for the exact pattern.
/// - every other family: no window.
fn window_per_layer(config: &ParallelModelConfig) -> Vec<Option<usize>> {
    let layer_count = config.num_hidden_layers;
    let window = match config.sliding_window {
        Some(window) if window > 0 => window,
        _ => return vec![None; layer_count],
    };
    match config.architecture {
        ModelArchitecture::Qwen3 => (0..layer_count)
            .map(|index| (index >= config.max_window_layers).then_some(window))
            .collect(),
        ModelArchitecture::Gemma3 => (0..layer_count)
            .map(|index| ((index + 1) % GEMMA3_SLIDING_WINDOW_PATTERN != 0).then_some(window))
            .collect(),
        ModelArchitecture::Mistral | ModelArchitecture::Gemma2 => {
            vec![Some(window); layer_count]
        }
        ModelArchitecture::Llama | ModelArchitecture::Qwen2 | ModelArchitecture::Gemma => {
            vec![None; layer_count]
        }
    }
}

/// Narrows a capacity-sized mask to the actual sequence and broadcasts it.
fn narrow_attention_mask(
    mask: &Tensor,
    batch_size: usize,
    sequence_length: usize,
) -> Result<Tensor> {
    mask.narrow(0, 0, sequence_length)?
        .narrow(1, 0, sequence_length)?
        .broadcast_as((batch_size, 1, sequence_length, sequence_length))
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
    /// Query scaling factor, `1 / sqrt(head_dimension)` for Llama/Qwen2, or
    /// `1 / sqrt(query_pre_attention_scalar)` for Gemma2/Gemma3.
    attention_scale: f64,
    /// Gemma2/Gemma3 `attn_logit_softcapping`: `tanh(logits / cap) * cap`.
    attention_logit_softcapping: Option<f64>,
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

        let attention = (query.matmul(&key.t()?)? * self.attention_scale)?;
        let attention = match self.attention_logit_softcapping {
            Some(cap) => ((attention / cap)?.tanh()? * cap)?,
            None => attention,
        };
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
    /// Sliding-window size for this layer, or `None` for full attention.
    window: Option<usize>,
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
    /// Gemma* embedding scale (`sqrt(hidden_size)`), applied once to the token
    /// embeddings and never again to the tied head.
    embedding_scale: Option<f64>,
    /// Gemma2/Gemma3 `final_logit_softcapping`: `tanh(logits / cap) * cap`.
    logit_softcapping: Option<f64>,
    causal_mask: Tensor,
    /// One capacity-sized mask per distinct sliding-window size.
    sliding_masks: Vec<(usize, Tensor)>,
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
            config.rms_norm_unit_offset,
        )?;

        let windows = window_per_layer(config);
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for (index, window) in windows.iter().enumerate() {
            let layer_prefix = format!("model.layers.{index}");
            let block = Self::load_block(
                &base_builder,
                &adapter_builder,
                config,
                lora,
                &layer_prefix,
                *window,
                device,
            )?;
            blocks.push(block);
        }

        let (global_cos, global_sin) = build_rotary_tables(config, config.rope_theta, device)?;
        let local_tables = match config.rope_local_base_frequency {
            Some(frequency) => Some(build_rotary_tables(config, frequency as f32, device)?),
            None => None,
        };
        for (index, block) in blocks.iter_mut().enumerate() {
            let (cos, sin) = match (&local_tables, windows[index]) {
                (Some((local_cos, local_sin)), Some(_)) => (local_cos.clone(), local_sin.clone()),
                _ => (global_cos.clone(), global_sin.clone()),
            };
            block.attention.cos = cos;
            block.attention.sin = sin;
        }

        let sequence_capacity = config.max_position_embeddings.min(2048);
        let mut sliding_masks: Vec<(usize, Tensor)> = Vec::new();
        for window in windows.iter().flatten() {
            if !sliding_masks
                .iter()
                .any(|(existing_window, _)| existing_window == window)
            {
                sliding_masks.push((
                    *window,
                    causal_window_mask(sequence_capacity, *window, device)?,
                ));
            }
        }

        Ok(Self {
            token_embedding,
            blocks,
            final_norm,
            language_model_head,
            tie_word_embeddings: config.tie_word_embeddings,
            embedding_scale: config.embedding_scale,
            logit_softcapping: config.logit_softcapping,
            causal_mask: causal_mask(sequence_capacity, device)?,
            sliding_masks,
            sequence_capacity,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn load_block(
        base_builder: &VarBuilder,
        adapter_builder: &VarBuilder,
        config: &ParallelModelConfig,
        lora: LoRAConfiguration,
        layer_prefix: &str,
        window: Option<usize>,
        device: &Device,
    ) -> anyhow::Result<TrainableBlock> {
        let input_norm = FrozenRmsNorm::from_base(
            &base_builder.get(
                config.hidden_size,
                &format!("{layer_prefix}.input_layernorm.weight"),
            )?,
            config.rms_norm_eps,
            config.rms_norm_unit_offset,
        )?;
        let post_attention_norm = FrozenRmsNorm::from_base(
            &base_builder.get(
                config.hidden_size,
                &format!("{layer_prefix}.post_attention_layernorm.weight"),
            )?,
            config.rms_norm_eps,
            config.rms_norm_unit_offset,
        )?;
        let head_dimension = config.head_dimension();
        let query_size = head_dimension * config.num_attention_heads;
        let key_value_size = head_dimension * config.num_key_value_heads;
        let attention_scale =
            1.0 / (config.query_pre_attention_scalar.unwrap_or(head_dimension) as f64).sqrt();

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
            attention_scale,
            attention_logit_softcapping: config.attention_logit_softcapping,
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
            window,
        })
    }

    /// Token embeddings, scaled by `embedding_scale` when the family requires
    /// it. The scale is applied here only, so a tied language-model head reads
    /// the unscaled `token_embedding` and never double-scales.
    fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, sequence_length) = token_ids.dims2()?;
        let hidden_size = self.token_embedding.dim(1)?;
        let flat_tokens = token_ids.flatten_all()?;
        let hidden = self
            .token_embedding
            .index_select(&flat_tokens, 0)?
            .to_dtype(DType::F32)?
            .reshape((batch_size, sequence_length, hidden_size))?;
        match self.embedding_scale {
            Some(scale) => hidden * scale,
            None => Ok(hidden),
        }
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
        let full_mask = narrow_attention_mask(&self.causal_mask, batch_size, sequence_length)?;
        let window_masks = self
            .sliding_masks
            .iter()
            .map(|(window, mask)| {
                Ok((
                    *window,
                    narrow_attention_mask(mask, batch_size, sequence_length)?,
                ))
            })
            .collect::<Result<Vec<(usize, Tensor)>>>()?;

        let mut hidden = self.embed_tokens(token_ids)?;

        for block in &self.blocks {
            let mask = match block.window {
                Some(window) => window_masks
                    .iter()
                    .find(|(candidate, _)| *candidate == window)
                    .map(|(_, mask)| mask)
                    .unwrap_or(&full_mask),
                None => &full_mask,
            };
            hidden = block.forward(&hidden, mask)?;
        }
        let hidden = self.final_norm.forward(&hidden)?;
        let (batch_size, sequence_length, hidden_size) = hidden.dims3()?;
        let head_weight = if self.tie_word_embeddings {
            &self.token_embedding
        } else {
            &self.language_model_head
        };
        let mut logits = hidden
            .reshape((batch_size * sequence_length, hidden_size))?
            .matmul(&head_weight.t()?)?
            .reshape((batch_size, sequence_length, head_weight.dim(0)?))?;
        if let Some(cap) = self.logit_softcapping {
            logits = ((logits / cap)?.tanh()? * cap)?;
        }
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

/// Builds the RoPE cosine/sine tables for the configured positions and base
/// frequency.
fn build_rotary_tables(
    config: &ParallelModelConfig,
    base_frequency: f32,
    device: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let head_dimension = config.head_dimension();
    let positions = config.max_position_embeddings.min(2048);
    let inverse_frequencies: Vec<f32> = (0..head_dimension)
        .step_by(2)
        .map(|index| 1f32 / base_frequency.powf(index as f32 / head_dimension as f32))
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

    #[test]
    fn unit_offset_rms_norm_adds_one_to_the_weight() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let input = Tensor::new(&[[1.0_f32, 2.0, 3.0, 4.0]], &device)?;
        let unit_weight = Tensor::ones((4,), DType::F32, &device)?;
        let plain_norm = FrozenRmsNorm::from_base(&unit_weight, 1e-5, false)?;
        let offset_norm = FrozenRmsNorm::from_base(&unit_weight, 1e-5, true)?;

        let plain_output = plain_norm
            .forward(&input)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let offset_output = offset_norm
            .forward(&input)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert!(
            plain_output.iter().any(|value| value.abs() > 1e-6),
            "the reference norm must be non-trivial"
        );
        for (plain_value, offset_value) in plain_output.iter().zip(offset_output.iter()) {
            assert!(
                (offset_value - 2.0 * plain_value).abs() < 1e-5,
                "with weight 1 the offset must double the output: plain {plain_value}, offset {offset_value}"
            );
        }
        Ok(())
    }

    #[test]
    fn embedding_scale_scales_the_hidden_state() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let tokens = Tensor::new(&[[1_u32, 2, 3]], &device)?;

        let mut scaled_config = tiny_config();
        scaled_config.embedding_scale = Some(2.0);
        let scaled_base = base_tensors(&scaled_config)?;
        let mut scaled_variables = VarMap::new();
        let scaled_model = TrainableLlama::load(
            &scaled_base,
            &scaled_config,
            LoRAConfiguration::new(2, 4.0),
            &mut scaled_variables,
            &device,
        )?;

        let plain_config = tiny_config();
        let plain_base = base_tensors(&plain_config)?;
        let mut plain_variables = VarMap::new();
        let plain_model = TrainableLlama::load(
            &plain_base,
            &plain_config,
            LoRAConfiguration::new(2, 4.0),
            &mut plain_variables,
            &device,
        )?;

        let scaled_hidden = scaled_model
            .embed_tokens(&tokens)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let plain_hidden = plain_model
            .embed_tokens(&tokens)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(scaled_hidden.len(), plain_hidden.len());
        assert!(
            plain_hidden.iter().any(|value| value.abs() > 1e-6),
            "the embedding must be non-trivial"
        );
        for (scaled_value, plain_value) in scaled_hidden.iter().zip(plain_hidden.iter()) {
            assert!(
                (scaled_value - 2.0 * plain_value).abs() < 1e-5,
                "embedding_scale must multiply the hidden state: scaled {scaled_value}, plain {plain_value}"
            );
        }
        Ok(())
    }

    #[test]
    fn logit_softcapping_bounds_the_logits() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut config = tiny_config();
        config.logit_softcapping = Some(5.0);
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
        let logits = model.forward(&tokens)?.flatten_all()?.to_vec1::<f32>()?;
        for value in &logits {
            assert!(value.is_finite(), "logits must be finite, got {value}");
            assert!(
                value.abs() <= 5.0 + 1e-4,
                "softcapping must bound the logits, got {value}"
            );
        }
        Ok(())
    }

    #[test]
    fn attention_softcapping_is_finite_and_differentiable() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut config = tiny_config();
        config.architecture = ModelArchitecture::Gemma2;
        config.rms_norm_unit_offset = true;
        config.embedding_scale = Some((config.hidden_size as f64).sqrt());
        config.attention_logit_softcapping = Some(50.0);
        config.logit_softcapping = Some(30.0);
        config.query_pre_attention_scalar = Some(config.head_dimension());
        config.sliding_window = Some(2);
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
        let logits = model.decision_logits(&tokens, &[3_usize])?;
        for value in logits.flatten_all()?.to_vec1::<f32>()? {
            assert!(value.is_finite(), "logits must be finite, got {value}");
        }

        let loss = candle_nn::loss::cross_entropy(&logits, &Tensor::new(&[7_u32], &device)?)?;
        let gradients = loss.backward()?;
        let mut found_gradient = false;
        for variable in &variable_map.all_vars() {
            if let Some(gradient) = gradients.get(variable) {
                found_gradient = true;
                assert!(gradient
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|value| value.is_finite()));
            }
        }
        assert!(
            found_gradient,
            "the softcapped attention must stay differentiable"
        );
        Ok(())
    }

    #[test]
    fn sliding_window_mask_masks_the_distant_past() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let sequence_length = 5;
        let window = 2;
        let mask = causal_window_mask(sequence_length, window, &device)?;
        let values = mask.flatten_all()?.to_vec1::<f32>()?;
        let at = |query: usize, key: usize| values[query * sequence_length + key];

        assert_eq!(at(0, 0), 0.0);
        assert_eq!(at(2, 0), 0.0, "key 2 positions back is still visible");
        assert!(
            at(3, 0) < 0.0 && at(3, 0).is_infinite(),
            "key 3 positions back must be masked"
        );
        assert_eq!(at(3, 1), 0.0);
        assert!(
            at(4, 1) < 0.0 && at(4, 1).is_infinite(),
            "key 3 positions back must be masked"
        );
        assert_eq!(at(4, 2), 0.0);
        assert!(
            at(1, 3) < 0.0 && at(1, 3).is_infinite(),
            "future keys must stay masked by the causal rule"
        );
        Ok(())
    }
}
