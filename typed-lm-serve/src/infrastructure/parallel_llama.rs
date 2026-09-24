//! Parallel-capable Llama forward implementation.
//!
//! Vendored from `candle-transformers` 0.11 so that two capabilities missing
//! from the upstream public API become available:
//!
//! 1. The key/value cache is observable, so the shared prefix cache of a
//!    request can be **broadcast across the attention batch dimension**
//!    ([`ParallelCache::broadcast_batch`]).
//! 2. Hidden states can be collected at **arbitrary positions per row**
//!    ([`ParallelLlama::logits_from_hidden_at_positions`]), which is required
//!    because each question suffix has a different length.
//!
//! Together they allow every question of a request to be evaluated in a
//! **single batched forward pass** on top of one prefilled prefix, the
//! "parallel evaluation via KV-cache broadcasting" pattern.

use std::collections::HashMap;
use std::f32::consts::PI;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::attention::{flash_attn as cpu_flash_attention, AttnMask};
use candle_nn::{embedding, rotary_emb, Embedding, Module, VarBuilder};
use candle_transformers::models::llama::{Llama3RopeConfig, Llama3RopeType};
use candle_transformers::models::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm};
use candle_transformers::utils::{build_causal_mask, repeat_kv};

use typed_lm_common::model_config::ParallelModelConfig;

/// Key/value cache with public batch broadcasting.
#[derive(Debug, Clone)]
pub struct ParallelCache {
    masks: HashMap<(usize, usize), Tensor>,
    use_key_value_cache: bool,
    key_values: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn default_inverse_frequencies(config: &ParallelModelConfig) -> Vec<f32> {
    let head_dimension = config.head_dimension();
    (0..head_dimension)
        .step_by(2)
        .map(|index| {
            1f32 / (config.rope_theta as f64).powf(index as f64 / head_dimension as f64) as f32
        })
        .collect()
}

impl ParallelCache {
    /// Builds an empty cache, precomputing the rotary embeddings.
    pub fn new(
        use_key_value_cache: bool,
        dtype: DType,
        config: &ParallelModelConfig,
        device: &Device,
    ) -> Result<Self> {
        let inverse_frequencies = match &config.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => default_inverse_frequencies(config),
            Some(rope_scaling) => {
                let low_frequency_wavelength = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_frequency_wavelength = rope_scaling.original_max_position_embeddings
                    as f32
                    / rope_scaling.high_freq_factor;
                default_inverse_frequencies(config)
                    .into_iter()
                    .map(|frequency| {
                        let wavelength = 2. * PI / frequency;
                        if wavelength < high_frequency_wavelength {
                            frequency
                        } else if wavelength > low_frequency_wavelength {
                            frequency / rope_scaling.factor
                        } else {
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelength
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            (1. - smooth) * frequency / rope_scaling.factor + smooth * frequency
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };
        let theta = Tensor::new(inverse_frequencies.as_slice(), device)?;
        let index_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = index_theta.cos()?.to_dtype(dtype)?;
        let sin = index_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            masks: HashMap::new(),
            use_key_value_cache,
            key_values: vec![None; config.num_hidden_layers],
            cos,
            sin,
            device: device.clone(),
        })
    }

    fn mask(&mut self, sequence_length: usize, index_position: usize) -> Result<Tensor> {
        let key_value_length = index_position + sequence_length;
        if let Some(mask) = self.masks.get(&(sequence_length, key_value_length)) {
            return Ok(mask.clone());
        }
        let mask = build_causal_mask(sequence_length, index_position, &self.device)?;
        self.masks
            .insert((sequence_length, key_value_length), mask.clone());
        Ok(mask)
    }

    /// Replicates the cached key/value tensors across `batch_size` rows.
    ///
    /// This is the literal KV-cache broadcast: the prefix was prefilled once
    /// with a single row, and every question branch receives an identical copy
    /// so all branches can be continued in one batched forward pass.
    pub fn broadcast_batch(&mut self, batch_size: usize) -> Result<()> {
        for (keys, values) in self.key_values.iter_mut().flatten() {
            let current_batch = keys.dims()[0];
            if current_batch != batch_size {
                *keys = keys
                    .expand((batch_size, keys.dims()[1], keys.dims()[2], keys.dims()[3]))?
                    .contiguous()?;
                *values = values
                    .expand((
                        batch_size,
                        values.dims()[1],
                        values.dims()[2],
                        values.dims()[3],
                    ))?
                    .contiguous()?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ParallelAttention {
    query_projection: Linear,
    key_projection: Linear,
    value_projection: Linear,
    output_projection: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dimension: usize,
    max_position_embeddings: usize,
    rotation_span: tracing::Span,
    span: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    mask.where_cond(&on_true, on_false)
}

/// Computes attention context with the fused CPU kernel, supporting GQA.
///
/// `query` has shape `(batch, sequence, heads, head_dim)`, `key` and `value`
/// have shape `(batch, key_value_sequence, key_value_heads, head_dim)` (grouped
/// query attention where the heads may differ), and `index_position` is the
/// position of the first query token in the key/value sequence. Returns the
/// context in `(batch, heads, sequence, head_dim)`.
///
/// The fused kernel keeps grouped query attention grouped (no key/value head
/// replication) and runs in `F32` internally, upcasting half-precision inputs
/// and narrowing the result back to the input dtype.
fn cpu_flash_attention_context(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    index_position: usize,
) -> Result<Tensor> {
    let head_dimension = query.dims()[3];
    let softmax_scale = 1.0 / (head_dimension as f32).sqrt();
    let attention_mask = AttnMask::causal_with_offset(index_position);
    match query.dtype() {
        DType::F32 => {
            cpu_flash_attention::<f32>(query, key, value, softmax_scale, attention_mask, None, None)
        }
        DType::F16 => {
            let query_f32 = query.to_dtype(DType::F32)?;
            let key_f32 = key.to_dtype(DType::F32)?;
            let value_f32 = value.to_dtype(DType::F32)?;
            cpu_flash_attention::<f32>(
                &query_f32,
                &key_f32,
                &value_f32,
                softmax_scale,
                attention_mask,
                None,
                None,
            )?
            .to_dtype(DType::F16)
        }
        DType::BF16 => {
            let query_f32 = query.to_dtype(DType::F32)?;
            let key_f32 = key.to_dtype(DType::F32)?;
            let value_f32 = value.to_dtype(DType::F32)?;
            cpu_flash_attention::<f32>(
                &query_f32,
                &key_f32,
                &value_f32,
                softmax_scale,
                attention_mask,
                None,
                None,
            )?
            .to_dtype(DType::BF16)
        }
        other => Err(candle_core::Error::Msg(format!(
            "unsupported dtype for CPU flash attention: {other:?}"
        ))),
    }
}

impl ParallelAttention {
    fn apply_rotary_embedding(
        &self,
        hidden: &Tensor,
        index_position: usize,
        cache: &ParallelCache,
    ) -> Result<Tensor> {
        let _enter = self.rotation_span.enter();
        let (_batch, _heads, sequence_length, _head_dimension) = hidden.dims4()?;
        let cos = cache.cos.narrow(0, index_position, sequence_length)?;
        let sin = cache.sin.narrow(0, index_position, sequence_length)?;
        rotary_emb::rope(hidden, &cos, &sin)
    }

    fn forward(
        &self,
        hidden: &Tensor,
        index_position: usize,
        block_index: usize,
        cache: &mut ParallelCache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
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
        let mut value = value
            .reshape((
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dimension,
            ))?
            .transpose(1, 2)?;

        let query = self.apply_rotary_embedding(&query, index_position, cache)?;
        let mut key = self.apply_rotary_embedding(&key, index_position, cache)?;

        if cache.use_key_value_cache {
            if let Some((cached_keys, cached_values)) = &cache.key_values[block_index] {
                key = Tensor::cat(&[cached_keys, &key], 2)?.contiguous()?;
                value = Tensor::cat(&[cached_values, &value], 2)?.contiguous()?;
                let keys_length = key.dims()[1];
                if keys_length > self.max_position_embeddings {
                    key = key
                        .narrow(
                            D::Minus1,
                            keys_length - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?
                }
                let values_length = value.dims()[1];
                if values_length > 2 * self.max_position_embeddings {
                    value = value
                        .narrow(
                            D::Minus1,
                            values_length - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?
                }
            }
            cache.key_values[block_index] = Some((key.clone(), value.clone()))
        }

        let key = repeat_kv(key, self.num_attention_heads / self.num_key_value_heads)?;
        let value = repeat_kv(value, self.num_attention_heads / self.num_key_value_heads)?;

        // On the CPU the fused flash-style kernel is both faster and more
        // memory-efficient than the matmul/softmax/matmul path, and it keeps
        // grouped query attention grouped. The result layout matches the
        // standard path: (batch, heads, sequence, head_dim).
        if query.device().is_cpu() {
            let query_in_sequence_layout = query.transpose(1, 2)?.contiguous()?;
            let key_in_sequence_layout = key.transpose(1, 2)?.contiguous()?;
            let value_in_sequence_layout = value.transpose(1, 2)?.contiguous()?;
            let context = cpu_flash_attention_context(
                &query_in_sequence_layout,
                &key_in_sequence_layout,
                &value_in_sequence_layout,
                index_position,
            )?;
            let output =
                context
                    .transpose(1, 2)?
                    .reshape(&[batch_size, sequence_length, hidden_size])?;
            return self.output_projection.forward(&output);
        }

        let input_dtype = query.dtype();
        let query = query.to_dtype(DType::F32)?;
        let key = key.to_dtype(DType::F32)?;
        let value = value.to_dtype(DType::F32)?;
        let attention = (query.matmul(&key.t()?)? / (self.head_dimension as f64).sqrt())?;
        let attention = if sequence_length == 1 {
            attention
        } else {
            let mask = cache
                .mask(sequence_length, index_position)?
                .broadcast_as(attention.shape())?;
            masked_fill(&attention, &mask, f32::NEG_INFINITY)?
        };
        let attention = candle_nn::ops::softmax_last_dim(&attention)?;
        let output = attention
            .matmul(&value.contiguous()?)?
            .to_dtype(input_dtype)?;

        let output =
            output
                .transpose(1, 2)?
                .reshape(&[batch_size, sequence_length, hidden_size])?;
        self.output_projection.forward(&output)
    }

    fn load(variable_builder: VarBuilder, config: &ParallelModelConfig) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attention");
        let rotation_span = tracing::span!(tracing::Level::TRACE, "attention-rotary");
        let input_size = config.hidden_size;
        let head_dimension = config.head_dimension();
        let query_size = head_dimension * config.num_attention_heads;
        let key_value_size = head_dimension * config.num_key_value_heads;
        let with_bias = config.has_query_key_value_bias();
        Ok(Self {
            query_projection: projection(
                input_size,
                query_size,
                with_bias,
                variable_builder.pp("q_proj"),
            )?,
            key_projection: projection(
                input_size,
                key_value_size,
                with_bias,
                variable_builder.pp("k_proj"),
            )?,
            value_projection: projection(
                input_size,
                key_value_size,
                with_bias,
                variable_builder.pp("v_proj"),
            )?,
            output_projection: projection(
                query_size,
                input_size,
                false,
                variable_builder.pp("o_proj"),
            )?,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dimension,
            max_position_embeddings: config.max_position_embeddings,
            rotation_span,
            span,
        })
    }
}

/// Loads a linear projection with or without bias.
///
/// `with_tracing::linear_b` adds a bias and applies it during `forward`; when
/// `with_bias` is false the call shape matches `linear_no_bias`. The bias is
/// therefore applied exactly once, inside the projection.
fn projection(
    input_size: usize,
    output_size: usize,
    with_bias: bool,
    variable_builder: VarBuilder,
) -> Result<Linear> {
    if with_bias {
        linear_b(input_size, output_size, true, variable_builder)
    } else {
        linear_no_bias(input_size, output_size, variable_builder)
    }
}

#[derive(Debug, Clone)]
struct ParallelMlp {
    gate_projection: Linear,
    up_projection: Linear,
    down_projection: Linear,
    span: tracing::Span,
}

impl ParallelMlp {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden = (candle_nn::ops::silu(&self.gate_projection.forward(hidden)?)?
            * self.up_projection.forward(hidden)?)?;
        self.down_projection.forward(&hidden)
    }

    fn load(variable_builder: VarBuilder, config: &ParallelModelConfig) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            gate_projection: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                variable_builder.pp("gate_proj"),
            )?,
            up_projection: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                variable_builder.pp("up_proj"),
            )?,
            down_projection: linear_no_bias(
                config.intermediate_size,
                config.hidden_size,
                variable_builder.pp("down_proj"),
            )?,
            span,
        })
    }
}

#[derive(Debug, Clone)]
struct ParallelBlock {
    input_norm: RmsNorm,
    attention: ParallelAttention,
    post_attention_norm: RmsNorm,
    mlp: ParallelMlp,
    span: tracing::Span,
}

impl ParallelBlock {
    fn forward(
        &self,
        hidden: &Tensor,
        index_position: usize,
        block_index: usize,
        cache: &mut ParallelCache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = hidden;
        let hidden = self.input_norm.forward(hidden)?;
        let hidden = (self
            .attention
            .forward(&hidden, index_position, block_index, cache)?
            + residual)?;
        let residual = &hidden;
        self.mlp
            .forward(&self.post_attention_norm.forward(&hidden)?)?
            + residual
    }

    fn load(variable_builder: VarBuilder, config: &ParallelModelConfig) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let input_norm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            variable_builder.pp("input_layernorm"),
        )?;
        let post_attention_norm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            variable_builder.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            input_norm,
            attention: ParallelAttention::load(variable_builder.pp("self_attn"), config)?,
            post_attention_norm,
            mlp: ParallelMlp::load(variable_builder.pp("mlp"), config)?,
            span,
        })
    }
}

/// Llama model with batch-broadcastable cache and per-row logit collection.
#[derive(Debug, Clone)]
pub struct ParallelLlama {
    token_embedding: Embedding,
    blocks: Vec<ParallelBlock>,
    final_norm: RmsNorm,
    language_model_head: Linear,
}

impl ParallelLlama {
    /// Runs every block and returns the hidden states for **all** positions,
    /// shaped `(batch, sequence_length, hidden_size)`.
    pub fn forward_hidden_all(
        &self,
        tokens: &Tensor,
        index_position: usize,
        cache: &mut ParallelCache,
    ) -> Result<Tensor> {
        let mut hidden = self.token_embedding.forward(tokens)?;
        for (block_index, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, index_position, block_index, cache)?;
        }
        Ok(hidden)
    }

    /// Applies the final norm and head to the hidden state of each row at its
    /// own position, returning `(batch, vocabulary)` logits in `F32`.
    pub fn logits_from_hidden_at_positions(
        &self,
        hidden: &Tensor,
        positions: &[usize],
    ) -> Result<Tensor> {
        let mut rows = Vec::with_capacity(positions.len());
        for (row, position) in positions.iter().enumerate() {
            rows.push(hidden.i((row, *position, ..))?);
        }
        let stacked = Tensor::stack(&rows, 0)?;
        let normalized = self.final_norm.forward(&stacked)?;
        let logits = self.language_model_head.forward(&normalized)?;
        logits.to_dtype(DType::F32)
    }

    /// Reference single-row forward returning last-position logits `(1, vocab)`.
    pub fn forward_last(
        &self,
        tokens: &Tensor,
        index_position: usize,
        cache: &mut ParallelCache,
    ) -> Result<Tensor> {
        let (_batch_size, sequence_length) = tokens.dims2()?;
        let hidden = self.forward_hidden_all(tokens, index_position, cache)?;
        let normalized = self.final_norm.forward(&hidden)?;
        let last = normalized.i((.., sequence_length - 1, ..))?.contiguous()?;
        let logits = self.language_model_head.forward(&last)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(variable_builder: VarBuilder, config: &ParallelModelConfig) -> Result<Self> {
        let token_embedding = embedding(
            config.vocab_size,
            config.hidden_size,
            variable_builder.pp("model.embed_tokens"),
        )?;
        let language_model_head = if config.tie_word_embeddings {
            Linear::from_weights(token_embedding.embeddings().clone(), None)
        } else {
            linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                variable_builder.pp("lm_head"),
            )?
        };
        let final_norm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            variable_builder.pp("model.norm"),
        )?;
        let blocks: Vec<ParallelBlock> = (0..config.num_hidden_layers)
            .map(|index| {
                ParallelBlock::load(variable_builder.pp(format!("model.layers.{index}")), config)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            token_embedding,
            blocks,
            final_norm,
            language_model_head,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_inverse_frequencies_have_head_dimension_halved() -> anyhow::Result<()> {
        let value = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "vocab_size": 100,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        });
        let config = ParallelModelConfig::from_llama_json(value)?;
        let frequencies = default_inverse_frequencies(&config);
        assert_eq!(frequencies.len(), (64 / 8) / 2);
        assert!(frequencies.iter().all(|value| *value > 0.0));
        Ok(())
    }

    /// Reference attention via explicit matmul/softmax, supporting GQA.
    fn reference_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        index_position: usize,
        num_key_value_groups: usize,
    ) -> anyhow::Result<Tensor> {
        let device = query.device();
        let (_batch_size, sequence_length, _heads, head_dimension) = query.dims4()?;
        let query = query.transpose(1, 2)?.contiguous()?;
        let key = repeat_kv(key.transpose(1, 2)?.contiguous()?, num_key_value_groups)?;
        let value = repeat_kv(value.transpose(1, 2)?.contiguous()?, num_key_value_groups)?;
        let scale = 1.0 / (head_dimension as f64).sqrt();
        let scores = (query.matmul(&key.t()?)? * scale)?;
        let mask = build_causal_mask(sequence_length, index_position, device)?
            .broadcast_as(scores.shape())?;
        let masked = masked_fill(&scores, &mask, f32::NEG_INFINITY)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&masked)?;
        let context = probabilities.matmul(&value.contiguous()?)?;
        Ok(context)
    }

    fn deterministic_tensor(
        shape: (usize, usize, usize, usize),
        device: &Device,
        seed: &mut u64,
    ) -> anyhow::Result<Tensor> {
        let count = shape.0 * shape.1 * shape.2 * shape.3;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            values.push(((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5);
        }
        Ok(Tensor::from_vec(values, shape, device)?)
    }

    #[test]
    fn cpu_flash_attention_matches_reference_with_grouped_heads() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let head_dimension = 8_usize;
        let heads = 4_usize;
        let key_value_heads = 2_usize;
        let groups = heads / key_value_heads;
        let mut seed = 0x5eed_u64;

        // B=1 exercises the single-batch causal kernel; B=2 exercises the
        // packed varlen path used for the batched suffix pass.
        for batch_size in [1_usize, 2] {
            for (sequence_length, index_position) in [(3_usize, 0_usize), (2, 5)] {
                let key_value_length = index_position + sequence_length;
                let query = deterministic_tensor(
                    (batch_size, sequence_length, heads, head_dimension),
                    &device,
                    &mut seed,
                )?;
                let key = deterministic_tensor(
                    (
                        batch_size,
                        key_value_length,
                        key_value_heads,
                        head_dimension,
                    ),
                    &device,
                    &mut seed,
                )?;
                let value = deterministic_tensor(
                    (
                        batch_size,
                        key_value_length,
                        key_value_heads,
                        head_dimension,
                    ),
                    &device,
                    &mut seed,
                )?;

                let flash = cpu_flash_attention_context(&query, &key, &value, index_position)?;
                let reference = reference_attention(&query, &key, &value, index_position, groups)?;

                let flash_values = flash.flatten_all()?.to_vec1::<f32>()?;
                let reference_values = reference.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(flash_values.len(), reference_values.len());
                for (position, (flash_value, reference_value)) in
                    flash_values.iter().zip(reference_values.iter()).enumerate()
                {
                    assert!(
                        (flash_value - reference_value).abs() < 1e-4,
                        "batch {batch_size} seq {sequence_length} offset {index_position} element {position}: \
                         flash {flash_value} vs reference {reference_value}"
                    );
                }
            }
        }
        Ok(())
    }

    /// Live test (ignored by default): proves the vendored model reproduces the
    /// upstream candle-transformers `Llama` logits exactly. Downloads the tiny
    /// random model once.
    #[test]
    #[ignore]
    fn vendored_forward_matches_candle_llama() -> anyhow::Result<()> {
        use candle_core::IndexOp as _;
        use candle_nn::VarBuilder;
        use candle_transformers::models::llama::{
            Cache as CandleCache, Config as LlamaRuntimeConfig, Llama as CandleLlama, LlamaConfig,
        };
        use typed_lm_common::device::DeviceResolver;
        use typed_lm_common::model_repository::ModelRepository;

        let device = DeviceResolver::resolve()?;
        let dtype = DType::F32;
        let repository =
            ModelRepository::new("hf-internal-testing/tiny-random-LlamaForCausalLM", None)?;
        let config_file = repository.download("config.json")?;
        let weights_file = repository.download("model.safetensors")?;
        let config_reader = std::fs::File::open(&config_file)?;
        let llama_config: LlamaConfig = serde_json::from_reader(config_reader)?;
        let reference_config: LlamaRuntimeConfig = llama_config.into_config(false);
        let vendored_config = ParallelModelConfig::from_json_for_architecture(
            serde_json::from_str(&std::fs::read_to_string(&config_file)?)?,
            typed_lm_common::checkpoint::ModelArchitecture::Llama,
        )?;

        let reference_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_file.as_path()], dtype, &device)?
        };
        let reference_model = CandleLlama::load(reference_builder, &reference_config)?;
        let vendored_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_file.as_path()], dtype, &device)?
        };
        let vendored_model = ParallelLlama::load(vendored_builder, &vendored_config)?;

        let tokens: Vec<u32> = vec![0, 5, 10, 15, 20, 25];
        let tensor = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
        let mut reference_cache = CandleCache::new(true, dtype, &reference_config, &device)?;
        let reference_logits = reference_model.forward(&tensor, 0, &mut reference_cache)?;
        let mut vendored_cache = ParallelCache::new(true, dtype, &vendored_config, &device)?;
        let vendored_logits = vendored_model.forward_last(&tensor, 0, &mut vendored_cache)?;

        for token_id in [0_usize, 1, 3, 100, 31999] {
            let reference_value = reference_logits.i((0, token_id))?.to_vec0::<f32>()?;
            let vendored_value = vendored_logits.i((0, token_id))?.to_vec0::<f32>()?;
            assert!(
                (reference_value - vendored_value).abs() < 1e-4,
                "token {token_id}: candle {reference_value} vs vendored {vendored_value}"
            );
        }
        Ok(())
    }

    /// Live test (ignored by default): proves the vendored model reproduces the
    /// upstream candle-transformers Qwen2 logits exactly (including q/k/v
    /// biases) on a synthetic checkpoint with a realistic `head_dim`.
    ///
    /// A synthetic model is used instead of a published tiny one because the
    /// tiny test models ship `head_dim = 2`, where candle's RoPE has an edge
    /// case that does not occur in real Qwen2 checkpoints (`head_dim` 64/128).
    /// The synthetic weights are generated deterministically, so the test is
    /// reproducible without a download.
    #[test]
    #[ignore]
    fn vendored_forward_matches_candle_qwen2() -> anyhow::Result<()> {
        use candle_nn::VarBuilder;
        use candle_transformers::models::qwen2::{
            Config as Qwen2Config, ModelForCausalLM as Qwen2,
        };
        use std::collections::HashMap;
        use typed_lm_common::checkpoint::ModelArchitecture;

        let device = Device::Cpu;
        let dtype = DType::F32;
        let hidden = 32_usize;
        let heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = hidden / heads; // 8, unlike the tiny models' 2.
        let layers = 2_usize;
        let intermediate = 64_usize;
        let vocabulary = 128_usize;

        let synthetic = serde_json::json!({
            "model_type": "qwen2",
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "vocab_size": vocabulary,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": false,
            "sliding_window": 64,
            "max_window_layers": 1,
            "use_sliding_window": false,
            "hidden_act": "silu"
        });
        let reference_config: Qwen2Config = serde_json::from_value(synthetic.clone())?;
        let vendored_config =
            ParallelModelConfig::from_json_for_architecture(synthetic, ModelArchitecture::Qwen2)?;

        // Deterministic pseudo-random weights.
        let seed = std::cell::Cell::new(1234_u64);
        let next = || {
            let value = seed.get();
            let value = value
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed.set(value);
            ((value >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let matrix = |rows: usize, columns: usize| -> anyhow::Result<Tensor> {
            let values: Vec<f32> = (0..rows * columns).map(|_| next() * 0.1).collect();
            Ok(Tensor::from_vec(values, (rows, columns), &device)?)
        };
        let vector = |size: usize| -> anyhow::Result<Tensor> {
            let values: Vec<f32> = (0..size).map(|_| next() * 0.1).collect();
            Ok(Tensor::from_vec(values, size, &device)?)
        };

        let mut weights: HashMap<String, Tensor> = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            matrix(vocabulary, hidden)?,
        );
        weights.insert("lm_head.weight".to_string(), matrix(vocabulary, hidden)?);
        let norm_weight = vector(hidden)?;
        weights.insert("model.norm.weight".to_string(), norm_weight.clone());
        for layer in 0..layers {
            let prefix = format!("model.layers.{layer}");
            weights.insert(format!("{prefix}.input_layernorm.weight"), vector(hidden)?);
            weights.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                vector(hidden)?,
            );
            weights.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                matrix(heads * head_dim, hidden)?,
            );
            weights.insert(
                format!("{prefix}.self_attn.q_proj.bias"),
                vector(heads * head_dim)?,
            );
            weights.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                matrix(kv_heads * head_dim, hidden)?,
            );
            weights.insert(
                format!("{prefix}.self_attn.k_proj.bias"),
                vector(kv_heads * head_dim)?,
            );
            weights.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                matrix(kv_heads * head_dim, hidden)?,
            );
            weights.insert(
                format!("{prefix}.self_attn.v_proj.bias"),
                vector(kv_heads * head_dim)?,
            );
            weights.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                matrix(hidden, heads * head_dim)?,
            );
            weights.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                matrix(intermediate, hidden)?,
            );
            weights.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                matrix(intermediate, hidden)?,
            );
            weights.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                matrix(hidden, intermediate)?,
            );
        }

        let reference_builder = VarBuilder::from_tensors(weights.clone(), dtype, &device);
        let mut reference_model = Qwen2::new(&reference_config, reference_builder)?;
        let vendored_builder = VarBuilder::from_tensors(weights.clone(), dtype, &device);
        let vendored_model = ParallelLlama::load(vendored_builder, &vendored_config)?;

        // The normalized hidden state must match the upstream base model. The
        // upstream `Model::forward` applies the final RMSNorm; the vendored
        // `forward_hidden_all` does not, so the norm is applied here.
        {
            use candle_nn::Module as _;
            use candle_transformers::models::qwen2::Model as Qwen2Base;
            use candle_transformers::models::with_tracing::RmsNorm;
            let base_builder = VarBuilder::from_tensors(weights.clone(), dtype, &device);
            let mut base = Qwen2Base::new(&reference_config, base_builder)?;
            let probe = Tensor::new(&[3u32, 9, 11, 5], &device)?.unsqueeze(0)?;
            let reference_hidden = base.forward(&probe, 0, None)?;
            let mut cache = ParallelCache::new(true, dtype, &vendored_config, &device)?;
            let vendored_hidden = vendored_model.forward_hidden_all(&probe, 0, &mut cache)?;
            let norm_builder = VarBuilder::from_tensors(
                {
                    let mut only_norm = std::collections::HashMap::new();
                    only_norm.insert("norm.weight".to_string(), norm_weight.clone());
                    only_norm
                },
                dtype,
                &device,
            );
            let final_norm = RmsNorm::new(hidden, 1e-6, norm_builder.pp("norm"))?;
            let vendored_normalized = final_norm.forward(&vendored_hidden)?;
            let reference_values = reference_hidden.i((0, 3, ..))?.to_vec1::<f32>()?;
            let vendored_values = vendored_normalized.i((0, 3, ..))?.to_vec1::<f32>()?;
            let maximum = reference_values
                .iter()
                .zip(vendored_values.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                maximum < 1e-5,
                "normalized hidden diverges by {maximum} (q/k/v bias applied twice?)"
            );
        }

        // Multi-token prefill and a single-token step must both match.
        for tokens in [vec![42_u32], vec![3_u32, 9, 11, 5]] {
            let tensor = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
            reference_model.clear_kv_cache();
            let reference_logits = reference_model.forward(&tensor, 0)?;
            let mut vendored_cache = ParallelCache::new(true, dtype, &vendored_config, &device)?;
            let vendored_logits = vendored_model.forward_last(&tensor, 0, &mut vendored_cache)?;
            for token_id in [0_usize, 1, 7, vocabulary / 2, vocabulary - 1] {
                let reference_value = reference_logits.i((0, 0, token_id))?.to_vec0::<f32>()?;
                let vendored_value = vendored_logits.i((0, token_id))?.to_vec0::<f32>()?;
                // Relative tolerance: logits have mixed magnitudes and F32
                // accumulation differs slightly between the two code paths.
                let tolerance = 1e-4 + 1e-3 * reference_value.abs().max(vendored_value.abs());
                assert!(
                    (reference_value - vendored_value).abs() < tolerance,
                    "tokens {tokens:?} token {token_id}: candle {reference_value} vs vendored {vendored_value}"
                );
            }
        }
        Ok(())
    }
}
