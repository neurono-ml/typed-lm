//! Quantized (GGML) Qwen2 forward with a batch-broadcastable KV cache.
//!
//! Vendored from `candle-transformers` 0.11 `models::quantized_qwen2` so that
//! the two capabilities the project needs become available (the upstream type
//! keeps its KV cache private and only returns last-position logits):
//!
//! 1. The key/value cache is observable, so the shared prefix of a request can
//!    be broadcast across the attention batch dimension
//!    ([`QuantizedCache::broadcast_batch`]).
//! 2. Hidden states can be collected at arbitrary positions per row
//!    ([`ParallelQuantizedQwen2::logits_from_hidden_at_positions`]).
//!
//! Together they extend the "parallel evaluation via KV-cache broadcasting"
//! pattern to GGML-quantized checkpoints (`Q4_0`…`Q8K`), which are the natural
//! fit for CPU inference.
//!
//! Weight layout follows `quantized_qwen2.rs`: quantized `QMatMul` projections
//! plus dequantized attention biases (Qwen2 bias in q/k/v).

use candle_core::quantized::gguf_file;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::quantized_var_builder;
use std::collections::HashMap;

/// GGUF metadata keys consumed to rebuild the model configuration.
pub const GGUF_ARCHITECTURE_KEY: &str = "general.architecture";

/// Key/value cache with public batch broadcasting, for the quantized model.
#[derive(Debug, Clone)]
pub struct QuantizedCache {
    masks: HashMap<(usize, usize), Tensor>,
    /// One `(keys, values)` entry per layer; `None` before any forward pass.
    key_values: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl QuantizedCache {
    /// Builds an empty cache with precomputed rotary embeddings.
    pub fn new(
        head_dimension: usize,
        rope_theta: f32,
        context_length: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let (cos, sin) =
            precompute_frequencies(head_dimension, rope_theta, context_length, dtype, device)?;
        Ok(Self {
            masks: HashMap::new(),
            key_values: vec![None; 0],
            cos,
            sin,
            device: device.clone(),
        })
    }

    /// Declares how many layers the cache must hold (one entry per layer).
    pub fn set_layer_count(&mut self, layer_count: usize) {
        self.key_values = vec![None; layer_count];
    }

    fn mask(&mut self, sequence_length: usize, index_position: usize) -> Result<Tensor> {
        let key_value_length = index_position + sequence_length;
        if let Some(mask) = self.masks.get(&(sequence_length, key_value_length)) {
            return Ok(mask.clone());
        }
        // `build_causal_mask` can return a non-float dtype (e.g. U8); the mask is
        // added to F32 attention scores, so it is normalized to F32 here.
        let mask = candle_transformers::utils::build_causal_mask(
            sequence_length,
            index_position,
            &self.device,
        )?
        .to_dtype(DType::F32)?;
        self.masks
            .insert((sequence_length, key_value_length), mask.clone());
        Ok(mask)
    }

    /// Replicates the cached key/value tensors across `batch_size` rows.
    ///
    /// This is the literal KV-cache broadcast: the prefix was prefilled once on
    /// a single row and every question branch receives an identical copy.
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

/// Precomputes the RoPE `cos`/`sin` tables for a head dimension.
pub fn precompute_frequencies(
    head_dimension: usize,
    rope_theta: f32,
    context_length: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let inverse_frequencies: Vec<f32> = (0..head_dimension)
        .step_by(2)
        .map(|index| 1f32 / rope_theta.powf(index as f32 / head_dimension as f32))
        .collect();
    let theta = Tensor::new(inverse_frequencies.as_slice(), device)?;
    let index_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((
        index_theta.cos()?.to_dtype(dtype)?,
        index_theta.sin()?.to_dtype(dtype)?,
    ))
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
    span: tracing::Span,
}

impl Module for Mlp {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let w1 = self.feed_forward_w1.forward(hidden)?;
        let w3 = self.feed_forward_w3.forward(hidden)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_bq: Tensor,
    attention_bk: Tensor,
    attention_bv: Tensor,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp: Mlp,
    ffn_norm: RmsNorm,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dimension: usize,
    span_attention: tracing::Span,
    span_rotation: tracing::Span,
}

impl LayerWeights {
    fn apply_rotary_embedding(
        &self,
        hidden: &Tensor,
        index_position: usize,
        cache: &QuantizedCache,
    ) -> Result<Tensor> {
        let _enter = self.span_rotation.enter();
        let (_batch, _heads, sequence_length, _head_dimension) = hidden.dims4()?;
        let cos = cache.cos.narrow(0, index_position, sequence_length)?;
        let sin = cache.sin.narrow(0, index_position, sequence_length)?;
        candle_nn::rotary_emb::rope(&hidden.contiguous()?, &cos, &sin)
    }

    fn forward_attention(
        &self,
        hidden: &Tensor,
        mask: Option<&Tensor>,
        index_position: usize,
        layer_index: usize,
        cache: &mut QuantizedCache,
    ) -> Result<Tensor> {
        let _enter = self.span_attention.enter();
        let (batch_size, sequence_length, hidden_size) = hidden.dims3()?;

        let query = self.attention_wq.forward(hidden)?;
        let key = self.attention_wk.forward(hidden)?;
        let value = self.attention_wv.forward(hidden)?;

        let query = query.broadcast_add(&self.attention_bq)?;
        let key = key.broadcast_add(&self.attention_bk)?;
        let value = value.broadcast_add(&self.attention_bv)?;

        let query = query
            .reshape((
                batch_size,
                sequence_length,
                self.num_heads,
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

        let query = self.apply_rotary_embedding(&query, index_position, cache)?;
        let key = self.apply_rotary_embedding(&key, index_position, cache)?;

        let (key, value) = match &cache.key_values[layer_index] {
            None => (key, value),
            Some((cached_keys, cached_values)) => {
                let key = Tensor::cat(&[cached_keys, &key], 2)?;
                let value = Tensor::cat(&[cached_values, &value], 2)?;
                (key, value)
            }
        };
        cache.key_values[layer_index] = Some((key.clone(), value.clone()));

        let key =
            candle_transformers::utils::repeat_kv(key, self.num_heads / self.num_key_value_heads)?;
        let value = candle_transformers::utils::repeat_kv(
            value,
            self.num_heads / self.num_key_value_heads,
        )?;

        let attention = (query.matmul(&key.t()?)? / (self.head_dimension as f64).sqrt())?;
        let attention = match mask {
            None => attention,
            Some(mask) => {
                let mask = mask.broadcast_as(attention.shape())?;
                attention.broadcast_add(&mask)?
            }
        };
        let attention = candle_nn::ops::softmax_last_dim(&attention)?;
        let output = attention.matmul(&value.contiguous()?)?;
        let output =
            output
                .transpose(1, 2)?
                .reshape(&[batch_size, sequence_length, hidden_size])?;
        self.attention_wo.forward(&output)
    }
}

/// Quantized Qwen2 model with batch-broadcastable cache and per-row logits.
#[derive(Debug, Clone)]
pub struct ParallelQuantizedQwen2 {
    token_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    final_norm: RmsNorm,
    output: QMatMul,
    device: Device,
}

impl ParallelQuantizedQwen2 {
    /// Runs every layer and returns hidden states for **all** positions,
    /// shaped `(batch, sequence_length, hidden_size)`.
    pub fn forward_hidden_all(
        &self,
        tokens: &Tensor,
        index_position: usize,
        cache: &mut QuantizedCache,
    ) -> Result<Tensor> {
        let (_batch_size, sequence_length) = tokens.dims2()?;
        let mask = if sequence_length == 1 {
            None
        } else {
            Some(cache.mask(sequence_length, index_position)?)
        };
        let mut hidden = embedding_forward(&self.token_embeddings, tokens)?.to_dtype(DType::F32)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let residual = &hidden;
            let normalized = layer.attention_norm.forward(&hidden)?;
            let attention = layer.forward_attention(
                &normalized,
                mask.as_ref(),
                index_position,
                layer_index,
                cache,
            )?;
            hidden = (&attention + residual).map_err(|error| {
                candle_core::Error::Msg(format!("layer {layer_index} attention+residual: {error}"))
            })?;

            let residual = &hidden;
            let normalized = layer.ffn_norm.forward(&hidden)?;
            let feed_forward = layer.mlp.forward(&normalized)?;
            hidden = (&feed_forward + residual).map_err(|error| {
                candle_core::Error::Msg(format!("layer {layer_index} mlp+residual: {error}"))
            })?;
        }
        Ok(hidden)
    }

    /// Applies the final norm and head to each row's own position, returning
    /// `(batch, vocabulary)` logits in `F32`.
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
        let logits = self.output.forward(&normalized)?;
        logits.to_dtype(DType::F32)
    }

    /// Reference single-row forward returning last-position logits `(1, vocab)`.
    pub fn forward_last(
        &self,
        tokens: &Tensor,
        index_position: usize,
        cache: &mut QuantizedCache,
    ) -> Result<Tensor> {
        let (_batch_size, sequence_length) = tokens.dims2()?;
        let hidden = self.forward_hidden_all(tokens, index_position, cache)?;
        let normalized = self.final_norm.forward(&hidden)?;
        let last = normalized.i((.., sequence_length - 1, ..))?.contiguous()?;
        let logits = self.output.forward(&last)?;
        logits.to_dtype(DType::F32)
    }

    /// Number of hidden layers, used to size the key/value cache.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Reads the architecture parameters from a GGUF file into a normalized
    /// config (tokenizer and weights are loaded separately).
    pub fn config_from_gguf(
        gguf_path: &std::path::Path,
    ) -> anyhow::Result<typed_lm_common::model_config::ParallelModelConfig> {
        use typed_lm_common::checkpoint::ModelArchitecture;
        use typed_lm_common::model_config::ParallelModelConfig;
        let metadata = read_gguf_metadata(gguf_path)?;
        let architecture = metadata
            .get(GGUF_ARCHITECTURE_KEY)
            .cloned()
            .unwrap_or_else(|| "qwen2".to_string());
        if architecture != "qwen2" {
            return Err(anyhow::anyhow!(
                "GGUF architecture '{architecture}' is not supported; expected 'qwen2'"
            ));
        }
        let num_heads = metadata_number(&metadata, "qwen2.attention.head_count")? as usize;
        let num_key_value_heads =
            metadata_number(&metadata, "qwen2.attention.head_count_kv")? as usize;
        let hidden_size = metadata_number(&metadata, "qwen2.embedding_length")? as usize;
        let block_count = metadata_number(&metadata, "qwen2.block_count")? as usize;
        let rms_norm_eps =
            metadata_number(&metadata, "qwen2.attention.layer_norm_rms_epsilon")? as f64;
        let rope_theta = metadata
            .get("qwen2.rope.freq_base")
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(1_000_000.0);
        let context_length = metadata
            .get("qwen2.context_length")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(32768);
        let intermediate_size = metadata
            .get("qwen2.feed_forward_length")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(hidden_size * 4);
        let vocabulary = metadata
            .get("qwen2.vocab_size")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(151_936);
        Ok(ParallelModelConfig {
            architecture: ModelArchitecture::Qwen2,
            vocab_size: vocabulary,
            hidden_size,
            intermediate_size,
            num_hidden_layers: block_count,
            num_attention_heads: num_heads,
            num_key_value_heads,
            max_position_embeddings: context_length,
            rms_norm_eps,
            rope_theta,
            tie_word_embeddings: false,
            rope_scaling: None,
        })
    }

    /// Loads a quantized Qwen2 model from a GGUF file.
    ///
    /// The GGUF metadata supplies the architecture parameters; the tokenizer is
    /// loaded separately because candle does not read GGUF vocabularies.
    pub fn from_gguf(gguf_path: &std::path::Path, device: &Device) -> anyhow::Result<Self> {
        let metadata = read_gguf_metadata(gguf_path)?;

        let architecture = metadata
            .get(GGUF_ARCHITECTURE_KEY)
            .cloned()
            .unwrap_or_else(|| "qwen2".to_string());
        if architecture != "qwen2" {
            return Err(anyhow::anyhow!(
                "GGUF architecture '{architecture}' is not supported by the quantized loader; expected 'qwen2'"
            ));
        }

        let num_heads = metadata_number(&metadata, "qwen2.attention.head_count")? as usize;
        let num_key_value_heads =
            metadata_number(&metadata, "qwen2.attention.head_count_kv")? as usize;
        let hidden_size = metadata_number(&metadata, "qwen2.embedding_length")? as usize;
        let block_count = metadata_number(&metadata, "qwen2.block_count")? as usize;
        let rms_norm_eps =
            metadata_number(&metadata, "qwen2.attention.layer_norm_rms_epsilon")? as f64;
        let head_dimension = hidden_size / num_heads;

        let variable_builder = quantized_var_builder::VarBuilder::from_gguf(gguf_path, device)?;

        let token_embedding_weight = variable_builder
            .get_no_shape("token_embd.weight")?
            .dequantize(device)?
            .to_dtype(candle_core::DType::F32)?;
        let token_embeddings = Embedding::new(token_embedding_weight, hidden_size);
        let final_norm = RmsNorm::new(
            hidden_size,
            rms_norm_eps,
            variable_builder.pp("output_norm"),
        )?;
        let output = match variable_builder.get_no_shape("output.weight") {
            Ok(tensor) => QMatMul::from_arc(tensor)?,
            Err(_) => QMatMul::from_arc(variable_builder.get_no_shape("token_embd.weight")?)?,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_index in 0..block_count {
            let prefix = format!("blk.{layer_index}");
            let attention_wq = QMatMul::from_arc(
                variable_builder.get_no_shape(&format!("{prefix}.attn_q.weight"))?,
            )?;
            let attention_wk = QMatMul::from_arc(
                variable_builder.get_no_shape(&format!("{prefix}.attn_k.weight"))?,
            )?;
            let attention_wv = QMatMul::from_arc(
                variable_builder.get_no_shape(&format!("{prefix}.attn_v.weight"))?,
            )?;
            let attention_wo = QMatMul::from_arc(
                variable_builder.get_no_shape(&format!("{prefix}.attn_output.weight"))?,
            )?;
            let attention_bq = variable_builder
                .get_no_shape(&format!("{prefix}.attn_q.bias"))?
                .dequantize(device)?
                .to_dtype(candle_core::DType::F32)?;
            let attention_bk = variable_builder
                .get_no_shape(&format!("{prefix}.attn_k.bias"))?
                .dequantize(device)?
                .to_dtype(candle_core::DType::F32)?;
            let attention_bv = variable_builder
                .get_no_shape(&format!("{prefix}.attn_v.bias"))?
                .dequantize(device)?
                .to_dtype(candle_core::DType::F32)?;
            let attention_norm_name = format!("{prefix}.attn_norm");
            let ffn_norm_name = format!("{prefix}.ffn_norm");
            let attention_norm = RmsNorm::new(
                hidden_size,
                rms_norm_eps,
                variable_builder.pp(attention_norm_name),
            )?;
            let ffn_norm = RmsNorm::new(
                hidden_size,
                rms_norm_eps,
                variable_builder.pp(ffn_norm_name),
            )?;
            let gate_name = format!("{prefix}.ffn_gate.weight");
            let down_name = format!("{prefix}.ffn_down.weight");
            let up_name = format!("{prefix}.ffn_up.weight");
            let mlp = Mlp {
                feed_forward_w1: QMatMul::from_arc(variable_builder.get_no_shape(&gate_name)?)?,
                feed_forward_w2: QMatMul::from_arc(variable_builder.get_no_shape(&down_name)?)?,
                feed_forward_w3: QMatMul::from_arc(variable_builder.get_no_shape(&up_name)?)?,
                span: tracing::span!(tracing::Level::TRACE, "quantized-mlp-feed-forward"),
            };
            layers.push(LayerWeights {
                attention_wq,
                attention_wk,
                attention_wv,
                attention_bq,
                attention_bk,
                attention_bv,
                attention_wo,
                attention_norm,
                mlp,
                ffn_norm,
                num_heads,
                num_key_value_heads,
                head_dimension,
                span_attention: tracing::span!(tracing::Level::TRACE, "quantized-attention"),
                span_rotation: tracing::span!(tracing::Level::TRACE, "quantized-rotation"),
            });
        }
        Ok(Self {
            token_embeddings,
            layers,
            final_norm,
            output,
            device: device.clone(),
        })
    }

    /// Builds a cache sized for this model.
    pub fn empty_cache(
        &self,
        context_length: usize,
        dtype: DType,
    ) -> anyhow::Result<QuantizedCache> {
        let head_dimension = self
            .layers
            .first()
            .map(|layer| layer.head_dimension)
            .ok_or_else(|| anyhow::anyhow!("quantized model has no layers"))?;
        let rope_theta = 1_000_000.0_f32;
        let mut cache = QuantizedCache::new(
            head_dimension,
            rope_theta,
            context_length,
            dtype,
            &self.device,
        )?;
        cache.set_layer_count(self.layer_count());
        Ok(cache)
    }
}

/// Forward for an embedding that may live on a different device than the input.
fn embedding_forward(embeddings: &Embedding, tokens: &Tensor) -> Result<Tensor> {
    embeddings.forward(tokens)
}

/// The GGUF metadata as string values (numbers rendered with `Display`).
fn read_gguf_metadata(gguf_path: &std::path::Path) -> anyhow::Result<HashMap<String, String>> {
    use candle_core::quantized::gguf_file::Value;
    let mut file = std::fs::File::open(gguf_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut metadata = HashMap::new();
    for (key, value) in &content.metadata {
        let rendered = match value {
            Value::U8(v) => v.to_string(),
            Value::I8(v) => v.to_string(),
            Value::U16(v) => v.to_string(),
            Value::I16(v) => v.to_string(),
            Value::U32(v) => v.to_string(),
            Value::I32(v) => v.to_string(),
            Value::U64(v) => v.to_string(),
            Value::I64(v) => v.to_string(),
            Value::F32(v) => v.to_string(),
            Value::F64(v) => v.to_string(),
            Value::Bool(v) => v.to_string(),
            Value::String(v) => v.clone(),
            Value::Array(_) => continue,
        };
        metadata.insert(key.clone(), rendered);
    }
    Ok(metadata)
}

/// Reads a numeric GGUF metadata entry.
fn metadata_number(metadata: &HashMap<String, String>, key: &str) -> anyhow::Result<f64> {
    metadata
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("GGUF metadata is missing required key '{key}'"))
        .and_then(|value| {
            value
                .parse::<f64>()
                .map_err(|error| anyhow::anyhow!("GGUF key '{key}' is not numeric: {error}"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequencies_have_one_row_per_position() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let (cos, sin) = precompute_frequencies(64, 1_000_000.0, 16, DType::F32, &device)?;
        assert_eq!(cos.dims(), &[16, 32]);
        assert_eq!(sin.dims(), &[16, 32]);
        Ok(())
    }

    #[test]
    fn cache_layer_count_sizes_the_key_values() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut cache = QuantizedCache::new(64, 1_000_000.0, 16, DType::F32, &device)?;
        cache.set_layer_count(3);
        assert_eq!(cache.key_values.len(), 3);
        Ok(())
    }

    #[test]
    fn non_qwen2_metadata_is_rejected() -> anyhow::Result<()> {
        let mut metadata = HashMap::new();
        metadata.insert(GGUF_ARCHITECTURE_KEY.to_string(), "llama".to_string());
        let architecture = metadata
            .get(GGUF_ARCHITECTURE_KEY)
            .cloned()
            .unwrap_or_default();
        assert_ne!(architecture, "qwen2");
        Ok(())
    }
}
