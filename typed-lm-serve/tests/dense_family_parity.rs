//! Live parity checks between the vendored dense forward and the upstream
//! `candle-transformers` models for the families added by the multi-architecture
//! work: Qwen3, Mistral, Gemma, Gemma2 and Gemma3.
//!
//! Each test builds a tiny synthetic checkpoint with deterministic weights,
//! loads it through both our `ParallelLlama` and the matching candle model, and
//! asserts the last-position logits agree. There is no download, but the tests
//! are marked `#[ignore]` because they exercise the full forward and are kept
//! out of the default CI run.
//!
//! The synthetic geometry is small but realistic (`head_dim` is a power of two
//! greater than two, grouped query attention is exercised, Gemma3 alternates
//! local and global layers). `query_pre_attn_scalar` is set to `head_dim` for
//! Gemma2/Gemma3 so our scaling matches candle's `1/sqrt(head_dim)`; official
//! Gemma checkpoints use that same value.

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use typed_lm_common::checkpoint::ModelArchitecture;
use typed_lm_common::model_config::ParallelModelConfig;
use typed_lm_serve::infrastructure::parallel_llama::{ParallelCache, ParallelLlama};

const HIDDEN: usize = 32;
const HEADS: usize = 4;
const HEAD_DIM: usize = 8;
const KV_HEADS: usize = 2;
const LAYERS: usize = 3;
const INTERMEDIATE: usize = 64;
const VOCAB: usize = 128;
const MAX_POSITIONS: usize = 64;
const SLIDING_WINDOW: usize = 8;
const EPS: f64 = 1e-6;

/// Deterministic pseudo-random source shared by every weight tensor.
struct Deterministic {
    state: u64,
}

impl Deterministic {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    fn matrix(&mut self, rows: usize, columns: usize) -> anyhow::Result<Tensor> {
        let values: Vec<f32> = (0..rows * columns).map(|_| self.next() * 0.1).collect();
        Ok(Tensor::from_vec(values, (rows, columns), &Device::Cpu)?)
    }

    fn vector(&mut self, size: usize) -> anyhow::Result<Tensor> {
        let values: Vec<f32> = (0..size).map(|_| self.next() * 0.1 + 1.0).collect();
        Ok(Tensor::from_vec(values, size, &Device::Cpu)?)
    }
}

/// Builds the canonical synthetic weight map for a configuration.
fn synthetic_weights(config: &ParallelModelConfig) -> anyhow::Result<HashMap<String, Tensor>> {
    let mut source = Deterministic::new(0xfeed_beef);
    let head_dimension = config.head_dimension();
    let query_size = head_dimension * config.num_attention_heads;
    let key_value_size = head_dimension * config.num_key_value_heads;
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        source.matrix(config.vocab_size, config.hidden_size)?,
    );
    // Emitted for the families whose candle model is untied (Mistral); the tied
    // families ignore it.
    weights.insert(
        "lm_head.weight".to_string(),
        source.matrix(config.vocab_size, config.hidden_size)?,
    );
    weights.insert(
        "model.norm.weight".to_string(),
        source.vector(config.hidden_size)?,
    );
    for index in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{index}");
        weights.insert(
            format!("{prefix}.input_layernorm.weight"),
            source.vector(config.hidden_size)?,
        );
        weights.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            source.vector(config.hidden_size)?,
        );
        if config.gemma_block_layout {
            weights.insert(
                format!("{prefix}.pre_feedforward_layernorm.weight"),
                source.vector(config.hidden_size)?,
            );
            weights.insert(
                format!("{prefix}.post_feedforward_layernorm.weight"),
                source.vector(config.hidden_size)?,
            );
        }
        let attention = format!("{prefix}.self_attn");
        for (projection, output) in [
            ("q_proj", query_size),
            ("k_proj", key_value_size),
            ("v_proj", key_value_size),
        ] {
            weights.insert(
                format!("{attention}.{projection}.weight"),
                source.matrix(output, config.hidden_size)?,
            );
            if config.has_query_key_value_bias() {
                weights.insert(
                    format!("{attention}.{projection}.bias"),
                    source.vector(output)?,
                );
            }
        }
        if config.per_head_query_key_norm {
            weights.insert(
                format!("{attention}.q_norm.weight"),
                source.vector(head_dimension)?,
            );
            weights.insert(
                format!("{attention}.k_norm.weight"),
                source.vector(head_dimension)?,
            );
        }
        weights.insert(
            format!("{attention}.o_proj.weight"),
            source.matrix(config.hidden_size, query_size)?,
        );
        if config.output_projection_bias() {
            weights.insert(
                format!("{attention}.o_proj.bias"),
                source.vector(config.hidden_size)?,
            );
        }
        let mlp = format!("{prefix}.mlp");
        weights.insert(
            format!("{mlp}.gate_proj.weight"),
            source.matrix(config.intermediate_size, config.hidden_size)?,
        );
        weights.insert(
            format!("{mlp}.up_proj.weight"),
            source.matrix(config.intermediate_size, config.hidden_size)?,
        );
        weights.insert(
            format!("{mlp}.down_proj.weight"),
            source.matrix(config.hidden_size, config.intermediate_size)?,
        );
    }
    Ok(weights)
}

/// Asserts the vendored last-position logits match a reference forward.
fn assert_matches_reference(
    config: &ParallelModelConfig,
    tokens: &[u32],
    reference: impl FnOnce(&HashMap<String, Tensor>, &Tensor) -> anyhow::Result<Tensor>,
) -> anyhow::Result<()> {
    let device = Device::Cpu;
    let weights = synthetic_weights(config)?;
    let tensor = Tensor::new(tokens, &device)?.unsqueeze(0)?;

    let reference_logits = reference(&weights, &tensor)?;

    let builder = VarBuilder::from_tensors(weights, DType::F32, &device);
    let model = ParallelLlama::load(builder, config)?;
    let mut cache = ParallelCache::new(true, DType::F32, config, &device)?;
    let vendored_logits = model.forward_last(&tensor, 0, &mut cache)?;

    for token_id in [0_usize, 1, 7, VOCAB / 2, VOCAB - 1] {
        let reference_value = reference_logits.i((0, 0, token_id))?.to_vec0::<f32>()?;
        let vendored_value = vendored_logits.i((0, token_id))?.to_vec0::<f32>()?;
        let tolerance = 1e-4 + 1e-3 * reference_value.abs().max(vendored_value.abs());
        assert!(
            (reference_value - vendored_value).abs() < tolerance,
            "token {token_id}: candle {reference_value} vs vendored {vendored_value}"
        );
    }
    Ok(())
}

fn qwen3_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "head_dim": HEAD_DIM,
        "num_key_value_heads": KV_HEADS,
        "max_position_embeddings": MAX_POSITIONS,
        "rms_norm_eps": EPS,
        "rope_theta": 1000000.0,
        "attention_bias": false,
        "tie_word_embeddings": true,
        "sliding_window": SLIDING_WINDOW,
        "max_window_layers": LAYERS,
        "use_sliding_window": false,
        "hidden_act": "silu"
    })
}

fn mistral_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "mistral",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "head_dim": HEAD_DIM,
        "num_key_value_heads": KV_HEADS,
        "max_position_embeddings": MAX_POSITIONS,
        "rms_norm_eps": EPS,
        "rope_theta": 10000.0,
        "sliding_window": SLIDING_WINDOW,
        "hidden_act": "silu"
    })
}

fn gemma_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma",
        "attention_bias": false,
        "head_dim": HEAD_DIM,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": EPS,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_POSITIONS,
        "tie_word_embeddings": true,
        "hidden_activation": "gelu_pytorch_tanh"
    })
}

fn gemma2_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma2",
        "attention_bias": true,
        "head_dim": HEAD_DIM,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": EPS,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
        "final_logit_softcapping": 30.0,
        "attn_logit_softcapping": 50.0,
        "query_pre_attn_scalar": HEAD_DIM,
        "sliding_window": SLIDING_WINDOW,
        "max_position_embeddings": MAX_POSITIONS,
        "tie_word_embeddings": true
    })
}

fn gemma3_json() -> serde_json::Value {
    // `num_key_value_heads` is 1: candle 0.11's `gemma3::Attention` passes a
    // transposed, non-contiguous `value_states` to `KvCache::append`, whose
    // `slice_set` requires contiguous tensors. With a single key/value head the
    // transposed view is still contiguous, so the upstream forward runs and the
    // Gemma3-specific behaviour (four-norm block, per-head q/k norm, local vs
    // global RoPE) is exercised. Our own forward has no such restriction.
    serde_json::json!({
        "model_type": "gemma3",
        "attention_bias": false,
        "head_dim": HEAD_DIM,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": 1,
        "rms_norm_eps": EPS,
        "rope_theta": 1000000.0,
        "rope_local_base_freq": 10000.0,
        "vocab_size": VOCAB,
        "final_logit_softcapping": 30.0,
        "attn_logit_softcapping": 50.0,
        "query_pre_attn_scalar": HEAD_DIM,
        "sliding_window": SLIDING_WINDOW,
        "sliding_window_pattern": 2,
        "max_position_embeddings": MAX_POSITIONS,
        "tie_word_embeddings": true
    })
}

#[test]
#[ignore]
fn vendored_forward_matches_candle_qwen3() -> anyhow::Result<()> {
    use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
    let json = qwen3_json();
    let reference_config: Config = serde_json::from_value(json.clone())?;
    let config = ParallelModelConfig::from_json_for_architecture(json, ModelArchitecture::Qwen3)?;
    assert_matches_reference(&config, &[3, 9, 11, 5], |weights, tokens| {
        let builder = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
        let mut model = ModelForCausalLM::new(&reference_config, builder)?;
        Ok(model.forward(tokens, 0)?)
    })
}

#[test]
#[ignore]
fn vendored_forward_matches_candle_mistral() -> anyhow::Result<()> {
    use candle_transformers::models::mistral::{Config, Model};
    let json = mistral_json();
    let reference_config: Config = serde_json::from_value(json.clone())?;
    let config = ParallelModelConfig::from_json_for_architecture(json, ModelArchitecture::Mistral)?;
    assert_matches_reference(&config, &[3, 9, 11, 5], |weights, tokens| {
        let builder = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
        let mut model = Model::new(&reference_config, builder)?;
        Ok(model.forward(tokens, 0)?)
    })
}

#[test]
#[ignore]
fn vendored_forward_matches_candle_gemma() -> anyhow::Result<()> {
    use candle_transformers::models::gemma::{Config, Model};
    let json = gemma_json();
    let reference_config: Config = serde_json::from_value(json.clone())?;
    let config = ParallelModelConfig::from_json_for_architecture(json, ModelArchitecture::Gemma)?;
    assert_matches_reference(&config, &[3, 9, 11, 5], |weights, tokens| {
        let builder = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
        let mut model = Model::new(false, &reference_config, builder)?;
        Ok(model.forward(tokens, 0)?)
    })
}

#[test]
#[ignore]
fn vendored_forward_matches_candle_gemma2() -> anyhow::Result<()> {
    use candle_transformers::models::gemma2::{Config, Model};
    let json = gemma2_json();
    let reference_config: Config = serde_json::from_value(json.clone())?;
    let config = ParallelModelConfig::from_json_for_architecture(json, ModelArchitecture::Gemma2)?;
    assert_matches_reference(&config, &[3, 9, 11, 5], |weights, tokens| {
        let builder = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
        let mut model = Model::new(false, &reference_config, builder)?;
        Ok(model.forward(tokens, 0)?)
    })
}

#[test]
#[ignore]
fn vendored_forward_matches_candle_gemma3() -> anyhow::Result<()> {
    use candle_transformers::models::gemma3::{Config, Model};
    let json = gemma3_json();
    let reference_config: Config = serde_json::from_value(json.clone())?;
    let config = ParallelModelConfig::from_json_for_architecture(json, ModelArchitecture::Gemma3)?;
    assert_matches_reference(&config, &[3, 9, 11, 5], |weights, tokens| {
        let builder = VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu);
        let mut model = Model::new(false, &reference_config, builder)?;
        Ok(model.forward(tokens, 0)?)
    })
}
