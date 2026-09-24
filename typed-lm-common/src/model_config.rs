//! Architecture-neutral model configuration for the vendored parallel forward.
//!
//! Llama and Qwen2 differ in a few details that matter to the forward pass:
//!
//! - Qwen2 biases its `q_proj`/`k_proj`/`v_proj`; Llama does not.
//! - Qwen2 places RoPE on `head_dim = hidden_size / num_attention_heads`
//!   exactly like Llama, so the inverse frequencies are identical.
//!
//! Everything else (RMSNorm, GQA, SwiGLU MLP, causal mask) is shared. This
//! struct captures the shared surface so `ParallelLlama` can serve both
//! architectures without duplicating the block code.

use candle_transformers::models::gemma::Config as GemmaConfig;
use candle_transformers::models::gemma2::Config as Gemma2Config;
use candle_transformers::models::gemma3::Config as Gemma3Config;
use candle_transformers::models::llama::{
    Config as LlamaRuntimeConfig, Llama3RopeConfig, Llama3RopeType, LlamaConfig,
};
use candle_transformers::models::mistral::Config as MistralConfig;
use candle_transformers::models::qwen2::Config as Qwen2Config;
use candle_transformers::models::qwen3::Config as Qwen3Config;

use crate::checkpoint::ModelArchitecture;

/// Architecture-neutral configuration consumed by the vendored forward pass.
///
/// This captures the union of the dense decoder families the workspace
/// supports. The base attention surface (RMSNorm, GQA, SwiGLU MLP, causal mask)
/// is shared; the fields after `rope_scaling` describe the per-family
/// variations handled by [`crate::architecture_traits::DenseArchitectureTraits`].
#[derive(Debug, Clone)]
pub struct ParallelModelConfig {
    pub architecture: ModelArchitecture,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    /// Optional Llama 3-style RoPE scaling (absent for Qwen2 and Llama 1/2).
    pub rope_scaling: Option<Llama3RopeConfig>,
    /// Attention projections carry biases (Qwen2; also Qwen3/Gemma per config).
    pub attention_bias: bool,
    /// Explicit head dimension (Qwen3, Mistral, Gemma*); `None` derives it from
    /// `hidden_size / num_attention_heads` (Llama, Qwen2).
    pub explicit_head_dimension: Option<usize>,
    /// Sliding-window attention size, when the family uses it.
    pub sliding_window: Option<usize>,
    /// Number of leading layers that do **not** use the sliding window (Qwen3).
    pub max_window_layers: usize,
    /// Gemma2/Gemma3 `final_logit_softcapping` (`tanh`-based logit cap).
    pub logit_softcapping: Option<f64>,
    /// Gemma2/Gemma3 `attn_logit_softcapping`.
    pub attention_logit_softcapping: Option<f64>,
    /// Gemma2/Gemma3 attention scaling denominator (`query_pre_attn_scalar`).
    pub query_pre_attention_scalar: Option<usize>,
    /// Gemma* use `(1 + weight)` instead of `weight` in RMSNorm.
    pub rms_norm_unit_offset: bool,
    /// Gemma* scale embeddings by `sqrt(hidden_size)`.
    pub embedding_scale: Option<f64>,
    /// Gemma3 local RoPE base frequency.
    pub rope_local_base_frequency: Option<f64>,
}

impl ParallelModelConfig {
    /// Attention projections carry biases (Qwen2 does, Llama does not).
    ///
    /// Reads the concrete flag parsed from `config.json`; do not use
    /// [`ModelArchitecture::has_query_key_value_bias`] here, because Qwen3 and
    /// Gemma* expose this through their config.
    pub fn has_query_key_value_bias(&self) -> bool {
        self.attention_bias
    }

    /// Head dimension used by RoPE and the attention reshape.
    ///
    /// Prefers the explicit value when the family provides one; otherwise it is
    /// `hidden_size / num_attention_heads`.
    pub fn head_dimension(&self) -> usize {
        self.explicit_head_dimension
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Builds the config from a Llama `config.json` value.
    pub fn from_llama_json(value: serde_json::Value) -> anyhow::Result<Self> {
        let llama_config: LlamaConfig = serde_json::from_value(value)
            .map_err(|error| anyhow::anyhow!("failed to parse the Llama configuration: {error}"))?;
        let runtime: LlamaRuntimeConfig = llama_config.into_config(false);
        let rope_scaling = match &runtime.rope_scaling {
            Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            })
            | None => None,
            Some(scaling) => Some(scaling.clone()),
        };
        Ok(Self {
            architecture: ModelArchitecture::Llama,
            vocab_size: runtime.vocab_size,
            hidden_size: runtime.hidden_size,
            intermediate_size: runtime.intermediate_size,
            num_hidden_layers: runtime.num_hidden_layers,
            num_attention_heads: runtime.num_attention_heads,
            num_key_value_heads: runtime.num_key_value_heads,
            max_position_embeddings: runtime.max_position_embeddings,
            rms_norm_eps: runtime.rms_norm_eps,
            rope_theta: runtime.rope_theta,
            tie_word_embeddings: runtime.tie_word_embeddings,
            rope_scaling,
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
        })
    }

    /// Builds the config from a Qwen2 `config.json` value.
    pub fn from_qwen2_json(value: serde_json::Value) -> anyhow::Result<Self> {
        let config: Qwen2Config = serde_json::from_value(value)
            .map_err(|error| anyhow::anyhow!("failed to parse the Qwen2 configuration: {error}"))?;
        Ok(Self {
            architecture: ModelArchitecture::Qwen2,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta as f32,
            tie_word_embeddings: config.tie_word_embeddings,
            rope_scaling: None,
            attention_bias: true,
            explicit_head_dimension: None,
            sliding_window: None,
            max_window_layers: 0,
            logit_softcapping: None,
            attention_logit_softcapping: None,
            query_pre_attention_scalar: None,
            rms_norm_unit_offset: false,
            embedding_scale: None,
            rope_local_base_frequency: None,
        })
    }

    /// Builds the config from a Qwen3 `config.json` value.
    pub fn from_qwen3_json(value: serde_json::Value) -> anyhow::Result<Self> {
        let config: Qwen3Config = serde_json::from_value(value)
            .map_err(|error| anyhow::anyhow!("failed to parse the Qwen3 configuration: {error}"))?;
        let sliding_window = if config.use_sliding_window {
            config.sliding_window
        } else {
            None
        };
        Ok(Self {
            architecture: ModelArchitecture::Qwen3,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta as f32,
            tie_word_embeddings: config.tie_word_embeddings,
            rope_scaling: None,
            attention_bias: config.attention_bias,
            explicit_head_dimension: Some(config.head_dim),
            sliding_window,
            max_window_layers: config.max_window_layers,
            logit_softcapping: None,
            attention_logit_softcapping: None,
            query_pre_attention_scalar: None,
            rms_norm_unit_offset: false,
            embedding_scale: None,
            rope_local_base_frequency: None,
        })
    }

    /// Builds the config from a Mistral `config.json` value.
    pub fn from_mistral_json(value: serde_json::Value) -> anyhow::Result<Self> {
        let config: MistralConfig = serde_json::from_value(value).map_err(|error| {
            anyhow::anyhow!("failed to parse the Mistral configuration: {error}")
        })?;
        Ok(Self {
            architecture: ModelArchitecture::Mistral,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta as f32,
            tie_word_embeddings: false,
            rope_scaling: None,
            attention_bias: false,
            explicit_head_dimension: config.head_dim,
            sliding_window: config.sliding_window,
            max_window_layers: 0,
            logit_softcapping: None,
            attention_logit_softcapping: None,
            query_pre_attention_scalar: None,
            rms_norm_unit_offset: false,
            embedding_scale: None,
            rope_local_base_frequency: None,
        })
    }

    /// Builds the config from a Gemma `config.json` value.
    pub fn from_gemma_json(value: serde_json::Value) -> anyhow::Result<Self> {
        let config: GemmaConfig = serde_json::from_value(value)
            .map_err(|error| anyhow::anyhow!("failed to parse the Gemma configuration: {error}"))?;
        Ok(Self {
            architecture: ModelArchitecture::Gemma,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta as f32,
            tie_word_embeddings: true,
            rope_scaling: None,
            attention_bias: config.attention_bias,
            explicit_head_dimension: Some(config.head_dim),
            sliding_window: None,
            max_window_layers: 0,
            logit_softcapping: None,
            attention_logit_softcapping: None,
            query_pre_attention_scalar: None,
            rms_norm_unit_offset: true,
            embedding_scale: Some((config.hidden_size as f64).sqrt()),
            rope_local_base_frequency: None,
        })
    }

    /// Builds the config from a Gemma2 `config.json` value.
    pub fn from_gemma2_json(value: serde_json::Value) -> anyhow::Result<Self> {
        let config: Gemma2Config = serde_json::from_value(value).map_err(|error| {
            anyhow::anyhow!("failed to parse the Gemma2 configuration: {error}")
        })?;
        Ok(Self {
            architecture: ModelArchitecture::Gemma2,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta as f32,
            tie_word_embeddings: true,
            rope_scaling: None,
            attention_bias: config.attention_bias,
            explicit_head_dimension: Some(config.head_dim),
            sliding_window: config.sliding_window,
            max_window_layers: 0,
            logit_softcapping: config.final_logit_softcapping,
            attention_logit_softcapping: config.attn_logit_softcapping,
            query_pre_attention_scalar: Some(config.query_pre_attn_scalar),
            rms_norm_unit_offset: true,
            embedding_scale: Some((config.hidden_size as f64).sqrt()),
            rope_local_base_frequency: None,
        })
    }

    /// Builds the config from a Gemma3 `config.json` value.
    pub fn from_gemma3_json(value: serde_json::Value) -> anyhow::Result<Self> {
        let config: Gemma3Config = serde_json::from_value(value).map_err(|error| {
            anyhow::anyhow!("failed to parse the Gemma3 configuration: {error}")
        })?;
        Ok(Self {
            architecture: ModelArchitecture::Gemma3,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta as f32,
            tie_word_embeddings: true,
            rope_scaling: None,
            attention_bias: config.attention_bias,
            explicit_head_dimension: Some(config.head_dim),
            sliding_window: Some(config.sliding_window),
            max_window_layers: 0,
            logit_softcapping: config.final_logit_softcapping,
            attention_logit_softcapping: config.attn_logit_softcapping,
            query_pre_attention_scalar: Some(config.query_pre_attn_scalar),
            rms_norm_unit_offset: true,
            embedding_scale: Some((config.hidden_size as f64).sqrt()),
            rope_local_base_frequency: Some(config.rope_local_base_freq),
        })
    }

    /// Builds the config from a `config.json` body and its detected architecture.
    pub fn from_json_for_architecture(
        value: serde_json::Value,
        architecture: ModelArchitecture,
    ) -> anyhow::Result<Self> {
        match architecture {
            ModelArchitecture::Llama => Self::from_llama_json(value),
            ModelArchitecture::Qwen2 => Self::from_qwen2_json(value),
            ModelArchitecture::Qwen3 => Self::from_qwen3_json(value),
            ModelArchitecture::Mistral => Self::from_mistral_json(value),
            ModelArchitecture::Gemma => Self::from_gemma_json(value),
            ModelArchitecture::Gemma2 => Self::from_gemma2_json(value),
            ModelArchitecture::Gemma3 => Self::from_gemma3_json(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_json_becomes_biasless_config() -> anyhow::Result<()> {
        let value = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 100,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        });
        let config = ParallelModelConfig::from_llama_json(value)?;
        assert_eq!(config.architecture, ModelArchitecture::Llama);
        assert!(!config.has_query_key_value_bias());
        assert_eq!(config.head_dimension(), 8);
        Ok(())
    }

    #[test]
    fn qwen2_json_becomes_biased_config() -> anyhow::Result<()> {
        let value = serde_json::json!({
            "model_type": "qwen2",
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true,
            "sliding_window": 32768,
            "max_window_layers": 21,
            "use_sliding_window": false,
            "hidden_act": "silu"
        });
        let config = ParallelModelConfig::from_qwen2_json(value)?;
        assert_eq!(config.architecture, ModelArchitecture::Qwen2);
        assert!(config.has_query_key_value_bias());
        assert_eq!(config.head_dimension(), 64);
        Ok(())
    }

    #[test]
    fn qwen3_json_carries_explicit_head_dimension_and_window() -> anyhow::Result<()> {
        let value = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "head_dim": 128,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "attention_bias": false,
            "tie_word_embeddings": true,
            "sliding_window": 32768,
            "max_window_layers": 28,
            "use_sliding_window": true,
            "hidden_act": "silu"
        });
        let config = ParallelModelConfig::from_qwen3_json(value)?;
        assert_eq!(config.architecture, ModelArchitecture::Qwen3);
        assert!(!config.has_query_key_value_bias());
        assert_eq!(config.head_dimension(), 128);
        assert_eq!(config.sliding_window, Some(32768));
        assert_eq!(config.max_window_layers, 28);
        Ok(())
    }

    #[test]
    fn qwen3_without_use_sliding_window_has_no_window() -> anyhow::Result<()> {
        let value = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "head_dim": 128,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "attention_bias": false,
            "tie_word_embeddings": true,
            "sliding_window": 32768,
            "max_window_layers": 28,
            "use_sliding_window": false,
            "hidden_act": "silu"
        });
        let config = ParallelModelConfig::from_qwen3_json(value)?;
        assert_eq!(config.sliding_window, None);
        Ok(())
    }

    #[test]
    fn mistral_json_is_biasless_with_window() -> anyhow::Result<()> {
        let value = serde_json::json!({
            "model_type": "mistral",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "hidden_act": "silu"
        });
        let config = ParallelModelConfig::from_mistral_json(value)?;
        assert_eq!(config.architecture, ModelArchitecture::Mistral);
        assert!(!config.has_query_key_value_bias());
        assert_eq!(config.head_dimension(), 128);
        assert_eq!(config.sliding_window, Some(4096));
        assert!(!config.tie_word_embeddings);
        Ok(())
    }

    #[test]
    fn gemma_json_uses_offset_norm_and_embedding_scale() -> anyhow::Result<()> {
        let value = serde_json::json!({
            "model_type": "gemma",
            "attention_bias": false,
            "head_dim": 256,
            "hidden_size": 2048,
            "intermediate_size": 16384,
            "num_attention_heads": 8,
            "num_hidden_layers": 18,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "vocab_size": 256000,
            "max_position_embeddings": 8192,
            "hidden_act": "gelu_pytorch_tanh"
        });
        let config = ParallelModelConfig::from_gemma_json(value)?;
        assert_eq!(config.architecture, ModelArchitecture::Gemma);
        assert!(config.rms_norm_unit_offset);
        assert_eq!(config.head_dimension(), 256);
        let scale = config
            .embedding_scale
            .ok_or_else(|| anyhow::anyhow!("gemma must carry an embedding scale"))?;
        assert!((scale - (2048f64).sqrt()).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn gemma2_json_carries_softcapping_and_scalar() -> anyhow::Result<()> {
        let value = serde_json::json!({
            "model_type": "gemma2",
            "attention_bias": false,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 2304,
            "intermediate_size": 9216,
            "num_attention_heads": 8,
            "num_hidden_layers": 26,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "vocab_size": 256000,
            "final_logit_softcapping": 30.0,
            "attn_logit_softcapping": 50.0,
            "query_pre_attn_scalar": 256,
            "sliding_window": 4096,
            "max_position_embeddings": 8192
        });
        let config = ParallelModelConfig::from_gemma2_json(value)?;
        assert_eq!(config.architecture, ModelArchitecture::Gemma2);
        assert_eq!(config.logit_softcapping, Some(30.0));
        assert_eq!(config.attention_logit_softcapping, Some(50.0));
        assert_eq!(config.query_pre_attention_scalar, Some(256));
        assert_eq!(config.sliding_window, Some(4096));
        assert!(config.rms_norm_unit_offset);
        assert!(config.rope_local_base_frequency.is_none());
        Ok(())
    }

    #[test]
    fn gemma3_json_carries_local_rope_base_frequency() -> anyhow::Result<()> {
        let value = serde_json::json!({
            "model_type": "gemma3",
            "attention_bias": false,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "intermediate_size": 6912,
            "num_attention_heads": 4,
            "num_hidden_layers": 26,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "vocab_size": 262144,
            "final_logit_softcapping": 30.0,
            "attn_logit_softcapping": 50.0,
            "query_pre_attn_scalar": 256,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
            "max_position_embeddings": 32768
        });
        let config = ParallelModelConfig::from_gemma3_json(value)?;
        assert_eq!(config.architecture, ModelArchitecture::Gemma3);
        assert_eq!(config.rope_local_base_frequency, Some(10000.0));
        assert_eq!(config.sliding_window, Some(1024));
        assert_eq!(config.query_pre_attention_scalar, Some(256));
        Ok(())
    }

    #[test]
    fn dispatcher_routes_every_architecture() -> anyhow::Result<()> {
        let minimal = |model_type: &str| {
            serde_json::json!({
                "model_type": model_type,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "head_dim": 8,
                "num_key_value_heads": 2,
                "vocab_size": 100,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "attention_bias": false,
                "tie_word_embeddings": true,
                "hidden_act": "silu",
                "hidden_activation": "silu",
                "sliding_window": 128,
                "max_window_layers": 2,
                "use_sliding_window": false,
                "final_logit_softcapping": 30.0,
                "attn_logit_softcapping": 50.0,
                "query_pre_attn_scalar": 64,
                "rope_local_base_freq": 10000.0,
                "sliding_window_pattern": 6
            })
        };
        for architecture in ModelArchitecture::SUPPORTED {
            let config = ParallelModelConfig::from_json_for_architecture(
                minimal(architecture.name()),
                architecture,
            )?;
            assert_eq!(config.architecture, architecture);
        }
        Ok(())
    }
}
