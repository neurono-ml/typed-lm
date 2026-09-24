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

use candle_transformers::models::llama::{
    Config as LlamaRuntimeConfig, Llama3RopeConfig, Llama3RopeType, LlamaConfig,
};
use candle_transformers::models::qwen2::Config as Qwen2Config;

use crate::infrastructure::checkpoint::ModelArchitecture;

/// Architecture-neutral configuration consumed by the vendored forward pass.
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
}

impl ParallelModelConfig {
    /// Attention projections carry biases (Qwen2 does, Llama does not).
    pub fn has_query_key_value_bias(&self) -> bool {
        self.architecture.has_query_key_value_bias()
    }

    /// Head dimension used by RoPE and the attention reshape.
    pub fn head_dimension(&self) -> usize {
        self.hidden_size / self.num_attention_heads
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
}
