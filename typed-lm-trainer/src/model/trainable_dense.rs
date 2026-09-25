//! Unified differentiable dense loader for every supported family.
//!
//! All supported families share the Llama decoder skeleton (RMSNorm,
//! grouped-query attention, SwiGLU MLP, causal mask) and differ only in the
//! capability flags captured by [`ParallelModelConfig`] and
//! [`DenseArchitectureTraits`]. The shared differentiable forward lives in
//! [`crate::model::trainable_llama`]; this module exposes a single entry point
//! that validates the configuration against the detected architecture before
//! delegating, so a checkpoint can never silently load a model whose forward
//! disagrees with its `config.json`.
//!
//! The result is the same [`TrainableLlama`] type (the architecture only changes
//! the base tensor shapes and the capability flags), which keeps the training
//! loop architecture-agnostic.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use typed_lm_common::architecture_traits::DenseArchitectureTraits;
use typed_lm_common::checkpoint::ModelArchitecture;
use typed_lm_common::model_config::ParallelModelConfig;

use crate::error::TrainerError;
use crate::model::trainable_llama::{LoRAConfiguration, TrainableLlama};

/// A trainable, LoRA-augmented dense model over a frozen base.
pub type TrainableDense = TrainableLlama;

/// Builds a trainable dense model for any supported architecture.
///
/// The configuration's architecture must be one the dense forward supports, and
/// the capability flags implied by that architecture must be consistent with the
/// parsed configuration (for example Qwen2 must request attention biases). The
/// checks fail with an actionable [`TrainerError::Model`] rather than producing a
/// numerically wrong forward pass.
pub fn load(
    base: &HashMap<String, Tensor>,
    config: &ParallelModelConfig,
    lora: LoRAConfiguration,
    variable_map: &mut VarMap,
    device: &Device,
) -> anyhow::Result<TrainableDense> {
    validate_configuration(config)?;
    TrainableLlama::load(base, config, lora, variable_map, device)
}

/// Validates that the configuration matches its detected architecture.
pub fn validate_configuration(config: &ParallelModelConfig) -> anyhow::Result<()> {
    if !is_supported_architecture(config.architecture) {
        return Err(TrainerError::Model(format!(
            "the dense loader received an unsupported architecture '{}'",
            config.architecture.name()
        ))
        .into());
    }
    let traits = DenseArchitectureTraits::for_architecture(config.architecture);
    if traits.explicit_head_dimension && config.explicit_head_dimension.is_none() {
        return Err(TrainerError::Model(format!(
            "the '{}' configuration must carry an explicit head dimension",
            config.architecture.name()
        ))
        .into());
    }
    if config.architecture == ModelArchitecture::Qwen2 && !config.has_query_key_value_bias() {
        return Err(TrainerError::Model(
            "the Qwen2 configuration must request query/key/value biases".to_string(),
        )
        .into());
    }
    Ok(())
}

/// Whether the dense trainable forward implements the given architecture.
pub fn is_supported_architecture(architecture: ModelArchitecture) -> bool {
    ModelArchitecture::SUPPORTED.contains(&architecture)
}

/// Whether the configuration drives a biased attention block.
pub fn uses_query_key_value_bias(config: &ParallelModelConfig) -> bool {
    config.has_query_key_value_bias()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_config(architecture: ModelArchitecture) -> ParallelModelConfig {
        let explicit_head_dimension = DenseArchitectureTraits::for_architecture(architecture)
            .explicit_head_dimension
            .then_some(8);
        ParallelModelConfig {
            architecture,
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            rope_scaling: None,
            attention_bias: architecture.has_query_key_value_bias(),
            explicit_head_dimension,
            sliding_window: None,
            max_window_layers: 0,
            logit_softcapping: None,
            attention_logit_softcapping: None,
            query_pre_attention_scalar: None,
            rms_norm_unit_offset: false,
            embedding_scale: None,
            hidden_activation: ParallelModelConfig::default_hidden_activation(architecture),
            rope_local_base_frequency: None,
            gemma_block_layout: false,
            per_head_query_key_norm: false,
            sliding_window_pattern: 0,
        }
    }

    #[test]
    fn every_supported_family_is_accepted() -> anyhow::Result<()> {
        for architecture in ModelArchitecture::SUPPORTED {
            assert!(is_supported_architecture(architecture));
            validate_configuration(&dense_config(architecture))?;
        }
        Ok(())
    }

    #[test]
    fn explicit_head_dimension_is_required_for_the_new_families() -> anyhow::Result<()> {
        for architecture in [
            ModelArchitecture::Qwen3,
            ModelArchitecture::Mistral,
            ModelArchitecture::Gemma,
            ModelArchitecture::Gemma2,
            ModelArchitecture::Gemma3,
        ] {
            let mut config = dense_config(architecture);
            config.explicit_head_dimension = None;
            assert!(validate_configuration(&config).is_err());
        }
        Ok(())
    }

    #[test]
    fn qwen2_must_request_biases() -> anyhow::Result<()> {
        let mut config = dense_config(ModelArchitecture::Qwen2);
        assert!(uses_query_key_value_bias(&config));
        config.attention_bias = false;
        assert!(validate_configuration(&config).is_err());
        Ok(())
    }

    #[test]
    fn a_biased_forward_produces_finite_logits() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let config = dense_config(ModelArchitecture::Qwen2);
        let base = biased_base(&config)?;
        let mut variable_map = VarMap::new();
        let model = load(
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
            assert!(value.is_finite());
        }
        Ok(())
    }

    fn biased_base(config: &ParallelModelConfig) -> anyhow::Result<HashMap<String, Tensor>> {
        let head_dimension = config.head_dimension();
        let query_size = head_dimension * config.num_attention_heads;
        let key_value_size = head_dimension * config.num_key_value_heads;
        let mut seed = 0xfeed_beef_u64;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            deterministic_tensor((config.vocab_size, config.hidden_size), &mut seed, 0.1)?,
        );
        tensors.insert(
            "model.norm.weight".to_string(),
            deterministic_vector(config.hidden_size, &mut seed)?,
        );
        for index in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{index}");
            tensors.insert(
                format!("{prefix}.input_layernorm.weight"),
                deterministic_vector(config.hidden_size, &mut seed)?,
            );
            tensors.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                deterministic_vector(config.hidden_size, &mut seed)?,
            );
            for (projection, output, input) in [
                ("self_attn.q_proj", query_size, config.hidden_size),
                ("self_attn.k_proj", key_value_size, config.hidden_size),
                ("self_attn.v_proj", key_value_size, config.hidden_size),
                ("self_attn.o_proj", config.hidden_size, query_size),
                (
                    "mlp.gate_proj",
                    config.intermediate_size,
                    config.hidden_size,
                ),
                ("mlp.up_proj", config.intermediate_size, config.hidden_size),
                (
                    "mlp.down_proj",
                    config.hidden_size,
                    config.intermediate_size,
                ),
            ] {
                tensors.insert(
                    format!("{prefix}.{projection}.weight"),
                    deterministic_tensor((output, input), &mut seed, 0.1)?,
                );
            }
            if config.has_query_key_value_bias() {
                for (projection, size) in [
                    ("self_attn.q_proj", query_size),
                    ("self_attn.k_proj", key_value_size),
                    ("self_attn.v_proj", key_value_size),
                ] {
                    tensors.insert(
                        format!("{prefix}.{projection}.bias"),
                        deterministic_tensor((size, 1), &mut seed, 0.1)?.reshape(size)?,
                    );
                }
            }
        }
        Ok(tensors)
    }

    fn deterministic_tensor(
        shape: (usize, usize),
        seed: &mut u64,
        scale: f32,
    ) -> anyhow::Result<Tensor> {
        let device = Device::Cpu;
        let count = shape.0 * shape.1;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            values.push((((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * scale);
        }
        Ok(Tensor::from_vec(values, shape, &device)?)
    }

    fn deterministic_vector(size: usize, seed: &mut u64) -> anyhow::Result<Tensor> {
        let device = Device::Cpu;
        let mut values = Vec::with_capacity(size);
        for _ in 0..size {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            values.push(1.0 + (((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 0.1);
        }
        Ok(Tensor::from_vec(values, size, &device)?)
    }
}
