//! Differentiable full-sequence Qwen2 forward with LoRA adapters.
//!
//! Qwen2 shares the Llama decoder skeleton (RMSNorm, grouped-query attention,
//! SwiGLU MLP, causal mask) and differs only in that the `q_proj`/`k_proj`/
//! `v_proj` projections carry biases. The shared differentiable forward lives in
//! [`crate::model::trainable_llama`]; this module exposes the Qwen2 entry point
//! and validates that the configuration really requests attention biases, so a
//! Qwen2 config can never silently load a biasless model.
//!
//! The result is the same [`TrainableLlama`] type (the architecture only changes
//! the base tensor shapes and the bias flag), which keeps the training loop
//! architecture-agnostic.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use typed_lm_common::checkpoint::ModelArchitecture;
use typed_lm_common::model_config::ParallelModelConfig;

use crate::error::TrainerError;
use crate::model::trainable_llama::{LoRAConfiguration, TrainableLlama};

/// A trainable, LoRA-augmented Qwen2 model over a frozen base.
pub type TrainableQwen2 = TrainableLlama;

/// Builds a trainable Qwen2 model, requiring attention biases in the config.
pub fn load(
    base: &HashMap<String, Tensor>,
    config: &ParallelModelConfig,
    lora: LoRAConfiguration,
    variable_map: &mut VarMap,
    device: &Device,
) -> anyhow::Result<TrainableQwen2> {
    if config.architecture != ModelArchitecture::Qwen2 {
        return Err(TrainerError::Model(format!(
            "the Qwen2 loader received a '{}' configuration",
            config.architecture.name()
        ))
        .into());
    }
    if !config.has_query_key_value_bias() {
        return Err(TrainerError::Model(
            "the Qwen2 configuration must request query/key/value biases".to_string(),
        )
        .into());
    }
    TrainableLlama::load(base, config, lora, variable_map, device)
}

/// Whether the configuration drives a Qwen2 biased attention block.
pub fn uses_query_key_value_bias(config: &ParallelModelConfig) -> bool {
    config.architecture == ModelArchitecture::Qwen2 && config.has_query_key_value_bias()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen2_config() -> ParallelModelConfig {
        ParallelModelConfig {
            architecture: ModelArchitecture::Qwen2,
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
        }
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
            // Qwen2 biases q/k/v only.
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
        Ok(tensors)
    }

    #[test]
    fn qwen2_uses_query_key_value_bias() {
        assert!(uses_query_key_value_bias(&qwen2_config()));
    }

    #[test]
    fn biased_forward_produces_finite_logits() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let config = qwen2_config();
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

    #[test]
    fn a_non_qwen2_configuration_is_rejected() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut config = qwen2_config();
        config.architecture = ModelArchitecture::Llama;
        let result = load(
            &HashMap::new(),
            &config,
            LoRAConfiguration::new(2, 4.0),
            &mut VarMap::new(),
            &device,
        );
        assert!(result.is_err());
        Ok(())
    }
}
