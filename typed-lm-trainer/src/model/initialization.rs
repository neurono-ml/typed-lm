//! Deterministic, dependency-free weight initialization for from-scratch
//! training.
//!
//! The trainer sometimes needs to start from random weights rather than a
//! pretrained checkpoint. Adding a full random-number-generation dependency
//! would be unnecessary: a small fixed-seed linear-congruential generator
//! (LCG) plus a Box-Muller normal transform gives a sequence that is fully
//! reproducible across machines and runs. The same `(config, configuration,
//! seed)` triple always produces byte-identical tensors.
//!
//! The initializer writes the canonical HuggingFace tensor names the trainable
//! forward reads, so the resulting map can be handed straight to
//! [`crate::model::trainable_llama::TrainableLlama::load`] (or saved as a
//! safetensors checkpoint) without any renaming step.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use typed_lm_common::model_config::ParallelModelConfig;

use crate::error::TrainerError;

/// Hyper-parameters for the deterministic weight initializer.
///
/// The `*_std` fields are the **standard deviation** of a zero-mean normal
/// distribution approximated by the deterministic sampler. They are not a
/// variance or a uniform range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitializationConfiguration {
    /// Standard deviation of the attention and MLP projection weights.
    pub initializer_range: f64,
    /// Standard deviation of the token-embedding (and untied language-model
    /// head) weights.
    pub embedding_std: f64,
    /// Constant value written to every RMSNorm weight.
    pub norm_weight: f64,
    /// Constant value written to every attention-projection bias (Qwen2 and the
    /// other families that carry query/key/value biases).
    pub bias_value: f64,
}

impl Default for InitializationConfiguration {
    fn default() -> Self {
        Self {
            initializer_range: 0.02,
            embedding_std: 0.02,
            norm_weight: 1.0,
            bias_value: 0.0,
        }
    }
}

/// A small, seedable linear-congruential generator with a normal transform.
///
/// The generator is intentionally minimal: a 64-bit LCG (the same constants
/// used by the `rand` crate's `Lcg64Xsh32::new`) advanced by wrapping
/// multiplication and addition. `next_unit` maps the top 53 bits to `[0, 1)`,
/// which is exactly the mantissa width of an `f64`, so every representable
/// IEEE-754 value in `[0, 1)` is reachable and the mapping is exact.
///
/// The same seed always yields the same sequence; different seeds diverge.
#[derive(Debug, Clone)]
pub struct DeterministicInitializer {
    state: u64,
}

impl DeterministicInitializer {
    /// Creates an initializer seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the LCG and returns a sample in `[0, 1)`.
    ///
    /// The generator uses the constants `6364136223846793005` (multiplier) and
    /// `1442695040888963407` (increment), both wrapping. The result is the top
    /// 53 bits of the new state divided by `2^53`.
    pub fn next_unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Returns a zero-mean, unit-variance normal sample via Box-Muller.
    ///
    /// Two uniforms are consumed. The first is nudged away from zero with
    /// `(1.0 - u).max(f64::MIN_POSITIVE)` so the logarithm in the radius term
    /// can never be taken at zero.
    pub fn next_normal(&mut self) -> f64 {
        let first = self.next_unit();
        let second = self.next_unit();
        let radius = (-2.0 * (1.0 - first).max(f64::MIN_POSITIVE).ln()).sqrt();
        radius * (2.0 * std::f64::consts::PI * second).cos()
    }

    /// Fills `count` values from the zero-mean normal scaled by
    /// `standard_deviation`.
    pub fn fill_normal(&mut self, count: usize, standard_deviation: f64) -> Vec<f32> {
        (0..count)
            .map(|_| (self.next_normal() * standard_deviation) as f32)
            .collect()
    }

    /// Fills `count` copies of `value`.
    ///
    /// Constants do not consume generator state, so a constant-heavy model
    /// draws the same random weights as an otherwise identical one regardless
    /// of where its constants appear.
    pub fn fill_constant(&mut self, count: usize, value: f64) -> Vec<f32> {
        vec![value as f32; count]
    }
}

/// Every canonical tensor name [`initialize_model_tensors`] writes for
/// `config`, sorted.
///
/// The list is conditional on the configuration: `lm_head.weight` appears only
/// when the embeddings are untied, and the query/key/value biases appear only
/// when the family carries them (`has_query_key_value_bias`).
pub fn canonical_tensor_names(config: &ParallelModelConfig) -> Vec<String> {
    let mut names = Vec::new();
    names.push("model.embed_tokens.weight".to_string());
    if !config.tie_word_embeddings {
        names.push("lm_head.weight".to_string());
    }
    names.push("model.norm.weight".to_string());
    for index in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{index}");
        names.push(format!("{prefix}.input_layernorm.weight"));
        names.push(format!("{prefix}.post_attention_layernorm.weight"));
        if config.gemma_block_layout {
            names.push(format!("{prefix}.pre_feedforward_layernorm.weight"));
            names.push(format!("{prefix}.post_feedforward_layernorm.weight"));
        }
        for projection in ["q_proj", "k_proj", "v_proj"] {
            names.push(format!("{prefix}.self_attn.{projection}.weight"));
            if config.has_query_key_value_bias() {
                names.push(format!("{prefix}.self_attn.{projection}.bias"));
            }
        }
        if config.per_head_query_key_norm {
            names.push(format!("{prefix}.self_attn.q_norm.weight"));
            names.push(format!("{prefix}.self_attn.k_norm.weight"));
        }
        names.push(format!("{prefix}.self_attn.o_proj.weight"));
        if config.output_projection_bias() {
            names.push(format!("{prefix}.self_attn.o_proj.bias"));
        }
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            names.push(format!("{prefix}.mlp.{projection}.weight"));
        }
    }
    names.sort();
    names
}

/// Builds the complete F32 tensor map for a from-scratch model.
///
/// The map carries the canonical HuggingFace names the trainable forward
/// reads. `lm_head.weight` is written only when the embeddings are **untied**;
/// when `config.tie_word_embeddings` is set, the head is not emitted because
/// the model reads the embedding matrix directly (`TrainableLlama` clones
/// `model.embed_tokens.weight` into its head when tied). Emitting it anyway
/// would be dead weight in the map and a second, divergent copy at serving
/// time.
///
/// The same `(config, configuration, seed)` always produces byte-identical
/// tensors, because the generator advances in a fixed iteration order.
pub fn initialize_model_tensors(
    config: &ParallelModelConfig,
    configuration: &InitializationConfiguration,
    seed: u64,
    device: &Device,
) -> anyhow::Result<HashMap<String, Tensor>> {
    validate_configuration(config)?;

    let head_dimension = config.head_dimension();
    let query_size = head_dimension * config.num_attention_heads;
    let key_value_size = head_dimension * config.num_key_value_heads;
    let with_query_key_value_bias = config.has_query_key_value_bias();

    let mut initializer = DeterministicInitializer::new(seed);
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        normal_tensor(
            &mut initializer,
            config.vocab_size,
            config.hidden_size,
            configuration.embedding_std,
            device,
        )?,
    );
    if !config.tie_word_embeddings {
        tensors.insert(
            "lm_head.weight".to_string(),
            normal_tensor(
                &mut initializer,
                config.vocab_size,
                config.hidden_size,
                configuration.embedding_std,
                device,
            )?,
        );
    }
    tensors.insert(
        "model.norm.weight".to_string(),
        constant_tensor(
            &mut initializer,
            config.hidden_size,
            configuration.norm_weight,
            device,
        )?,
    );

    for index in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{index}");
        tensors.insert(
            format!("{prefix}.input_layernorm.weight"),
            constant_tensor(
                &mut initializer,
                config.hidden_size,
                configuration.norm_weight,
                device,
            )?,
        );
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            constant_tensor(
                &mut initializer,
                config.hidden_size,
                configuration.norm_weight,
                device,
            )?,
        );
        if config.gemma_block_layout {
            tensors.insert(
                format!("{prefix}.pre_feedforward_layernorm.weight"),
                constant_tensor(
                    &mut initializer,
                    config.hidden_size,
                    configuration.norm_weight,
                    device,
                )?,
            );
            tensors.insert(
                format!("{prefix}.post_feedforward_layernorm.weight"),
                constant_tensor(
                    &mut initializer,
                    config.hidden_size,
                    configuration.norm_weight,
                    device,
                )?,
            );
        }

        let attention_prefix = format!("{prefix}.self_attn");
        for (projection, output_size) in [
            ("q_proj", query_size),
            ("k_proj", key_value_size),
            ("v_proj", key_value_size),
        ] {
            tensors.insert(
                format!("{attention_prefix}.{projection}.weight"),
                normal_tensor(
                    &mut initializer,
                    output_size,
                    config.hidden_size,
                    configuration.initializer_range,
                    device,
                )?,
            );
            if with_query_key_value_bias {
                tensors.insert(
                    format!("{attention_prefix}.{projection}.bias"),
                    constant_tensor(
                        &mut initializer,
                        output_size,
                        configuration.bias_value,
                        device,
                    )?,
                );
            }
        }
        if config.per_head_query_key_norm {
            for norm in ["q_norm", "k_norm"] {
                tensors.insert(
                    format!("{attention_prefix}.{norm}.weight"),
                    constant_tensor(
                        &mut initializer,
                        head_dimension,
                        configuration.norm_weight,
                        device,
                    )?,
                );
            }
        }
        tensors.insert(
            format!("{attention_prefix}.o_proj.weight"),
            normal_tensor(
                &mut initializer,
                config.hidden_size,
                query_size,
                configuration.initializer_range,
                device,
            )?,
        );
        if config.output_projection_bias() {
            tensors.insert(
                format!("{attention_prefix}.o_proj.bias"),
                constant_tensor(
                    &mut initializer,
                    config.hidden_size,
                    configuration.bias_value,
                    device,
                )?,
            );
        }

        let mlp_prefix = format!("{prefix}.mlp");
        tensors.insert(
            format!("{mlp_prefix}.gate_proj.weight"),
            normal_tensor(
                &mut initializer,
                config.intermediate_size,
                config.hidden_size,
                configuration.initializer_range,
                device,
            )?,
        );
        tensors.insert(
            format!("{mlp_prefix}.up_proj.weight"),
            normal_tensor(
                &mut initializer,
                config.intermediate_size,
                config.hidden_size,
                configuration.initializer_range,
                device,
            )?,
        );
        tensors.insert(
            format!("{mlp_prefix}.down_proj.weight"),
            normal_tensor(
                &mut initializer,
                config.hidden_size,
                config.intermediate_size,
                configuration.initializer_range,
                device,
            )?,
        );
    }

    Ok(tensors)
}

/// Validates the shape relationships the initializer relies on.
fn validate_configuration(config: &ParallelModelConfig) -> anyhow::Result<()> {
    if config.num_attention_heads == 0 {
        return Err(
            TrainerError::initialization("num_attention_heads must be greater than zero").into(),
        );
    }
    if !config
        .hidden_size
        .is_multiple_of(config.num_attention_heads)
    {
        return Err(TrainerError::initialization(format!(
            "hidden_size ({}) must be divisible by num_attention_heads ({})",
            config.hidden_size, config.num_attention_heads
        ))
        .into());
    }
    if config.num_key_value_heads == 0 {
        return Err(
            TrainerError::initialization("num_key_value_heads must be greater than zero").into(),
        );
    }
    if !config
        .num_attention_heads
        .is_multiple_of(config.num_key_value_heads)
    {
        return Err(TrainerError::initialization(format!(
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            config.num_attention_heads, config.num_key_value_heads
        ))
        .into());
    }
    Ok(())
}

/// Draws a `(rows, columns)` normal tensor from the initializer.
fn normal_tensor(
    initializer: &mut DeterministicInitializer,
    rows: usize,
    columns: usize,
    standard_deviation: f64,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let values = initializer.fill_normal(rows * columns, standard_deviation);
    Ok(Tensor::from_vec(values, (rows, columns), device)?)
}

/// Builds a one-dimensional constant tensor from the initializer.
fn constant_tensor(
    initializer: &mut DeterministicInitializer,
    size: usize,
    value: f64,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let values = initializer.fill_constant(size, value);
    Ok(Tensor::from_vec(values, size, device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use typed_lm_common::architecture_traits::DenseArchitectureTraits;
    use typed_lm_common::checkpoint::ModelArchitecture;

    /// A tiny Llama-family-shaped configuration for the fast unit tests.
    ///
    /// The per-family flags are derived from [`DenseArchitectureTraits`] so a
    /// test can build a Qwen2 (biased) or Llama (biasless) model from the same
    /// skeleton, and `explicit_head_dimension` follows the family's
    /// requirement.
    fn tiny_config(architecture: ModelArchitecture) -> ParallelModelConfig {
        let traits = DenseArchitectureTraits::for_architecture(architecture);
        let hidden_size = 16;
        let num_attention_heads = 4;
        ParallelModelConfig {
            architecture,
            vocab_size: 32,
            hidden_size,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads,
            num_key_value_heads: 2,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            rope_scaling: None,
            attention_bias: traits.attention_bias,
            explicit_head_dimension: traits
                .explicit_head_dimension
                .then_some(hidden_size / num_attention_heads),
            sliding_window: None,
            max_window_layers: 0,
            logit_softcapping: None,
            attention_logit_softcapping: None,
            query_pre_attention_scalar: None,
            rms_norm_unit_offset: traits.rms_norm_unit_offset,
            embedding_scale: traits
                .scales_embeddings
                .then_some((hidden_size as f64).sqrt()),
            hidden_activation: ParallelModelConfig::default_hidden_activation(architecture),
            rope_local_base_frequency: None,
            gemma_block_layout: traits.gemma_block_layout,
            per_head_query_key_norm: traits.per_head_query_key_norm,
            sliding_window_pattern: 0,
        }
    }

    /// Flattens every canonical tensor in a stable, sorted order.
    fn flattened_tensor_values(
        tensors: &HashMap<String, Tensor>,
        config: &ParallelModelConfig,
    ) -> anyhow::Result<Vec<f32>> {
        let mut values = Vec::new();
        for name in canonical_tensor_names(config) {
            let tensor = tensors
                .get(&name)
                .ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
            values.extend(tensor.flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(values)
    }

    #[test]
    fn same_seed_produces_identical_sequences() {
        let mut first = DeterministicInitializer::new(99);
        let mut second = DeterministicInitializer::new(99);
        assert_eq!(first.fill_normal(64, 0.5), second.fill_normal(64, 0.5));
    }

    #[test]
    fn different_seeds_diverge() {
        let mut first = DeterministicInitializer::new(1);
        let mut second = DeterministicInitializer::new(2);
        assert_ne!(first.fill_normal(16, 1.0), second.fill_normal(16, 1.0));
    }

    #[test]
    fn fill_constant_is_constant() {
        let mut initializer = DeterministicInitializer::new(7);
        assert_eq!(initializer.fill_constant(8, 3.5), vec![3.5_f32; 8]);
    }

    #[test]
    fn normal_samples_are_within_a_reasonable_range() {
        let mut initializer = DeterministicInitializer::new(1234);
        let standard_deviation = 1.0;
        for value in initializer.fill_normal(5000, standard_deviation) {
            assert!(
                value.abs() < 10.0 * standard_deviation as f32,
                "outlier sample {value}"
            );
        }
    }

    #[test]
    fn initialize_model_tensors_carries_every_canonical_name() -> anyhow::Result<()> {
        let config = tiny_config(ModelArchitecture::Llama);
        let device = Device::Cpu;
        let tensors = initialize_model_tensors(
            &config,
            &InitializationConfiguration::default(),
            42,
            &device,
        )?;
        let mut keys: Vec<String> = tensors.keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, canonical_tensor_names(&config));
        Ok(())
    }

    #[test]
    fn initialize_model_tensors_is_deterministic_for_the_same_seed() -> anyhow::Result<()> {
        let config = tiny_config(ModelArchitecture::Llama);
        let device = Device::Cpu;
        let configuration = InitializationConfiguration::default();
        let first = initialize_model_tensors(&config, &configuration, 2024, &device)?;
        let second = initialize_model_tensors(&config, &configuration, 2024, &device)?;
        assert_eq!(
            flattened_tensor_values(&first, &config)?,
            flattened_tensor_values(&second, &config)?
        );
        Ok(())
    }

    #[test]
    fn initialization_depends_on_the_seed() -> anyhow::Result<()> {
        let config = tiny_config(ModelArchitecture::Llama);
        let device = Device::Cpu;
        let configuration = InitializationConfiguration::default();
        let first = initialize_model_tensors(&config, &configuration, 1, &device)?;
        let second = initialize_model_tensors(&config, &configuration, 2, &device)?;
        assert_ne!(
            flattened_tensor_values(&first, &config)?,
            flattened_tensor_values(&second, &config)?
        );
        Ok(())
    }

    #[test]
    fn qwen2_configuration_includes_qkv_biases() {
        let config = tiny_config(ModelArchitecture::Qwen2);
        let names = canonical_tensor_names(&config);
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.bias".to_string()));
        assert!(names.contains(&"model.layers.0.self_attn.k_proj.bias".to_string()));
        assert!(names.contains(&"model.layers.0.self_attn.v_proj.bias".to_string()));
    }

    #[test]
    fn llama_configuration_omits_qkv_biases() {
        let config = tiny_config(ModelArchitecture::Llama);
        let names = canonical_tensor_names(&config);
        assert!(!names.contains(&"model.layers.0.self_attn.q_proj.bias".to_string()));
        assert!(!names.contains(&"model.layers.0.self_attn.k_proj.bias".to_string()));
        assert!(!names.contains(&"model.layers.0.self_attn.v_proj.bias".to_string()));
    }

    #[test]
    fn gemma3_configuration_carries_the_new_block_and_norm_tensors() -> anyhow::Result<()> {
        let config = tiny_config(ModelArchitecture::Gemma3);
        assert!(config.gemma_block_layout);
        assert!(config.per_head_query_key_norm);
        let names = canonical_tensor_names(&config);
        for expected in [
            "model.layers.0.pre_feedforward_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
        ] {
            assert!(names.contains(&expected.to_string()), "missing {expected}");
        }
        // The deterministic initializer must write exactly the canonical names.
        let tensors = initialize_model_tensors(
            &config,
            &InitializationConfiguration::default(),
            11,
            &Device::Cpu,
        )?;
        let mut keys: Vec<String> = tensors.keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, names);
        Ok(())
    }

    #[test]
    fn tied_embeddings_omit_the_lm_head() {
        let mut config = tiny_config(ModelArchitecture::Llama);
        config.tie_word_embeddings = true;
        assert!(!canonical_tensor_names(&config).contains(&"lm_head.weight".to_string()));
    }

    #[test]
    fn untied_embeddings_include_the_lm_head() {
        let mut config = tiny_config(ModelArchitecture::Llama);
        config.tie_word_embeddings = false;
        assert!(canonical_tensor_names(&config).contains(&"lm_head.weight".to_string()));
    }

    #[test]
    fn an_invalid_head_configuration_is_an_error() {
        let mut config = tiny_config(ModelArchitecture::Llama);
        config.num_attention_heads = 3;
        let device = Device::Cpu;
        let result =
            initialize_model_tensors(&config, &InitializationConfiguration::default(), 0, &device);
        assert!(result.is_err());
    }
}
