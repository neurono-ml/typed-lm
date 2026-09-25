//! Saving and loading trained checkpoints.
//!
//! Two artifact flavours are written here:
//!
//! - a **LoRA adapter** — a `safetensors` file holding the adapter variables
//!   plus a companion `adapter_config.json` that records the rank, alpha and the
//!   source model so a later quantization step can merge the adapter into the
//!   base without guessing. Only the adapter tensors are written: the frozen
//!   base is never duplicated, keeping artifacts small and the merge explicit.
//! - a **full dense checkpoint** (`--method full`/`from-scratch`) — a complete
//!   `model.safetensors` with canonical Hugging Face names, a `config.json`
//!   reparseable by [`ParallelModelConfig::from_json_for_architecture`] and a
//!   copy of `tokenizer.json`, so `typed-lm-serve` can serve it directly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use serde_json::json;

use typed_lm_common::checkpoint::{ModelArchitecture, TOKENIZER_NAME};
use typed_lm_common::model_config::ParallelModelConfig;

use crate::error::TrainerError;

/// Name of the adapter weights file inside an output directory.
pub const ADAPTER_WEIGHTS_NAME: &str = "adapter.safetensors";

/// Name of the adapter metadata file inside an output directory.
pub const ADAPTER_CONFIG_NAME: &str = "adapter_config.json";

/// Name of the dense weights file inside a full checkpoint directory.
pub const FULL_WEIGHTS_NAME: &str = "model.safetensors";

/// Name of the configuration file inside a full checkpoint directory.
pub const FULL_CONFIG_NAME: &str = "config.json";

/// Persisted adapter metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterConfiguration {
    /// Adapter rank.
    pub rank: usize,
    /// Adapter scaling numerator.
    pub alpha: f64,
    /// Source model identifier (`--model-id`).
    pub model_identifier: String,
    /// Detected model architecture name (`llama`/`qwen2`).
    pub architecture: String,
}

impl AdapterConfiguration {
    pub fn new(
        rank: usize,
        alpha: f64,
        model_identifier: impl Into<String>,
        architecture: impl Into<String>,
    ) -> Self {
        Self {
            rank,
            alpha,
            model_identifier: model_identifier.into(),
            architecture: architecture.into(),
        }
    }
}

/// Saves the adapter variables and metadata into `output_directory`.
pub fn save_adapter(
    variable_map: &VarMap,
    configuration: &AdapterConfiguration,
    output_directory: &Path,
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(output_directory).map_err(TrainerError::Io)?;
    let tensors = adapter_tensors(variable_map)?;
    let weights_path = output_directory.join(ADAPTER_WEIGHTS_NAME);
    candle_core::safetensors::save(&tensors, &weights_path)?;

    let configuration_path = output_directory.join(ADAPTER_CONFIG_NAME);
    let json = serde_json::to_string_pretty(configuration).map_err(|error| {
        TrainerError::Training(format!("failed to serialize adapter config: {error}"))
    })?;
    std::fs::write(&configuration_path, json)?;
    Ok(weights_path)
}

/// Loads the adapter tensors from `output_directory` into a `VarMap`.
pub fn load_adapter(
    output_directory: &Path,
    device: &Device,
) -> anyhow::Result<(VarMap, AdapterConfiguration)> {
    let weights_path = output_directory.join(ADAPTER_WEIGHTS_NAME);
    let configuration_path = output_directory.join(ADAPTER_CONFIG_NAME);
    let configuration_json = std::fs::read_to_string(&configuration_path)?;
    let configuration: AdapterConfiguration =
        serde_json::from_str(&configuration_json).map_err(|error| {
            TrainerError::Training(format!("failed to parse adapter config: {error}"))
        })?;

    let tensors = candle_core::safetensors::load(&weights_path, device)?;
    let mut variable_map = VarMap::new();
    let builder =
        candle_nn::VarBuilder::from_varmap(&variable_map, candle_core::DType::F32, device);
    for (name, tensor) in &tensors {
        let shape = tensor.shape().clone();
        let variable = builder.get_with_hints(shape, name, candle_nn::Init::Const(0.0))?;
        variable_map.set_one(name, tensor)?;
        let _ = variable;
    }
    Ok((variable_map, configuration))
}

/// Collects the adapter variables into a name -> tensor map.
fn adapter_tensors(variable_map: &VarMap) -> anyhow::Result<HashMap<String, Tensor>> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let data = variable_map.data();
    let guard = data.lock().map_err(|error| {
        TrainerError::Training(format!("failed to lock the variable map: {error}"))
    })?;
    for (name, variable) in guard.iter() {
        tensors.insert(name.clone(), variable.as_tensor().clone());
    }
    Ok(tensors)
}

/// Saves a complete dense checkpoint into `output_directory`.
///
/// Writes `model.safetensors` with canonical Hugging Face tensor names, a
/// `config.json` reparseable by [`ParallelModelConfig::from_json_for_architecture`]
/// and a copy of the tokenizer, so the artifact loads in `typed-lm-serve`
/// without further processing. Returns the weights path.
pub fn save_full_checkpoint(
    variables: &HashMap<String, Tensor>,
    configuration: &ParallelModelConfig,
    tokenizer_source: &Path,
    output_directory: &Path,
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(output_directory).map_err(TrainerError::Io)?;

    let weights_path = output_directory.join(FULL_WEIGHTS_NAME);
    candle_core::safetensors::save(variables, &weights_path)?;

    let configuration_path = output_directory.join(FULL_CONFIG_NAME);
    let value = configuration_to_json(configuration)?;
    let json = serde_json::to_string_pretty(&value).map_err(|error| {
        TrainerError::Training(format!(
            "failed to serialize full checkpoint config: {error}"
        ))
    })?;
    std::fs::write(&configuration_path, json)?;

    copy_tokenizer(tokenizer_source, &output_directory.join(TOKENIZER_NAME))?;
    Ok(weights_path)
}

/// Loads a complete dense checkpoint written by [`save_full_checkpoint`].
///
/// Reads `model.safetensors` on `device` and parses `config.json` through
/// [`ParallelModelConfig::from_json_for_architecture`] using its `model_type`.
pub fn load_full_checkpoint(
    output_directory: &Path,
    device: &Device,
) -> anyhow::Result<(HashMap<String, Tensor>, ParallelModelConfig)> {
    let weights_path = output_directory.join(FULL_WEIGHTS_NAME);
    let configuration_path = output_directory.join(FULL_CONFIG_NAME);
    let configuration_json = std::fs::read_to_string(&configuration_path)?;
    let value: serde_json::Value = serde_json::from_str(&configuration_json).map_err(|error| {
        TrainerError::Training(format!("failed to parse full checkpoint config: {error}"))
    })?;

    let model_type = value
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            TrainerError::Training(
                "full checkpoint config.json has no 'model_type' field".to_string(),
            )
        })?;
    let architecture = ModelArchitecture::from_model_type(model_type).ok_or_else(|| {
        TrainerError::Training(format!(
            "unsupported architecture '{model_type}' in full checkpoint config.json; supported: {}",
            ModelArchitecture::supported_names()
        ))
    })?;
    let configuration = ParallelModelConfig::from_json_for_architecture(value, architecture)?;

    let tensors = candle_core::safetensors::load(&weights_path, device)?;
    Ok((tensors, configuration))
}

/// Converts a [`ParallelModelConfig`] into the Hugging Face JSON shape.
///
/// Emits every field `read_config`/`from_json_for_architecture` needs to
/// reconstruct the configuration for all seven dense families, including the
/// per-family fields (explicit head dimension, sliding window, logit
/// soft-capping, query pre-attention scalar and activation) that only some
/// `config.json` variants declare.
pub fn configuration_to_json(
    configuration: &ParallelModelConfig,
) -> anyhow::Result<serde_json::Value> {
    let architecture = configuration.architecture;
    let mut value = json!({
        "model_type": architecture.name(),
        "vocab_size": configuration.vocab_size,
        "hidden_size": configuration.hidden_size,
        "intermediate_size": configuration.intermediate_size,
        "num_hidden_layers": configuration.num_hidden_layers,
        "num_attention_heads": configuration.num_attention_heads,
        "num_key_value_heads": configuration.num_key_value_heads,
        "max_position_embeddings": configuration.max_position_embeddings,
        "rms_norm_eps": configuration.rms_norm_eps,
        "rope_theta": configuration.rope_theta,
        "tie_word_embeddings": configuration.tie_word_embeddings,
        "attention_bias": configuration.attention_bias,
        "max_window_layers": configuration.max_window_layers,
    });
    let object = value.as_object_mut().ok_or_else(|| {
        TrainerError::Training("failed to build the full checkpoint config object".to_string())
    })?;

    if let Some(head_dimension) = configuration.explicit_head_dimension {
        object.insert("head_dim".to_string(), json!(head_dimension));
    }
    if let Some(sliding_window) = configuration.sliding_window {
        object.insert("sliding_window".to_string(), json!(sliding_window));
    }

    match architecture {
        ModelArchitecture::Qwen2 => {
            object.insert(
                "sliding_window".to_string(),
                json!(configuration
                    .sliding_window
                    .unwrap_or(configuration.max_position_embeddings)),
            );
            object.insert("use_sliding_window".to_string(), json!(false));
        }
        ModelArchitecture::Qwen3 => {
            object.insert(
                "use_sliding_window".to_string(),
                json!(configuration.sliding_window.is_some()),
            );
        }
        ModelArchitecture::Gemma3 => {
            object.insert(
                "sliding_window".to_string(),
                json!(configuration
                    .sliding_window
                    .unwrap_or(configuration.max_position_embeddings)),
            );
            object.insert(
                "sliding_window_pattern".to_string(),
                json!(configuration.sliding_window_pattern.max(1)),
            );
        }
        ModelArchitecture::Llama
        | ModelArchitecture::Mistral
        | ModelArchitecture::Gemma
        | ModelArchitecture::Gemma2 => {}
    }

    if let Some(logit_softcapping) = configuration.logit_softcapping {
        object.insert(
            "final_logit_softcapping".to_string(),
            json!(logit_softcapping),
        );
    }
    if let Some(attention_logit_softcapping) = configuration.attention_logit_softcapping {
        object.insert(
            "attn_logit_softcapping".to_string(),
            json!(attention_logit_softcapping),
        );
    }
    if let Some(query_pre_attention_scalar) = configuration.query_pre_attention_scalar {
        object.insert(
            "query_pre_attn_scalar".to_string(),
            json!(query_pre_attention_scalar),
        );
    }
    if let Some(rope_local_base_frequency) = configuration.rope_local_base_frequency {
        object.insert(
            "rope_local_base_freq".to_string(),
            json!(rope_local_base_frequency),
        );
    }

    let activation = hidden_activation_name(architecture);
    match architecture {
        // Gemma v1 accepts `hidden_act` and `hidden_activation`, but candle
        // rejects a config that sets both, so only the canonical `hidden_act`
        // is written. Gemma2/Gemma3 only declare `hidden_activation`.
        ModelArchitecture::Gemma => {
            object.insert("hidden_act".to_string(), json!(activation));
        }
        ModelArchitecture::Gemma2 | ModelArchitecture::Gemma3 => {
            object.insert("hidden_activation".to_string(), json!(activation));
        }
        ModelArchitecture::Llama
        | ModelArchitecture::Qwen2
        | ModelArchitecture::Qwen3
        | ModelArchitecture::Mistral => {
            object.insert("hidden_act".to_string(), json!(activation));
        }
    }

    Ok(value)
}

/// Canonical `hidden_act` name for each architecture family.
fn hidden_activation_name(architecture: ModelArchitecture) -> &'static str {
    match architecture {
        ModelArchitecture::Gemma | ModelArchitecture::Gemma2 | ModelArchitecture::Gemma3 => {
            "gelu_pytorch_tanh"
        }
        ModelArchitecture::Llama
        | ModelArchitecture::Qwen2
        | ModelArchitecture::Qwen3
        | ModelArchitecture::Mistral => "silu",
    }
}

/// Copies the tokenizer into `destination`, accepting a file or a directory.
///
/// A directory source is resolved to its `tokenizer.json` entry.
fn copy_tokenizer(tokenizer_source: &Path, destination: &Path) -> anyhow::Result<()> {
    let resolved = if tokenizer_source.is_dir() {
        tokenizer_source.join(TOKENIZER_NAME)
    } else {
        tokenizer_source.to_path_buf()
    };
    if !resolved.is_file() {
        return Err(TrainerError::Training(format!(
            "tokenizer source '{}' does not point at a tokenizer.json file",
            tokenizer_source.display()
        ))
        .into());
    }
    std::fs::copy(&resolved, destination)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn sample_variable_map(device: &Device) -> anyhow::Result<VarMap> {
        let variable_map = VarMap::new();
        let builder = candle_nn::VarBuilder::from_varmap(&variable_map, DType::F32, device);
        let _down = builder.get_with_hints(
            (2, 4),
            "model.layers.0.lora_a",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.1,
            },
        )?;
        let _up =
            builder.get_with_hints((3, 2), "model.layers.0.lora_b", candle_nn::Init::Const(0.5))?;
        Ok(variable_map)
    }

    fn sample_full_weights(device: &Device) -> anyhow::Result<HashMap<String, Tensor>> {
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::new(&[[1f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device)?,
        );
        weights.insert(
            "lm_head.weight".to_string(),
            Tensor::new(&[[0.5f32, -0.5, 1.5, -1.5]], device)?,
        );
        weights.insert(
            "model.norm.weight".to_string(),
            Tensor::new(&[1f32, 1.0, 1.0, 1.0], device)?,
        );
        Ok(weights)
    }

    fn write_tokenizer_file(path: &Path) -> anyhow::Result<()> {
        std::fs::write(path, "{}")?;
        Ok(())
    }

    fn minimal_configuration_json(architecture: ModelArchitecture) -> serde_json::Value {
        let mut configuration = json!({
            "model_type": architecture.name(),
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": true
        });
        if let Some(object) = configuration.as_object_mut() {
            match architecture {
                ModelArchitecture::Qwen2 => {
                    object.insert("sliding_window".to_string(), json!(64));
                    object.insert("max_window_layers".to_string(), json!(1));
                    object.insert("use_sliding_window".to_string(), json!(false));
                    object.insert("hidden_act".to_string(), json!("silu"));
                }
                ModelArchitecture::Qwen3 => {
                    object.insert("head_dim".to_string(), json!(4));
                    object.insert("attention_bias".to_string(), json!(false));
                    object.insert("sliding_window".to_string(), json!(64));
                    object.insert("max_window_layers".to_string(), json!(2));
                    object.insert("use_sliding_window".to_string(), json!(true));
                    object.insert("hidden_act".to_string(), json!("silu"));
                }
                ModelArchitecture::Mistral => {
                    object.insert("head_dim".to_string(), json!(4));
                    object.insert("sliding_window".to_string(), json!(64));
                    object.insert("hidden_act".to_string(), json!("silu"));
                }
                ModelArchitecture::Gemma => {
                    object.insert("attention_bias".to_string(), json!(false));
                    object.insert("head_dim".to_string(), json!(4));
                    object.insert("hidden_act".to_string(), json!("gelu_pytorch_tanh"));
                    object.insert("hidden_activation".to_string(), json!("gelu_pytorch_tanh"));
                }
                ModelArchitecture::Gemma2 => {
                    object.insert("attention_bias".to_string(), json!(false));
                    object.insert("head_dim".to_string(), json!(4));
                    object.insert("hidden_activation".to_string(), json!("gelu_pytorch_tanh"));
                    object.insert("final_logit_softcapping".to_string(), json!(30.0));
                    object.insert("attn_logit_softcapping".to_string(), json!(50.0));
                    object.insert("query_pre_attn_scalar".to_string(), json!(4));
                    object.insert("sliding_window".to_string(), json!(64));
                }
                ModelArchitecture::Gemma3 => {
                    object.insert("attention_bias".to_string(), json!(false));
                    object.insert("head_dim".to_string(), json!(4));
                    object.insert("hidden_activation".to_string(), json!("gelu_pytorch_tanh"));
                    object.insert("rope_local_base_freq".to_string(), json!(10000.0));
                    object.insert("final_logit_softcapping".to_string(), json!(30.0));
                    object.insert("attn_logit_softcapping".to_string(), json!(50.0));
                    object.insert("query_pre_attn_scalar".to_string(), json!(4));
                    object.insert("sliding_window".to_string(), json!(64));
                    object.insert("sliding_window_pattern".to_string(), json!(6));
                }
                ModelArchitecture::Llama => {}
            }
        }
        configuration
    }

    #[test]
    fn adapter_round_trips_through_safetensors() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let variable_map = sample_variable_map(&device)?;
        let configuration = AdapterConfiguration::new(2, 4.0, "hf-internal-testing/tiny", "llama");
        let directory = tempfile::tempdir()?;
        let weights_path = save_adapter(&variable_map, &configuration, directory.path())?;
        assert!(weights_path.exists());
        assert!(directory.path().join(ADAPTER_CONFIG_NAME).exists());

        let (_, loaded_configuration) = load_adapter(directory.path(), &device)?;
        assert_eq!(loaded_configuration, configuration);
        Ok(())
    }

    #[test]
    fn missing_adapter_files_are_an_error() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let directory = tempfile::tempdir()?;
        let result = load_adapter(directory.path(), &device);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn saved_metadata_preserves_rank_and_alpha() -> anyhow::Result<()> {
        let configuration = AdapterConfiguration::new(8, 16.0, "model", "qwen2");
        let json = serde_json::to_string(&configuration)?;
        let parsed: AdapterConfiguration = serde_json::from_str(&json)?;
        assert_eq!(parsed.rank, 8);
        assert!((parsed.alpha - 16.0).abs() < 1e-9);
        assert_eq!(parsed.architecture, "qwen2");
        Ok(())
    }

    #[test]
    fn full_checkpoint_round_trips_config_and_weights() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let architectures = [
            ModelArchitecture::Llama,
            ModelArchitecture::Qwen2,
            ModelArchitecture::Gemma2,
        ];
        for architecture in architectures {
            let configuration = ParallelModelConfig::from_json_for_architecture(
                minimal_configuration_json(architecture),
                architecture,
            )?;
            let weights = sample_full_weights(&device)?;
            let directory = tempfile::tempdir()?;
            let tokenizer_source = directory.path().join("source-tokenizer.json");
            write_tokenizer_file(&tokenizer_source)?;
            let output_directory = directory.path().join("checkpoint");

            let weights_path = save_full_checkpoint(
                &weights,
                &configuration,
                &tokenizer_source,
                &output_directory,
            )?;
            assert!(weights_path.exists());
            assert!(output_directory.join(FULL_CONFIG_NAME).exists());
            assert!(output_directory.join(TOKENIZER_NAME).exists());

            let (loaded_weights, loaded_configuration) =
                load_full_checkpoint(&output_directory, &device)?;
            assert_eq!(loaded_configuration.architecture, architecture);
            assert_eq!(loaded_configuration.vocab_size, configuration.vocab_size);
            assert_eq!(
                loaded_configuration.head_dimension(),
                configuration.head_dimension()
            );

            let loaded_embedding = loaded_weights
                .get("model.embed_tokens.weight")
                .ok_or_else(|| anyhow::anyhow!("loaded embedding weight is missing"))?
                .to_vec2::<f32>()?;
            let source_embedding = weights
                .get("model.embed_tokens.weight")
                .ok_or_else(|| anyhow::anyhow!("source embedding weight is missing"))?
                .to_vec2::<f32>()?;
            assert_eq!(loaded_embedding, source_embedding);
        }
        Ok(())
    }

    #[test]
    fn the_config_json_round_trips_every_family() -> anyhow::Result<()> {
        for architecture in ModelArchitecture::SUPPORTED {
            let configuration = ParallelModelConfig::from_json_for_architecture(
                minimal_configuration_json(architecture),
                architecture,
            )?;
            let value = configuration_to_json(&configuration)?;
            let reparsed = ParallelModelConfig::from_json_for_architecture(value, architecture)?;
            assert_eq!(reparsed.architecture, architecture);
            assert_eq!(reparsed.head_dimension(), configuration.head_dimension());
            assert_eq!(reparsed.attention_bias, configuration.attention_bias);
        }
        Ok(())
    }

    #[test]
    fn a_tokenizer_directory_is_accepted() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let configuration = ParallelModelConfig::from_json_for_architecture(
            minimal_configuration_json(ModelArchitecture::Llama),
            ModelArchitecture::Llama,
        )?;
        let weights = sample_full_weights(&device)?;
        let directory = tempfile::tempdir()?;
        let tokenizer_source = directory.path().join("tokenizer-source");
        std::fs::create_dir_all(&tokenizer_source)?;
        write_tokenizer_file(&tokenizer_source.join(TOKENIZER_NAME))?;
        let output_directory = directory.path().join("checkpoint");

        save_full_checkpoint(
            &weights,
            &configuration,
            &tokenizer_source,
            &output_directory,
        )?;
        assert!(output_directory.join(TOKENIZER_NAME).exists());
        Ok(())
    }

    #[test]
    fn a_missing_tokenizer_is_an_error() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let configuration = ParallelModelConfig::from_json_for_architecture(
            minimal_configuration_json(ModelArchitecture::Llama),
            ModelArchitecture::Llama,
        )?;
        let weights = sample_full_weights(&device)?;
        let directory = tempfile::tempdir()?;
        let missing_source = directory.path().join("does-not-exist.json");
        let output_directory = directory.path().join("checkpoint");

        let result =
            save_full_checkpoint(&weights, &configuration, &missing_source, &output_directory);
        assert!(result.is_err());
        Ok(())
    }
}
