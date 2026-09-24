//! Saving and loading a trained LoRA adapter.
//!
//! A checkpoint is a `safetensors` file holding the adapter variables plus a
//! companion `adapter_config.json` that records the rank, alpha and the source
//! model so a later quantization step can merge the adapter into the base
//! without guessing. Only the adapter tensors are written: the frozen base is
//! never duplicated, keeping artifacts small and the merge explicit.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use crate::error::TrainerError;

/// Name of the adapter weights file inside an output directory.
pub const ADAPTER_WEIGHTS_NAME: &str = "adapter.safetensors";

/// Name of the adapter metadata file inside an output directory.
pub const ADAPTER_CONFIG_NAME: &str = "adapter_config.json";

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
}
