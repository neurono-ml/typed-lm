//! Merging LoRA adapters into the base and exporting quantized artifacts.
//!
//! Export has two jobs:
//!
//! 1. **Merge** — fold each LoRA adapter into its frozen base weight and drop
//!    the adapter, producing a plain dense `model.safetensors`.
//! 2. **Quantize** — post-training quantization (PTQ) of the merged dense
//!    weights to FP8 or FP4, writing both `model.safetensors` and a
//!    `quantization_config.json` that records the scheme and block size.
//!
//! FP8 stores native `F8_E4M3` tensors with a per-row `*_scale`. Candle 0.11
//! can store but not convert `F4`, so FP4 is written as the packed E2M1 nibbles
//! in a `U8` tensor plus a `U8` exponent tensor named with the `_scale` suffix
//! (see [`typed_lm_common::quantization::scale_tensor_name`]); the loader
//! dequantizes both back to dense F32.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use typed_lm_common::quantization::{
    dequantize, quantize, scale_tensor_name, QuantizationConfig, QuantizationScheme,
};

use crate::error::TrainerError;

/// Name of the dense/quantized weights file written by export.
pub const MODEL_WEIGHTS_NAME: &str = "model.safetensors";

/// Name of the quantization metadata file written by export.
pub const QUANTIZATION_CONFIG_NAME: &str = "quantization_config.json";

/// One named weight to export.
#[derive(Debug, Clone)]
pub struct ExportableWeight {
    pub name: String,
    pub tensor: Tensor,
}

/// Merges an adapter delta into a base weight map.
///
/// `adapter` holds pairs keyed by the base weight name, e.g.
/// `model.layers.0.self_attn.q_proj.weight`; the adapter tensor is a dense
/// delta of the same shape. Merging removes the delta entries from the output,
/// so the exported checkpoint is a plain dense model.
pub fn merge_adapter(
    base: HashMap<String, Tensor>,
    adapter: HashMap<String, Tensor>,
) -> anyhow::Result<HashMap<String, Tensor>> {
    let mut merged: HashMap<String, Tensor> = HashMap::with_capacity(base.len());
    for (name, weight) in base {
        match adapter.get(&name) {
            Some(delta) => {
                if delta.dims() != weight.dims() {
                    return Err(TrainerError::Quantization(format!(
                        "adapter delta for '{name}' has shape {:?}, expected {:?}",
                        delta.dims(),
                        weight.dims()
                    ))
                    .into());
                }
                merged.insert(name, (weight + delta)?.to_dtype(DType::F32)?);
            }
            None => {
                merged.insert(name, weight.to_dtype(DType::F32)?);
            }
        };
    }
    Ok(merged)
}

/// Quantizes and writes an artifact into `output_directory`.
///
/// `weights` is the merged dense F32 weight map. For [`QuantizationScheme::None`]
/// the weights are written densly; for FP8/FP4 they are quantized with a
/// `quantization_config.json` companion.
pub fn export_quantized(
    weights: HashMap<String, Tensor>,
    scheme: QuantizationScheme,
    output_directory: &Path,
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(output_directory).map_err(TrainerError::Io)?;
    let configuration = QuantizationConfig::new(scheme);
    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(weights.len() * 2);

    for (name, weight) in weights {
        let weight = weight.to_dtype(DType::F32)?;
        match scheme {
            QuantizationScheme::None => {
                tensors.insert(name, weight);
            }
            QuantizationScheme::Fp8 => {
                let (quantized, scales) = quantize(&configuration, &weight)?;
                tensors.insert(name.clone(), quantized);
                tensors.insert(scale_tensor_name(&name), scales);
            }
            QuantizationScheme::Fp4 => {
                let (packed, exponents) = quantize(&configuration, &weight)?;
                // Candle cannot serialize F4/F8E8M0, so the packed nibbles and
                // the exponents are stored as U8 with the `_scale` suffix.
                tensors.insert(name.clone(), packed.to_dtype(DType::U8)?);
                tensors.insert(scale_tensor_name(&name), exponents.to_dtype(DType::U8)?);
            }
        }
    }

    let weights_path = output_directory.join(MODEL_WEIGHTS_NAME);
    candle_core::safetensors::save(&tensors, &weights_path)?;
    std::fs::write(
        output_directory.join(QUANTIZATION_CONFIG_NAME),
        configuration.to_json()?,
    )
    .map_err(TrainerError::Io)?;
    Ok(weights_path)
}

/// Reads a quantized artifact back and dequantizes it to dense F32.
///
/// Used by round-trip tests and by callers that need the dense weights after a
/// PTQ export. The scheme is taken from the companion `quantization_config.json`.
pub fn load_quantized(
    output_directory: &Path,
    device: &Device,
) -> anyhow::Result<HashMap<String, Tensor>> {
    let configuration_json =
        std::fs::read_to_string(output_directory.join(QUANTIZATION_CONFIG_NAME))?;
    let configuration = QuantizationConfig::from_json(&configuration_json)?;
    let tensors =
        candle_core::safetensors::load(output_directory.join(MODEL_WEIGHTS_NAME), device)?;
    if configuration.scheme == QuantizationScheme::None {
        return Ok(tensors);
    }
    let mut dense: HashMap<String, Tensor> = HashMap::with_capacity(tensors.len());
    for (name, tensor) in &tensors {
        if name.ends_with("_scale") {
            continue;
        }
        let scale = tensors.get(&scale_tensor_name(name)).ok_or_else(|| {
            TrainerError::Quantization(format!(
                "quantized weight '{name}' is missing its scale tensor"
            ))
        })?;
        let element_count = weight_element_count(tensor)?;
        let restored = dequantize(&configuration, tensor, scale, element_count)?;
        dense.insert(name.clone(), restored.to_dtype(DType::F32)?);
    }
    Ok(dense)
}

/// Element count of a stored weight (halved for packed FP4 bytes).
fn weight_element_count(tensor: &Tensor) -> anyhow::Result<usize> {
    // `dequantize` for FP4 expects the *original* element count, i.e. two
    // nibbles per stored byte.
    Ok(tensor.elem_count() * 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_weights(device: &Device) -> anyhow::Result<HashMap<String, Tensor>> {
        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Tensor::new(&[[1.0_f32, -2.0], [0.5, 3.0]], device)?,
        );
        weights.insert(
            "model.norm.weight".to_string(),
            Tensor::new(&[1.0_f32, 1.0], device)?,
        );
        Ok(weights)
    }

    #[test]
    fn merging_adds_the_adapter_and_drops_it() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let base = dense_weights(&device)?;
        let mut adapter = HashMap::new();
        adapter.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Tensor::new(&[[0.5_f32, 0.5], [0.5, 0.5]], &device)?,
        );
        let merged = merge_adapter(base, adapter)?;
        let value = merged
            .get("model.layers.0.self_attn.q_proj.weight")
            .ok_or_else(|| anyhow::anyhow!("missing merged weight"))?
            .to_vec2::<f32>()?;
        assert_eq!(value, vec![vec![1.5, -1.5], vec![1.0, 3.5]]);
        Ok(())
    }

    #[test]
    fn a_shape_mismatched_adapter_is_an_error() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let base = dense_weights(&device)?;
        let mut adapter = HashMap::new();
        adapter.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Tensor::new(&[0.5_f32, 0.5], &device)?,
        );
        assert!(merge_adapter(base, adapter).is_err());
        Ok(())
    }

    #[test]
    fn none_scheme_writes_dense_weights() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let directory = tempfile::tempdir()?;
        let weights = dense_weights(&device)?;
        let path = export_quantized(weights, QuantizationScheme::None, directory.path())?;
        assert!(path.exists());
        assert!(directory.path().join(QUANTIZATION_CONFIG_NAME).exists());
        let restored = load_quantized(directory.path(), &device)?;
        assert_eq!(restored.len(), 2);
        Ok(())
    }

    #[test]
    fn fp8_export_round_trips_through_the_loader() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let directory = tempfile::tempdir()?;
        let weights = dense_weights(&device)?;
        export_quantized(weights, QuantizationScheme::Fp8, directory.path())?;
        let restored = load_quantized(directory.path(), &device)?;
        assert_eq!(restored.len(), 2);
        for tensor in restored.values() {
            assert_eq!(tensor.dtype(), DType::F32);
        }
        Ok(())
    }

    #[test]
    fn fp4_export_round_trips_through_the_loader() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let directory = tempfile::tempdir()?;
        let weights = dense_weights(&device)?;
        export_quantized(weights, QuantizationScheme::Fp4, directory.path())?;
        let restored = load_quantized(directory.path(), &device)?;
        assert_eq!(restored.len(), 2);
        // Packed FP4 is dequantized to a flat F32 tensor covering every original
        // element (the 2x2 query weight holds four values).
        let query = restored
            .get("model.layers.0.self_attn.q_proj.weight")
            .ok_or_else(|| anyhow::anyhow!("missing query weight"))?;
        assert_eq!(query.elem_count(), 4);
        assert_eq!(query.dtype(), DType::F32);
        Ok(())
    }

    #[test]
    fn a_missing_config_is_an_error() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let directory = tempfile::tempdir()?;
        assert!(load_quantized(directory.path(), &device).is_err());
        Ok(())
    }

    #[test]
    fn fp4_config_records_the_block_size() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let directory = tempfile::tempdir()?;
        export_quantized(
            dense_weights(&device)?,
            QuantizationScheme::Fp4,
            directory.path(),
        )?;
        let json = std::fs::read_to_string(directory.path().join(QUANTIZATION_CONFIG_NAME))?;
        let configuration = QuantizationConfig::from_json(&json)?;
        assert_eq!(configuration.scheme, QuantizationScheme::Fp4);
        assert_eq!(
            configuration.block_size,
            typed_lm_common::quantization::MXFP4_BLOCK_SIZE
        );
        Ok(())
    }
}
