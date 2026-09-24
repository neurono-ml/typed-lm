//! Building a frozen base `VarMap` from a resolved checkpoint.
//!
//! The trainer only trains a LoRA adapter, so every base weight stays frozen:
//! this module dequantizes low-precision (FP8/FP4) checkpoints to dense F32,
//! reads safetensors/GGUF/PyTorch/NumPy layouts into an in-memory tensor map,
//! and records which names came from the checkpoint. A [`FrozenBase`] exposes
//! that map plus the frozen names, so the trainable model can wrap each base
//! projection while the loader guarantees the base is never a trainable
//! variable.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use typed_lm_common::checkpoint::{WeightKind, WeightLayout};
use typed_lm_common::checkpoint_resolver::LoadableCheckpoint;
use typed_lm_common::model_config::ParallelModelConfig;
use typed_lm_common::quantization::{dequantize_checkpoint_tensors, QuantizationScheme};

use crate::error::TrainerError;

/// A base checkpoint loaded as dense F32 tensors, frozen for LoRA training.
#[derive(Debug, Clone)]
pub struct FrozenBase {
    /// Every base tensor, dense and in F32.
    tensors: HashMap<String, Tensor>,
    /// The weight kind detected for the source checkpoint.
    weight_kind: WeightKind,
}

impl FrozenBase {
    /// Wraps an in-memory tensor map as a frozen base (no checkpoint read).
    pub fn from_tensors(
        tensors: HashMap<String, Tensor>,
        weight_kind: WeightKind,
    ) -> anyhow::Result<Self> {
        if tensors.is_empty() {
            return Err(TrainerError::Model("the base tensor map is empty".to_string()).into());
        }
        let mut dense: HashMap<String, Tensor> = HashMap::with_capacity(tensors.len());
        for (name, tensor) in tensors {
            dense.insert(name, tensor.to_dtype(DType::F32)?);
        }
        Ok(Self {
            tensors: dense,
            weight_kind,
        })
    }

    /// Loads a resolved checkpoint into dense F32 base tensors.
    ///
    /// `scheme` is applied only for low-precision float checkpoints; dense and
    /// GGML-quantized checkpoints keep their existing loading path.
    pub fn from_checkpoint(
        checkpoint: &LoadableCheckpoint,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let scheme = scheme_for_weight_kind(checkpoint.weight_kind);
        let tensors = load_checkpoint_tensors(&checkpoint.resolved.layout, scheme, device)?;
        Self::from_tensors(tensors, checkpoint.weight_kind)
    }

    /// The frozen base tensors.
    pub fn tensors(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }

    /// Looks up one base tensor by name.
    pub fn tensor(&self, name: &str) -> anyhow::Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| TrainerError::Model(format!("missing base tensor '{name}'")).into())
    }

    /// Number of base tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the base carries no tensors (never true for a loaded base).
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// The detected weight kind of the source checkpoint.
    pub fn weight_kind(&self) -> WeightKind {
        self.weight_kind
    }

    /// Names of the frozen base tensors, sorted for determinism.
    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }

    /// Dense F32 base weight of a projection (no bias), by prefix.
    ///
    /// Used by the trainable model builder to obtain each frozen projection.
    pub fn projection_weight(&self, prefix: &str) -> anyhow::Result<Tensor> {
        self.tensor(&format!("{prefix}.weight")).cloned()
    }

    /// Dense F32 base bias of a projection, when the architecture has one.
    pub fn projection_bias(&self, prefix: &str) -> anyhow::Result<Option<Tensor>> {
        Ok(self.tensors.get(&format!("{prefix}.bias")).cloned())
    }

    /// The model configuration inferred from a config.json body.
    pub fn configuration(
        &self,
        value: serde_json::Value,
        architecture: typed_lm_common::checkpoint::ModelArchitecture,
    ) -> anyhow::Result<ParallelModelConfig> {
        ParallelModelConfig::from_json_for_architecture(value, architecture)
    }
}

/// Maps a detected weight kind to the dequantization scheme used on load.
pub fn scheme_for_weight_kind(weight_kind: WeightKind) -> QuantizationScheme {
    match weight_kind {
        WeightKind::Float8 => QuantizationScheme::Fp8,
        WeightKind::Float4 => QuantizationScheme::Fp4,
        WeightKind::Dense | WeightKind::Quantized | WeightKind::UnsupportedFloat8 => {
            QuantizationScheme::None
        }
    }
}

/// Reads every weight of a layout into an in-memory dense F32 tensor map.
fn load_checkpoint_tensors(
    layout: &WeightLayout,
    scheme: QuantizationScheme,
    device: &Device,
) -> anyhow::Result<HashMap<String, Tensor>> {
    match layout {
        WeightLayout::Safetensors { files } => {
            let mut tensors: HashMap<String, Tensor> = HashMap::new();
            for file in files {
                for (name, tensor) in candle_core::safetensors::load(file, device)? {
                    tensors.insert(name, tensor);
                }
            }
            if scheme.is_quantized() {
                dequantize_checkpoint_tensors(tensors, scheme)
            } else {
                Ok(tensors)
            }
        }
        WeightLayout::Gguf { file } => load_gguf_tensors(file, device),
        WeightLayout::Pytorch { file } => Ok(candle_core::safetensors::load(file, device)?),
        WeightLayout::Numpy { file } => load_numpy_tensors(file, device),
    }
}

/// Dequantizes a GGUF file's tensors to dense F32 by name.
fn load_gguf_tensors(
    file: &std::path::Path,
    device: &Device,
) -> anyhow::Result<HashMap<String, Tensor>> {
    use candle_core::quantized::gguf_file;

    let mut reader = std::fs::File::open(file)?;
    let content = gguf_file::Content::read(&mut reader)?;
    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(content.tensor_infos.len());
    for name in content.tensor_infos.keys() {
        let quantized = content.tensor(&mut reader, name, &Device::Cpu)?;
        let dense = quantized.dequantize(device)?.to_dtype(DType::F32)?;
        tensors.insert(name.clone(), dense);
    }
    Ok(tensors)
}

/// Reads a NumPy `.npz` archive into an in-memory tensor map.
fn load_numpy_tensors(
    file: &std::path::Path,
    device: &Device,
) -> anyhow::Result<HashMap<String, Tensor>> {
    let variable_builder = candle_nn::VarBuilder::from_npz(file, DType::F32, device)?;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for name in npz_tensor_names(file)? {
        tensors.insert(
            name.clone(),
            variable_builder.get_unchecked(&name)?.detach(),
        );
    }
    Ok(tensors)
}

/// Lists the array names inside a `.npz` archive using its zip directory.
///
/// A `.npz` file is a zip archive whose members are `<name>.npy`; the names are
/// read from the central directory records (signature `0x02014b50`) without
/// decompressing any data.
fn npz_tensor_names(file: &std::path::Path) -> anyhow::Result<Vec<String>> {
    const CENTRAL_DIRECTORY_SIGNATURE: u32 = 0x0201_4b50;
    const RECORD_HEADER_BYTES: usize = 46;
    const NAME_LENGTH_OFFSET: usize = 28;

    let bytes = std::fs::read(file)?;
    if bytes.len() < RECORD_HEADER_BYTES {
        return Err(TrainerError::Model(format!(
            "'{}' is too small to be an npz archive",
            file.display()
        ))
        .into());
    }
    let mut names: Vec<String> = Vec::new();
    let mut index = 0_usize;
    while index + RECORD_HEADER_BYTES <= bytes.len() {
        let signature = u32::from_le_bytes([
            bytes[index],
            bytes[index + 1],
            bytes[index + 2],
            bytes[index + 3],
        ]);
        if signature != CENTRAL_DIRECTORY_SIGNATURE {
            index += 1;
            continue;
        }
        let name_length = u16::from_le_bytes([
            bytes[index + NAME_LENGTH_OFFSET],
            bytes[index + NAME_LENGTH_OFFSET + 1],
        ]) as usize;
        let name_start = index + RECORD_HEADER_BYTES;
        if name_start + name_length > bytes.len() {
            break;
        }
        if let Ok(raw) = std::str::from_utf8(&bytes[name_start..name_start + name_length]) {
            let trimmed = raw
                .strip_suffix(".npy")
                .unwrap_or(raw)
                .trim_start_matches(['/', '.']);
            if !trimmed.is_empty() {
                names.push(trimmed.to_string());
            }
        }
        index = name_start + name_length;
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_tensors() -> HashMap<String, Tensor> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        if let Ok(weight) = Tensor::new(&[[1.0_f32, 2.0], [3.0, 4.0]], &device) {
            tensors.insert("model.layers.0.weight".to_string(), weight);
        }
        tensors
    }

    #[test]
    fn from_tensors_forces_dense_f32() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "layer.weight".to_string(),
            Tensor::new(&[1.0_f32, 2.0], &device)?.to_dtype(DType::BF16)?,
        );
        let base = FrozenBase::from_tensors(tensors, WeightKind::Dense)?;
        assert_eq!(base.tensor("layer.weight")?.dtype(), DType::F32);
        assert_eq!(base.len(), 1);
        assert!(!base.is_empty());
        Ok(())
    }

    #[test]
    fn an_empty_base_is_rejected() {
        let result = FrozenBase::from_tensors(HashMap::new(), WeightKind::Dense);
        assert!(result.is_err());
    }

    #[test]
    fn a_missing_tensor_is_an_error() -> anyhow::Result<()> {
        let base = FrozenBase::from_tensors(dummy_tensors(), WeightKind::Dense)?;
        assert!(base.tensor("does.not.exist").is_err());
        Ok(())
    }

    #[test]
    fn projection_lookup_reads_weight_and_optional_bias() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "self_attn.q_proj.weight".to_string(),
            Tensor::new(&[[1.0_f32]], &device)?,
        );
        tensors.insert(
            "self_attn.q_proj.bias".to_string(),
            Tensor::new(&[0.5_f32], &device)?,
        );
        let base = FrozenBase::from_tensors(tensors, WeightKind::Dense)?;
        assert!(base.projection_weight("self_attn.q_proj").is_ok());
        assert!(base.projection_bias("self_attn.q_proj")?.is_some());
        assert!(base.projection_bias("self_attn.k_proj")?.is_none());
        Ok(())
    }

    #[test]
    fn tensor_names_are_sorted() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert("b".to_string(), Tensor::new(&[1.0_f32], &device)?);
        tensors.insert("a".to_string(), Tensor::new(&[2.0_f32], &device)?);
        let base = FrozenBase::from_tensors(tensors, WeightKind::Dense)?;
        assert_eq!(base.tensor_names(), vec!["a".to_string(), "b".to_string()]);
        Ok(())
    }

    #[test]
    fn safetensors_fp8_checkpoint_is_dequantized() -> anyhow::Result<()> {
        use candle_core::safetensors;

        let device = Device::Cpu;
        let directory = tempfile::tempdir()?;
        let file = directory.path().join("model.safetensors");
        let weights = Tensor::new(&[[1.0_f32, -2.0], [0.5, 3.0]], &device)?;
        let (quantized, scales) =
            typed_lm_common::quantization::quantize_fp8_per_channel(&weights)?;
        let mut map = HashMap::new();
        map.insert("layer.weight".to_string(), quantized);
        map.insert(
            typed_lm_common::quantization::scale_tensor_name("layer.weight"),
            scales,
        );
        safetensors::save(&map, &file)?;

        let tensors = load_checkpoint_tensors(
            &WeightLayout::Safetensors { files: vec![file] },
            QuantizationScheme::Fp8,
            &device,
        )?;
        // The scale tensor is consumed by the dequantization.
        assert_eq!(tensors.len(), 1);
        let restored = tensors
            .get("layer.weight")
            .ok_or_else(|| anyhow::anyhow!("missing dequantized weight"))?;
        assert_eq!(restored.dims(), &[2, 2]);
        assert_eq!(restored.dtype(), DType::F32);
        Ok(())
    }

    #[test]
    fn scheme_mapping_covers_every_weight_kind() {
        assert_eq!(
            scheme_for_weight_kind(WeightKind::Float8),
            QuantizationScheme::Fp8
        );
        assert_eq!(
            scheme_for_weight_kind(WeightKind::Float4),
            QuantizationScheme::Fp4
        );
        assert_eq!(
            scheme_for_weight_kind(WeightKind::Dense),
            QuantizationScheme::None
        );
        assert_eq!(
            scheme_for_weight_kind(WeightKind::Quantized),
            QuantizationScheme::None
        );
    }
}
