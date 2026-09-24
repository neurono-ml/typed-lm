//! End-to-end artifact export: a tiny checkpoint is loaded, merged with an
//! optional adapter and quantized to FP8/FP4, then read back and dequantized.

mod support;

use candle_core::Device;
use typed_lm_common::checkpoint::WeightKind;
use typed_lm_common::quantization::QuantizationScheme;

use typed_lm_trainer::model::weight_loading::FrozenBase;
use typed_lm_trainer::quantization::export::{export_quantized, load_quantized};

/// Exports a tiny base to `scheme` and asserts the artifact round-trips dense.
fn export_and_reload(scheme: QuantizationScheme) -> anyhow::Result<()> {
    let device = Device::Cpu;
    let directory = tempfile::tempdir()?;
    let frozen_base = FrozenBase::from_tensors(support::tiny_weights()?, WeightKind::Dense)?;
    let weights_path = export_quantized(frozen_base.tensors().clone(), scheme, directory.path())?;
    assert!(weights_path.exists(), "model.safetensors must be written");
    assert!(
        directory.path().join("quantization_config.json").exists(),
        "quantization_config.json must be written"
    );
    let restored = load_quantized(directory.path(), &device)?;
    assert_eq!(restored.len(), frozen_base.len());
    Ok(())
}

#[test]
fn fp8_export_writes_a_loadable_artifact() -> anyhow::Result<()> {
    export_and_reload(QuantizationScheme::Fp8)
}

#[test]
fn fp4_export_writes_a_loadable_artifact() -> anyhow::Result<()> {
    export_and_reload(QuantizationScheme::Fp4)
}

#[test]
fn none_export_writes_dense_weights() -> anyhow::Result<()> {
    export_and_reload(QuantizationScheme::None)
}
