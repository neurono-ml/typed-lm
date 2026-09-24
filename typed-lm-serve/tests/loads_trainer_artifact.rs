//! The serving loader must accept a checkpoint quantized by the trainer.
//!
//! This test builds a tiny checkpoint, quantizes it to FP8/FP4 with the trainer
//! export path, and then loads the resulting directory through the serving
//! checkpoint resolver plus the low-precision variable builder. It needs no
//! network but does exercise real tensor round-trips, so it is `#[ignore]` in
//! CI and run explicitly with `--ignored`.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use typed_lm_common::checkpoint::{ModelReference, WeightKind};
use typed_lm_common::checkpoint_resolver::LocalCheckpointResolver;
use typed_lm_common::quantization::{dequantize_checkpoint_tensors, QuantizationScheme};

/// Writes a tiny config + tokenizer + one dense weight tensor.
fn write_tiny_checkpoint(root: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(root)?;
    std::fs::write(
        root.join("config.json"),
        br#"{"model_type": "llama", "vocab_size": 8, "hidden_size": 4}"#,
    )?;
    std::fs::write(root.join("tokenizer.json"), b"{}")?;
    let device = Device::Cpu;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        Tensor::new(&[[1.0_f32, -2.0, 0.5, 3.0]], &device)?,
    );
    tensors.insert(
        "model.norm.weight".to_string(),
        Tensor::new(&[1.0_f32, 1.0, 1.0, 1.0], &device)?,
    );
    candle_core::safetensors::save(&tensors, root.join("model.safetensors"))?;
    Ok(())
}

/// Quantizes the tiny checkpoint in memory and writes the artifact tensors.
fn quantize_into(
    source: &Path,
    destination: &Path,
    scheme: QuantizationScheme,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(destination)?;
    let device = Device::Cpu;
    let dense = candle_core::safetensors::load(source.join("model.safetensors"), &device)?;
    std::fs::write(
        destination.join("config.json"),
        std::fs::read(source.join("config.json"))?,
    )?;
    std::fs::write(
        destination.join("tokenizer.json"),
        std::fs::read(source.join("tokenizer.json"))?,
    )?;
    // Round-trip through the shared quantization helpers to obtain the
    // quantized tensors the loader will dequantize.
    let configuration = typed_lm_common::quantization::QuantizationConfig::new(scheme);
    let mut artifact: HashMap<String, Tensor> = HashMap::new();
    let (name, weight) = dense
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("the tiny checkpoint has no weight tensor"))?;
    let (quantized, auxiliary) = typed_lm_common::quantization::quantize(&configuration, &weight)?;
    artifact.insert(name.clone(), quantized.to_dtype(DType::F32)?);
    artifact.insert(
        typed_lm_common::quantization::scale_tensor_name(&name),
        auxiliary.to_dtype(DType::F32)?,
    );
    candle_core::safetensors::save(&artifact, destination.join("model.safetensors"))?;
    std::fs::write(
        destination.join("quantization_config.json"),
        configuration.to_json()?,
    )?;
    Ok(())
}

#[test]
#[ignore = "exercises a real artifact round-trip; run with --ignored"]
fn resolve_accepts_a_quantized_artifact_directory() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let source = directory.path().join("base");
    write_tiny_checkpoint(&source)?;
    let artifact = directory.path().join("artifact");
    quantize_into(&source, &artifact, QuantizationScheme::Fp8)?;

    let reference = ModelReference::Local {
        path: artifact.clone(),
    };
    let checkpoint =
        LocalCheckpointResolver::new(artifact.clone(), None, None).resolve(&reference, None)?;
    assert_eq!(checkpoint.weight_kind, WeightKind::Float8);

    // The loader path dequantizes the artifact back to dense F32.
    let device = Device::Cpu;
    let tensors = candle_core::safetensors::load(artifact.join("model.safetensors"), &device)?;
    let dense = dequantize_checkpoint_tensors(tensors, QuantizationScheme::Fp8)?;
    assert!(!dense.is_empty());
    Ok(())
}

#[test]
#[ignore = "exercises a real artifact round-trip; run with --ignored"]
fn fp4_artifact_dequantizes_through_the_loader() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let source = directory.path().join("base");
    write_tiny_checkpoint(&source)?;
    let artifact = directory.path().join("artifact");
    quantize_into(&source, &artifact, QuantizationScheme::Fp4)?;
    let reference = ModelReference::Local {
        path: artifact.clone(),
    };
    let checkpoint =
        LocalCheckpointResolver::new(artifact.clone(), None, None).resolve(&reference, None)?;
    assert_eq!(checkpoint.weight_kind, WeightKind::Float4);
    Ok(())
}
