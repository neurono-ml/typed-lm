//! The serving loader must accept a checkpoint quantized by the trainer.
//!
//! This test builds a tiny checkpoint and writes a quantized artifact the same
//! way the trainer's export does (native `F8_E4M3` + per-channel scales for FP8,
//! packed `U8` nibbles + `U8` exponents for FP4), then loads the directory
//! through the serving checkpoint resolver and the shared dequantizer. It needs
//! no network but does exercise real tensor round-trips, so it is `#[ignore]` in
//! CI and run explicitly with `--ignored`.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use typed_lm_common::checkpoint::{ModelReference, WeightKind};
use typed_lm_common::checkpoint_resolver::LocalCheckpointResolver;
use typed_lm_common::quantization::{
    dequantize_checkpoint_tensors, scale_tensor_name, shape_tensor_name, QuantizationConfig,
    QuantizationScheme,
};

/// Writes a minimal valid tokenizer so the resolver accepts the directory.
fn write_tokenizer(path: &Path) -> anyhow::Result<()> {
    use std::collections::HashMap as TokenMap;
    use tokenizers::models::wordlevel::WordLevelBuilder;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::Tokenizer;

    let vocabulary: TokenMap<String, u32> = [("A".to_string(), 0_u32), ("[UNK]".to_string(), 1)]
        .into_iter()
        .collect();
    let word_level = WordLevelBuilder::default()
        .vocab(vocabulary)
        .unk_token("[UNK]".to_string())
        .build()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let mut tokenizer = Tokenizer::new(word_level);
    tokenizer.with_pre_tokenizer(Whitespace);
    tokenizer
        .save(path, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(())
}

/// Writes a quantized artifact the way the trainer export does, into `destination`.
///
/// The dtype is preserved so the resolver detects the low-precision scheme:
/// FP8 keeps native `F8_E4M3` tensors; FP4 uses packed `U8` nibbles plus `U8`
/// exponents (safetensors/candle cannot convert `F4`).
fn write_quantized_artifact(destination: &Path, scheme: QuantizationScheme) -> anyhow::Result<()> {
    std::fs::create_dir_all(destination)?;
    std::fs::write(
        destination.join("config.json"),
        br#"{"model_type": "llama", "vocab_size": 8, "hidden_size": 4}"#,
    )?;
    write_tokenizer(&destination.join("tokenizer.json"))?;

    let device = Device::Cpu;
    let weights = Tensor::new(&[[1.0_f32, -2.0, 0.5, 3.0]], &device)?;
    let configuration = QuantizationConfig::new(scheme);
    let mut artifact: HashMap<String, Tensor> = HashMap::new();
    let name = "model.embed_tokens.weight".to_string();
    match scheme {
        QuantizationScheme::Fp8 => {
            let (quantized, scales) =
                typed_lm_common::quantization::quantize(&configuration, &weights)?;
            artifact.insert(name.clone(), quantized);
            artifact.insert(scale_tensor_name(&name), scales);
        }
        QuantizationScheme::Fp4 => {
            let (packed, exponents) = typed_lm_common::quantization::quantize_fp4_mxfp4(&weights)?;
            let shape: Vec<u32> = weights
                .dims()
                .iter()
                .map(|dimension| *dimension as u32)
                .collect();
            artifact.insert(name.clone(), packed.to_dtype(DType::U8)?);
            artifact.insert(scale_tensor_name(&name), exponents.to_dtype(DType::U8)?);
            // The original shape is stored so a 2-D weight round-trips exactly.
            artifact.insert(
                shape_tensor_name(&name),
                Tensor::from_vec(shape, (weights.dims().len(),), &device)?,
            );
        }
        QuantizationScheme::None => {
            artifact.insert(name.clone(), weights.to_dtype(DType::F32)?);
        }
    }
    candle_core::safetensors::save(&artifact, destination.join("model.safetensors"))?;
    std::fs::write(
        destination.join("quantization_config.json"),
        configuration.to_json()?,
    )?;
    Ok(())
}

#[test]
#[ignore = "exercises a real artifact round-trip; run with --ignored"]
fn resolve_accepts_an_fp8_artifact_directory() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let artifact = directory.path().join("artifact");
    write_quantized_artifact(&artifact, QuantizationScheme::Fp8)?;

    let reference = ModelReference::Local {
        path: artifact.clone(),
    };
    let checkpoint =
        LocalCheckpointResolver::new(artifact.clone(), None, None).resolve(&reference, None)?;
    assert_eq!(checkpoint.weight_kind, WeightKind::Float8);

    // The loader dequantizes the artifact back to dense F32.
    let device = Device::Cpu;
    let tensors = candle_core::safetensors::load(artifact.join("model.safetensors"), &device)?;
    let dense = dequantize_checkpoint_tensors(tensors, QuantizationScheme::Fp8)?;
    assert!(!dense.is_empty());
    for tensor in dense.values() {
        assert_eq!(tensor.dtype(), DType::F32);
    }
    Ok(())
}

#[test]
#[ignore = "exercises a real artifact round-trip; run with --ignored"]
fn resolve_accepts_an_fp4_artifact_directory() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let artifact = directory.path().join("artifact");
    write_quantized_artifact(&artifact, QuantizationScheme::Fp4)?;

    let reference = ModelReference::Local {
        path: artifact.clone(),
    };
    let checkpoint =
        LocalCheckpointResolver::new(artifact.clone(), None, None).resolve(&reference, None)?;
    assert_eq!(checkpoint.weight_kind, WeightKind::Float4);

    let device = Device::Cpu;
    let tensors = candle_core::safetensors::load(artifact.join("model.safetensors"), &device)?;
    let dense = dequantize_checkpoint_tensors(tensors, QuantizationScheme::Fp4)?;
    assert!(!dense.is_empty());
    let weight = dense
        .get("model.embed_tokens.weight")
        .ok_or_else(|| anyhow::anyhow!("missing dequantized embedding weight"))?;
    assert_eq!(
        weight.dims(),
        &[1, 4],
        "FP4 must restore the original shape"
    );
    Ok(())
}
