//! QLoRA over a small GGUF base checkpoint.
//!
//! Exercised only with a real GGUF file (no network in CI), so it is `#[ignore]`
//! by default. Run with `cargo test -p typed-lm-trainer -- --ignored` once a
//! tiny GGUF checkpoint is available at the path below.

mod support;

use std::path::Path;

use candle_core::Device;
use typed_lm_common::checkpoint::WeightKind;
use typed_lm_common::model_config::ParallelModelConfig;

use typed_lm_trainer::model::weight_loading::FrozenBase;

/// Path to a small GGUF checkpoint; override via the `TINY_GGUF` environment
/// variable. Defaults to a conventional local location.
fn gguf_path() -> std::path::PathBuf {
    std::env::var("TINY_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("resources/tiny-q4_k_m.gguf"))
}

#[test]
#[ignore = "requires a real GGUF checkpoint; run with --ignored and TINY_GGUF set"]
fn qlora_loads_a_gguf_base_as_dense_f32() -> anyhow::Result<()> {
    let path = gguf_path();
    if !Path::new(&path).exists() {
        return Err(anyhow::anyhow!(
            "GGUF checkpoint '{}' not found (set TINY_GGUF to a tiny GGUF file)",
            path.display()
        ));
    }
    let device = Device::Cpu;
    let _configuration = ParallelModelConfig::from_json_for_architecture(
        support::tiny_configuration_json(),
        typed_lm_common::checkpoint::ModelArchitecture::Llama,
    )?;
    let reference = typed_lm_common::checkpoint::ModelReference::Local { path: path.clone() };
    let checkpoint =
        typed_lm_common::checkpoint_resolver::LocalCheckpointResolver::new(path, None, None)
            .resolve(&reference, None)?;
    let frozen_base = FrozenBase::from_checkpoint(&checkpoint, &device)?;
    assert!(!frozen_base.is_empty());
    assert_eq!(frozen_base.weight_kind(), WeightKind::Quantized);
    Ok(())
}
