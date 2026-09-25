//! Live end-to-end GPU test: real checkpoint -> LoRA on CUDA -> FP8 PTQ.
//!
//! Unlike the dummy tests, this one downloads a real (tiny) Hugging Face
//! checkpoint and runs the full trainer pipeline on the GPU. It is marked
//! `#[ignore]` because it needs network access, a CUDA device and a
//! `--features cuda` build, so it never runs in CI:
//!
//! ```bash
//! cargo test -p typed-lm-trainer --features cuda --test live_gpu_e2e -- --ignored --nocapture
//! ```
//!
//! The checkpoint is a tiny random Llama (`hf-internal-testing/tiny-random-
//! LlamaForCausalLM`) so the run is quick while still exercising real weights,
//! real tokenization and the CUDA code paths.

mod support;

use candle_core::Device;
use typed_lm_common::checkpoint::{ModelArchitecture, ModelReference};
use typed_lm_common::checkpoint_resolver::LocalCheckpointResolver;
use typed_lm_common::model_config::ParallelModelConfig;
use typed_lm_common::prompt_template::PromptTemplate;
use typed_lm_common::quantization::QuantizationScheme;
use typed_lm_common::tokenizer::load_tokenizer;

use typed_lm_trainer::dataset::collate::{build_batches, tokenize_item};
use typed_lm_trainer::dataset::discovery::discover_dataset_files;
use typed_lm_trainer::dataset::loader::load_records;
use typed_lm_trainer::dataset::record::expand_records;
use typed_lm_trainer::model::trainable_llama::{LoRAConfiguration, TrainableLlama};
use typed_lm_trainer::model::weight_loading::FrozenBase;
use typed_lm_trainer::quantization::export::{export_quantized, load_quantized};
use typed_lm_trainer::training::r#loop::{train, TrainingLoopConfiguration};

/// Repository of the tiny deterministic Llama used by the live test.
const TINY_MODEL_REPOSITORY: &str = "hf-internal-testing/tiny-random-LlamaForCausalLM";

/// Downloads `TINY_MODEL_REPOSITORY` into the local Hugging Face cache.
///
/// Returns the snapshot directory holding `config.json`, `tokenizer.json` and
/// `model.safetensors`.
fn download_tiny_checkpoint() -> anyhow::Result<std::path::PathBuf> {
    use hf_hub::{split_id, HFClientSync};
    let client = HFClientSync::new().map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let (owner, name) = split_id(TINY_MODEL_REPOSITORY);
    let repository = client.model(owner, name);
    let configuration = repository
        .download_file()
        .filename("config.json")
        .send()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let snapshot = configuration
        .parent()
        .ok_or_else(|| anyhow::anyhow!("downloaded config has no parent directory"))?
        .to_path_buf();
    repository
        .download_file()
        .filename("tokenizer.json")
        .send()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    repository
        .download_file()
        .filename("model.safetensors")
        .send()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(snapshot)
}

#[test]
#[ignore = "requires network, a CUDA device and a --features cuda build"]
fn train_and_quantize_on_cuda() -> anyhow::Result<()> {
    // A CUDA build without a GPU must fail loudly rather than silently test CPU.
    let device = Device::new_cuda(0).map_err(|error| {
        anyhow::anyhow!(
            "the live GPU test needs a CUDA device (build with --features cuda): {error}"
        )
    })?;

    let checkpoint_root = download_tiny_checkpoint()?;
    let reference = ModelReference::Local {
        path: checkpoint_root.clone(),
    };
    let checkpoint = LocalCheckpointResolver::new(checkpoint_root.clone(), None, None)
        .resolve(&reference, None)?;
    assert_eq!(checkpoint.architecture, ModelArchitecture::Llama);

    let configuration = ParallelModelConfig::from_json_for_architecture(
        serde_json::from_slice(&std::fs::read(checkpoint.resolved.config_file.clone())?)?,
        checkpoint.architecture,
    )?;

    // The dummy tokenizer covers the labels but not the free-form state, so the
    // dataset is kept to the exact words the tiny vocabulary knows.
    let directory = tempfile::tempdir()?;
    let dataset = directory.path().join("dataset.jsonl");
    support::write_jev_dataset(&dataset)?;

    // Load the real weights on the GPU and build the batches.
    let frozen_base = FrozenBase::from_checkpoint(&checkpoint, &device)?;
    assert!(!frozen_base.is_empty());
    let files = discover_dataset_files(&dataset)?;
    let records = load_records(&files)?;
    let items = expand_records(&records)?;
    let tokenizer = load_tokenizer(&checkpoint.resolved.tokenizer_file)?;
    let template = PromptTemplate::for_architecture(configuration.architecture);
    let tokenized = items
        .iter()
        .map(|item| tokenize_item(&tokenizer, template, "", item))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let batches = build_batches(tokenized, 2, 128, &device)?;
    assert!(!batches.is_empty());

    // Train LoRA on CUDA.
    let mut variable_map = candle_nn::VarMap::new();
    let model = TrainableLlama::load(
        frozen_base.tensors(),
        &configuration,
        LoRAConfiguration::new(4, 8.0),
        &mut variable_map,
        &device,
    )?;
    let loop_configuration = TrainingLoopConfiguration {
        epochs: 2,
        learning_rate: 1e-2,
        warmup_steps: 0,
        weight_decay: 0.0,
        maximum_gradient_norm: 1.0,
        gradient_accumulation_steps: 1,
        minimum_improvement: 0.0,
        early_stop_patience: 0,
        minimum_learning_rate_ratio: 0.0,
    };
    let outcome = train(&model, &batches, &loop_configuration, &device)?;
    assert!(!outcome.epochs.is_empty(), "expected at least one epoch");

    // Post-training quantization to FP8 and reload the artifact on the GPU.
    let quantized_directory = directory.path().join("quantized");
    export_quantized(
        frozen_base.tensors().clone(),
        QuantizationScheme::Fp8,
        &quantized_directory,
    )?;
    assert!(quantized_directory.join("model.safetensors").exists());
    assert!(quantized_directory
        .join("quantization_config.json")
        .exists());
    let restored = load_quantized(&quantized_directory, &device)?;
    assert_eq!(restored.len(), frozen_base.len());
    for tensor in restored.values() {
        assert_eq!(tensor.dtype(), candle_core::DType::F32);
    }
    Ok(())
}
