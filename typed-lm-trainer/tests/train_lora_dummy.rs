//! End-to-end LoRA training over a synthetic Jev dataset.
//!
//! Uses the library pipeline directly (rather than the CLI) so it exercises the
//! exact path `main` wires together: discovery -> records -> items -> tokenize ->
//! batches -> trainable model -> epoch loop -> adapter save. No network, no GPU.

mod support;

use candle_core::Device;
use candle_nn::VarMap;
use typed_lm_common::checkpoint::ModelArchitecture;
use typed_lm_common::prompt_template::PromptTemplate;
use typed_lm_common::tokenizer::load_tokenizer;

use typed_lm_common::checkpoint::WeightKind;
use typed_lm_common::model_config::ParallelModelConfig;
use typed_lm_trainer::dataset::collate::{build_batches, tokenize_item};
use typed_lm_trainer::dataset::discovery::discover_dataset_files;
use typed_lm_trainer::dataset::loader::load_records;
use typed_lm_trainer::dataset::record::expand_records;
use typed_lm_trainer::model::trainable_llama::{LoRAConfiguration, TrainableLlama};
use typed_lm_trainer::model::weight_loading::FrozenBase;
use typed_lm_trainer::training::checkpoint::{load_adapter, save_adapter, AdapterConfiguration};
use typed_lm_trainer::training::r#loop::{train, TrainingLoopConfiguration};

#[test]
fn lora_overfit_dummy_reduces_loss_and_saves_an_adapter() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let directory = tempfile::tempdir()?;
    let checkpoint = directory.path().join("model");
    support::write_tiny_checkpoint(&checkpoint)?;
    let dataset = directory.path().join("dataset.jsonl");
    support::write_jev_dataset(&dataset)?;

    // Load the frozen base and configuration.
    let frozen_base = FrozenBase::from_tensors(support::tiny_weights()?, WeightKind::Dense)?;
    let configuration = ParallelModelConfig::from_json_for_architecture(
        support::tiny_configuration_json(),
        ModelArchitecture::Llama,
    )?;

    // Build the decision-position batches.
    let files = discover_dataset_files(&dataset)?;
    let records = load_records(&files)?;
    let items = expand_records(&records)?;
    let tokenizer = load_tokenizer(&checkpoint.join("tokenizer.json"))?;
    let template = PromptTemplate::for_architecture(configuration.architecture);
    let tokenized = items
        .iter()
        .map(|item| tokenize_item(&tokenizer, template, "", item))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let batches = build_batches(tokenized, 2, 64, &device)?;
    assert!(!batches.is_empty());

    // Attach LoRA adapters and train.
    let mut variable_map = VarMap::new();
    let model = TrainableLlama::load(
        frozen_base.tensors(),
        &configuration,
        LoRAConfiguration::new(4, 8.0),
        &mut variable_map,
        &device,
    )?;
    let loop_configuration = TrainingLoopConfiguration {
        epochs: 5,
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
    let first = outcome
        .epochs
        .first()
        .ok_or_else(|| anyhow::anyhow!("no epoch metrics recorded"))?;
    let last = outcome
        .epochs
        .last()
        .ok_or_else(|| anyhow::anyhow!("no epoch metrics recorded"))?;
    assert!(
        last.mean_loss < first.mean_loss,
        "LoRA overfit must reduce the loss: {} -> {}",
        first.mean_loss,
        last.mean_loss
    );

    // Save and reload the adapter.
    let adapter_directory = directory.path().join("adapter");
    let configuration_metadata =
        AdapterConfiguration::new(4, 8.0, "tiny-llama", configuration.architecture.name());
    save_adapter(&variable_map, &configuration_metadata, &adapter_directory)?;
    let (_, loaded_configuration) = load_adapter(&adapter_directory, &device)?;
    assert_eq!(loaded_configuration.rank, 4);
    Ok(())
}
