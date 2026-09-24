//! End-to-end from-scratch training over a synthetic Jev dataset (CPU).
//!
//! Uses the library pipeline directly (rather than the CLI) so it exercises the
//! exact path `main` wires together for `--method from-scratch`: deterministic
//! initialization from geometry -> fully-trainable dense model -> epoch loop ->
//! complete checkpoint save -> reload. No network, no GPU, no real model.

mod support;

use candle_core::Device;
use candle_nn::VarMap;
use typed_lm_common::architecture_traits::DenseArchitectureTraits;
use typed_lm_common::checkpoint::ModelArchitecture;
use typed_lm_common::prompt_template::PromptTemplate;
use typed_lm_common::tokenizer::load_tokenizer;

use typed_lm_common::model_config::ParallelModelConfig;
use typed_lm_trainer::dataset::collate::{build_batches, tokenize_item};
use typed_lm_trainer::dataset::discovery::discover_dataset_files;
use typed_lm_trainer::dataset::loader::load_records;
use typed_lm_trainer::dataset::record::expand_records;
use typed_lm_trainer::model::initialization::{
    initialize_model_tensors, InitializationConfiguration,
};
use typed_lm_trainer::model::trainable_full::TrainableFull;
use typed_lm_trainer::training::checkpoint::{load_full_checkpoint, save_full_checkpoint};
use typed_lm_trainer::training::r#loop::{train, TrainingLoopConfiguration};

/// Geometry for a tiny from-scratch model matching the synthetic tokenizer.
fn tiny_configuration(architecture: ModelArchitecture) -> ParallelModelConfig {
    let traits = DenseArchitectureTraits::for_architecture(architecture);
    ParallelModelConfig {
        architecture,
        vocab_size: 64,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        tie_word_embeddings: false,
        rope_scaling: None,
        attention_bias: architecture.has_query_key_value_bias(),
        explicit_head_dimension: traits.explicit_head_dimension.then_some(4),
        sliding_window: None,
        max_window_layers: 0,
        logit_softcapping: None,
        attention_logit_softcapping: None,
        query_pre_attention_scalar: None,
        rms_norm_unit_offset: traits.rms_norm_unit_offset,
        embedding_scale: traits.scales_embeddings.then_some(4.0),
        rope_local_base_frequency: None,
    }
}

#[test]
fn from_scratch_overfit_reduces_loss_and_saves_a_complete_checkpoint() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let directory = tempfile::tempdir()?;
    let checkpoint = directory.path().join("model");
    support::write_tiny_checkpoint(&checkpoint)?;
    let dataset = directory.path().join("dataset.jsonl");
    support::write_jev_dataset(&dataset)?;

    let configuration = tiny_configuration(ModelArchitecture::Llama);

    // Deterministically initialize every parameter.
    let tensors = initialize_model_tensors(
        &configuration,
        &InitializationConfiguration::default(),
        42,
        &device,
    )?;
    assert!(!tensors.is_empty());

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

    // Build the fully-trainable model and train every parameter.
    let mut variable_map = VarMap::new();
    let model = TrainableFull::from_initialized(&tensors, &configuration, &mut variable_map, &device)?;
    assert!(!model.all_trainable_variables().is_empty());

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
        "from-scratch overfit must reduce the loss: {} -> {}",
        first.mean_loss,
        last.mean_loss
    );

    // Persist and reload a complete checkpoint.
    let output = directory.path().join("output");
    let state = model.state_dict()?;
    let weights_path = save_full_checkpoint(
        &state,
        &configuration,
        &checkpoint.join("tokenizer.json"),
        &output,
    )?;
    assert!(weights_path.exists());
    assert!(output.join("config.json").exists());
    assert!(output.join("tokenizer.json").exists());

    let (reloaded, reloaded_configuration) = load_full_checkpoint(&output, &device)?;
    assert_eq!(reloaded_configuration.architecture, ModelArchitecture::Llama);
    assert_eq!(reloaded.len(), state.len());
    Ok(())
}

#[test]
fn from_scratch_is_deterministic_for_a_fixed_seed() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let configuration = tiny_configuration(ModelArchitecture::Qwen3);
    let first = initialize_model_tensors(
        &configuration,
        &InitializationConfiguration::default(),
        7,
        &device,
    )?;
    let second = initialize_model_tensors(
        &configuration,
        &InitializationConfiguration::default(),
        7,
        &device,
    )?;
    let mut names: Vec<&String> = first.keys().collect();
    names.sort();
    for name in names {
        let left = first
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing {name}"))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let right = second
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing {name}"))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(left, right, "tensor {name} must be seed-deterministic");
    }
    Ok(())
}
