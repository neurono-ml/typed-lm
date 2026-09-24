//! `typed-lm-trainer` entry point.
//!
//! `train` loads a frozen base checkpoint, attaches LoRA adapters, builds
//! decision-position batches from a Jev-native dataset and runs the epoch loop,
//! then saves the adapter (and, when requested, a post-training-quantized
//! artifact). `quantize` merges an optional adapter into the base and applies
//! PTQ. Both paths are thin wiring over the `dataset`, `model`, `training` and
//! `quantization` modules.

use std::collections::HashMap;

use candle_core::DType;
use candle_nn::VarMap;
use typed_lm_common::checkpoint::ModelReference;
use typed_lm_common::checkpoint_resolver::{LoadableCheckpoint, LocalCheckpointResolver};
use typed_lm_common::model_config::ParallelModelConfig;
use typed_lm_common::quantization::QuantizationScheme;
use typed_lm_common::tokenizer::load_tokenizer;

use typed_lm_trainer::cli::{
    Command, ModelGeometryArguments, QuantizationMode, QuantizeArguments, TrainArguments,
    TrainerArguments, TrainingMethod,
};
use typed_lm_trainer::dataset::collate::{build_batches, tokenize_item};
use typed_lm_trainer::dataset::discovery::discover_dataset_files;
use typed_lm_trainer::dataset::loader::load_records;
use typed_lm_trainer::dataset::record::expand_records;
use typed_lm_trainer::error::TrainerError;
use typed_lm_trainer::model::initialization::{
    initialize_model_tensors, InitializationConfiguration,
};
use typed_lm_trainer::model::trainable_dense;
use typed_lm_trainer::model::trainable_full::TrainableFull;
use typed_lm_trainer::model::trainable_llama::LoRAConfiguration;
use typed_lm_trainer::model::weight_loading::FrozenBase;
use typed_lm_trainer::quantization::export::{export_quantized, merge_adapter};
use typed_lm_trainer::training::checkpoint::{
    save_adapter, save_full_checkpoint, AdapterConfiguration,
};
use typed_lm_trainer::training::r#loop::{train, TrainingLoopConfiguration};

/// Location of the loaded context used as the system prompt during training.
///
/// Training reuses the same context as serving so the prompts (and therefore
/// the decision positions) line up. An absent file means an empty context.
const CONTEXT_FILE: &str = "resources/memory.md";

#[tokio::main]
async fn main() -> Result<(), TrainerError> {
    let arguments = TrainerArguments::parse_with_configuration()?;
    match arguments.command {
        Command::Train(train_arguments) => run_train(train_arguments).await,
        Command::Quantize(quantize_arguments) => run_quantize(quantize_arguments).await,
    }
}

/// Runs the fine-tuning pipeline.
async fn run_train(arguments: TrainArguments) -> Result<(), TrainerError> {
    let method = arguments
        .training_method()
        .map_err(|error| TrainerError::Configuration(error.to_string()))?;
    let quantization_mode = arguments
        .quantization_mode()
        .map_err(|error| TrainerError::Configuration(error.to_string()))?;
    let training_scheme = arguments
        .quantization_scheme()
        .map_err(|error| TrainerError::Configuration(error.to_string()))?;
    // `--quantization-mode training` keeps the base quantized during training,
    // so it only makes sense together with a low-precision scheme. In
    // `post-training` mode (`--quantization` set) the merged adapter is
    // quantized at the end of the run, and `none` simply skips that step.
    if quantization_mode == QuantizationMode::Training
        && training_scheme == QuantizationScheme::None
    {
        return Err(TrainerError::Configuration(
            "--quantization-mode training requires --quantization fp8 or fp4".to_string(),
        ));
    }

    let device = arguments.device.resolve()?;
    if method.trains_all_parameters() {
        return run_train_full_parameter(
            arguments,
            method,
            quantization_mode,
            training_scheme,
            &device,
        )
        .await;
    }

    let reference = ModelReference::resolve(&arguments.model_id);
    let checkpoint = resolve_local_checkpoint(&reference)?;
    tracing::info!(
        model = %reference.describe(),
        method = method.name(),
        "loading frozen base checkpoint"
    );
    let frozen_base = FrozenBase::from_checkpoint(&checkpoint, &device)?;
    let configuration = read_model_configuration(&checkpoint)?;

    // Build the decision-position batches first: this validates the dataset and
    // the answer labels before any adapter allocation.
    let dataset_files = discover_dataset_files(&arguments.dataset).map_err(TrainerError::from)?;
    let records = load_records(&dataset_files)?;
    let items = expand_records(&records)?;
    let tokenizer = load_tokenizer(&checkpoint.resolved.tokenizer_file)?;
    let template = typed_lm_common::prompt_template::PromptTemplate::for_architecture(
        configuration.architecture,
    );
    let context_text = load_context_text();
    let tokenized: Vec<_> = items
        .iter()
        .map(|item| tokenize_item(&tokenizer, template, &context_text, item))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let batches = build_batches(
        tokenized,
        arguments.batch_size,
        arguments.max_sequence_length,
        &device,
    )?;
    tracing::info!(
        records = records.len(),
        items = items.len(),
        batches = batches.len(),
        "dataset collated"
    );

    // Attach LoRA adapters over the frozen base.
    let mut variable_map = VarMap::new();
    let lora = LoRAConfiguration::new(arguments.lora_rank, arguments.lora_alpha);
    // The unified dense loader validates the configuration against the detected
    // architecture and delegates to the shared differentiable forward.
    let model = trainable_dense::load(
        frozen_base.tensors(),
        &configuration,
        lora,
        &mut variable_map,
        &device,
    )?;

    let loop_configuration = TrainingLoopConfiguration {
        epochs: arguments.epochs,
        learning_rate: arguments.learning_rate,
        warmup_steps: arguments.warmup_steps,
        weight_decay: arguments.weight_decay,
        maximum_gradient_norm: arguments.maximum_gradient_norm,
        gradient_accumulation_steps: arguments.gradient_accumulation_steps,
        minimum_improvement: arguments.minimum_improvement,
        early_stop_patience: arguments.early_stop_patience,
        minimum_learning_rate_ratio: 0.0,
    };
    let outcome = train(&model, &batches, &loop_configuration, &device)?;
    tracing::info!(
        epochs = outcome.epochs.len(),
        stopped_early = outcome.stopped_early,
        "training finished"
    );

    // Persist the adapter and its metadata.
    let adapter_configuration = AdapterConfiguration::new(
        arguments.lora_rank,
        arguments.lora_alpha,
        arguments.model_id.clone(),
        configuration.architecture.name(),
    );
    save_adapter(
        &variable_map,
        &adapter_configuration,
        &arguments.output_directory,
    )?;
    tracing::info!(
        directory = %arguments.output_directory.display(),
        "adapter saved"
    );

    // Optional post-training quantization of the merged weights.
    if quantization_mode == QuantizationMode::PostTraining
        && training_scheme != QuantizationScheme::None
    {
        let merged = merge_lora_into_base(&variable_map, &frozen_base, &configuration)?;
        export_quantized(merged, training_scheme, &arguments.output_directory)?;
        tracing::info!(scheme = training_scheme.name(), "quantized artifact saved");
    }
    Ok(())
}

/// Runs `--method full` (from a checkpoint) or `--method from-scratch`.
///
/// Both methods train every parameter. `full` initializes the model from an
/// existing dense checkpoint; `from-scratch` initializes it deterministically
/// from the geometry in `--architecture`/`--hidden-size`/… (or the TOML
/// `[model]` section) and a seed. The result is a complete dense checkpoint
/// (`model.safetensors` + `config.json` + `tokenizer.json`) that the server
/// serves directly.
async fn run_train_full_parameter(
    arguments: TrainArguments,
    method: TrainingMethod,
    quantization_mode: QuantizationMode,
    training_scheme: QuantizationScheme,
    device: &candle_core::Device,
) -> Result<(), TrainerError> {
    let from_scratch = method == TrainingMethod::FromScratch;
    let (configuration, tokenizer_path, source_tensors) = if from_scratch {
        let configuration = configuration_from_geometry(&arguments.geometry)?;
        let tokenizer_path = resolve_scratch_tokenizer(&arguments)?;
        let initialization = InitializationConfiguration::default();
        let tensors =
            initialize_model_tensors(&configuration, &initialization, arguments.seed, device)?;
        tracing::info!(
            architecture = configuration.architecture.name(),
            seed = arguments.seed,
            "initialized random weights for from-scratch training"
        );
        (configuration, tokenizer_path, tensors)
    } else {
        let reference = ModelReference::resolve(&arguments.model_id);
        let checkpoint = resolve_local_checkpoint(&reference)?;
        let configuration = read_model_configuration(&checkpoint)?;
        let frozen_base = FrozenBase::from_checkpoint(&checkpoint, device)?;
        (
            configuration,
            checkpoint.resolved.tokenizer_file.clone(),
            frozen_base.tensors().clone(),
        )
    };

    // Build the decision-position batches.
    let dataset_files = discover_dataset_files(&arguments.dataset).map_err(TrainerError::from)?;
    let records = load_records(&dataset_files)?;
    let items = expand_records(&records)?;
    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let template = typed_lm_common::prompt_template::PromptTemplate::for_architecture(
        configuration.architecture,
    );
    let context_text = load_context_text();
    let tokenized: Vec<_> = items
        .iter()
        .map(|item| tokenize_item(&tokenizer, template, &context_text, item))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let batches = build_batches(
        tokenized,
        arguments.batch_size,
        arguments.max_sequence_length,
        device,
    )?;
    tracing::info!(
        records = records.len(),
        items = items.len(),
        batches = batches.len(),
        "dataset collated"
    );

    let mut variable_map = VarMap::new();
    let model = TrainableFull::from_initialized(
        &source_tensors,
        &configuration,
        &mut variable_map,
        device,
    )?;

    let loop_configuration = TrainingLoopConfiguration {
        epochs: arguments.epochs,
        learning_rate: arguments.learning_rate,
        warmup_steps: arguments.warmup_steps,
        weight_decay: arguments.weight_decay,
        maximum_gradient_norm: arguments.maximum_gradient_norm,
        gradient_accumulation_steps: arguments.gradient_accumulation_steps,
        minimum_improvement: arguments.minimum_improvement,
        early_stop_patience: arguments.early_stop_patience,
        minimum_learning_rate_ratio: 0.0,
    };
    let outcome = train(&model, &batches, &loop_configuration, device)?;
    tracing::info!(
        epochs = outcome.epochs.len(),
        stopped_early = outcome.stopped_early,
        "training finished"
    );

    // Persist the complete checkpoint (weights + config + tokenizer).
    let state = model.state_dict()?;
    let weights_path = save_full_checkpoint(
        &state,
        &configuration,
        &tokenizer_path,
        &arguments.output_directory,
    )?;
    tracing::info!(
        directory = %arguments.output_directory.display(),
        weights = %weights_path.display(),
        "full checkpoint saved"
    );

    // Optional post-training quantization of the complete checkpoint.
    if quantization_mode == QuantizationMode::PostTraining
        && training_scheme != QuantizationScheme::None
    {
        export_quantized(state, training_scheme, &arguments.output_directory)?;
        tracing::info!(scheme = training_scheme.name(), "quantized artifact saved");
    }
    Ok(())
}

/// Builds a [`ParallelModelConfig`] from explicit CLI/TOML geometry.
fn configuration_from_geometry(
    geometry: &ModelGeometryArguments,
) -> Result<ParallelModelConfig, TrainerError> {
    let architecture_name = geometry.architecture.as_deref().ok_or_else(|| {
        TrainerError::Configuration(
            "--method from-scratch requires --architecture (or a [model] section)".to_string(),
        )
    })?;
    let architecture =
        typed_lm_common::checkpoint::ModelArchitecture::from_model_type(architecture_name)
            .ok_or_else(|| {
                TrainerError::Configuration(format!(
                    "unsupported architecture '{architecture_name}': expected one of {}",
                    typed_lm_common::checkpoint::ModelArchitecture::supported_names()
                ))
            })?;
    let required = |value: Option<usize>, flag: &str| -> Result<usize, TrainerError> {
        value.ok_or_else(|| {
            TrainerError::Configuration(format!(
                "--method from-scratch requires {flag} (or a [model] section)"
            ))
        })
    };
    let hidden_size = required(geometry.hidden_size, "--hidden-size")?;
    let num_attention_heads = required(geometry.num_attention_heads, "--num-attention-heads")?;
    let traits = typed_lm_common::architecture_traits::DenseArchitectureTraits::for_architecture(
        architecture,
    );
    let explicit_head_dimension = traits
        .explicit_head_dimension
        .then_some(hidden_size / num_attention_heads);
    Ok(ParallelModelConfig {
        architecture,
        vocab_size: required(geometry.vocab_size, "--vocab-size")?,
        hidden_size,
        intermediate_size: geometry.intermediate_size.unwrap_or(hidden_size * 4),
        num_hidden_layers: required(geometry.num_hidden_layers, "--num-hidden-layers")?,
        num_attention_heads,
        num_key_value_heads: geometry.num_key_value_heads.unwrap_or(num_attention_heads),
        max_position_embeddings: geometry.max_position_embeddings.unwrap_or(2048),
        rms_norm_eps: geometry.rms_norm_eps.unwrap_or(1e-6),
        rope_theta: geometry.rope_theta.unwrap_or(1000000.0),
        tie_word_embeddings: geometry.tie_word_embeddings.unwrap_or(false),
        rope_scaling: None,
        attention_bias: architecture.has_query_key_value_bias(),
        explicit_head_dimension,
        sliding_window: None,
        max_window_layers: 0,
        logit_softcapping: None,
        attention_logit_softcapping: None,
        query_pre_attention_scalar: None,
        rms_norm_unit_offset: traits.rms_norm_unit_offset,
        embedding_scale: traits
            .scales_embeddings
            .then_some((hidden_size as f64).sqrt()),
        rope_local_base_frequency: None,
    })
}

/// Locates the tokenizer for a from-scratch run.
///
/// A from-scratch run has no checkpoint, so the tokenizer must be provided
/// explicitly via `--tokenizer-file` or the `[tokenizer] file` TOML key.
/// Without it the run cannot build dataset batches and fails with an actionable
/// configuration error.
fn resolve_scratch_tokenizer(
    arguments: &TrainArguments,
) -> Result<std::path::PathBuf, TrainerError> {
    arguments.tokenizer_file.clone().ok_or_else(|| {
        TrainerError::Configuration(
            "--method from-scratch requires a tokenizer: pass --tokenizer-file or set \
             [tokenizer] file in the configuration file"
                .to_string(),
        )
    })
}

/// Runs the post-training quantization pipeline.
async fn run_quantize(arguments: QuantizeArguments) -> Result<(), TrainerError> {
    let scheme = arguments
        .quantization_scheme()
        .map_err(|error| TrainerError::Configuration(error.to_string()))?;
    let device = arguments.device.resolve()?;
    let reference = ModelReference::resolve(&arguments.model_id);
    let checkpoint = resolve_local_checkpoint(&reference)?;
    let frozen_base = FrozenBase::from_checkpoint(&checkpoint, &device)?;

    let merged = match &arguments.adapter_directory {
        Some(adapter_directory) => {
            let (variable_map, _adapter_configuration) =
                typed_lm_trainer::training::checkpoint::load_adapter(adapter_directory, &device)?;
            let configuration = read_model_configuration(&checkpoint)?;
            merge_lora_into_base(&variable_map, &frozen_base, &configuration)?
        }
        None => frozen_base.tensors().clone(),
    };
    let weights_path = export_quantized(merged, scheme, &arguments.output_directory)?;
    tracing::info!(
        path = %weights_path.display(),
        scheme = scheme.name(),
        "quantized artifact written"
    );
    Ok(())
}

/// Resolves a model reference to a loadable checkpoint, requiring a local path.
///
/// Training needs the tokenizer, config and weights on disk; a Hub identifier
/// must be downloaded first (the serving loader can prefetch it).
fn resolve_local_checkpoint(reference: &ModelReference) -> anyhow::Result<LoadableCheckpoint> {
    match reference {
        ModelReference::Local { path } => {
            LocalCheckpointResolver::new(path.clone(), None, None).resolve(reference, None)
        }
        ModelReference::Hub { repository } => Err(anyhow::anyhow!(
            "training requires a local checkpoint; '{repository}' is a Hub identifier. \
             Download it first and pass the local directory via --model-id."
        )),
    }
}

/// Reads `config.json` into a [`ParallelModelConfig`].
fn read_model_configuration(
    checkpoint: &LoadableCheckpoint,
) -> anyhow::Result<ParallelModelConfig> {
    let bytes = std::fs::read(&checkpoint.resolved.config_file).map_err(|error| {
        TrainerError::Model(format!(
            "failed to read config '{}': {error}",
            checkpoint.resolved.config_file.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        TrainerError::Model(format!(
            "failed to parse config '{}': {error}",
            checkpoint.resolved.config_file.display()
        ))
    })?;
    ParallelModelConfig::from_json_for_architecture(value, checkpoint.architecture)
}

/// Loads the shared context used as the training system prompt.
fn load_context_text() -> String {
    std::fs::read_to_string(CONTEXT_FILE).unwrap_or_default()
}

/// Folds every LoRA adapter delta into the frozen base weight map.
fn merge_lora_into_base(
    variable_map: &VarMap,
    frozen_base: &FrozenBase,
    configuration: &ParallelModelConfig,
) -> anyhow::Result<HashMap<String, candle_core::Tensor>> {
    let adapter_tensors = adapter_deltas(variable_map, configuration)?;
    merge_adapter(frozen_base.tensors().clone(), adapter_tensors)
}

/// Extracts the adapter variables under their `VarMap` names.
fn adapter_variables(
    variable_map: &VarMap,
) -> anyhow::Result<HashMap<String, candle_core::Tensor>> {
    let data = variable_map.data();
    let guard = data.lock().map_err(|error| {
        TrainerError::Training(format!("failed to lock the variable map: {error}"))
    })?;
    let mut variables: HashMap<String, candle_core::Tensor> = HashMap::with_capacity(guard.len());
    for (name, variable) in guard.iter() {
        variables.insert(name.clone(), variable.as_tensor().clone());
    }
    Ok(variables)
}

/// Derives the merged deltas for the dense base weight names.
///
/// The exported adapter tensors are named `...lora_a`/`...lora_b`; the merged
/// delta for a base weight `X.weight` is `B @ A` and keeps the base weight's
/// name so [`merge_adapter`] can add it.
fn adapter_deltas(
    variable_map: &VarMap,
    configuration: &ParallelModelConfig,
) -> anyhow::Result<HashMap<String, candle_core::Tensor>> {
    let mut variables = adapter_variables(variable_map)?;
    let mut deltas: HashMap<String, candle_core::Tensor> = HashMap::new();
    for layer_index in 0..configuration.num_hidden_layers {
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let prefix = format!("model.layers.{layer_index}.self_attn.{projection}");
            insert_projection_delta(&mut variables, &mut deltas, &prefix)?;
        }
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let prefix = format!("model.layers.{layer_index}.mlp.{projection}");
            insert_projection_delta(&mut variables, &mut deltas, &prefix)?;
        }
    }
    Ok(deltas)
}

/// Computes the merged delta of one LoRA-wrapped projection, if present.
fn insert_projection_delta(
    variables: &mut HashMap<String, candle_core::Tensor>,
    deltas: &mut HashMap<String, candle_core::Tensor>,
    prefix: &str,
) -> anyhow::Result<()> {
    let down_name = format!("{prefix}.lora_a");
    let up_name = format!("{prefix}.lora_b");
    let (Some(down), Some(up)) = (variables.remove(&down_name), variables.remove(&up_name)) else {
        return Ok(());
    };
    // `lora_a` is (rank, input), `lora_b` is (output, rank); delta = B @ A.
    let delta = up.matmul(&down)?.to_dtype(DType::F32)?;
    deltas.insert(format!("{prefix}.weight"), delta);
    Ok(())
}
