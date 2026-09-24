//! Configuration resolution with **CLI > TOML > default** precedence.
//!
//! Clap materializes defaults into the parsed value, so "the flag was absent"
//! cannot be decided from the value alone. The resolver therefore inspects
//! [`ArgMatches::value_source`] and only lets a TOML value override a field when
//! its CLI flag was **not** provided on the command line. The resulting
//! precedence is:
//!
//! 1. an explicit CLI flag,
//! 2. otherwise the corresponding TOML key,
//! 3. otherwise the clap default.
//!
//! The resolution is pure: it reads the parsed matches and the configuration
//! file and produces the effective [`TrainArguments`] plus the derived
//! initialization configuration, so every combination is unit-testable without
//! touching the filesystem or a model.

use clap::parser::ValueSource;
use clap::ArgMatches;

use crate::cli::{ModelGeometryArguments, TrainArguments};
use crate::configuration_file::{ConfigurationFile, InitializationSection, ModelSection};
use crate::model::initialization::InitializationConfiguration;

/// Whether a CLI flag was provided explicitly rather than defaulted.
pub fn flag_is_explicit(matches: &ArgMatches, flag: &str) -> bool {
    matches.value_source(flag) == Some(ValueSource::CommandLine)
}

/// Resolves the train arguments with CLI over TOML over default precedence.
///
/// `matches` are the parsed matches of the `train` subcommand; `configuration`
/// is the (possibly empty) TOML file. The returned arguments carry the effective
/// values the training run must use.
pub fn resolve_train_arguments(
    matches: &ArgMatches,
    configuration: &ConfigurationFile,
) -> anyhow::Result<TrainArguments> {
    let mut arguments =
        <TrainArguments as clap::FromArgMatches>::from_arg_matches(matches).map_err(|error| {
            crate::error::TrainerError::Configuration(format!(
                "failed to rebuild the train arguments from matches: {error}"
            ))
        })?;
    let run = configuration.run();

    if !flag_is_explicit(matches, "model_id") {
        if let Some(value) = run.and_then(|section| section.model_id.as_ref()) {
            arguments.model_id = value.clone();
        }
    }
    if !flag_is_explicit(matches, "dataset") {
        if let Some(value) = configuration
            .dataset()
            .and_then(|section| section.path.as_ref())
        {
            arguments.dataset = value.into();
        }
    }
    if !flag_is_explicit(matches, "output_directory") {
        if let Some(value) = run.and_then(|section| section.output_directory.as_ref()) {
            arguments.output_directory = value.into();
        }
    }
    if !flag_is_explicit(matches, "method") {
        if let Some(value) = run.and_then(|section| section.method.as_ref()) {
            arguments.method = value.clone();
        }
    }
    if !flag_is_explicit(matches, "quantization") {
        if let Some(value) = run.and_then(|section| section.quantization.as_ref()) {
            arguments.quantization = value.clone();
        }
    }
    if !flag_is_explicit(matches, "quantization_mode") {
        if let Some(value) = run.and_then(|section| section.quantization_mode.as_ref()) {
            arguments.quantization_mode = value.clone();
        }
    }
    if !flag_is_explicit(matches, "device") {
        if let Some(value) = run.and_then(|section| section.device.as_deref()) {
            arguments.device = parse_trainer_device(value)?;
        }
    }
    if !flag_is_explicit(matches, "seed") {
        if let Some(value) = run.and_then(|section| section.seed) {
            arguments.seed = value;
        }
    }
    if !flag_is_explicit(matches, "tokenizer_file") {
        if let Some(value) = configuration
            .tokenizer()
            .and_then(|section| section.file.as_ref())
        {
            arguments.tokenizer_file = Some(value.into());
        }
    }
    if !flag_is_explicit(matches, "lora_rank") {
        if let Some(value) = run.and_then(|section| section.lora_rank) {
            arguments.lora_rank = value;
        }
    }
    if !flag_is_explicit(matches, "lora_alpha") {
        if let Some(value) = run.and_then(|section| section.lora_alpha) {
            arguments.lora_alpha = value;
        }
    }
    if !flag_is_explicit(matches, "lora_dropout") {
        if let Some(value) = run.and_then(|section| section.lora_dropout) {
            arguments.lora_dropout = value;
        }
    }
    if !flag_is_explicit(matches, "epochs") {
        if let Some(value) = run.and_then(|section| section.epochs) {
            arguments.epochs = value;
        }
    }
    if !flag_is_explicit(matches, "batch_size") {
        if let Some(value) = run.and_then(|section| section.batch_size) {
            arguments.batch_size = value;
        }
    }
    if !flag_is_explicit(matches, "gradient_accumulation_steps") {
        if let Some(value) = run.and_then(|section| section.gradient_accumulation_steps) {
            arguments.gradient_accumulation_steps = value;
        }
    }
    if !flag_is_explicit(matches, "learning_rate") {
        if let Some(value) = run.and_then(|section| section.learning_rate) {
            arguments.learning_rate = value;
        }
    }
    if !flag_is_explicit(matches, "warmup_steps") {
        if let Some(value) = run.and_then(|section| section.warmup_steps) {
            arguments.warmup_steps = value;
        }
    }
    if !flag_is_explicit(matches, "weight_decay") {
        if let Some(value) = run.and_then(|section| section.weight_decay) {
            arguments.weight_decay = value;
        }
    }
    if !flag_is_explicit(matches, "maximum_gradient_norm") {
        if let Some(value) = run.and_then(|section| section.maximum_gradient_norm) {
            arguments.maximum_gradient_norm = value;
        }
    }
    if !flag_is_explicit(matches, "max_sequence_length") {
        if let Some(value) = run.and_then(|section| section.max_sequence_length) {
            arguments.max_sequence_length = value;
        }
    }
    if !flag_is_explicit(matches, "minimum_improvement") {
        if let Some(value) = run.and_then(|section| section.minimum_improvement) {
            arguments.minimum_improvement = value;
        }
    }
    if !flag_is_explicit(matches, "early_stop_patience") {
        if let Some(value) = run.and_then(|section| section.early_stop_patience) {
            arguments.early_stop_patience = value;
        }
    }

    arguments.geometry = resolve_geometry(matches, configuration.model())?;
    Ok(arguments)
}

/// Resolves the explicit geometry with CLI over TOML over default precedence.
pub fn resolve_geometry(
    matches: &ArgMatches,
    model: Option<&ModelSection>,
) -> anyhow::Result<ModelGeometryArguments> {
    let mut geometry =
        <ModelGeometryArguments as clap::FromArgMatches>::from_arg_matches(matches).map_err(
            |error| {
                crate::error::TrainerError::Configuration(format!(
                    "failed to rebuild the geometry arguments from matches: {error}"
                ))
            },
        )?;
    let Some(model) = model else {
        return Ok(geometry);
    };

    if !flag_is_explicit(matches, "architecture") {
        if let Some(value) = model.architecture.as_ref() {
            geometry.architecture = Some(value.clone());
        }
    }
    if !flag_is_explicit(matches, "hidden_size") {
        if let Some(value) = model.hidden_size {
            geometry.hidden_size = Some(value);
        }
    }
    if !flag_is_explicit(matches, "intermediate_size") {
        if let Some(value) = model.intermediate_size {
            geometry.intermediate_size = Some(value);
        }
    }
    if !flag_is_explicit(matches, "num_hidden_layers") {
        if let Some(value) = model.num_hidden_layers {
            geometry.num_hidden_layers = Some(value);
        }
    }
    if !flag_is_explicit(matches, "num_attention_heads") {
        if let Some(value) = model.num_attention_heads {
            geometry.num_attention_heads = Some(value);
        }
    }
    if !flag_is_explicit(matches, "num_key_value_heads") {
        if let Some(value) = model.num_key_value_heads {
            geometry.num_key_value_heads = Some(value);
        }
    }
    if !flag_is_explicit(matches, "vocab_size") {
        if let Some(value) = model.vocab_size {
            geometry.vocab_size = Some(value);
        }
    }
    if !flag_is_explicit(matches, "max_position_embeddings") {
        if let Some(value) = model.max_position_embeddings {
            geometry.max_position_embeddings = Some(value);
        }
    }
    if !flag_is_explicit(matches, "rope_theta") {
        if let Some(value) = model.rope_theta {
            geometry.rope_theta = Some(value);
        }
    }
    if !flag_is_explicit(matches, "rms_norm_eps") {
        if let Some(value) = model.rms_norm_eps {
            geometry.rms_norm_eps = Some(value);
        }
    }
    if !flag_is_explicit(matches, "tie_word_embeddings") {
        if let Some(value) = model.tie_word_embeddings {
            geometry.tie_word_embeddings = Some(value);
        }
    }
    Ok(geometry)
}

/// Maps the TOML `[initialization]` section over the initializer defaults.
///
/// Absent keys keep the [`InitializationConfiguration::default`] value, so a
/// missing section is equivalent to the default initializer.
pub fn resolve_initialization_configuration(
    section: Option<&InitializationSection>,
) -> InitializationConfiguration {
    let mut configuration = InitializationConfiguration::default();
    let Some(section) = section else {
        return configuration;
    };
    if let Some(value) = section.initializer_range {
        configuration.initializer_range = value;
    }
    if let Some(value) = section.embedding_std {
        configuration.embedding_std = value;
    }
    if let Some(value) = section.norm_weight {
        configuration.norm_weight = value;
    }
    if let Some(value) = section.bias_value {
        configuration.bias_value = value;
    }
    configuration
}

/// Parses a TOML `device` value into the CLI enum, mirroring clap's choices.
fn parse_trainer_device(value: &str) -> anyhow::Result<crate::cli::TrainerDevice> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(crate::cli::TrainerDevice::Auto),
        "cpu" => Ok(crate::cli::TrainerDevice::Cpu),
        "cuda" => Ok(crate::cli::TrainerDevice::Cuda),
        other => Err(crate::error::TrainerError::Configuration(format!(
            "invalid device '{other}': expected 'auto', 'cpu' or 'cuda'"
        ))
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::TrainerArguments;
    use clap::CommandFactory;

    fn train_matches(arguments: &[&str]) -> anyhow::Result<ArgMatches> {
        let command = TrainerArguments::command();
        let matches = command.try_get_matches_from(arguments)?;
        let train = matches
            .subcommand_matches("train")
            .ok_or_else(|| anyhow::anyhow!("expected the train subcommand"))?;
        Ok(train.clone())
    }

    fn configuration(contents: &str) -> anyhow::Result<ConfigurationFile> {
        crate::configuration_file::parse_configuration_file(contents)
    }

    #[test]
    fn explicit_cli_epochs_wins_over_the_toml() -> anyhow::Result<()> {
        let matches = train_matches(&[
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--epochs",
            "7",
        ])?;
        let configuration = configuration("[run]\nepochs = 5\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert_eq!(resolved.epochs, 7);
        Ok(())
    }

    #[test]
    fn toml_epochs_wins_over_the_default() -> anyhow::Result<()> {
        let matches = train_matches(&["typed-lm-trainer", "train", "--dataset", "data"])?;
        let configuration = configuration("[run]\nepochs = 5\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert_eq!(resolved.epochs, 5);
        Ok(())
    }

    #[test]
    fn epoch_default_holds_when_neither_is_present() -> anyhow::Result<()> {
        let matches = train_matches(&["typed-lm-trainer", "train", "--dataset", "data"])?;
        let configuration = configuration("[run]\nseed = 1\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert_eq!(resolved.epochs, 3);
        Ok(())
    }

    #[test]
    fn explicit_cli_method_wins_over_the_toml() -> anyhow::Result<()> {
        let matches = train_matches(&[
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--method",
            "qlora",
        ])?;
        let configuration = configuration("[run]\nmethod = \"full\"\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert_eq!(resolved.method, "qlora");
        Ok(())
    }

    #[test]
    fn toml_method_wins_over_the_default() -> anyhow::Result<()> {
        let matches = train_matches(&["typed-lm-trainer", "train", "--dataset", "data"])?;
        let configuration = configuration("[run]\nmethod = \"from-scratch\"\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert_eq!(resolved.method, "from-scratch");
        Ok(())
    }

    #[test]
    fn method_default_holds_when_neither_is_present() -> anyhow::Result<()> {
        let matches = train_matches(&["typed-lm-trainer", "train", "--dataset", "data"])?;
        let configuration = configuration("[run]\nseed = 1\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert_eq!(resolved.method, "lora");
        Ok(())
    }

    #[test]
    fn explicit_cli_learning_rate_wins_over_the_toml() -> anyhow::Result<()> {
        let matches = train_matches(&[
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--learning-rate",
            "0.5",
        ])?;
        let configuration = configuration("[run]\nlearning_rate = 0.01\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert!((resolved.learning_rate - 0.5).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn toml_learning_rate_wins_over_the_default() -> anyhow::Result<()> {
        let matches = train_matches(&["typed-lm-trainer", "train", "--dataset", "data"])?;
        let configuration = configuration("[run]\nlearning_rate = 0.01\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert!((resolved.learning_rate - 0.01).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn learning_rate_default_holds_when_neither_is_present() -> anyhow::Result<()> {
        let matches = train_matches(&["typed-lm-trainer", "train", "--dataset", "data"])?;
        let configuration = configuration("[run]\nseed = 1\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert!((resolved.learning_rate - 1e-4).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn explicit_cli_seed_wins_over_the_toml() -> anyhow::Result<()> {
        let matches = train_matches(&[
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--seed",
            "99",
        ])?;
        let configuration = configuration("[run]\nseed = 1\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert_eq!(resolved.seed, 99);
        Ok(())
    }

    #[test]
    fn toml_seed_wins_over_the_default() -> anyhow::Result<()> {
        let matches = train_matches(&["typed-lm-trainer", "train", "--dataset", "data"])?;
        let configuration = configuration("[run]\nseed = 1\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert_eq!(resolved.seed, 1);
        Ok(())
    }

    #[test]
    fn seed_default_holds_when_neither_is_present() -> anyhow::Result<()> {
        let matches = train_matches(&["typed-lm-trainer", "train", "--dataset", "data"])?;
        let configuration = configuration("[run]\nepochs = 1\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert_eq!(resolved.seed, 42);
        Ok(())
    }

    #[test]
    fn toml_dataset_and_output_directory_apply_when_absent_from_cli() -> anyhow::Result<()> {
        // `--dataset` is required by clap, so it is always explicit; only the
        // optional output directory can be sourced from the file.
        let matches = train_matches(&["typed-lm-trainer", "train", "--dataset", "cli.jsonl"])?;
        let configuration =
            configuration("[dataset]\npath = \"file.jsonl\"\n[run]\noutput_directory = \"out\"\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert_eq!(resolved.dataset.to_string_lossy(), "cli.jsonl");
        assert_eq!(resolved.output_directory.to_string_lossy(), "out");
        Ok(())
    }

    #[test]
    fn toml_device_is_parsed_and_overridden_by_the_cli() -> anyhow::Result<()> {
        let matches = train_matches(&["typed-lm-trainer", "train", "--dataset", "data"])?;
        let configuration = configuration("[run]\ndevice = \"cpu\"\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert_eq!(resolved.device, crate::cli::TrainerDevice::Cpu);

        let explicit = train_matches(&[
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--device",
            "cuda",
        ])?;
        let resolved_explicit = resolve_train_arguments(&explicit, &configuration)?;
        assert_eq!(resolved_explicit.device, crate::cli::TrainerDevice::Cuda);
        Ok(())
    }

    #[test]
    fn an_invalid_toml_device_is_an_error() -> anyhow::Result<()> {
        let matches = train_matches(&["typed-lm-trainer", "train", "--dataset", "data"])?;
        let configuration = configuration("[run]\ndevice = \"tpu\"\n")?;
        assert!(resolve_train_arguments(&matches, &configuration).is_err());
        Ok(())
    }

    #[test]
    fn toml_geometry_applies_and_cli_geometry_wins() -> anyhow::Result<()> {
        let matches = train_matches(&[
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--architecture",
            "llama",
        ])?;
        let configuration = configuration(
            "[model]\narchitecture = \"qwen3\"\nhidden_size = 128\nnum_attention_heads = 8\n",
        )?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        // The CLI architecture wins; the TOML geometry still fills the rest.
        assert_eq!(resolved.geometry.architecture.as_deref(), Some("llama"));
        assert_eq!(resolved.geometry.hidden_size, Some(128));
        assert_eq!(resolved.geometry.num_attention_heads, Some(8));
        Ok(())
    }

    #[test]
    fn toml_geometry_applies_when_the_cli_is_absent() -> anyhow::Result<()> {
        let matches = train_matches(&["typed-lm-trainer", "train", "--dataset", "data"])?;
        let configuration = configuration("[model]\narchitecture = \"gemma2\"\nhidden_size = 64\n")?;
        let resolved = resolve_train_arguments(&matches, &configuration)?;
        assert_eq!(resolved.geometry.architecture.as_deref(), Some("gemma2"));
        assert_eq!(resolved.geometry.hidden_size, Some(64));
        assert!(resolved.geometry.intermediate_size.is_none());
        Ok(())
    }

    #[test]
    fn initialization_defaults_hold_without_a_section() {
        let configuration = resolve_initialization_configuration(None);
        let default = InitializationConfiguration::default();
        assert!((configuration.initializer_range - default.initializer_range).abs() < 1e-12);
        assert!((configuration.embedding_std - default.embedding_std).abs() < 1e-12);
        assert!((configuration.norm_weight - default.norm_weight).abs() < 1e-12);
        assert!((configuration.bias_value - default.bias_value).abs() < 1e-12);
    }

    #[test]
    fn initialization_section_overrides_the_defaults() -> anyhow::Result<()> {
        let configuration =
            configuration("[initialization]\ninitializer_range = 0.05\nnorm_weight = 2.0\n")?;
        let resolved = resolve_initialization_configuration(configuration.initialization());
        assert!((resolved.initializer_range - 0.05).abs() < 1e-12);
        assert!((resolved.norm_weight - 2.0).abs() < 1e-12);
        // The untouched keys keep their defaults.
        assert!((resolved.embedding_std - 0.02).abs() < 1e-12);
        assert!((resolved.bias_value - 0.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn flag_explicitness_is_reported() -> anyhow::Result<()> {
        let matches = train_matches(&[
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--epochs",
            "2",
        ])?;
        assert!(flag_is_explicit(&matches, "epochs"));
        assert!(!flag_is_explicit(&matches, "seed"));
        Ok(())
    }
}
