//! TOML configuration file schema for the trainer.
//!
//! The trainer accepts an optional TOML file that can express every CLI
//! parameter, so a long invocation can be versioned next to the dataset. Every
//! field is optional, so a partial file is valid and only the keys that are
//! present override the CLI defaults; explicit CLI flags take precedence at
//! merge time.
//!
//! Sections mirror the runtime concerns: `[run]` holds the training
//! hyper-parameters, `[model]` the architecture geometry, `[initialization]`
//! the from-scratch weight initializers, `[dataset]` the dataset location and
//! `[tokenizer]` the tokenizer artifact. Unknown keys are rejected so a typo
//! surfaces as a configuration error instead of being silently ignored.

use std::path::Path;

use serde::Deserialize;

use crate::error::TrainerError;

/// Run section: training method, hyper-parameters and execution device.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunSection {
    /// Training method: `lora`, `qlora`, `full` or `from-scratch`.
    pub method: Option<String>,
    /// Deterministic initialization seed.
    pub seed: Option<u64>,
    /// Directory that receives the trained adapter and metadata.
    pub output_directory: Option<String>,
    /// Model reference: a Hugging Face repository id or a local path.
    pub model_id: Option<String>,
    /// Quantization target: `none`, `fp8` or `fp4`.
    pub quantization: Option<String>,
    /// When quantization happens: `post-training` or `training`.
    pub quantization_mode: Option<String>,
    /// Execution device: `auto`, `cpu` or `cuda`.
    pub device: Option<String>,
    /// Global gradient-norm clipping threshold.
    pub maximum_gradient_norm: Option<f64>,
    /// Minimum loss improvement that resets the early-stop patience.
    pub minimum_improvement: Option<f32>,
    /// Epochs without improvement before stopping early (`0` disables it).
    pub early_stop_patience: Option<usize>,
    /// Maximum prompt length; longer items are skipped.
    pub max_sequence_length: Option<usize>,
    /// Linear warmup steps before the cosine decay.
    pub warmup_steps: Option<usize>,
    /// AdamW weight decay.
    pub weight_decay: Option<f64>,
    /// Peak learning rate.
    pub learning_rate: Option<f64>,
    /// Batch size per step.
    pub batch_size: Option<usize>,
    /// Micro-batches accumulated before an optimizer step.
    pub gradient_accumulation_steps: Option<usize>,
    /// Number of epochs.
    pub epochs: Option<usize>,
    /// LoRA rank (`r`).
    pub lora_rank: Option<usize>,
    /// LoRA scaling numerator (`alpha`); effective scale is `alpha / rank`.
    pub lora_alpha: Option<f64>,
    /// LoRA dropout applied to the adapter activations.
    pub lora_dropout: Option<f32>,
}

/// Model section: explicit architecture geometry for checkpoint-free training.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSection {
    /// Architecture family (`llama`, `qwen2`, `qwen3`, `mistral`, `gemma`, ...).
    pub architecture: Option<String>,
    /// Vocabulary size.
    pub vocab_size: Option<usize>,
    /// Hidden dimension of the model.
    pub hidden_size: Option<usize>,
    /// Feed-forward intermediate dimension.
    pub intermediate_size: Option<usize>,
    /// Number of transformer blocks.
    pub num_hidden_layers: Option<usize>,
    /// Number of query attention heads.
    pub num_attention_heads: Option<usize>,
    /// Number of key/value attention heads (grouped-query attention).
    pub num_key_value_heads: Option<usize>,
    /// Maximum supported sequence length.
    pub max_position_embeddings: Option<usize>,
    /// Rotary embedding base frequency.
    pub rope_theta: Option<f32>,
    /// Root-mean-square normalization epsilon.
    pub rms_norm_eps: Option<f64>,
    /// Whether the input and output embeddings share weights.
    pub tie_word_embeddings: Option<bool>,
}

/// Initialization section: weight initializers for from-scratch training.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InitializationSection {
    /// Standard deviation of the general weight initializer.
    pub initializer_range: Option<f64>,
    /// Standard deviation of the embedding initializer.
    pub embedding_std: Option<f64>,
    /// Value assigned to normalization weights.
    pub norm_weight: Option<f64>,
    /// Value assigned to bias tensors.
    pub bias_value: Option<f64>,
}

/// Dataset section: where the training data comes from.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetSection {
    /// Dataset path: a `.jsonl`/`.json` file or a directory (auto-detected).
    pub path: Option<String>,
}

/// Tokenizer section: where the tokenizer artifact comes from.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerSection {
    /// Path to a `tokenizer.json` artifact.
    pub file: Option<String>,
}

/// Root of the TOML configuration file.
///
/// Every section is optional so a partial file stays valid; omitted sections
/// leave the corresponding CLI defaults untouched.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigurationFile {
    /// Training hyper-parameters and execution settings.
    pub run: Option<RunSection>,
    /// Model geometry.
    pub model: Option<ModelSection>,
    /// From-scratch initialization parameters.
    pub initialization: Option<InitializationSection>,
    /// Dataset location.
    pub dataset: Option<DatasetSection>,
    /// Tokenizer location.
    pub tokenizer: Option<TokenizerSection>,
}

impl ConfigurationFile {
    /// Borrows the `[run]` section when present.
    pub fn run(&self) -> Option<&RunSection> {
        self.run.as_ref()
    }

    /// Borrows the `[model]` section when present.
    pub fn model(&self) -> Option<&ModelSection> {
        self.model.as_ref()
    }

    /// Borrows the `[initialization]` section when present.
    pub fn initialization(&self) -> Option<&InitializationSection> {
        self.initialization.as_ref()
    }

    /// Borrows the `[dataset]` section when present.
    pub fn dataset(&self) -> Option<&DatasetSection> {
        self.dataset.as_ref()
    }

    /// Borrows the `[tokenizer]` section when present.
    pub fn tokenizer(&self) -> Option<&TokenizerSection> {
        self.tokenizer.as_ref()
    }

    /// Whether the file declared no section at all.
    pub fn is_empty(&self) -> bool {
        self.run.is_none()
            && self.model.is_none()
            && self.initialization.is_none()
            && self.dataset.is_none()
            && self.tokenizer.is_none()
    }
}

/// Reads and parses a TOML configuration file from disk.
///
/// I/O failures surface as [`TrainerError::Io`]; parse failures (malformed
/// TOML, unknown key or wrong type) surface as
/// [`TrainerError::ConfigurationFile`] carrying the path and the reason.
pub fn load_configuration_file(path: &Path) -> anyhow::Result<ConfigurationFile> {
    let contents = std::fs::read_to_string(path).map_err(TrainerError::from)?;
    parse_configuration_with_path(&contents, &path.display().to_string())
}

/// Parses a TOML configuration from a string in memory.
///
/// Used by tests and by callers that already hold the contents; the error
/// message names the file as `"<inline>"`.
pub fn parse_configuration_file(contents: &str) -> anyhow::Result<ConfigurationFile> {
    parse_configuration_with_path(contents, "<inline>")
}

fn parse_configuration_with_path(
    contents: &str,
    path: &str,
) -> anyhow::Result<ConfigurationFile> {
    toml::from_str::<ConfigurationFile>(contents).map_err(|error| {
        TrainerError::configuration_file(path, error.to_string()).into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPLETE_FILE: &str = r#"
[run]
method = "from-scratch"
seed = 42
output_directory = "output/scratch"
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
quantization = "none"
quantization_mode = "post-training"
device = "auto"
maximum_gradient_norm = 1.0
minimum_improvement = 0.0
early_stop_patience = 0
max_sequence_length = 1024
warmup_steps = 10
weight_decay = 0.0
learning_rate = 1e-4
batch_size = 4
gradient_accumulation_steps = 1
epochs = 3
lora_rank = 16
lora_alpha = 32.0
lora_dropout = 0.0

[model]
architecture = "qwen3"
vocab_size = 151936
hidden_size = 1024
intermediate_size = 4096
num_hidden_layers = 16
num_attention_heads = 16
num_key_value_heads = 4
max_position_embeddings = 4096
rope_theta = 1000000.0
rms_norm_eps = 1e-6
tie_word_embeddings = true

[initialization]
initializer_range = 0.02
embedding_std = 0.02
norm_weight = 1.0
bias_value = 0.0

[dataset]
path = "resources/dataset.jsonl"

[tokenizer]
file = "tokenizer.json"
"#;

    #[test]
    fn a_complete_file_parses_every_section() -> anyhow::Result<()> {
        let configuration = parse_configuration_file(COMPLETE_FILE)?;

        let run = configuration
            .run()
            .ok_or_else(|| anyhow::anyhow!("expected the run section"))?;
        assert_eq!(run.method.as_deref(), Some("from-scratch"));
        assert_eq!(run.seed, Some(42));
        assert_eq!(run.lora_rank, Some(16));
        assert_eq!(run.learning_rate, Some(1e-4));

        let model = configuration
            .model()
            .ok_or_else(|| anyhow::anyhow!("expected the model section"))?;
        assert_eq!(model.architecture.as_deref(), Some("qwen3"));
        assert_eq!(model.hidden_size, Some(1024));
        assert_eq!(model.tie_word_embeddings, Some(true));

        let initialization = configuration
            .initialization()
            .ok_or_else(|| anyhow::anyhow!("expected the initialization section"))?;
        assert_eq!(initialization.initializer_range, Some(0.02));

        let dataset = configuration
            .dataset()
            .ok_or_else(|| anyhow::anyhow!("expected the dataset section"))?;
        assert_eq!(dataset.path.as_deref(), Some("resources/dataset.jsonl"));

        let tokenizer = configuration
            .tokenizer()
            .ok_or_else(|| anyhow::anyhow!("expected the tokenizer section"))?;
        assert_eq!(tokenizer.file.as_deref(), Some("tokenizer.json"));

        assert!(!configuration.is_empty());
        Ok(())
    }

    #[test]
    fn a_partial_file_is_valid() -> anyhow::Result<()> {
        let configuration = parse_configuration_file("[run]\nseed = 7\nepochs = 5\n")?;

        let run = configuration
            .run()
            .ok_or_else(|| anyhow::anyhow!("expected the run section"))?;
        assert_eq!(run.seed, Some(7));
        assert_eq!(run.epochs, Some(5));
        assert!(run.method.is_none());
        assert!(configuration.model().is_none());
        assert!(configuration.initialization().is_none());
        assert!(configuration.dataset().is_none());
        assert!(configuration.tokenizer().is_none());
        assert!(!configuration.is_empty());
        Ok(())
    }

    #[test]
    fn an_empty_file_is_valid_and_empty() -> anyhow::Result<()> {
        let configuration = parse_configuration_file("")?;
        assert!(configuration.is_empty());
        assert!(configuration.run().is_none());
        assert!(configuration.model().is_none());
        assert!(configuration.initialization().is_none());
        assert!(configuration.dataset().is_none());
        assert!(configuration.tokenizer().is_none());
        Ok(())
    }

    #[test]
    fn an_unknown_key_is_rejected() -> anyhow::Result<()> {
        let result = parse_configuration_file("[run]\nunknown_key = 1\n");
        assert!(result.is_err());
        let Err(error) = result else {
            return Ok(());
        };
        let message = error.to_string();
        assert!(message.contains("unknown_key"), "message was: {message}");
        Ok(())
    }

    #[test]
    fn a_wrong_type_is_rejected() -> anyhow::Result<()> {
        let result = parse_configuration_file("[run]\nepochs = \"three\"\n");
        assert!(result.is_err());
        let Err(error) = result else {
            return Ok(());
        };
        let message = error.to_string();
        assert!(message.contains("epochs"), "message was: {message}");
        assert!(message.contains("<inline>"), "message was: {message}");
        Ok(())
    }

    #[test]
    fn malformed_toml_is_rejected() -> anyhow::Result<()> {
        let result = parse_configuration_file("[run\n");
        assert!(result.is_err());
        let Err(error) = result else {
            return Ok(());
        };
        assert!(error.to_string().contains("<inline>"));
        Ok(())
    }

    #[test]
    fn the_error_names_the_file_path() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("broken.toml");
        std::fs::write(&path, "[run\n")?;

        let result = load_configuration_file(&path);
        assert!(result.is_err());
        let Err(error) = result else {
            return Ok(());
        };
        let rendered = error.to_string();
        assert!(
            rendered.contains(&path.display().to_string()),
            "message was: {rendered}"
        );
        Ok(())
    }

    #[test]
    fn a_missing_file_is_an_io_error() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("absent.toml");

        let result = load_configuration_file(&path);
        assert!(result.is_err());
        let Err(error) = result else {
            return Ok(());
        };
        let Some(trainer_error) = error.downcast_ref::<TrainerError>() else {
            return Err(anyhow::anyhow!("expected a TrainerError, got: {error}"));
        };
        assert!(matches!(trainer_error, TrainerError::Io(_)));
        Ok(())
    }
}
