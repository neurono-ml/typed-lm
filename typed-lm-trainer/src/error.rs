//! Typed trainer errors, mapped to actionable messages.
//!
//! Every fallible stage of the trainer (dataset, model loading, training and
//! quantization) contributes one variant so a failure carries enough context to
//! be actionable without a panic. `Other` keeps `anyhow` context for the paths
//! that already produce rich messages.

/// Errors surfaced by the trainer.
#[derive(Debug, thiserror::Error)]
pub enum TrainerError {
    /// I/O failure while reading or writing artifacts.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid configuration (bad flag combination, out-of-range value).
    #[error("invalid configuration: {0}")]
    Configuration(String),
    /// A configuration file could not be read, parsed or validated.
    #[error("configuration file error in '{path}': {message}")]
    ConfigurationFile {
        /// Path of the offending configuration file.
        path: String,
        /// Human-readable reason, ideally naming the key.
        message: String,
    },
    /// Failure while initializing from-scratch weights.
    #[error("initialization error: {0}")]
    Initialization(String),
    /// Failure while discovering, parsing or collating a dataset.
    #[error("dataset error: {0}")]
    Dataset(String),
    /// Failure while resolving or loading a model checkpoint.
    #[error("model error: {0}")]
    Model(String),
    /// Failure while building a precision policy.
    #[error("precision error: {0}")]
    Precision(String),
    /// Failure during the training loop (forward, backward or update).
    #[error("training error: {0}")]
    Training(String),
    /// Failure during quantization or artifact export.
    #[error("quantization error: {0}")]
    Quantization(String),
    /// Any other failure, wrapped for context.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl TrainerError {
    /// Build a configuration-file error from a path and a human-readable reason.
    pub fn configuration_file(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ConfigurationFile {
            path: path.into(),
            message: message.into(),
        }
    }

    /// Build an initialization error from a human-readable reason.
    pub fn initialization(message: impl Into<String>) -> Self {
        Self::Initialization(message.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_errors_are_wrapped_with_context() {
        let error = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let wrapped = TrainerError::from(error);
        assert!(wrapped.to_string().contains("io error"));
    }

    #[test]
    fn configuration_errors_carry_their_message() {
        let error = TrainerError::Configuration("rank must be positive".to_string());
        assert!(error.to_string().contains("rank must be positive"));
        assert!(error.to_string().contains("invalid configuration"));
    }

    #[test]
    fn dataset_model_training_and_quantization_are_distinct_variants() {
        let dataset = TrainerError::Dataset("bad line".to_string());
        let model = TrainerError::Model("missing weights".to_string());
        let training = TrainerError::Training("nan loss".to_string());
        let quantization = TrainerError::Quantization("fp4 scale missing".to_string());
        assert!(dataset.to_string().starts_with("dataset error"));
        assert!(model.to_string().starts_with("model error"));
        assert!(training.to_string().starts_with("training error"));
        assert!(quantization.to_string().starts_with("quantization error"));
    }

    #[test]
    fn anyhow_errors_convert_through_other() {
        let error: TrainerError = anyhow::anyhow!("wrapped").into();
        assert!(error.to_string().contains("wrapped"));
    }

    #[test]
    fn configuration_file_error_formats_path_and_message() -> anyhow::Result<()> {
        let error = TrainerError::ConfigurationFile {
            path: "config/train.toml".to_string(),
            message: "missing key: optimizer.learning_rate".to_string(),
        };
        let rendered = error.to_string();
        assert!(rendered.contains("config/train.toml"));
        assert!(rendered.contains("missing key: optimizer.learning_rate"));
        Ok(())
    }

    #[test]
    fn configuration_file_constructor_builds_the_variant() -> anyhow::Result<()> {
        let error = TrainerError::configuration_file("config/train.toml", "unknown key: batch");
        let rendered = error.to_string();
        assert!(matches!(error, TrainerError::ConfigurationFile { .. }));
        assert!(rendered.contains("config/train.toml"));
        assert!(rendered.contains("unknown key: batch"));
        Ok(())
    }

    #[test]
    fn initialization_constructor_carries_the_prefix() -> anyhow::Result<()> {
        let error = TrainerError::initialization("cannot allocate weight matrix");
        assert!(matches!(error, TrainerError::Initialization(_)));
        assert!(error.to_string().starts_with("initialization error"));
        assert!(error.to_string().contains("cannot allocate weight matrix"));
        Ok(())
    }
}
