//! Typed trainer errors, mapped to actionable messages.
//!
//! Wave 6 expands the variants; the skeleton provides the shared error enum so
//! `main.rs` can propagate failures with `?` and never panic.

/// Errors surfaced by the trainer.
#[derive(Debug, thiserror::Error)]
pub enum TrainerError {
    /// I/O failure while reading or writing artifacts.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Any other failure, wrapped for context.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
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
}
