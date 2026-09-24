use std::sync::Arc;

use crate::domain::evaluator::Evaluator;

/// Shared application state cloned into every Actix worker via `web::Data`.
///
/// The evaluator handle is `Arc`-cloned per worker; the underlying language
/// model is loaded once at startup and never mutated per request. The
/// evaluator holds the prefilled system-context KV-cache; each request clones
/// it and runs isolated forward passes, discarding the clones afterwards.
/// `system_sequence_length` is the token length of that base cache.
pub struct ApplicationState {
    pub evaluator: Arc<dyn Evaluator + Send + Sync>,
    pub served_model_name: String,
    pub context_name: String,
    pub startup_seconds: f64,
    pub system_sequence_length: usize,
}

/// Backwards-compatible alias used across handlers and tests.
pub type SharedState = ApplicationState;
