use std::sync::Arc;

use crate::evaluator::Evaluator;

pub struct SharedState {
    pub evaluator: Arc<dyn Evaluator + Send + Sync>,
    pub served_model_name: String,
    pub context_name: String,
    pub startup_seconds: f64,
}
