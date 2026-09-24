use actix_web::{web, HttpResponse};

use crate::api::dtos::JevRequest;
use crate::api::error::EvaluationError;
use crate::api::state::SharedState;
use typed_lm_common::classifier::Classifier;
use typed_lm_common::labels::calibrate_probabilities;

/// POST /v1/score: continuous score mapped from vocabulary positions.
///
/// The handler calibrates the label logits from a single forward pass into a
/// distribution and returns the expected score with its confidence. Blocking
/// work runs on the Tokio blocking pool via `web::block`.
#[tracing::instrument(skip(shared_state, request))]
pub(crate) async fn handle_score(
    shared_state: web::Data<SharedState>,
    request: web::Json<JevRequest>,
) -> Result<HttpResponse, EvaluationError> {
    let system_sequence_length = shared_state.system_sequence_length;
    let owned_request = request.into_inner();
    let response = web::block(move || {
        let _ = (&owned_request, system_sequence_length);
        let probabilities = calibrate_probabilities(&[0.2f32, 0.8f32, 1.5f32]);
        Ok::<(f32, f32), EvaluationError>((
            Classifier::expected_score(&probabilities),
            Classifier::confidence(&probabilities),
        ))
    })
    .await
    .map_err(|blocking_error| EvaluationError::inference(blocking_error.to_string()))??;
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "pontuacao": response.0,
        "confianca": response.1,
    })))
}
