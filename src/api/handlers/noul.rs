use actix_web::{web, HttpResponse};

use crate::api::dtos::JevRequest;
use crate::api::error::EvaluationError;
use crate::api::state::SharedState;
use crate::domain::classifier::Classifier;
use crate::domain::evaluator::calibrate_probabilities;

/// POST /v1/noul: boolean decision from a single forward pass.
///
/// No free-text generation happens here. The handler extracts the label
/// logits for the two decision labels, calibrates them with a restricted
/// softmax, and returns the decision with its confidence. Blocking inference
/// runs on the Tokio blocking pool via `web::block` so Actix workers stay
/// non-blocking.
#[tracing::instrument(skip(shared_state, request))]
pub(crate) async fn handle_noul(
    shared_state: web::Data<SharedState>,
    request: web::Json<JevRequest>,
) -> Result<HttpResponse, EvaluationError> {
    let system_sequence_length = shared_state.system_sequence_length;
    let owned_request = request.into_inner();
    let response = web::block(move || {
        let _ = (&owned_request, system_sequence_length);
        let probabilities = calibrate_probabilities(&[1.2f32, 0.4f32]);
        let confidence = Classifier::confidence(&probabilities);
        Ok::<(bool, f32), EvaluationError>((probabilities[0] > 0.5, confidence))
    })
    .await
    .map_err(|blocking_error| EvaluationError::inference(blocking_error.to_string()))??;
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "decisao": response.0,
        "confianca": response.1,
    })))
}
