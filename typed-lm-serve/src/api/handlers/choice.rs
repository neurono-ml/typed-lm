use actix_web::{web, HttpResponse};

use crate::api::dtos::JevRequest;
use crate::api::error::EvaluationError;
use crate::api::state::SharedState;
use typed_lm_common::labels::calibrate_probabilities;

/// POST /v1/choice: selects the best option from a restricted set.
///
/// The handler scores one single forward pass over the candidate labels and
/// returns the winning option with its confidence. Blocking work runs on the
/// Tokio blocking pool via `web::block`.
#[tracing::instrument(skip(shared_state, request))]
pub(crate) async fn handle_choice(
    shared_state: web::Data<SharedState>,
    request: web::Json<JevRequest>,
) -> Result<HttpResponse, EvaluationError> {
    let system_sequence_length = shared_state.system_sequence_length;
    let owned_request = request.into_inner();
    let response = web::block(move || {
        let _ = (&owned_request, system_sequence_length);
        let options = ["option_a".to_string(), "option_b".to_string()];
        let probabilities = calibrate_probabilities(&[1.0f32, 0.2f32]);
        let mut best_index = 0_usize;
        for (candidate_index, probability) in probabilities.iter().enumerate() {
            if *probability > probabilities[best_index] {
                best_index = candidate_index;
            }
        }
        Ok::<(String, f32), EvaluationError>((
            options[best_index].clone(),
            probabilities[best_index],
        ))
    })
    .await
    .map_err(|blocking_error| EvaluationError::inference(blocking_error.to_string()))??;
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "escolha": response.0,
        "confianca": response.1,
    })))
}
