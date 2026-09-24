use actix_web::{web, HttpResponse};

use crate::api::dtos::SystemOneRequest;
use crate::api::error::EvaluationError;

use crate::api::state::SharedState;

#[tracing::instrument(skip(shared_state, request), fields(model = %request.model, questions = request.questions.len()))]
pub(crate) async fn handle_systemone(
    shared_state: web::Data<SharedState>,
    request: web::Json<SystemOneRequest>,
) -> Result<HttpResponse, EvaluationError> {
    let question_count = request.questions.len();
    tracing::info!(
        "received systemone request for model '{}' with {question_count} question(s)",
        request.model
    );
    request
        .validate()
        .map_err(EvaluationError::invalid_request)?;
    tracing::debug!("request validation passed; dispatching blocking inference");
    let owned_request = request.into_inner();
    let evaluator_handle = shared_state.evaluator.clone();
    // Candle inference blocks the thread for seconds; run it on the
    // dedicated blocking pool so Actix workers keep accepting requests.
    let response = web::block(move || evaluator_handle.evaluate(&owned_request))
        .await
        .map_err(|blocking_error| {
            tracing::error!("blocking inference pool failed: {blocking_error}");
            EvaluationError::inference(blocking_error.to_string())
        })??;
    tracing::info!(
        "systemone request for model '{}' answered with {} answer(s)",
        response.model,
        response.answers.len()
    );
    Ok(HttpResponse::Ok().json(response))
}
