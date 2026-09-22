use actix_web::{web, HttpResponse};

use crate::evaluation_error::EvaluationError;
use crate::jev::SystemOneRequest;

use super::SharedState;

pub(crate) async fn handle_systemone(
    shared_state: web::Data<SharedState>,
    request: web::Json<SystemOneRequest>,
) -> Result<HttpResponse, EvaluationError> {
    request
        .validate()
        .map_err(EvaluationError::invalid_request)?;
    let response = shared_state.evaluator.evaluate(&request)?;
    Ok(HttpResponse::Ok().json(response))
}
