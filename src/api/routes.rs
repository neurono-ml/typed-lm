use actix_web::web;

use crate::evaluation_error::EvaluationError;

use super::health::{handle_health, handle_liveness};
use super::models::handle_models;
use super::systemone::handle_systemone;

pub fn configure(application_configuration: &mut web::ServiceConfig) {
    application_configuration
        .app_data(web::JsonConfig::default().error_handler(|_, _| {
            EvaluationError::invalid_request("request body failed validation").into()
        }))
        .service(
            web::scope("/v1")
                .route("/systemone", web::post().to(handle_systemone))
                .route("/models", web::get().to(handle_models)),
        )
        .route("/health", web::get().to(handle_health))
        .route("/health/live", web::get().to(handle_liveness));
}
