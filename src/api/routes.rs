use actix_web::web;

use crate::api::error::EvaluationError;

use super::handlers::{
    handle_choice, handle_health, handle_liveness, handle_models, handle_noul, handle_score,
    handle_systemone,
};

pub fn configure(application_configuration: &mut web::ServiceConfig) {
    application_configuration
        .app_data(web::JsonConfig::default().error_handler(|_, _| {
            EvaluationError::invalid_request("request body failed validation").into()
        }))
        .service(
            web::scope("/v1")
                .route("/systemone", web::post().to(handle_systemone))
                .route("/noul", web::post().to(handle_noul))
                .route("/choice", web::post().to(handle_choice))
                .route("/score", web::post().to(handle_score))
                .route("/models", web::get().to(handle_models)),
        )
        .route("/health", web::get().to(handle_health))
        .route("/health/live", web::get().to(handle_liveness));
}
