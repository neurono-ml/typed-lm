use actix_web::{web, HttpResponse};

use crate::jev::{HealthResponse, LiveResponse};

use super::SharedState;

pub(crate) async fn handle_health(shared_state: web::Data<SharedState>) -> HttpResponse {
    HttpResponse::Ok().json(HealthResponse {
        status: "ok".to_string(),
        startup_seconds: Some(shared_state.startup_seconds),
    })
}

pub(crate) async fn handle_liveness() -> HttpResponse {
    HttpResponse::Ok().json(LiveResponse {
        status: "ok".to_string(),
    })
}
