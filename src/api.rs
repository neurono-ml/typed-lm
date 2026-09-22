use actix_web::{web, HttpResponse};
use std::sync::Arc;

use crate::evaluation_error::EvaluationError;
use crate::evaluator::Evaluator;
use crate::jev::{
    HealthResponse, LiveResponse, ModelEntry, ModelsResponse, OpenAIModelEntry, SystemOneRequest,
};

const JEV_LATEST_MODEL_ALIAS: &str = "jev-latest";
const MODEL_OWNER: &str = "manaca";

pub struct SharedState {
    pub evaluator: Arc<dyn Evaluator + Send + Sync>,
    pub served_model_name: String,
    pub context_name: String,
    pub startup_seconds: f64,
}

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

async fn handle_systemone(
    shared_state: web::Data<SharedState>,
    request: web::Json<SystemOneRequest>,
) -> Result<HttpResponse, EvaluationError> {
    request
        .validate()
        .map_err(EvaluationError::invalid_request)?;
    let response = shared_state.evaluator.evaluate(&request)?;
    Ok(HttpResponse::Ok().json(response))
}

fn model_identifiers(served_model_name: &str) -> Vec<String> {
    let mut identifiers = vec![served_model_name.to_string()];
    if served_model_name != JEV_LATEST_MODEL_ALIAS {
        identifiers.push(JEV_LATEST_MODEL_ALIAS.to_string());
    }
    identifiers
}

async fn handle_models(shared_state: web::Data<SharedState>) -> HttpResponse {
    let identifiers = model_identifiers(&shared_state.served_model_name);
    let data = identifiers
        .iter()
        .map(|identifier| OpenAIModelEntry {
            id: identifier.clone(),
            object: "model".to_string(),
            owned_by: MODEL_OWNER.to_string(),
        })
        .collect();
    let models = identifiers
        .iter()
        .map(|identifier| ModelEntry {
            name: identifier.clone(),
            description: format!(
                "Jev-compatible model served from context '{}'",
                shared_state.context_name
            ),
            release_date: "unknown".to_string(),
        })
        .collect();
    HttpResponse::Ok().json(ModelsResponse {
        object: "list".to_string(),
        data,
        models,
    })
}

async fn handle_health(shared_state: web::Data<SharedState>) -> HttpResponse {
    HttpResponse::Ok().json(HealthResponse {
        status: "ok".to_string(),
        startup_seconds: Some(shared_state.startup_seconds),
    })
}

async fn handle_liveness() -> HttpResponse {
    HttpResponse::Ok().json(LiveResponse {
        status: "ok".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::MockEvaluator;
    use actix_web::{http::StatusCode, test as actix_test, App};

    const SERVED_MODEL_NAME: &str = "manaca-test-model";
    const CONTEXT_NAME: &str = "test-context";

    fn shared_state_with_mock() -> web::Data<SharedState> {
        web::Data::new(SharedState {
            evaluator: Arc::new(MockEvaluator::new(SERVED_MODEL_NAME.to_string())),
            served_model_name: SERVED_MODEL_NAME.to_string(),
            context_name: CONTEXT_NAME.to_string(),
            startup_seconds: 1.5,
        })
    }

    fn valid_systemone_payload() -> serde_json::Value {
        serde_json::json!({
            "model": SERVED_MODEL_NAME,
            "state": "customer was charged twice",
            "questions": {
                "refund": {"type": "noul", "instructions": "Should the customer be refunded?"},
                "department": {
                    "type": "choice",
                    "instructions": "Which department should handle this?",
                    "criteria": {"billing": "Payment issues", "technical": "Bug reports"}
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is this?",
                    "criteria": ["Routine", "Urgent", "Emergency"]
                }
            }
        })
    }

    #[test]
    fn shared_state_holds_send_sync_evaluator() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SharedState>();
        let state = shared_state_with_mock();
        let _: Arc<dyn Evaluator + Send + Sync> = state.evaluator.clone();
    }

    #[actix_web::test]
    async fn systemone_answers_every_question_type() {
        let application = actix_test::init_service(
            App::new()
                .app_data(shared_state_with_mock())
                .configure(configure),
        )
        .await;
        let request = actix_test::TestRequest::post()
            .uri("/v1/systemone")
            .set_json(valid_systemone_payload())
            .to_request();
        let response = actix_test::call_service(&application, request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value = actix_test::read_body_json(response).await;
        assert_eq!(body["model"], SERVED_MODEL_NAME);
        assert_eq!(body["answers"]["refund"]["type"], "noul");
        assert_eq!(body["answers"]["department"]["type"], "choice");
        assert_eq!(body["answers"]["urgency"]["type"], "score");
    }

    #[actix_web::test]
    async fn models_lists_served_model_with_alias() {
        let application = actix_test::init_service(
            App::new()
                .app_data(shared_state_with_mock())
                .configure(configure),
        )
        .await;
        let request = actix_test::TestRequest::get()
            .uri("/v1/models")
            .to_request();
        let response = actix_test::call_service(&application, request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value = actix_test::read_body_json(response).await;
        assert_eq!(body["object"], "list");
        let identifiers: Vec<&str> = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["id"].as_str().unwrap())
            .collect();
        assert!(identifiers.contains(&SERVED_MODEL_NAME));
        assert!(identifiers.contains(&"jev-latest"));
        let names: Vec<&str> = body["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&SERVED_MODEL_NAME));
    }

    #[actix_web::test]
    async fn systemone_rejects_empty_questions_with_unprocessable_entity() {
        let application = actix_test::init_service(
            App::new()
                .app_data(shared_state_with_mock())
                .configure(configure),
        )
        .await;
        let payload = serde_json::json!({
            "model": SERVED_MODEL_NAME,
            "state": "customer was charged twice",
            "questions": {}
        });
        let request = actix_test::TestRequest::post()
            .uri("/v1/systemone")
            .set_json(payload)
            .to_request();
        let response = actix_test::call_service(&application, request).await;
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body: serde_json::Value = actix_test::read_body_json(response).await;
        assert!(body["error"]["message"].is_string());
    }

    #[actix_web::test]
    async fn systemone_rejects_unknown_question_type_with_unprocessable_entity() {
        let application = actix_test::init_service(
            App::new()
                .app_data(shared_state_with_mock())
                .configure(configure),
        )
        .await;
        let payload = serde_json::json!({
            "model": SERVED_MODEL_NAME,
            "state": "customer was charged twice",
            "questions": {"mystery": {"type": "unknown"}}
        });
        let request = actix_test::TestRequest::post()
            .uri("/v1/systemone")
            .set_json(payload)
            .to_request();
        let response = actix_test::call_service(&application, request).await;
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body: serde_json::Value = actix_test::read_body_json(response).await;
        assert!(body["error"]["message"].is_string());
    }

    #[actix_web::test]
    async fn systemone_maps_inference_failure_to_internal_server_error() {
        struct FailingEvaluator;

        impl Evaluator for FailingEvaluator {
            fn evaluate(
                &self,
                _request: &SystemOneRequest,
            ) -> Result<crate::jev::SystemOneResponse, EvaluationError> {
                Err(EvaluationError::inference("simulated forward pass failure"))
            }
        }

        let failing_state = web::Data::new(SharedState {
            evaluator: Arc::new(FailingEvaluator),
            served_model_name: SERVED_MODEL_NAME.to_string(),
            context_name: CONTEXT_NAME.to_string(),
            startup_seconds: 1.5,
        });
        let application =
            actix_test::init_service(App::new().app_data(failing_state).configure(configure)).await;
        let request = actix_test::TestRequest::post()
            .uri("/v1/systemone")
            .set_json(valid_systemone_payload())
            .to_request();
        let response = actix_test::call_service(&application, request).await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body: serde_json::Value = actix_test::read_body_json(response).await;
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("simulated forward pass failure"));
    }

    #[actix_web::test]
    async fn health_reports_startup_time() {
        let application = actix_test::init_service(
            App::new()
                .app_data(shared_state_with_mock())
                .configure(configure),
        )
        .await;
        let request = actix_test::TestRequest::get().uri("/health").to_request();
        let response = actix_test::call_service(&application, request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value = actix_test::read_body_json(response).await;
        assert_eq!(body["status"], "ok");
        assert_eq!(body["startup_seconds"], 1.5);
    }

    #[actix_web::test]
    async fn liveness_reports_ok() {
        let application = actix_test::init_service(
            App::new()
                .app_data(shared_state_with_mock())
                .configure(configure),
        )
        .await;
        let request = actix_test::TestRequest::get()
            .uri("/health/live")
            .to_request();
        let response = actix_test::call_service(&application, request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value = actix_test::read_body_json(response).await;
        assert_eq!(body["status"], "ok");
    }
}
