use super::{configure, SharedState};
use crate::evaluation_error::EvaluationError;
use crate::evaluator::{Evaluator, MockEvaluator};
use crate::jev::SystemOneRequest;
use actix_web::{http::StatusCode, test as actix_test, web, App};
use std::sync::Arc;

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

/// Live integration test against the real weights and `resources/memory.md`.
///
/// Ignored by default so CI never downloads model weights; run explicitly with
/// `cargo test -- --ignored`. It validates context-anchored answers: a
/// double charge 60 days old is still refundable (fact 4 overrides the
/// 30-day window) and routes to the billing department.
#[actix_web::test]
#[ignore]
async fn live_candle_evaluator_answers_context_anchored_questions() {
    use crate::context::{ContextProvider, FileContextProvider};
    use crate::device::DeviceResolver;
    use crate::evaluator::CandleEvaluator;
    use crate::model::LanguageModel;
    use crate::repository::ModelRepository;
    use std::path::Path;

    let served_model_name = "jev-latest".to_string();
    let execution_device = DeviceResolver::resolve().unwrap();
    let model_files = ModelRepository::new("recogna-nlp/bode-1b-instruct")
        .unwrap()
        .files()
        .unwrap();
    let language_model = LanguageModel::load(
        &model_files.config,
        &model_files.weights,
        &model_files.tokenizer,
        &execution_device,
    )
    .unwrap();
    let context_provider = FileContextProvider::new(Path::new("resources/memory.md"));
    let loaded_context = context_provider.load().unwrap();
    assert!(loaded_context.contains("GreenLeaf"));
    let leaked_model: &'static LanguageModel = Box::leak(Box::new(language_model));
    let evaluator = CandleEvaluator::new(leaked_model, loaded_context, served_model_name.clone());
    let shared_state = web::Data::new(SharedState {
        evaluator: Arc::new(evaluator),
        served_model_name: served_model_name.clone(),
        context_name: context_provider.name(),
        startup_seconds: 0.0,
    });
    let application =
        actix_test::init_service(App::new().app_data(shared_state).configure(configure)).await;
    let payload = serde_json::json!({
        "model": served_model_name,
        "state": "The customer found a duplicate charge for order #4821 on their credit card. The duplicate charge is 60 days old and they request a full refund of the duplicate.",
        "questions": {
            "refund_eligible": {
                "type": "noul",
                "instructions": "The customer is eligible for a full refund under the store policy."
            },
            "responsible_department": {
                "type": "choice",
                "instructions": "Which department should handle this case?",
                "criteria": {
                    "billing": "Double charges and payment errors",
                    "logistics": "Damaged, lost, or late shipments",
                    "product_support": "Defective-item troubleshooting, replacements, and setup help"
                }
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this case?",
                "criteria": ["Routine", "Urgent", "Emergency"]
            }
        }
    });
    let request = actix_test::TestRequest::post()
        .uri("/v1/systemone")
        .set_json(payload)
        .to_request();
    let response = actix_test::call_service(&application, request).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = actix_test::read_body_json(response).await;
    let noul = body["answers"]["refund_eligible"]["noul"].as_f64().unwrap();
    assert!(
        noul > 0.5,
        "double charge is always refundable per memory fact 4, got noul={noul}"
    );
    let department = body["answers"]["responsible_department"]["choice"]
        .as_str()
        .unwrap();
    assert_eq!(department, "billing");
    let urgency = body["answers"]["urgency"]["score"].as_f64().unwrap();
    assert!(
        (0.0..=2.0).contains(&urgency),
        "urgency score must stay within the legend range, got {urgency}"
    );
}
