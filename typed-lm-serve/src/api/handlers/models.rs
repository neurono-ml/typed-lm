use actix_web::{web, HttpResponse};

use crate::api::dtos::{ModelEntry, ModelsResponse, OpenAIModelEntry};

use crate::api::state::SharedState;

const MODEL_OWNER: &str = "typed-lm";

pub(crate) async fn handle_models(shared_state: web::Data<SharedState>) -> HttpResponse {
    let identifiers = [shared_state.served_model_name.clone()];
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
                "typed-lm model served from context '{}'",
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
