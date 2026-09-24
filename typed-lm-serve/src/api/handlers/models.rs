use actix_web::{web, HttpResponse};

use crate::api::dtos::{ModelEntry, ModelsResponse, OpenAIModelEntry};

use crate::api::state::SharedState;

const JEV_LATEST_MODEL_ALIAS: &str = "jev-latest";
const MODEL_OWNER: &str = "typed-lm";

fn model_identifiers(served_model_name: &str) -> Vec<String> {
    let mut identifiers = vec![served_model_name.to_string()];
    if served_model_name != JEV_LATEST_MODEL_ALIAS {
        identifiers.push(JEV_LATEST_MODEL_ALIAS.to_string());
    }
    identifiers
}

pub(crate) async fn handle_models(shared_state: web::Data<SharedState>) -> HttpResponse {
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
