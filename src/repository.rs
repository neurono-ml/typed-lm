use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use std::path::PathBuf;

/// Locates the model files in the local cache (downloading them if needed).
///
/// Authentication: if the model is gated, export one of
/// `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` with a token that has access.
pub struct ModelRepository {
    repo: ApiRepo,
}

impl ModelRepository {
    pub fn new(model_id: &str) -> anyhow::Result<Self> {
        let token = std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .ok();
        let api = ApiBuilder::new().with_token(token).build()?;
        Ok(Self {
            repo: api.model(model_id.to_string()),
        })
    }

    pub fn files(&self) -> anyhow::Result<ModelFiles> {
        let get = |name: &str| {
            self.repo.get(name).map_err(|e| {
                anyhow::anyhow!(
                    "failed to download '{name}': {e}. \
                     If the model is gated, set HF_TOKEN (or pass --model-id of a public model)."
                )
            })
        };
        Ok(ModelFiles {
            tokenizer: get("tokenizer.json")?,
            config: get("config.json")?,
            weights: get("model.safetensors")?,
        })
    }
}

/// Accessory struct: groups the three paths of the same repository.
/// (Legitimate exception to the "one struct per module" rule: it belongs to `ModelRepository`.)
pub struct ModelFiles {
    pub tokenizer: PathBuf,
    pub config: PathBuf,
    pub weights: PathBuf,
}
