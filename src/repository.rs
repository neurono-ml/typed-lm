use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use std::path::PathBuf;

/// Locates the model files in the local cache (downloading them if needed).
///
/// Authentication for gated models must be provided explicitly through
/// [`crate::cli::ServeArgs::hugging_face_token`] (flag `--hf-token`,
/// falling back to the `HF_TOKEN` environment variable via clap).
/// This type never reads environment variables itself.
pub struct ModelRepository {
    repo: ApiRepo,
    #[allow(dead_code)]
    model_identifier: String,
    #[allow(dead_code)]
    hugging_face_token: Option<String>,
}

impl ModelRepository {
    pub fn new(model_identifier: &str, hugging_face_token: Option<String>) -> anyhow::Result<Self> {
        let api = ApiBuilder::new()
            .with_token(hugging_face_token.clone())
            .build()?;
        Ok(Self {
            repo: api.model(model_identifier.to_string()),
            model_identifier: model_identifier.to_string(),
            hugging_face_token,
        })
    }

    /// Returns the model identifier this repository was built for.
    #[allow(dead_code)]
    pub fn model_identifier(&self) -> &str {
        &self.model_identifier
    }

    /// Returns the explicitly provided Hugging Face token, if any.
    #[allow(dead_code)]
    pub fn hugging_face_token(&self) -> Option<&str> {
        self.hugging_face_token.as_deref()
    }

    pub fn files(&self) -> anyhow::Result<ModelFiles> {
        let get = |name: &str| {
            self.repo.get(name).map_err(|e| {
                anyhow::anyhow!(
                    "failed to download '{name}': {e}. \
                     If the model is gated, pass --hf-token (or HF_TOKEN) of a token with access."
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repository_carries_explicit_token_without_network() {
        let repository = ModelRepository::new(
            "some-organization/some-model",
            Some("secret-token-value".to_string()),
        )
        .unwrap();
        assert_eq!(
            repository.model_identifier(),
            "some-organization/some-model"
        );
        assert_eq!(repository.hugging_face_token(), Some("secret-token-value"));
    }

    #[test]
    fn repository_supports_missing_token_for_public_models() {
        let repository = ModelRepository::new("some-organization/some-model", None).unwrap();
        assert_eq!(repository.hugging_face_token(), None);
    }
}
