use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use std::path::PathBuf;

/// Files a repository must expose to be servable by this crate.
///
/// This is the conservative default used by [`missing_model_files`]: a minimal
/// servable checkpoint. Detection-aware resolution (see
/// [`crate::checkpoint_resolver`]) decides the real required
/// set per format, so GGUF repositories without `config.json` are accepted
/// there while still being reported by this pure helper.
pub const REQUIRED_MODEL_FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

/// Returns the required model files absent from `available_file_names`.
///
/// Pure function: makes the "this repository is not a checkpoint" diagnosis
/// testable without network access.
pub fn missing_model_files(available_file_names: &[String]) -> Vec<&'static str> {
    REQUIRED_MODEL_FILES
        .iter()
        .copied()
        .filter(|required| {
            !available_file_names
                .iter()
                .any(|available| available == required)
        })
        .collect()
}

/// Locates the model files in the local cache (downloading them if needed).
///
/// Authentication for gated models must be provided explicitly through
/// [`typed-lm-serve ServeArguments::hugging_face_token`] (flag `--hf-token`,
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
        Self::with_revision(model_identifier, "main", hugging_face_token)
    }

    /// Builds a repository handle pinned to a specific revision (branch, tag or
    /// commit hash). Falls back to `main` when `revision` is empty.
    pub fn with_revision(
        model_identifier: &str,
        revision: &str,
        hugging_face_token: Option<String>,
    ) -> anyhow::Result<Self> {
        let api = ApiBuilder::new()
            .with_token(hugging_face_token.clone())
            .build()?;
        let revision = if revision.is_empty() {
            "main"
        } else {
            revision
        };
        let repo = api.repo(hf_hub::Repo::with_revision(
            model_identifier.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));
        Ok(Self {
            repo,
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

    /// Lists the file names exposed by the repository (a single Hub request).
    pub fn available_file_names(&self) -> anyhow::Result<Vec<String>> {
        let repository_info = self.repo.info().map_err(|error| {
            anyhow::anyhow!(
                "failed to inspect repository '{}': {error}. \
                 If the model is gated, pass --hf-token (or HF_TOKEN) of a token with access.",
                self.model_identifier
            )
        })?;
        Ok(repository_info
            .siblings
            .iter()
            .map(|sibling| sibling.rfilename.clone())
            .collect())
    }

    /// Downloads (or reuses from cache) a single named file.
    pub fn download(&self, name: &str) -> anyhow::Result<PathBuf> {
        self.repo.get(name).map_err(|error| {
            anyhow::anyhow!(
                "failed to download '{name}': {error}. \
                 If the model is gated, pass --hf-token (or HF_TOKEN) of a token with access."
            )
        })
    }

    /// Verifies, before downloading, that the repository actually exposes a
    /// servable checkpoint.
    ///
    /// Without this preflight a missing file surfaces as a bare `404` that is
    /// easy to misread as an authentication problem. Many Hugging Face
    /// repositories (for example `harshatheg/Qwen-2.5-1B-RLCD`, which only
    /// contains the demo source code) share the model name but hold no weights.
    ///
    /// A repository is servable when it exposes either a dense layout
    /// (`config.json` + `tokenizer.json` + weights) or a GGUF file. GGUF
    /// repositories omit `config.json` (parameters come from the GGUF metadata)
    /// but still need a tokenizer.
    pub fn ensure_servable_checkpoint(&self) -> anyhow::Result<()> {
        let available_file_names = self.available_file_names()?;
        let has_gguf = available_file_names
            .iter()
            .any(|name| name.ends_with(".gguf"));
        if has_gguf {
            return Ok(());
        }
        let missing = missing_model_files(&available_file_names);
        if missing.is_empty() {
            return Ok(());
        }
        Err(anyhow::anyhow!(
            "repository '{}' is not a servable checkpoint: it provides no {}. \
             A servable model must expose config.json, tokenizer.json and model.safetensors, \
             or a GGUF file plus tokenizer.json. \
             The repository looks like code-only or an unsupported layout (for example MLX). \
             Point --model-id at a full-precision Llama or Qwen2 repository \
             (for example Qwen/Qwen2.5-1.5B-Instruct) or a GGUF one \
             (for example Qwen/Qwen2.5-1.5B-Instruct-GGUF).",
            self.model_identifier,
            missing.join(", ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repository_carries_explicit_token_without_network() -> anyhow::Result<()> {
        let repository = ModelRepository::new(
            "some-organization/some-model",
            Some("secret-token-value".to_string()),
        )?;
        assert_eq!(
            repository.model_identifier(),
            "some-organization/some-model"
        );
        assert_eq!(repository.hugging_face_token(), Some("secret-token-value"));
        Ok(())
    }

    #[test]
    fn repository_supports_missing_token_for_public_models() -> anyhow::Result<()> {
        let repository = ModelRepository::new("some-organization/some-model", None)?;
        assert_eq!(repository.hugging_face_token(), None);
        Ok(())
    }

    #[test]
    fn complete_checkpoint_reports_nothing_missing() {
        let available = vec![
            "config.json".to_string(),
            "tokenizer.json".to_string(),
            "model.safetensors".to_string(),
            "README.md".to_string(),
        ];
        assert!(missing_model_files(&available).is_empty());
    }

    #[test]
    fn code_only_repository_reports_every_required_file_missing() {
        // The actual sibling list of `harshatheg/Qwen-2.5-1B-RLCD`, which is the
        // parallel-constrained-decoding demo source, not a checkpoint.
        let available: Vec<String> = [
            "README.md",
            "app.py",
            "core/engine.py",
            "core/schema.py",
            "presets/support_triage.json",
            "server/app.py",
            "web/index.html",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        assert_eq!(
            missing_model_files(&available),
            vec!["config.json", "tokenizer.json", "model.safetensors"]
        );
    }

    #[test]
    fn partial_checkpoint_reports_only_the_absent_file() {
        let available = vec!["config.json".to_string(), "model.safetensors".to_string()];
        assert_eq!(missing_model_files(&available), vec!["tokenizer.json"]);
    }
}
