use clap::Parser;
use std::path::PathBuf;

use crate::infrastructure::session_cache::{
    DEFAULT_SESSION_CACHE_ENTRIES, DEFAULT_SESSION_CACHE_TOKENS,
};
use typed_lm_common::device::ModelDtype;

/// typed-lm-serve: deterministic evaluation via Candle, Jev-format HTTP API.
#[derive(Parser, Debug)]
#[command(
    name = "typed-lm-serve",
    about = "Jev-compatible evaluation API over local Candle inference"
)]
pub struct ServeArguments {
    /// Listen interface.
    #[arg(long, env = "HOST", default_value = "0.0.0.0")]
    pub host: String,

    /// Listen port.
    #[arg(long, env = "PORT", default_value_t = 8080)]
    pub port: u16,

    /// Model reference: a Hugging Face repository identifier or a local path.
    ///
    /// Supported layouts are detected automatically: safetensors (single or
    /// sharded), GGUF (dense or quantized), PyTorch `.pth`/`.bin` and NumPy
    /// `.npz`. Architectures: Llama and Qwen2.
    #[arg(
        long = "model-id",
        env = "MODEL_ID",
        default_value = "Qwen/Qwen2.5-1.5B-Instruct",
        alias = "model"
    )]
    pub model_id: String,

    /// Hugging Face revision (branch, tag or commit hash); default `main`.
    #[arg(long, env = "MODEL_REVISION", default_value = "main")]
    pub model_revision: String,

    /// Explicit weight file, overriding automatic layout detection.
    /// Useful when a repository or directory holds several candidates.
    #[arg(long, env = "WEIGHTS_FILE")]
    pub weights_file: Option<PathBuf>,

    /// Explicit `tokenizer.json`, overriding the one next to the weights.
    #[arg(long, env = "TOKENIZER_FILE")]
    pub tokenizer_file: Option<PathBuf>,

    /// Explicit `config.json`, overriding the one next to the weights.
    #[arg(long, env = "CONFIG_FILE")]
    pub config_file: Option<PathBuf>,

    /// Memory/context file evaluated with each request.
    /// Optional and replaceable by retrieval without changing the interface.
    /// When absent, the evaluator receives an empty context.
    #[arg(long, env = "CONTEXT_PATH")]
    pub context_path: Option<PathBuf>,

    /// Public model name announced in /v1/models and in responses.
    #[arg(long, env = "SERVED_MODEL_NAME", default_value = "jev-latest")]
    pub served_model_name: String,

    /// Hugging Face access token for gated models.
    /// Falls back to the HF_TOKEN environment variable when the flag is omitted.
    #[arg(long = "hf-token", env = "HF_TOKEN")]
    pub hugging_face_token: Option<String>,

    /// Weight numeric type for inference.
    /// `auto` keeps F32 on the CPU and picks F16 on CUDA/Metal accelerators.
    #[arg(long, env = "MODEL_DTYPE", value_enum, default_value = "auto")]
    pub model_dtype: ModelDtype,

    /// Number of distinct evaluated states whose prefilled prefix cache is
    /// retained for reuse across requests. `0` disables session caching.
    #[arg(
        long = "session-cache-entries",
        env = "SESSION_CACHE_ENTRIES",
        default_value_t = DEFAULT_SESSION_CACHE_ENTRIES
    )]
    pub session_cache_entries: usize,

    /// Total token budget across all retained session prefixes; the least
    /// recently used entries are evicted first. `0` disables session caching.
    #[arg(
        long = "session-cache-tokens",
        env = "SESSION_CACHE_TOKENS",
        default_value_t = DEFAULT_SESSION_CACHE_TOKENS
    )]
    pub session_cache_tokens: usize,
}

/// Internal server binding derived from CLI arguments (no clap involved).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerBinding {
    pub host: String,
    pub port: u16,
}

impl ServerBinding {
    pub fn from_serve_arguments(arguments: &ServeArguments) -> Self {
        Self {
            host: arguments.host.clone(),
            port: arguments.port,
        }
    }

    pub fn listen_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_serve_arguments(host: &str, port: u16) -> ServeArguments {
        ServeArguments {
            host: host.to_string(),
            port,
            model_id: "some-organization/some-model".to_string(),
            model_revision: "main".to_string(),
            weights_file: None,
            tokenizer_file: None,
            config_file: None,
            context_path: None,
            served_model_name: "typed-lm-test-model".to_string(),
            hugging_face_token: None,
            model_dtype: ModelDtype::Auto,
            session_cache_entries: DEFAULT_SESSION_CACHE_ENTRIES,
            session_cache_tokens: DEFAULT_SESSION_CACHE_TOKENS,
        }
    }

    #[test]
    fn server_binding_copies_host_and_port_from_internal_arguments() {
        let arguments = sample_serve_arguments("127.0.0.1", 9090);
        let binding = ServerBinding::from_serve_arguments(&arguments);
        assert_eq!(binding.host, "127.0.0.1");
        assert_eq!(binding.port, 9090);
    }

    #[test]
    fn listen_address_combines_host_and_port() {
        let binding = ServerBinding {
            host: "0.0.0.0".to_string(),
            port: 8080,
        };
        assert_eq!(binding.listen_address(), "0.0.0.0:8080");
    }

    #[test]
    fn model_dtype_defaults_to_auto_and_parses_overrides() -> anyhow::Result<()> {
        let default_arguments = ServeArguments::try_parse_from(["typed-lm-serve"])?;
        assert_eq!(default_arguments.model_dtype, ModelDtype::Auto);

        let overridden =
            ServeArguments::try_parse_from(["typed-lm-serve", "--model-dtype", "bf16"])?;
        assert_eq!(overridden.model_dtype, ModelDtype::Bf16);
        Ok(())
    }
}
