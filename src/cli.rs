use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

/// Manaca Jev-like: avaliação determinística via Candle, API HTTP formato Jev.
#[derive(Parser, Debug)]
#[command(
    name = "manaca-jev-like",
    about = "Jev-compatible evaluation API over local Candle inference"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Sobe a API HTTP no formato Jev (POST /v1/systemone).
    Serve(ServeArgs),
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Interface de escuta.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Porta de escuta.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Hugging Face model id (checkpoint arquitetura Llama com
    /// config.json, tokenizer.json e model.safetensors).
    #[arg(long, default_value = "TinyLlama/TinyLlama-1.1B-Chat-v1.0")]
    pub model_id: String,

    /// Arquivo de memória/contexto avaliado junto com cada request.
    /// Será substituível por RAG sem trocar a interface.
    #[arg(long, default_value = "resources/memory.md")]
    pub context: PathBuf,

    /// Public model name announced in /v1/models and in responses.
    #[arg(long, default_value = "jev-latest")]
    pub served_model_name: String,

    /// Hugging Face access token for gated models.
    /// Falls back to the HF_TOKEN environment variable when the flag is omitted.
    #[arg(long = "hf-token", env = "HF_TOKEN")]
    pub hugging_face_token: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn environment_lock() -> &'static Mutex<()> {
        static ENVIRONMENT_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        ENVIRONMENT_GUARD.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn serve_subcommand_parses_all_options() {
        let cli = Cli::try_parse_from([
            "manaca-jev-like",
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--model-id",
            "some-org/some-model",
            "--context",
            "memory.md",
            "--served-model-name",
            "manaca-1",
        ])
        .unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(args.host, "0.0.0.0");
        assert_eq!(args.port, 8080);
        assert_eq!(args.model_id, "some-org/some-model");
        assert_eq!(args.context, PathBuf::from("memory.md"));
        assert_eq!(args.served_model_name, "manaca-1");
    }

    #[test]
    fn serve_uses_sensible_defaults() {
        let _guard = environment_lock().lock().unwrap();
        let previous_value = std::env::var("HF_TOKEN").ok();
        std::env::remove_var("HF_TOKEN");
        let cli = Cli::try_parse_from(["manaca-jev-like", "serve"]).unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(args.host, "0.0.0.0");
        assert_eq!(args.port, 8080);
        assert_eq!(args.context, PathBuf::from("resources/memory.md"));
        if let Some(previous) = previous_value {
            std::env::set_var("HF_TOKEN", previous);
        }
    }

    #[test]
    fn serve_parses_explicit_hugging_face_token() {
        let cli = Cli::try_parse_from([
            "manaca-jev-like",
            "serve",
            "--hf-token",
            "secret-token-value",
        ])
        .unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(
            args.hugging_face_token.as_deref(),
            Some("secret-token-value")
        );
    }

    #[test]
    fn serve_defaults_to_no_hugging_face_token() {
        let _guard = environment_lock().lock().unwrap();
        let previous_value = std::env::var("HF_TOKEN").ok();
        std::env::remove_var("HF_TOKEN");
        let cli = Cli::try_parse_from(["manaca-jev-like", "serve"]).unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(args.hugging_face_token, None);
        if let Some(previous) = previous_value {
            std::env::set_var("HF_TOKEN", previous);
        }
    }

    #[test]
    fn serve_falls_back_to_hugging_face_token_environment() {
        let _guard = environment_lock().lock().unwrap();
        let previous_value = std::env::var("HF_TOKEN").ok();
        std::env::set_var("HF_TOKEN", "environment-token-value");
        let cli = Cli::try_parse_from(["manaca-jev-like", "serve"]).unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(
            args.hugging_face_token.as_deref(),
            Some("environment-token-value")
        );
        if let Some(previous) = previous_value {
            std::env::set_var("HF_TOKEN", previous);
        } else {
            std::env::remove_var("HF_TOKEN");
        }
    }
}
