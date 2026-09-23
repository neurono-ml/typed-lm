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
    #[arg(long, env = "HOST", default_value = "0.0.0.0")]
    pub host: String,

    /// Porta de escuta.
    #[arg(long, env = "PORT", default_value_t = 8080)]
    pub port: u16,

    /// Hugging Face model id (checkpoint arquitetura Llama com
    /// config.json, tokenizer.json e model.safetensors).
    #[arg(
        long,
        env = "MODEL_ID",
        default_value = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )]
    pub model_id: String,

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn environment_lock() -> &'static Mutex<()> {
        static ENVIRONMENT_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        ENVIRONMENT_GUARD.get_or_init(|| Mutex::new(()))
    }

    fn lock_environment() -> std::sync::MutexGuard<'static, ()> {
        environment_lock()
            .lock()
            .unwrap_or_else(|poison_error| poison_error.into_inner())
    }

    const MANAGED_VARIABLES: [&str; 6] = [
        "HOST",
        "PORT",
        "MODEL_ID",
        "CONTEXT_PATH",
        "SERVED_MODEL_NAME",
        "HF_TOKEN",
    ];

    fn snapshot_environment() -> Vec<(String, Option<String>)> {
        MANAGED_VARIABLES
            .iter()
            .map(|name| ((*name).to_string(), std::env::var(*name).ok()))
            .collect()
    }

    fn restore_environment(snapshot: Vec<(String, Option<String>)>) {
        for (name, previous_value) in snapshot {
            if let Some(previous) = previous_value {
                std::env::set_var(name, previous);
            } else {
                std::env::remove_var(name);
            }
        }
    }

    fn clear_managed_variables() {
        for name in MANAGED_VARIABLES {
            std::env::remove_var(name);
        }
    }

    #[test]
    fn serve_subcommand_parses_all_options() {
        let _guard = lock_environment();
        let snapshot = snapshot_environment();
        clear_managed_variables();
        let cli = Cli::try_parse_from([
            "manaca-jev-like",
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--model-id",
            "some-org/some-model",
            "--context-path",
            "memory.md",
            "--served-model-name",
            "manaca-1",
        ])
        .unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(args.host, "0.0.0.0");
        assert_eq!(args.port, 8080);
        assert_eq!(args.model_id, "some-org/some-model");
        assert_eq!(args.context_path, Some(PathBuf::from("memory.md")));
        assert_eq!(args.served_model_name, "manaca-1");
        restore_environment(snapshot);
    }

    #[test]
    fn serve_uses_sensible_defaults() {
        let _guard = lock_environment();
        let snapshot = snapshot_environment();
        clear_managed_variables();
        let cli = Cli::try_parse_from(["manaca-jev-like", "serve"]).unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(args.host, "0.0.0.0");
        assert_eq!(args.port, 8080);
        assert_eq!(args.model_id, "TinyLlama/TinyLlama-1.1B-Chat-v1.0");
        assert_eq!(args.context_path, None);
        assert_eq!(args.served_model_name, "jev-latest");
        restore_environment(snapshot);
    }

    #[test]
    fn serve_parses_explicit_hugging_face_token() {
        let _guard = lock_environment();
        let snapshot = snapshot_environment();
        clear_managed_variables();
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
        restore_environment(snapshot);
    }

    #[test]
    fn serve_defaults_to_no_hugging_face_token() {
        let _guard = lock_environment();
        let snapshot = snapshot_environment();
        clear_managed_variables();
        let cli = Cli::try_parse_from(["manaca-jev-like", "serve"]).unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(args.hugging_face_token, None);
        restore_environment(snapshot);
    }

    #[test]
    fn serve_falls_back_to_hugging_face_token_environment() {
        let _guard = lock_environment();
        let snapshot = snapshot_environment();
        clear_managed_variables();
        std::env::set_var("HF_TOKEN", "environment-token-value");
        let cli = Cli::try_parse_from(["manaca-jev-like", "serve"]).unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(
            args.hugging_face_token.as_deref(),
            Some("environment-token-value")
        );
        restore_environment(snapshot);
    }

    #[test]
    fn serve_falls_back_to_host_environment() {
        let _guard = lock_environment();
        let snapshot = snapshot_environment();
        clear_managed_variables();
        std::env::set_var("HOST", "127.0.0.1");
        let cli = Cli::try_parse_from(["manaca-jev-like", "serve"]).unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(args.host, "127.0.0.1");
        restore_environment(snapshot);
    }

    #[test]
    fn serve_falls_back_to_port_environment() {
        let _guard = lock_environment();
        let snapshot = snapshot_environment();
        clear_managed_variables();
        std::env::set_var("PORT", "9090");
        let cli = Cli::try_parse_from(["manaca-jev-like", "serve"]).unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(args.port, 9090);
        restore_environment(snapshot);
    }

    #[test]
    fn serve_falls_back_to_model_id_environment() {
        let _guard = lock_environment();
        let snapshot = snapshot_environment();
        clear_managed_variables();
        std::env::set_var("MODEL_ID", "some-org/environment-model");
        let cli = Cli::try_parse_from(["manaca-jev-like", "serve"]).unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(args.model_id, "some-org/environment-model");
        restore_environment(snapshot);
    }

    #[test]
    fn serve_falls_back_to_context_environment() {
        let _guard = lock_environment();
        let snapshot = snapshot_environment();
        clear_managed_variables();
        std::env::set_var("CONTEXT_PATH", "custom/memory.md");
        let cli = Cli::try_parse_from(["manaca-jev-like", "serve"]).unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(args.context_path, Some(PathBuf::from("custom/memory.md")));
        restore_environment(snapshot);
    }

    #[test]
    fn serve_falls_back_to_served_model_name_environment() {
        let _guard = lock_environment();
        let snapshot = snapshot_environment();
        clear_managed_variables();
        std::env::set_var("SERVED_MODEL_NAME", "environment-model");
        let cli = Cli::try_parse_from(["manaca-jev-like", "serve"]).unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(args.served_model_name, "environment-model");
        restore_environment(snapshot);
    }

    #[test]
    fn serve_flags_take_precedence_over_environment() {
        let _guard = lock_environment();
        let snapshot = snapshot_environment();
        clear_managed_variables();
        std::env::set_var("HOST", "127.0.0.1");
        std::env::set_var("PORT", "9090");
        std::env::set_var("MODEL_ID", "some-org/environment-model");
        std::env::set_var("CONTEXT_PATH", "custom/memory.md");
        std::env::set_var("SERVED_MODEL_NAME", "environment-model");
        let cli = Cli::try_parse_from([
            "manaca-jev-like",
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--model-id",
            "some-org/flag-model",
            "--context-path",
            "resources/memory.md",
            "--served-model-name",
            "flag-model",
        ])
        .unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(args.host, "0.0.0.0");
        assert_eq!(args.port, 8080);
        assert_eq!(args.model_id, "some-org/flag-model");
        assert_eq!(
            args.context_path,
            Some(PathBuf::from("resources/memory.md"))
        );
        assert_eq!(args.served_model_name, "flag-model");
        restore_environment(snapshot);
    }
}
