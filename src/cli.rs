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
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Porta de escuta.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Hugging Face model id (checkpoint arquitetura Llama com
    /// config.json, tokenizer.json e model.safetensors).
    #[arg(long, default_value = "recogna-nlp/bode-1b-instruct")]
    pub model_id: String,

    /// Arquivo de memória/contexto avaliado junto com cada request.
    /// Será substituível por RAG sem trocar a interface.
    #[arg(long, default_value = "resources/memory.md")]
    pub context: PathBuf,

    /// Nome público do modelo anunciado em /v1/models e nas respostas.
    #[arg(long, default_value = "jev-latest")]
    pub served_model_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let cli = Cli::try_parse_from(["manaca-jev-like", "serve"]).unwrap();
        let Command::Serve(args) = cli.command;
        assert_eq!(args.host, "127.0.0.1");
        assert_eq!(args.port, 8080);
        assert_eq!(args.context, PathBuf::from("resources/memory.md"));
    }
}
