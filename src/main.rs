use clap::Parser;

use config::cli::Cli;

mod api;
mod bootstrap;
mod config;
mod domain;
mod infrastructure;
/// Service name reported to Jaeger as the trace resource.
const SERVICE_NAME: &str = "manaca-jev-like";

/// Tokio is the async runtime. Actix Web runs on top of it: handlers stay
/// async and non-blocking, while blocking Candle inference is dispatched to
/// the Tokio blocking pool via `web::block` (see the API handlers).
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _telemetry_guard = bootstrap::telemetry::initialize_telemetry(SERVICE_NAME)?;
    tracing::info!("starting manaca jev-like service");
    let command_line = Cli::parse();
    bootstrap::startup::run(command_line).await
}
