use clap::Parser;

use typed_lm_common::telemetry;
use typed_lm_serve::cli::ServeArguments;

/// Service name reported to Jaeger as the trace resource.
const SERVICE_NAME: &str = "typed-lm-serve";

/// Tokio is the async runtime. Actix Web runs on top of it: handlers stay
/// async and non-blocking, while blocking Candle inference is dispatched to
/// the Tokio blocking pool via `web::block` (see the API handlers).
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _telemetry_guard = telemetry::initialize_telemetry(SERVICE_NAME)?;
    tracing::info!("starting typed-lm serve");
    let arguments = ServeArguments::parse();
    typed_lm_serve::bootstrap::startup::run(arguments).await
}
