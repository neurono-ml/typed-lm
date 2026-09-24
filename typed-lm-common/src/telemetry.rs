use opentelemetry::{trace::TracerProvider as _, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{runtime, trace::TracerProvider};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Default OTLP/gRPC endpoint pointing at the Jaeger companion service
/// declared in `.devcontainer/docker-compose.yml`.
pub const DEFAULT_OTLP_ENDPOINT: &str = "http://jaeger:4317";
/// Environment variable overriding the OTLP endpoint (standard name).
pub const OTLP_ENDPOINT_VARIABLE: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
/// Default filter applied when `RUST_LOG` is not set.
pub const DEFAULT_LOG_FILTER: &str = "info,actix_web=info";

/// Resolves the OTLP endpoint, preferring the standard environment variable.
pub fn otlp_endpoint() -> String {
    std::env::var(OTLP_ENDPOINT_VARIABLE)
        .ok()
        .filter(|candidate| !candidate.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_OTLP_ENDPOINT.to_string())
}

/// Holds the tracer provider for the process lifetime.
///
/// Dropping the guard shuts the provider down so buffered spans are flushed
/// to Jaeger instead of being lost on exit.
pub struct TelemetryGuard {
    tracer_provider: TracerProvider,
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        if let Err(error) = self.tracer_provider.shutdown() {
            eprintln!("failed to shut down tracer provider: {error}");
        }
    }
}

/// Installs the global tracing subscriber.
///
/// Logs go to stdout through the formatting layer while spans and events are
/// also exported as OTLP traces to Jaeger. When Jaeger is unreachable the
/// exporter retries in the background; local logs are unaffected.
pub fn initialize_telemetry(service_name: &str) -> anyhow::Result<TelemetryGuard> {
    let endpoint = otlp_endpoint();
    let span_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint.clone())
        .with_timeout(std::time::Duration::from_secs(5))
        .build()?;
    let span_processor =
        opentelemetry_sdk::trace::BatchSpanProcessor::builder(span_exporter, runtime::Tokio)
            .build();
    let tracer_provider = TracerProvider::builder()
        .with_span_processor(span_processor)
        .with_resource(opentelemetry_sdk::Resource::new(vec![KeyValue::new(
            "service.name",
            service_name.to_string(),
        )]))
        .build();
    let tracer = tracer_provider.tracer(service_name.to_string());
    let opentelemetry_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    let environment_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(DEFAULT_LOG_FILTER));
    tracing_subscriber::registry()
        .with(environment_filter)
        .with(tracing_subscriber::fmt::layer())
        .with(opentelemetry_layer)
        .try_init()
        .map_err(|error| anyhow::anyhow!("failed to install tracing subscriber: {error}"))?;
    tracing::info!("exporting traces to Jaeger via OTLP endpoint '{endpoint}'");
    Ok(TelemetryGuard { tracer_provider })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_endpoint_targets_jaeger_companion_service() {
        assert_eq!(DEFAULT_OTLP_ENDPOINT, "http://jaeger:4317");
    }

    #[test]
    fn empty_endpoint_variable_falls_back_to_default() {
        let variable = OTLP_ENDPOINT_VARIABLE;
        let previous = std::env::var(variable).ok();
        std::env::set_var(variable, "   ");
        assert_eq!(otlp_endpoint(), DEFAULT_OTLP_ENDPOINT);
        match previous {
            Some(previous_value) => std::env::set_var(variable, previous_value),
            None => std::env::remove_var(variable),
        }
    }
}
