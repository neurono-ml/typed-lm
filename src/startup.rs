use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use actix_web::{web, App, HttpServer};

use crate::api::SharedState;
use crate::cli::{Cli, Command, ServeArgs};
use crate::context::{ContextProvider, FileContextProvider};
use crate::device::DeviceResolver;
use crate::evaluator::{CandleEvaluator, Evaluator};
use crate::model::LanguageModel;
use crate::repository::ModelRepository;

/// Runs the parsed command line interface until completion.
pub async fn run(command_line: Cli) -> anyhow::Result<()> {
    match command_line.command {
        Command::Serve(serve_arguments) => run_serve_command(serve_arguments).await,
    }
}

/// Measures elapsed startup time in seconds.
pub fn elapsed_startup_seconds(started: Instant) -> f64 {
    started.elapsed().as_secs_f64()
}

/// Formats the listen address reported on startup.
pub fn listen_address(host: &str, port: u16) -> String {
    format!("{host}:{port}")
}

/// Builds the shared application state handed to every worker.
pub fn build_shared_state(
    evaluator: Arc<dyn Evaluator + Send + Sync>,
    served_model_name: String,
    context_name: String,
    startup_seconds: f64,
) -> SharedState {
    SharedState {
        evaluator,
        served_model_name,
        context_name,
        startup_seconds,
    }
}

/// Name reported when the server runs without a memory context file.
pub const ABSENT_CONTEXT_NAME: &str = "none";

/// Loads the memory context file, mapping IO failures to a clear message.
fn load_memory_context(path: &Path) -> anyhow::Result<(String, String)> {
    let context_provider = FileContextProvider::new(path);
    let context_name = context_provider.name();
    let loaded_context = context_provider.load().map_err(|error| {
        anyhow::anyhow!(
            "failed to load memory context from '{}': {error}",
            path.display()
        )
    })?;
    Ok((loaded_context, context_name))
}

/// Resolves the optional memory context into loaded text and a source name.
/// A missing path yields an empty context named [`ABSENT_CONTEXT_NAME`].
fn resolve_memory_context(path: Option<&Path>) -> anyhow::Result<(String, String)> {
    match path {
        None => Ok((String::new(), ABSENT_CONTEXT_NAME.to_string())),
        Some(present_path) => load_memory_context(present_path),
    }
}

async fn run_serve_command(serve_arguments: ServeArgs) -> anyhow::Result<()> {
    let started = Instant::now();
    let execution_device = DeviceResolver::resolve()?;
    let model_files = ModelRepository::new(
        &serve_arguments.model_id,
        serve_arguments.hugging_face_token.clone(),
    )?
    .files()?;
    let language_model = LanguageModel::load(
        &model_files.config,
        &model_files.weights,
        &model_files.tokenizer,
        &execution_device,
    )?;
    let (loaded_context, context_name) =
        resolve_memory_context(serve_arguments.context_path.as_deref())?;
    let served_model_name = serve_arguments.served_model_name.clone();
    // The HTTP server factory requires 'static state, so the model is
    // heap-leaked once at startup and borrowed for the process lifetime.
    let leaked_model: &'static LanguageModel = Box::leak(Box::new(language_model));
    let evaluator = CandleEvaluator::new(leaked_model, loaded_context, served_model_name.clone());
    let startup_seconds = elapsed_startup_seconds(started);
    let shared_state = web::Data::new(build_shared_state(
        Arc::new(evaluator),
        served_model_name.clone(),
        context_name.clone(),
        startup_seconds,
    ));
    let address = listen_address(&serve_arguments.host, serve_arguments.port);
    println!("serving model '{served_model_name}' with context '{context_name}' on {address}");
    HttpServer::new(move || {
        App::new()
            .app_data(shared_state.clone())
            .configure(crate::api::configure)
    })
    .bind((serve_arguments.host.as_str(), serve_arguments.port))?
    .run()
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::MockEvaluator;

    #[test]
    fn startup_seconds_are_finite_and_non_negative() {
        let started = Instant::now();
        let seconds = elapsed_startup_seconds(started);
        assert!(seconds.is_finite());
        assert!(seconds >= 0.0);
    }

    #[test]
    fn address_combines_host_and_port() {
        assert_eq!(listen_address("127.0.0.1", 8080), "127.0.0.1:8080");
        assert_eq!(listen_address("0.0.0.0", 1), "0.0.0.0:1");
    }

    #[test]
    fn shared_state_carries_startup_metadata() {
        let evaluator: Arc<dyn Evaluator + Send + Sync> =
            Arc::new(MockEvaluator::new("manaca-test-model".to_string()));
        let state = build_shared_state(
            evaluator,
            "manaca-test-model".to_string(),
            "file:resources/memory.md".to_string(),
            2.5,
        );
        assert_eq!(state.served_model_name, "manaca-test-model");
        assert_eq!(state.context_name, "file:resources/memory.md");
        assert!((state.startup_seconds - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn absent_context_resolves_to_empty_text_and_none_name() {
        let (loaded_context, context_name) = resolve_memory_context(None).unwrap();
        assert!(loaded_context.is_empty());
        assert_eq!(context_name, ABSENT_CONTEXT_NAME);
    }

    #[test]
    fn present_missing_file_reports_its_path() {
        let missing = Path::new("missing-memory-for-test-12345.md");
        let error = resolve_memory_context(Some(missing)).unwrap_err();
        assert!(error
            .to_string()
            .contains("missing-memory-for-test-12345.md"));
    }

    #[test]
    fn serving_without_context_builds_empty_state() {
        let (loaded_context, context_name) = resolve_memory_context(None).unwrap();
        let evaluator: Arc<dyn Evaluator + Send + Sync> =
            Arc::new(MockEvaluator::new("manaca-test-model".to_string()));
        let state = build_shared_state(
            evaluator,
            "manaca-test-model".to_string(),
            context_name,
            0.0,
        );
        assert!(loaded_context.is_empty());
        assert_eq!(state.context_name, ABSENT_CONTEXT_NAME);
    }

    #[test]
    fn present_memory_file_loads_with_source_name() {
        let path = Path::new("resources/memory.md");
        if !path.exists() {
            return;
        }
        let (loaded_context, context_name) = resolve_memory_context(Some(path)).unwrap();
        assert!(loaded_context.contains("GreenLeaf"));
        assert_eq!(context_name, "file:resources/memory.md");
    }
}
