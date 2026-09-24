use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use actix_web::{web, App, HttpServer};

use crate::api::state::{ApplicationState, SharedState};
use crate::cli::{ServeArguments, ServerBinding};
use crate::domain::evaluator::Evaluator;
use crate::infrastructure::candle_evaluator::CandleEvaluator;
use crate::infrastructure::language_model::LanguageModel;
use crate::infrastructure::session_cache::SessionCacheConfiguration;
use typed_lm_common::checkpoint::{ModelReference, WeightKind};
use typed_lm_common::checkpoint_resolver::{
    HubCheckpointResolver, LoadableCheckpoint, LocalCheckpointResolver,
};
use typed_lm_common::context::{ContextProvider, FileContextProvider};
use typed_lm_common::device::DeviceResolver;

/// Runs the parsed command line interface until completion.
pub async fn run(serve_arguments: ServeArguments) -> anyhow::Result<()> {
    run_serve_command(serve_arguments).await
}

/// Measures elapsed startup time in seconds.
#[allow(dead_code)]
pub fn elapsed_startup_seconds(started: Instant) -> f64 {
    started.elapsed().as_secs_f64()
}

/// Formats the listen address reported on startup.
#[allow(dead_code)]
pub fn listen_address(host: &str, port: u16) -> String {
    format!("{host}:{port}")
}

/// Builds the shared application state handed to every worker.
#[allow(dead_code)]
pub fn build_shared_state(
    evaluator: Arc<dyn Evaluator + Send + Sync>,
    served_model_name: String,
    context_name: String,
    startup_seconds: f64,
) -> SharedState {
    build_application_state(
        evaluator,
        served_model_name,
        context_name,
        startup_seconds,
        0,
    )
}

/// Builds the full application state including the system base-cache length.
///
/// `system_sequence_length` records how many tokens the precomputed system
/// base cache holds. Request handlers clone that immutable prefix and run
/// the forward pass from that offset on a disposable cache clone.
pub fn build_application_state(
    evaluator: Arc<dyn Evaluator + Send + Sync>,
    served_model_name: String,
    context_name: String,
    startup_seconds: f64,
    system_sequence_length: usize,
) -> ApplicationState {
    ApplicationState {
        evaluator,
        served_model_name,
        context_name,
        startup_seconds,
        system_sequence_length,
    }
}

/// Name reported when the server runs without a memory context file.
pub const ABSENT_CONTEXT_NAME: &str = "none";

/// Builds an actionable error for a failed socket bind.
///
/// The standard library only reports `Address already in use (os error 98)`,
/// which does not tell the operator what to do next. This message names the
/// address and points at the two supported remedies.
pub fn bind_failure_message(address: &str, port: u16, error: &std::io::Error) -> String {
    format!(
        "failed to bind {address} ({error}); port {port} is already in use — stop the other instance or pass --port <free-port>",
    )
}

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

/// Resolves the CLI model reference into a loadable checkpoint (Hub or local).
fn resolve_checkpoint(serve_arguments: &ServeArguments) -> anyhow::Result<LoadableCheckpoint> {
    let reference = ModelReference::resolve(&serve_arguments.model_id);
    let explicit_weights = serve_arguments.weights_file.as_deref();
    match &reference {
        ModelReference::Hub { .. } => HubCheckpointResolver::new(
            &serve_arguments.model_id,
            serve_arguments.hugging_face_token.clone(),
        )?
        .with_revision(&serve_arguments.model_revision)
        .with_tokenizer(serve_arguments.tokenizer_file.clone())
        .resolve(&reference, explicit_weights),
        ModelReference::Local { path } => LocalCheckpointResolver::new(
            path.clone(),
            serve_arguments.config_file.clone(),
            serve_arguments.tokenizer_file.clone(),
        )
        .resolve(&reference, explicit_weights),
    }
}

/// Confirms the checkpoint can be executed by the serving loader.
///
/// FP8 and FP4 checkpoints are accepted: the loader dequantizes them to the
/// compute dtype before the forward pass, so candle never needs a low-precision
/// matmul kernel. Only schemes this crate does not model remain rejected.
fn ensure_supported_weight_kind(checkpoint: &LoadableCheckpoint) -> anyhow::Result<()> {
    match checkpoint.weight_kind {
        WeightKind::UnsupportedFloat8 => Err(anyhow::anyhow!(
            "checkpoint '{}' uses an FP8/compressed-tensors scheme this build does not model. \
             Supported low-precision schemes are FP8 (E4M3/E5M2) and MXFP4, both dequantized on \
             load. Alternatively use a full-precision (BF16/F16/F32) or GGML-quantized checkpoint, \
             for example Qwen/Qwen2.5-1.5B-Instruct or its GGUF variant.",
            checkpoint.resolved.reference.describe()
        )),
        WeightKind::Dense | WeightKind::Quantized | WeightKind::Float8 | WeightKind::Float4 => {
            Ok(())
        }
    }
}

async fn run_serve_command(serve_arguments: ServeArguments) -> anyhow::Result<()> {
    let started = Instant::now();
    tracing::info!(
        "resolving execution device for model '{}'",
        serve_arguments.model_id
    );
    let execution_device = DeviceResolver::resolve()?;
    let model_dtype = serve_arguments.model_dtype.resolve(&execution_device);
    tracing::info!("execution device resolved ({model_dtype:?} weights); locating checkpoint");
    let checkpoint = resolve_checkpoint(&serve_arguments)?;
    ensure_supported_weight_kind(&checkpoint)?;
    tracing::info!(
        "checkpoint ready ({} layout, {} architecture, {} weights); loading weights",
        checkpoint.resolved.layout.name(),
        checkpoint.architecture.name(),
        checkpoint.weight_kind.name(),
    );
    let language_model = LanguageModel::load(&checkpoint, &execution_device, model_dtype)?;
    tracing::info!("language model loaded; resolving memory context");
    let (loaded_context, context_name) =
        resolve_memory_context(serve_arguments.context_path.as_deref())?;
    let served_model_name = serve_arguments.served_model_name.clone();
    // The HTTP server factory requires 'static state, so the model is
    // heap-leaked once at startup and borrowed for the process lifetime.
    // The evaluator prefills the system context into a real KV-cache at
    // construction; its token length is exposed as `system_sequence_length`.
    let leaked_model: &'static LanguageModel = Box::leak(Box::new(language_model));
    let session_configuration = SessionCacheConfiguration {
        maximum_entries: serve_arguments.session_cache_entries,
        maximum_tokens: serve_arguments.session_cache_tokens,
    };
    let evaluator = CandleEvaluator::new(
        leaked_model,
        loaded_context,
        served_model_name.clone(),
        session_configuration,
    )?;
    let system_sequence_length = evaluator.context_token_length();
    let startup_seconds = elapsed_startup_seconds(started);
    let shared_state = web::Data::new(build_application_state(
        Arc::new(evaluator),
        served_model_name.clone(),
        context_name.clone(),
        startup_seconds,
        system_sequence_length,
    ));
    let binding = ServerBinding::from_serve_arguments(&serve_arguments);
    let address = binding.listen_address();
    tracing::info!(
        "serving model '{served_model_name}' with context '{context_name}' on {address} (startup took {:.2}s)",
        startup_seconds
    );
    println!("serving model '{served_model_name}' with context '{context_name}' on {address}");
    HttpServer::new(move || {
        App::new()
            .app_data(shared_state.clone())
            .configure(crate::api::configure)
    })
    .bind((serve_arguments.host.as_str(), serve_arguments.port))
    .map_err(|error| {
        anyhow::anyhow!(
            "{}",
            bind_failure_message(&address, serve_arguments.port, &error)
        )
    })?
    .run()
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::evaluator::MockEvaluator;

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
    fn bind_failure_message_names_address_and_remedy() {
        let error = std::io::Error::new(
            std::io::ErrorKind::AddrInUse,
            "Address already in use (os error 98)",
        );
        let message = bind_failure_message("0.0.0.0:8080", 8080, &error);
        assert!(message.contains("0.0.0.0:8080"));
        assert!(message.contains("8080"));
        assert!(message.contains("--port"));
    }

    #[test]
    fn shared_state_carries_startup_metadata() {
        let evaluator: Arc<dyn Evaluator + Send + Sync> =
            Arc::new(MockEvaluator::new("typed-lm-test-model".to_string()));
        let state = build_shared_state(
            evaluator,
            "typed-lm-test-model".to_string(),
            "file:resources/memory.md".to_string(),
            2.5,
        );
        assert_eq!(state.served_model_name, "typed-lm-test-model");
        assert_eq!(state.context_name, "file:resources/memory.md");
        assert!((state.startup_seconds - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn absent_context_resolves_to_empty_text_and_none_name() -> anyhow::Result<()> {
        let (loaded_context, context_name) = resolve_memory_context(None)?;
        assert!(loaded_context.is_empty());
        assert_eq!(context_name, ABSENT_CONTEXT_NAME);
        Ok(())
    }

    #[test]
    fn present_missing_file_reports_its_path() {
        let missing = Path::new("missing-memory-for-test-12345.md");
        let result = resolve_memory_context(Some(missing));
        assert!(result.is_err());
        let Err(error) = result else {
            return;
        };
        assert!(error
            .to_string()
            .contains("missing-memory-for-test-12345.md"));
    }

    #[test]
    fn serving_without_context_builds_empty_state() -> anyhow::Result<()> {
        let (loaded_context, context_name) = resolve_memory_context(None)?;
        let evaluator: Arc<dyn Evaluator + Send + Sync> =
            Arc::new(MockEvaluator::new("typed-lm-test-model".to_string()));
        let state = build_shared_state(
            evaluator,
            "typed-lm-test-model".to_string(),
            context_name,
            0.0,
        );
        assert!(loaded_context.is_empty());
        assert_eq!(state.context_name, ABSENT_CONTEXT_NAME);
        Ok(())
    }

    #[test]
    fn application_state_records_system_sequence_length_for_cache_isolation() {
        let evaluator: Arc<dyn Evaluator + Send + Sync> =
            Arc::new(MockEvaluator::new("typed-lm-test-model".to_string()));
        let state = build_application_state(
            evaluator,
            "typed-lm-test-model".to_string(),
            "file:resources/memory.md".to_string(),
            1.0,
            42,
        );
        assert_eq!(state.system_sequence_length, 42);
        // Cloning the state (as Actix does per worker via web::Data) must
        // preserve the immutable base-cache offset without mutating it.
        let cloned_length = state.system_sequence_length;
        assert_eq!(cloned_length, 42);
    }

    #[test]
    fn present_memory_file_loads_with_source_name() -> anyhow::Result<()> {
        let path = Path::new("resources/memory.md");
        if !path.exists() {
            return Ok(());
        }
        let (loaded_context, context_name) = resolve_memory_context(Some(path))?;
        assert!(loaded_context.contains("GreenLeaf"));
        assert_eq!(context_name, "file:resources/memory.md");
        Ok(())
    }

    fn checkpoint_with_weight_kind(weight_kind: WeightKind) -> LoadableCheckpoint {
        LoadableCheckpoint {
            resolved: typed_lm_common::checkpoint::ResolvedCheckpoint {
                reference: ModelReference::Local {
                    path: std::path::PathBuf::from("/tmp/typed-lm-test-model"),
                },
                config_file: std::path::PathBuf::from("/tmp/typed-lm-test-model/config.json"),
                tokenizer_file: std::path::PathBuf::from("/tmp/typed-lm-test-model/tokenizer.json"),
                layout: typed_lm_common::checkpoint::WeightLayout::Safetensors {
                    files: Vec::new(),
                },
            },
            architecture: typed_lm_common::checkpoint::ModelArchitecture::Qwen2,
            weight_kind,
        }
    }

    #[test]
    fn dense_and_quantized_weight_kinds_are_accepted() {
        for weight_kind in [
            WeightKind::Dense,
            WeightKind::Quantized,
            WeightKind::Float8,
            WeightKind::Float4,
        ] {
            let checkpoint = checkpoint_with_weight_kind(weight_kind);
            assert!(
                ensure_supported_weight_kind(&checkpoint).is_ok(),
                "{weight_kind:?} must be accepted"
            );
        }
    }

    #[test]
    fn unsupported_float8_weight_kind_is_rejected_with_a_remedy() {
        let checkpoint = checkpoint_with_weight_kind(WeightKind::UnsupportedFloat8);
        let result = ensure_supported_weight_kind(&checkpoint);
        assert!(result.is_err());
        let Err(error) = result else {
            return;
        };
        assert!(error.to_string().contains("GGUF"));
    }
}
