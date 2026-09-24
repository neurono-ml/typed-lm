//! Checkpoint discovery: format and source detection for local paths and
//! Hugging Face repositories.
//!
//! The server no longer hard-codes a single weight layout. Given a model
//! reference, this module decides, without user input beyond the reference:
//!
//! 1. **Source** — a local filesystem path or a Hugging Face repository.
//! 2. **Layout** — sharded/single safetensors, GGUF, PyTorch or NumPy.
//! 3. **Architecture** — Llama or Qwen2 (other decoder families are rejected
//!    with an actionable error).
//! 4. **Weight kind** — dense (BF16/F16/F32) or GGML-quantized.
//!
//! Pure detection functions take file-name lists (or directory contents) and
//! are therefore unit-testable without network access or real weights.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Weight file name recognized by the safetensors loader (single file).
pub const SINGLE_SAFETENSORS_NAME: &str = "model.safetensors";

/// Index file describing a sharded safetensors checkpoint.
pub const SAFETENSORS_INDEX_NAME: &str = "model.safetensors.index.json";

/// Tokenizer file required by the `tokenizers` crate.
pub const TOKENIZER_NAME: &str = "tokenizer.json";

/// Model configuration file required by dense (non-GGUF) checkpoints.
pub const CONFIG_NAME: &str = "config.json";

/// Reference passed on the command line: a local path or a Hub repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelReference {
    /// Repository identifier on the Hugging Face Hub (e.g. `Qwen/Qwen2.5-1.5B`).
    Hub { repository: String },
    /// Path on the local filesystem (a directory or a weight file).
    Local { path: PathBuf },
}

impl ModelReference {
    /// Resolves a raw `--model-id` value into a concrete reference.
    ///
    /// Anything that exists on disk is treated as local; everything else is
    /// interpreted as a Hugging Face repository identifier. This keeps the
    /// flag usable with both origins without a mode switch.
    pub fn resolve(raw: &str) -> Self {
        let path = Path::new(raw);
        if path.exists() {
            Self::Local {
                path: path.to_path_buf(),
            }
        } else {
            Self::Hub {
                repository: raw.to_string(),
            }
        }
    }

    /// Human-readable description used in logs and error messages.
    pub fn describe(&self) -> String {
        match self {
            Self::Hub { repository } => format!("hub:{repository}"),
            Self::Local { path } => format!("local:{}", path.display()),
        }
    }
}

/// On-disk organization of a checkpoint's weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightLayout {
    /// One or more safetensors files (sharding supported by candle).
    Safetensors { files: Vec<PathBuf> },
    /// A single GGUF file (dense or GGML-quantized).
    Gguf { file: PathBuf },
    /// A PyTorch `pth`/`bin` file.
    Pytorch { file: PathBuf },
    /// A NumPy `.npz` archive.
    Numpy { file: PathBuf },
}

impl WeightLayout {
    /// Short label used in diagnostics.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Safetensors { .. } => "safetensors",
            Self::Gguf { .. } => "gguf",
            Self::Pytorch { .. } => "pytorch",
            Self::Numpy { .. } => "numpy",
        }
    }
}

/// Decoder architecture understood by the vendored parallel forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    Llama,
    Qwen2,
}

impl ModelArchitecture {
    /// Maps the `model_type`/`general.architecture` value to an architecture.
    pub fn from_model_type(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "llama" => Some(Self::Llama),
            "qwen2" => Some(Self::Qwen2),
            _ => None,
        }
    }

    /// Whether the attention projections carry biases (Qwen2 does, Llama does not).
    pub fn has_query_key_value_bias(self) -> bool {
        match self {
            Self::Llama => false,
            Self::Qwen2 => true,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
        }
    }
}

/// Numerical kind of the weights, driving dense vs. quantized loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightKind {
    /// Dense floating point weights (safetensors/pth/npz or GGUF F32/F16/BF16).
    Dense,
    /// GGML-quantized weights requiring the `QMatMul` path.
    Quantized,
    /// FP8 (`F8_E4M3`) or `compressed-tensors` weights that candle 0.11 cannot
    /// execute (it has no FP8 matmul kernel).
    UnsupportedFloat8,
}

/// Everything needed to load a model, resolved before touching the weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCheckpoint {
    pub reference: ModelReference,
    pub config_file: PathBuf,
    pub tokenizer_file: PathBuf,
    pub layout: WeightLayout,
}

/// File names available in a checkpoint (Hub siblings or directory listing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvailableFiles {
    pub names: Vec<String>,
}

impl AvailableFiles {
    pub fn new(names: Vec<String>) -> Self {
        Self { names }
    }

    fn contains(&self, name: &str) -> bool {
        self.names.iter().any(|candidate| candidate == name)
    }

    /// Public membership check (used by resolvers deciding on shard indexes).
    pub fn has(&self, name: &str) -> bool {
        self.contains(name)
    }

    /// All entries ending with the given suffix, sorted for determinism.
    fn with_suffix(&self, suffix: &str) -> Vec<String> {
        let mut matches: Vec<String> = self
            .names
            .iter()
            .filter(|name| name.ends_with(suffix))
            .cloned()
            .collect();
        matches.sort();
        matches
    }
}

/// Errors raised while detecting the checkpoint layout.
#[derive(Debug)]
pub enum DetectionError {
    /// The `model.safetensors.index.json` body could not be parsed.
    InvalidIndex { reason: String },
    /// Several candidate weight files were found and none was selected.
    AmbiguousWeights { candidates: Vec<String> },
    /// No file matching any supported format was found.
    NoWeights,
    /// `config.json` is required for a dense checkpoint but is absent.
    MissingConfig,
    /// `tokenizer.json` is required but is absent.
    MissingTokenizer,
}

impl std::fmt::Display for DetectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidIndex { reason } => write!(
                formatter,
                "failed to parse the safetensors shard index: {reason}"
            ),
            Self::AmbiguousWeights { candidates } => write!(
                formatter,
                "multiple weight files were found ({}); pass --weights-file to select one: {}",
                candidates.len(),
                candidates.join(", ")
            ),
            Self::NoWeights => write!(
                formatter,
                "no supported weight file was found; expected one of model.safetensors, \
                 model.safetensors.index.json, *.gguf, *.pth, *.bin or *.npz"
            ),
            Self::MissingConfig => write!(
                formatter,
                "config.json is required for dense checkpoints and was not found"
            ),
            Self::MissingTokenizer => write!(
                formatter,
                "tokenizer.json is required and was not found; pass --tokenizer-file to point at one"
            ),
        }
    }
}

impl std::error::Error for DetectionError {}

/// Detects the weight layout from the available file names.
///
/// Detection order is explicit and deterministic:
/// index-backed safetensors, single safetensors, GGUF, PyTorch, NumPy.
/// An optional `explicit_weights_file` short-circuits the search.
pub fn detect_weight_layout(
    available: &AvailableFiles,
    explicit_weights_file: Option<&Path>,
) -> Result<WeightLayout, DetectionError> {
    if let Some(file) = explicit_weights_file {
        return Ok(layout_from_extension(file));
    }

    if available.contains(SAFETENSORS_INDEX_NAME) {
        return Err(DetectionError::InvalidIndex {
            reason: "the shard index must be read from disk or the repository to list its shards"
                .to_string(),
        });
    }

    if available.contains(SINGLE_SAFETENSORS_NAME) {
        return Ok(WeightLayout::Safetensors {
            files: vec![PathBuf::from(SINGLE_SAFETENSORS_NAME)],
        });
    }

    let safetensors = available.with_suffix(".safetensors");
    if safetensors.len() == 1 {
        return Ok(WeightLayout::Safetensors {
            files: safetensors.into_iter().map(PathBuf::from).collect(),
        });
    }
    if safetensors.len() > 1 {
        return Err(DetectionError::AmbiguousWeights {
            candidates: safetensors,
        });
    }

    let gguf = available.with_suffix(".gguf");
    if gguf.len() == 1 {
        return Ok(WeightLayout::Gguf {
            file: PathBuf::from(&gguf[0]),
        });
    }
    if gguf.len() > 1 {
        return Err(DetectionError::AmbiguousWeights { candidates: gguf });
    }

    let pytorch: Vec<String> = available
        .with_suffix(".pth")
        .into_iter()
        .chain(available.with_suffix(".bin"))
        .collect();
    if pytorch.len() == 1 {
        return Ok(WeightLayout::Pytorch {
            file: PathBuf::from(&pytorch[0]),
        });
    }
    if pytorch.len() > 1 {
        return Err(DetectionError::AmbiguousWeights {
            candidates: pytorch,
        });
    }

    let numpy = available.with_suffix(".npz");
    if numpy.len() == 1 {
        return Ok(WeightLayout::Numpy {
            file: PathBuf::from(&numpy[0]),
        });
    }
    if numpy.len() > 1 {
        return Err(DetectionError::AmbiguousWeights { candidates: numpy });
    }

    Err(DetectionError::NoWeights)
}

/// Classifies a weight file by its extension (used for explicit selection).
fn layout_from_extension(file: &Path) -> WeightLayout {
    match file
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("gguf") => WeightLayout::Gguf {
            file: file.to_path_buf(),
        },
        Some("pth") | Some("bin") => WeightLayout::Pytorch {
            file: file.to_path_buf(),
        },
        Some("npz") => WeightLayout::Numpy {
            file: file.to_path_buf(),
        },
        _ => WeightLayout::Safetensors {
            files: vec![file.to_path_buf()],
        },
    }
}

/// Parses `model.safetensors.index.json` into the ordered shard file names.
///
/// The index maps every tensor name to the shard file that holds it; the
/// distinct file names are returned sorted so loading is deterministic.
pub fn shard_files_from_index(index_json: &str) -> Result<Vec<String>, DetectionError> {
    let parsed: serde_json::Value =
        serde_json::from_str(index_json).map_err(|error| DetectionError::InvalidIndex {
            reason: error.to_string(),
        })?;
    let mapping = parsed
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| DetectionError::InvalidIndex {
            reason: "the index has no 'weight_map' object".to_string(),
        })?;
    let mut shards: Vec<String> = mapping
        .values()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect();
    shards.sort();
    shards.dedup();
    if shards.is_empty() {
        return Err(DetectionError::InvalidIndex {
            reason: "the 'weight_map' references no shard file".to_string(),
        });
    }
    Ok(shards)
}

/// Infers the weight kind from a safetensors dtype map (tensor name -> dtype).
///
/// Only full-precision dtypes are loadable by candle 0.11; FP8/compressed
/// checkpoints are detected so the caller can produce an actionable error.
pub fn weight_kind_from_safetensors_dtypes(tensor_dtypes: &HashMap<String, String>) -> WeightKind {
    let mut has_float8 = false;
    let mut has_dense = false;
    for dtype in tensor_dtypes.values() {
        if dtype.starts_with("F8") || dtype.contains("FP8") {
            has_float8 = true;
        } else if matches!(dtype.as_str(), "F32" | "F16" | "BF16" | "F64") {
            has_dense = true;
        }
    }
    if has_float8 {
        WeightKind::UnsupportedFloat8
    } else if has_dense {
        WeightKind::Dense
    } else {
        // No floating-point tensor at all: treat as dense and let the loader
        // report the concrete dtype mismatch.
        WeightKind::Dense
    }
}

/// Infers the weight kind from GGML layer dtypes (`true` when quantized).
///
/// The input lists one GGML dtype per weight tensor; a single value above the
/// dense set (`F32`/`F16`/`BF16`) marks the whole checkpoint as quantized.
/// Kept for the GGUF inspection path and exercised by unit tests.
#[allow(dead_code)]
pub fn weight_kind_from_ggml_dtypes(layer_dtypes: &[String]) -> WeightKind {
    let quantized = layer_dtypes.iter().any(|dtype| {
        dtype.starts_with('Q') || dtype.starts_with("MXFP") || dtype.starts_with("IQ")
    });
    if quantized {
        WeightKind::Quantized
    } else {
        WeightKind::Dense
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(names: &[&str]) -> AvailableFiles {
        AvailableFiles::new(names.iter().map(|name| name.to_string()).collect())
    }

    #[test]
    fn hub_reference_is_used_when_path_is_absent() {
        let reference = ModelReference::resolve("Qwen/Qwen2.5-1.5B-Instruct");
        assert_eq!(
            reference,
            ModelReference::Hub {
                repository: "Qwen/Qwen2.5-1.5B-Instruct".to_string()
            }
        );
        assert!(reference.describe().starts_with("hub:"));
    }

    #[test]
    fn local_reference_is_used_when_path_exists() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let raw = directory.path().to_string_lossy().to_string();
        let reference = ModelReference::resolve(&raw);
        match reference {
            ModelReference::Local { path } => assert_eq!(path, directory.path()),
            other => panic!("expected a local reference, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn single_safetensors_is_detected() -> anyhow::Result<()> {
        let available = files(&["config.json", "tokenizer.json", "model.safetensors"]);
        let layout = detect_weight_layout(&available, None).map_err(to_error)?;
        assert_eq!(
            layout,
            WeightLayout::Safetensors {
                files: vec![PathBuf::from("model.safetensors")]
            }
        );
        Ok(())
    }

    #[test]
    fn named_single_safetensors_is_detected() -> anyhow::Result<()> {
        let available = files(&["config.json", "tokenizer.json", "qwen-model.safetensors"]);
        let layout = detect_weight_layout(&available, None).map_err(to_error)?;
        assert_eq!(layout.name(), "safetensors");
        Ok(())
    }

    #[test]
    fn gguf_is_detected() -> anyhow::Result<()> {
        let available = files(&["qwen2.5-1.5b-q4_k_m.gguf", "tokenizer.json"]);
        let layout = detect_weight_layout(&available, None).map_err(to_error)?;
        assert_eq!(
            layout,
            WeightLayout::Gguf {
                file: PathBuf::from("qwen2.5-1.5b-q4_k_m.gguf")
            }
        );
        Ok(())
    }

    #[test]
    fn pytorch_and_numpy_are_detected() -> anyhow::Result<()> {
        let pytorch =
            detect_weight_layout(&files(&["pytorch_model.bin"]), None).map_err(to_error)?;
        assert_eq!(pytorch.name(), "pytorch");
        let numpy = detect_weight_layout(&files(&["weights.npz"]), None).map_err(to_error)?;
        assert_eq!(numpy.name(), "numpy");
        Ok(())
    }

    #[test]
    fn several_gguf_files_require_an_explicit_choice() {
        let available = files(&["model-q4_k_m.gguf", "model-q8_0.gguf"]);
        let result = detect_weight_layout(&available, None);
        assert!(result.is_err());
        let Err(DetectionError::AmbiguousWeights { candidates }) = result else {
            return;
        };
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn explicit_file_overrides_detection() -> anyhow::Result<()> {
        let available = files(&["model-q4_k_m.gguf", "model-q8_0.gguf"]);
        let layout =
            detect_weight_layout(&available, Some(Path::new("chosen.gguf"))).map_err(to_error)?;
        assert_eq!(
            layout,
            WeightLayout::Gguf {
                file: PathBuf::from("chosen.gguf")
            }
        );
        Ok(())
    }

    #[test]
    fn no_weights_is_an_error() {
        assert!(matches!(
            detect_weight_layout(&files(&["README.md", "config.json"]), None),
            Err(DetectionError::NoWeights)
        ));
    }

    #[test]
    fn shard_index_is_parsed_into_sorted_distinct_files() -> anyhow::Result<()> {
        let index = serde_json::json!({
            "metadata": {"total_size": 123},
            "weight_map": {
                "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.norm.weight": "model-00002-of-00002.safetensors"
            }
        });
        let shards = shard_files_from_index(&index.to_string()).map_err(to_error)?;
        assert_eq!(
            shards,
            vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string()
            ]
        );
        Ok(())
    }

    #[test]
    fn malformed_shard_index_reports_a_reason() {
        let result = shard_files_from_index("{ not json");
        assert!(result.is_err());
        let Err(DetectionError::InvalidIndex { reason }) = result else {
            return;
        };
        assert!(!reason.is_empty());
    }

    #[test]
    fn architecture_accepts_llama_and_qwen2_only() {
        assert_eq!(
            ModelArchitecture::from_model_type("llama"),
            Some(ModelArchitecture::Llama)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("Qwen2"),
            Some(ModelArchitecture::Qwen2)
        );
        assert_eq!(ModelArchitecture::from_model_type("qwen3"), None);
        assert!(!ModelArchitecture::Llama.has_query_key_value_bias());
        assert!(ModelArchitecture::Qwen2.has_query_key_value_bias());
    }

    #[test]
    fn float8_dtypes_are_flagged_unsupported() {
        let mut dtypes = HashMap::new();
        dtypes.insert("lm_head.weight".to_string(), "BF16".to_string());
        dtypes.insert("layer.weight".to_string(), "F8_E4M3".to_string());
        assert_eq!(
            weight_kind_from_safetensors_dtypes(&dtypes),
            WeightKind::UnsupportedFloat8
        );
    }

    #[test]
    fn dense_dtypes_are_flagged_supported() {
        let mut dtypes = HashMap::new();
        dtypes.insert("embed.weight".to_string(), "BF16".to_string());
        dtypes.insert("norm.weight".to_string(), "F32".to_string());
        assert_eq!(
            weight_kind_from_safetensors_dtypes(&dtypes),
            WeightKind::Dense
        );
    }

    #[test]
    fn ggml_quantized_dtypes_are_detected() {
        let dtypes = vec!["Q4_K".to_string(), "F32".to_string()];
        assert_eq!(weight_kind_from_ggml_dtypes(&dtypes), WeightKind::Quantized);
        let dense = vec!["F16".to_string(), "F32".to_string()];
        assert_eq!(weight_kind_from_ggml_dtypes(&dense), WeightKind::Dense);
    }

    /// Converts the displayable detection error into `anyhow` for `?` usage.
    fn to_error(error: DetectionError) -> anyhow::Error {
        anyhow::anyhow!(error)
    }
}
