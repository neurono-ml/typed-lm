//! Concrete checkpoint resolution: turns a [`ModelReference`] into the files
//! needed to load a model, downloading from the Hugging Face Hub when required.
//!
//! Two resolvers share the same output ([`ResolvedCheckpoint`]):
//!
//! - [`HubCheckpointResolver`] inspects a repository's file list (via the Hub
//!   API), downloads `config.json`/`tokenizer.json` when needed and, for
//!   sharded safetensors, reads the shard index to learn the shard names.
//! - [`LocalCheckpointResolver`] reads a directory (or a single weight file)
//!   from disk, using an explicitly provided config/tokenizer when the folder
//!   does not carry them.
//!
//! The layout/architecture/kind *detection* itself lives in
//! [`crate::infrastructure::checkpoint`] as pure functions; this module only
//! provides the concrete file access those functions expect.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::infrastructure::checkpoint::{
    detect_weight_layout, shard_files_from_index, weight_kind_from_safetensors_dtypes,
    AvailableFiles, DetectionError, ModelArchitecture, ModelReference, ResolvedCheckpoint,
    WeightKind, WeightLayout, CONFIG_NAME, SAFETENSORS_INDEX_NAME, TOKENIZER_NAME,
};
use crate::infrastructure::model_repository::ModelRepository;

/// The fully resolved checkpoint plus its inferred metadata.
#[derive(Debug, Clone)]
pub struct LoadableCheckpoint {
    pub resolved: ResolvedCheckpoint,
    pub architecture: ModelArchitecture,
    pub weight_kind: WeightKind,
}

/// Resolves a Hub repository into a loadable checkpoint.
pub struct HubCheckpointResolver {
    repository: ModelRepository,
    revision: String,
    tokenizer_override: Option<PathBuf>,
}

impl HubCheckpointResolver {
    pub fn new(model_identifier: &str, hugging_face_token: Option<String>) -> anyhow::Result<Self> {
        Ok(Self {
            repository: ModelRepository::new(model_identifier, hugging_face_token)?,
            revision: "main".to_string(),
            tokenizer_override: None,
        })
    }

    /// Pins the resolver to a specific Hub revision (branch, tag or commit).
    pub fn with_revision(mut self, revision: &str) -> Self {
        self.revision = revision.to_string();
        self
    }

    /// Uses an explicit local `tokenizer.json` instead of the repository one.
    ///
    /// GGUF repositories often omit `tokenizer.json` (the vocabulary is inside
    /// the GGUF, which candle does not read), so the tokenizer is taken from the
    /// corresponding dense repository or a local file.
    pub fn with_tokenizer(mut self, tokenizer_file: Option<PathBuf>) -> Self {
        self.tokenizer_override = tokenizer_file;
        self
    }

    /// Downloads/records every file required by the detected layout.
    pub fn resolve(
        &self,
        reference: &ModelReference,
        explicit_weights_file: Option<&Path>,
    ) -> anyhow::Result<LoadableCheckpoint> {
        // Re-pin the repository to the requested revision before touching files.
        let repository = ModelRepository::with_revision(
            self.repository.model_identifier(),
            &self.revision,
            self.repository.hugging_face_token().map(str::to_string),
        )?;
        repository
            .ensure_servable_checkpoint()
            .map_err(|error| anyhow::anyhow!("checkpoint preflight failed: {error}"))?;
        let available = AvailableFiles::new(repository.available_file_names()?);

        // `config.json` is required for dense layouts; GGUF repositories omit
        // it because parameters come from the GGUF metadata.
        let has_config = available.has(CONFIG_NAME);
        let config_file = if has_config {
            repository
                .download(CONFIG_NAME)
                .map_err(|error| anyhow::anyhow!("failed to fetch {CONFIG_NAME}: {error}"))?
        } else {
            PathBuf::new()
        };
        let tokenizer_file = match &self.tokenizer_override {
            Some(path) if path.exists() => path.clone(),
            Some(path) => {
                return Err(anyhow::anyhow!(
                    "tokenizer override '{}' does not exist",
                    path.display()
                ))
            }
            None => repository
                .download(TOKENIZER_NAME)
                .map_err(|error| anyhow::anyhow!("failed to fetch {TOKENIZER_NAME}: {error}"))?,
        };
        let architecture = if has_config {
            read_architecture_from_config(&config_file)?
        } else {
            ModelArchitecture::Qwen2
        };

        let has_shard_index = available.has(SAFETENSORS_INDEX_NAME);
        let layout = match explicit_weights_file {
            Some(file) => {
                let detected = detect_weight_layout(&available, Some(file)).map_err(to_anyhow)?;
                download_layout(&repository, detected)?
            }
            None if has_shard_index => {
                let index_path = repository.download(SAFETENSORS_INDEX_NAME)?;
                let index_json = std::fs::read_to_string(&index_path)?;
                let shards = shard_files_from_index(&index_json).map_err(to_anyhow)?;
                let mut files = Vec::with_capacity(shards.len());
                for shard in &shards {
                    files.push(repository.download(shard)?);
                }
                WeightLayout::Safetensors { files }
            }
            None => {
                // The detected layout carries file *names*; download each of
                // them so the loader receives real cache paths.
                let detected = detect_weight_layout(&available, None).map_err(to_anyhow)?;
                download_layout(&repository, detected)?
            }
        };

        let weight_kind = self.detect_weight_kind(&layout)?;
        Ok(LoadableCheckpoint {
            resolved: ResolvedCheckpoint {
                reference: reference.clone(),
                config_file,
                tokenizer_file,
                layout,
            },
            architecture,
            weight_kind,
        })
    }

    /// Infers the weight kind from the safetensors headers (dense) or, for a
    /// GGUF file, from its GGML tensor dtypes.
    fn detect_weight_kind(&self, layout: &WeightLayout) -> anyhow::Result<WeightKind> {
        match layout {
            WeightLayout::Safetensors { files } => {
                let mut dtypes: HashMap<String, String> = HashMap::new();
                for file in files {
                    for (name, dtype) in read_safetensors_dtypes(file)? {
                        dtypes.insert(name, dtype);
                    }
                }
                Ok(weight_kind_from_safetensors_dtypes(&dtypes))
            }
            WeightLayout::Gguf { .. } => {
                // GGUF dtype inspection happens when the file is opened by the
                // quantized loader; treat it as quantized-capable for now and
                // let the loader refine it (dense GGUF is also accepted).
                Ok(WeightKind::Quantized)
            }
            WeightLayout::Pytorch { .. } | WeightLayout::Numpy { .. } => Ok(WeightKind::Dense),
        }
    }
}

/// Resolves a local path into a loadable checkpoint.
pub struct LocalCheckpointResolver {
    path: PathBuf,
    config_override: Option<PathBuf>,
    tokenizer_override: Option<PathBuf>,
}

impl LocalCheckpointResolver {
    pub fn new(
        path: PathBuf,
        config_override: Option<PathBuf>,
        tokenizer_override: Option<PathBuf>,
    ) -> Self {
        Self {
            path,
            config_override,
            tokenizer_override,
        }
    }

    /// Resolves the local checkpoint, discovering the layout from disk.
    pub fn resolve(
        &self,
        reference: &ModelReference,
        explicit_weights_file: Option<&Path>,
    ) -> anyhow::Result<LoadableCheckpoint> {
        let (directory, available_names, explicit_from_path) = self.enumerate()?;
        let available = AvailableFiles::new(available_names);

        let chosen_weights = match explicit_weights_file {
            Some(file) => Some(file.to_path_buf()),
            None => explicit_from_path,
        };

        let layout = self.resolve_layout(&directory, &available, chosen_weights.as_deref())?;
        let is_gguf = matches!(layout, WeightLayout::Gguf { .. });
        // GGUF checkpoints may omit config.json (parameters come from metadata).
        let config_file = match (is_gguf, &self.config_override) {
            (true, None) => PathBuf::new(),
            _ => self.resolve_companion(&directory, &self.config_override, CONFIG_NAME)?,
        };
        let tokenizer_file =
            self.resolve_companion(&directory, &self.tokenizer_override, TOKENIZER_NAME)?;
        let architecture = if config_file.exists() {
            read_architecture_from_config(&config_file)?
        } else {
            ModelArchitecture::Qwen2
        };
        let weight_kind = match &layout {
            WeightLayout::Safetensors { files } => {
                let mut dtypes: HashMap<String, String> = HashMap::new();
                for file in files {
                    for (name, dtype) in read_safetensors_dtypes(file)? {
                        dtypes.insert(name, dtype);
                    }
                }
                weight_kind_from_safetensors_dtypes(&dtypes)
            }
            WeightLayout::Gguf { .. } => WeightKind::Quantized,
            WeightLayout::Pytorch { .. } | WeightLayout::Numpy { .. } => WeightKind::Dense,
        };

        Ok(LoadableCheckpoint {
            resolved: ResolvedCheckpoint {
                reference: reference.clone(),
                config_file,
                tokenizer_file,
                layout,
            },
            architecture,
            weight_kind,
        })
    }

    /// Returns the base directory, its file names and a single weight file
    /// when the provided path is itself a file.
    fn enumerate(&self) -> anyhow::Result<(PathBuf, Vec<String>, Option<PathBuf>)> {
        if self.path.is_file() {
            let directory = self
                .path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."));
            let explicit = Some(self.path.clone());
            let names = list_directory_names(&directory)?;
            Ok((directory, names, explicit))
        } else if self.path.is_dir() {
            Ok((self.path.clone(), list_directory_names(&self.path)?, None))
        } else {
            Err(anyhow::anyhow!(
                "local model path '{}' does not exist",
                self.path.display()
            ))
        }
    }

    /// Builds the layout, expanding a shard index when present.
    ///
    /// Relative weight paths returned by detection are joined with `directory`
    /// so the loader receives absolute paths.
    fn resolve_layout(
        &self,
        directory: &Path,
        available: &AvailableFiles,
        explicit_weights: Option<&Path>,
    ) -> anyhow::Result<WeightLayout> {
        if let Some(file) = explicit_weights {
            return Ok(absolutize_layout(
                detect_weight_layout(available, Some(file)).map_err(to_anyhow)?,
                directory,
            ));
        }
        if available
            .names
            .iter()
            .any(|name| name == SAFETENSORS_INDEX_NAME)
        {
            let index_path = directory.join(SAFETENSORS_INDEX_NAME);
            let index_json = std::fs::read_to_string(&index_path)?;
            let shards = shard_files_from_index(&index_json).map_err(to_anyhow)?;
            let files = shards.iter().map(|shard| directory.join(shard)).collect();
            return Ok(WeightLayout::Safetensors { files });
        }
        Ok(absolutize_layout(
            detect_weight_layout(available, None).map_err(to_anyhow)?,
            directory,
        ))
    }

    /// Resolves a companion file (config/tokenizer) from an override or the
    /// checkpoint directory.
    fn resolve_companion(
        &self,
        directory: &Path,
        override_path: &Option<PathBuf>,
        default_name: &str,
    ) -> anyhow::Result<PathBuf> {
        if let Some(path) = override_path {
            if !path.exists() {
                return Err(anyhow::anyhow!(
                    "override file '{}' does not exist",
                    path.display()
                ));
            }
            return Ok(path.clone());
        }
        let candidate = directory.join(default_name);
        if candidate.exists() {
            Ok(candidate)
        } else if default_name == CONFIG_NAME {
            Err(anyhow::anyhow!(DetectionError::MissingConfig))
        } else {
            Err(anyhow::anyhow!(DetectionError::MissingTokenizer))
        }
    }
}

/// Reads `model_type` from a `config.json` file.
pub fn read_architecture_from_config(config_file: &Path) -> anyhow::Result<ModelArchitecture> {
    let bytes = std::fs::read(config_file)?;
    let parsed: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| anyhow::anyhow!("failed to parse '{}': {error}", config_file.display()))?;
    let model_type = parsed
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("'{}' has no 'model_type' field", config_file.display()))?;
    ModelArchitecture::from_model_type(model_type).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported model_type '{model_type}' in '{}': only llama and qwen2 are supported",
            config_file.display()
        )
    })
}

/// Lists the file names directly under a directory (sorted for determinism).
fn list_directory_names(directory: &Path) -> anyhow::Result<Vec<String>> {
    let mut names: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            if let Some(name) = entry.file_name().to_str() {
                names.push(name.to_string());
            }
        }
    }
    names.sort();
    Ok(names)
}

/// Reads the safetensors header of a file into a tensor-name -> dtype map.
///
/// Only the JSON header is parsed (first 8 bytes hold its length as a
/// little-endian `u64`), so no tensor data is loaded.
pub fn read_safetensors_dtypes(file: &Path) -> anyhow::Result<HashMap<String, String>> {
    use std::io::Read as _;
    let mut handle = std::fs::File::open(file)?;
    let mut length_bytes = [0_u8; 8];
    handle.read_exact(&mut length_bytes)?;
    let header_length = u64::from_le_bytes(length_bytes);
    const MAXIMUM_HEADER_BYTES: u64 = 128 * 1024 * 1024;
    if header_length > MAXIMUM_HEADER_BYTES {
        return Err(anyhow::anyhow!(
            "refusing to parse safetensors header of {} bytes in '{}': file looks corrupted",
            header_length,
            file.display()
        ));
    }
    let mut header_bytes = vec![0_u8; header_length as usize];
    handle.read_exact(&mut header_bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes).map_err(|error| {
        anyhow::anyhow!(
            "failed to parse safetensors header of '{}': {error}",
            file.display()
        )
    })?;
    let mapping = header.as_object().ok_or_else(|| {
        anyhow::anyhow!(
            "unexpected safetensors header in '{}': top level is not an object",
            file.display()
        )
    })?;
    let mut dtypes = HashMap::new();
    for (tensor_name, descriptor) in mapping {
        if tensor_name == "__metadata__" {
            continue;
        }
        if let Some(dtype) = descriptor.get("dtype").and_then(serde_json::Value::as_str) {
            dtypes.insert(tensor_name.clone(), dtype.to_string());
        }
    }
    Ok(dtypes)
}

/// Converts a [`DetectionError`] into `anyhow` for `?`-based propagation.
fn to_anyhow(error: DetectionError) -> anyhow::Error {
    anyhow::anyhow!(error)
}

/// Downloads every weight file named by a detected layout, returning the same
/// layout with concrete cache paths.
fn download_layout(
    repository: &ModelRepository,
    layout: WeightLayout,
) -> anyhow::Result<WeightLayout> {
    let fetch = |file: PathBuf| -> anyhow::Result<PathBuf> {
        let name = file
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("weight file name is not valid UTF-8: {file:?}"))?;
        repository.download(name)
    };
    match layout {
        WeightLayout::Safetensors { files } => {
            let mut downloaded = Vec::with_capacity(files.len());
            for file in files {
                downloaded.push(fetch(file)?);
            }
            Ok(WeightLayout::Safetensors { files: downloaded })
        }
        WeightLayout::Gguf { file } => Ok(WeightLayout::Gguf { file: fetch(file)? }),
        WeightLayout::Pytorch { file } => Ok(WeightLayout::Pytorch { file: fetch(file)? }),
        WeightLayout::Numpy { file } => Ok(WeightLayout::Numpy { file: fetch(file)? }),
    }
}

/// Joins relative weight paths in a layout with the checkpoint directory.
fn absolutize_layout(layout: WeightLayout, directory: &Path) -> WeightLayout {
    let join = |file: PathBuf| {
        if file.is_absolute() {
            file
        } else {
            directory.join(file)
        }
    };
    match layout {
        WeightLayout::Safetensors { files } => WeightLayout::Safetensors {
            files: files.into_iter().map(join).collect(),
        },
        WeightLayout::Gguf { file } => WeightLayout::Gguf { file: join(file) },
        WeightLayout::Pytorch { file } => WeightLayout::Pytorch { file: join(file) },
        WeightLayout::Numpy { file } => WeightLayout::Numpy { file: join(file) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::checkpoint::SINGLE_SAFETENSORS_NAME;
    use std::io::Write as _;

    /// Writes a minimal safetensors file with the given dtype map.
    fn write_safetensors(path: &Path, dtypes: &[(&str, &str)]) -> anyhow::Result<()> {
        let mut header = serde_json::Map::new();
        let mut offset = 0_u64;
        for (name, dtype) in dtypes {
            header.insert(
                (*name).to_string(),
                serde_json::json!({"dtype": dtype, "shape": [4], "data_offsets": [offset, offset + 4]}),
            );
            offset += 4;
        }
        let header_bytes = serde_json::to_vec(&header)?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        file.write_all(&header_bytes)?;
        file.write_all(&vec![0_u8; offset as usize])?;
        Ok(())
    }

    #[test]
    fn architecture_is_read_from_config_json() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let config = directory.path().join(CONFIG_NAME);
        std::fs::write(&config, br#"{"model_type": "qwen2", "hidden_size": 896}"#)?;
        assert_eq!(
            read_architecture_from_config(&config)?,
            ModelArchitecture::Qwen2
        );
        Ok(())
    }

    #[test]
    fn unknown_architecture_is_rejected() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let config = directory.path().join(CONFIG_NAME);
        std::fs::write(&config, br#"{"model_type": "qwen3"}"#)?;
        let result = read_architecture_from_config(&config);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn safetensors_dtypes_are_read_without_tensor_data() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let weights = directory.path().join(SINGLE_SAFETENSORS_NAME);
        write_safetensors(
            &weights,
            &[("embed.weight", "BF16"), ("layer.weight", "F32")],
        )?;
        let dtypes = read_safetensors_dtypes(&weights)?;
        assert_eq!(dtypes.get("embed.weight").map(String::as_str), Some("BF16"));
        assert_eq!(dtypes.get("layer.weight").map(String::as_str), Some("F32"));
        Ok(())
    }

    #[test]
    fn local_directory_resolves_single_safetensors() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        std::fs::write(
            directory.path().join(CONFIG_NAME),
            br#"{"model_type": "llama"}"#,
        )?;
        std::fs::write(directory.path().join(TOKENIZER_NAME), b"{}")?;
        write_safetensors(
            &directory.path().join(SINGLE_SAFETENSORS_NAME),
            &[("embed.weight", "BF16")],
        )?;
        let reference = ModelReference::Local {
            path: directory.path().to_path_buf(),
        };
        let resolver = LocalCheckpointResolver::new(directory.path().to_path_buf(), None, None);
        let checkpoint = resolver.resolve(&reference, None)?;
        assert_eq!(checkpoint.architecture, ModelArchitecture::Llama);
        assert_eq!(checkpoint.weight_kind, WeightKind::Dense);
        assert_eq!(checkpoint.resolved.layout.name(), "safetensors");
        Ok(())
    }

    #[test]
    fn local_directory_expands_shard_index() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        std::fs::write(
            directory.path().join(CONFIG_NAME),
            br#"{"model_type": "qwen2"}"#,
        )?;
        std::fs::write(directory.path().join(TOKENIZER_NAME), b"{}")?;
        let shard_a = "model-00001-of-00002.safetensors";
        let shard_b = "model-00002-of-00002.safetensors";
        write_safetensors(&directory.path().join(shard_a), &[("a", "BF16")])?;
        write_safetensors(&directory.path().join(shard_b), &[("b", "BF16")])?;
        let index = serde_json::json!({
            "weight_map": {"a": shard_a, "b": shard_b}
        });
        std::fs::write(
            directory.path().join(SAFETENSORS_INDEX_NAME),
            index.to_string(),
        )?;
        let reference = ModelReference::Local {
            path: directory.path().to_path_buf(),
        };
        let resolver = LocalCheckpointResolver::new(directory.path().to_path_buf(), None, None);
        let checkpoint = resolver.resolve(&reference, None)?;
        match checkpoint.resolved.layout {
            WeightLayout::Safetensors { files } => assert_eq!(files.len(), 2),
            other => panic!("expected sharded safetensors, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn local_directory_without_config_reports_missing_config() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join(TOKENIZER_NAME), b"{}")?;
        write_safetensors(
            &directory.path().join(SINGLE_SAFETENSORS_NAME),
            &[("embed.weight", "BF16")],
        )?;
        let reference = ModelReference::Local {
            path: directory.path().to_path_buf(),
        };
        let resolver = LocalCheckpointResolver::new(directory.path().to_path_buf(), None, None);
        let result = resolver.resolve(&reference, None);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn local_directory_without_tokenizer_reports_missing_tokenizer() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        std::fs::write(
            directory.path().join(CONFIG_NAME),
            br#"{"model_type": "llama"}"#,
        )?;
        write_safetensors(
            &directory.path().join(SINGLE_SAFETENSORS_NAME),
            &[("embed.weight", "BF16")],
        )?;
        let reference = ModelReference::Local {
            path: directory.path().to_path_buf(),
        };
        let resolver = LocalCheckpointResolver::new(directory.path().to_path_buf(), None, None);
        let result = resolver.resolve(&reference, None);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn local_single_gguf_file_is_resolved() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        std::fs::write(
            directory.path().join(CONFIG_NAME),
            br#"{"model_type": "qwen2"}"#,
        )?;
        std::fs::write(directory.path().join(TOKENIZER_NAME), b"{}")?;
        let gguf = directory.path().join("model-q4_k_m.gguf");
        std::fs::write(&gguf, b"GGUF")?;
        let reference = ModelReference::Local { path: gguf.clone() };
        let resolver = LocalCheckpointResolver::new(gguf.clone(), None, None);
        let checkpoint = resolver.resolve(&reference, None)?;
        assert_eq!(checkpoint.weight_kind, WeightKind::Quantized);
        match checkpoint.resolved.layout {
            WeightLayout::Gguf { file } => assert_eq!(
                file.file_name().and_then(std::ffi::OsStr::to_str),
                Some("model-q4_k_m.gguf")
            ),
            other => panic!("expected gguf, got {other:?}"),
        }
        Ok(())
    }
}
