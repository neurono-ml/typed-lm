use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::infrastructure::parallel_llama::{ParallelCache, ParallelLlama};
use crate::infrastructure::parallel_quantized_qwen2::{ParallelQuantizedQwen2, QuantizedCache};
use typed_lm_common::checkpoint::{ModelArchitecture, WeightKind};
use typed_lm_common::checkpoint_resolver::LoadableCheckpoint;
use typed_lm_common::model_config::ParallelModelConfig;
use typed_lm_common::prompt_template::PromptTemplate;
use typed_lm_common::quantization::{dequantize_checkpoint_tensors, QuantizationScheme};

/// Architecture-agnostic key/value cache handle.
///
/// Dense checkpoints use [`ParallelCache`]; GGUF-quantized checkpoints use the
/// quantized [`QuantizedCache`]. Both expose `broadcast_batch`, so the
/// evaluator stays generic over the storage strategy.
#[derive(Debug, Clone)]
pub enum ModelCache {
    Dense(ParallelCache),
    Quantized(QuantizedCache),
}

impl ModelCache {
    /// Replicates the cached key/value tensors across `batch_size` rows.
    pub fn broadcast_batch(&mut self, batch_size: usize) -> anyhow::Result<()> {
        match self {
            Self::Dense(cache) => cache.broadcast_batch(batch_size)?,
            Self::Quantized(cache) => cache.broadcast_batch(batch_size)?,
        }
        Ok(())
    }
}

/// The loaded model weights, either dense or GGML-quantized.
#[derive(Debug, Clone)]
enum ParallelModel {
    Dense(ParallelLlama),
    Quantized(ParallelQuantizedQwen2),
}

impl ParallelModel {
    fn forward_last(
        &self,
        tokens: &Tensor,
        index_position: usize,
        cache: &mut ModelCache,
    ) -> anyhow::Result<Tensor> {
        match (self, cache) {
            (Self::Dense(model), ModelCache::Dense(cache)) => {
                Ok(model.forward_last(tokens, index_position, cache)?)
            }
            (Self::Quantized(model), ModelCache::Quantized(cache)) => {
                Ok(model.forward_last(tokens, index_position, cache)?)
            }
            _ => Err(anyhow::anyhow!(
                "cache kind does not match the loaded model kind"
            )),
        }
    }

    fn forward_hidden_all(
        &self,
        tokens: &Tensor,
        index_position: usize,
        cache: &mut ModelCache,
    ) -> anyhow::Result<Tensor> {
        match (self, cache) {
            (Self::Dense(model), ModelCache::Dense(cache)) => {
                Ok(model.forward_hidden_all(tokens, index_position, cache)?)
            }
            (Self::Quantized(model), ModelCache::Quantized(cache)) => {
                Ok(model.forward_hidden_all(tokens, index_position, cache)?)
            }
            _ => Err(anyhow::anyhow!(
                "cache kind does not match the loaded model kind"
            )),
        }
    }

    fn logits_from_hidden_at_positions(
        &self,
        hidden: &Tensor,
        positions: &[usize],
    ) -> anyhow::Result<Tensor> {
        match self {
            Self::Dense(model) => Ok(model.logits_from_hidden_at_positions(hidden, positions)?),
            Self::Quantized(model) => Ok(model.logits_from_hidden_at_positions(hidden, positions)?),
        }
    }
}

/// Loads weights, config and tokenizer, exposing the model ready for inference.
///
/// The model is architecture-aware (Llama or Qwen2) and format-aware (dense
/// safetensors/pth/npz or GGUF quantized). Detection is done by
/// [`typed_lm_common::checkpoint_resolver`]; this type consumes the
/// resolved checkpoint and builds the vendored parallel forward pass so the
/// key/value cache can be broadcast across the attention batch dimension.
pub struct LanguageModel {
    model: ParallelModel,
    config: ParallelModelConfig,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    prompt_template: PromptTemplate,
    maximum_sequence_length: usize,
}

impl LanguageModel {
    /// Loads a model from a resolved checkpoint.
    pub fn load(
        checkpoint: &LoadableCheckpoint,
        device: &Device,
        dtype: DType,
    ) -> anyhow::Result<Self> {
        let architecture = checkpoint.architecture;
        let prompt_template = PromptTemplate::for_architecture(architecture);
        let tokenizer = Tokenizer::from_file(&checkpoint.resolved.tokenizer_file)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;

        match &checkpoint.resolved.layout {
            typed_lm_common::checkpoint::WeightLayout::Safetensors { files } => {
                let config = read_config(&checkpoint.resolved.config_file, architecture)?;
                let variable_builder = if checkpoint.weight_kind.is_low_precision_float() {
                    Self::low_precision_variable_builder(
                        files,
                        scheme_for_weight_kind(checkpoint.weight_kind),
                        device,
                        dtype,
                    )?
                } else {
                    unsafe { VarBuilder::from_mmaped_safetensors(files.as_slice(), dtype, device)? }
                };
                let model = ParallelLlama::load(variable_builder, &config)?;
                Ok(Self::assemble(
                    ParallelModel::Dense(model),
                    config,
                    tokenizer,
                    device,
                    dtype,
                    prompt_template,
                ))
            }
            typed_lm_common::checkpoint::WeightLayout::Gguf { file } => {
                let model = ParallelQuantizedQwen2::from_gguf(file, device)?;
                // GGUF metadata carries the architecture parameters; the tokenizer
                // is loaded from the companion `tokenizer.json`.
                Self::load_quantized(
                    model,
                    file,
                    &checkpoint.resolved.config_file,
                    tokenizer,
                    device,
                    dtype,
                    prompt_template,
                    architecture,
                )
            }
            typed_lm_common::checkpoint::WeightLayout::Pytorch { file } => {
                let config = read_config(&checkpoint.resolved.config_file, architecture)?;
                let variable_builder = VarBuilder::from_pth(file, dtype, device)?;
                let model = ParallelLlama::load(variable_builder, &config)?;
                Ok(Self::assemble(
                    ParallelModel::Dense(model),
                    config,
                    tokenizer,
                    device,
                    dtype,
                    prompt_template,
                ))
            }
            typed_lm_common::checkpoint::WeightLayout::Numpy { file } => {
                let config = read_config(&checkpoint.resolved.config_file, architecture)?;
                let variable_builder = VarBuilder::from_npz(file, dtype, device)?;
                let model = ParallelLlama::load(variable_builder, &config)?;
                Ok(Self::assemble(
                    ParallelModel::Dense(model),
                    config,
                    tokenizer,
                    device,
                    dtype,
                    prompt_template,
                ))
            }
        }
    }

    /// Builds a variable builder from a low-precision (FP8/FP4) safetensors
    /// checkpoint by dequantizing every weight to dense F32 first.
    ///
    /// Candle 0.11 can store `F8_E4M3`/`F4` tensors but has no matmul kernel for
    /// them, so the forward pass cannot consume them directly. Each safetensors
    /// file is read into memory, the low-precision weights are dequantized with
    /// the shared [`dequantize_checkpoint_tensors`] routine (which consumes the
    /// paired scale tensors), and the resulting dense map is handed to
    /// [`VarBuilder::from_tensors`].
    fn low_precision_variable_builder(
        files: &[PathBuf],
        scheme: QuantizationScheme,
        device: &Device,
        dtype: DType,
    ) -> anyhow::Result<VarBuilder<'static>> {
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for file in files {
            for (name, tensor) in candle_core::safetensors::load(file, device)? {
                tensors.insert(name, tensor);
            }
        }
        let dense_weights = dequantize_checkpoint_tensors(tensors, scheme)?;
        Ok(VarBuilder::from_tensors(dense_weights, dtype, device))
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        model: ParallelModel,
        config: ParallelModelConfig,
        tokenizer: Tokenizer,
        device: &Device,
        dtype: DType,
        prompt_template: PromptTemplate,
    ) -> Self {
        let maximum_sequence_length = config.max_position_embeddings;
        Self {
            model,
            config,
            tokenizer,
            device: device.clone(),
            dtype,
            prompt_template,
            maximum_sequence_length,
        }
    }

    /// Loads a quantized model whose config comes from an optional
    /// `config.json` (GGUF metadata otherwise).
    #[allow(clippy::too_many_arguments)]
    fn load_quantized(
        model: ParallelQuantizedQwen2,
        gguf_file: &Path,
        config_file: &Path,
        tokenizer: Tokenizer,
        device: &Device,
        dtype: DType,
        prompt_template: PromptTemplate,
        architecture: ModelArchitecture,
    ) -> anyhow::Result<Self> {
        let config = if config_file.exists() {
            read_config(config_file, architecture)?
        } else {
            // A GGUF-only checkpoint: derive the parameters from the GGUF
            // metadata so the cache sizing and RoPE match the weights.
            ParallelQuantizedQwen2::config_from_gguf(gguf_file)?
        };
        Ok(Self::assemble(
            ParallelModel::Quantized(model),
            config,
            tokenizer,
            device,
            dtype,
            prompt_template,
        ))
    }

    /// Prompt template in use (Manacá markers or Qwen ChatML).
    pub fn prompt_template(&self) -> &PromptTemplate {
        &self.prompt_template
    }

    /// Builds the system prompt using the Manacá markers.
    ///
    /// Kept as a stable, weights-free helper for tests and external callers;
    /// request-time prompt building goes through [`Self::prompt_template`].
    #[allow(dead_code)]
    pub fn system_prompt(system_text: &str) -> String {
        PromptTemplate::Manaca.system_prompt(system_text)
    }

    /// Builds the user prompt using the Manacá markers.
    #[allow(dead_code)]
    pub fn user_prompt(question: &str) -> String {
        PromptTemplate::Manaca.user_prompt(question)
    }

    /// Builds the full Manacá prompt (pure function, testable without weights).
    #[allow(dead_code)]
    pub fn full_prompt(system_text: &str, question: &str) -> String {
        PromptTemplate::Manaca.full_prompt(system_text, question)
    }

    /// Continues a forward pass over `token_identifiers`, writing the new
    /// key/value entries into `cache` at `index_position`.
    #[tracing::instrument(skip(self, token_identifiers, cache))]
    pub fn continue_forward(
        &self,
        token_identifiers: &[u32],
        index_position: usize,
        cache: &mut ModelCache,
    ) -> anyhow::Result<Tensor> {
        if token_identifiers.is_empty() {
            return Err(anyhow::anyhow!(
                "cannot run a forward pass over an empty token sequence"
            ));
        }
        let tensor = Tensor::new(token_identifiers, &self.device)?.unsqueeze(0)?;
        self.model.forward_last(&tensor, index_position, cache)
    }

    /// Creates an empty key/value cache for this model (no tokens seen yet).
    pub fn empty_cache(&self) -> anyhow::Result<ModelCache> {
        match &self.model {
            ParallelModel::Dense(_) => Ok(ModelCache::Dense(ParallelCache::new(
                true,
                self.dtype,
                &self.config,
                &self.device,
            )?)),
            ParallelModel::Quantized(model) => Ok(ModelCache::Quantized(
                model.empty_cache(self.maximum_sequence_length, self.dtype)?,
            )),
        }
    }

    /// Prefills `token_identifiers` from position 0, returning the populated
    /// cache and the last-position logits. This is the "single broadcast
    /// prefill": the cache can be broadcast and continued for many suffixes.
    #[tracing::instrument(skip(self, token_identifiers))]
    pub fn prefill(&self, token_identifiers: &[u32]) -> anyhow::Result<(ModelCache, Tensor)> {
        let mut cache = self.empty_cache()?;
        let logits = self.continue_forward(token_identifiers, 0, &mut cache)?;
        Ok((cache, logits))
    }

    /// Evaluates several question suffixes in a **single batched forward pass**
    /// on top of one shared, broadcast prefix cache.
    #[tracing::instrument(skip(self, suffix_token_sequences, cache))]
    pub fn score_suffixes_batched(
        &self,
        suffix_token_sequences: &[Vec<u32>],
        index_position: usize,
        cache: &mut ModelCache,
    ) -> anyhow::Result<Tensor> {
        let (padded_tokens, maximum_length, positions) =
            pad_suffix_token_sequences(suffix_token_sequences)?;
        let batch_size = suffix_token_sequences.len();
        let suffix_tensor =
            Tensor::from_vec(padded_tokens, (batch_size, maximum_length), &self.device)?;
        cache.broadcast_batch(batch_size)?;
        let hidden = self
            .model
            .forward_hidden_all(&suffix_tensor, index_position, cache)?;
        self.model
            .logits_from_hidden_at_positions(&hidden, &positions)
    }

    /// Runs a single forward pass over the full prompt at position 0 and
    /// returns the logits tensor plus the total sequence length.
    #[allow(dead_code)]
    #[tracing::instrument(skip(self, prompt))]
    pub fn forward_full(&self, prompt: &str) -> anyhow::Result<(Tensor, usize)> {
        tracing::debug!("starting full forward pass");
        let token_identifiers = self.encode_tokens(prompt, true)?;
        let sequence_length = token_identifiers.len();
        let (_cache, logits) = self.prefill(&token_identifiers)?;
        tracing::debug!("forward finished for {sequence_length} tokens");
        Ok((logits, sequence_length))
    }

    /// Extracts the scalar logit of a token from the last-position logits.
    pub fn extract_logit(logits: &Tensor, token_id: u32) -> anyhow::Result<f32> {
        let value = logits.i((0, token_id as usize))?;
        Ok(value.to_vec0::<f32>()?)
    }

    /// Scalar logit of a token at the last sequence position.
    pub fn token_logit(&self, logits: &Tensor, token_id: u32) -> anyhow::Result<f32> {
        Self::extract_logit(logits, token_id)
    }

    /// Encodes text into vocabulary identifiers.
    pub fn encode_tokens(&self, text: &str, add_special: bool) -> anyhow::Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, add_special)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Number of tokens for a text fragment (used for admission budgets).
    #[allow(dead_code)]
    pub fn token_count(&self, text: &str, add_special: bool) -> usize {
        self.encode_tokens(text, add_special)
            .map(|token_identifiers| token_identifiers.len())
            .unwrap_or(usize::MAX)
    }

    /// Resolves a single-token answer label, trying common spacing variants.
    ///
    /// Kept alongside [`Self::label_token_ids`]; the scoring path uses the full
    /// variant list, this returns only the first match.
    #[allow(dead_code)]
    pub fn label_token_id(&self, label: &str) -> Option<u32> {
        resolve_single_token_id(
            &self.tokenizer,
            &[label.to_string(), format!(" {label}"), format!("\n{label}")],
        )
    }

    /// All single-token ids a label can be emitted as, in priority order.
    ///
    /// The scoring path reads the logit of *every* variant so the calibration
    /// does not depend on which spacing variant the tokenizer happens to rank
    /// first (Qwen tokenizes both `"A"` and `" A"` as single tokens).
    pub fn label_token_ids(&self, label: &str) -> Vec<u32> {
        let mut identifiers = Vec::new();
        for candidate in [label.to_string(), format!(" {label}"), format!("\n{label}")] {
            if let Ok(encoding) = self.tokenizer.encode(candidate.as_str(), false) {
                if encoding.get_ids().len() == 1 {
                    let identifier = encoding.get_ids()[0];
                    if !identifiers.contains(&identifier) {
                        identifiers.push(identifier);
                    }
                }
            }
        }
        identifiers
    }
}

/// Reads a `config.json` body into the normalized parallel model configuration.
fn read_config(
    config_file: &Path,
    architecture: ModelArchitecture,
) -> anyhow::Result<ParallelModelConfig> {
    let bytes = std::fs::read(config_file)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| anyhow::anyhow!("failed to parse '{}': {error}", config_file.display()))?;
    ParallelModelConfig::from_json_for_architecture(value, architecture)
}

/// Maps a resolved weight kind to the dequantization scheme used on load.
///
/// Dense and GGML-quantized checkpoints never reach the low-precision loader;
/// the mapping is total so callers stay exhaustive.
fn scheme_for_weight_kind(weight_kind: WeightKind) -> QuantizationScheme {
    match weight_kind {
        WeightKind::Float8 => QuantizationScheme::Fp8,
        WeightKind::Float4 => QuantizationScheme::Fp4,
        WeightKind::Dense | WeightKind::Quantized | WeightKind::UnsupportedFloat8 => {
            QuantizationScheme::None
        }
    }
}

/// Pads variable-length suffix token sequences to a common length for batching.
///
/// Rows are right-padded with the zero token; under a causal mask the padding
/// cannot influence earlier positions, so the batched result matches scoring
/// each suffix on its own. Pure and testable without weights.
pub fn pad_suffix_token_sequences(
    suffix_token_sequences: &[Vec<u32>],
) -> anyhow::Result<(Vec<u32>, usize, Vec<usize>)> {
    if suffix_token_sequences.is_empty() {
        return Err(anyhow::anyhow!(
            "cannot pad an empty batch of question suffixes"
        ));
    }
    let batch_size = suffix_token_sequences.len();
    let maximum_length = suffix_token_sequences
        .iter()
        .map(|suffix| suffix.len())
        .max()
        .unwrap_or(0);
    if maximum_length == 0 {
        return Err(anyhow::anyhow!(
            "cannot pad a batch whose suffixes are all empty"
        ));
    }
    let mut padded_tokens = vec![0_u32; batch_size * maximum_length];
    let mut positions = Vec::with_capacity(batch_size);
    for (row, suffix) in suffix_token_sequences.iter().enumerate() {
        let row_offset = row * maximum_length;
        padded_tokens[row_offset..row_offset + suffix.len()].copy_from_slice(suffix);
        positions.push(suffix.len() - 1);
    }
    Ok((padded_tokens, maximum_length, positions))
}

/// Returns the vocabulary id when a candidate string encodes to exactly one
/// token, trying each candidate in order. Pure function over the tokenizer so
/// answer-label resolution stays testable without model weights.
fn resolve_single_token_id(tokenizer: &Tokenizer, candidates: &[String]) -> Option<u32> {
    for candidate in candidates {
        if let Ok(encoding) = tokenizer.encode(candidate.as_str(), false) {
            if encoding.get_ids().len() == 1 {
                return Some(encoding.get_ids()[0]);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_token_resolution_accepts_only_single_token_candidates() -> anyhow::Result<()> {
        use std::collections::HashMap;
        use tokenizers::models::wordlevel::WordLevelBuilder;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;

        let vocab: HashMap<String, u32> = [
            ("A".to_string(), 0),
            ("B".to_string(), 1),
            ("[UNK]".to_string(), 2),
        ]
        .into_iter()
        .collect();
        let word_level = WordLevelBuilder::default()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let mut tokenizer = Tokenizer::new(word_level);
        tokenizer.with_pre_tokenizer(Whitespace);

        assert_eq!(
            resolve_single_token_id(&tokenizer, &["A".to_string()]),
            Some(0)
        );
        assert_eq!(
            resolve_single_token_id(&tokenizer, &["A B".to_string()]),
            None
        );
        assert_eq!(
            resolve_single_token_id(&tokenizer, &["A B".to_string(), "B".to_string()]),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn llama_prompt_helpers_are_preserved() {
        assert_eq!(
            LanguageModel::full_prompt("rules", "Is it true?"),
            "<|system|>\nrules\n<|user|>\nIs it true?\n<|model|>\n"
        );
    }

    #[test]
    fn extract_logit_reads_token_from_batch_vocab_matrix() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(vec![vec![1.0f32, 2.0, 3.0]], &device)?;
        let value = LanguageModel::extract_logit(&logits, 2)?;
        assert!((value - 3.0).abs() < 1e-6);
        let first_value = LanguageModel::extract_logit(&logits, 0)?;
        assert!((first_value - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn extract_logit_rejects_out_of_vocab_token() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::zeros((1, 4), DType::F32, &device)?;
        assert!(LanguageModel::extract_logit(&logits, 99).is_err());
        Ok(())
    }

    #[test]
    fn suffix_padding_is_row_major_with_last_real_positions() -> anyhow::Result<()> {
        let sequences = vec![vec![10, 11, 12], vec![20], vec![30, 31]];
        let (padded, length, positions) = pad_suffix_token_sequences(&sequences)?;
        assert_eq!(length, 3);
        assert_eq!(positions, vec![2, 0, 1]);
        assert_eq!(padded, vec![10, 11, 12, 20, 0, 0, 30, 31, 0]);
        Ok(())
    }

    #[test]
    fn suffix_padding_rejects_empty_inputs() {
        assert!(pad_suffix_token_sequences(&[]).is_err());
        assert!(pad_suffix_token_sequences(&[Vec::new(), Vec::new()]).is_err());
    }

    /// Live test (ignored): the batched broadcast path matches sequential
    /// scoring for the real model. Downloads the weights once.
    #[test]
    #[ignore]
    fn batched_suffixes_match_sequential() -> anyhow::Result<()> {
        use typed_lm_common::checkpoint::ModelReference;
        use typed_lm_common::checkpoint_resolver::{HubCheckpointResolver, LoadableCheckpoint};
        use typed_lm_common::device::{DeviceResolver, ModelDtype};

        let model_identifier =
            std::env::var("MODEL_ID").unwrap_or_else(|_| "Qwen/Qwen2.5-1.5B-Instruct".to_string());
        let device = DeviceResolver::resolve()?;
        let dtype = ModelDtype::Auto.resolve(&device);
        let reference = ModelReference::resolve(&model_identifier);
        let checkpoint: LoadableCheckpoint =
            HubCheckpointResolver::new(&model_identifier, None)?.resolve(&reference, None)?;
        let model = LanguageModel::load(&checkpoint, &device, dtype)?;

        let prefix: Vec<u32> = (2..40).collect();
        let suffixes: Vec<Vec<u32>> = vec![
            vec![100, 101, 102, 103],
            vec![200],
            vec![300, 301, 302, 303, 304, 305],
        ];
        let prefix_length = prefix.len();
        let (shared_cache, _) = model.prefill(&prefix)?;

        let mut sequential_logits = Vec::new();
        for suffix in &suffixes {
            let mut sequential_cache = shared_cache.clone();
            sequential_logits.push(model.continue_forward(
                suffix,
                prefix_length,
                &mut sequential_cache,
            )?);
        }

        let mut broadcast_cache = shared_cache.clone();
        let batched =
            model.score_suffixes_batched(&suffixes, prefix_length, &mut broadcast_cache)?;

        for (row, sequential) in sequential_logits.iter().enumerate() {
            for token_id in [0_u32, 1, 7, 123, 1000] {
                let sequential_value = LanguageModel::extract_logit(sequential, token_id)?;
                let batched_value = batched.i((row, token_id as usize))?.to_vec0::<f32>()?;
                assert!(
                    (sequential_value - batched_value).abs() < 1e-3,
                    "row {row} token {token_id}: sequential {sequential_value} vs batched {batched_value}"
                );
            }
        }
        Ok(())
    }

    /// Live benchmark (ignored): reports the latency of the prefill and the
    /// batched suffix pass for the real model on the resolved device.
    #[test]
    #[ignore]
    fn reports_latency_breakdown() -> anyhow::Result<()> {
        use std::time::Instant;
        use typed_lm_common::checkpoint::ModelReference;
        use typed_lm_common::checkpoint_resolver::HubCheckpointResolver;
        use typed_lm_common::device::{DeviceResolver, ModelDtype};

        let model_identifier =
            std::env::var("MODEL_ID").unwrap_or_else(|_| "Qwen/Qwen2.5-1.5B-Instruct".to_string());
        let device = DeviceResolver::resolve()?;
        let dtype = ModelDtype::Auto.resolve(&device);
        let reference = ModelReference::resolve(&model_identifier);
        let checkpoint =
            HubCheckpointResolver::new(&model_identifier, None)?.resolve(&reference, None)?;
        let model = LanguageModel::load(&checkpoint, &device, dtype)?;
        println!("device: {device:?}, dtype: {dtype:?}");

        // Warm up.
        let warm = model.prefill(&(2..40).collect::<Vec<u32>>())?.1;
        let _ = warm;

        for prefix_length in [64_usize, 256, 1024] {
            let prefix: Vec<u32> = (0..prefix_length)
                .map(|index| (index % 1000) as u32 + 3)
                .collect();
            let prefix_start = Instant::now();
            let (shared_cache, _) = model.prefill(&prefix)?;
            let prefix_elapsed = prefix_start.elapsed();

            let suffixes: Vec<Vec<u32>> = (0..5)
                .map(|row| vec![100 + row as u32, 200, 300, 400])
                .collect();
            let batched_start = Instant::now();
            let mut cache = shared_cache.clone();
            model.score_suffixes_batched(&suffixes, prefix_length, &mut cache)?;
            let batched_elapsed = batched_start.elapsed();

            let single_start = Instant::now();
            let mut single_cache = shared_cache.clone();
            model.continue_forward(&[42], prefix_length, &mut single_cache)?;
            let single_elapsed = single_start.elapsed();

            println!(
                "prefix {prefix_length:>5}: prefill {prefix_elapsed:>10.2?} | 5 batched suffixes {batched_elapsed:>10.2?} | single next-token {single_elapsed:>10.2?}"
            );
        }
        Ok(())
    }
}
