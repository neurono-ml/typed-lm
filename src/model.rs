use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig};
use std::path::Path;
use tokenizers::Tokenizer;

/// Loads weights, config and tokenizer, exposing the model ready for inference.
///
/// Fixed issues:
/// - `Llama::load` and `Cache::new` require `&Config`, not `&LlamaConfig`:
///   the conversion is done via `LlamaConfig::into_config(false)`.
/// - `VarBuilder::from_mmaped_safetensors` is `unsafe`: called inside an `unsafe` block.
pub struct LanguageModel {
    llama: Llama,
    config: Config,
    tokenizer: Tokenizer,
    device: Device,
}

impl LanguageModel {
    pub fn load(
        config_file: &Path,
        weights_file: &Path,
        tokenizer_file: &Path,
        device: &Device,
    ) -> anyhow::Result<Self>
    where
        anyhow::Error: From<candle_core::Error> + From<std::io::Error>,
    {
        let open_file = std::fs::File::open(config_file)?;
        let llama_config: LlamaConfig = serde_json::from_reader(open_file)?;
        // FIX: convert serializable LlamaConfig into runtime Config.
        let config: Config = llama_config.into_config(false);

        // FIX: unsafe function requires an unsafe block.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_file], DType::F32, device)? };

        let llama = Llama::load(vb, &config)?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_file).map_err(|e| anyhow::anyhow!(e.to_string()))?;

        Ok(Self {
            llama,
            config,
            tokenizer,
            device: device.clone(),
        })
    }

    /// Builds the system prompt (pure function, testable without weights).
    pub fn system_prompt(system_text: &str) -> String {
        format!("<|system|>\n{}\n", system_text)
    }

    /// Builds the user prompt (pure function, testable without weights).
    pub fn user_prompt(question: &str) -> String {
        format!("<|user|>\n{}\n<|model|>\n", question)
    }

    /// Builds the full classification prompt (pure function, testable without weights).
    pub fn full_prompt(system_text: &str, question: &str) -> String {
        format!(
            "{}{}",
            Self::system_prompt(system_text),
            Self::user_prompt(question)
        )
    }

    /// Runs a single forward pass over the full prompt at position 0 and
    /// returns the logits tensor plus the total sequence length.
    ///
    /// A single pass is required: candle-transformers 0.6 builds a square
    /// `(seq_len, seq_len)` causal mask, so resuming a cache at `index_pos > 0`
    /// with more than one token fails to broadcast.
    pub fn forward_full(&self, prompt: &str) -> anyhow::Result<(Tensor, usize)> {
        let mut cache = Cache::new(true, DType::F32, &self.config, &self.device)?;
        let ids = self.tokenize(prompt, true)?;
        let n = ids.len();
        let tensor = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.llama.forward(&tensor, 0, &mut cache)?;
        Ok((logits, n))
    }

    /// Extracts the scalar logit of a token from the last-position logits
    /// (pure function, testable with a synthetic tensor, no weights needed).
    ///
    /// NOTE: `Llama::forward` already selects the last sequence position, so its
    /// output is shaped `[batch, vocab]` — there is no sequence axis to index.
    pub fn extract_logit(logits: &Tensor, token_id: u32) -> anyhow::Result<f32> {
        let value = logits.i((0, token_id as usize))?;
        Ok(value.to_vec0::<f32>()?)
    }

    /// Scalar logit of a token at the last sequence position.
    pub fn token_logit(&self, logits: &Tensor, token_id: u32) -> anyhow::Result<f32> {
        Self::extract_logit(logits, token_id)
    }

    /// Resolves a token id, trying with and without a leading space.
    /// Returns an error instead of panicking via `unwrap`.
    pub fn token_id(&self, with_space: &str, without_space: &str) -> anyhow::Result<u32> {
        self.tokenizer
            .token_to_id(with_space)
            .or_else(|| self.tokenizer.token_to_id(without_space))
            .ok_or_else(|| {
                anyhow::anyhow!("tokens '{with_space}'/'{without_space}' not found in vocabulary")
            })
    }

    fn tokenize(&self, text: &str, add_special: bool) -> anyhow::Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, add_special)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Number of tokens for a text fragment (used for admission budgets).
    pub fn token_count(&self, text: &str, add_special: bool) -> usize {
        self.tokenize(text, add_special)
            .map(|ids| ids.len())
            .unwrap_or(usize::MAX)
    }

    /// Resolves a single-token answer label, trying common spacing variants.
    /// Returns None when the label is not exactly one vocabulary token.
    pub fn label_token_id(&self, label: &str) -> Option<u32> {
        for candidate in [format!(" {label}"), label.to_string(), format!("\n{label}")] {
            if let Some(id) = self.tokenizer.token_to_id(&candidate) {
                let ids = self.tokenizer.encode(candidate.clone(), false).ok()?;
                if ids.get_ids().len() == 1 && ids.get_ids()[0] == id {
                    return Some(id);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_wraps_content_with_markers() {
        let p = LanguageModel::system_prompt("rules here");
        assert!(p.starts_with("<|system|>\n"));
        assert!(p.ends_with("rules here\n"));
    }

    #[test]
    fn user_prompt_wraps_question_and_hands_off_to_model() {
        let p = LanguageModel::user_prompt("Is the sky blue?");
        assert!(p.starts_with("<|user|>\n"));
        assert!(p.contains("Is the sky blue?"));
        assert!(p.ends_with("<|model|>\n"));
    }

    #[test]
    fn full_prompt_concatenates_system_then_user() {
        let p = LanguageModel::full_prompt("rules", "Is it true?");
        assert_eq!(p, "<|system|>\nrules\n<|user|>\nIs it true?\n<|model|>\n");
    }

    #[test]
    fn extract_logit_reads_token_from_batch_vocab_matrix() {
        let device = Device::Cpu;
        // Last-position logits shaped (1, 3): batch=1, vocab=3.
        let logits = Tensor::new(vec![vec![1.0f32, 2.0, 3.0]], &device).unwrap();
        let v = LanguageModel::extract_logit(&logits, 2).unwrap();
        assert!((v - 3.0).abs() < 1e-6);
        let v0 = LanguageModel::extract_logit(&logits, 0).unwrap();
        assert!((v0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn extract_logit_rejects_out_of_vocab_token() {
        let device = Device::Cpu;
        let logits = Tensor::zeros((1, 4), candle_core::DType::F32, &device).unwrap();
        assert!(LanguageModel::extract_logit(&logits, 99).is_err());
    }
}
