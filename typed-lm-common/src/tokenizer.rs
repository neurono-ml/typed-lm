//! Tokenizer helpers shared by serving and training.
//!
//! The single-forward-pass classifier reads the logit of an answer label
//! (`A`, `B`, ...). Which vocabulary id a label maps to depends on the
//! tokenizer's whitespace handling, so both the server and the trainer must
//! resolve labels the same way. This module centralizes that resolution and is
//! pure over the `tokenizers::Tokenizer`, keeping it testable without weights.

use std::path::Path;

use tokenizers::Tokenizer;

/// Candidate text renderings of one label, in priority order.
///
/// Qwen-style tokenizers encode both `"A"` and `" A"` as a single token, while
/// Llama-style ones prefer the leading-space variant after a newline. Reading
/// every variant keeps calibration independent of the tokenizer's ranking.
fn label_candidates(label: &str) -> [String; 3] {
    [label.to_string(), format!(" {label}"), format!("\n{label}")]
}

/// Loads a `tokenizers::Tokenizer` from a `tokenizer.json` file.
pub fn load_tokenizer(tokenizer_file: &Path) -> anyhow::Result<Tokenizer> {
    Tokenizer::from_file(tokenizer_file)
        .map_err(|error| anyhow::anyhow!("failed to load '{}': {error}", tokenizer_file.display()))
}

/// Returns the vocabulary id when a candidate string encodes to exactly one
/// token, trying each candidate in order. Pure function over the tokenizer so
/// answer-label resolution stays testable without model weights.
pub fn resolve_single_token_id(tokenizer: &Tokenizer, candidates: &[String]) -> Option<u32> {
    for candidate in candidates {
        if let Ok(encoding) = tokenizer.encode(candidate.as_str(), false) {
            if encoding.get_ids().len() == 1 {
                return Some(encoding.get_ids()[0]);
            }
        }
    }
    None
}

/// Resolves a single-token answer label, trying common spacing variants.
pub fn label_token_id(tokenizer: &Tokenizer, label: &str) -> Option<u32> {
    resolve_single_token_id(tokenizer, &label_candidates(label))
}

/// All single-token ids a label can be emitted as, in priority order.
///
/// The scoring path reads the logit of *every* variant so the calibration does
/// not depend on which spacing variant the tokenizer happens to rank first.
pub fn label_token_ids(tokenizer: &Tokenizer, label: &str) -> Vec<u32> {
    let mut identifiers = Vec::new();
    for candidate in label_candidates(label) {
        if let Ok(encoding) = tokenizer.encode(candidate.as_str(), false) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashMap;
    use tokenizers::models::wordlevel::WordLevelBuilder;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    /// Builds a tiny `WordLevel` tokenizer with a fixed vocabulary.
    fn word_level_tokenizer(vocabulary: &[(&str, u32)]) -> anyhow::Result<Tokenizer> {
        let words: AHashMap<String, u32> = vocabulary
            .iter()
            .map(|(token, identifier)| (token.to_string(), *identifier))
            .collect();
        let word_level = WordLevelBuilder::default()
            .vocab(words)
            .unk_token("[UNK]".to_string())
            .build()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let mut tokenizer = Tokenizer::new(word_level);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        Ok(tokenizer)
    }

    #[test]
    fn single_token_resolution_accepts_only_single_token_candidates() -> anyhow::Result<()> {
        let tokenizer = word_level_tokenizer(&[("A", 0), ("B", 1), ("[UNK]", 2)])?;

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
    fn label_token_ids_collects_distinct_single_token_variants() -> anyhow::Result<()> {
        // `Whitespace` strips the leading space, so "A" and " A" both encode to
        // id 0; the newline variant is unknown and must be ignored.
        let tokenizer = word_level_tokenizer(&[("A", 0), ("[UNK]", 2)])?;
        let identifiers = label_token_ids(&tokenizer, "A");
        assert_eq!(identifiers, vec![0]);
        Ok(())
    }

    #[test]
    fn label_token_id_returns_the_first_single_token_match() -> anyhow::Result<()> {
        let tokenizer = word_level_tokenizer(&[("A", 0), ("[UNK]", 2)])?;
        assert_eq!(label_token_id(&tokenizer, "A"), Some(0));
        // A multi-word label encodes to several tokens and has no single id.
        assert_eq!(label_token_id(&tokenizer, "A B"), None);
        Ok(())
    }

    #[test]
    fn loading_a_missing_tokenizer_file_is_an_error() {
        let result = load_tokenizer(Path::new("does-not-exist-tokenizer.json"));
        assert!(result.is_err());
    }

    #[test]
    fn loading_a_valid_tokenizer_file_succeeds() -> anyhow::Result<()> {
        let tokenizer = word_level_tokenizer(&[("A", 0), ("[UNK]", 2)])?;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("tokenizer.json");
        tokenizer
            .save(&path, false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let reloaded = load_tokenizer(&path)?;
        assert_eq!(label_token_id(&reloaded, "A"), Some(0));
        Ok(())
    }
}
