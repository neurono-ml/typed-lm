//! Tokenization and batch collation for training.
//!
//! Training mirrors serving exactly: for each item the trainer builds
//! `system_prompt(context) + state_prefix(state) + question_completion(question)`
//! with the same [`PromptTemplate`] the server uses, so the last token of the
//! prompt is the *decision position* where the server reads the answer-label
//! logits. The cross-entropy loss is applied only there, against the item's
//! answer-label token.
//!
//! Items that share a state are grouped so a batch reuses one state prefix;
//! within a state, items are ordered by length (length bucketing) to minimize
//! padding, and items longer than `--max-sequence-length` are skipped.

use std::collections::BTreeMap;

use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;
use typed_lm_common::prompt_template::PromptTemplate;
use typed_lm_common::rendering::{content_to_text, render_question_text};
use typed_lm_common::tokenizer::label_token_id;

use crate::dataset::record::TrainingItem;

/// A training item tokenized and anchored at its decision position.
#[derive(Debug, Clone)]
pub struct TokenizedItem {
    /// Full prompt token ids (`system + state prefix + question completion`).
    pub input_ids: Vec<u32>,
    /// Index of the last prompt token, where the label is predicted.
    pub decision_position: usize,
    /// Vocabulary id of the answer label (`A`, `B`, ...).
    pub label_token_id: u32,
    /// Canonical state text used to group items that share a prefix.
    pub state_key: String,
}

/// One padded batch ready for a forward/backward pass.
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// `(batch_size, sequence_length)` token ids, right-padded with `0`.
    pub input_ids: Tensor,
    /// `(batch_size, sequence_length)` decision mask (`1` only at the decision).
    pub decision_mask: Tensor,
    /// `(batch_size,)` decision positions.
    pub decision_positions: Tensor,
    /// `(batch_size,)` answer-label token ids.
    pub label_token_ids: Tensor,
    /// Padded sequence length shared by the batch.
    pub sequence_length: usize,
}

/// Builds the decision mask: all zeros except a single `1` at the decision.
fn decision_mask_row(sequence_length: usize, decision_position: usize) -> Vec<u32> {
    let mut mask = vec![0_u32; sequence_length];
    if decision_position < sequence_length {
        mask[decision_position] = 1;
    }
    mask
}

/// Tokenizes one item into its prompt ids, decision position and label id.
pub fn tokenize_item(
    tokenizer: &Tokenizer,
    template: PromptTemplate,
    context_text: &str,
    item: &TrainingItem,
) -> anyhow::Result<TokenizedItem> {
    let state_text = content_to_text(&item.state);
    let question_text = render_question_text(&item.question);
    let prompt = format!(
        "{}{}{}",
        template.system_prompt(context_text),
        template.state_prefix(&state_text),
        template.question_completion(&question_text),
    );
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|error| anyhow::anyhow!("failed to tokenize a training prompt: {error}"))?;
    let input_ids = encoding.get_ids().to_vec();
    if input_ids.is_empty() {
        return Err(anyhow::anyhow!(
            "training prompt tokenized to an empty sequence"
        ));
    }
    let decision_position = input_ids.len() - 1;
    let label_token_id = label_token_id(tokenizer, &item.answer_label).ok_or_else(|| {
        anyhow::anyhow!(
            "answer label '{}' is not a single vocabulary token",
            item.answer_label
        )
    })?;
    Ok(TokenizedItem {
        input_ids,
        decision_position,
        label_token_id,
        state_key: state_text,
    })
}

/// Groups items by their shared state prefix.
pub fn group_by_state(items: Vec<TokenizedItem>) -> BTreeMap<String, Vec<TokenizedItem>> {
    let mut grouped: BTreeMap<String, Vec<TokenizedItem>> = BTreeMap::new();
    for item in items {
        grouped
            .entry(item.state_key.clone())
            .or_default()
            .push(item);
    }
    grouped
}

/// Pads items into a single batch on `device`.
///
/// The items must be non-empty and share the same state prefix (the caller
/// groups them first); padding uses token id `0`.
pub fn collate_batch(items: &[TokenizedItem], device: &Device) -> anyhow::Result<TrainingBatch> {
    if items.is_empty() {
        return Err(anyhow::anyhow!("cannot collate an empty batch"));
    }
    let sequence_length = items
        .iter()
        .map(|item| item.input_ids.len())
        .max()
        .unwrap_or(0);
    let batch_size = items.len();

    let mut flattened_ids: Vec<u32> = Vec::with_capacity(batch_size * sequence_length);
    let mut flattened_mask: Vec<u32> = Vec::with_capacity(batch_size * sequence_length);
    let mut decision_positions: Vec<u32> = Vec::with_capacity(batch_size);
    let mut label_token_ids: Vec<u32> = Vec::with_capacity(batch_size);
    for item in items {
        flattened_ids.extend_from_slice(&item.input_ids);
        flattened_ids.extend(std::iter::repeat_n(
            0_u32,
            sequence_length - item.input_ids.len(),
        ));
        flattened_mask.extend(decision_mask_row(sequence_length, item.decision_position));
        decision_positions.push(item.decision_position as u32);
        label_token_ids.push(item.label_token_id);
    }

    let input_ids = Tensor::from_vec(flattened_ids, (batch_size, sequence_length), device)?;
    let decision_mask = Tensor::from_vec(flattened_mask, (batch_size, sequence_length), device)?;
    let decision_positions = Tensor::from_vec(decision_positions, (batch_size,), device)?;
    let label_token_ids = Tensor::from_vec(label_token_ids, (batch_size,), device)?;
    Ok(TrainingBatch {
        input_ids,
        decision_mask,
        decision_positions,
        label_token_ids,
        sequence_length,
    })
}

/// Builds length-bucketed batches grouped by shared state.
///
/// Items longer than `max_sequence_length` are dropped; if that leaves nothing,
/// the call fails so a misconfigured maximum is not silently ignored.
pub fn build_batches(
    items: Vec<TokenizedItem>,
    batch_size: usize,
    max_sequence_length: usize,
    device: &Device,
) -> anyhow::Result<Vec<TrainingBatch>> {
    if batch_size == 0 {
        return Err(anyhow::anyhow!("batch size must be greater than zero"));
    }
    let mut batches: Vec<TrainingBatch> = Vec::new();
    for (_state, mut state_items) in group_by_state(items) {
        state_items.retain(|item| item.input_ids.len() <= max_sequence_length);
        state_items.sort_by_key(|item| item.input_ids.len());
        for chunk in state_items.chunks(batch_size) {
            batches.push(collate_batch(chunk, device)?);
        }
    }
    if batches.is_empty() {
        return Err(anyhow::anyhow!(
            "no training item fits within the maximum sequence length of {max_sequence_length}"
        ));
    }
    Ok(batches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashMap;
    use tokenizers::models::wordlevel::WordLevelBuilder;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use typed_lm_common::contract::Content;
    use typed_lm_common::labels::label_sequence;

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

    /// A tokenizer whose vocabulary covers the template markers and labels.
    fn template_tokenizer() -> anyhow::Result<Tokenizer> {
        let vocabulary: Vec<(&str, u32)> = vec![
            ("<|system|>", 0),
            ("<|user|>", 1),
            ("<|model|>", 2),
            ("State:", 3),
            ("Question:", 4),
            ("A", 5),
            ("B", 6),
            ("C", 7),
            ("[UNK]", 8),
        ];
        word_level_tokenizer(&vocabulary)
    }

    fn noul_item(state: &str, answer_index: usize) -> TrainingItem {
        TrainingItem {
            state: Content::Text(state.to_string()),
            question_identifier: "refund".to_string(),
            question: typed_lm_common::contract::Question::Noul {
                instructions: Content::Text("Refund?".to_string()),
                criteria: typed_lm_common::contract::NoulCriteria::default(),
            },
            answer: if answer_index == 0 { "yes" } else { "no" }.to_string(),
            answer_index,
            answer_label: label_sequence(answer_index),
            label_count: 2,
        }
    }

    #[test]
    fn tokenizes_with_the_decision_position_at_the_last_token() -> anyhow::Result<()> {
        let tokenizer = template_tokenizer()?;
        let item = noul_item("charged twice", 0);
        let tokenized = tokenize_item(&tokenizer, PromptTemplate::Manaca, "rules", &item)?;
        assert_eq!(
            tokenized.decision_position,
            tokenized.input_ids.len() - 1,
            "the decision is the last prompt token"
        );
        assert_eq!(tokenized.label_token_id, 5, "label A maps to id 5");
        assert_eq!(tokenized.state_key, "charged twice");
        Ok(())
    }

    #[test]
    fn groups_items_by_shared_state() -> anyhow::Result<()> {
        let tokenizer = template_tokenizer()?;
        let first = tokenize_item(
            &tokenizer,
            PromptTemplate::Manaca,
            "",
            &noul_item("state one", 0),
        )?;
        let second = tokenize_item(
            &tokenizer,
            PromptTemplate::Manaca,
            "",
            &noul_item("state two", 1),
        )?;
        let grouped = group_by_state(vec![first, second]);
        assert_eq!(grouped.len(), 2);
        Ok(())
    }

    #[test]
    fn collate_pads_and_marks_only_the_decision() -> anyhow::Result<()> {
        let tokenizer = template_tokenizer()?;
        let short = tokenize_item(
            &tokenizer,
            PromptTemplate::Manaca,
            "",
            &noul_item("state one", 0),
        )?;
        let long = tokenize_item(
            &tokenizer,
            PromptTemplate::Manaca,
            "",
            &noul_item("a much longer state value", 1),
        )?;
        let batch = collate_batch(&[short, long], &Device::Cpu)?;
        assert_eq!(batch.input_ids.dims(), &[2, batch.sequence_length]);
        let mask = batch.decision_mask.to_vec2::<u32>()?;
        for row in &mask {
            let ones: u32 = row.iter().sum();
            assert_eq!(ones, 1, "exactly one decision per row");
        }
        let positions = batch.decision_positions.to_vec1::<u32>()?;
        assert_eq!(positions.len(), 2);
        let labels = batch.label_token_ids.to_vec1::<u32>()?;
        assert_eq!(labels, vec![5, 6]);
        Ok(())
    }

    #[test]
    fn build_batches_buckets_by_state_and_length() -> anyhow::Result<()> {
        let tokenizer = template_tokenizer()?;
        let items: Vec<TokenizedItem> = ["s one", "s two", "s three"]
            .iter()
            .enumerate()
            .map(|(index, state)| {
                tokenize_item(
                    &tokenizer,
                    PromptTemplate::Manaca,
                    "",
                    &noul_item(state, index % 2),
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let batches = build_batches(items, 2, 512, &Device::Cpu)?;
        // Three distinct states => three single-item batches.
        assert_eq!(batches.len(), 3);
        Ok(())
    }

    #[test]
    fn build_batches_skips_items_beyond_max_sequence_length() -> anyhow::Result<()> {
        let tokenizer = template_tokenizer()?;
        let item = tokenize_item(
            &tokenizer,
            PromptTemplate::Manaca,
            "",
            &noul_item("some state", 0),
        )?;
        let result = build_batches(vec![item], 1, 0, &Device::Cpu);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn a_zero_batch_size_is_an_error() -> anyhow::Result<()> {
        let result = build_batches(Vec::new(), 0, 512, &Device::Cpu);
        assert!(result.is_err());
        Ok(())
    }
}
