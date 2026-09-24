//! Answer label arithmetic shared by serving and training.
//!
//! A question with `n` candidates consumes the first `n` labels of the
//! spreadsheet-style sequence (`A`, `B`, ... `Z`, `AA`, ...). Both the server
//! (reading label logits) and the trainer (building the decision-position loss
//! mask) must agree on this mapping, so it lives in `typed-lm-common`.

use crate::classifier::Classifier;
use crate::contract::Question;

/// Softmax temperature applied to every answer distribution.
pub const SCORING_TEMPERATURE: f32 = 1.0;

/// Converts a zero-based option index into a spreadsheet-style answer label:
/// 0 becomes `A`, 25 becomes `Z`, 26 becomes `AA`, 27 becomes `AB`, and so on.
pub fn label_sequence(option_index: usize) -> String {
    let mut label = String::new();
    let mut remaining = option_index + 1;
    while remaining > 0 {
        remaining -= 1;
        let character = (b'A' + (remaining % 26) as u8) as char;
        label.insert(0, character);
        remaining /= 26;
    }
    label
}

/// Builds one answer label per option (`A` through `Z`, then `AA` and beyond).
pub fn option_labels(option_count: usize) -> Vec<String> {
    (0..option_count).map(label_sequence).collect()
}

/// Counts how many answer labels one question needs (two for boolean questions).
pub fn label_count_for_question(question: &Question) -> usize {
    match question {
        Question::Noul { .. } => 2,
        Question::Choice { criteria, .. } => criteria.len(),
        Question::Score { criteria, .. } => criteria.len(),
    }
}

/// Calibrates raw label logits into a probability distribution.
///
/// Boolean questions carry exactly two labels and go through the binary
/// softmax; larger label sets use the temperature-scaled softmax. Both
/// paths compute the same distribution family, so `normalize` and
/// `binary_softmax` agree on two-logit inputs.
pub fn calibrate_probabilities(logit_values: &[f32]) -> Vec<f32> {
    if logit_values.len() == 2 {
        let classification = Classifier::binary_softmax(logit_values[0], logit_values[1]);
        return vec![classification.true_prob, classification.false_prob];
    }
    Classifier::normalize(logit_values, SCORING_TEMPERATURE)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn labels_start_at_a_and_roll_over_after_z() {
        assert_eq!(label_sequence(0), "A");
        assert_eq!(label_sequence(1), "B");
        assert_eq!(label_sequence(25), "Z");
        assert_eq!(label_sequence(26), "AA");
        assert_eq!(label_sequence(27), "AB");
        assert_eq!(label_sequence(51), "AZ");
        assert_eq!(label_sequence(52), "BA");
    }

    #[test]
    fn option_labels_cover_two_rounds_without_repetition() {
        let labels = option_labels(28);
        assert_eq!(labels.len(), 28);
        assert_eq!(labels[0], "A");
        assert_eq!(labels[25], "Z");
        assert_eq!(labels[26], "AA");
        let mut unique = labels.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), labels.len());
    }

    #[test]
    fn calibration_matches_hand_computed_softmax() {
        let probabilities = calibrate_probabilities(&[2.0, 0.0]);
        assert!((probabilities[0] - 0.8808).abs() < 1e-4);
        assert!((probabilities[1] - 0.1192).abs() < 1e-4);
    }

    #[test]
    fn calibration_sums_to_one() {
        let probabilities = calibrate_probabilities(&[2.0, 1.0, 0.5]);
        let total: f32 = probabilities.iter().sum();
        assert!((total - 1.0).abs() < EPSILON);
        assert!(probabilities[0] > probabilities[1]);
        assert!(probabilities[1] > probabilities[2]);
    }

    #[test]
    fn label_count_is_two_for_noul_and_criteria_len_otherwise() -> anyhow::Result<()> {
        let raw = serde_json::json!({
            "n": {"type": "noul", "instructions": "Q?"},
            "c": {"type": "choice", "instructions": "Q?", "criteria": {"a": null, "b": null}},
            "s": {"type": "score", "instructions": "Q?", "criteria": ["x", "y", "z"]}
        });
        let questions: std::collections::HashMap<String, Question> = serde_json::from_value(raw)?;
        let noul = questions
            .get("n")
            .ok_or_else(|| anyhow::anyhow!("missing n"))?;
        let choice = questions
            .get("c")
            .ok_or_else(|| anyhow::anyhow!("missing c"))?;
        let score = questions
            .get("s")
            .ok_or_else(|| anyhow::anyhow!("missing s"))?;
        assert_eq!(label_count_for_question(noul), 2);
        assert_eq!(label_count_for_question(choice), 2);
        assert_eq!(label_count_for_question(score), 3);
        Ok(())
    }
}
