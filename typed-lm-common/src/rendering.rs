//! Deterministic text rendering of the request contract.
//!
//! Serving and training must build byte-identical prompts for the same
//! request, otherwise the trainer's decision positions would not line up with
//! the logits the server reads at inference time. Both crates therefore share
//! these renderers.

use crate::contract::{Content, Question};
use crate::labels::label_sequence;

/// Renders structured or free-text content as plain text.
pub fn content_to_text(content: &Content) -> String {
    match content {
        Content::Text(text) => text.clone(),
        Content::Structured(value) => value.to_string(),
    }
}

/// Renders the human-readable text evaluated for one question.
pub fn render_question_text(question: &Question) -> String {
    match question {
        Question::Noul {
            instructions,
            criteria,
        } => {
            format!(
                "{}\nYes means {}. No means {}.",
                content_to_text(instructions),
                criteria.yes,
                criteria.no
            )
        }
        Question::Choice {
            instructions,
            criteria,
        } => {
            let mut option_names: Vec<&String> = criteria.keys().collect();
            option_names.sort();
            let listed: Vec<String> = option_names
                .iter()
                .enumerate()
                .map(|(option_index, option_name)| {
                    let description = criteria
                        .get(*option_name)
                        .and_then(|candidate| candidate.clone())
                        .unwrap_or_else(|| (*option_name).clone());
                    format!(
                        "{}: {} ({})",
                        label_sequence(option_index),
                        option_name,
                        description
                    )
                })
                .collect();
            format!("{}\n{}", content_to_text(instructions), listed.join("\n"))
        }
        Question::Score {
            instructions,
            criteria,
        } => {
            let listed: Vec<String> = criteria
                .iter()
                .enumerate()
                .map(|(level_index, level_name)| {
                    format!("{}: {}", label_sequence(level_index), level_name)
                })
                .collect();
            format!("{}\n{}", content_to_text(instructions), listed.join("\n"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_content_is_returned_verbatim() {
        assert_eq!(
            content_to_text(&Content::Text("hello".to_string())),
            "hello"
        );
    }

    #[test]
    fn structured_content_is_rendered_as_json() {
        let rendered = content_to_text(&Content::Structured(serde_json::json!({"a": 1})));
        assert!(rendered.contains("\"a\""));
    }

    #[test]
    fn noul_question_mentions_yes_and_no_labels() -> anyhow::Result<()> {
        let question: Question = serde_json::from_value(serde_json::json!({
            "type": "noul", "instructions": "Refund?"
        }))?;
        let rendered = render_question_text(&question);
        assert!(rendered.contains("Refund?"));
        assert!(rendered.contains("Yes means Yes."));
        assert!(rendered.contains("No means No."));
        Ok(())
    }

    #[test]
    fn choice_question_lists_sorted_option_labels() -> anyhow::Result<()> {
        let question: Question = serde_json::from_value(serde_json::json!({
            "type": "choice", "instructions": "Route?",
            "criteria": {"billing": "Payments", "technical": "Bugs"}
        }))?;
        let rendered = render_question_text(&question);
        let billing = rendered
            .find("A: billing (Payments)")
            .ok_or_else(|| anyhow::anyhow!("missing billing option: {rendered}"))?;
        let technical = rendered
            .find("B: technical (Bugs)")
            .ok_or_else(|| anyhow::anyhow!("missing technical option: {rendered}"))?;
        assert!(billing < technical);
        Ok(())
    }

    #[test]
    fn score_question_numbers_each_level() -> anyhow::Result<()> {
        let question: Question = serde_json::from_value(serde_json::json!({
            "type": "score", "instructions": "Urgent?",
            "criteria": ["Routine", "Urgent", "Emergency"]
        }))?;
        let rendered = render_question_text(&question);
        assert!(rendered.contains("A: Routine"));
        assert!(rendered.contains("C: Emergency"));
        Ok(())
    }
}
