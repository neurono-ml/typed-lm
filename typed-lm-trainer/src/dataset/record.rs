//! Jev-native training records and their expansion into decision items.
//!
//! A dataset record mirrors the serving request contract (`state` plus a map of
//! typed `questions`) and adds a ground-truth `answer` to every question. The
//! answer is the semantic choice — `yes`/`no` for a `noul` question, an option
//! name for a `choice`, a level name for a `score` — and is mapped to one of the
//! spreadsheet-style answer labels (`A`, `B`, ...) that the server reads logits
//! for. Keeping the mapping here guarantees the trainer optimizes exactly the
//! label the server scores.

use std::collections::BTreeMap;

use serde_json::Value;
use typed_lm_common::contract::{Content, Question};
use typed_lm_common::labels::label_sequence;

/// One question together with its ground-truth answer.
#[derive(Debug, Clone)]
pub struct TrainingQuestion {
    pub question: Question,
    pub answer: String,
}

/// One dataset record: a state and its answered questions.
#[derive(Debug, Clone)]
pub struct TrainingRecord {
    pub state: Content,
    pub questions: BTreeMap<String, TrainingQuestion>,
}

/// One question expanded into a training item with its answer label.
///
/// `answer_index` is the zero-based position in the question's candidate order
/// and `answer_label` its spreadsheet letter, matching what the server scores.
#[derive(Debug, Clone)]
pub struct TrainingItem {
    pub state: Content,
    pub question_identifier: String,
    pub question: Question,
    pub answer: String,
    pub answer_index: usize,
    pub answer_label: String,
    pub label_count: usize,
}

/// Maps an answer to its candidate index for the question type.
///
/// The candidate order matches [`typed_lm_common::rendering::render_question_text`]:
/// boolean questions are `yes` then `no`; a choice sorts its option names
/// lexicographically; a score follows the declared level order.
fn answer_index_for(question: &Question, answer: &str) -> Result<(usize, usize), String> {
    match question {
        Question::Noul { criteria, .. } => {
            let normalized = answer.trim().to_ascii_lowercase();
            let yes_matches =
                normalized == "yes" || normalized == criteria.yes.to_ascii_lowercase();
            let no_matches = normalized == "no" || normalized == criteria.no.to_ascii_lowercase();
            if yes_matches {
                Ok((0, 2))
            } else if no_matches {
                Ok((1, 2))
            } else {
                Err(format!(
                    "noul answer '{answer}' must be 'yes' or 'no' (or the declared criteria labels)"
                ))
            }
        }
        Question::Choice { criteria, .. } => {
            let mut option_names: Vec<&String> = criteria.keys().collect();
            option_names.sort();
            option_names
                .iter()
                .position(|name| name.as_str() == answer)
                .map(|index| (index, option_names.len()))
                .ok_or_else(|| format!("choice answer '{answer}' is not one of the criteria"))
        }
        Question::Score { criteria, .. } => criteria
            .iter()
            .position(|level| level == answer)
            .map(|index| (index, criteria.len()))
            .ok_or_else(|| format!("score answer '{answer}' is not one of the declared levels")),
    }
}

/// Parses one Jev-native JSON record.
///
/// Each question object is the serving-contract shape plus an `answer` string.
/// The `answer` key is stripped before the question is parsed so the contract's
/// strict (`deny_unknown_fields`) deserialization still applies.
pub fn parse_record(value: &Value) -> anyhow::Result<TrainingRecord> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("record must be a JSON object"))?;
    let state_value = object
        .get("state")
        .ok_or_else(|| anyhow::anyhow!("record is missing 'state'"))?
        .clone();
    let state: Content = serde_json::from_value(state_value)
        .map_err(|error| anyhow::anyhow!("invalid 'state': {error}"))?;

    let questions_value = object
        .get("questions")
        .ok_or_else(|| anyhow::anyhow!("record is missing 'questions'"))?;
    let questions_object = questions_value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("'questions' must be an object"))?;
    if questions_object.is_empty() {
        return Err(anyhow::anyhow!("'questions' must not be empty"));
    }

    let mut questions: BTreeMap<String, TrainingQuestion> = BTreeMap::new();
    for (identifier, question_value) in questions_object {
        let question_object = question_value
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("question '{identifier}' must be a JSON object"))?;
        let answer = question_object
            .get("answer")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("question '{identifier}' is missing a string 'answer'"))?
            .to_string();
        let mut stripped = question_object.clone();
        stripped.remove("answer");
        let question: Question = serde_json::from_value(Value::Object(stripped))
            .map_err(|error| anyhow::anyhow!("question '{identifier}' is invalid: {error}"))?;
        questions.insert(identifier.clone(), TrainingQuestion { question, answer });
    }

    Ok(TrainingRecord { state, questions })
}

/// Expands records into one training item per answered question.
///
/// Records keep their input order; within a record the questions follow their
/// sorted identifier order (the map is a `BTreeMap`), which makes the expansion
/// reproducible.
pub fn expand_records(records: &[TrainingRecord]) -> anyhow::Result<Vec<TrainingItem>> {
    let mut items: Vec<TrainingItem> = Vec::new();
    for record in records {
        for (identifier, answered) in &record.questions {
            let (answer_index, label_count) =
                answer_index_for(&answered.question, &answered.answer)
                    .map_err(|error| anyhow::anyhow!("question '{identifier}': {error}"))?;
            items.push(TrainingItem {
                state: record.state.clone(),
                question_identifier: identifier.clone(),
                question: answered.question.clone(),
                answer: answered.answer.clone(),
                answer_index,
                answer_label: label_sequence(answer_index),
                label_count,
            });
        }
    }
    if items.is_empty() {
        return Err(anyhow::anyhow!("dataset produced no training items"));
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record_value() -> Value {
        serde_json::json!({
            "state": "charged twice",
            "questions": {
                "refund": {"type": "noul", "instructions": "Refund?", "answer": "yes"},
                "dept": {
                    "type": "choice", "instructions": "Dept?",
                    "criteria": {"billing": "Payments", "technical": "Bugs"},
                    "answer": "technical"
                },
                "urg": {
                    "type": "score", "instructions": "Urgent?",
                    "criteria": ["Routine", "Urgent", "Emergency"],
                    "answer": "Urgent"
                }
            }
        })
    }

    #[test]
    fn parses_all_three_question_types_with_answers() -> anyhow::Result<()> {
        let record = parse_record(&record_value())?;
        assert_eq!(record.questions.len(), 3);
        let refund = record
            .questions
            .get("refund")
            .ok_or_else(|| anyhow::anyhow!("missing refund question"))?;
        assert_eq!(refund.answer, "yes");
        Ok(())
    }

    #[test]
    fn a_missing_answer_is_an_error() {
        let value = serde_json::json!({
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "Q?"}}
        });
        assert!(parse_record(&value).is_err());
    }

    #[test]
    fn an_unknown_question_type_is_an_error() {
        let value = serde_json::json!({
            "state": "x",
            "questions": {"q": {"type": "unknown", "instructions": "Q?", "answer": "yes"}}
        });
        assert!(parse_record(&value).is_err());
    }

    #[test]
    fn empty_questions_is_an_error() {
        let value = serde_json::json!({"state": "x", "questions": {}});
        assert!(parse_record(&value).is_err());
    }

    #[test]
    fn expansion_maps_answers_to_spreadsheet_labels() -> anyhow::Result<()> {
        let record = parse_record(&record_value())?;
        let items = expand_records(&[record])?;
        assert_eq!(items.len(), 3);
        // Questions are expanded in sorted identifier order: dept, refund, urg.
        let by_identifier = |identifier: &str| -> anyhow::Result<&TrainingItem> {
            items
                .iter()
                .find(|item| item.question_identifier == identifier)
                .ok_or_else(|| anyhow::anyhow!("missing item '{identifier}'"))
        };
        let dept = by_identifier("dept")?;
        // Sorted option names: billing (A), technical (B).
        assert_eq!(dept.answer_label, "B");
        assert_eq!(dept.answer_index, 1);
        assert_eq!(dept.label_count, 2);
        let refund = by_identifier("refund")?;
        assert_eq!(refund.answer_label, "A");
        let urgent = by_identifier("urg")?;
        assert_eq!(urgent.answer_label, "B");
        assert_eq!(urgent.label_count, 3);
        Ok(())
    }

    #[test]
    fn an_answer_outside_the_criteria_is_an_error() -> anyhow::Result<()> {
        let value = serde_json::json!({
            "state": "x",
            "questions": {
                "dept": {
                    "type": "choice", "instructions": "Dept?",
                    "criteria": {"billing": null, "technical": null},
                    "answer": "shipping"
                }
            }
        });
        let record = parse_record(&value)?;
        let result = expand_records(&[record]);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn custom_noul_criteria_labels_are_accepted() -> anyhow::Result<()> {
        let value = serde_json::json!({
            "state": "x",
            "questions": {
                "q": {
                    "type": "noul", "instructions": "Q?",
                    "criteria": {"yes": "Approved", "no": "Rejected"},
                    "answer": "Rejected"
                }
            }
        });
        let record = parse_record(&value)?;
        let items = expand_records(&[record])?;
        assert_eq!(items[0].answer_label, "B");
        Ok(())
    }
}
