//! Jev request contract shared by the serving and training crates.
//!
//! These types are the request half of the official Jev contract
//! (`POST /v1/systemone`): the evaluated `state`, the served `model` name and
//! the typed `questions` map. The response half lives in `typed-lm-serve`, as
//! it is only produced by the server.

use serde::Deserialize;
use std::collections::HashMap;

/// Free-text or structured content accepted by the contract.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Structured(serde_json::Value),
}

/// Positive/negative labels for a boolean (`noul`) question.
#[derive(Debug, Clone, Deserialize)]
pub struct NoulCriteria {
    #[serde(default = "default_yes", alias = "true")]
    pub yes: String,
    #[serde(default = "default_no", alias = "false")]
    pub no: String,
}

fn default_yes() -> String {
    "Yes".to_string()
}

fn default_no() -> String {
    "No".to_string()
}

impl Default for NoulCriteria {
    fn default() -> Self {
        Self {
            yes: default_yes(),
            no: default_no(),
        }
    }
}

/// One typed question belonging to a request.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum Question {
    #[serde(rename = "noul")]
    Noul {
        instructions: Content,
        #[serde(default)]
        criteria: NoulCriteria,
    },
    #[serde(rename = "choice")]
    Choice {
        instructions: Content,
        criteria: HashMap<String, Option<String>>,
    },
    #[serde(rename = "score")]
    Score {
        instructions: Content,
        criteria: Vec<String>,
    },
}

/// Full single-forward-pass evaluation request.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SystemOneRequest {
    pub state: Content,
    pub model: String,
    pub questions: HashMap<String, Question>,
}

impl SystemOneRequest {
    /// Structural validation of the official contract (no quantitative limits).
    pub fn validate(&self) -> Result<(), String> {
        if self.questions.is_empty() {
            return Err("questions must not be empty".to_string());
        }
        for (question_identifier, question) in &self.questions {
            match question {
                Question::Choice { criteria, .. } => {
                    if criteria.is_empty() {
                        return Err(format!(
                            "question '{question_identifier}': choice criteria must not be empty"
                        ));
                    }
                }
                Question::Score { criteria, .. } => {
                    if !(2..=10).contains(&criteria.len()) {
                        return Err(format!(
                            "question '{question_identifier}': score requires between 2 and 10 levels"
                        ));
                    }
                }
                Question::Noul { .. } => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_three_question_types() -> anyhow::Result<()> {
        let raw = serde_json::json!({
            "model": "jev-latest",
            "state": "charged twice",
            "questions": {
                "refund": {"type": "noul", "instructions": "Refund?"},
                "dept": {"type": "choice", "instructions": "Dept?",
                         "criteria": {"billing": "Payments", "technical": "Bugs"}},
                "urg": {"type": "score", "instructions": "Urgent?",
                        "criteria": ["Routine", "Urgent", "Emergency"]}
            }
        });
        let request: SystemOneRequest = serde_json::from_value(raw)?;
        assert_eq!(request.questions.len(), 3);
        Ok(())
    }

    #[test]
    fn rejects_unknown_question_type() {
        let raw = serde_json::json!({
            "model": "jev-latest",
            "state": "x",
            "questions": {"q": {"type": "unknown"}}
        });
        assert!(serde_json::from_value::<SystemOneRequest>(raw).is_err());
    }

    #[test]
    fn instructions_are_required_by_the_official_contract() {
        let raw = serde_json::json!({
            "model": "jev-latest",
            "state": "x",
            "questions": {"q": {"type": "noul"}}
        });
        assert!(serde_json::from_value::<SystemOneRequest>(raw).is_err());
    }

    fn request_with(questions: serde_json::Value) -> anyhow::Result<SystemOneRequest> {
        let raw = serde_json::json!({
            "model": "jev-latest", "state": "x", "questions": questions
        });
        Ok(serde_json::from_value(raw)?)
    }

    #[test]
    fn empty_choice_criteria_is_rejected() -> anyhow::Result<()> {
        let request = request_with(serde_json::json!({
            "c": {"type": "choice", "instructions": "Q?", "criteria": {}}
        }))?;
        assert!(request.validate().is_err());
        Ok(())
    }

    #[test]
    fn score_requires_between_two_and_ten_levels() -> anyhow::Result<()> {
        let one = request_with(serde_json::json!({
            "s": {"type": "score", "instructions": "Q?", "criteria": ["only"]}
        }))?;
        assert!(one.validate().is_err());
        let eleven = request_with(serde_json::json!({
            "s": {"type": "score", "instructions": "Q?",
                  "criteria": ["0","1","2","3","4","5","6","7","8","9","10"]}
        }))?;
        assert!(eleven.validate().is_err());
        let three = request_with(serde_json::json!({
            "s": {"type": "score", "instructions": "Q?",
                  "criteria": ["low", "mid", "high"]}
        }))?;
        assert!(three.validate().is_ok());
        Ok(())
    }

    #[test]
    fn empty_questions_map_is_rejected() -> anyhow::Result<()> {
        let raw = serde_json::json!({
            "model": "jev-latest", "state": "x", "questions": {}
        });
        let request: SystemOneRequest = serde_json::from_value(raw)?;
        assert!(request.validate().is_err());
        Ok(())
    }
}
