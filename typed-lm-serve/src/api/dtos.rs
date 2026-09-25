//! HTTP response DTOs for the Jev contract.
//!
//! The request half of the contract now lives in `typed-lm-common::contract`;
//! these aliases keep the serving code reading naturally while the response
//! types stay server-only.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use typed_lm_common::contract::{Content, NoulCriteria, Question, SystemOneRequest};

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct NoulAnswer {
    #[serde(rename = "type")]
    pub kind: String,
    pub noul: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChoiceAnswer {
    #[serde(rename = "type")]
    pub kind: String,
    pub choice: String,
    pub probabilities: HashMap<String, f32>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoreAnswer {
    #[serde(rename = "type")]
    pub kind: String,
    pub score: f32,
    pub legend: HashMap<String, String>,
    pub probabilities: HashMap<String, f32>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Answer {
    Noul(NoulAnswer),
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: HashMap<String, Answer>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub name: String,
    pub description: String,
    pub release_date: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAIModelEntry {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<OpenAIModelEntry>,
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub startup_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveResponse {
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub message: String,
}

impl ErrorBody {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_three_question_types() -> anyhow::Result<()> {
        let raw = serde_json::json!({
            "model": "typed-lm",
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
            "model": "typed-lm",
            "state": "x",
            "questions": {"q": {"type": "unknown"}}
        });
        assert!(serde_json::from_value::<SystemOneRequest>(raw).is_err());
    }

    #[test]
    fn instructions_are_required_by_the_official_contract() {
        let raw = serde_json::json!({
            "model": "typed-lm",
            "state": "x",
            "questions": {"q": {"type": "noul"}}
        });
        assert!(serde_json::from_value::<SystemOneRequest>(raw).is_err());
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
            "model": "typed-lm", "state": "x", "questions": {}
        });
        let request: SystemOneRequest = serde_json::from_value(raw)?;
        assert!(request.validate().is_err());
        Ok(())
    }

    fn request_with(questions: serde_json::Value) -> anyhow::Result<SystemOneRequest> {
        let raw = serde_json::json!({
            "model": "typed-lm", "state": "x", "questions": questions
        });
        Ok(serde_json::from_value(raw)?)
    }

    #[test]
    fn response_serializes_typed_answers() -> anyhow::Result<()> {
        let mut answers = HashMap::new();
        answers.insert(
            "r".to_string(),
            Answer::Noul(NoulAnswer {
                kind: "noul".to_string(),
                noul: 0.9,
            }),
        );
        let response = SystemOneResponse {
            model: "typed-lm".to_string(),
            answers,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 2,
            },
        };
        let serialized: serde_json::Value = serde_json::to_value(&response)?;
        assert_eq!(serialized["answers"]["r"]["type"], "noul");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Jev single-question contract (POST /v1/noul, /v1/choice, /v1/score)
// ---------------------------------------------------------------------------

/// Shared Jev request envelope: evaluated facts plus validation schema.
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JevRequest {
    pub estado: serde_json::Value,
    pub schema: serde_json::Value,
}

/// Response for POST /v1/noul.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoulResponse {
    pub decisao: bool,
    pub confianca: f32,
}

/// Response for POST /v1/choice.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceResponse {
    pub escolha: String,
    pub confianca: f32,
}

/// Response for POST /v1/score.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreResponse {
    pub pontuacao: f32,
    pub confianca: f32,
}
