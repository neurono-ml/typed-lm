use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Structured(serde_json::Value),
}

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

impl Default for NoulCriteria {
    fn default() -> Self {
        Self {
            yes: default_yes(),
            no: default_no(),
        }
    }
}

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
    fn parses_all_three_question_types() {
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
        let req: SystemOneRequest = serde_json::from_value(raw).unwrap();
        assert_eq!(req.questions.len(), 3);
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

    #[test]
    fn empty_choice_criteria_is_rejected() {
        let req = request_with(serde_json::json!({
            "c": {"type": "choice", "instructions": "Q?", "criteria": {}}
        }));
        assert!(req.validate().is_err());
    }

    #[test]
    fn score_requires_between_two_and_ten_levels() {
        let one = request_with(serde_json::json!({
            "s": {"type": "score", "instructions": "Q?", "criteria": ["only"]}
        }));
        assert!(one.validate().is_err());
        let eleven = request_with(serde_json::json!({
            "s": {"type": "score", "instructions": "Q?",
                  "criteria": ["0","1","2","3","4","5","6","7","8","9","10"]}
        }));
        assert!(eleven.validate().is_err());
        let three = request_with(serde_json::json!({
            "s": {"type": "score", "instructions": "Q?",
                  "criteria": ["low", "mid", "high"]}
        }));
        assert!(three.validate().is_ok());
    }

    #[test]
    fn empty_questions_map_is_rejected() {
        let raw = serde_json::json!({
            "model": "jev-latest", "state": "x", "questions": {}
        });
        let req: SystemOneRequest = serde_json::from_value(raw).unwrap();
        assert!(req.validate().is_err());
    }

    fn request_with(questions: serde_json::Value) -> SystemOneRequest {
        let raw = serde_json::json!({
            "model": "jev-latest", "state": "x", "questions": questions
        });
        serde_json::from_value(raw).unwrap()
    }

    #[test]
    fn response_serializes_typed_answers() {
        let mut answers = HashMap::new();
        answers.insert(
            "r".to_string(),
            Answer::Noul(NoulAnswer {
                kind: "noul".to_string(),
                noul: 0.9,
            }),
        );
        let resp = SystemOneResponse {
            model: "jev-latest".to_string(),
            answers,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 2,
            },
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["answers"]["r"]["type"], "noul");
    }
}
