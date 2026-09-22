use rig_core::message::{AssistantContent, Message, UserContent};
use std::collections::HashMap;

use crate::classifier::Classifier;
use crate::evaluation_error::EvaluationError;
use crate::jev::{
    Answer, ChoiceAnswer, Content, NoulAnswer, Question, ScoreAnswer, SystemOneRequest,
    SystemOneResponse, Usage,
};
use crate::model::LanguageModel;

/// Softmax temperature applied to every answer distribution.
const SCORING_TEMPERATURE: f32 = 1.0;
/// Probability reported by [`MockEvaluator`] for boolean questions.
#[cfg(test)]
const MOCK_NOUL_PROBABILITY: f32 = 0.75;

/// Abstraction over the evaluation pipeline.
///
/// Production code uses [`CandleEvaluator`] (real weights); API tests and
/// CI use [`MockEvaluator`] so no heavy model download is required.
pub trait Evaluator: Send + Sync {
    fn evaluate(&self, request: &SystemOneRequest) -> Result<SystemOneResponse, EvaluationError>;
}

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

/// Calibrates raw label logits into a probability distribution.
///
/// Boolean questions carry exactly two labels and go through the binary
/// softmax; larger label sets use the temperature-scaled softmax. Both
/// paths compute the same distribution family, so `normalize` and
/// `binary_softmax` agree on two-logit inputs.
///
/// The scoring intentionally reads Candle logits: Rig orchestrates the
/// conversation history (preamble plus user message) but does not expose
/// logprobs, so the single forward pass distribution always comes from Candle.
pub fn calibrate_probabilities(logit_values: &[f32]) -> Vec<f32> {
    if logit_values.len() == 2 {
        let classification = Classifier::binary_softmax(logit_values[0], logit_values[1]);
        return vec![classification.true_prob, classification.false_prob];
    }
    Classifier::normalize(logit_values, SCORING_TEMPERATURE)
}

/// Accepts a model name when it matches the served model or a Jev alias.
pub fn is_supported_model(requested_model: &str, served_model_name: &str) -> bool {
    requested_model == served_model_name || requested_model.starts_with("jev-")
}

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

/// Builds the Rig conversation history: the loaded context as preamble and
/// one user message carrying the evaluated state plus the question text.
pub fn build_conversation_history(
    loaded_context: &str,
    state_text: &str,
    question_text: &str,
) -> Vec<Message> {
    vec![
        Message::system(loaded_context.to_string()),
        Message::user(format!(
            "State:\n{state_text}\n\nQuestion:\n{question_text}"
        )),
    ]
}

/// Extracts the plain text carried by one Rig message.
pub fn message_text(message: &Message) -> String {
    match message {
        Message::System { content } => content.clone(),
        Message::User { content } => content
            .iter()
            .filter_map(|item| match item {
                UserContent::Text(text) => Some(text.text.clone()),
                UserContent::ToolResult(_) => None,
                UserContent::Image(_)
                | UserContent::Audio(_)
                | UserContent::Video(_)
                | UserContent::Document(_) => None,
            })
            .collect::<Vec<String>>()
            .join("\n"),
        Message::Assistant { content, .. } => content
            .iter()
            .filter_map(|item| match item {
                AssistantContent::Text(text) => Some(text.text.clone()),
                AssistantContent::ToolCall(_)
                | AssistantContent::Reasoning(_)
                | AssistantContent::Image(_) => None,
            })
            .collect::<Vec<String>>()
            .join("\n"),
    }
}

/// Renders the Rig conversation history into the single prompt evaluated by Candle.
pub fn render_prompt(history: &[Message]) -> String {
    let mut system_section = String::new();
    let mut user_sections: Vec<String> = Vec::new();
    for message in history {
        match message {
            Message::System { .. } => {
                if system_section.is_empty() {
                    system_section = message_text(message);
                } else {
                    system_section = format!("{system_section}\n{}", message_text(message));
                }
            }
            Message::User { .. } | Message::Assistant { .. } => {
                user_sections.push(message_text(message));
            }
        }
    }
    LanguageModel::full_prompt(&system_section, &user_sections.join("\n"))
}

fn build_noul_answer(probabilities: &[f32]) -> Answer {
    let noul_probability = probabilities.first().copied().unwrap_or(0.0);
    Answer::Noul(NoulAnswer {
        kind: "noul".to_string(),
        noul: noul_probability,
    })
}

fn build_choice_answer(option_names: &[String], probabilities: &[f32]) -> Answer {
    let mut best_index = 0;
    for (candidate_index, probability) in probabilities.iter().enumerate() {
        if *probability > probabilities[best_index] {
            best_index = candidate_index;
        }
    }
    let distribution: HashMap<String, f32> = option_names
        .iter()
        .enumerate()
        .map(|(option_index, option_name)| {
            (
                option_name.clone(),
                probabilities.get(option_index).copied().unwrap_or(0.0),
            )
        })
        .collect();
    let choice = option_names.get(best_index).cloned().unwrap_or_default();
    Answer::Choice(ChoiceAnswer {
        kind: "choice".to_string(),
        choice,
        probabilities: distribution,
        confidence: Classifier::confidence(probabilities),
    })
}

fn build_score_answer(level_names: &[String], probabilities: &[f32]) -> Answer {
    let legend: HashMap<String, String> = level_names
        .iter()
        .enumerate()
        .map(|(level_index, level_name)| (level_index.to_string(), level_name.clone()))
        .collect();
    let distribution: HashMap<String, f32> = level_names
        .iter()
        .enumerate()
        .map(|(level_index, _)| {
            (
                level_index.to_string(),
                probabilities.get(level_index).copied().unwrap_or(0.0),
            )
        })
        .collect();
    Answer::Score(ScoreAnswer {
        kind: "score".to_string(),
        score: Classifier::expected_score(probabilities),
        legend,
        probabilities: distribution,
        confidence: Classifier::confidence(probabilities),
    })
}

/// Builds a typed answer from a calibrated distribution for one question.
pub fn answer_from_probabilities(question: &Question, probabilities: &[f32]) -> Answer {
    match question {
        Question::Noul { .. } => build_noul_answer(probabilities),
        Question::Choice { criteria, .. } => {
            let mut option_names: Vec<String> = criteria.keys().cloned().collect();
            option_names.sort();
            build_choice_answer(&option_names, probabilities)
        }
        Question::Score { criteria, .. } => build_score_answer(criteria, probabilities),
    }
}

/// Counts how many answer labels one question needs (two for boolean questions).
pub fn label_count_for_question(question: &Question) -> usize {
    match question {
        Question::Noul { .. } => 2,
        Question::Choice { criteria, .. } => criteria.len(),
        Question::Score { criteria, .. } => criteria.len(),
    }
}

/// Real evaluator: Rig-orchestrated prompts scored with Candle logits.
///
/// The loaded context is fixed at startup; every question clones the
/// conversation through [`LanguageModel::forward_full`], which builds a fresh
/// key-value cache per call, so per-request isolation holds by construction.
pub struct CandleEvaluator<'model_lifetime> {
    language_model: &'model_lifetime LanguageModel,
    loaded_context: String,
    served_model_name: String,
}

impl<'model_lifetime> CandleEvaluator<'model_lifetime> {
    pub fn new(
        language_model: &'model_lifetime LanguageModel,
        loaded_context: String,
        served_model_name: String,
    ) -> Self {
        Self {
            language_model,
            loaded_context,
            served_model_name,
        }
    }

    fn evaluate_single_question(
        &self,
        question: &Question,
        state_text: &str,
    ) -> Result<(Answer, usize), EvaluationError> {
        let question_text = render_question_text(question);
        let history = build_conversation_history(&self.loaded_context, state_text, &question_text);
        let prompt = render_prompt(&history);
        let input_tokens = self.language_model.token_count(&prompt, true);
        let (logits_tensor, _) = self
            .language_model
            .forward_full(&prompt)
            .map_err(EvaluationError::from)?;
        let labels = option_labels(label_count_for_question(question));
        let mut logit_values: Vec<f32> = Vec::with_capacity(labels.len());
        for label in &labels {
            let token_identifier = self.language_model.label_token_id(label).ok_or_else(|| {
                EvaluationError::inference(format!(
                    "answer label '{label}' is not a single vocabulary token"
                ))
            })?;
            let logit_value = self
                .language_model
                .token_logit(&logits_tensor, token_identifier)
                .map_err(EvaluationError::from)?;
            logit_values.push(logit_value);
        }
        let probabilities = calibrate_probabilities(&logit_values);
        Ok((
            answer_from_probabilities(question, &probabilities),
            input_tokens,
        ))
    }
}

impl Evaluator for CandleEvaluator<'_> {
    fn evaluate(&self, request: &SystemOneRequest) -> Result<SystemOneResponse, EvaluationError> {
        request
            .validate()
            .map_err(EvaluationError::invalid_request)?;
        if !is_supported_model(&request.model, &self.served_model_name) {
            return Err(EvaluationError::unknown_model(format!(
                "model '{}' is not served (serving '{}')",
                request.model, self.served_model_name
            )));
        }
        let state_text = content_to_text(&request.state);
        let mut ordered_identifiers: Vec<&String> = request.questions.keys().collect();
        ordered_identifiers.sort();
        let mut answers: HashMap<String, Answer> = HashMap::with_capacity(request.questions.len());
        let mut input_tokens = 0_usize;
        for identifier in ordered_identifiers {
            let question = request.questions.get(identifier).ok_or_else(|| {
                EvaluationError::invalid_request(format!(
                    "question '{identifier}' disappeared during evaluation"
                ))
            })?;
            let (answer, question_tokens) = self.evaluate_single_question(question, &state_text)?;
            answers.insert(identifier.clone(), answer);
            input_tokens = input_tokens.saturating_add(question_tokens);
        }
        Ok(SystemOneResponse {
            model: request.model.clone(),
            answers,
            usage: Usage {
                input_tokens,
                output_tokens: request.questions.len().saturating_add(1),
            },
        })
    }
}

/// Predictable test double: returns fixed answers without loading weights.
///
/// API tests depend on this evaluator instead of [`CandleEvaluator`],
/// keeping CI free of heavy model downloads. It is compiled only for
/// tests because this is a binary crate: no external target can import it.
#[cfg(test)]
pub struct MockEvaluator {
    served_model_name: String,
}

#[cfg(test)]
impl MockEvaluator {
    pub fn new(served_model_name: String) -> Self {
        Self { served_model_name }
    }

    fn mock_answer(&self, question: &Question) -> Answer {
        match question {
            Question::Noul { .. } => Answer::Noul(NoulAnswer {
                kind: "noul".to_string(),
                noul: MOCK_NOUL_PROBABILITY,
            }),
            Question::Choice { criteria, .. } => {
                let mut option_names: Vec<String> = criteria.keys().cloned().collect();
                option_names.sort();
                let option_count = option_names.len().max(1);
                let uniform = 1.0 / option_count as f32;
                let distribution: HashMap<String, f32> = option_names
                    .iter()
                    .map(|option_name| (option_name.clone(), uniform))
                    .collect();
                let choice = option_names.first().cloned().unwrap_or_default();
                Answer::Choice(ChoiceAnswer {
                    kind: "choice".to_string(),
                    choice,
                    probabilities: distribution,
                    confidence: 0.0,
                })
            }
            Question::Score { criteria, .. } => {
                let legend: HashMap<String, String> = criteria
                    .iter()
                    .enumerate()
                    .map(|(level_index, level_name)| (level_index.to_string(), level_name.clone()))
                    .collect();
                let option_count = criteria.len().max(1);
                let uniform = 1.0 / option_count as f32;
                let distribution: HashMap<String, f32> = criteria
                    .iter()
                    .enumerate()
                    .map(|(level_index, _)| (level_index.to_string(), uniform))
                    .collect();
                Answer::Score(ScoreAnswer {
                    kind: "score".to_string(),
                    score: (option_count as f32 - 1.0) / 2.0,
                    legend,
                    probabilities: distribution,
                    confidence: 0.0,
                })
            }
        }
    }
}

#[cfg(test)]
impl Evaluator for MockEvaluator {
    fn evaluate(&self, request: &SystemOneRequest) -> Result<SystemOneResponse, EvaluationError> {
        request
            .validate()
            .map_err(EvaluationError::invalid_request)?;
        if !is_supported_model(&request.model, &self.served_model_name) {
            return Err(EvaluationError::unknown_model(format!(
                "model '{}' is not served (serving '{}')",
                request.model, self.served_model_name
            )));
        }
        let mut answers: HashMap<String, Answer> = HashMap::with_capacity(request.questions.len());
        for (identifier, question) in &request.questions {
            answers.insert(identifier.clone(), self.mock_answer(question));
        }
        Ok(SystemOneResponse {
            model: request.model.clone(),
            answers,
            usage: Usage {
                input_tokens: request.questions.len().saturating_mul(10),
                output_tokens: request.questions.len().saturating_add(1),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::ResponseError as _;
    use candle_core::{Device, Tensor};

    const EPSILON: f32 = 1e-5;

    fn request_with_model(model: &str, questions: serde_json::Value) -> SystemOneRequest {
        let raw = serde_json::json!({
            "model": model,
            "state": "charged twice",
            "questions": questions
        });
        serde_json::from_value(raw).unwrap()
    }

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
    fn calibration_from_simulated_tensor_sums_to_one() {
        let device = Device::Cpu;
        // Simulated last-position logits shaped (1, 3): batch 1, vocabulary 3.
        let simulated = Tensor::new(vec![vec![2.0f32, 1.0, 0.5]], &device).unwrap();
        let logit_values = vec![
            LanguageModel::extract_logit(&simulated, 0).unwrap(),
            LanguageModel::extract_logit(&simulated, 1).unwrap(),
            LanguageModel::extract_logit(&simulated, 2).unwrap(),
        ];
        let probabilities = calibrate_probabilities(&logit_values);
        let total: f32 = probabilities.iter().sum();
        assert!((total - 1.0).abs() < EPSILON);
        assert!(probabilities[0] > probabilities[1]);
        assert!(probabilities[1] > probabilities[2]);
    }

    #[test]
    fn calibration_matches_hand_computed_softmax() {
        let probabilities = calibrate_probabilities(&[2.0, 0.0]);
        assert!((probabilities[0] - 0.8808).abs() < 1e-4);
        assert!((probabilities[1] - 0.1192).abs() < 1e-4);
    }

    #[test]
    fn conversation_history_carries_context_state_and_question() {
        let history = build_conversation_history("shop rules", "order total", "refund?");
        assert_eq!(history.len(), 2);
        assert!(message_text(&history[0]).contains("shop rules"));
        let user_text = message_text(&history[1]);
        assert!(user_text.contains("order total"));
        assert!(user_text.contains("refund?"));
    }

    #[test]
    fn rendered_prompt_keeps_system_before_user() {
        let history = build_conversation_history("shop rules", "order total", "refund?");
        let prompt = render_prompt(&history);
        let system_position = prompt.find("shop rules").unwrap();
        let user_position = prompt.find("order total").unwrap();
        assert!(system_position < user_position);
    }

    #[test]
    fn supported_models_match_served_name_or_jev_prefix() {
        assert!(is_supported_model("manaca-1", "manaca-1"));
        assert!(is_supported_model("jev-latest", "manaca-1"));
        assert!(!is_supported_model("ghost-model", "manaca-1"));
    }

    #[test]
    fn noul_answer_reports_probability_of_first_label() {
        let raw = serde_json::json!({
            "refund": {"type": "noul", "instructions": "Refund?"}
        });
        let request = request_with_model("jev-latest", raw);
        let question = request.questions.get("refund").unwrap();
        let answer = answer_from_probabilities(question, &[0.8, 0.2]);
        match answer {
            Answer::Noul(noul) => assert!((noul.noul - 0.8).abs() < EPSILON),
            other => panic!("expected noul answer, got {other:?}"),
        }
    }

    #[test]
    fn choice_answer_selects_highest_probability_option() {
        let raw = serde_json::json!({
            "department": {
                "type": "choice",
                "instructions": "Route?",
                "criteria": {"billing": "Payments", "technical": "Bugs"}
            }
        });
        let request = request_with_model("jev-latest", raw);
        let question = request.questions.get("department").unwrap();
        // Sorted option names are [billing, technical]; favor the second one.
        let answer = answer_from_probabilities(question, &[0.25, 0.75]);
        match answer {
            Answer::Choice(choice) => {
                assert_eq!(choice.choice, "technical");
                assert!((choice.probabilities["technical"] - 0.75).abs() < EPSILON);
            }
            other => panic!("expected choice answer, got {other:?}"),
        }
    }

    #[test]
    fn score_answer_returns_expected_value_with_legend() {
        let raw = serde_json::json!({
            "urgency": {
                "type": "score",
                "instructions": "Urgent?",
                "criteria": ["Routine", "Urgent", "Emergency"]
            }
        });
        let request = request_with_model("jev-latest", raw);
        let question = request.questions.get("urgency").unwrap();
        let answer = answer_from_probabilities(question, &[0.0, 0.0, 1.0]);
        match answer {
            Answer::Score(score) => {
                assert!((score.score - 2.0).abs() < EPSILON);
                assert_eq!(score.legend["0"], "Routine");
                assert_eq!(score.legend["2"], "Emergency");
            }
            other => panic!("expected score answer, got {other:?}"),
        }
    }

    #[test]
    fn mock_evaluator_answers_every_question_type() {
        let evaluator = MockEvaluator::new("manaca-1".to_string());
        let request = request_with_model(
            "manaca-1",
            serde_json::json!({
                "refund": {"type": "noul", "instructions": "Refund?"},
                "department": {
                    "type": "choice",
                    "instructions": "Route?",
                    "criteria": {"billing": "Payments", "technical": "Bugs"}
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Urgent?",
                    "criteria": ["Routine", "Urgent", "Emergency"]
                }
            }),
        );
        let response = evaluator.evaluate(&request).unwrap();
        assert_eq!(response.answers.len(), 3);
        assert_eq!(response.usage.output_tokens, 4);
        match response.answers.get("refund").unwrap() {
            Answer::Noul(noul) => {
                assert!((noul.noul - MOCK_NOUL_PROBABILITY).abs() < EPSILON);
            }
            other => panic!("expected noul answer, got {other:?}"),
        }
    }

    #[test]
    fn mock_evaluator_rejects_invalid_requests() {
        let evaluator = MockEvaluator::new("manaca-1".to_string());
        let request = request_with_model(
            "manaca-1",
            serde_json::json!({
                "department": {
                    "type": "choice", "instructions": "Route?", "criteria": {}
                }
            }),
        );
        let error = evaluator.evaluate(&request).unwrap_err();
        assert_eq!(
            error.status_code(),
            actix_web::http::StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    #[test]
    fn mock_evaluator_rejects_unknown_models() {
        let evaluator = MockEvaluator::new("manaca-1".to_string());
        let request = request_with_model(
            "ghost-model",
            serde_json::json!({"refund": {"type": "noul", "instructions": "Refund?"}}),
        );
        let error = evaluator.evaluate(&request).unwrap_err();
        match error {
            EvaluationError::UnknownModel(_) => {}
            other => panic!("expected unknown model error, got {other:?}"),
        }
    }

    #[test]
    fn mock_evaluator_accepts_jev_prefixed_models() {
        let evaluator = MockEvaluator::new("manaca-1".to_string());
        let request = request_with_model(
            "jev-experimental",
            serde_json::json!({"refund": {"type": "noul", "instructions": "Refund?"}}),
        );
        assert!(evaluator.evaluate(&request).is_ok());
    }
}
