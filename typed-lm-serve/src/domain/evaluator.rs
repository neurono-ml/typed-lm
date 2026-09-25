use rig_core::message::{AssistantContent, Message, UserContent};
use std::collections::HashMap;

#[cfg(test)]
use crate::api::dtos::Usage;
use crate::api::dtos::{
    Answer, ChoiceAnswer, NoulAnswer, Question, ScoreAnswer, SystemOneRequest, SystemOneResponse,
};
use crate::api::error::EvaluationError;
use typed_lm_common::classifier::Classifier;
use typed_lm_common::prompt_template::PromptTemplate;

#[cfg(test)]
use typed_lm_common::labels::{calibrate_probabilities, label_sequence, option_labels};

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

/// Length of the longest token prefix shared by every sequence.
///
/// This is the exact number of tokens that can be prefilled once into a
/// key/value cache and reused across all the sequences (the "single broadcast
/// prefill" of parallel evaluation). Returns `0` for an empty slice.
pub fn longest_common_prefix(sequences: &[Vec<u32>]) -> usize {
    let Some(first) = sequences.first() else {
        return 0;
    };
    let mut shared_length = first.len();
    for sequence in &sequences[1..] {
        let mut matching = 0;
        while matching < shared_length
            && matching < sequence.len()
            && first[matching] == sequence[matching]
        {
            matching += 1;
        }
        shared_length = matching;
        if shared_length == 0 {
            break;
        }
    }
    shared_length
}

/// Accepts a request when its model name matches the served model name.
pub fn is_supported_model(requested_model: &str, served_model_name: &str) -> bool {
    requested_model == served_model_name
}

/// Builds the Rig conversation history: the loaded context as preamble and
/// one user message carrying the evaluated state plus the question text.
///
/// Retained as the reference (monolithic) renderer: the production path builds
/// the prompt through [`PromptTemplate::state_prefix`] and
/// [`PromptTemplate::question_completion`], and the equivalence between both
/// renderings is asserted in tests.
#[allow(dead_code)]
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
#[allow(dead_code)]
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
///
/// Reference (monolithic) renderer; see [`build_conversation_history`].
#[allow(dead_code)]
pub fn render_prompt(history: &[Message], template: PromptTemplate) -> String {
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
    template.full_prompt(&system_section, &user_sections.join("\n"))
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

/// Predictable test double: returns fixed answers without loading weights.
///
/// API tests depend on this evaluator instead of `CandleEvaluator`,
/// keeping CI free of heavy model downloads.
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
    use crate::api::dtos::{Answer, SystemOneRequest};
    use crate::api::error::EvaluationError;
    use crate::infrastructure::language_model::LanguageModel;
    use actix_web::ResponseError as _;
    use candle_core::{Device, Tensor};

    const EPSILON: f32 = 1e-5;

    fn request_with_model(
        model: &str,
        questions: serde_json::Value,
    ) -> anyhow::Result<SystemOneRequest> {
        let raw = serde_json::json!({
            "model": model,
            "state": "charged twice",
            "questions": questions
        });
        Ok(serde_json::from_value(raw)?)
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
    fn longest_common_prefix_counts_shared_leading_tokens() {
        let sequences = vec![vec![1, 2, 3, 4], vec![1, 2, 9], vec![1, 2, 3, 0]];
        assert_eq!(longest_common_prefix(&sequences), 2);
        assert_eq!(longest_common_prefix(&[vec![7, 7, 7]]), 3);
        assert_eq!(longest_common_prefix(&[vec![1, 2], vec![3, 4]]), 0);
        assert_eq!(longest_common_prefix(&[]), 0);
    }

    #[test]
    fn longest_common_prefix_handles_nested_sequences() {
        let sequences = vec![vec![5, 6], vec![5, 6, 7, 8]];
        assert_eq!(longest_common_prefix(&sequences), 2);
    }

    #[test]
    fn calibration_from_simulated_tensor_sums_to_one() -> anyhow::Result<()> {
        let device = Device::Cpu;
        // Simulated last-position logits shaped (1, 3): batch 1, vocabulary 3.
        let simulated = Tensor::new(vec![vec![2.0f32, 1.0, 0.5]], &device)?;
        let logit_values = vec![
            LanguageModel::extract_logit(&simulated, 0)?,
            LanguageModel::extract_logit(&simulated, 1)?,
            LanguageModel::extract_logit(&simulated, 2)?,
        ];
        let probabilities = calibrate_probabilities(&logit_values);
        let total: f32 = probabilities.iter().sum();
        assert!((total - 1.0).abs() < EPSILON);
        assert!(probabilities[0] > probabilities[1]);
        assert!(probabilities[1] > probabilities[2]);
        Ok(())
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
        let prompt = render_prompt(&history, PromptTemplate::Manaca);
        let system_position = prompt.find("shop rules").unwrap_or(usize::MAX);
        let user_position = prompt.find("order total").unwrap_or(usize::MAX);
        assert_ne!(system_position, usize::MAX);
        assert_ne!(user_position, usize::MAX);
        assert!(system_position < user_position);
    }

    #[test]
    fn rendered_chatml_prompt_embeds_the_context_verbatim() {
        let context = "Fact 7: The logistics department handles damaged shipments.";
        let history = build_conversation_history(context, "smashed box", "Who handles it?");
        let prompt = render_prompt(&history, PromptTemplate::ChatMl);
        assert!(
            prompt.contains(context),
            "the loaded context must appear verbatim in the rendered prompt: {prompt}"
        );
        assert!(prompt.starts_with("<|im_start|>system\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn rendered_manaca_family_prompt_embeds_the_context_verbatim() {
        let context = "Fact 4: a double charge is always fully refundable.";
        let history = build_conversation_history(context, "billed twice", "Refund?");
        let prompt = render_prompt(&history, PromptTemplate::Manaca);
        assert!(
            prompt.contains(context),
            "the loaded context must appear verbatim in the rendered prompt: {prompt}"
        );
    }

    #[test]
    fn system_prompt_is_a_token_prefix_of_the_full_prompt() {
        // Absolute prerequisite for prefix reuse: the system prompt (prefilled
        // once at startup) must be a literal character prefix of every full
        // prompt, for both templates.
        for template in [PromptTemplate::Manaca, PromptTemplate::ChatMl] {
            let context = "loaded memory context";
            let system = template.system_prompt(context);
            let full = template.full_prompt(context, "State:\n...\n\nQuestion:\n...");
            assert!(
                full.starts_with(&system),
                "system prompt must prefix the full prompt for prefix reuse"
            );
        }
    }

    #[test]
    fn supported_models_match_the_served_name() {
        assert!(is_supported_model("typed-lm-1", "typed-lm-1"));
        assert!(!is_supported_model("typedef-lm", "typed-lm-1"));
        assert!(!is_supported_model("ghost-model", "typed-lm-1"));
    }

    #[test]
    fn noul_answer_reports_probability_of_first_label() -> anyhow::Result<()> {
        let raw = serde_json::json!({
            "refund": {"type": "noul", "instructions": "Refund?"}
        });
        let request = request_with_model("typed-lm", raw)?;
        let missing_question = anyhow::anyhow!("missing question 'refund'");
        let question = request.questions.get("refund").ok_or(missing_question)?;
        let answer = answer_from_probabilities(question, &[0.8, 0.2]);
        match answer {
            Answer::Noul(noul) => assert!((noul.noul - 0.8).abs() < EPSILON),
            other => panic!("expected noul answer, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn choice_answer_selects_highest_probability_option() -> anyhow::Result<()> {
        let raw = serde_json::json!({
            "department": {
                "type": "choice",
                "instructions": "Route?",
                "criteria": {"billing": "Payments", "technical": "Bugs"}
            }
        });
        let request = request_with_model("typed-lm", raw)?;
        let missing_question = anyhow::anyhow!("missing question 'department'");
        let question = request
            .questions
            .get("department")
            .ok_or(missing_question)?;
        // Sorted option names are [billing, technical]; favor the second one.
        let answer = answer_from_probabilities(question, &[0.25, 0.75]);
        match answer {
            Answer::Choice(choice) => {
                assert_eq!(choice.choice, "technical");
                assert!((choice.probabilities["technical"] - 0.75).abs() < EPSILON);
            }
            other => panic!("expected choice answer, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn score_answer_returns_expected_value_with_legend() -> anyhow::Result<()> {
        let raw = serde_json::json!({
            "urgency": {
                "type": "score",
                "instructions": "Urgent?",
                "criteria": ["Routine", "Urgent", "Emergency"]
            }
        });
        let request = request_with_model("typed-lm", raw)?;
        let missing_question = anyhow::anyhow!("missing question 'urgency'");
        let question = request.questions.get("urgency").ok_or(missing_question)?;
        let answer = answer_from_probabilities(question, &[0.0, 0.0, 1.0]);
        match answer {
            Answer::Score(score) => {
                assert!((score.score - 2.0).abs() < EPSILON);
                assert_eq!(score.legend["0"], "Routine");
                assert_eq!(score.legend["2"], "Emergency");
            }
            other => panic!("expected score answer, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn mock_evaluator_answers_every_question_type() -> anyhow::Result<()> {
        let evaluator = MockEvaluator::new("typed-lm-1".to_string());
        let request = request_with_model(
            "typed-lm-1",
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
        )?;
        let response = evaluator
            .evaluate(&request)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(response.answers.len(), 3);
        assert_eq!(response.usage.output_tokens, 4);
        let missing_answer = anyhow::anyhow!("missing answer 'refund'");
        match response.answers.get("refund").ok_or(missing_answer)? {
            Answer::Noul(noul) => {
                assert!((noul.noul - MOCK_NOUL_PROBABILITY).abs() < EPSILON);
            }
            other => panic!("expected noul answer, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn mock_evaluator_rejects_invalid_requests() -> anyhow::Result<()> {
        let evaluator = MockEvaluator::new("typed-lm-1".to_string());
        let request = request_with_model(
            "typed-lm-1",
            serde_json::json!({
                "department": {
                    "type": "choice", "instructions": "Route?", "criteria": {}
                }
            }),
        )?;
        let result = evaluator.evaluate(&request);
        assert!(result.is_err());
        let Err(error) = result else {
            return Ok(());
        };
        assert_eq!(
            error.status_code(),
            actix_web::http::StatusCode::UNPROCESSABLE_ENTITY
        );
        Ok(())
    }

    #[test]
    fn mock_evaluator_rejects_unknown_models() -> anyhow::Result<()> {
        let evaluator = MockEvaluator::new("typed-lm-1".to_string());
        let request = request_with_model(
            "ghost-model",
            serde_json::json!({"refund": {"type": "noul", "instructions": "Refund?"}}),
        )?;
        let result = evaluator.evaluate(&request);
        assert!(result.is_err());
        let Err(error) = result else {
            return Ok(());
        };
        match error {
            EvaluationError::UnknownModel(_) => {}
            other => panic!("expected unknown model error, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn mock_evaluator_accepts_only_the_served_model_name() -> anyhow::Result<()> {
        let evaluator = MockEvaluator::new("typed-lm-1".to_string());

        let accepted = request_with_model(
            "typed-lm-1",
            serde_json::json!({"refund": {"type": "noul", "instructions": "Refund?"}}),
        )?;
        assert!(evaluator.evaluate(&accepted).is_ok());

        let rejected = request_with_model(
            "jev-experimental",
            serde_json::json!({"refund": {"type": "noul", "instructions": "Refund?"}}),
        )?;
        let result = evaluator.evaluate(&rejected);
        let Err(error) = result else {
            return Err(anyhow::anyhow!("an unrelated model name must be rejected"));
        };
        match error {
            EvaluationError::UnknownModel(_) => {}
            other => panic!("expected unknown model error, got {other:?}"),
        }
        Ok(())
    }
}
