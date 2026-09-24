//! Prompt templates per model family.
//!
//! The server classifies answers from a single forward pass, so the prompt
//! layout must match the checkpoint's training format; otherwise the label
//! logits are meaningless. Two templates are supported:
//!
//! - [`PromptTemplate::Manaca`] — the marker format used by the Manacá-1B base
//!   model (`<|system|>` / `<|user|>` / `<|model|>`).
//! - [`PromptTemplate::ChatMl`] — the ChatML format used by Qwen2.5-Instruct
//!   (`<|im_start|>` / `<|im_end|>`), ending on the assistant header.
//!
//! Both satisfy the invariant the prefix-reuse evaluator depends on:
//! `full_prompt = system_prompt + user_prompt`, so the system prompt is a
//! literal token prefix of every question prompt.

use crate::checkpoint::ModelArchitecture;

/// Prompt layout for a model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptTemplate {
    /// Manacá base markers.
    Manaca,
    /// Qwen2.5-Instruct ChatML.
    ChatMl,
}

/// Constant instruction appended to the ChatML system prompt so the model
/// answers with a single option letter instead of free text.
const CHATML_ANSWER_INSTRUCTION: &str = "Answer with a single option letter and nothing else.";

impl PromptTemplate {
    /// Selects the template matching the detected architecture.
    ///
    /// Llama checkpoints in this ecosystem use the Manacá markers; every other
    /// supported dense family (Qwen2/Qwen3, Mistral, Gemma*) is served with the
    /// ChatML template its instruct variants expect.
    pub fn for_architecture(architecture: ModelArchitecture) -> Self {
        match architecture {
            ModelArchitecture::Llama => Self::Manaca,
            ModelArchitecture::Qwen2
            | ModelArchitecture::Qwen3
            | ModelArchitecture::Mistral
            | ModelArchitecture::Gemma
            | ModelArchitecture::Gemma2
            | ModelArchitecture::Gemma3 => Self::ChatMl,
        }
    }

    /// Builds the system prompt (the prefilled, shared prefix).
    ///
    /// For ChatML the loaded context is followed by a fixed answer-format
    /// instruction. Instruct models need to be told to answer with a single
    /// option letter; without it the logits of `A`/`B`/`C` do not reflect the
    /// choice. The instruction is constant across requests, so the system
    /// prompt stays a stable prefix (the prefix-reuse invariant holds).
    pub fn system_prompt(self, system_text: &str) -> String {
        match self {
            Self::Manaca => format!("<|system|>\n{system_text}\n"),
            Self::ChatMl => format!(
                "<|im_start|>system\n{system_text}\n\n{}<|im_end|>\n",
                CHATML_ANSWER_INSTRUCTION
            ),
        }
    }

    /// Builds the user prompt (state + question).
    pub fn user_prompt(self, question: &str) -> String {
        match self {
            Self::Manaca => format!("<|user|>\n{question}\n<|model|>\n"),
            Self::ChatMl => {
                format!("<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")
            }
        }
    }

    /// Opening markers of the user turn followed by the state header.
    ///
    /// This is the part of the user turn that precedes the question text. The
    /// server tokenizes `system_prompt(context) + state_prefix(state)` once and
    /// reuses the resulting key/value cache across every question of a request
    /// (and, keyed by the state, across requests), so it is kept as a separate
    /// unit from [`Self::question_completion`].
    pub fn state_prefix(self, state_text: &str) -> String {
        match self {
            Self::Manaca => format!("<|user|>\nState:\n{state_text}\n\nQuestion:\n"),
            Self::ChatMl => format!("<|im_start|>user\nState:\n{state_text}\n\nQuestion:\n"),
        }
    }

    /// The question text followed by the closing markers of the user turn.
    ///
    /// `state_prefix(state) + question_completion(question)` reproduces the body
    /// previously built as `"State:\n{state}\n\nQuestion:\n{question}"` wrapped
    /// by [`Self::user_prompt`], so the concatenated prompt is the same string
    /// the monolithic renderer produced.
    pub fn question_completion(self, question_text: &str) -> String {
        match self {
            Self::Manaca => format!("{question_text}\n<|model|>\n"),
            Self::ChatMl => format!("{question_text}<|im_end|>\n<|im_start|>assistant\n"),
        }
    }

    /// Full classification prompt: system immediately followed by user.
    pub fn full_prompt(self, system_text: &str, question: &str) -> String {
        format!(
            "{}{}",
            self.system_prompt(system_text),
            self.user_prompt(question)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_selects_the_template() {
        assert_eq!(
            PromptTemplate::for_architecture(ModelArchitecture::Llama),
            PromptTemplate::Manaca
        );
        assert_eq!(
            PromptTemplate::for_architecture(ModelArchitecture::Qwen2),
            PromptTemplate::ChatMl
        );
    }

    #[test]
    fn manaca_template_matches_legacy_markers() {
        let template = PromptTemplate::Manaca;
        assert_eq!(template.system_prompt("rules"), "<|system|>\nrules\n");
        assert_eq!(template.user_prompt("Q?"), "<|user|>\nQ?\n<|model|>\n");
        assert_eq!(
            template.full_prompt("rules", "Q?"),
            "<|system|>\nrules\n<|user|>\nQ?\n<|model|>\n"
        );
    }

    #[test]
    fn chatml_template_ends_on_the_assistant_header() {
        let template = PromptTemplate::ChatMl;
        let system = template.system_prompt("rules");
        assert!(system.starts_with("<|im_start|>system\n"));
        assert!(system.contains("rules"));
        assert!(system.contains("Answer with a single option letter"));
        assert!(system.ends_with("<|im_end|>\n"));
        assert_eq!(
            template.user_prompt("Q?"),
            "<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n"
        );
        let full = template.full_prompt("rules", "Q?");
        assert!(full.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn full_prompt_starts_with_system_prompt_for_prefix_reuse() {
        for template in [PromptTemplate::Manaca, PromptTemplate::ChatMl] {
            let system = template.system_prompt("context");
            let full = template.full_prompt("context", "question");
            assert!(full.starts_with(&system));
        }
    }

    #[test]
    fn structured_state_and_question_reproduce_the_user_prompt() {
        // The structured split must concatenate back to the exact body the
        // monolithic user prompt carried: "State:\n{state}\n\nQuestion:\n{question}".
        for template in [PromptTemplate::Manaca, PromptTemplate::ChatMl] {
            let state = "charged twice";
            let question = "Refund?";
            let body = format!("State:\n{state}\n\nQuestion:\n{question}");
            let monolithic_user = template.user_prompt(&body);
            let structured_user = format!(
                "{}{}",
                template.state_prefix(state),
                template.question_completion(question)
            );
            assert_eq!(structured_user, monolithic_user);
        }
    }

    #[test]
    fn structured_full_prompt_reproduces_the_monolithic_prompt() {
        for template in [PromptTemplate::Manaca, PromptTemplate::ChatMl] {
            let context = "refund policy";
            let state = "billed twice";
            let question = "Refund?";
            let monolithic = template.full_prompt(
                context,
                &format!("State:\n{state}\n\nQuestion:\n{question}"),
            );
            let structured = format!(
                "{}{}{}",
                template.system_prompt(context),
                template.state_prefix(state),
                template.question_completion(question)
            );
            assert_eq!(structured, monolithic);
        }
    }
}
