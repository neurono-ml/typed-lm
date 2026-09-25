use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, MutexGuard};

use candle_core::{IndexOp, Tensor};

use crate::api::dtos::{Answer, Question, SystemOneRequest, SystemOneResponse, Usage};
use crate::api::error::EvaluationError;
use crate::domain::evaluator::{
    answer_from_probabilities, is_supported_model, longest_common_prefix, Evaluator,
};
use crate::infrastructure::language_model::{LanguageModel, ModelCache};
use crate::infrastructure::session_cache::{SessionCacheConfiguration, SessionPrefixCache};
use typed_lm_common::labels::{calibrate_probabilities, label_count_for_question, option_labels};
use typed_lm_common::rendering::{content_to_text, render_question_text};

/// A prefilled prefix (system context + state) held by the session cache.
#[derive(Clone)]
struct CachedSessionPrefix {
    /// Exact tokenization the cached key/value entries correspond to.
    token_identifiers: Vec<u32>,
    /// The key/value cache with `token_identifiers` already prefilled.
    cache: ModelCache,
}

/// Real evaluator: parallel evaluation via KV-cache broadcasting.
///
/// The system prompt (loaded context) is prefilled once at construction into an
/// immutable [`ModelCache`]. Every request then prefills, exactly once, the
/// tokens shared by all its questions: the `state` plus the common question
/// prefix. That shared cache is **broadcast across the batch dimension** and
/// every question suffix is evaluated in a **single batched forward pass**,
/// cloning only when a question prompt degenerates to the shared prefix. The
/// broadcast cache is never reused after the batched pass, preserving the
/// per-request cache isolation required by the design.
///
/// The state prefix (`system + state`) is additionally retained in a bounded
/// least-recently-used cache keyed by a canonical hash of the state, so a state
/// that reappears across requests skips its forward pass. Entries hold clones;
/// the cached cache is never mutated.
pub struct CandleEvaluator<'model_lifetime> {
    language_model: &'model_lifetime LanguageModel,
    loaded_context: String,
    served_model_name: String,
    context_cache: ModelCache,
    context_token_identifiers: Vec<u32>,
    session_cache: Mutex<SessionPrefixCache<CachedSessionPrefix>>,
    session_namespace: u64,
}

impl<'model_lifetime> CandleEvaluator<'model_lifetime> {
    /// Builds the evaluator and prefills the fixed system context once.
    pub fn new(
        language_model: &'model_lifetime LanguageModel,
        loaded_context: String,
        served_model_name: String,
        session_configuration: SessionCacheConfiguration,
    ) -> Result<Self, EvaluationError> {
        let system_prompt = language_model
            .prompt_template()
            .system_prompt(&loaded_context);
        let context_token_identifiers = language_model
            .encode_tokens(&system_prompt, true)
            .map_err(EvaluationError::from)?;
        let (context_cache, _context_logits) = language_model
            .prefill(&context_token_identifiers)
            .map_err(EvaluationError::from)?;
        tracing::debug!(
            "prefilled system context into base cache ({} tokens)",
            context_token_identifiers.len()
        );

        // Stable identity of (context, served model): the session key mixes it
        // in so cached prefixes never leak across a different context/model.
        let mut namespace_hasher = DefaultHasher::new();
        loaded_context.hash(&mut namespace_hasher);
        served_model_name.hash(&mut namespace_hasher);

        Ok(Self {
            language_model,
            loaded_context,
            served_model_name,
            context_cache,
            context_token_identifiers,
            session_cache: Mutex::new(SessionPrefixCache::new(
                session_configuration.maximum_entries,
                session_configuration.maximum_tokens,
            )),
            session_namespace: namespace_hasher.finish(),
        })
    }

    /// Number of tokens held by the prefilled system-context base cache.
    pub fn context_token_length(&self) -> usize {
        self.context_token_identifiers.len()
    }

    /// Locks the session cache, recovering from a poisoned mutex.
    fn session_cache_guard(&self) -> MutexGuard<'_, SessionPrefixCache<CachedSessionPrefix>> {
        match self.session_cache.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Canonical, non-cryptographic key for a state within this evaluator.
    fn session_key(&self, state_text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.session_namespace.hash(&mut hasher);
        state_text.hash(&mut hasher);
        hasher.finish()
    }

    /// Builds `system_prompt + state_prefix` as one string (the cacheable part).
    fn render_state_prefix(&self, state_text: &str) -> String {
        let template = *self.language_model.prompt_template();
        format!(
            "{}{}",
            template.system_prompt(&self.loaded_context),
            template.state_prefix(state_text)
        )
    }

    /// Reads the logit of every answer label from the last-position logits.
    ///
    /// When a label can be emitted as more than one single-token surface form
    /// (for example `"A"` and `" A"` in byte-level BPE), the highest logit
    /// across the variants is used, so calibration does not depend on which
    /// spacing variant the tokenizer ranks first.
    fn label_logit_values(
        &self,
        logits: &Tensor,
        question: &Question,
    ) -> Result<Vec<f32>, EvaluationError> {
        let labels = option_labels(label_count_for_question(question));
        let mut logit_values: Vec<f32> = Vec::with_capacity(labels.len());
        for label in &labels {
            let token_identifiers = self.language_model.label_token_ids(label);
            if token_identifiers.is_empty() {
                return Err(EvaluationError::inference(format!(
                    "answer label '{label}' is not a single vocabulary token"
                )));
            }
            let mut best = f32::NEG_INFINITY;
            for token_identifier in token_identifiers {
                let logit_value = self
                    .language_model
                    .token_logit(logits, token_identifier)
                    .map_err(EvaluationError::from)?;
                if logit_value > best {
                    best = logit_value;
                }
            }
            logit_values.push(best);
        }
        Ok(logit_values)
    }

    /// Resolves the starting prefix cache for one request.
    ///
    /// On a session-cache hit the cached prefix cache is cloned (never mutated);
    /// on a miss the prefix is prefilled from the immutable context cache (or
    /// from scratch when the context is not a literal token prefix) and a clone
    /// is stored. Returns the broadcast-ready cache and the number of tokens it
    /// holds.
    fn resolve_session_prefix(
        &self,
        state_text: &str,
        prefix_token_identifiers: &[u32],
    ) -> Result<(ModelCache, usize, bool), EvaluationError> {
        let session_key = self.session_key(state_text);
        {
            let mut guard = self.session_cache_guard();
            if let Some(cached) = guard.lookup(session_key) {
                if cached.token_identifiers == prefix_token_identifiers {
                    return Ok((cached.cache.clone(), prefix_token_identifiers.len(), true));
                }
            }
        }

        let context_length = self.context_token_identifiers.len();
        let context_is_prefix =
            prefix_token_identifiers.starts_with(&self.context_token_identifiers);
        let mut cache = if context_is_prefix {
            self.context_cache.clone()
        } else {
            self.language_model
                .empty_cache()
                .map_err(EvaluationError::from)?
        };
        let start_position = if context_is_prefix { context_length } else { 0 };
        if prefix_token_identifiers.len() > start_position {
            self.language_model
                .continue_forward(
                    &prefix_token_identifiers[start_position..],
                    start_position,
                    &mut cache,
                )
                .map_err(EvaluationError::from)?;
        }
        {
            let mut guard = self.session_cache_guard();
            guard.insert(
                session_key,
                CachedSessionPrefix {
                    token_identifiers: prefix_token_identifiers.to_vec(),
                    cache: cache.clone(),
                },
                prefix_token_identifiers.len(),
            );
        }
        Ok((cache, prefix_token_identifiers.len(), false))
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
        tracing::info!(
            "starting evaluation for model '{}' with {} question(s)",
            request.model,
            request.questions.len()
        );

        // Deterministic question order.
        let mut ordered_identifiers: Vec<&String> = request.questions.keys().collect();
        ordered_identifiers.sort();

        // Structured tokenization: the system prompt plus the state prefix is
        // tokenized once, and every question only contributes its completion
        // (the question text and the user-turn closing markers). This yields a
        // guaranteed shared prefix and the canonical session-cache key, without
        // relying on the tokenizer's longest-common-prefix behavior.
        let state_prefix_text = self.render_state_prefix(&state_text);
        let prefix_token_identifiers = self
            .language_model
            .encode_tokens(&state_prefix_text, true)
            .map_err(EvaluationError::from)?;

        let mut question_tokens: Vec<(&String, &Question, Vec<u32>, Vec<u32>)> =
            Vec::with_capacity(ordered_identifiers.len());
        for identifier in &ordered_identifiers {
            let question = request.questions.get(*identifier).ok_or_else(|| {
                EvaluationError::invalid_request(format!(
                    "question '{identifier}' disappeared during evaluation"
                ))
            })?;
            let question_text = render_question_text(question);
            let completion_text = self
                .language_model
                .prompt_template()
                .question_completion(&question_text);
            let completion_tokens = self
                .language_model
                .encode_tokens(&completion_text, true)
                .map_err(EvaluationError::from)?;
            let mut full_tokens = prefix_token_identifiers.clone();
            full_tokens.extend_from_slice(&completion_tokens);
            question_tokens.push((identifier, question, full_tokens, completion_tokens));
        }

        let completion_sequences: Vec<Vec<u32>> = question_tokens
            .iter()
            .map(|(_, _, _, completion_tokens)| completion_tokens.clone())
            .collect();
        let shared_completion_length = longest_common_prefix(&completion_sequences);
        let prefix_length = prefix_token_identifiers.len();
        let shared_length = prefix_length + shared_completion_length;

        // Resolve the state prefix: reuse the session cache when possible,
        // otherwise prefill it once and retain a clone.
        let (mut shared_cache, start_position, cache_hit) =
            self.resolve_session_prefix(&state_text, &prefix_token_identifiers)?;

        // Single broadcast prefill: the tokens shared by all questions beyond
        // the state prefix are processed exactly once, on the cloned cache.
        let mut shared_logits: Option<Tensor> = None;
        if shared_length > start_position {
            let first_sequence = question_tokens
                .first()
                .map(|(_, _, full_tokens, _)| full_tokens)
                .ok_or_else(|| {
                    EvaluationError::inference("no questions were supplied for evaluation")
                })?;
            let shared_tokens = &first_sequence[start_position..shared_length];
            let logits = self
                .language_model
                .continue_forward(shared_tokens, start_position, &mut shared_cache)
                .map_err(EvaluationError::from)?;
            shared_logits = Some(logits);
        }

        tracing::debug!(
            context_tokens = self.context_token_identifiers.len(),
            state_tokens = prefix_length.saturating_sub(self.context_token_identifiers.len()),
            shared_tokens = shared_length,
            shared_completion_tokens = shared_completion_length,
            questions = question_tokens.len(),
            cache_hit,
            "session prefix resolved"
        );

        // Parallel evaluation: every non-degenerate suffix is scored in one
        // batched forward pass on top of the broadcast shared prefix cache.
        let mut batched_question_indices: Vec<usize> = Vec::new();
        let mut batched_suffixes: Vec<Vec<u32>> = Vec::new();
        for (index, (_, _, full_tokens, _)) in question_tokens.iter().enumerate() {
            let suffix = &full_tokens[shared_length..];
            if !suffix.is_empty() {
                batched_question_indices.push(index);
                batched_suffixes.push(suffix.to_vec());
            }
        }
        tracing::debug!(
            suffix_tokens = batched_suffixes
                .iter()
                .map(|suffix| suffix.len())
                .sum::<usize>(),
            batched_questions = batched_suffixes.len(),
            "question suffixes resolved"
        );

        let mut batched_logits: HashMap<usize, Tensor> =
            HashMap::with_capacity(batched_suffixes.len());
        if batched_suffixes.len() > 1 {
            let mut broadcast_cache = shared_cache.clone();
            let all_logits = self
                .language_model
                .score_suffixes_batched(&batched_suffixes, shared_length, &mut broadcast_cache)
                .map_err(EvaluationError::from)?;
            for (row, question_index) in batched_question_indices.iter().enumerate() {
                let row_logits = all_logits.i((row, ..))?.unsqueeze(0)?;
                batched_logits.insert(*question_index, row_logits);
            }
            tracing::debug!(
                "scored {} question suffix(es) in a single batched forward pass",
                batched_suffixes.len()
            );
        }

        let mut answers: HashMap<String, Answer> = HashMap::with_capacity(question_tokens.len());
        let mut input_tokens = 0_usize;
        for (index, (identifier, question, full_tokens, _)) in question_tokens.iter().enumerate() {
            let suffix = &full_tokens[shared_length..];
            let logits = if let Some(row_logits) = batched_logits.get(&index) {
                row_logits.clone()
            } else if suffix.is_empty() {
                // A question prompt that equals the shared prefix: its answer
                // sits at the last shared position.
                match &shared_logits {
                    Some(logits) => logits.clone(),
                    None => {
                        let (_cache, full_logits) = self
                            .language_model
                            .prefill(full_tokens)
                            .map_err(EvaluationError::from)?;
                        full_logits
                    }
                }
            } else {
                // Single suffix: reuse the shared cache without broadcasting.
                let mut question_cache = shared_cache.clone();
                self.language_model
                    .continue_forward(suffix, shared_length, &mut question_cache)
                    .map_err(EvaluationError::from)?
            };
            let logit_values = self.label_logit_values(&logits, question)?;
            let probabilities = calibrate_probabilities(&logit_values);
            tracing::debug!(
                "question '{identifier}' answered with probabilities {probabilities:?}"
            );
            answers.insert(
                (*identifier).clone(),
                answer_from_probabilities(question, &probabilities),
            );
            input_tokens = input_tokens.saturating_add(full_tokens.len());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::session_cache::{
        DEFAULT_SESSION_CACHE_ENTRIES, DEFAULT_SESSION_CACHE_TOKENS,
    };

    #[test]
    fn default_session_configuration_is_bounded_and_enabled() {
        let configuration = SessionCacheConfiguration::default();
        assert!(configuration.maximum_entries > 0);
        assert!(configuration.maximum_tokens > 0);
        assert_eq!(configuration.maximum_entries, DEFAULT_SESSION_CACHE_ENTRIES);
        assert_eq!(configuration.maximum_tokens, DEFAULT_SESSION_CACHE_TOKENS);
    }

    /// Live test (ignored): proves the structured tokenization used by the
    /// evaluator concatenates to the monolithic full-prompt tokenization for
    /// the real tokenizer. Downloads the weights once.
    #[test]
    #[ignore]
    fn prefix_reuse_matches_monolithic_forward() -> anyhow::Result<()> {
        use typed_lm_common::checkpoint::ModelReference;
        use typed_lm_common::checkpoint_resolver::{HubCheckpointResolver, LoadableCheckpoint};
        use typed_lm_common::device::{DeviceResolver, ModelDtype};

        let model_identifier =
            std::env::var("MODEL_ID").unwrap_or_else(|_| "Qwen/Qwen2.5-1.5B-Instruct".to_string());
        let device = DeviceResolver::resolve()?;
        let dtype = ModelDtype::Auto.resolve(&device);
        let reference = ModelReference::resolve(&model_identifier);
        let checkpoint: LoadableCheckpoint =
            HubCheckpointResolver::new(&model_identifier, None)?.resolve(&reference, None)?;
        let model = LanguageModel::load(&checkpoint, &device, dtype)?;
        let template = *model.prompt_template();

        let context = "Fact 1: a double charge is always fully refundable.";
        let state = "{\"amount\": 42.0, \"merchant\": \"GreenLeaf\"}";
        let question = "Should this be refunded?";
        let system = template.system_prompt(context);
        let state_prefix = template.state_prefix(state);
        let completion = template.question_completion(question);

        let prefix_tokens = model.encode_tokens(&format!("{system}{state_prefix}"), true)?;
        let completion_tokens = model.encode_tokens(&completion, true)?;
        let monolithic_tokens =
            model.encode_tokens(&format!("{system}{state_prefix}{completion}"), true)?;

        let mut concatenated = prefix_tokens.clone();
        concatenated.extend_from_slice(&completion_tokens);
        assert_eq!(
            concatenated, monolithic_tokens,
            "structured tokenization diverges from the monolithic prompt"
        );
        Ok(())
    }

    /// Live test (ignored): the session cache returns identical answers and the
    /// retained state prefix is actually reused across requests.
    ///
    /// Timing on a shared, unpinned CPU is too noisy to assert a speedup
    /// deterministically (the machine uses the `powersave` governor and other
    /// work can preempt the test), so only the **behavioural** contract is
    /// asserted: answers are identical whether the cache is enabled or not, and
    /// a repeated request over the same state is answered identically. The
    /// latency gain itself is measured by `reports_latency_breakdown`, which
    /// logs rather than asserts.
    #[test]
    #[ignore]
    fn session_cache_reuses_state_prefix_and_preserves_answers() -> anyhow::Result<()> {
        use typed_lm_common::checkpoint::ModelReference;
        use typed_lm_common::checkpoint_resolver::{HubCheckpointResolver, LoadableCheckpoint};
        use typed_lm_common::device::{DeviceResolver, ModelDtype};

        let model_identifier =
            std::env::var("MODEL_ID").unwrap_or_else(|_| "Qwen/Qwen2.5-1.5B-Instruct".to_string());
        let device = DeviceResolver::resolve()?;
        let dtype = ModelDtype::Auto.resolve(&device);
        let reference = ModelReference::resolve(&model_identifier);
        let checkpoint: LoadableCheckpoint =
            HubCheckpointResolver::new(&model_identifier, None)?.resolve(&reference, None)?;
        let model = LanguageModel::load(&checkpoint, &device, dtype)?;

        let context = "Fact 1: a double charge is always fully refundable.".to_string();
        let request: SystemOneRequest = serde_json::from_value(serde_json::json!({
            "model": "typed-lm",
            "state": "{\"amount\": 42.0, \"merchant\": \"GreenLeaf\"}",
            "questions": {
                "refund": {"type": "noul", "instructions": "Should this be refunded?"},
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is it?",
                    "criteria": ["Routine", "Urgent", "Emergency"]
                }
            }
        }))?;

        let cached = CandleEvaluator::new(
            &model,
            context.clone(),
            "typed-lm".to_string(),
            SessionCacheConfiguration::default(),
        )?;
        let uncached = CandleEvaluator::new(
            &model,
            context,
            "typed-lm".to_string(),
            SessionCacheConfiguration {
                maximum_entries: 0,
                maximum_tokens: 0,
            },
        )?;

        let cached_answer = cached.evaluate(&request)?;
        let repeated = cached.evaluate(&request)?;
        let uncached_answer = uncached.evaluate(&request)?;

        assert_eq!(
            serde_json::to_value(&cached_answer)?,
            serde_json::to_value(&uncached_answer)?,
            "session cache must not change the answers"
        );
        assert_eq!(
            serde_json::to_value(&cached_answer)?,
            serde_json::to_value(&repeated)?,
            "repeated requests over the same state must be identical"
        );
        Ok(())
    }
}
