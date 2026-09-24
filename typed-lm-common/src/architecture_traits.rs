//! Dense architecture capabilities consumed by the vendored forward pass.
//!
//! All supported families share the same decoder skeleton (RMSNorm, grouped
//! query attention, SwiGLU MLP, causal mask). They differ in a small, fixed set
//! of details that the forward pass must branch on:
//!
//! - attention projection biases,
//! - an explicit head dimension (Qwen3/Mistral/Gemma*),
//! - sliding-window attention (Qwen3/Mistral/Gemma2/Gemma3),
//! - logit soft-capping and the pre-attention scalar (Gemma2/Gemma3),
//! - the RMSNorm `+1` unit offset and the embedding scale (Gemma*),
//! - a separate local RoPE base frequency (Gemma3).
//!
//! Rather than copying the forward pass once per family, this module derives a
//! [`DenseArchitectureTraits`] value from the [`ModelArchitecture`] so a single
//! parametrized implementation serves every dense family. The per-family
//! defaults encode the *architectural* contract; the concrete `config.json`
//! still refines the values that are data-dependent (see
//! [`ParallelModelConfig`](crate::model_config::ParallelModelConfig)).

use crate::checkpoint::ModelArchitecture;

/// Capability flags for a dense decoder family.
///
/// These are the *defaults implied by the family*. Values that a checkpoint can
/// override (for example `attention_bias` on Qwen3/Gemma*) are authoritative in
/// the parsed [`ParallelModelConfig`](crate::model_config::ParallelModelConfig);
/// the flags here document which branches the forward pass must support.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseArchitectureTraits {
    /// The family itself, for diagnostics.
    pub architecture: ModelArchitecture,
    /// Attention projections may carry biases.
    pub attention_bias: bool,
    /// The family always carries an explicit `head_dim` (never derived).
    pub explicit_head_dimension: bool,
    /// The family may use sliding-window attention.
    pub supports_sliding_window: bool,
    /// The family applies `final_logit_softcapping`.
    pub supports_logit_softcapping: bool,
    /// The family applies `attn_logit_softcapping`.
    pub supports_attention_logit_softcapping: bool,
    /// The family uses `query_pre_attn_scalar` for the attention scale.
    pub supports_query_pre_attention_scalar: bool,
    /// RMSNorm uses `(1 + weight)` (Gemma*).
    pub rms_norm_unit_offset: bool,
    /// Embeddings are scaled by `sqrt(hidden_size)` (Gemma*).
    pub scales_embeddings: bool,
    /// The family carries a local RoPE base frequency (Gemma3).
    pub supports_local_rope_base_frequency: bool,
}

impl DenseArchitectureTraits {
    /// Derives the capability flags for an architecture.
    pub fn for_architecture(architecture: ModelArchitecture) -> Self {
        match architecture {
            ModelArchitecture::Llama => Self::llama(),
            ModelArchitecture::Qwen2 => Self::qwen2(),
            ModelArchitecture::Qwen3 => Self::qwen3(),
            ModelArchitecture::Mistral => Self::mistral(),
            ModelArchitecture::Gemma => Self::gemma(),
            ModelArchitecture::Gemma2 => Self::gemma2(),
            ModelArchitecture::Gemma3 => Self::gemma3(),
        }
    }

    fn llama() -> Self {
        Self {
            architecture: ModelArchitecture::Llama,
            attention_bias: false,
            explicit_head_dimension: false,
            supports_sliding_window: false,
            supports_logit_softcapping: false,
            supports_attention_logit_softcapping: false,
            supports_query_pre_attention_scalar: false,
            rms_norm_unit_offset: false,
            scales_embeddings: false,
            supports_local_rope_base_frequency: false,
        }
    }

    fn qwen2() -> Self {
        Self {
            architecture: ModelArchitecture::Qwen2,
            attention_bias: true,
            explicit_head_dimension: false,
            supports_sliding_window: false,
            supports_logit_softcapping: false,
            supports_attention_logit_softcapping: false,
            supports_query_pre_attention_scalar: false,
            rms_norm_unit_offset: false,
            scales_embeddings: false,
            supports_local_rope_base_frequency: false,
        }
    }

    fn qwen3() -> Self {
        Self {
            architecture: ModelArchitecture::Qwen3,
            attention_bias: true,
            explicit_head_dimension: true,
            supports_sliding_window: true,
            supports_logit_softcapping: false,
            supports_attention_logit_softcapping: false,
            supports_query_pre_attention_scalar: false,
            rms_norm_unit_offset: false,
            scales_embeddings: false,
            supports_local_rope_base_frequency: false,
        }
    }

    fn mistral() -> Self {
        Self {
            architecture: ModelArchitecture::Mistral,
            attention_bias: false,
            explicit_head_dimension: true,
            supports_sliding_window: true,
            supports_logit_softcapping: false,
            supports_attention_logit_softcapping: false,
            supports_query_pre_attention_scalar: false,
            rms_norm_unit_offset: false,
            scales_embeddings: false,
            supports_local_rope_base_frequency: false,
        }
    }

    fn gemma() -> Self {
        Self {
            architecture: ModelArchitecture::Gemma,
            attention_bias: true,
            explicit_head_dimension: true,
            supports_sliding_window: false,
            supports_logit_softcapping: false,
            supports_attention_logit_softcapping: false,
            supports_query_pre_attention_scalar: false,
            rms_norm_unit_offset: true,
            scales_embeddings: true,
            supports_local_rope_base_frequency: false,
        }
    }

    fn gemma2() -> Self {
        Self {
            architecture: ModelArchitecture::Gemma2,
            attention_bias: true,
            explicit_head_dimension: true,
            supports_sliding_window: true,
            supports_logit_softcapping: true,
            supports_attention_logit_softcapping: true,
            supports_query_pre_attention_scalar: true,
            rms_norm_unit_offset: true,
            scales_embeddings: true,
            supports_local_rope_base_frequency: false,
        }
    }

    fn gemma3() -> Self {
        Self {
            architecture: ModelArchitecture::Gemma3,
            attention_bias: true,
            explicit_head_dimension: true,
            supports_sliding_window: true,
            supports_logit_softcapping: true,
            supports_attention_logit_softcapping: true,
            supports_query_pre_attention_scalar: true,
            rms_norm_unit_offset: true,
            scales_embeddings: true,
            supports_local_rope_base_frequency: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_supported_architecture_has_traits() {
        for architecture in ModelArchitecture::SUPPORTED {
            let traits = DenseArchitectureTraits::for_architecture(architecture);
            assert_eq!(traits.architecture, architecture);
        }
    }

    #[test]
    fn attention_bias_defaults_match_llama_and_qwen2() {
        assert!(
            !DenseArchitectureTraits::for_architecture(ModelArchitecture::Llama).attention_bias
        );
        assert!(
            DenseArchitectureTraits::for_architecture(ModelArchitecture::Qwen2).attention_bias
        );
    }

    #[test]
    fn explicit_head_dimension_is_required_for_new_families() {
        for architecture in [
            ModelArchitecture::Qwen3,
            ModelArchitecture::Mistral,
            ModelArchitecture::Gemma,
            ModelArchitecture::Gemma2,
            ModelArchitecture::Gemma3,
        ] {
            assert!(
                DenseArchitectureTraits::for_architecture(architecture).explicit_head_dimension,
                "{architecture:?} must carry an explicit head dimension"
            );
        }
        for architecture in [ModelArchitecture::Llama, ModelArchitecture::Qwen2] {
            assert!(
                !DenseArchitectureTraits::for_architecture(architecture).explicit_head_dimension
            );
        }
    }

    #[test]
    fn sliding_window_is_only_for_the_expected_families() {
        for architecture in [
            ModelArchitecture::Qwen3,
            ModelArchitecture::Mistral,
            ModelArchitecture::Gemma2,
            ModelArchitecture::Gemma3,
        ] {
            assert!(
                DenseArchitectureTraits::for_architecture(architecture).supports_sliding_window
            );
        }
        for architecture in [
            ModelArchitecture::Llama,
            ModelArchitecture::Qwen2,
            ModelArchitecture::Gemma,
        ] {
            assert!(
                !DenseArchitectureTraits::for_architecture(architecture).supports_sliding_window
            );
        }
    }

    #[test]
    fn softcapping_is_gemma2_and_gemma3_only() {
        for architecture in [ModelArchitecture::Gemma2, ModelArchitecture::Gemma3] {
            let traits = DenseArchitectureTraits::for_architecture(architecture);
            assert!(traits.supports_logit_softcapping);
            assert!(traits.supports_attention_logit_softcapping);
            assert!(traits.supports_query_pre_attention_scalar);
        }
        for architecture in [
            ModelArchitecture::Llama,
            ModelArchitecture::Qwen2,
            ModelArchitecture::Qwen3,
            ModelArchitecture::Mistral,
            ModelArchitecture::Gemma,
        ] {
            let traits = DenseArchitectureTraits::for_architecture(architecture);
            assert!(!traits.supports_logit_softcapping);
            assert!(!traits.supports_attention_logit_softcapping);
            assert!(!traits.supports_query_pre_attention_scalar);
        }
    }

    #[test]
    fn gemma_families_share_norm_offset_and_embedding_scale() {
        for architecture in [
            ModelArchitecture::Gemma,
            ModelArchitecture::Gemma2,
            ModelArchitecture::Gemma3,
        ] {
            let traits = DenseArchitectureTraits::for_architecture(architecture);
            assert!(traits.rms_norm_unit_offset);
            assert!(traits.scales_embeddings);
        }
        for architecture in [
            ModelArchitecture::Llama,
            ModelArchitecture::Qwen2,
            ModelArchitecture::Qwen3,
            ModelArchitecture::Mistral,
        ] {
            let traits = DenseArchitectureTraits::for_architecture(architecture);
            assert!(!traits.rms_norm_unit_offset);
            assert!(!traits.scales_embeddings);
        }
    }

    #[test]
    fn local_rope_base_frequency_is_gemma3_only() {
        assert!(
            DenseArchitectureTraits::for_architecture(ModelArchitecture::Gemma3)
                .supports_local_rope_base_frequency
        );
        for architecture in [
            ModelArchitecture::Llama,
            ModelArchitecture::Qwen2,
            ModelArchitecture::Qwen3,
            ModelArchitecture::Mistral,
            ModelArchitecture::Gemma,
            ModelArchitecture::Gemma2,
        ] {
            assert!(
                !DenseArchitectureTraits::for_architecture(architecture)
                    .supports_local_rope_base_frequency
            );
        }
    }
}
