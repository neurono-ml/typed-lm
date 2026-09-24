//! Trainable model wrappers and LoRA/QLoRA adapters.
//!
//! The model tree separates four concerns:
//!
//! - [`precision`] — dtype casts that follow a [`PrecisionPolicy`](typed_lm_common::device::PrecisionPolicy).
//! - [`lora`] — the trainable low-rank adapter wrapped around a frozen projection.
//! - [`weight_loading`] — turning a resolved checkpoint into a frozen dense base.
//! - [`trainable_llama`] — the differentiable full-sequence forward over a `VarMap`.
//! - [`trainable_dense`] — the unified dense loader that validates the
//!   configuration against the detected architecture for every supported family.
//! - [`initialization`] — the deterministic, seeded weight initializer used by
//!   from-scratch training.
//! - [`trainable_linear`]/[`trainable_rms_norm`] — fully-trainable primitives
//!   whose weights are `Var`s (full-parameter training).
pub mod initialization;
pub mod lora;
pub mod precision;
pub mod trainable_dense;
pub mod trainable_full;
pub mod trainable_linear;
pub mod trainable_llama;
pub mod trainable_rms_norm;
pub mod weight_loading;
