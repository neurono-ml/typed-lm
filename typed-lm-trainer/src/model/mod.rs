//! Trainable model wrappers and LoRA/QLoRA adapters.
//!
//! The model tree separates four concerns:
//!
//! - [`precision`] — dtype casts that follow a [`PrecisionPolicy`](typed_lm_common::device::PrecisionPolicy).
//! - [`lora`] — the trainable low-rank adapter wrapped around a frozen projection.
//! - [`weight_loading`] — turning a resolved checkpoint into a frozen dense base.
//! - [`trainable_llama`]/[`trainable_qwen2`] — the differentiable full-sequence
//!   forward over a `VarMap`.

pub mod lora;
pub mod precision;
pub mod trainable_llama;
pub mod trainable_qwen2;
pub mod weight_loading;
