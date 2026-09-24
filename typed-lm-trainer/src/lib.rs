//! `typed-lm-trainer`: LoRA/QLoRA fine-tuning and post-training quantization
//! for the `typed-lm` monorepo.
//!
//! The crate is intentionally split into small modules (dataset, model,
//! training, quantization) so each concern stays testable in isolation. This
//! skeleton declares the module tree; the implementations land in Waves 3-6.

pub mod cli;
pub mod dataset;
pub mod error;
pub mod model;
pub mod quantization;
pub mod training;
