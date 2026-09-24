//! Shared foundations for the `typed-lm` workspace.
//!
//! This crate holds everything the serving (`typed-lm-serve`) and training
//! (`typed-lm-trainer`) binaries must agree on: the Jev request contract, answer
//! label arithmetic, deterministic prompt rendering, checkpoint detection,
//! device/dtype resolution, model configuration and telemetry.

pub mod checkpoint;
pub mod checkpoint_resolver;
pub mod classifier;
pub mod context;
pub mod contract;
pub mod device;
pub mod labels;
pub mod model_config;
pub mod model_repository;
pub mod prompt_template;
pub mod quantization;
pub mod rendering;
pub mod telemetry;
pub mod tokenizer;
