//! Training loop, loss, optimizer and checkpointing.
//!
//! The training tree breaks down as:
//!
//! - [`loss`] — decision-position cross-entropy and the optional KL calibration.
//! - [`optimizer`] — AdamW, warmup/cosine schedule, gradient clipping and
//!   accumulation.
//! - [`checkpoint`] — saving and loading the LoRA adapter.
//! - [`r#loop`] — the generic epoch loop over a [`r#loop::TrainableModel`].

pub mod checkpoint;
pub mod r#loop;
pub mod loss;
pub mod optimizer;
