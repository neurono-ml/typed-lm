use clap::Parser;

use typed_lm_trainer::cli::{Command, TrainerArguments};
use typed_lm_trainer::error::TrainerError;

/// Trains and quantizes models for the `typed-lm` serving stack.
///
/// Dispatch is a thin shim in this skeleton; Waves 3-6 connect the real
/// dataset, model, training and quantization pipelines.
#[tokio::main]
async fn main() -> Result<(), TrainerError> {
    let arguments = TrainerArguments::parse();
    match arguments.command {
        Command::Train(_train_arguments) => Err(TrainerError::Other(anyhow::anyhow!(
            "train is not implemented yet (arrives in a later wave)"
        ))),
        Command::Quantize(_quantize_arguments) => Err(TrainerError::Other(anyhow::anyhow!(
            "quantize is not implemented yet (arrives in a later wave)"
        ))),
    }
}
