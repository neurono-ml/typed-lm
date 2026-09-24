//! Command-line interface for the trainer.
//!
//! Wave 6 fills the full `train`/`quantize` argument sets; this skeleton
//! declares only the subcommand enum so `--help` works.

use clap::{Parser, Subcommand};

/// typed-lm-trainer: fine-tune and quantize the served language models.
#[derive(Parser, Debug)]
#[command(
    name = "typed-lm-trainer",
    about = "LoRA/QLoRA fine-tuning and post-training quantization for typed-lm"
)]
pub struct TrainerArguments {
    #[command(subcommand)]
    pub command: Command,
}

/// Trainer subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Fine-tune a base model with LoRA or QLoRA.
    Train(TrainArguments),
    /// Quantize a fine-tuned or base model to FP8/FP4.
    Quantize(QuantizeArguments),
}

/// Arguments for the `train` subcommand (extended in Wave 6).
#[derive(clap::Args, Debug, Clone, Default)]
pub struct TrainArguments {
    /// Training method: `lora` or `qlora`.
    #[arg(long, default_value = "lora")]
    pub method: String,
}

/// Arguments for the `quantize` subcommand (extended in Wave 6).
#[derive(clap::Args, Debug, Clone, Default)]
pub struct QuantizeArguments {
    /// Quantization target: `none`, `fp8` or `fp4`.
    #[arg(long, default_value = "fp8")]
    pub quantization: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_defaults_to_lora_method() -> anyhow::Result<()> {
        let arguments = TrainerArguments::try_parse_from(["typed-lm-trainer", "train"])?;
        let Command::Train(train) = arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert_eq!(train.method, "lora");
        Ok(())
    }

    #[test]
    fn quantize_defaults_to_fp8() -> anyhow::Result<()> {
        let arguments = TrainerArguments::try_parse_from(["typed-lm-trainer", "quantize"])?;
        let Command::Quantize(quantize) = arguments.command else {
            return Err(anyhow::anyhow!("expected the quantize subcommand"));
        };
        assert_eq!(quantize.quantization, "fp8");
        Ok(())
    }
}
