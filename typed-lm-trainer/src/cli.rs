//! Command-line interface for the trainer.
//!
//! Two subcommands share the same vocabulary as the rest of the workspace:
//! `train` fine-tunes a base checkpoint with LoRA/QLoRA, and `quantize` applies
//! post-training quantization (FP8/FP4) to a merged or base checkpoint. Every
//! flag carries a sane default so `typed-lm-trainer train --dataset data.jsonl`
//! is a complete invocation.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use typed_lm_common::quantization::QuantizationScheme;

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

/// Training method selected by `--method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingMethod {
    /// LoRA adapters over a dense base checkpoint.
    Lora,
    /// QLoRA adapters over a quantized (dequantized-on-load) base checkpoint.
    QLoRa,
}

impl TrainingMethod {
    /// Parses the `--method` flag value.
    pub fn from_flag(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "lora" => Some(Self::Lora),
            "qlora" => Some(Self::QLoRa),
            _ => None,
        }
    }

    /// Canonical flag name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Lora => "lora",
            Self::QLoRa => "qlora",
        }
    }
}

/// When quantization happens relative to training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    /// Quantize the merged adapter after training (post-training quantization).
    PostTraining,
    /// Keep the base quantized throughout training (QLoRA).
    Training,
}

impl QuantizationMode {
    /// Parses the `--quantization-mode` flag value.
    pub fn from_flag(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "post-training" => Some(Self::PostTraining),
            "training" => Some(Self::Training),
            _ => None,
        }
    }

    /// Canonical flag name.
    pub fn name(self) -> &'static str {
        match self {
            Self::PostTraining => "post-training",
            Self::Training => "training",
        }
    }
}

/// Arguments for the `train` subcommand.
#[derive(clap::Args, Debug, Clone)]
pub struct TrainArguments {
    /// Model reference: a Hugging Face repository id or a local path.
    #[arg(long, default_value = "Qwen/Qwen2.5-1.5B-Instruct")]
    pub model_id: String,

    /// Dataset path: a `.jsonl`/`.json` file or a directory (auto-detected).
    #[arg(long)]
    pub dataset: PathBuf,

    /// Directory that receives the trained adapter and metadata.
    #[arg(long, default_value = "output/train")]
    pub output_directory: PathBuf,

    /// Training method: `lora` or `qlora`.
    #[arg(long, default_value = "lora")]
    pub method: String,

    /// LoRA rank (`r`).
    #[arg(long, default_value_t = 16)]
    pub lora_rank: usize,

    /// LoRA scaling numerator (`alpha`); effective scale is `alpha / rank`.
    #[arg(long, default_value_t = 32.0)]
    pub lora_alpha: f64,

    /// LoRA dropout applied to the adapter activations.
    #[arg(long, default_value_t = 0.0)]
    pub lora_dropout: f32,

    /// Number of epochs.
    #[arg(long, default_value_t = 3)]
    pub epochs: usize,

    /// Batch size per step (items that share a state may be bucketed together).
    #[arg(long, default_value_t = 4)]
    pub batch_size: usize,

    /// Micro-batches accumulated before an optimizer step.
    #[arg(long, default_value_t = 1)]
    pub gradient_accumulation_steps: usize,

    /// Peak learning rate.
    #[arg(long, default_value_t = 1e-4)]
    pub learning_rate: f64,

    /// Linear warmup steps before the cosine decay.
    #[arg(long, default_value_t = 10)]
    pub warmup_steps: usize,

    /// AdamW weight decay.
    #[arg(long, default_value_t = 0.0)]
    pub weight_decay: f64,

    /// Global gradient-norm clipping threshold.
    #[arg(long, default_value_t = 1.0)]
    pub maximum_gradient_norm: f64,

    /// Maximum prompt length; longer items are skipped.
    #[arg(long, default_value_t = 1024)]
    pub max_sequence_length: usize,

    /// Minimum loss improvement that resets the early-stop patience.
    #[arg(long, default_value_t = 0.0)]
    pub minimum_improvement: f32,

    /// Epochs without improvement before stopping early (`0` disables it).
    #[arg(long, default_value_t = 0)]
    pub early_stop_patience: usize,

    /// Quantization applied when `--quantization-mode training`.
    #[arg(long, default_value = "none")]
    pub quantization: String,

    /// When quantization happens: `post-training` or `training`.
    #[arg(long, default_value = "post-training")]
    pub quantization_mode: String,
}

/// Arguments for the `quantize` subcommand.
#[derive(clap::Args, Debug, Clone)]
pub struct QuantizeArguments {
    /// Model reference whose merged weights are quantized.
    #[arg(long, default_value = "Qwen/Qwen2.5-1.5B-Instruct")]
    pub model_id: String,

    /// Directory holding the trained adapter to merge (optional).
    #[arg(long)]
    pub adapter_directory: Option<PathBuf>,

    /// Directory that receives the quantized artifact.
    #[arg(long, default_value = "output/quantized")]
    pub output_directory: PathBuf,

    /// Quantization target: `none`, `fp8` or `fp4`.
    #[arg(long, default_value = "fp8")]
    pub quantization: String,
}

impl TrainArguments {
    /// Resolves the parsed `--method` value into a [`TrainingMethod`].
    pub fn training_method(&self) -> anyhow::Result<TrainingMethod> {
        TrainingMethod::from_flag(&self.method).ok_or_else(|| {
            anyhow::anyhow!(
                "invalid --method '{}': expected 'lora' or 'qlora'",
                self.method
            )
        })
    }

    /// Resolves `--quantization-mode` into a [`QuantizationMode`].
    pub fn quantization_mode(&self) -> anyhow::Result<QuantizationMode> {
        QuantizationMode::from_flag(&self.quantization_mode).ok_or_else(|| {
            anyhow::anyhow!(
                "invalid --quantization-mode '{}': expected 'post-training' or 'training'",
                self.quantization_mode
            )
        })
    }

    /// Resolves `--quantization` into a [`QuantizationScheme`].
    pub fn quantization_scheme(&self) -> anyhow::Result<QuantizationScheme> {
        QuantizationScheme::from_flag(&self.quantization).ok_or_else(|| {
            anyhow::anyhow!(
                "invalid --quantization '{}': expected 'none', 'fp8' or 'fp4'",
                self.quantization
            )
        })
    }
}

impl QuantizeArguments {
    /// Resolves `--quantization` into a [`QuantizationScheme`].
    pub fn quantization_scheme(&self) -> anyhow::Result<QuantizationScheme> {
        QuantizationScheme::from_flag(&self.quantization).ok_or_else(|| {
            anyhow::anyhow!(
                "invalid --quantization '{}': expected 'none', 'fp8' or 'fp4'",
                self.quantization
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_defaults_to_lora_method_and_full_arguments() -> anyhow::Result<()> {
        let arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data.jsonl",
        ])?;
        let Command::Train(train) = arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert_eq!(train.method, "lora");
        assert_eq!(train.training_method()?, TrainingMethod::Lora);
        assert_eq!(train.lora_rank, 16);
        assert_eq!(train.epochs, 3);
        assert_eq!(train.quantization_mode()?, QuantizationMode::PostTraining);
        Ok(())
    }

    #[test]
    fn train_accepts_qlora_and_training_mode() -> anyhow::Result<()> {
        let arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--method",
            "qlora",
            "--quantization",
            "fp4",
            "--quantization-mode",
            "training",
        ])?;
        let Command::Train(train) = arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert_eq!(train.training_method()?, TrainingMethod::QLoRa);
        assert_eq!(train.quantization_mode()?, QuantizationMode::Training);
        assert_eq!(train.quantization_scheme()?, QuantizationScheme::Fp4);
        Ok(())
    }

    #[test]
    fn train_requires_a_dataset() {
        assert!(TrainerArguments::try_parse_from(["typed-lm-trainer", "train"]).is_err());
    }

    #[test]
    fn an_invalid_method_is_rejected() -> anyhow::Result<()> {
        let arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--method",
            "full",
        ])?;
        let Command::Train(train) = arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert!(train.training_method().is_err());
        Ok(())
    }

    #[test]
    fn quantize_defaults_to_fp8() -> anyhow::Result<()> {
        let arguments = TrainerArguments::try_parse_from(["typed-lm-trainer", "quantize"])?;
        let Command::Quantize(quantize) = arguments.command else {
            return Err(anyhow::anyhow!("expected the quantize subcommand"));
        };
        assert_eq!(quantize.quantization, "fp8");
        assert_eq!(quantize.quantization_scheme()?, QuantizationScheme::Fp8);
        assert!(quantize.adapter_directory.is_none());
        Ok(())
    }

    #[test]
    fn method_and_mode_flag_names_round_trip() {
        assert_eq!(
            TrainingMethod::from_flag("LORA"),
            Some(TrainingMethod::Lora)
        );
        assert_eq!(
            TrainingMethod::from_flag("QLoRA"),
            Some(TrainingMethod::QLoRa)
        );
        assert_eq!(TrainingMethod::Lora.name(), "lora");
        assert_eq!(
            QuantizationMode::from_flag("training"),
            Some(QuantizationMode::Training)
        );
        assert_eq!(QuantizationMode::PostTraining.name(), "post-training");
    }
}
