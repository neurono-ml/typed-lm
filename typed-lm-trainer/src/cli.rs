//! Command-line interface for the trainer.
//!
//! Two subcommands share the same vocabulary as the rest of the workspace:
//! `train` fine-tunes a base checkpoint with LoRA/QLoRA, and `quantize` applies
//! post-training quantization (FP8/FP4) to a merged or base checkpoint. Every
//! flag carries a sane default so `typed-lm-trainer train --dataset data.jsonl`
//! is a complete invocation.

use std::path::PathBuf;

use candle_core::Device;
use clap::{CommandFactory, Parser, Subcommand};

use typed_lm_common::device::DeviceResolver;
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
    /// Full fine-tuning of every parameter from an existing checkpoint.
    Full,
    /// Training of every parameter from a randomly initialized model.
    FromScratch,
}

impl TrainingMethod {
    /// Parses the `--method` flag value.
    pub fn from_flag(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "lora" => Some(Self::Lora),
            "qlora" => Some(Self::QLoRa),
            "full" => Some(Self::Full),
            "from-scratch" => Some(Self::FromScratch),
            _ => None,
        }
    }

    /// Canonical flag name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Lora => "lora",
            Self::QLoRa => "qlora",
            Self::Full => "full",
            Self::FromScratch => "from-scratch",
        }
    }

    /// Whether this method produces a saved LoRA/QLoRA adapter.
    pub fn is_adapter_method(self) -> bool {
        matches!(self, Self::Lora | Self::QLoRa)
    }

    /// Whether this method trains every parameter of the model.
    pub fn trains_all_parameters(self) -> bool {
        matches!(self, Self::Full | Self::FromScratch)
    }
}

/// Execution device requested by `--device`.
///
/// `auto` keeps the same precedence the server uses (CUDA > Metal > CPU), so a
/// binary built with the `cuda` feature trains on the GPU without extra flags;
/// `cpu` and `cuda` pin the choice explicitly (useful for parity checks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum TrainerDevice {
    /// CUDA when available, otherwise CPU.
    Auto,
    /// Force the CPU.
    Cpu,
    /// Force the first CUDA device (fails when no GPU is available).
    Cuda,
}

impl TrainerDevice {
    /// Resolves the requested device into a concrete candle device.
    pub fn resolve(self) -> anyhow::Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda => Device::new_cuda(0)
                .map_err(|error| anyhow::anyhow!("failed to open CUDA device 0: {error}")),
            Self::Auto => DeviceResolver::resolve(),
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

/// Explicit model geometry for `--method from-scratch`/`--method full`
/// when it must not come from a checkpoint `config.json`.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct ModelGeometryArguments {
    /// Architecture family (`llama`, `qwen2`, `qwen3`, `mistral`, `gemma`, `gemma2`, `gemma3`).
    #[arg(long)]
    pub architecture: Option<String>,
    /// Hidden dimension of the model.
    #[arg(long)]
    pub hidden_size: Option<usize>,
    /// Feed-forward intermediate dimension.
    #[arg(long)]
    pub intermediate_size: Option<usize>,
    /// Number of transformer blocks.
    #[arg(long)]
    pub num_hidden_layers: Option<usize>,
    /// Number of query attention heads.
    #[arg(long)]
    pub num_attention_heads: Option<usize>,
    /// Number of key/value attention heads (grouped-query attention).
    #[arg(long)]
    pub num_key_value_heads: Option<usize>,
    /// Vocabulary size.
    #[arg(long)]
    pub vocab_size: Option<usize>,
    /// Maximum supported sequence length.
    #[arg(long)]
    pub max_position_embeddings: Option<usize>,
    /// Rotary embedding base frequency.
    #[arg(long)]
    pub rope_theta: Option<f32>,
    /// Root-mean-square normalization epsilon.
    #[arg(long)]
    pub rms_norm_eps: Option<f64>,
    /// Whether the input and output embeddings share weights.
    #[arg(long)]
    pub tie_word_embeddings: Option<bool>,
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

    /// Execution device: `auto` (CUDA > Metal > CPU), `cpu` or `cuda`.
    #[arg(long, value_enum, default_value = "auto")]
    pub device: TrainerDevice,

    /// Explicit model geometry for full fine-tuning without a checkpoint.
    #[command(flatten)]
    pub geometry: ModelGeometryArguments,

    /// Deterministic initialization seed (`--method from-scratch`).
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Optional TOML configuration file; explicit CLI flags take precedence.
    #[arg(long)]
    pub configuration_file: Option<PathBuf>,

    /// Tokenizer artifact used by `--method from-scratch` (`tokenizer.json`).
    ///
    /// Checkpoint-based methods read the tokenizer from the checkpoint; a
    /// from-scratch run has none, so the tokenizer must be provided here or via
    /// the TOML `[tokenizer] file` key.
    #[arg(long)]
    pub tokenizer_file: Option<PathBuf>,
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

    /// Execution device: `auto` (CUDA > Metal > CPU), `cpu` or `cuda`.
    #[arg(long, value_enum, default_value = "auto")]
    pub device: TrainerDevice,
}

impl TrainerArguments {
    /// Parses the command line, applies the optional TOML configuration file and
    /// returns the resolved arguments with **CLI > TOML > default** precedence.
    ///
    /// The first pass lets clap parse the raw command line (so `--configuration-file`
    /// and the subcommand are known); the configuration file is then loaded, and
    /// [`resolve_train_arguments`](crate::configuration_resolution::resolve_train_arguments)
    /// fills every field that was not provided explicitly. A malformed or
    /// unknown-key TOML file is a typed configuration error.
    pub fn parse_with_configuration() -> anyhow::Result<Self> {
        let matches = Self::command().get_matches();
        Self::from_matches_with_configuration(&matches)
    }

    /// Applies a configuration file to already-parsed matches.
    ///
    /// Kept separate from [`Self::parse_with_configuration`] so integration tests
    /// can drive the resolution from a controlled `ArgMatches` instance.
    pub fn from_matches_with_configuration(matches: &clap::ArgMatches) -> anyhow::Result<Self> {
        match matches.subcommand() {
            Some(("train", train_matches)) => {
                let configuration = Self::load_configuration_from(train_matches)?;
                let resolved = crate::configuration_resolution::resolve_train_arguments(
                    train_matches,
                    &configuration,
                )?;
                Ok(Self {
                    command: Command::Train(resolved),
                })
            }
            _ => <Self as clap::FromArgMatches>::from_arg_matches(matches)
                .map_err(|error| anyhow::anyhow!(error)),
        }
    }

    /// Loads the TOML file named by `--configuration-file`, if the flag is set.
    fn load_configuration_from(
        matches: &clap::ArgMatches,
    ) -> anyhow::Result<crate::configuration_file::ConfigurationFile> {
        match matches.get_one::<PathBuf>("configuration_file") {
            Some(path) => crate::configuration_file::load_configuration_file(path),
            None => Ok(crate::configuration_file::ConfigurationFile::default()),
        }
    }
}

impl TrainArguments {
    /// Resolves the parsed `--method` value into a [`TrainingMethod`].
    pub fn training_method(&self) -> anyhow::Result<TrainingMethod> {
        TrainingMethod::from_flag(&self.method).ok_or_else(|| {
            anyhow::anyhow!(
                "invalid --method '{}': expected 'lora', 'qlora', 'full' or 'from-scratch'",
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

    /// Whether the resolved training method needs an explicit model geometry.
    pub fn requires_geometry(&self) -> anyhow::Result<bool> {
        let method = self.training_method()?;
        Ok(method.trains_all_parameters())
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
        assert_eq!(train.device, TrainerDevice::Auto);
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
            "bogus",
        ])?;
        let Command::Train(train) = arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert!(train.training_method().is_err());
        Ok(())
    }

    #[test]
    fn full_and_from_scratch_methods_round_trip() -> anyhow::Result<()> {
        let full_arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--method",
            "full",
        ])?;
        let Command::Train(full_train) = full_arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert_eq!(full_train.training_method()?, TrainingMethod::Full);
        assert_eq!(full_train.training_method()?.name(), "full");
        assert!(full_train.training_method()?.trains_all_parameters());
        assert!(!full_train.training_method()?.is_adapter_method());

        let from_scratch_arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--method",
            "from-scratch",
        ])?;
        let Command::Train(from_scratch_train) = from_scratch_arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert_eq!(
            from_scratch_train.training_method()?,
            TrainingMethod::FromScratch
        );
        assert_eq!(from_scratch_train.training_method()?.name(), "from-scratch");
        assert!(from_scratch_train.training_method()?.trains_all_parameters());
        assert!(!from_scratch_train.training_method()?.is_adapter_method());
        Ok(())
    }

    #[test]
    fn adapter_methods_are_flagged_as_adapters() -> anyhow::Result<()> {
        assert!(TrainingMethod::Lora.is_adapter_method());
        assert!(TrainingMethod::QLoRa.is_adapter_method());
        assert!(!TrainingMethod::Lora.trains_all_parameters());
        assert!(!TrainingMethod::QLoRa.trains_all_parameters());
        assert_eq!(TrainingMethod::from_flag("FULL"), Some(TrainingMethod::Full));
        assert_eq!(
            TrainingMethod::from_flag("FROM-SCRATCH"),
            Some(TrainingMethod::FromScratch)
        );
        Ok(())
    }

    #[test]
    fn configuration_file_flag_is_parsed() -> anyhow::Result<()> {
        let arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--configuration-file",
            "path.toml",
        ])?;
        let Command::Train(train) = arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert_eq!(
            train.configuration_file,
            Some(PathBuf::from("path.toml"))
        );
        Ok(())
    }

    #[test]
    fn seed_flag_defaults_and_parses() -> anyhow::Result<()> {
        let default_arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
        ])?;
        let Command::Train(default_train) = default_arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert_eq!(default_train.seed, 42);

        let seeded_arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--seed",
            "7",
        ])?;
        let Command::Train(seeded_train) = seeded_arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert_eq!(seeded_train.seed, 7);
        Ok(())
    }

    #[test]
    fn geometry_flags_parse() -> anyhow::Result<()> {
        let arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--architecture",
            "qwen3",
            "--hidden-size",
            "64",
            "--intermediate-size",
            "128",
            "--num-hidden-layers",
            "4",
            "--num-attention-heads",
            "8",
            "--num-key-value-heads",
            "2",
            "--vocab-size",
            "256",
            "--max-position-embeddings",
            "512",
            "--rope-theta",
            "10000.0",
            "--rms-norm-eps",
            "0.00001",
            "--tie-word-embeddings",
            "true",
        ])?;
        let Command::Train(train) = arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        let geometry = train.geometry;
        assert_eq!(geometry.architecture.as_deref(), Some("qwen3"));
        assert_eq!(geometry.hidden_size, Some(64));
        assert_eq!(geometry.intermediate_size, Some(128));
        assert_eq!(geometry.num_hidden_layers, Some(4));
        assert_eq!(geometry.num_attention_heads, Some(8));
        assert_eq!(geometry.num_key_value_heads, Some(2));
        assert_eq!(geometry.vocab_size, Some(256));
        assert_eq!(geometry.max_position_embeddings, Some(512));
        assert_eq!(geometry.rope_theta, Some(10000.0));
        assert_eq!(geometry.rms_norm_eps, Some(0.00001));
        assert_eq!(geometry.tie_word_embeddings, Some(true));
        Ok(())
    }

    #[test]
    fn geometry_fields_default_to_none() -> anyhow::Result<()> {
        let arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
        ])?;
        let Command::Train(train) = arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        let geometry = train.geometry;
        assert!(geometry.architecture.is_none());
        assert!(geometry.hidden_size.is_none());
        assert!(geometry.intermediate_size.is_none());
        assert!(geometry.num_hidden_layers.is_none());
        assert!(geometry.num_attention_heads.is_none());
        assert!(geometry.num_key_value_heads.is_none());
        assert!(geometry.vocab_size.is_none());
        assert!(geometry.max_position_embeddings.is_none());
        assert!(geometry.rope_theta.is_none());
        assert!(geometry.rms_norm_eps.is_none());
        assert!(geometry.tie_word_embeddings.is_none());
        Ok(())
    }

    #[test]
    fn requires_geometry_depends_on_the_method() -> anyhow::Result<()> {
        let full_arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--method",
            "full",
        ])?;
        let Command::Train(full_train) = full_arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert!(full_train.requires_geometry()?);

        let from_scratch_arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--method",
            "from-scratch",
        ])?;
        let Command::Train(from_scratch_train) = from_scratch_arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert!(from_scratch_train.requires_geometry()?);

        let lora_arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--method",
            "lora",
        ])?;
        let Command::Train(lora_train) = lora_arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert!(!lora_train.requires_geometry()?);

        let qlora_arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--method",
            "qlora",
        ])?;
        let Command::Train(qlora_train) = qlora_arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert!(!qlora_train.requires_geometry()?);
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
        assert_eq!(quantize.device, TrainerDevice::Auto);
        Ok(())
    }

    #[test]
    fn device_flag_is_parsed_for_both_subcommands() -> anyhow::Result<()> {
        let train_arguments = TrainerArguments::try_parse_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--device",
            "cuda",
        ])?;
        let Command::Train(train) = train_arguments.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert_eq!(train.device, TrainerDevice::Cuda);

        let quantize_arguments =
            TrainerArguments::try_parse_from(["typed-lm-trainer", "quantize", "--device", "cpu"])?;
        let Command::Quantize(quantize) = quantize_arguments.command else {
            return Err(anyhow::anyhow!("expected the quantize subcommand"));
        };
        assert_eq!(quantize.device, TrainerDevice::Cpu);
        Ok(())
    }

    #[test]
    fn cpu_device_resolves_without_a_gpu() -> anyhow::Result<()> {
        let device = TrainerDevice::Cpu.resolve()?;
        assert!(device.is_cpu());
        Ok(())
    }

    #[test]
    fn a_configuration_file_supplies_absent_values() -> anyhow::Result<()> {
        use clap::CommandFactory;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("training.toml");
        std::fs::write(
            &path,
            "[run]\nmethod = \"from-scratch\"\nseed = 7\nepochs = 9\n",
        )?;
        let matches = TrainerArguments::command().try_get_matches_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--configuration-file",
            path.to_string_lossy().as_ref(),
        ])?;
        let resolved = TrainerArguments::from_matches_with_configuration(&matches)?;
        let Command::Train(train) = resolved.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert_eq!(train.method, "from-scratch");
        assert_eq!(train.seed, 7);
        assert_eq!(train.epochs, 9);
        assert_eq!(train.training_method()?, TrainingMethod::FromScratch);
        Ok(())
    }

    #[test]
    fn an_explicit_flag_overrides_the_configuration_file() -> anyhow::Result<()> {
        use clap::CommandFactory;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("training.toml");
        std::fs::write(&path, "[run]\nmethod = \"from-scratch\"\nepochs = 9\n")?;
        let matches = TrainerArguments::command().try_get_matches_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--epochs",
            "2",
            "--configuration-file",
            path.to_string_lossy().as_ref(),
        ])?;
        let resolved = TrainerArguments::from_matches_with_configuration(&matches)?;
        let Command::Train(train) = resolved.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert_eq!(train.epochs, 2);
        assert_eq!(train.method, "from-scratch");
        Ok(())
    }

    #[test]
    fn a_broken_configuration_file_is_an_error() -> anyhow::Result<()> {
        use clap::CommandFactory;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("broken.toml");
        std::fs::write(&path, "[run\n")?;
        let matches = TrainerArguments::command().try_get_matches_from([
            "typed-lm-trainer",
            "train",
            "--dataset",
            "data",
            "--configuration-file",
            path.to_string_lossy().as_ref(),
        ])?;
        assert!(TrainerArguments::from_matches_with_configuration(&matches).is_err());
        Ok(())
    }

    #[test]
    fn no_configuration_file_keeps_the_defaults() -> anyhow::Result<()> {
        use clap::CommandFactory;
        let matches = TrainerArguments::command()
            .try_get_matches_from(["typed-lm-trainer", "train", "--dataset", "data"])?;
        let resolved = TrainerArguments::from_matches_with_configuration(&matches)?;
        let Command::Train(train) = resolved.command else {
            return Err(anyhow::anyhow!("expected the train subcommand"));
        };
        assert_eq!(train.method, "lora");
        assert_eq!(train.epochs, 3);
        assert_eq!(train.seed, 42);
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
