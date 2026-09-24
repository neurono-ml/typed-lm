//! LoRA (low-rank adaptation) linear layer.
//!
//! A [`LoRALinear`] wraps a frozen base projection with a trainable low-rank
//! update `B @ A`. The base weight is a plain (non-variable) `Tensor`, so the
//! gradient of the update never reaches it; only the `A`/`B` `Var`s are
//! trainable. Following the original LoRA recipe, `A` is initialized with a
//! small Gaussian and `B` with zeros, so the adapter starts as the identity
//! (the initial forward pass equals the frozen base). The update is scaled by
//! `alpha / rank`.
//!
//! [`LoRALinear::merge`] folds the adapter back into a dense weight so a
//! fine-tuned model can be exported without the LoRA branches.

use candle_core::{Module, Result, Tensor, Var};
use candle_nn::{Init, VarBuilder};

use crate::error::TrainerError;

/// A frozen linear projection plus a trainable low-rank update.
#[derive(Debug, Clone)]
pub struct LoRALinear {
    /// Frozen base weight, shaped `(output_features, input_features)`.
    base_weight: Tensor,
    /// Optional frozen base bias.
    base_bias: Option<Tensor>,
    /// Low-rank down-projection, shaped `(rank, input_features)`.
    down_projection: Var,
    /// Low-rank up-projection, shaped `(output_features, rank)`.
    up_projection: Var,
    /// `alpha / rank` scaling applied to the low-rank update.
    scaling: f64,
    /// The adapter rank.
    rank: usize,
}

impl LoRALinear {
    /// Builds a LoRA layer around a frozen base weight.
    ///
    /// `down_projection`/`up_projection` are added to `variable_builder` so they
    /// participate in the `VarMap` and appear in `all_vars` for the optimizer.
    pub fn new(
        base_weight: Tensor,
        base_bias: Option<Tensor>,
        rank: usize,
        alpha: f64,
        variable_builder: &VarBuilder,
        path: &str,
    ) -> anyhow::Result<Self> {
        if rank == 0 {
            return Err(TrainerError::Configuration(
                "LoRA rank must be greater than zero".to_string(),
            )
            .into());
        }
        let (output_features, input_features) = base_weight.dims2()?;
        let branch = variable_builder.pp(path);
        let down_projection = branch.get_with_hints(
            (rank, input_features),
            "lora_a",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0 / (input_features as f64).sqrt(),
            },
        )?;
        let up_projection =
            branch.get_with_hints((output_features, rank), "lora_b", Init::Const(0.0))?;
        Ok(Self {
            base_weight,
            base_bias,
            down_projection: Var::from_tensor(&down_projection)?,
            up_projection: Var::from_tensor(&up_projection)?,
            scaling: alpha / rank as f64,
            rank,
        })
    }

    /// The frozen base weight, shaped `(output_features, input_features)`.
    pub fn base_weight(&self) -> &Tensor {
        &self.base_weight
    }

    /// The trainable `A` (down-projection) variable.
    pub fn down_projection(&self) -> &Var {
        &self.down_projection
    }

    /// The trainable `B` (up-projection) variable.
    pub fn up_projection(&self) -> &Var {
        &self.up_projection
    }

    /// `alpha / rank` scaling applied to the adapter.
    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    /// The scalar `alpha` implied by the stored scaling and the adapter rank.
    pub fn alpha(&self) -> f64 {
        self.scaling * self.rank as f64
    }

    /// The adapter rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Computes only the low-rank update `(B @ A) * scaling`.
    pub fn adapter_delta(&self) -> Result<Tensor> {
        let down = self.down_projection.as_tensor();
        let up = self.up_projection.as_tensor();
        let delta = up.matmul(down)?;
        delta * self.scaling
    }

    /// Folds the adapter into the frozen base weight, returning a dense weight.
    ///
    /// The result is detached from the autograd graph: exporting a merged weight
    /// is an inference operation, not a training step.
    pub fn merged_weight(&self) -> Result<Tensor> {
        let merged = (self.base_weight.clone() + self.adapter_delta()?)?;
        Ok(merged.detach())
    }

    /// A dense (weight, bias) pair equivalent to this layer after merging.
    pub fn merged_parameters(&self) -> Result<(Tensor, Option<Tensor>)> {
        Ok((self.merged_weight()?, self.base_bias.clone()))
    }
}

impl Module for LoRALinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (base, delta) = match input.dims() {
            [batch, sequence, features] => {
                let flat = input.reshape((batch * sequence, *features))?;
                let base = flat.matmul(&self.base_weight.t()?)?;
                let down = flat.matmul(&self.down_projection.as_tensor().t()?)?;
                let up = down.matmul(&self.up_projection.as_tensor().t()?)?;
                (
                    base.reshape((*batch, *sequence, self.base_weight.dim(0)?))?,
                    up.reshape((*batch, *sequence, self.base_weight.dim(0)?))?,
                )
            }
            _ => {
                let base = input.matmul(&self.base_weight.t()?)?;
                let down = input.matmul(&self.down_projection.as_tensor().t()?)?;
                let up = down.matmul(&self.up_projection.as_tensor().t()?)?;
                (base, up)
            }
        };
        let base = match &self.base_bias {
            Some(bias) => base.broadcast_add(bias)?,
            None => base,
        };
        base + (delta * self.scaling)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    fn build_layer(rank: usize, alpha: f64) -> anyhow::Result<(LoRALinear, VarMap)> {
        let device = Device::Cpu;
        let base_weight = Tensor::new(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, 1.0]], &device)?;
        let variable_map = VarMap::new();
        let builder = VarBuilder::from_varmap(&variable_map, candle_core::DType::F32, &device);
        let layer = LoRALinear::new(base_weight, None, rank, alpha, &builder, "proj")?;
        Ok((layer, variable_map))
    }

    #[test]
    fn zero_initialized_b_equals_the_frozen_base() -> anyhow::Result<()> {
        let (layer, _variable_map) = build_layer(2, 4.0)?;
        let device = Device::Cpu;
        let input = Tensor::new(&[[1.0_f32, 2.0]], &device)?;
        let base = input.matmul(&layer.base_weight().t()?)?;
        let adapted = layer.forward(&input)?;
        let base_values = base.flatten_all()?.to_vec1::<f32>()?;
        let adapted_values = adapted.flatten_all()?.to_vec1::<f32>()?;
        for (left, right) in base_values.iter().zip(adapted_values.iter()) {
            assert!((left - right).abs() < 1e-6, "B=0 must start at the base");
        }
        Ok(())
    }

    #[test]
    fn a_zero_rank_is_an_error() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let base_weight = Tensor::new(&[[1.0_f32, 0.0]], &device)?;
        let variable_map = VarMap::new();
        let variable_builder =
            VarBuilder::from_varmap(&variable_map, candle_core::DType::F32, &device);
        let result = LoRALinear::new(base_weight, None, 0, 4.0, &variable_builder, "proj");
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn nonzero_adapter_changes_the_output() -> anyhow::Result<()> {
        let (layer, _variable_map) = build_layer(2, 4.0)?;
        let device = Device::Cpu;
        let input = Tensor::new(&[[1.0_f32, 2.0]], &device)?;
        let base = input.matmul(&layer.base_weight().t()?)?;
        // Inject a non-zero B so the adapter contributes.
        layer.up_projection().set(&Tensor::new(
            &[[0.5_f32, 0.0], [0.0, 0.5], [0.5, 0.5]],
            &device,
        )?)?;
        let adapted = layer.forward(&input)?;
        let base_values = base.flatten_all()?.to_vec1::<f32>()?;
        let adapted_values = adapted.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            base_values
                .iter()
                .zip(adapted_values.iter())
                .any(|(left, right)| (left - right).abs() > 1e-4),
            "a non-zero adapter must change the output"
        );
        Ok(())
    }

    #[test]
    fn merge_preserves_the_forward_output() -> anyhow::Result<()> {
        let (layer, _variable_map) = build_layer(2, 2.0)?;
        let device = Device::Cpu;
        layer.up_projection().set(&Tensor::new(
            &[[0.25_f32, -0.5], [0.75, 0.1], [-0.2, 0.3]],
            &device,
        )?)?;
        let input = Tensor::new(&[[1.0_f32, 2.0], [0.5, -1.0]], &device)?;
        let adapted = layer.forward(&input)?;
        let (merged_weight, merged_bias) = layer.merged_parameters()?;
        assert!(merged_bias.is_none());
        let merged = input.matmul(&merged_weight.t()?)?;
        let adapted_values = adapted.flatten_all()?.to_vec1::<f32>()?;
        let merged_values = merged.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(adapted_values.len(), merged_values.len());
        for (left, right) in adapted_values.iter().zip(merged_values.iter()) {
            assert!(
                (left - right).abs() < 1e-5,
                "merge must preserve the output"
            );
        }
        Ok(())
    }

    #[test]
    fn merge_is_detached_from_autograd() -> anyhow::Result<()> {
        let (layer, _variable_map) = build_layer(2, 2.0)?;
        let merged = layer.merged_weight()?;
        assert!(!merged.is_variable());
        Ok(())
    }

    #[test]
    fn scaling_reflects_alpha_over_rank() -> anyhow::Result<()> {
        let (layer, _variable_map) = build_layer(4, 8.0)?;
        assert!((layer.scaling() - 2.0).abs() < 1e-9);
        assert_eq!(layer.rank(), 4);
        assert!((layer.alpha() - 8.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn base_bias_is_added_in_the_forward_pass() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let base_weight = Tensor::new(&[[1.0_f32, 0.0], [0.0, 1.0]], &device)?;
        let base_bias = Tensor::new(&[10.0_f32, 20.0], &device)?;
        let variable_map = VarMap::new();
        let variable_builder =
            VarBuilder::from_varmap(&variable_map, candle_core::DType::F32, &device);
        let layer = LoRALinear::new(
            base_weight,
            Some(base_bias),
            2,
            4.0,
            &variable_builder,
            "proj",
        )?;
        let input = Tensor::new(&[[1.0_f32, 1.0]], &device)?;
        let output = layer.forward(&input)?;
        let values = output.flatten_all()?.to_vec1::<f32>()?;
        assert!((values[0] - 11.0).abs() < 1e-5);
        assert!((values[1] - 21.0).abs() < 1e-5);
        Ok(())
    }
}
