//! A fully-trainable linear projection.
//!
//! Unlike [`crate::model::lora::LoRALinear`], whose base projection is frozen,
//! [`TrainableLinear`] keeps both the weight and the optional bias as
//! [`Var`]s. This is the building block for full-parameter and from-scratch
//! training, where every parameter must receive a gradient and appear in the
//! `VarMap` handed to the optimizer.
//!
//! The layer stores its parameters in row-major `(output_features,
//! input_features)` order, matching the dense checkpoint convention, and its
//! forward pass computes `input @ weight^T (+ bias)`.

use candle_core::{Module, Result, Tensor, Var};
use candle_nn::{Init, VarBuilder};

use crate::error::TrainerError;

/// A linear projection whose weight (and optional bias) are trainable variables.
#[derive(Debug, Clone)]
pub struct TrainableLinear {
    /// Trainable weight, shaped `(output_features, input_features)`.
    weight: Var,
    /// Optional trainable bias, shaped `(output_features,)`.
    bias: Option<Var>,
}

impl TrainableLinear {
    /// Builds a fresh layer with random weight and zero-initialized bias.
    ///
    /// The weight is drawn from a zero-mean Gaussian with standard deviation
    /// `0.02`; the bias, when requested, starts at zero. Both variables are
    /// created in `variable_builder` under `path` so they participate in the
    /// `VarMap` and appear in `all_vars` for the optimizer.
    pub fn new(
        output_features: usize,
        input_features: usize,
        with_bias: bool,
        variable_builder: &VarBuilder,
        path: &str,
    ) -> anyhow::Result<Self> {
        let branch = variable_builder.pp(path);
        let weight = Var::from_tensor(&branch.get_with_hints(
            (output_features, input_features),
            "weight",
            Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?)?;
        let bias = if with_bias {
            Some(Var::from_tensor(&branch.get_with_hints(
                (output_features,),
                "bias",
                Init::Const(0.0),
            )?)?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    /// Builds a layer whose variables are set from the provided tensors.
    ///
    /// This is the entry point for `full` fine-tuning: the variables are
    /// created in `variable_builder` (so they are registered in the `VarMap`)
    /// and immediately overwritten with the checkpoint values. The provided
    /// weight must be 2-D `(output_features, input_features)` and, when given,
    /// the bias must be 1-D `(output_features,)`; a mismatch is an error.
    pub fn from_tensors(
        weight: Tensor,
        bias: Option<Tensor>,
        variable_builder: &VarBuilder,
        path: &str,
    ) -> anyhow::Result<Self> {
        let (output_features, input_features) = weight.dims2()?;
        let provided_bias_features = match &bias {
            Some(bias_tensor) => Some(bias_tensor.dims1()?),
            None => None,
        };
        if let Some(provided_features) = provided_bias_features {
            if provided_features != output_features {
                return Err(TrainerError::Model(format!(
                    "'{path}' bias shape mismatch: expected ({output_features},), got ({provided_features},)"
                ))
                .into());
            }
        }

        let branch = variable_builder.pp(path);
        let weight_variable = Var::from_tensor(&branch.get_with_hints(
            (output_features, input_features),
            "weight",
            Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?)?;
        weight_variable.set(&weight)?;

        let bias_variable = match bias {
            Some(bias_tensor) => {
                let variable = Var::from_tensor(&branch.get_with_hints(
                    (output_features,),
                    "bias",
                    Init::Const(0.0),
                )?)?;
                variable.set(&bias_tensor)?;
                Some(variable)
            }
            None => None,
        };

        Ok(Self {
            weight: weight_variable,
            bias: bias_variable,
        })
    }

    /// The trainable weight, shaped `(output_features, input_features)`.
    pub fn weight(&self) -> &Var {
        &self.weight
    }

    /// The optional trainable bias, shaped `(output_features,)`.
    pub fn bias(&self) -> Option<&Var> {
        self.bias.as_ref()
    }

    /// All trainable variables in a deterministic order: weight, then bias.
    pub fn variables(&self) -> Vec<Var> {
        let mut collected = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            collected.push(bias.clone());
        }
        collected
    }
}

impl Module for TrainableLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match input.dims() {
            [batch, sequence, features] => {
                let flat = input.reshape((*batch * *sequence, *features))?;
                let projected = flat.matmul(&self.weight.as_tensor().t()?)?;
                let projected = match &self.bias {
                    Some(bias) => projected.broadcast_add(bias.as_tensor())?,
                    None => projected,
                };
                projected.reshape((*batch, *sequence, self.weight.dim(0)?))
            }
            _ => {
                let projected = input.matmul(&self.weight.as_tensor().t()?)?;
                match &self.bias {
                    Some(bias) => projected.broadcast_add(bias.as_tensor()),
                    None => Ok(projected),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    fn build_layer(with_bias: bool) -> anyhow::Result<(TrainableLinear, VarMap)> {
        let device = Device::Cpu;
        let variable_map = VarMap::new();
        let variable_builder =
            VarBuilder::from_varmap(&variable_map, candle_core::DType::F32, &device);
        let layer = TrainableLinear::new(3, 2, with_bias, &variable_builder, "projection")?;
        Ok((layer, variable_map))
    }

    fn assert_tensors_close(left: &Tensor, right: &Tensor) -> anyhow::Result<()> {
        let left_values = left.flatten_all()?.to_vec1::<f32>()?;
        let right_values = right.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(left_values.len(), right_values.len());
        for (left_value, right_value) in left_values.iter().zip(right_values.iter()) {
            assert!(
                (left_value - right_value).abs() < 1e-5,
                "expected {left_value} to match {right_value}"
            );
        }
        Ok(())
    }

    #[test]
    fn forward_matches_a_manual_matmul() -> anyhow::Result<()> {
        let (layer, _variable_map) = build_layer(false)?;
        let device = Device::Cpu;
        let weight = Tensor::new(&[[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &device)?;
        layer.weight().set(&weight)?;
        let input = Tensor::new(&[[1.0_f32, 1.0], [2.0, 0.5]], &device)?;

        let output = layer.forward(&input)?;
        let expected = input.matmul(&weight.t()?)?;
        assert_tensors_close(&output, &expected)?;
        Ok(())
    }

    #[test]
    fn forward_broadcasts_the_bias() -> anyhow::Result<()> {
        let (layer, _variable_map) = build_layer(true)?;
        let device = Device::Cpu;
        layer.weight().set(&Tensor::zeros((3, 2), candle_core::DType::F32, &device)?)?;
        let bias = layer
            .bias()
            .ok_or_else(|| anyhow::anyhow!("bias must be present"))?;
        bias.set(&Tensor::new(&[10.0_f32, 20.0, 30.0], &device)?)?;
        let input = Tensor::new(&[[1.0_f32, 2.0], [3.0, 4.0]], &device)?;

        let output = layer.forward(&input)?;
        let expected = Tensor::new(&[[10.0_f32, 20.0, 30.0], [10.0, 20.0, 30.0]], &device)?;
        assert_tensors_close(&output, &expected)?;
        Ok(())
    }

    #[test]
    fn three_dimensional_input_is_supported() -> anyhow::Result<()> {
        let (layer, _variable_map) = build_layer(false)?;
        let device = Device::Cpu;
        let input = Tensor::new(
            &[
                [[1.0_f32, 1.0], [2.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0]],
            ],
            &device,
        )?;
        let output = layer.forward(&input)?;
        assert_eq!(output.dims(), &[2, 2, 3]);
        Ok(())
    }

    #[test]
    fn variables_lists_weight_and_optional_bias() -> anyhow::Result<()> {
        let (without_bias, _first_map) = build_layer(false)?;
        assert_eq!(without_bias.variables().len(), 1);

        let (with_bias, _second_map) = build_layer(true)?;
        let variables = with_bias.variables();
        assert_eq!(variables.len(), 2);
        assert_eq!(variables[0].dims(), without_bias.weight().dims());
        Ok(())
    }

    #[test]
    fn from_tensors_preserves_the_provided_values() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::new(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, 1.0]], &device)?;
        let bias = Tensor::new(&[0.5_f32, -0.5, 2.0], &device)?;
        let variable_map = VarMap::new();
        let variable_builder =
            VarBuilder::from_varmap(&variable_map, candle_core::DType::F32, &device);
        let layer = TrainableLinear::from_tensors(
            weight.clone(),
            Some(bias.clone()),
            &variable_builder,
            "projection",
        )?;

        let input = Tensor::new(&[[1.0_f32, 2.0], [3.0, 4.0]], &device)?;
        let output = layer.forward(&input)?;
        let expected = input.matmul(&weight.t()?)?.broadcast_add(&bias)?;
        assert_tensors_close(&output, &expected)?;
        Ok(())
    }

    #[test]
    fn a_shape_mismatch_in_from_tensors_is_an_error() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::new(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, 1.0]], &device)?;
        let wrong_bias = Tensor::new(&[0.5_f32, -0.5], &device)?;
        let variable_map = VarMap::new();
        let variable_builder =
            VarBuilder::from_varmap(&variable_map, candle_core::DType::F32, &device);
        let result = TrainableLinear::from_tensors(
            weight,
            Some(wrong_bias),
            &variable_builder,
            "projection",
        );
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn the_weight_receives_a_gradient() -> anyhow::Result<()> {
        let (layer, _variable_map) = build_layer(true)?;
        let device = Device::Cpu;
        let input = Tensor::new(&[[1.0_f32, 2.0]], &device)?;
        let output = layer.forward(&input)?;
        let loss = output.powf(2.0)?.sum_all()?;
        let gradients = loss.backward()?;
        let weight_gradient = gradients
            .get(layer.weight())
            .ok_or_else(|| anyhow::anyhow!("the weight gradient must be present"))?;
        let values = weight_gradient.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(values.len(), 6);
        assert!(values.iter().all(|value| value.is_finite()));
        Ok(())
    }
}
