//! Fully-trainable RMSNorm whose weight is a candle `Var`.
//!
//! The fused `candle_nn::ops::rms_norm` kernel has no CPU backward pass in
//! candle 0.11, so full-parameter and from-scratch training recompute the norm
//! from differentiable primitives. Unlike the LoRA path in
//! [`crate::model::trainable_llama`] (whose norm weight is frozen), this module
//! owns a `Var` weight so the optimizer updates it. The Gemma* families offset
//! the norm weight by one (`1 + weight`), which is selected with `unit_offset`.

use candle_core::{Module, Result, Tensor, Var, D};
use candle_nn::{Init, VarBuilder};

use crate::error::TrainerError;

/// A trainable RMSNorm with a `Var` weight shaped `(hidden_size,)`.
///
/// When `unit_offset` is set the effective multiplicative weight is
/// `(1 + weight)`, matching the Gemma* families that store a zero-centered norm
/// weight; otherwise the effective weight is the weight itself.
#[derive(Debug, Clone)]
pub struct TrainableRmsNorm {
    weight: Var,
    epsilon: f64,
    unit_offset: bool,
}

impl TrainableRmsNorm {
    /// Creates a norm weight initialized to one through the variable builder.
    ///
    /// The resulting [`Var`] participates in the [`VarBuilder`]'s [`VarMap`] so
    /// the optimizer discovers it. `path` names the variable in the map.
    ///
    /// [`VarMap`]: candle_nn::VarMap
    pub fn new(
        hidden_size: usize,
        epsilon: f64,
        unit_offset: bool,
        variable_builder: &VarBuilder,
        path: &str,
    ) -> anyhow::Result<Self> {
        let branch = variable_builder.pp(path);
        let weight = branch.get_with_hints((hidden_size,), "weight", Init::Const(1.0))?;
        let weight = Var::from_tensor(&weight)?;
        Ok(Self {
            weight,
            epsilon,
            unit_offset,
        })
    }

    /// Creates a norm whose weight is set from an existing rank-1 tensor.
    ///
    /// The tensor must be rank 1; its length defines the hidden size. The
    /// variable is registered in the [`VarBuilder`]'s [`VarMap`] under `path`
    /// and its contents are overwritten with the provided weight.
    ///
    /// [`VarMap`]: candle_nn::VarMap
    pub fn from_tensor(
        weight: Tensor,
        epsilon: f64,
        unit_offset: bool,
        variable_builder: &VarBuilder,
        path: &str,
    ) -> anyhow::Result<Self> {
        let dimensions = weight.dims();
        if dimensions.len() != 1 {
            return Err(TrainerError::Model(format!(
                "the RMSNorm weight must be rank 1, got rank {}",
                dimensions.len()
            ))
            .into());
        }
        let hidden_size = dimensions[0];
        let branch = variable_builder.pp(path);
        let created = branch.get_with_hints((hidden_size,), "weight", Init::Const(1.0))?;
        let variable = Var::from_tensor(&created)?;
        variable.set(&weight)?;
        Ok(Self {
            weight: variable,
            epsilon,
            unit_offset,
        })
    }

    /// The trainable norm weight, shaped `(hidden_size,)`.
    pub fn weight(&self) -> &Var {
        &self.weight
    }

    /// The trainable variables owned by this module (only the weight).
    pub fn variables(&self) -> Vec<Var> {
        vec![self.weight.clone()]
    }

    /// The multiplicative weight, `weight` or `(weight + 1.0)` when unit-offset.
    ///
    /// The result stays connected to the underlying [`Var`] so gradients flow
    /// to the weight during back-propagation.
    pub fn effective_weight(&self) -> Result<Tensor> {
        let weight = self.weight.as_tensor();
        if self.unit_offset {
            weight + 1.0
        } else {
            Ok(weight.clone())
        }
    }
}

impl Module for TrainableRmsNorm {
    /// Differentiable RMSNorm built from primitive ops.
    ///
    /// Mirrors [`FrozenRmsNorm`](crate::model::trainable_llama) but reads the
    /// effective weight from the [`Var`] every call, keeping the graph connected.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden_size = input.dim(D::Minus1)?;
        let mean_square = (input.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let normalized = input.broadcast_div(&(mean_square + self.epsilon)?.sqrt()?)?;
        normalized.broadcast_mul(&self.effective_weight()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    const EPSILON: f64 = 1e-5;

    fn build_builder(device: &Device) -> anyhow::Result<(VarMap, VarBuilder<'_>)> {
        let variable_map = VarMap::new();
        let builder = VarBuilder::from_varmap(&variable_map, DType::F32, device);
        Ok((variable_map, builder))
    }

    #[test]
    fn forward_matches_a_manual_rms_norm() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let (_variable_map, builder) = build_builder(&device)?;
        let norm = TrainableRmsNorm::new(4, EPSILON, false, &builder, "norm")?;

        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let input = Tensor::new(&[values], &device)?;
        let output = norm.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;

        let mean_square = values.iter().map(|value| value * value).sum::<f32>() / 4.0;
        let denominator = (mean_square + EPSILON as f32).sqrt();
        for (actual, expected) in output.iter().zip(values.iter().map(|v| v / denominator)) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "expected {expected}, got {actual}"
            );
        }
        Ok(())
    }

    #[test]
    fn unit_offset_adds_one_to_the_weight() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let (_variable_map, builder) = build_builder(&device)?;
        let zeros = Tensor::zeros((4,), DType::F32, &device)?;
        let offset_norm =
            TrainableRmsNorm::from_tensor(zeros.clone(), EPSILON, true, &builder, "offset")?;
        let plain_norm = TrainableRmsNorm::from_tensor(zeros, EPSILON, false, &builder, "plain")?;

        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let input = Tensor::new(&[values], &device)?;
        let offset_output = offset_norm
            .forward(&input)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let plain_output = plain_norm
            .forward(&input)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let mean_square = values.iter().map(|value| value * value).sum::<f32>() / 4.0;
        let denominator = (mean_square + EPSILON as f32).sqrt();
        for (actual, expected) in offset_output
            .iter()
            .zip(values.iter().map(|v| v / denominator))
        {
            assert!(
                (actual - expected).abs() < 1e-5,
                "unit offset must normalize as weight one: expected {expected}, got {actual}"
            );
        }
        for actual in &plain_output {
            assert!(
                actual.abs() < 1e-6,
                "a zero weight without offset must zero the output, got {actual}"
            );
        }
        Ok(())
    }

    #[test]
    fn from_tensor_preserves_the_provided_weight() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let (_variable_map, builder) = build_builder(&device)?;
        let weight = Tensor::new(&[0.5_f32, 1.5, -2.0], &device)?;
        let norm = TrainableRmsNorm::from_tensor(weight.clone(), EPSILON, false, &builder, "norm")?;

        let stored = norm.weight().as_tensor().flatten_all()?.to_vec1::<f32>()?;
        let expected = weight.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(stored.len(), expected.len());
        for (actual, expected) in stored.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "expected {expected}, got {actual}"
            );
        }
        Ok(())
    }

    #[test]
    fn the_weight_receives_a_gradient() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let (_variable_map, builder) = build_builder(&device)?;
        let norm = TrainableRmsNorm::new(4, EPSILON, false, &builder, "norm")?;

        let input = Tensor::new(&[[1.0_f32, 2.0, 3.0, 4.0]], &device)?;
        let output = norm.forward(&input)?;
        let loss = output.sum_all()?;
        let gradients = loss.backward()?;
        let gradient = gradients
            .get(norm.weight())
            .ok_or_else(|| anyhow::anyhow!("the weight must receive a gradient"))?;
        let values = gradient.flatten_all()?.to_vec1::<f32>()?;
        assert!(!values.is_empty(), "the gradient must not be empty");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "the weight gradient must be finite"
        );
        Ok(())
    }

    #[test]
    fn variables_contains_the_weight() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let (_variable_map, builder) = build_builder(&device)?;
        let weight = Tensor::new(&[0.5_f32, 1.5, -2.0], &device)?;
        let norm = TrainableRmsNorm::from_tensor(weight, EPSILON, false, &builder, "norm")?;

        let variables = norm.variables();
        assert_eq!(variables.len(), 1, "the only variable is the weight");
        let exposed = variables[0].as_tensor().flatten_all()?.to_vec1::<f32>()?;
        let direct = norm.weight().as_tensor().flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(exposed, direct);
        Ok(())
    }

    #[test]
    fn a_wrong_rank_weight_is_rejected() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let (_variable_map, builder) = build_builder(&device)?;
        let matrix = Tensor::zeros((2, 3), DType::F32, &device)?;
        let result = TrainableRmsNorm::from_tensor(matrix, EPSILON, false, &builder, "norm");
        assert!(result.is_err(), "a rank-2 weight must be rejected");
        Ok(())
    }
}
