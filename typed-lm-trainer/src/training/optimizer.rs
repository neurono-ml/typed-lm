//! Optimizer schedule, gradient clipping and accumulation.
//!
//! [`build_optimizer`] constructs a `candle_nn::optim::AdamW` over the adapter
//! variables. [`LearningRateSchedule`] implements linear warmup followed by a
//! cosine decay to `minimum_ratio * peak`; [`GradientAccumulator`] averages the
//! micro-batch losses before a single optimizer step; [`clip_gradients`] rescales
//! the gradient store by the global norm so a single explosive update cannot
//! destabilize training.

use std::collections::HashMap;

use candle_core::backprop::GradStore;
use candle_core::{Result, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};

/// Hyper-parameters for the AdamW optimizer used by the trainer.
#[derive(Debug, Clone, Copy)]
pub struct OptimizerConfiguration {
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}

impl Default for OptimizerConfiguration {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Builds an AdamW optimizer over the given variables.
pub fn build_optimizer(
    variables: Vec<Var>,
    configuration: OptimizerConfiguration,
) -> Result<AdamW> {
    let params = ParamsAdamW {
        lr: configuration.learning_rate,
        beta1: configuration.beta1,
        beta2: configuration.beta2,
        eps: configuration.epsilon,
        weight_decay: configuration.weight_decay,
    };
    AdamW::new(variables, params)
}

/// Linear warmup followed by cosine decay.
#[derive(Debug, Clone, Copy)]
pub struct LearningRateSchedule {
    peak: f64,
    minimum_ratio: f64,
    warmup_steps: usize,
    total_steps: usize,
}

impl LearningRateSchedule {
    /// Builds a schedule; a zero `warmup_steps` starts at the peak rate.
    pub fn new(peak: f64, warmup_steps: usize, total_steps: usize, minimum_ratio: f64) -> Self {
        Self {
            peak,
            minimum_ratio: minimum_ratio.clamp(0.0, 1.0),
            warmup_steps,
            total_steps: total_steps.max(1),
        }
    }

    /// The learning rate at `step` (zero-based, before the optimizer step).
    pub fn learning_rate(&self, step: usize) -> f64 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            let progress = (step + 1) as f64 / self.warmup_steps as f64;
            return self.peak * progress;
        }
        let decay_span = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let decay_step = step.saturating_sub(self.warmup_steps).min(decay_span);
        let progress = decay_step as f64 / decay_span as f64;
        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        let minimum = self.peak * self.minimum_ratio;
        minimum + (self.peak - minimum) * cosine
    }
}

/// Applies the schedule to an optimizer before a step.
pub fn apply_learning_rate<O: Optimizer>(
    optimizer: &mut O,
    schedule: &LearningRateSchedule,
    step: usize,
) {
    optimizer.set_learning_rate(schedule.learning_rate(step));
}

/// Accumulates micro-batch gradients, averaging them before a step.
#[derive(Debug, Default)]
pub struct GradientAccumulator {
    accumulated: HashMap<candle_core::TensorId, Tensor>,
    micro_batch_count: usize,
}

impl GradientAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of micro-batches accumulated so far.
    pub fn micro_batch_count(&self) -> usize {
        self.micro_batch_count
    }

    /// Adds one micro-batch loss gradient, scaled to be averaged later.
    pub fn accumulate(&mut self, variables: &[Var], gradients: &GradStore) -> Result<()> {
        self.micro_batch_count += 1;
        for variable in variables {
            let Some(gradient) = gradients.get(variable) else {
                continue;
            };
            let identifier = variable.id();
            let scaled = gradient.detach();
            match self.accumulated.get(&identifier) {
                Some(existing) => {
                    self.accumulated.insert(identifier, (existing + scaled)?);
                }
                None => {
                    self.accumulated.insert(identifier, scaled);
                }
            }
        }
        Ok(())
    }

    /// Returns the mean gradient store and resets the accumulator.
    pub fn take_averaged(&mut self, variables: &[Var]) -> Result<GradStore> {
        let mut gradients = GradStore::default();
        let divisor = self.micro_batch_count.max(1) as f64;
        for variable in variables {
            if let Some(accumulated) = self.accumulated.remove(&variable.id()) {
                let averaged = (accumulated / divisor)?;
                gradients.insert(variable, averaged);
            }
        }
        self.micro_batch_count = 0;
        Ok(gradients)
    }
}

/// Clips gradients by the global L2 norm, returning the applied scale factor.
///
/// If the global norm is at most `maximum_norm`, gradients are left untouched
/// (scale `1.0`); otherwise every gradient is multiplied by
/// `maximum_norm / global_norm`.
pub fn clip_gradients(
    variables: &[Var],
    gradients: &mut GradStore,
    maximum_norm: f64,
) -> Result<f64> {
    if maximum_norm <= 0.0 {
        return Ok(1.0);
    }
    let mut squared_sum = 0.0_f64;
    for variable in variables {
        if let Some(gradient) = gradients.get(variable) {
            let values = gradient.flatten_all()?.to_vec1::<f32>()?;
            for value in values {
                squared_sum += (value as f64) * (value as f64);
            }
        }
    }
    let global_norm = squared_sum.sqrt();
    if global_norm <= maximum_norm || global_norm == 0.0 {
        return Ok(1.0);
    }
    let scale = maximum_norm / global_norm;
    let mut updates: Vec<(Var, Tensor)> = Vec::with_capacity(variables.len());
    for variable in variables {
        if let Some(gradient) = gradients.get(variable) {
            updates.push((variable.clone(), (gradient * scale)?));
        }
    }
    for (variable, clipped) in updates {
        gradients.insert(&variable, clipped);
    }
    Ok(scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn warmup_increases_then_cosine_decays() {
        let schedule = LearningRateSchedule::new(1.0, 10, 100, 0.1);
        assert!(schedule.learning_rate(0) < schedule.learning_rate(9));
        assert!(
            (schedule.learning_rate(9) - 1.0).abs() < 1e-9,
            "peak at warmup"
        );
        assert!(schedule.learning_rate(50) < schedule.learning_rate(15));
        assert!(schedule.learning_rate(100) < schedule.learning_rate(50));
        // The cosine tail approaches the configured minimum ratio.
        assert!((schedule.learning_rate(100) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn schedule_is_monotone_during_warmup() {
        let schedule = LearningRateSchedule::new(2.0, 20, 200, 0.0);
        let mut previous = 0.0;
        for step in 0..20 {
            let rate = schedule.learning_rate(step);
            assert!(rate >= previous, "warmup must be non-decreasing");
            previous = rate;
        }
    }

    #[test]
    fn zero_warmup_starts_at_the_peak() {
        let schedule = LearningRateSchedule::new(3.0, 0, 10, 0.0);
        assert!((schedule.learning_rate(0) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn clipping_scales_down_a_large_gradient() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let variable = Var::from_tensor(&Tensor::new(&[3.0_f32, 4.0], &device)?)?;
        let mut gradients = GradStore::default();
        // Gradient norm is 3^2 + 4^2 = 25 -> norm 5.
        gradients.insert(&variable, Tensor::new(&[3.0_f32, 4.0], &device)?);
        let scale = clip_gradients(std::slice::from_ref(&variable), &mut gradients, 1.0)?;
        assert!((scale - 0.2).abs() < 1e-6);
        let clipped = gradients
            .get(&variable)
            .ok_or_else(|| anyhow::anyhow!("missing clipped gradient"))?;
        let values = clipped.to_vec1::<f32>()?;
        let norm = (values[0] * values[0] + values[1] * values[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "clipped norm must be 1");
        Ok(())
    }

    #[test]
    fn clipping_leaves_a_small_gradient_untouched() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let variable = Var::from_tensor(&Tensor::new(&[0.1_f32], &device)?)?;
        let mut gradients = GradStore::default();
        gradients.insert(&variable, Tensor::new(&[0.1_f32], &device)?);
        let scale = clip_gradients(std::slice::from_ref(&variable), &mut gradients, 1.0)?;
        assert!((scale - 1.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn accumulator_averages_two_micro_batches() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let variable = Var::from_tensor(&Tensor::new(&[0.0_f32], &device)?)?;
        let mut accumulator = GradientAccumulator::new();
        let mut first = GradStore::default();
        first.insert(&variable, Tensor::new(&[2.0_f32], &device)?);
        accumulator.accumulate(std::slice::from_ref(&variable), &first)?;
        let mut second = GradStore::default();
        second.insert(&variable, Tensor::new(&[4.0_f32], &device)?);
        accumulator.accumulate(std::slice::from_ref(&variable), &second)?;
        assert_eq!(accumulator.micro_batch_count(), 2);
        let averaged = accumulator.take_averaged(std::slice::from_ref(&variable))?;
        let value = averaged
            .get(&variable)
            .ok_or_else(|| anyhow::anyhow!("missing averaged gradient"))?
            .to_vec1::<f32>()?;
        assert!((value[0] - 3.0).abs() < 1e-6);
        assert_eq!(accumulator.micro_batch_count(), 0);
        Ok(())
    }

    #[test]
    fn optimizer_steps_with_the_schedule_learning_rate() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let variable = Var::from_tensor(&Tensor::new(&[1.0_f32], &device)?)?;
        let mut optimizer =
            build_optimizer(vec![variable.clone()], OptimizerConfiguration::default())?;
        let schedule = LearningRateSchedule::new(0.1, 0, 10, 0.0);
        apply_learning_rate(&mut optimizer, &schedule, 0);
        assert!((optimizer.learning_rate() - 0.1).abs() < 1e-9);
        Ok(())
    }
}
