//! Epoch training loop with logging, metrics and early stopping.
//!
//! The loop ties the dataset, the trainable model, the loss and the optimizer
//! together. It is generic over a [`TrainableModel`], so the same loop drives
//! Llama and Qwen2 and can be unit-tested with a lightweight mock that returns
//! predictable logits without real weights. Each epoch reports the mean loss and
//! optional accuracy; training stops early when the loss does not improve by
//! `minimum_improvement` for `patience` consecutive epochs.

use std::collections::HashMap;

use candle_core::{backprop::GradStore, Device, Tensor, Var};
use candle_nn::optim::Optimizer;

use crate::dataset::collate::TrainingBatch;
use crate::training::loss::decision_loss;
use crate::training::optimizer::{
    apply_learning_rate, build_optimizer, clip_gradients, GradientAccumulator,
    LearningRateSchedule, OptimizerConfiguration,
};

/// A model the training loop can optimize.
pub trait TrainableModel {
    /// Variables updated by the optimizer.
    fn variables(&self) -> Vec<Var>;

    /// Full-vocabulary logits `(batch, sequence, vocabulary)` for a batch.
    fn forward(&self, batch: &TrainingBatch) -> anyhow::Result<Tensor>;

    /// Runs one forward/backward pass, returning the loss value and gradients.
    fn compute_gradients(&self, batch: &TrainingBatch) -> anyhow::Result<(f32, GradStore)> {
        let logits = self.forward(batch)?;
        let positions = batch.decision_positions.to_vec1::<u32>()?;
        let positions: Vec<usize> = positions
            .iter()
            .map(|position| *position as usize)
            .collect();
        let loss = decision_loss(&logits, &positions, &batch.label_token_ids)?;
        let value = loss.to_vec0::<f32>()?;
        let gradients = loss.backward()?;
        Ok((value, gradients))
    }
}

/// Configuration for one training run.
#[derive(Debug, Clone, Copy)]
pub struct TrainingLoopConfiguration {
    pub epochs: usize,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub maximum_gradient_norm: f64,
    pub gradient_accumulation_steps: usize,
    pub minimum_improvement: f32,
    pub early_stop_patience: usize,
    pub minimum_learning_rate_ratio: f64,
}

impl Default for TrainingLoopConfiguration {
    fn default() -> Self {
        Self {
            epochs: 3,
            learning_rate: 1e-3,
            warmup_steps: 0,
            weight_decay: 0.01,
            maximum_gradient_norm: 1.0,
            gradient_accumulation_steps: 1,
            minimum_improvement: 0.0,
            early_stop_patience: 0,
            minimum_learning_rate_ratio: 0.1,
        }
    }
}

/// Metrics reported for one epoch.
#[derive(Debug, Clone, Copy)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub mean_loss: f32,
    pub learning_rate: f64,
}

/// Outcome of a full training run.
#[derive(Debug, Clone)]
pub struct TrainingOutcome {
    pub epochs: Vec<EpochMetrics>,
    pub stopped_early: bool,
}

/// Trains `model` over `batches` for the configured number of epochs.
pub fn train<M: TrainableModel>(
    model: &M,
    batches: &[TrainingBatch],
    configuration: &TrainingLoopConfiguration,
    device: &Device,
) -> anyhow::Result<TrainingOutcome> {
    if batches.is_empty() {
        return Err(anyhow::anyhow!("cannot train on zero batches"));
    }
    let variables = model.variables();
    if variables.is_empty() {
        return Err(anyhow::anyhow!("the model exposes no trainable variables"));
    }
    let optimizer_configuration = OptimizerConfiguration {
        learning_rate: configuration.learning_rate,
        weight_decay: configuration.weight_decay,
        ..OptimizerConfiguration::default()
    };
    let mut optimizer = build_optimizer(variables.clone(), optimizer_configuration)?;

    let steps_per_epoch = batches
        .len()
        .div_ceil(configuration.gradient_accumulation_steps.max(1));
    let total_steps = (steps_per_epoch * configuration.epochs).max(1);
    let schedule = LearningRateSchedule::new(
        configuration.learning_rate,
        configuration.warmup_steps,
        total_steps,
        configuration.minimum_learning_rate_ratio,
    );

    let mut metrics: Vec<EpochMetrics> = Vec::with_capacity(configuration.epochs);
    let mut global_step = 0_usize;
    let mut best_loss = f32::INFINITY;
    let mut epochs_without_improvement = 0_usize;
    let mut stopped_early = false;

    for epoch in 0..configuration.epochs {
        let mut accumulator = GradientAccumulator::new();
        let mut running_loss = 0.0_f32;
        let mut micro_batches_since_step = 0_usize;
        let mut epoch_learning_rate = configuration.learning_rate;

        for batch in batches {
            let (loss, gradients) = model.compute_gradients(batch)?;
            running_loss += loss;
            accumulator.accumulate(&variables, &gradients)?;
            micro_batches_since_step += 1;

            let accumulation_target = configuration.gradient_accumulation_steps.max(1);
            if micro_batches_since_step >= accumulation_target {
                let mut averaged = accumulator.take_averaged(&variables)?;
                clip_gradients(
                    &variables,
                    &mut averaged,
                    configuration.maximum_gradient_norm,
                )?;
                apply_learning_rate(&mut optimizer, &schedule, global_step);
                epoch_learning_rate = optimizer.learning_rate();
                optimizer.step(&averaged)?;
                global_step += 1;
                micro_batches_since_step = 0;
            }
        }
        // Flush any partial accumulation window at the end of the epoch.
        if micro_batches_since_step > 0 {
            let mut averaged = accumulator.take_averaged(&variables)?;
            clip_gradients(
                &variables,
                &mut averaged,
                configuration.maximum_gradient_norm,
            )?;
            apply_learning_rate(&mut optimizer, &schedule, global_step);
            epoch_learning_rate = optimizer.learning_rate();
            optimizer.step(&averaged)?;
            global_step += 1;
        }

        let mean_loss = running_loss / batches.len() as f32;
        tracing::info!(
            epoch,
            mean_loss,
            learning_rate = epoch_learning_rate,
            "epoch completed"
        );
        metrics.push(EpochMetrics {
            epoch,
            mean_loss,
            learning_rate: epoch_learning_rate,
        });

        if mean_loss + configuration.minimum_improvement < best_loss {
            best_loss = mean_loss;
            epochs_without_improvement = 0;
        } else {
            epochs_without_improvement += 1;
            if configuration.early_stop_patience > 0
                && epochs_without_improvement >= configuration.early_stop_patience
            {
                tracing::info!(epoch, "early stopping triggered");
                stopped_early = true;
                break;
            }
        }
    }

    let _ = device;
    Ok(TrainingOutcome {
        epochs: metrics,
        stopped_early,
    })
}

/// Per-epoch metrics keyed by epoch index, for callers that want a map view.
pub fn metrics_by_epoch(outcome: &TrainingOutcome) -> HashMap<usize, EpochMetrics> {
    outcome
        .epochs
        .iter()
        .map(|metrics| (metrics.epoch, *metrics))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    /// A mock model whose single parameter directly scales the decision logits.
    struct MockModel {
        variable: Var,
        target: usize,
        vocabulary: usize,
        device: Device,
    }

    impl TrainableModel for MockModel {
        fn variables(&self) -> Vec<Var> {
            vec![self.variable.clone()]
        }

        fn forward(&self, batch: &TrainingBatch) -> anyhow::Result<Tensor> {
            let (batch_size, sequence_length) = batch.input_ids.dims2()?;
            let scale = self
                .variable
                .as_tensor()
                .flatten_all()?
                .reshape((1, 1, 1))?;
            // A one-hot target mask scaled by the trainable value keeps the
            // variable in the autograd graph (no host round-trip).
            let mut mask = vec![0.0_f32; batch_size * sequence_length * self.vocabulary];
            for row in 0..batch_size {
                let position = batch.decision_positions.to_vec1::<u32>()?[row] as usize;
                let index = (row * sequence_length + position) * self.vocabulary + self.target;
                mask[index] = 3.0;
            }
            let mask = Tensor::from_vec(
                mask,
                (batch_size, sequence_length, self.vocabulary),
                &self.device,
            )?;
            Ok(mask.broadcast_mul(&scale)?)
        }
    }

    fn dummy_batch(device: &Device) -> anyhow::Result<TrainingBatch> {
        let input_ids = Tensor::new(&[[1_u32, 2, 3]], device)?;
        let decision_mask = Tensor::new(&[[0_u32, 0, 1]], device)?;
        let decision_positions = Tensor::new(&[2_u32], device)?;
        let label_token_ids = Tensor::new(&[1_u32], device)?;
        Ok(TrainingBatch {
            input_ids,
            decision_mask,
            decision_positions,
            label_token_ids,
            sequence_length: 3,
        })
    }

    fn mock_model(device: &Device) -> anyhow::Result<MockModel> {
        let variable_map = VarMap::new();
        let builder = candle_nn::VarBuilder::from_varmap(&variable_map, DType::F32, device);
        let variable = builder.get_with_hints(
            (1,),
            "mock.scale",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.5,
            },
        )?;
        Ok(MockModel {
            variable: Var::from_tensor(&variable)?,
            target: 1,
            vocabulary: 4,
            device: device.clone(),
        })
    }

    #[test]
    fn a_dummy_model_overfits_and_reduces_its_loss() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let model = mock_model(&device)?;
        let batches = vec![dummy_batch(&device)?];
        let configuration = TrainingLoopConfiguration {
            epochs: 30,
            learning_rate: 0.5,
            maximum_gradient_norm: 100.0,
            ..TrainingLoopConfiguration::default()
        };
        let outcome = train(&model, &batches, &configuration, &device)?;
        let first = outcome
            .epochs
            .first()
            .ok_or_else(|| anyhow::anyhow!("no metrics recorded"))?;
        let last = outcome
            .epochs
            .last()
            .ok_or_else(|| anyhow::anyhow!("no metrics recorded"))?;
        assert!(
            last.mean_loss < first.mean_loss,
            "loss must fall: first {}, last {}",
            first.mean_loss,
            last.mean_loss
        );
        Ok(())
    }

    #[test]
    fn early_stopping_triggers_on_a_plateau() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let model = mock_model(&device)?;
        let batches = vec![dummy_batch(&device)?];
        let configuration = TrainingLoopConfiguration {
            epochs: 100,
            learning_rate: 0.0,
            maximum_gradient_norm: 1.0,
            early_stop_patience: 3,
            minimum_improvement: 1.0,
            ..TrainingLoopConfiguration::default()
        };
        let outcome = train(&model, &batches, &configuration, &device)?;
        assert!(outcome.stopped_early);
        assert!(outcome.epochs.len() < configuration.epochs);
        Ok(())
    }

    #[test]
    fn zero_batches_is_an_error() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let model = mock_model(&device)?;
        let result = train(&model, &[], &TrainingLoopConfiguration::default(), &device);
        assert!(result.is_err());
        Ok(())
    }
}
