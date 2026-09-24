//! Training losses.
//!
//! The trainer mirrors serving: the decision-position logits (restricted to the
//! question's candidate labels) are turned into a distribution and compared to
//! the ground-truth label. Three pieces compose:
//!
//! - [`decision_loss`] — cross-entropy over the full vocabulary at the decision
//!   position of every row, exactly the server's single-forward decision.
//! - [`restricted_decision_loss`] — cross-entropy restricted to the candidate
//!   label ids, matching [`typed_lm_common::labels::calibrate_probabilities`].
//! - [`calibration_divergence`] — an optional KL term that keeps the trained
//!   distribution close to a reference probability distribution.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use typed_lm_common::labels::calibrate_probabilities;

/// Cross-entropy of the full-vocabulary logits at each decision position.
///
/// `logits` is `(batch, sequence, vocabulary)` and `decision_positions` holds one
/// position per row; `label_token_ids` holds the target vocabulary id per row.
pub fn decision_loss(
    logits: &Tensor,
    decision_positions: &[usize],
    label_token_ids: &Tensor,
) -> Result<Tensor> {
    let logits = logits.to_dtype(DType::F32)?;
    let mut rows = Vec::with_capacity(decision_positions.len());
    for (row, position) in decision_positions.iter().enumerate() {
        rows.push(logits.i((row, *position, ..))?);
    }
    let decision_logits = Tensor::stack(&rows, 0)?;
    let target = label_token_ids.to_dtype(DType::U32)?;
    candle_nn::loss::cross_entropy(&decision_logits, &target)
}

/// Cross-entropy restricted to the candidate labels of each row.
///
/// Answers are scored exactly the way the server calibrates them: only the
/// candidate label ids participate in the softmax, so the loss cannot be reduced
/// by inflating unrelated vocabulary logits. `candidate_label_ids` holds, per
/// row, the vocabulary ids of its answer labels; the target is the index of the
/// correct candidate in that row's list.
pub fn restricted_decision_loss(
    logits: &Tensor,
    decision_positions: &[usize],
    candidate_label_ids: &[Vec<u32>],
    target_indices: &[u32],
) -> anyhow::Result<Tensor> {
    if decision_positions.len() != candidate_label_ids.len()
        || decision_positions.len() != target_indices.len()
    {
        return Err(anyhow::anyhow!(
            "restricted loss inputs disagree on the batch size"
        ));
    }
    let logits = logits.to_dtype(DType::F32)?;
    let mut row_losses: Vec<f32> = Vec::with_capacity(decision_positions.len());
    for (row, ((position, candidates), target_index)) in decision_positions
        .iter()
        .zip(candidate_label_ids.iter())
        .zip(target_indices.iter())
        .enumerate()
    {
        if candidates.is_empty() {
            return Err(anyhow::anyhow!("row {row} has no candidate labels"));
        }
        let target_position = *target_index as usize;
        if target_position >= candidates.len() {
            return Err(anyhow::anyhow!(
                "row {row} target index {target_position} is outside its {} candidates",
                candidates.len()
            ));
        }
        let full_row = logits.i((row, *position, ..))?;
        let mut candidate_logits = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            candidate_logits.push(full_row.i(*candidate as usize)?.to_scalar::<f32>()?);
        }
        let probabilities = calibrate_probabilities(&candidate_logits);
        let probability = probabilities[target_position].max(1e-12);
        row_losses.push(-probability.ln());
    }
    let device = logits.device();
    Ok(Tensor::from_vec(row_losses, (decision_positions.len(),), device)?.mean_all()?)
}

/// KL divergence `KL(reference || candidate)` between two distributions.
///
/// Both inputs are logit or probability vectors; they are normalized with
/// [`calibrate_probabilities`] first. The result is zero when the two
/// distributions agree, which is how the calibration term stays neutral until
/// the trained distribution drifts.
pub fn calibration_divergence(reference: &[f32], candidate: &[f32]) -> anyhow::Result<f32> {
    if reference.len() != candidate.len() || reference.is_empty() {
        return Err(anyhow::anyhow!(
            "calibration distributions must be non-empty and equally sized"
        ));
    }
    let reference = calibrate_probabilities(reference);
    let candidate = calibrate_probabilities(candidate);
    let mut divergence = 0.0_f32;
    for (reference_probability, candidate_probability) in reference.iter().zip(candidate.iter()) {
        let reference_probability = reference_probability.max(1e-12);
        let candidate_probability = candidate_probability.max(1e-12);
        divergence += reference_probability * (reference_probability / candidate_probability).ln();
    }
    Ok(divergence)
}

/// Combines a decision loss with an optional KL calibration term.
pub fn combined_loss(
    decision: &Tensor,
    calibration: Option<f32>,
    calibration_weight: f32,
) -> Result<Tensor> {
    match calibration {
        Some(divergence) if calibration_weight != 0.0 => {
            let penalty = Tensor::full(divergence * calibration_weight, (), decision.device())?;
            &penalty + decision
        }
        _ => Ok(decision.clone()),
    }
}

/// A device helper kept for callers constructing tensors from scalars.
pub fn scalar(value: f32, device: &Device) -> Result<Tensor> {
    Tensor::full(value, (), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_loss_is_lower_for_the_correct_label() -> anyhow::Result<()> {
        let device = Device::Cpu;
        // One row, three positions, vocabulary of 4; the decision is position 2.
        let logits = Tensor::new(
            &[[
                [0.0_f32, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 3.0, 0.2, 0.3],
            ]],
            &device,
        )?;
        let target = Tensor::new(&[1_u32], &device)?;
        let loss = decision_loss(&logits, &[2], &target)?;
        // The correct label has the largest logit, so the loss is small.
        assert!(loss.to_vec0::<f32>()? < 0.2);
        Ok(())
    }

    #[test]
    fn restricted_loss_prefers_the_correct_candidate() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(
            &[[
                [0.0_f32, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 3.0, 0.2, 0.3],
            ]],
            &device,
        )?;
        let candidates = vec![vec![0_u32, 1]];
        let loss = restricted_decision_loss(&logits, &[2], &candidates, &[1])?;
        assert!(loss.to_vec0::<f32>()? < 0.2);
        Ok(())
    }

    #[test]
    fn restricted_loss_rejects_an_out_of_range_target() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[[0.0_f32, 1.0], [0.0, 1.0]]], &device)?;
        let result = restricted_decision_loss(&logits, &[1], &[vec![0_u32, 1]], &[5]);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn kl_is_zero_for_identical_distributions() -> anyhow::Result<()> {
        let divergence = calibration_divergence(&[2.0, 1.0, 0.5], &[2.0, 1.0, 0.5])?;
        assert!(divergence.abs() < 1e-6, "KL must vanish for equal inputs");
        Ok(())
    }

    #[test]
    fn kl_is_positive_for_different_distributions() -> anyhow::Result<()> {
        let divergence = calibration_divergence(&[3.0, 0.0], &[0.0, 3.0])?;
        assert!(divergence > 0.0);
        Ok(())
    }

    #[test]
    fn combined_loss_without_calibration_returns_the_decision_loss() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let decision = Tensor::new(2.0_f32, &device)?;
        let combined = combined_loss(&decision, None, 0.5)?;
        assert!((combined.to_vec0::<f32>()? - 2.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn combined_loss_adds_the_weighted_calibration_term() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let decision = Tensor::new(2.0_f32, &device)?;
        let combined = combined_loss(&decision, Some(1.0), 0.5)?;
        assert!((combined.to_vec0::<f32>()? - 2.5).abs() < 1e-6);
        Ok(())
    }
}
