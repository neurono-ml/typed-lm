/// Calibrates two raw logits into probabilities via binary softmax.
/// Numerically stable (subtracts the peak before exp), matching `normalize`.
pub struct Classifier;

impl Classifier {
    pub fn binary_softmax(true_logit: f32, false_logit: f32) -> Classification {
        let peak = true_logit.max(false_logit);
        let exponent_true = (true_logit - peak).exp();
        let exponent_false = (false_logit - peak).exp();
        let sum = exponent_true + exponent_false;
        Classification {
            true_prob: exponent_true / sum,
            false_prob: exponent_false / sum,
        }
    }

    /// Normalizes raw logits into probabilities via temperature-scaled softmax.
    /// Numerically stable (subtracts the peak before exp). Mirrors openjev scoring.
    pub fn normalize(logits: &[f32], temperature: f32) -> Vec<f32> {
        let peak = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = logits
            .iter()
            .map(|v| ((v - peak) / temperature).exp())
            .collect();
        let total: f32 = weights.iter().sum();
        weights.iter().map(|w| w / total).collect()
    }

    /// Distribution concentration: 1 minus normalized entropy. In [0, 1].
    pub fn confidence(probs: &[f32]) -> f32 {
        let entropy: f32 = probs
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| -p * p.ln())
            .sum();
        let n = probs.len() as f32;
        (1.0 - entropy / n.ln()).clamp(0.0, 1.0)
    }

    pub fn expected_score(probs: &[f32]) -> f32 {
        probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum()
    }
}

/// Accessory struct: probability pair produced by the classification.
pub struct Classification {
    pub true_prob: f32,
    pub false_prob: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn equal_logits_yield_half_each() {
        let r = Classifier::binary_softmax(1.0, 1.0);
        assert!((r.true_prob - 0.5).abs() < EPS);
        assert!((r.false_prob - 0.5).abs() < EPS);
    }

    #[test]
    fn probabilities_sum_to_one() {
        for (lt, lf) in [(2.0, 0.5), (-1.0, 3.0), (0.0, 0.0), (10.0, -10.0)] {
            let r = Classifier::binary_softmax(lt, lf);
            assert!((r.true_prob + r.false_prob - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn larger_logit_gets_larger_probability() {
        let r = Classifier::binary_softmax(2.0, 0.0);
        assert!(r.true_prob > r.false_prob);
        // softmax(2,0) = e^2/(e^2+1) ~= 0.8808
        assert!((r.true_prob - 0.8808).abs() < 1e-4);
    }

    #[test]
    fn sides_are_symmetric_when_swapped() {
        let a = Classifier::binary_softmax(1.5, -0.5);
        let b = Classifier::binary_softmax(-0.5, 1.5);
        assert!((a.true_prob - b.false_prob).abs() < EPS);
        assert!((a.false_prob - b.true_prob).abs() < EPS);
    }

    #[test]
    fn probabilities_stay_in_valid_range() {
        let r = Classifier::binary_softmax(5.0, -3.0);
        assert!((0.0..=1.0).contains(&r.true_prob));
        assert!((0.0..=1.0).contains(&r.false_prob));
        assert!(r.true_prob > 0.999);
    }
}
