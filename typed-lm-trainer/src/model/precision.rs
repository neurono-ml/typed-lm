//! Cast helpers that follow a [`PrecisionPolicy`].
//!
//! Training keeps the *master* weights and the optimizer state in F32 on every
//! device, feeds the matmuls a narrower *compute* dtype (BF16 on accelerators,
//! F32 on the CPU) and always reduces the loss in F32. These wrappers make the
//! policy explicit at every cast site instead of scattering raw `to_dtype`
//! calls, so a lower-precision forward can never silently drift the master
//! parameters or the loss.
//!
//! The master dtype is validated to be F32: a policy whose master is not F32 is
//! rejected with a typed error, because the trainer relies on F32 accumulation
//! for numerical parity between CPU and CUDA.

use candle_core::{DType, Device, Tensor};
use typed_lm_common::device::PrecisionPolicy;

use crate::error::TrainerError;

/// Cast helpers bound to a resolved precision policy.
#[derive(Debug, Clone, Copy)]
pub struct PrecisionCasts {
    policy: PrecisionPolicy,
}

impl PrecisionCasts {
    /// Validates the policy (master weights must be F32) and returns the casts.
    pub fn new(policy: PrecisionPolicy) -> Result<Self, TrainerError> {
        if policy.master != DType::F32 {
            return Err(TrainerError::Precision(format!(
                "the master weight dtype must be F32, got {:?}",
                policy.master
            )));
        }
        Ok(Self { policy })
    }

    /// Resolves the policy for a device (honoring an explicit compute override).
    pub fn resolve(device: &Device, compute_override: Option<DType>) -> Result<Self, TrainerError> {
        Self::new(PrecisionPolicy::resolve(device, compute_override))
    }

    /// The underlying policy.
    pub fn policy(&self) -> PrecisionPolicy {
        self.policy
    }

    /// Casts a tensor into the master (trainable) dtype.
    pub fn to_master(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        tensor.to_dtype(self.policy.master)
    }

    /// Casts a tensor into the compute dtype used by matmuls.
    pub fn to_compute(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        tensor.to_dtype(self.policy.compute)
    }

    /// Casts a tensor into the reduction dtype used by losses.
    pub fn to_reduction(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        tensor.to_dtype(self.policy.reduction)
    }

    /// Casts a tensor from compute/reduction precision back to master precision.
    pub fn back_to_master(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        self.to_master(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_non_f32_master_is_rejected() {
        let policy = PrecisionPolicy {
            master: DType::BF16,
            compute: DType::BF16,
            reduction: DType::F32,
        };
        assert!(PrecisionCasts::new(policy).is_err());
    }

    #[test]
    fn cpu_resolves_to_f32_compute_with_f32_master() -> anyhow::Result<()> {
        let casts = PrecisionCasts::resolve(&Device::Cpu, None)?;
        assert_eq!(casts.policy().master, DType::F32);
        assert_eq!(casts.policy().compute, DType::F32);
        assert_eq!(casts.policy().reduction, DType::F32);
        Ok(())
    }

    #[test]
    fn explicit_compute_override_is_honored() -> anyhow::Result<()> {
        let casts = PrecisionCasts::resolve(&Device::Cpu, Some(DType::BF16))?;
        assert_eq!(casts.policy().compute, DType::BF16);
        assert_eq!(casts.policy().master, DType::F32);
        Ok(())
    }

    #[test]
    fn casting_to_master_round_trips_a_bf16_compute_tensor() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let casts = PrecisionCasts::resolve(&device, Some(DType::BF16))?;
        let original = Tensor::new(&[1.0_f32, 2.0, 3.0], &device)?;
        let compute = casts.to_compute(&original)?;
        assert_eq!(compute.dtype(), DType::BF16);
        let restored = casts.back_to_master(&compute)?;
        assert_eq!(restored.dtype(), DType::F32);
        let original_values = original.to_vec1::<f32>()?;
        let restored_values = restored.to_vec1::<f32>()?;
        for (left, right) in original_values.iter().zip(restored_values.iter()) {
            // BF16 keeps 8 mantissa bits; the round-trip is exact for these
            // small integers.
            assert!((left - right).abs() < 1e-3);
        }
        Ok(())
    }

    #[test]
    fn casting_to_reduction_preserves_values() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let casts = PrecisionCasts::resolve(&device, None)?;
        let tensor = Tensor::new(&[0.25_f32, 0.5], &device)?;
        let reduced = casts.to_reduction(&tensor)?;
        assert_eq!(reduced.dtype(), DType::F32);
        assert_eq!(reduced.to_vec1::<f32>()?, vec![0.25, 0.5]);
        Ok(())
    }
}
