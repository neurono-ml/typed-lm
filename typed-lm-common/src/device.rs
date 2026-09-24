use candle_core::{DType, Device};

/// Resolves the physical execution device (CUDA > Metal > CPU).
pub struct DeviceResolver;

impl DeviceResolver {
    pub fn resolve() -> anyhow::Result<Device> {
        if candle_core::utils::cuda_is_available() {
            Ok(Device::new_cuda(0)?)
        } else if candle_core::utils::metal_is_available() {
            Ok(Device::new_metal(0)?)
        } else {
            Ok(Device::Cpu)
        }
    }
}

/// Weight numeric type requested for the loaded model.
///
/// `auto` keeps `F32` on the CPU (where candle computes in `F32` anyway and a
/// narrower load would only add conversions) and selects `F16` on accelerators
/// (CUDA/Metal), roughly halving memory bandwidth during the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ModelDtype {
    Auto,
    F32,
    F16,
    Bf16,
}

impl ModelDtype {
    /// Maps the requested weight type to a concrete candle `DType` for the
    /// resolved execution device.
    pub fn resolve(self, device: &Device) -> DType {
        match self {
            ModelDtype::F32 => DType::F32,
            ModelDtype::F16 => DType::F16,
            ModelDtype::Bf16 => DType::BF16,
            ModelDtype::Auto => {
                if device.is_cpu() {
                    DType::F32
                } else {
                    DType::F16
                }
            }
        }
    }
}

/// Numeric precision policy used by the trainer.
///
/// Functional parity between CPU and CUDA is achieved by keeping the master
/// weights (and the optimizer state) in F32 on every device, using a narrower
/// precision only for the compute-heavy matmuls, and always reducing the loss
/// in F32. That keeps the trained adapter numerically equivalent across devices
/// without requiring the same throughput.
///
/// - `master` — dtype of the trainable parameters and optimizer state.
/// - `compute` — dtype of the activations fed to matmul/attention kernels.
/// - `reduction` — dtype of loss/softmax reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionPolicy {
    pub master: DType,
    pub compute: DType,
    pub reduction: DType,
}

impl PrecisionPolicy {
    /// Resolves the policy for a device, honoring an explicit compute override.
    ///
    /// With no override the compute dtype is F32 on the CPU and BF16 on any
    /// accelerator. Master and reduction stay F32 in every case so training
    /// quality does not depend on the device.
    pub fn resolve(device: &Device, compute_override: Option<DType>) -> Self {
        let compute = match compute_override {
            Some(explicit) => explicit,
            None if device.is_cpu() => DType::F32,
            None => DType::BF16,
        };
        Self {
            master: DType::F32,
            compute,
            reduction: DType::F32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_dtypes_are_returned_unchanged() {
        assert_eq!(ModelDtype::F32.resolve(&Device::Cpu), DType::F32);
        assert_eq!(ModelDtype::F16.resolve(&Device::Cpu), DType::F16);
        assert_eq!(ModelDtype::Bf16.resolve(&Device::Cpu), DType::BF16);
    }

    #[test]
    fn automatic_dtype_uses_f32_on_cpu() {
        assert_eq!(ModelDtype::Auto.resolve(&Device::Cpu), DType::F32);
    }

    #[test]
    fn precision_policy_keeps_master_and_reduction_in_f32_on_cpu() {
        let policy = PrecisionPolicy::resolve(&Device::Cpu, None);
        assert_eq!(policy.master, DType::F32);
        assert_eq!(policy.compute, DType::F32);
        assert_eq!(policy.reduction, DType::F32);
    }

    #[test]
    fn precision_policy_honors_an_explicit_compute_override() {
        let policy = PrecisionPolicy::resolve(&Device::Cpu, Some(DType::BF16));
        assert_eq!(policy.master, DType::F32);
        assert_eq!(policy.compute, DType::BF16);
        assert_eq!(policy.reduction, DType::F32);
    }
}
