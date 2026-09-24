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
}
