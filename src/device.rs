use candle_core::Device;

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
