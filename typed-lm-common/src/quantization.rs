//! Post-training quantization schemes shared by serving and training.
//!
//! Two low-precision float formats are supported:
//!
//! - **FP8 (E4M3)** — one byte per weight, scaled per output channel. Candle 0.11
//!   stores and casts `F8_E4M3` natively, so quantization and dequantization use
//!   the tensor cast path.
//! - **FP4 (MXFP4)** — a 4-bit E2M1 nibble per weight plus one shared
//!   power-of-two exponent (`F8_E8M0`) per block of 32 weights. Candle stores
//!   `F4`/`F8_E8M0` but has no arithmetic for them, so the packing and unpacking
//!   are implemented here as pure bit operations and the values are widened to
//!   F32 for any compute.
//!
//! Both formats dequantize to a dense F32 tensor, which is what the loader and
//! the trainer consume — no FP8/FP4 matmul kernel is required.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Number of weights sharing one MXFP4 exponent.
pub const MXFP4_BLOCK_SIZE: usize = 32;

/// Selectable quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationScheme {
    /// No quantization; weights stay dense.
    None,
    /// FP8 E4M3 with per-output-channel scaling.
    Fp8,
    /// MXFP4 with one F8E8M0 exponent per block of 32 weights.
    Fp4,
}

impl QuantizationScheme {
    /// Parses the `--quantization` flag value.
    pub fn from_flag(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "none" => Some(Self::None),
            "fp8" => Some(Self::Fp8),
            "fp4" => Some(Self::Fp4),
            _ => None,
        }
    }

    /// Canonical flag name.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Fp8 => "fp8",
            Self::Fp4 => "fp4",
        }
    }

    /// Whether this scheme actually shrinks the weights.
    pub fn is_quantized(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Persistent description of how a tensor was quantized.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub scheme: QuantizationScheme,
    /// Block size for block-wise schemes (MXFP4 uses 32).
    #[serde(default)]
    pub block_size: usize,
}

impl QuantizationConfig {
    pub fn new(scheme: QuantizationScheme) -> Self {
        Self {
            scheme,
            block_size: MXFP4_BLOCK_SIZE,
        }
    }

    /// Serializes the config to the JSON written next to a quantized artifact.
    pub fn to_json(&self) -> anyhow::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| anyhow::anyhow!("failed to serialize quantization config: {error}"))
    }

    /// Parses the JSON written next to a quantized artifact.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        serde_json::from_str(json)
            .map_err(|error| anyhow::anyhow!("failed to parse quantization config: {error}"))
    }
}

/// Quantizes a dense F32 tensor to FP8 E4M3 with per-row scaling.
///
/// Each row is divided by its absolute maximum before the cast, so the row maps
/// into the representable E4M3 range. Returns the quantized tensor and the
/// per-row scales needed to dequantize it. A rank-1 tensor (a normalization
/// weight, for example) has no rows and shares one scale instead.
pub fn quantize_fp8_per_channel(weights: &Tensor) -> CandleResult<(Tensor, Tensor)> {
    let weights = weights.to_dtype(DType::F32)?;
    if weights.rank() < 2 {
        let absolute_maximum = weights.abs()?.max_all()?;
        let floor = Tensor::full(1e-12_f32, (), weights.device())?;
        let scale = absolute_maximum.maximum(&floor)?;
        let normalized = weights.broadcast_div(&scale)?;
        let quantized = normalized.to_dtype(DType::F8E4M3)?;
        return Ok((quantized, scale.reshape((1,))?.to_dtype(DType::F32)?));
    }
    let rows = weights.dims()[0];
    let absolute_maximum = weights.abs()?.max_keepdim(1)?;
    let floor = Tensor::full(1e-12_f32, absolute_maximum.dims(), weights.device())?;
    let scales = absolute_maximum.maximum(&floor)?;
    let normalized = weights.broadcast_div(&scales)?;
    let quantized = normalized.to_dtype(DType::F8E4M3)?;
    let scales = scales.reshape(rows)?;
    Ok((quantized, scales.to_dtype(DType::F32)?))
}

/// Dequantizes an FP8 E4M3 tensor back to dense F32.
///
/// `scales` holds one value per row (or a single value for a rank-1 tensor);
/// the result has the original shape.
pub fn dequantize_fp8_per_channel(quantized: &Tensor, scales: &Tensor) -> CandleResult<Tensor> {
    let dense = quantized.to_dtype(DType::F32)?;
    if dense.rank() < 2 {
        let scale = scales.reshape(())?;
        return dense.broadcast_mul(&scale);
    }
    let rows = dense.dims()[0];
    let broadcast_scales = scales.reshape((rows, 1))?;
    dense.broadcast_mul(&broadcast_scales)
}

/// Max representable magnitude of an MXFP4 E2M1 nibble (values are ±6).
const MXFP4_MAX: f32 = 6.0;

/// Quantizes a dense F32 tensor to MXFP4 (packed E2M1 + F8E8M0 block exponent).
///
/// Weights are blocked along the last dimension in groups of
/// [`MXFP4_BLOCK_SIZE`]; each block shares a power-of-two exponent chosen so the
/// largest magnitude maps close to [`MXFP4_MAX`]. The packed nibbles are
/// returned as a `U8` tensor with two nibbles per byte and the exponents as a
/// `U8` tensor of E8M0-encoded exponents.
pub fn quantize_fp4_mxfp4(weights: &Tensor) -> CandleResult<(Tensor, Tensor)> {
    let weights = weights.to_dtype(DType::F32)?;
    let flat = weights.flatten_all()?.to_vec1::<f32>()?;
    let block_count = flat.len().div_ceil(MXFP4_BLOCK_SIZE);

    let mut packed = vec![0_u8; flat.len().div_ceil(2)];
    let mut exponents = vec![0_u8; block_count];

    for (block_index, exponent_slot) in exponents.iter_mut().enumerate() {
        let start = block_index * MXFP4_BLOCK_SIZE;
        let end = (start + MXFP4_BLOCK_SIZE).min(flat.len());
        let block = &flat[start..end];
        let block_maximum = block
            .iter()
            .fold(0.0_f32, |accumulator, value| accumulator.max(value.abs()));
        // Power-of-two scale: exponent = ceil(log2(block_max / MXFP4_MAX)).
        let raw_exponent = if block_maximum > 0.0 {
            (block_maximum / MXFP4_MAX).log2().ceil()
        } else {
            0.0
        };
        let exponent_byte = encode_e8m0_exponent(raw_exponent);
        *exponent_slot = exponent_byte;
        let block_scale = pow2(raw_exponent);
        for (offset, value) in block.iter().enumerate() {
            let normalized = if block_scale > 0.0 {
                value / block_scale
            } else {
                0.0
            };
            let nibble = encode_e2m1(normalized);
            let absolute_index = start + offset;
            let byte_index = absolute_index / 2;
            if absolute_index.is_multiple_of(2) {
                packed[byte_index] = (packed[byte_index] & 0xF0) | nibble;
            } else {
                packed[byte_index] = (packed[byte_index] & 0x0F) | (nibble << 4);
            }
        }
    }

    let device = weights.device();
    let packed_tensor = Tensor::from_vec(packed, (flat.len().div_ceil(2),), device)?;
    let exponent_tensor = Tensor::from_vec(exponents, (block_count,), device)?;
    Ok((packed_tensor, exponent_tensor))
}

/// Dequantizes an MXFP4 tensor back to dense F32.
///
/// `original_shape` is the shape of the weight before quantization; the result
/// has exactly that shape and element count. The shape is required because the
/// packed nibbles are stored flat and, with an odd element count, the last byte
/// carries one padding nibble.
pub fn dequantize_fp4_mxfp4(
    packed: &Tensor,
    exponents: &Tensor,
    original_shape: &[usize],
) -> CandleResult<Tensor> {
    let element_count: usize = original_shape.iter().product();
    let packed_bytes = packed.flatten_all()?.to_vec1::<u8>()?;
    let exponent_bytes = exponents.flatten_all()?.to_vec1::<u8>()?;
    let mut output = Vec::with_capacity(element_count);
    for element_index in 0..element_count {
        let byte_index = element_index / 2;
        let byte = packed_bytes.get(byte_index).copied().unwrap_or(0);
        let nibble = if element_index % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        let block_index = element_index / MXFP4_BLOCK_SIZE;
        let exponent_byte = exponent_bytes.get(block_index).copied().unwrap_or(0);
        let block_scale = decode_e8m0_exponent(exponent_byte);
        output.push(decode_e2m1(nibble) * block_scale);
    }
    Tensor::from_vec(output, original_shape, packed.device())
}

/// 2 raised to `exponent` (exact for the FP4 block scale by construction).
fn pow2(exponent: f32) -> f32 {
    2.0_f32.powf(exponent)
}

/// Encodes a power-of-two exponent into an E8M0 byte (biased by 127).
fn encode_e8m0_exponent(exponent: f32) -> u8 {
    let biased = (exponent + 127.0).round();
    biased.clamp(0.0, 255.0) as u8
}

/// Decodes an E8M0 byte back into its power-of-two scale.
fn decode_e8m0_exponent(byte: u8) -> f32 {
    // 0xFF is the E8M0 NaN sentinel; treat it as a zero scale.
    if byte == 0xFF {
        return 0.0;
    }
    pow2(byte as f32 - 127.0)
}

/// E2M1 codebook (sign, exponent, mantissa) for the 16 representable values.
const E2M1_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Encodes a normalized value into the nearest E2M1 nibble (round to nearest).
fn encode_e2m1(value: f32) -> u8 {
    let mut best_index = 0_usize;
    let mut best_distance = f32::INFINITY;
    for (index, candidate) in E2M1_VALUES.iter().enumerate() {
        let distance = (value - candidate).abs();
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }
    best_index as u8
}

/// Decodes an E2M1 nibble back into its value.
fn decode_e2m1(nibble: u8) -> f32 {
    E2M1_VALUES[(nibble & 0x0F) as usize]
}

/// Dequantizes a tensor according to a stored [`QuantizationConfig`].
///
/// For FP8 the input is the E4M3 tensor and `auxiliary` its per-row scales;
/// for FP4 the input is the packed nibbles and `auxiliary` the block exponents.
/// `original_shape` is the weight shape before quantization (FP4 needs it to
/// restore the exact shape and, with an odd count, drop the padding nibble).
pub fn dequantize(
    config: &QuantizationConfig,
    quantized: &Tensor,
    auxiliary: &Tensor,
    original_shape: &[usize],
) -> anyhow::Result<Tensor> {
    match config.scheme {
        QuantizationScheme::None => Ok(quantized.clone()),
        QuantizationScheme::Fp8 => dequantize_fp8_per_channel(quantized, auxiliary)
            .map_err(|error| anyhow::anyhow!("failed to dequantize FP8 weights: {error}")),
        QuantizationScheme::Fp4 => dequantize_fp4_mxfp4(quantized, auxiliary, original_shape)
            .map_err(|error| anyhow::anyhow!("failed to dequantize FP4 weights: {error}")),
    }
}

/// Quantizes a dense tensor according to a scheme, returning weights + auxiliary.
pub fn quantize(config: &QuantizationConfig, weights: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
    match config.scheme {
        QuantizationScheme::None => {
            Ok((weights.clone(), Tensor::full(0_f32, (), weights.device())?))
        }
        QuantizationScheme::Fp8 => quantize_fp8_per_channel(weights)
            .map_err(|error| anyhow::anyhow!("failed to quantize FP8 weights: {error}")),
        QuantizationScheme::Fp4 => quantize_fp4_mxfp4(weights)
            .map_err(|error| anyhow::anyhow!("failed to quantize FP4 weights: {error}")),
    }
}

/// Casts a dense tensor to the compute dtype named by a device policy.
///
/// Small helper used by training loops that keep master weights in F32 but run
/// matmuls in the policy's compute dtype.
pub fn cast_for_compute(weights: &Tensor, compute: DType) -> CandleResult<Tensor> {
    weights.to_dtype(compute)
}

/// Convenience helper: builds a dense tensor from a slice on a device.
pub fn dense_from_slice(values: &[f32], device: &Device) -> CandleResult<Tensor> {
    Tensor::from_slice(values, (values.len(),), device)
}

/// Name of the scale tensor paired with a quantized weight tensor.
///
/// Follows the `compressed-tensors` naming used by FP8/MXFP4 checkpoints:
/// `model.layers.0.mlp.down_proj.weight` pairs with
/// `model.layers.0.mlp.down_proj.weight_scale`.
pub fn scale_tensor_name(weight_name: &str) -> String {
    format!("{weight_name}_scale")
}

/// Name of the tensor recording a weight's original (pre-quantization) shape.
///
/// FP4 stores the packed nibbles flat, so the original shape cannot be
/// recovered from the stored tensor; the export writes it as a `U32` tensor
/// `model.layers.0.proj.weight_shape` and the loader reads it back.
pub fn shape_tensor_name(weight_name: &str) -> String {
    format!("{weight_name}_shape")
}

/// Whether a stored tensor name is quantization metadata, not a weight.
///
/// Metadata tensors (`*_scale`, `*_shape`) pair with a weight and are consumed
/// during dequantization; they must never leak into the model weights.
pub fn is_quantization_metadata(name: &str) -> bool {
    name.ends_with("_scale") || name.ends_with("_shape")
}

/// Dequantizes every weight of a checkpoint tensor map to dense F32.
///
/// The scheme is homogeneous across the checkpoint (that is what the trainer
/// emits), so a single [`QuantizationScheme`] drives the whole map.
///
/// - `Fp8` — each weight is scaled by its paired `*_scale` tensor (per output
///   channel). A missing scale leaves the raw E4M3 value widened to F32.
/// - `Fp4` — each weight holds packed E2M1 nibbles (two per byte) and its
///   paired `*_scale` tensor holds the E8M0 block exponents; a missing scale is
///   an error, because the block scale cannot be reconstructed. The paired
///   `*_shape` tensor (when present) restores the original weight shape.
/// - `None` — the map is returned unchanged.
///
/// Paired metadata tensors are consumed by the dequantization, so they never
/// leak into the model weights handed to the variable builder.
pub fn dequantize_checkpoint_tensors(
    mut tensors: HashMap<String, Tensor>,
    scheme: QuantizationScheme,
) -> anyhow::Result<HashMap<String, Tensor>> {
    if scheme == QuantizationScheme::None {
        return Ok(tensors);
    }
    let weight_names: Vec<String> = tensors
        .keys()
        .filter(|name| !is_quantization_metadata(name))
        .cloned()
        .collect();
    let mut output = HashMap::with_capacity(weight_names.len());
    for weight_name in weight_names {
        let Some(weight) = tensors.remove(&weight_name) else {
            continue;
        };
        let scale = tensors.remove(&scale_tensor_name(&weight_name));
        let original_shape = tensors
            .remove(&shape_tensor_name(&weight_name))
            .map(|shape| shape.to_vec1::<u32>())
            .transpose()?
            .map(|dimensions| {
                dimensions
                    .iter()
                    .map(|dimension| *dimension as usize)
                    .collect::<Vec<usize>>()
            });
        let dequantized = match scheme {
            QuantizationScheme::Fp8 => match scale {
                Some(scale) => dequantize_fp8_per_channel(&weight, &scale)?,
                None => weight.to_dtype(DType::F32)?,
            },
            QuantizationScheme::Fp4 => {
                let scale = scale.ok_or_else(|| {
                    anyhow::anyhow!(
                        "FP4 weight '{weight_name}' is missing its '{}' exponent tensor",
                        scale_tensor_name(&weight_name)
                    )
                })?;
                let shape = original_shape.unwrap_or_else(|| vec![weight.elem_count() * 2]);
                dequantize_fp4_mxfp4(&weight, &scale, &shape)?
            }
            QuantizationScheme::None => weight,
        };
        output.insert(weight_name, dequantized.to_dtype(DType::F32)?);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheme_flag_round_trips() {
        assert_eq!(
            QuantizationScheme::from_flag("none"),
            Some(QuantizationScheme::None)
        );
        assert_eq!(
            QuantizationScheme::from_flag("FP8"),
            Some(QuantizationScheme::Fp8)
        );
        assert_eq!(
            QuantizationScheme::from_flag("fp4"),
            Some(QuantizationScheme::Fp4)
        );
        assert_eq!(QuantizationScheme::from_flag("int8"), None);
        assert_eq!(QuantizationScheme::Fp8.name(), "fp8");
        assert!(QuantizationScheme::Fp4.is_quantized());
        assert!(!QuantizationScheme::None.is_quantized());
    }

    #[test]
    fn config_json_round_trips() -> anyhow::Result<()> {
        let config = QuantizationConfig::new(QuantizationScheme::Fp4);
        let json = config.to_json()?;
        let parsed = QuantizationConfig::from_json(&json)?;
        assert_eq!(parsed.scheme, QuantizationScheme::Fp4);
        assert_eq!(parsed.block_size, MXFP4_BLOCK_SIZE);
        Ok(())
    }

    #[test]
    fn malformed_config_json_is_an_error() {
        assert!(QuantizationConfig::from_json("{ not json").is_err());
    }

    #[test]
    fn fp8_round_trip_is_close_to_the_original() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let weights = Tensor::new(vec![vec![1.0_f32, -2.0, 0.5, 3.0]], &device)?;
        let (quantized, scales) = quantize_fp8_per_channel(&weights)?;
        assert_eq!(quantized.dtype(), DType::F8E4M3);
        let restored = dequantize_fp8_per_channel(&quantized, &scales)?;
        assert_eq!(restored.dims(), weights.dims());
        let original = weights.flatten_all()?.to_vec1::<f32>()?;
        let round_tripped = restored.flatten_all()?.to_vec1::<f32>()?;
        for (original_value, restored_value) in original.iter().zip(round_tripped.iter()) {
            // E4M3 keeps only three mantissa bits: ~6.25% relative precision.
            let tolerance = original_value.abs().max(1.0) * 0.07;
            assert!(
                (original_value - restored_value).abs() <= tolerance,
                "FP8 round-trip drifted: {original_value} vs {restored_value}"
            );
        }
        Ok(())
    }

    #[test]
    fn fp8_round_trip_handles_rank_one_weights() -> anyhow::Result<()> {
        let device = Device::Cpu;
        // A normalization weight is rank-1 and must use a single shared scale.
        let weights = Tensor::from_vec(vec![1.0_f32, -2.0, 0.5, 3.0], (4,), &device)?;
        let (quantized, scales) = quantize_fp8_per_channel(&weights)?;
        assert_eq!(scales.dims(), &[1]);
        let restored = dequantize_fp8_per_channel(&quantized, &scales)?;
        assert_eq!(restored.dims(), &[4]);
        let original = weights.to_vec1::<f32>()?;
        let round_tripped = restored.to_vec1::<f32>()?;
        for (original_value, restored_value) in original.iter().zip(round_tripped.iter()) {
            let tolerance = original_value.abs().max(1.0) * 0.07;
            assert!(
                (original_value - restored_value).abs() <= tolerance,
                "rank-1 FP8 round-trip drifted: {original_value} vs {restored_value}"
            );
        }
        Ok(())
    }

    #[test]
    fn fp4_round_trip_stays_within_block_tolerance() -> anyhow::Result<()> {
        let device = Device::Cpu;
        // A 64-element block spans two MXFP4 blocks of 32.
        let values: Vec<f32> = (0..64).map(|index| (index as f32 - 32.0) / 8.0).collect();
        let weights = Tensor::from_vec(values.clone(), (64,), &device)?;
        let (packed, exponents) = quantize_fp4_mxfp4(&weights)?;
        assert_eq!(exponents.dims(), &[2]);
        assert_eq!(packed.dims(), &[32]);
        let restored = dequantize_fp4_mxfp4(&packed, &exponents, &[64])?;
        assert_eq!(restored.dims(), &[64]);
        let round_tripped = restored.to_vec1::<f32>()?;
        // E2M1 has coarse steps; allow the block's representable precision.
        for (original_value, restored_value) in values.iter().zip(round_tripped.iter()) {
            let tolerance = original_value.abs().max(1.0) * 0.5;
            assert!(
                (original_value - restored_value).abs() <= tolerance,
                "FP4 round-trip drifted: {original_value} vs {restored_value}"
            );
        }
        Ok(())
    }

    #[test]
    fn fp4_handles_odd_element_counts() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let weights = Tensor::from_vec(vec![1.0_f32; 33], (33,), &device)?;
        let (packed, exponents) = quantize_fp4_mxfp4(&weights)?;
        assert_eq!(packed.dims(), &[17]);
        assert_eq!(exponents.dims(), &[2]);
        let restored = dequantize_fp4_mxfp4(&packed, &exponents, &[33])?;
        assert_eq!(restored.dims(), &[33]);
        Ok(())
    }

    #[test]
    fn e2m1_codebook_round_trips() {
        // Nibbles 0 (0.0) and 8 (-0.0) decode to values that compare equal, so
        // the round-trip is asserted for the distinct values only.
        for nibble in 0_u8..16 {
            let value = decode_e2m1(nibble);
            if value == 0.0 {
                continue;
            }
            assert_eq!(encode_e2m1(value), nibble);
        }
        assert_eq!(decode_e2m1(0), 0.0);
        assert_eq!(decode_e2m1(8), -0.0);
    }

    #[test]
    fn e8m0_exponent_round_trips_for_representable_values() {
        for exponent in -8_i32..=8 {
            let byte = encode_e8m0_exponent(exponent as f32);
            let decoded = decode_e8m0_exponent(byte);
            assert!((decoded - pow2(exponent as f32)).abs() < 1e-6);
        }
    }

    #[test]
    fn config_driven_quantize_dequantize_round_trips() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let weights = Tensor::new(vec![vec![1.0_f32, 2.0, -3.0]], &device)?;
        let config = QuantizationConfig::new(QuantizationScheme::Fp8);
        let (quantized, auxiliary) = quantize(&config, &weights)?;
        let restored = dequantize(&config, &quantized, &auxiliary, weights.dims())?;
        assert_eq!(restored.elem_count(), weights.elem_count());
        Ok(())
    }

    #[test]
    fn no_op_quantization_returns_the_input() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let weights = Tensor::new(vec![1.0_f32, 2.0], &device)?;
        let config = QuantizationConfig::new(QuantizationScheme::None);
        let (quantized, _auxiliary) = quantize(&config, &weights)?;
        assert_eq!(quantized.to_vec1::<f32>()?, vec![1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn scale_tensor_name_appends_the_convention_suffix() {
        assert_eq!(
            scale_tensor_name("model.layers.0.mlp.down_proj.weight"),
            "model.layers.0.mlp.down_proj.weight_scale"
        );
        assert_eq!(
            shape_tensor_name("model.layers.0.mlp.down_proj.weight"),
            "model.layers.0.mlp.down_proj.weight_shape"
        );
        assert!(is_quantization_metadata("layer.weight_scale"));
        assert!(is_quantization_metadata("layer.weight_shape"));
        assert!(!is_quantization_metadata("layer.weight"));
    }

    #[test]
    fn fp4_round_trip_preserves_a_two_dimensional_shape() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..6).map(|index| index as f32 / 2.0).collect();
        let weights = Tensor::from_vec(values, (2, 3), &device)?;
        let (packed, exponents) = quantize_fp4_mxfp4(&weights)?;
        let restored = dequantize_fp4_mxfp4(&packed, &exponents, &[2, 3])?;
        assert_eq!(restored.dims(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn fp4_odd_element_count_stays_exact_with_the_original_shape() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let weights = Tensor::from_vec(vec![1.0_f32; 33], (33,), &device)?;
        let (packed, exponents) = quantize_fp4_mxfp4(&weights)?;
        let restored = dequantize_fp4_mxfp4(&packed, &exponents, &[33])?;
        assert_eq!(restored.elem_count(), 33, "no padding nibble may leak");
        Ok(())
    }

    #[test]
    fn checkpoint_tensor_map_fp8_dequantizes_and_consumes_scales() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let name = "model.layers.0.weight".to_string();
        let weights = Tensor::new(vec![vec![1.0_f32, -2.0, 0.5, 3.0]], &device)?;
        let (quantized, scales) = quantize_fp8_per_channel(&weights)?;
        let mut tensors = HashMap::new();
        tensors.insert(name.clone(), quantized);
        tensors.insert(scale_tensor_name(&name), scales);

        let dequantized = dequantize_checkpoint_tensors(tensors, QuantizationScheme::Fp8)?;
        assert_eq!(dequantized.len(), 1, "the scale tensor must be consumed");
        let restored = dequantized
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("missing weight '{name}'"))?;
        assert_eq!(restored.dtype(), DType::F32);
        assert_eq!(restored.dims(), &[1, 4]);
        Ok(())
    }

    #[test]
    fn checkpoint_tensor_map_fp4_dequantizes_with_packed_nibbles() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let name = "model.layers.0.proj.weight".to_string();
        let values: Vec<f32> = (0..32).map(|index| index as f32 / 8.0).collect();
        let weights = Tensor::from_vec(values, (32,), &device)?;
        let (packed, exponents) = quantize_fp4_mxfp4(&weights)?;
        let mut tensors = HashMap::new();
        tensors.insert(name.clone(), packed);
        tensors.insert(scale_tensor_name(&name), exponents);
        tensors.insert(
            shape_tensor_name(&name),
            Tensor::from_vec(vec![32_u32], (1,), &device)?,
        );

        let dequantized = dequantize_checkpoint_tensors(tensors, QuantizationScheme::Fp4)?;
        assert_eq!(dequantized.len(), 1);
        let restored = dequantized
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("missing weight '{name}'"))?;
        // 32 nibbles pack into 16 bytes, dequantized back to 32 F32 values.
        assert_eq!(restored.dims(), &[32]);
        assert_eq!(restored.dtype(), DType::F32);
        Ok(())
    }

    #[test]
    fn checkpoint_tensor_map_fp4_uses_the_stored_shape_for_two_dimensions() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let name = "model.layers.0.proj.weight".to_string();
        let values: Vec<f32> = (0..64).map(|index| index as f32 / 8.0).collect();
        let weights = Tensor::from_vec(values, (8, 8), &device)?;
        let (packed, exponents) = quantize_fp4_mxfp4(&weights)?;
        let mut tensors = HashMap::new();
        tensors.insert(name.clone(), packed);
        tensors.insert(scale_tensor_name(&name), exponents);
        tensors.insert(
            shape_tensor_name(&name),
            Tensor::from_vec(vec![8_u32, 8_u32], (2,), &device)?,
        );

        let dequantized = dequantize_checkpoint_tensors(tensors, QuantizationScheme::Fp4)?;
        assert_eq!(dequantized.len(), 1, "metadata tensors must be consumed");
        let restored = dequantized
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("missing weight '{name}'"))?;
        assert_eq!(restored.dims(), &[8, 8]);
        Ok(())
    }

    #[test]
    fn checkpoint_tensor_map_fp4_without_scales_is_an_error() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let name = "model.layers.0.proj.weight".to_string();
        let weights = Tensor::from_vec(vec![1.0_f32; 32], (32,), &device)?;
        let (packed, _exponents) = quantize_fp4_mxfp4(&weights)?;
        let mut tensors = HashMap::new();
        tensors.insert(name, packed);
        let result = dequantize_checkpoint_tensors(tensors, QuantizationScheme::Fp4);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn checkpoint_tensor_map_none_is_returned_unchanged() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "embed.weight".to_string(),
            Tensor::new(vec![1.0_f32, 2.0], &device)?,
        );
        let unchanged = dequantize_checkpoint_tensors(tensors.clone(), QuantizationScheme::None)?;
        assert_eq!(unchanged.len(), tensors.len());
        Ok(())
    }
}
