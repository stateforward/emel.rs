//! Maintained Sortformer log-mel feature extraction.
//!
//! Rust port of the pinned C++ implementation at
//! `emel.cpp/src/emel/diarization/sortformer/encoder/feature_extractor/detail.cpp`
//! (source commit `843a117386ef17dc5a50549bbfc821074c2141d6`). The model
//! contract is validated once at construction; dispatch receives caller-owned
//! PCM, output, and reusable scratch, and performs no heap allocation.
//!
//! Source-fixed contract: mono 16 kHz PCM; 400-sample centered windows;
//! 160-sample hop; 512-point FFT; 257 power bins; 128 mel bins; 1,504
//! feature frames from 240,640 input samples; preemphasis 0.97; power-spectrum
//! accumulation; and `log(acc + 1/2^24)` guarding.

#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    reason = "fixed-size source-parity extraction preserves the pinned scalar arithmetic and index proofs"
)]

use emel_model::bridge::Data;

/// Input sample rate fixed by the maintained Sortformer feature contract.
pub const SAMPLE_RATE: usize = 16_000;
/// Number of input channels fixed by the maintained contract.
pub const CHANNEL_COUNT: usize = 1;
/// Number of samples in each centered analysis window.
pub const WINDOW_LENGTH: usize = 400;
/// Distance between adjacent analysis windows.
pub const HOP_LENGTH: usize = 160;
/// FFT length used by the pinned extractor.
pub const FFT_SIZE: usize = 512;
/// Number of non-negative FFT bins (`FFT_SIZE / 2 + 1`).
pub const FFT_BIN_COUNT: usize = (FFT_SIZE / 2) + 1;
/// Number of mel filter-bank rows.
pub const FEATURE_BIN_COUNT: usize = 128;
/// Number of output feature frames.
pub const FEATURE_FRAME_COUNT: usize = 1_504;
/// Required mono PCM sample count.
pub const REQUIRED_SAMPLE_COUNT: usize = 240_640;
/// Required flattened output feature count.
pub const REQUIRED_FEATURE_COUNT: usize = FEATURE_FRAME_COUNT * FEATURE_BIN_COUNT;
/// Source-fixed preemphasis coefficient.
pub const PREEMPHASIS: f32 = 0.97;
/// Guard added before taking each mel-energy logarithm.
pub const LOG_ZERO_GUARD: f32 = 1.0 / 16_777_216.0;

const FILTER_BANK_NAME: &[u8] = b"prep.feat.fb";
const WINDOW_NAME: &[u8] = b"prep.feat.win";
const F32_BYTES: usize = core::mem::size_of::<f32>();
const FILTER_BANK_VALUE_COUNT: usize = FFT_BIN_COUNT * FEATURE_BIN_COUNT;
const FILTER_BANK_BYTE_COUNT: usize = FILTER_BANK_VALUE_COUNT * F32_BYTES;
const WINDOW_BYTE_COUNT: usize = WINDOW_LENGTH * F32_BYTES;
const FFT_WINDOW_PADDING: usize = (FFT_SIZE - WINDOW_LENGTH) / 2;
const PI: f32 = core::f32::consts::PI;

/// Construction or dispatch failures for the feature route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Error {
    /// The model architecture is not Sortformer.
    WrongArchitecture,
    /// A required tensor is absent or not resident.
    MissingTensor,
    /// A tensor has an unexpected rank or dimensions.
    Shape,
    /// A tensor has a non-f32 serialized type or wrong byte length.
    TensorContract,
    /// Caller-provided input/output capacity is not exact.
    Capacity,
    /// The model payload contains a malformed f32 value.
    NonFiniteModelValue,
}

/// Caller-owned fixed scratch for one synchronous extraction.
///
/// Construct once and reuse for every request. Extraction never allocates or
/// grows this storage during dispatch.
#[derive(Debug)]
#[allow(clippy::large_stack_arrays)]
pub struct Workspace {
    real: [f32; FFT_SIZE],
    imag: [f32; FFT_SIZE],
    power: [f32; FFT_BIN_COUNT],
}

impl Workspace {
    /// Creates zeroed scratch storage for one extraction route.
    #[must_use]
    #[allow(clippy::large_stack_arrays)]
    pub const fn new() -> Self {
        Self {
            real: [0.0; FFT_SIZE],
            imag: [0.0; FFT_SIZE],
            power: [0.0; FFT_BIN_COUNT],
        }
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable model-bound feature-extractor binding.
///
/// Both resident payloads are decoded during construction, so this binding
/// does not borrow the source [`Data`] after `from_model` returns.
#[derive(Debug)]
pub struct Binding {
    filter_bank: Box<[f32]>,
    window: Box<[f32]>,
}

/// Caller-owned synchronous route over a validated binding and reusable scratch.
#[allow(
    missing_debug_implementations,
    reason = "the route borrows caller-owned mutable workspace and binding state"
)]
pub struct Route<'a> {
    binding: &'a Binding,
    workspace: &'a mut Workspace,
}

impl<'a> Route<'a> {
    /// Associates a binding with caller-owned reusable scratch.
    #[must_use]
    pub const fn new(binding: &'a Binding, workspace: &'a mut Workspace) -> Self {
        Self { binding, workspace }
    }

    /// Extracts exactly [`REQUIRED_FEATURE_COUNT`] log-mel values.
    ///
    /// The operation is bounded and allocation-free after construction.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Capacity`] when the PCM or feature output length does
    /// not match the fixed extractor contract.
    pub fn extract(&mut self, pcm: &[f32], features: &mut [f32]) -> Result<(), Error> {
        self.binding.extract(pcm, features, self.workspace)
    }
}

impl Binding {
    /// Decodes and validates the pinned feature tensors from model-owned data.
    ///
    /// The returned binding owns copies of both payloads and remains valid if
    /// the source model is subsequently dropped or reused.
    ///
    /// # Errors
    ///
    /// Returns an error when the architecture, tensor metadata, resident
    /// payload, or model values violate the fixed extractor contract.
    pub fn from_model(data: &Data) -> Result<Self, Error> {
        if data.architecture_name() != emel_model::sortformer::ARCHITECTURE_NAME {
            return Err(Error::WrongArchitecture);
        }
        Ok(Self {
            filter_bank: decode_tensor(
                data,
                FILTER_BANK_NAME,
                3,
                [FFT_BIN_COUNT as u64, FEATURE_BIN_COUNT as u64, 1, 1],
                FILTER_BANK_BYTE_COUNT,
            )?,
            window: decode_tensor(
                data,
                WINDOW_NAME,
                1,
                [WINDOW_LENGTH as u64, 1, 1, 1],
                WINDOW_BYTE_COUNT,
            )?,
        })
    }

    /// Extracts using caller-provided reusable workspace.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Capacity`] when the PCM or feature output length does
    /// not match the fixed extractor contract.
    pub fn extract(
        &self,
        pcm: &[f32],
        features: &mut [f32],
        workspace: &mut Workspace,
    ) -> Result<(), Error> {
        if pcm.len() != REQUIRED_SAMPLE_COUNT || features.len() != REQUIRED_FEATURE_COUNT {
            return Err(Error::Capacity);
        }

        for frame in 0..FEATURE_FRAME_COUNT {
            let feature_offset = frame * FEATURE_BIN_COUNT;
            load_fft_frame(pcm, frame, &self.window, workspace);
            fft_in_place(&mut workspace.real, &mut workspace.imag);
            compute_power_spectrum(&workspace.real, &workspace.imag, &mut workspace.power);
            compute_mel_features(
                &self.filter_bank,
                &workspace.power,
                &mut features[feature_offset..feature_offset + FEATURE_BIN_COUNT],
            );
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn test_fixture() -> Self {
        let mut filter_bank = vec![0.0; FILTER_BANK_VALUE_COUNT];
        for mel in 0..FEATURE_BIN_COUNT {
            filter_bank[mel * FFT_BIN_COUNT] = 1.0;
        }
        Self::from_parts(filter_bank, vec![1.0; WINDOW_LENGTH])
    }

    #[cfg(test)]
    fn from_parts(filter_bank: Vec<f32>, window: Vec<f32>) -> Self {
        Self {
            filter_bank: filter_bank.into_boxed_slice(),
            window: window.into_boxed_slice(),
        }
    }
}

fn decode_tensor(
    data: &Data,
    name: &[u8],
    rank: usize,
    dimensions: [u64; 4],
    byte_len: usize,
) -> Result<Box<[f32]>, Error> {
    let tensor = data.tensor_named(name).ok_or(Error::MissingTensor)?;
    let metadata = tensor.metadata().ok_or(Error::TensorContract)?;
    let actual_rank = usize::try_from(metadata.dimension_count()).map_err(|_| Error::Shape)?;
    if actual_rank != rank || metadata.tensor_type().wire_code() != 0 {
        return Err(Error::TensorContract);
    }
    if metadata.dimensions() != dimensions {
        return Err(Error::Shape);
    }
    let bytes = tensor.bytes().ok_or(Error::MissingTensor)?;
    if bytes.len() != byte_len || !bytes.len().is_multiple_of(F32_BYTES) {
        return Err(Error::TensorContract);
    }
    let mut values = Vec::with_capacity(bytes.len() / F32_BYTES);
    for chunk in bytes.chunks_exact(F32_BYTES) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !value.is_finite() {
            return Err(Error::NonFiniteModelValue);
        }
        values.push(value);
    }
    Ok(values.into_boxed_slice())
}

fn preemphasized_sample(pcm: &[f32], index: isize) -> f32 {
    if index < 0 {
        return 0.0;
    }
    let index = index as usize;
    let Some(&current) = pcm.get(index) else {
        return 0.0;
    };
    if index == 0 {
        current
    } else {
        current - PREEMPHASIS * pcm[index - 1]
    }
}

fn load_fft_frame(pcm: &[f32], frame: usize, window: &[f32], workspace: &mut Workspace) {
    let frame_start = frame * HOP_LENGTH;
    for sample in 0..FFT_SIZE {
        let pcm_index = frame_start as isize + sample as isize - (FFT_SIZE / 2) as isize;
        let mut value = preemphasized_sample(pcm, pcm_index);
        if (FFT_WINDOW_PADDING..FFT_WINDOW_PADDING + WINDOW_LENGTH).contains(&sample) {
            value *= window[sample - FFT_WINDOW_PADDING];
        } else {
            value = 0.0;
        }
        workspace.real[sample] = value;
        workspace.imag[sample] = 0.0;
    }
}

fn fft_in_place(real: &mut [f32; FFT_SIZE], imag: &mut [f32; FFT_SIZE]) {
    let mut j = 0_usize;
    for i in 1..FFT_SIZE {
        let mut bit = FFT_SIZE >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= FFT_SIZE {
        let angle = (-2.0 * PI) / len as f32;
        let w_len_real = angle.cos();
        let w_len_imag = angle.sin();
        for offset in (0..FFT_SIZE).step_by(len) {
            let mut w_real = 1.0;
            let mut w_imag = 0.0;
            let half = len >> 1;
            for index in 0..half {
                let even = offset + index;
                let odd = even + half;
                let odd_real = real[odd] * w_real - imag[odd] * w_imag;
                let odd_imag = real[odd] * w_imag + imag[odd] * w_real;
                real[odd] = real[even] - odd_real;
                imag[odd] = imag[even] - odd_imag;
                real[even] += odd_real;
                imag[even] += odd_imag;
                let next_w_real = w_real * w_len_real - w_imag * w_len_imag;
                let next_w_imag = w_real * w_len_imag + w_imag * w_len_real;
                w_real = next_w_real;
                w_imag = next_w_imag;
            }
        }
        len <<= 1;
    }
}

fn compute_power_spectrum(
    real: &[f32; FFT_SIZE],
    imag: &[f32; FFT_SIZE],
    power: &mut [f32; FFT_BIN_COUNT],
) {
    for bin in 0..FFT_BIN_COUNT {
        power[bin] = real[bin] * real[bin] + imag[bin] * imag[bin];
    }
}

fn compute_mel_features(
    filter_bank: &[f32],
    power: &[f32; FFT_BIN_COUNT],
    feature_row: &mut [f32],
) {
    for mel in 0..FEATURE_BIN_COUNT {
        let row_base = mel * FFT_BIN_COUNT;
        let mut acc = 0.0;
        for bin in 0..FFT_BIN_COUNT {
            acc += filter_bank[row_base + bin] * power[bin];
        }
        feature_row[mel] = (acc + LOG_ZERO_GUARD).ln();
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::suboptimal_flops)]
mod tests {
    use super::*;

    fn binding() -> Binding {
        let mut filter_bank = vec![0.0; FILTER_BANK_VALUE_COUNT];
        for mel in 0..FEATURE_BIN_COUNT {
            filter_bank[mel * FFT_BIN_COUNT] = 1.0;
        }
        Binding::from_parts(filter_bank, vec![1.0; WINDOW_LENGTH])
    }

    #[test]
    fn extraction_is_finite_deterministic_and_fully_written() {
        let binding = binding();
        let pcm: Vec<f32> = (0..REQUIRED_SAMPLE_COUNT)
            .map(|index| (index % 37) as f32 * 0.001 - 0.02)
            .collect();
        let mut first = vec![f32::NAN; REQUIRED_FEATURE_COUNT];
        let mut second = vec![f32::NAN; REQUIRED_FEATURE_COUNT];
        let mut workspace = Workspace::default();
        binding.extract(&pcm, &mut first, &mut workspace).unwrap();
        binding.extract(&pcm, &mut second, &mut workspace).unwrap();
        assert_eq!(first, second);
        assert!(first.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn extraction_rejects_hostile_shapes_and_capacity() {
        let binding = binding();
        let mut workspace = Workspace::default();
        let mut features = vec![0.0; REQUIRED_FEATURE_COUNT];
        assert_eq!(
            binding.extract(&[], &mut features, &mut workspace),
            Err(Error::Capacity)
        );
        let pcm = vec![0.0; REQUIRED_SAMPLE_COUNT];
        assert_eq!(
            binding.extract(&pcm, &mut [], &mut workspace),
            Err(Error::Capacity)
        );
    }
}
