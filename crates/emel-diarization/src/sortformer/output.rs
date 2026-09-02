//! Model-bound native Sortformer speaker-probability output.
//!
//! The binding owns decoded F32 module tensors. [`Route::compute`] uses only
//! caller-owned fixed workspace after construction, so the numeric path does
//! not allocate or retain model-source borrows during dispatch.

use emel_kernels::any::sortformer::{self, DenseBatch};
use emel_model::bridge::Data;

use super::pipeline::sm::{
    FRAME_COUNT, HIDDEN_DIM, REQUIRED_HIDDEN_VALUE_COUNT, REQUIRED_PROBABILITY_VALUE_COUNT,
    SegmentRecord, SPEAKER_COUNT,
};

const FRAME_COUNT_USIZE: usize = FRAME_COUNT as usize;
const HIDDEN_DIM_USIZE: usize = HIDDEN_DIM as usize;
const SPEAKER_COUNT_USIZE: usize = SPEAKER_COUNT as usize;
const F32_BYTES: usize = core::mem::size_of::<f32>();

const FH2H_WEIGHT_NAME: &[u8] = b"mods.fh2h.w";
const FH2H_BIAS_NAME: &[u8] = b"mods.fh2h.b";
const SH2S_WEIGHT_NAME: &[u8] = b"mods.sh2s.w";
const SH2S_BIAS_NAME: &[u8] = b"mods.sh2s.b";

const FH2H_WEIGHT_VALUES: usize = HIDDEN_DIM_USIZE * HIDDEN_DIM_USIZE;
const FH2H_BIAS_VALUES: usize = HIDDEN_DIM_USIZE;
const SH2S_WEIGHT_VALUES: usize = SPEAKER_COUNT_USIZE * HIDDEN_DIM_USIZE;
const SH2S_BIAS_VALUES: usize = SPEAKER_COUNT_USIZE;
const FH2H_WEIGHT_BYTES: usize = FH2H_WEIGHT_VALUES * F32_BYTES;
const FH2H_BIAS_BYTES: usize = FH2H_BIAS_VALUES * F32_BYTES;
const SH2S_WEIGHT_BYTES: usize = SH2S_WEIGHT_VALUES * F32_BYTES;
const SH2S_BIAS_BYTES: usize = SH2S_BIAS_VALUES * F32_BYTES;

/// Native output binding failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Error {
    /// The model architecture or required tensor is absent.
    ModelInvalid,
    /// A tensor has an unexpected type, shape, or serialized length.
    TensorContract,
    /// Caller-provided input or output capacity is not exact.
    Shape,
    /// The dense kernel rejected a validated operation.
    Kernel,
    /// Activity threshold is not finite or lies outside `[0, 1]`.
    Threshold,
    /// Caller-provided segment capacity is insufficient.
    SegmentCapacity,
}

/// Fixed scratch for one synchronous native output computation.
///
/// Construct once and reuse. The arrays are intentionally sized for the
/// largest dense stage; the second stage reuses the same transposition space.
#[derive(Debug)]
#[allow(clippy::large_stack_arrays, clippy::large_stack_frames)]
pub struct Workspace {
    input_relu: [f32; REQUIRED_HIDDEN_VALUE_COUNT],
    hidden_stage: [f32; REQUIRED_HIDDEN_VALUE_COUNT],
    transposed_input: [f32; REQUIRED_HIDDEN_VALUE_COUNT],
    transposed_output: [f32; REQUIRED_HIDDEN_VALUE_COUNT],
    logits: [f32; REQUIRED_PROBABILITY_VALUE_COUNT],
}

impl Default for Workspace {
    #[allow(clippy::large_stack_arrays, clippy::large_stack_frames)]
    fn default() -> Self {
        Self {
            input_relu: [0.0; REQUIRED_HIDDEN_VALUE_COUNT],
            hidden_stage: [0.0; REQUIRED_HIDDEN_VALUE_COUNT],
            transposed_input: [0.0; REQUIRED_HIDDEN_VALUE_COUNT],
            transposed_output: [0.0; REQUIRED_HIDDEN_VALUE_COUNT],
            logits: [0.0; REQUIRED_PROBABILITY_VALUE_COUNT],
        }
    }
}

/// Immutable model-bound native output tensors.
#[derive(Debug)]
pub struct Binding {
    fh2h_weights: Box<[f32]>,
    fh2h_bias: Box<[f32]>,
    sh2s_weights: Box<[f32]>,
    sh2s_bias: Box<[f32]>,
}

/// Synchronous native output route over a binding and reusable workspace.
#[derive(Debug)]
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

    /// Computes frame-major speaker probabilities without dispatch allocation.
    pub fn compute(&mut self, hidden: &[f32], probabilities: &mut [f32]) -> Result<(), Error> {
        if hidden.len() != REQUIRED_HIDDEN_VALUE_COUNT
            || probabilities.len() != REQUIRED_PROBABILITY_VALUE_COUNT
        {
            return Err(Error::Shape);
        }
        self.binding.compute(hidden, probabilities, self.workspace)
    }

    /// Decodes speaker-major contiguous active runs from frame-major probabilities.
    /// Runs use inclusive threshold activation and flush an active final run.
    pub fn decode_segments(
        &mut self,
        probabilities: &[f32],
        threshold: f32,
        segments: &mut [SegmentRecord],
        segment_count: &mut i32,
    ) -> Result<(), Error> {
        *segment_count = 0;
        if probabilities.len() != REQUIRED_PROBABILITY_VALUE_COUNT {
            return Err(Error::Shape);
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(Error::Threshold);
        }
        for speaker in 0..SPEAKER_COUNT {
            let mut start_frame = -1;
            let mut max_probability = 0.0_f32;
            for frame in 0..FRAME_COUNT {
                let offset = (frame as usize) * SPEAKER_COUNT_USIZE + speaker as usize;
                let probability = probabilities[offset];
                if probability >= threshold {
                    if start_frame < 0 {
                        start_frame = frame;
                        max_probability = probability;
                    } else {
                        max_probability = max_probability.max(probability);
                    }
                } else if start_frame >= 0 {
                    append_segment(segments, segment_count, speaker, start_frame, frame, max_probability)?;
                    start_frame = -1;
                    max_probability = 0.0;
                }
            }
            if start_frame >= 0 {
                append_segment(segments, segment_count, speaker, start_frame, FRAME_COUNT, max_probability)?;
            }
        }
        Ok(())
    }
}

fn append_segment(
    segments: &mut [SegmentRecord],
    segment_count: &mut i32,
    speaker: i32,
    start_frame: i32,
    end_frame: i32,
    max_probability: f32,
) -> Result<(), Error> {
    let index = usize::try_from(*segment_count).map_err(|_| Error::SegmentCapacity)?;
    let Some(segment) = segments.get_mut(index) else {
        return Err(Error::SegmentCapacity);
    };
    *segment = SegmentRecord {
        speaker,
        start_frame,
        end_frame,
        start_seconds: (start_frame as f32) * 0.08,
        end_seconds: (end_frame as f32) * 0.08,
        max_probability,
    };
    *segment_count += 1;
    Ok(())
}

impl Binding {
    fn decode_tensor(
        data: &Data,
        name: &[u8],
        dimensions: &[u64],
        byte_len: usize,
    ) -> Result<Box<[f32]>, Error> {
        let tensor = data.tensor_named(name).ok_or(Error::ModelInvalid)?;
        let metadata = tensor.metadata().ok_or(Error::ModelInvalid)?;
        let rank =
            usize::try_from(metadata.dimension_count()).map_err(|_| Error::TensorContract)?;
        let actual = metadata.dimensions();
        if rank > actual.len()
            || rank != dimensions.len()
            || metadata.tensor_type().wire_code() != 0
            || actual[..rank] != dimensions[..]
        {
            return Err(Error::TensorContract);
        }

        let bytes = tensor.bytes().ok_or(Error::ModelInvalid)?;
        if bytes.len() != byte_len || !bytes.len().is_multiple_of(F32_BYTES) {
            return Err(Error::TensorContract);
        }

        let (chunks, remainder) = bytes.as_chunks::<F32_BYTES>();
        if !remainder.is_empty() {
            return Err(Error::TensorContract);
        }
        let mut values = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            values.push(f32::from_le_bytes(*chunk));
        }
        Ok(values.into_boxed_slice())
    }

    /// Decodes the exact resident F32 output tensors from model-owned data.
    ///
    /// The returned binding owns all decoded payloads and does not borrow
    /// `data` after construction.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ModelInvalid`] when the model architecture or required
    /// tensor data is absent, or [`Error::TensorContract`] for an unexpected
    /// tensor type, shape, or serialized length.
    pub fn from_model(data: &Data) -> Result<Self, Error> {
        if data.architecture_name() != emel_model::sortformer::ARCHITECTURE_NAME {
            return Err(Error::ModelInvalid);
        }
        Ok(Self {
            fh2h_weights: Self::decode_tensor(
                data,
                FH2H_WEIGHT_NAME,
                &[HIDDEN_DIM as u64, HIDDEN_DIM as u64],
                FH2H_WEIGHT_BYTES,
            )?,
            fh2h_bias: Self::decode_tensor(
                data,
                FH2H_BIAS_NAME,
                &[HIDDEN_DIM as u64],
                FH2H_BIAS_BYTES,
            )?,
            // GGUF metadata is [192, 4], while dense rows are four 192-wide
            // output rows in the resident scalar payload.
            sh2s_weights: Self::decode_tensor(
                data,
                SH2S_WEIGHT_NAME,
                &[HIDDEN_DIM as u64, SPEAKER_COUNT as u64],
                SH2S_WEIGHT_BYTES,
            )?,
            sh2s_bias: Self::decode_tensor(
                data,
                SH2S_BIAS_NAME,
                &[SPEAKER_COUNT as u64],
                SH2S_BIAS_BYTES,
            )?,
        })
    }

    fn compute(
        &self,
        hidden: &[f32],
        probabilities: &mut [f32],
        workspace: &mut Workspace,
    ) -> Result<(), Error> {
        for (destination, value) in workspace.input_relu.iter_mut().zip(hidden.iter().copied()) {
            *destination = value.max(0.0);
        }

        if !sortformer::dense_batch(DenseBatch {
            input_rows: &workspace.input_relu,
            row_count: FRAME_COUNT_USIZE,
            input_dim: HIDDEN_DIM_USIZE,
            weights: &self.fh2h_weights,
            bias: &self.fh2h_bias,
            output_dim: HIDDEN_DIM_USIZE,
            transposed_input: &mut workspace.transposed_input,
            transposed_output: &mut workspace.transposed_output,
            output_rows: &mut workspace.hidden_stage,
        }) {
            return Err(Error::Kernel);
        }

        for value in &mut workspace.hidden_stage {
            *value = value.max(0.0);
        }

        if !sortformer::dense_batch(DenseBatch {
            input_rows: &workspace.hidden_stage,
            row_count: FRAME_COUNT_USIZE,
            input_dim: HIDDEN_DIM_USIZE,
            weights: &self.sh2s_weights,
            bias: &self.sh2s_bias,
            output_dim: SPEAKER_COUNT_USIZE,
            transposed_input: &mut workspace.transposed_input,
            transposed_output: &mut workspace.transposed_output,
            output_rows: &mut workspace.logits,
        }) {
            return Err(Error::Kernel);
        }

        for (destination, logit) in probabilities
            .iter_mut()
            .zip(workspace.logits.iter().copied())
        {
            *destination = sigmoid(logit);
        }
        Ok(())
    }

    #[cfg(test)]
    fn from_parts(
        fh2h_weights: Vec<f32>,
        fh2h_bias: Vec<f32>,
        sh2s_weights: Vec<f32>,
        sh2s_bias: Vec<f32>,
    ) -> Self {
        Self {
            fh2h_weights: fh2h_weights.into_boxed_slice(),
            fh2h_bias: fh2h_bias.into_boxed_slice(),
            sh2s_weights: sh2s_weights.into_boxed_slice(),
            sh2s_bias: sh2s_bias.into_boxed_slice(),
        }
    }
}

#[inline]
fn sigmoid(value: f32) -> f32 {
    let clamped = value.clamp(-80.0, 80.0);
    (1.0 / (1.0 + (-clamped).exp())).clamp(0.0, 1.0)
}

/// Number of hidden values consumed by the route.
pub const INPUT_VALUE_COUNT: usize = REQUIRED_HIDDEN_VALUE_COUNT;
/// Number of frame-major probabilities produced by the route.
pub const OUTPUT_VALUE_COUNT: usize = REQUIRED_PROBABILITY_VALUE_COUNT;

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn native_output_is_frame_major_and_relu_sigmoid_numeric() {
        let mut fh2h_weights = vec![0.0; FH2H_WEIGHT_VALUES];
        for index in 0..HIDDEN_DIM_USIZE {
            fh2h_weights[index * HIDDEN_DIM_USIZE + index] = 1.0;
        }
        let mut sh2s_weights = vec![0.0; SH2S_WEIGHT_VALUES];
        sh2s_weights[0] = 1.0;
        sh2s_weights[HIDDEN_DIM_USIZE + 1] = 2.0;
        let binding = Binding::from_parts(
            fh2h_weights,
            vec![0.0; FH2H_BIAS_VALUES],
            sh2s_weights,
            vec![0.0; SH2S_BIAS_VALUES],
        );
        let mut workspace = Workspace::default();
        let mut route = Route::new(&binding, &mut workspace);
        let mut hidden = vec![0.0; INPUT_VALUE_COUNT];
        hidden[0] = 2.0;
        hidden[1] = -3.0;
        hidden[HIDDEN_DIM_USIZE] = 1.0;
        let mut probabilities = vec![f32::NAN; OUTPUT_VALUE_COUNT];

        route.compute(&hidden, &mut probabilities).unwrap();

        let expected = 1.0 / (1.0 + (-2.0_f32).exp());
        assert!((probabilities[0] - expected).abs() < 1e-6);
        assert!((probabilities[1] - 0.5).abs() < 1e-6);
        assert!(
            (probabilities[SPEAKER_COUNT_USIZE] - (1.0 / (1.0 + (-1.0_f32).exp()))).abs() < 1e-6
        );
    }

    #[test]
    fn hostile_shapes_leave_public_output_untouched() {
        let binding = Binding::from_parts(
            vec![0.0; FH2H_WEIGHT_VALUES],
            vec![0.0; FH2H_BIAS_VALUES],
            vec![0.0; SH2S_WEIGHT_VALUES],
            vec![0.0; SH2S_BIAS_VALUES],
        );
        let mut workspace = Workspace::default();
        let mut route = Route::new(&binding, &mut workspace);
        let hidden = vec![0.0; INPUT_VALUE_COUNT - 1];
        let mut probabilities = vec![7.0; OUTPUT_VALUE_COUNT];

        assert_eq!(
            route.compute(&hidden, &mut probabilities),
            Err(Error::Shape)
        );
        assert!(probabilities.iter().all(|value| *value == 7.0));

        let hidden = vec![0.0; INPUT_VALUE_COUNT];
        let mut probabilities = vec![7.0; OUTPUT_VALUE_COUNT - 1];
        assert_eq!(
            route.compute(&hidden, &mut probabilities),
            Err(Error::Shape)
        );
        assert!(probabilities.iter().all(|value| *value == 7.0));
    }

    fn empty_route() -> Route<'static> {
        let binding = Box::leak(Box::new(Binding::from_parts(
            vec![0.0; FH2H_WEIGHT_VALUES],
            vec![0.0; FH2H_BIAS_VALUES],
            vec![0.0; SH2S_WEIGHT_VALUES],
            vec![0.0; SH2S_BIAS_VALUES],
        )));
        let workspace = Box::leak(Box::new(Workspace::default()));
        Route::new(binding, workspace)
    }

    #[test]
    fn native_decoder_emits_speaker_major_runs_with_inclusive_threshold_and_flush() {
        let mut route = empty_route();
        let mut probabilities = vec![0.0; OUTPUT_VALUE_COUNT];
        for frame in 1..4 {
            probabilities[frame * SPEAKER_COUNT_USIZE] = if frame == 2 { 0.5 } else { 0.75 };
        }
        for frame in 2..6 {
            probabilities[frame * SPEAKER_COUNT_USIZE + 1] = 0.875;
        }
        let mut segments = [SegmentRecord::default(); 4];
        let mut count = -1;

        route
            .decode_segments(&probabilities, 0.5, &mut segments, &mut count)
            .unwrap();

        assert_eq!(count, 2);
        assert_eq!(segments[0].speaker, 0);
        assert_eq!(segments[0].start_frame, 1);
        assert_eq!(segments[0].end_frame, 4);
        assert_eq!(segments[0].start_seconds, 0.08);
        assert_eq!(segments[0].end_seconds, 0.32);
        assert_eq!(segments[0].max_probability, 0.75);
        assert_eq!(segments[1].speaker, 1);
        assert_eq!(segments[1].start_frame, 2);
        assert_eq!(segments[1].end_frame, 6);
        assert_eq!(segments[1].start_seconds, 0.16);
        assert_eq!(segments[1].end_seconds, 0.48);
        assert_eq!(segments[1].max_probability, 0.875);
    }

    #[test]
    fn native_decoder_resets_count_on_invalid_input_and_reports_capacity() {
        let mut route = empty_route();
        let probabilities = vec![0.75; OUTPUT_VALUE_COUNT];
        let mut segments = [SegmentRecord::default(); 1];
        let mut count = 99;

        assert_eq!(
            route.decode_segments(&probabilities[..OUTPUT_VALUE_COUNT - 1], 0.5, &mut segments, &mut count),
            Err(Error::Shape)
        );
        assert_eq!(count, 0);
        assert_eq!(
            route.decode_segments(&probabilities, f32::NAN, &mut segments, &mut count),
            Err(Error::Threshold)
        );
        assert_eq!(count, 0);
        assert_eq!(
            route.decode_segments(&probabilities, 0.5, &mut segments, &mut count),
            Err(Error::SegmentCapacity)
        );
        assert_eq!(count, 1);
        assert_eq!(segments[0].speaker, 0);
    }
 }
