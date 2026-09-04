//! Model-bound native Sortformer encoder projection.

use emel_kernels::any::sortformer::{self, DenseBatch};
use emel_model::bridge::Data;

use super::sm::{ENCODER_DIM, FRAME_COUNT, HIDDEN_DIM};

const WEIGHT_NAME: &[u8] = b"mods.ep.w";
const BIAS_NAME: &[u8] = b"mods.ep.b";
const F32_BYTES: usize = core::mem::size_of::<f32>();
const WEIGHT_BYTES: usize = HIDDEN_DIM * ENCODER_DIM * F32_BYTES;
const BIAS_BYTES: usize = HIDDEN_DIM * F32_BYTES;

pub const INPUT_VALUE_COUNT: usize = FRAME_COUNT * ENCODER_DIM;
pub const OUTPUT_VALUE_COUNT: usize = FRAME_COUNT * HIDDEN_DIM;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Error {
    ModelInvalid,
    TensorContract,
    Shape,
    Kernel,
}

#[derive(Debug)]
#[allow(clippy::large_stack_arrays, clippy::large_stack_frames)]
pub struct Workspace {
    transposed_input: [f32; INPUT_VALUE_COUNT],
    transposed_output: [f32; OUTPUT_VALUE_COUNT],
}

impl Default for Workspace {
    #[allow(clippy::large_stack_arrays, clippy::large_stack_frames)]
    fn default() -> Self {
        Self {
            transposed_input: [0.0; INPUT_VALUE_COUNT],
            transposed_output: [0.0; OUTPUT_VALUE_COUNT],
        }
    }
}

#[derive(Debug)]
pub struct Binding {
    weights: Box<[f32]>,
    bias: Box<[f32]>,
}
/// Caller-owned native projection binding and preallocated workspace.
///
/// The route retains no model storage beyond the validated binding reference;
/// callers retain ownership of both the binding and reusable workspace.
#[derive(Debug)]
pub struct Route<'a> {
    binding: &'a Binding,
    workspace: &'a mut Workspace,
}

impl<'a> Route<'a> {
    #[must_use]
    pub const fn new(binding: &'a Binding, workspace: &'a mut Workspace) -> Self {
        Self { binding, workspace }
    }
    /// Projects encoder frames into hidden frames.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] for incorrect buffer lengths or [`Error::Kernel`]
    /// when the underlying dense kernel rejects the operation.
    pub fn project(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), Error> {
        self.binding.project(input, output, self.workspace)
    }
}

impl Binding {
    fn decode_tensor(
        data: &Data,
        name: &[u8],
        dims: &[u64],
        byte_len: usize,
    ) -> Result<Box<[f32]>, Error> {
        let tensor = data.tensor_named(name).ok_or(Error::ModelInvalid)?;
        let metadata = tensor.metadata().ok_or(Error::ModelInvalid)?;
        let rank =
            usize::try_from(metadata.dimension_count()).map_err(|_| Error::TensorContract)?;
        let actual = metadata.dimensions();
        if rank > actual.len()
            || rank != dims.len()
            || metadata.tensor_type().wire_code() != 0
            || actual[..rank] != dims[..]
        {
            return Err(Error::TensorContract);
        }
        let bytes = tensor.bytes().ok_or(Error::ModelInvalid)?;
        if bytes.len() != byte_len || !bytes.len().is_multiple_of(F32_BYTES) {
            return Err(Error::TensorContract);
        }
        let mut values = Vec::with_capacity(bytes.len() / F32_BYTES);
        let (chunks, _) = bytes.as_chunks::<F32_BYTES>();
        for chunk in chunks {
            values.push(f32::from_le_bytes(*chunk));
        }
        Ok(values.into_boxed_slice())
    }

    /// Decodes the pinned projection tensors from model data.
    ///
    /// # Errors
    ///
    /// Returns an error when the architecture, tensor metadata, or tensor byte
    /// lengths do not match the pinned projection contract.
    pub fn from_model(data: &Data) -> Result<Self, Error> {
        if data.architecture_name() != emel_model::sortformer::ARCHITECTURE_NAME {
            return Err(Error::ModelInvalid);
        }
        Ok(Self {
            weights: Self::decode_tensor(
                data,
                WEIGHT_NAME,
                &[ENCODER_DIM as u64, HIDDEN_DIM as u64],
                WEIGHT_BYTES,
            )?,
            bias: Self::decode_tensor(data, BIAS_NAME, &[HIDDEN_DIM as u64], BIAS_BYTES)?,
        })
    }

    /// Projects one encoder chunk using the supplied reusable workspace.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] for incorrect buffer lengths or [`Error::Kernel`]
    /// when the underlying dense kernel rejects the operation.
    pub fn project(
        &self,
        input: &[f32],
        output: &mut [f32],
        workspace: &mut Workspace,
    ) -> Result<(), Error> {
        if input.len() != INPUT_VALUE_COUNT || output.len() != OUTPUT_VALUE_COUNT {
            return Err(Error::Shape);
        }
        if sortformer::dense_batch(DenseBatch {
            input_rows: input,
            row_count: FRAME_COUNT,
            input_dim: ENCODER_DIM,
            weights: &self.weights,
            bias: &self.bias,
            output_dim: HIDDEN_DIM,
            transposed_input: &mut workspace.transposed_input,
            transposed_output: &mut workspace.transposed_output,
            output_rows: output,
        }) {
            Ok(())
        } else {
            Err(Error::Kernel)
        }
    }

    #[cfg(test)]
    fn from_parts(weights: Vec<f32>, bias: Vec<f32>) -> Self {
        Self {
            weights: weights.into_boxed_slice(),
            bias: bias.into_boxed_slice(),
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    clippy::float_cmp
)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_projection_matches_scalar_reference_and_writes_all_outputs() {
        let weights: Vec<f32> = (0..HIDDEN_DIM * ENCODER_DIM)
            .map(|i| (i % 17) as f32 * 0.01 - 0.08)
            .collect();
        let bias: Vec<f32> = (0..HIDDEN_DIM).map(|i| i as f32 * 0.125 - 0.5).collect();
        let binding = Binding::from_parts(weights.clone(), bias.clone());
        let input: Vec<f32> = (0..INPUT_VALUE_COUNT)
            .map(|i| (i % 31) as f32 * 0.02 - 0.3)
            .collect();
        let mut output = vec![f32::NAN; OUTPUT_VALUE_COUNT];
        let mut workspace = Workspace::default();
        binding
            .project(&input, &mut output, &mut workspace)
            .unwrap();
        for frame in 0..FRAME_COUNT {
            for col in 0..HIDDEN_DIM {
                let mut expected = bias[col];
                for row in 0..ENCODER_DIM {
                    expected += weights[col * ENCODER_DIM + row] * input[frame * ENCODER_DIM + row];
                }
                assert_eq!(output[frame * HIDDEN_DIM + col], expected);
            }
        }
        assert!(output.iter().all(|value| !value.is_nan()));
    }

    #[test]
    fn projection_rejects_wrong_shapes() {
        let binding =
            Binding::from_parts(vec![0.0; HIDDEN_DIM * ENCODER_DIM], vec![0.0; HIDDEN_DIM]);
        let mut workspace = Workspace::default();
        let mut output = vec![0.0; OUTPUT_VALUE_COUNT];
        assert_eq!(
            binding.project(&[], &mut output, &mut workspace),
            Err(Error::Shape)
        );
    }
}
