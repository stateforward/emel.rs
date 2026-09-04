//! Model-bound native Sortformer transformer execution.
//!
//! Tensor names and dimensions follow the pinned `transformer::detail` contract:
//! `te.l{layer}.{suffix}`, 18 layers and 16 tensors per layer. Binding owns
//! decoded resident weights; dispatch reuses one caller-owned kernel workspace.

use emel_kernels::any::sortformer::{self, TransformerLayerWorkspace};
use emel_model::bridge::Data;

use super::sm::{FRAME_COUNT, HIDDEN_DIM, TRANSFORMER_LAYER_COUNT};

const F32_BYTES: usize = core::mem::size_of::<f32>();
const INNER_DIM: usize = 768;
const LAYER_TENSOR_COUNT: usize = 16;
const HIDDEN_MATRIX: usize = HIDDEN_DIM * HIDDEN_DIM;
const FF_IN_MATRIX: usize = HIDDEN_DIM * INNER_DIM;
const FF_OUT_MATRIX: usize = INNER_DIM * HIDDEN_DIM;

const SUFFIXES: [&[u8]; LAYER_TENSOR_COUNT] = [
    b"sa.k.b", b"sa.k.w", b"sa.o.b", b"sa.o.w", b"sa.q.b", b"sa.q.w", b"sa.v.b", b"sa.v.w",
    b"ln1.b", b"ln1.w", b"ln2.b", b"ln2.w", b"ff.di.b", b"ff.di.w", b"ff.do.b", b"ff.do.w",
];
const LENGTHS: [usize; LAYER_TENSOR_COUNT] = [
    HIDDEN_DIM,
    HIDDEN_MATRIX,
    HIDDEN_DIM,
    HIDDEN_MATRIX,
    HIDDEN_DIM,
    HIDDEN_MATRIX,
    HIDDEN_DIM,
    HIDDEN_MATRIX,
    HIDDEN_DIM,
    HIDDEN_DIM,
    HIDDEN_DIM,
    HIDDEN_DIM,
    INNER_DIM,
    FF_IN_MATRIX,
    HIDDEN_DIM,
    FF_OUT_MATRIX,
];
const DIMS: [[u64; 2]; LAYER_TENSOR_COUNT] = [
    [HIDDEN_DIM as u64, 0],
    [HIDDEN_DIM as u64, HIDDEN_DIM as u64],
    [HIDDEN_DIM as u64, 0],
    [HIDDEN_DIM as u64, HIDDEN_DIM as u64],
    [HIDDEN_DIM as u64, 0],
    [HIDDEN_DIM as u64, HIDDEN_DIM as u64],
    [HIDDEN_DIM as u64, 0],
    [HIDDEN_DIM as u64, HIDDEN_DIM as u64],
    [HIDDEN_DIM as u64, 0],
    [HIDDEN_DIM as u64, 0],
    [HIDDEN_DIM as u64, 0],
    [HIDDEN_DIM as u64, 0],
    [INNER_DIM as u64, 0],
    [HIDDEN_DIM as u64, INNER_DIM as u64],
    [HIDDEN_DIM as u64, 0],
    [INNER_DIM as u64, HIDDEN_DIM as u64],
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Error {
    ModelInvalid,
    TensorContract,
    Shape,
    Kernel,
}

#[derive(Debug, Default)]
pub struct Workspace {
    pub(crate) kernel: TransformerLayerWorkspace,
}

#[derive(Debug)]
struct Layer {
    tensors: [Box<[f32]>; LAYER_TENSOR_COUNT],
}

/// Immutable model weights for all pinned transformer layers.
#[derive(Debug)]
pub struct Binding {
    layers: Box<[Layer]>,
}

/// Caller-owned native transformer binding and reusable workspace.
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

    /// Executes one pinned transformer layer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] for invalid layer or buffer dimensions, or
    /// [`Error::Kernel`] when a lower kernel rejects execution.
    pub fn execute(
        &mut self,
        layer: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        self.binding
            .execute(layer, input, output, &mut self.workspace.kernel)
    }
}

impl Binding {
    fn tensor_name(layer: usize, suffix: &[u8], out: &mut [u8; 32]) -> Option<usize> {
        let mut at = 0;
        out[at] = b't';
        at += 1;
        out[at] = b'e';
        at += 1;
        out[at] = b'.';
        at += 1;
        out[at] = b'l';
        at += 1;
        let mut divisor = 100;
        let mut started = false;
        while divisor != 0 {
            let digit = (layer / divisor) % 10;
            if digit != 0 || started || divisor == 1 {
                out[at] = b'0' + u8::try_from(digit).expect("decimal digit");
                at += 1;
                started = true;
            }
            divisor /= 10;
        }
        out[at] = b'.';
        at += 1;
        if at + suffix.len() > out.len() {
            return None;
        }
        out[at..at + suffix.len()].copy_from_slice(suffix);
        Some(at + suffix.len())
    }

    fn decode_tensor(
        data: &Data,
        name: &[u8],
        dims: [u64; 2],
        rank: usize,
        length: usize,
    ) -> Result<Box<[f32]>, Error> {
        let tensor = data.tensor_named(name).ok_or(Error::ModelInvalid)?;
        let metadata = tensor.metadata().ok_or(Error::ModelInvalid)?;
        let actual = metadata.dimensions();
        if metadata.tensor_type().wire_code() != 0
            || metadata.dimension_count() as usize != rank
            || actual[0] != dims[0]
            || (rank == 2 && actual[1] != dims[1])
            || (rank == 1 && actual[1] != 1)
        {
            return Err(Error::TensorContract);
        }
        let bytes = tensor.bytes().ok_or(Error::ModelInvalid)?;
        let byte_len = length.checked_mul(F32_BYTES).ok_or(Error::TensorContract)?;
        if bytes.len() != byte_len {
            return Err(Error::TensorContract);
        }
        let mut values = Vec::with_capacity(length);
        let (chunks, _) = bytes.as_chunks::<F32_BYTES>();
        for chunk in chunks {
            values.push(f32::from_le_bytes(*chunk));
        }
        Ok(values.into_boxed_slice())
    }

    /// Decodes the pinned transformer tensors from model data.
    ///
    /// # Errors
    ///
    /// Returns an error when the architecture, tensor metadata, or tensor byte
    /// lengths do not match the pinned transformer contract.
    pub fn from_model(data: &Data) -> Result<Self, Error> {
        if data.architecture_name() != emel_model::sortformer::ARCHITECTURE_NAME {
            return Err(Error::ModelInvalid);
        }
        let mut layers = Vec::with_capacity(TRANSFORMER_LAYER_COUNT);
        for layer in 0..TRANSFORMER_LAYER_COUNT {
            let mut tensors = Vec::with_capacity(LAYER_TENSOR_COUNT);
            for index in 0..LAYER_TENSOR_COUNT {
                let mut name = [0_u8; 32];
                let length = Self::tensor_name(layer, SUFFIXES[index], &mut name)
                    .ok_or(Error::TensorContract)?;
                tensors.push(Self::decode_tensor(
                    data,
                    &name[..length],
                    DIMS[index],
                    if DIMS[index][1] == 0 { 1 } else { 2 },
                    LENGTHS[index],
                )?);
            }
            let tensors: [Box<[f32]>; LAYER_TENSOR_COUNT] =
                tensors.try_into().map_err(|_| Error::TensorContract)?;
            layers.push(Layer { tensors });
        }
        Ok(Self {
            layers: layers.into_boxed_slice(),
        })
    }

    fn execute(
        &self,
        layer: usize,
        input: &[f32],
        output: &mut [f32],
        workspace: &mut TransformerLayerWorkspace,
    ) -> Result<(), Error> {
        if layer >= TRANSFORMER_LAYER_COUNT
            || input.len() != FRAME_COUNT * HIDDEN_DIM
            || output.len() != input.len()
        {
            return Err(Error::Shape);
        }
        let t = &self.layers[layer].tensors;
        let frame_values = FRAME_COUNT * HIDDEN_DIM;
        if !sortformer::transpose_dense_input(
            input,
            FRAME_COUNT,
            HIDDEN_DIM,
            &mut workspace.dense_transposed_input[..frame_values],
        ) {
            return Err(Error::Kernel);
        }
        for (weight, bias, projected) in [
            (&t[5], &t[4], &mut workspace.query),
            (&t[1], &t[0], &mut workspace.key),
            (&t[7], &t[6], &mut workspace.value),
        ] {
            if !sortformer::dense_batch_from_transposed(&mut sortformer::DenseBatchFromTransposed {
                transposed_input: &workspace.dense_transposed_input[..frame_values],
                row_count: FRAME_COUNT,
                input_dim: HIDDEN_DIM,
                weights: weight,
                bias,
                output_dim: HIDDEN_DIM,
                transposed_output: &mut workspace.dense_transposed_output[..frame_values],
                output_rows: &mut projected[..frame_values],
            }) {
                return Err(Error::Kernel);
            }
        }
        let ok = sortformer::transformer_batch_192(
            input,
            &workspace.query,
            &workspace.key,
            &workspace.value,
            &t[3],
            &t[2],
            &t[9],
            &t[8],
            &t[13],
            &t[12],
            &t[15],
            &t[14],
            &t[11],
            &t[10],
            FRAME_COUNT,
            &mut workspace.scores,
            &mut workspace.attended,
            &mut workspace.residual,
            &mut workspace.normalized,
            &mut workspace.feed_forward,
            &mut workspace.output,
            &mut workspace.dense_transposed_input,
            &mut workspace.dense_transposed_output,
            output,
        );
        if ok { Ok(()) } else { Err(Error::Kernel) }
    }

    #[cfg(test)]
    fn zero() -> Self {
        let mut layers = Vec::with_capacity(TRANSFORMER_LAYER_COUNT);
        for _ in 0..TRANSFORMER_LAYER_COUNT {
            let tensors = LENGTHS.map(|length| vec![0.0; length].into_boxed_slice());
            layers.push(Layer { tensors });
        }
        Self {
            layers: layers.into_boxed_slice(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_layer_is_numeric_and_rejects_hostile_shapes() {
        let binding = Binding::zero();
        let mut workspace = Workspace::default();
        let mut route = Route::new(&binding, &mut workspace);
        let input = vec![1.0; FRAME_COUNT * HIDDEN_DIM];
        let mut output = vec![f32::NAN; FRAME_COUNT * HIDDEN_DIM];
        route.execute(0, &input, &mut output).unwrap();
        assert!(output.iter().all(|value| *value == 0.0));
        assert_eq!(
            route.execute(TRANSFORMER_LAYER_COUNT, &input, &mut output),
            Err(Error::Shape)
        );
        assert_eq!(route.execute(0, &[], &mut output), Err(Error::Shape));
    }
}
