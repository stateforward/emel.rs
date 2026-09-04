//! Safe Moshi text-embedding model binding.
//!
//! The maintained binding follows the pinned Moshi tensor contract: `lm.text_emb.weight`
//! is the text embedding and `lm.emb.0.weight` is audio. This adapter consumes only
//! the explicit text tensor and does not infer or substitute an audio embedding.
//!
//! The binding is constructed once from model-owned storage. The resulting
//! row input only borrows that storage and creates kernel views; it performs no
//! allocation on the executor dispatch path.

use core::fmt;

use emel_kernels::any::get_rows::{IndexView, OpGetRowsF32Bytes};
use emel_kernels::any::tensor_view::{ByteTensorView, DType, Layout, TensorViewMut};
use emel_model::bridge::{MoshiComponent, MoshiLmBindingInput, TensorView};
use emel_tensor::dtype::SerializedType;

/// Name of the Moshi text embedding tensor consumed by this adapter.
pub const TEXT_EMBEDDING_TENSOR: &[u8] = b"lm.text_emb.weight";

/// Errors returned while binding the Moshi text embedding tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MoshiTextEmbeddingBindingError {
    /// The supplied model is not the Moshi architecture.
    WrongArchitecture,
    /// The supplied model is not a Moshi LM component.
    WrongComponent,
    /// The model's Moshi LM metadata is invalid.
    InvalidHParams,
    /// The required tensor name is absent.
    MissingTensor,
    /// Tensor metadata could not be decoded from the model-owned record.
    InvalidMetadata,
    /// Tensor bytes are not resident in model-owned storage.
    NonResidentTensor,
    /// Resident storage does not match the metadata-declared byte length.
    StorageLengthMismatch,
    /// The tensor's serialized dtype is not the supported F32 path.
    WrongDType(SerializedType),
    /// The tensor rank or dimensions do not match the Moshi row orientation.
    WrongShape { actual: [u64; 4] },
    /// The row destination does not have the exact embedding width.
    DestinationLengthMismatch,
    /// The row operation's layout arithmetic cannot be represented.
    LayoutCapacity,
}

impl fmt::Display for MoshiTextEmbeddingBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongArchitecture => formatter.write_str("wrong Moshi model architecture"),
            Self::WrongComponent => formatter.write_str("wrong Moshi model component"),
            Self::InvalidHParams => formatter.write_str("invalid Moshi LM metadata"),
            Self::MissingTensor => formatter.write_str("missing Moshi text embedding tensor"),
            Self::InvalidMetadata => formatter.write_str("invalid Moshi text embedding metadata"),
            Self::NonResidentTensor => {
                formatter.write_str("Moshi text embedding tensor is not resident")
            }
            Self::StorageLengthMismatch => {
                formatter.write_str("Moshi text embedding storage length mismatch")
            }
            Self::WrongDType(dtype) => write!(
                formatter,
                "unsupported Moshi text embedding dtype {dtype:?}"
            ),
            Self::WrongShape { actual } => {
                write!(formatter, "wrong Moshi text embedding shape {actual:?}")
            }
            Self::DestinationLengthMismatch => {
                formatter.write_str("Moshi text embedding row destination length mismatch")
            }
            Self::LayoutCapacity => {
                formatter.write_str("Moshi text embedding layout capacity overflow")
            }
        }
    }
}

impl std::error::Error for MoshiTextEmbeddingBindingError {}

/// A validated, immutable Moshi text embedding owned by a model `Data` value.
///
/// This type retains no model-private record or raw pointer. Its byte view is
/// borrowed from the model and therefore cannot outlive the model owner.
#[derive(Clone, Copy, Debug)]
pub struct MoshiTextEmbeddingBinding<'a> {
    source: &'a [u8],
    layout: Layout,
    dimension: usize,
    text_card: usize,
}

impl<'a> MoshiTextEmbeddingBinding<'a> {
    /// The supported path is intentionally F32-only.  F16, BF16, and all
    /// quantized representations are rejected rather than converted.
    ///
    /// # Errors
    ///
    /// Returns a typed error when architecture, component, metadata, tensor
    /// residency, dtype, shape, or storage contracts are invalid.
    pub fn try_from_lm(
        model: MoshiLmBindingInput<'a>,
    ) -> Result<Self, MoshiTextEmbeddingBindingError> {
        if model.architecture_name() != b"moshi" {
            return Err(MoshiTextEmbeddingBindingError::WrongArchitecture);
        }
        if model.component() != MoshiComponent::Lm {
            return Err(MoshiTextEmbeddingBindingError::WrongComponent);
        }
        let hparams = model.hparams();
        hparams
            .validate()
            .map_err(|_| MoshiTextEmbeddingBindingError::InvalidHParams)?;
        let dimension = usize::try_from(hparams.dim())
            .map_err(|_| MoshiTextEmbeddingBindingError::InvalidHParams)?;
        let text_card = usize::try_from(hparams.text_card())
            .map_err(|_| MoshiTextEmbeddingBindingError::InvalidHParams)?;
        let tensor = model
            .tensor_named(TEXT_EMBEDDING_TENSOR)
            .ok_or(MoshiTextEmbeddingBindingError::MissingTensor)?;
        Self::try_from_tensor(tensor, dimension, text_card)
    }

    fn try_from_tensor(
        tensor: TensorView<'a>,
        dimension: usize,
        text_card: usize,
    ) -> Result<Self, MoshiTextEmbeddingBindingError> {
        let metadata = tensor
            .metadata()
            .ok_or(MoshiTextEmbeddingBindingError::InvalidMetadata)?;
        if metadata.tensor_type() != SerializedType::F32 {
            return Err(MoshiTextEmbeddingBindingError::WrongDType(
                metadata.tensor_type(),
            ));
        }
        let expected_shape = [
            u64::try_from(dimension).map_err(|_| MoshiTextEmbeddingBindingError::InvalidHParams)?,
            u64::try_from(text_card).map_err(|_| MoshiTextEmbeddingBindingError::InvalidHParams)?,
            1,
            1,
        ];
        if metadata.dimension_count() != 2 || metadata.dimensions() != expected_shape {
            return Err(MoshiTextEmbeddingBindingError::WrongShape {
                actual: metadata.dimensions(),
            });
        }
        let expected_bytes = dimension
            .checked_mul(text_card)
            .and_then(|count| count.checked_mul(core::mem::size_of::<f32>()))
            .ok_or(MoshiTextEmbeddingBindingError::LayoutCapacity)?;
        if metadata.data_size() != u64::try_from(expected_bytes).unwrap_or(u64::MAX) {
            return Err(MoshiTextEmbeddingBindingError::StorageLengthMismatch);
        }
        let bytes = match tensor.byte_view() {
            Some(bytes) => bytes,
            None if metadata.storage().is_none() => {
                return Err(MoshiTextEmbeddingBindingError::NonResidentTensor);
            }
            None => return Err(MoshiTextEmbeddingBindingError::StorageLengthMismatch),
        };
        if bytes.len() != expected_bytes {
            return Err(MoshiTextEmbeddingBindingError::StorageLengthMismatch);
        }
        let layout = Layout::contiguous(DType::F32, expected_shape)
            .ok_or(MoshiTextEmbeddingBindingError::LayoutCapacity)?;
        Ok(Self {
            source: bytes,
            layout,
            dimension,
            text_card,
        })
    }

    /// Returns the embedding width in F32 values.
    #[must_use]
    pub const fn dimension(self) -> usize {
        self.dimension
    }

    /// Returns the number of rows accepted by the source tensor.
    #[must_use]
    pub const fn text_card(self) -> usize {
        self.text_card
    }

    /// Creates a typed row-gather input for a later executor transition.
    ///
    /// The returned value owns no storage and can be dispatched through
    /// [`emel_kernels::any::get_rows::GetRowsKernel::process_event`].
    ///
    /// # Errors
    ///
    /// Returns a typed error when the destination width or row-operation layout
    /// cannot satisfy the validated embedding contract.
    pub fn row<'b>(
        &'b self,
        index: &'b i32,
        destination: &'b mut [f32],
    ) -> Result<MoshiTextEmbeddingRow<'b>, MoshiTextEmbeddingBindingError> {
        if destination.len() != self.dimension {
            return Err(MoshiTextEmbeddingBindingError::DestinationLengthMismatch);
        }
        let destination_layout = Layout::contiguous(
            DType::F32,
            [u64::try_from(self.dimension).unwrap_or(u64::MAX), 1, 1, 1],
        )
        .ok_or(MoshiTextEmbeddingBindingError::LayoutCapacity)?;
        let destination = TensorViewMut::new(destination, destination_layout);
        let indices = IndexView::new(core::slice::from_ref(index), [1, 1, 1]);
        let source = ByteTensorView::new(self.source, self.layout);
        Ok(MoshiTextEmbeddingRow {
            operation: OpGetRowsF32Bytes::new(source, indices, destination),
        })
    }
}

/// Typed operation input handed to the later Moshi executor row transition.
#[derive(Debug)]
pub struct MoshiTextEmbeddingRow<'a> {
    operation: OpGetRowsF32Bytes<'a>,
}

impl<'a> MoshiTextEmbeddingRow<'a> {
    /// Converts this input into the maintained F32 byte-backed row operation.
    #[must_use]
    pub const fn into_kernel_operation(self) -> OpGetRowsF32Bytes<'a> {
        self.operation
    }
}

#[cfg(test)]
#[allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::cast_precision_loss,
    clippy::float_cmp
)]
mod tests {
    use super::super::executor::sm::{
        ExecutorError, NativeRowRun, SpeechPredictorMoshiNativeRowExecutor, process_native_row,
    };
    use super::*;
    use emel_kernels::any::Kernel;
    use emel_kernels::any::get_rows::GetRowsError;
    use emel_model::bridge::{
        Data, DataError, MoshiLmDataInput, MoshiLmHParams, MoshiLmHParamsInput, TensorBinding,
        TensorInput, TensorMetadata, TensorMetadataInput,
    };

    fn hparams() -> MoshiLmHParams {
        MoshiLmHParams::try_new(&MoshiLmHParamsInput {
            card: 32,
            n_q: 2,
            dep_q: 1,
            inference_dep_q: 1,
            text_card: 8,
            text_padding_id: 0,
            dim: 4,
            num_layers: 1,
            num_heads: 1,
            context: 8,
            max_period: 100,
            dim_feedforward: 4,
            depformer_dim: 4,
            depformer_num_heads: 1,
            depformer_num_layers: 1,
            depformer_dim_feedforward: 4,
            depformer_context: 8,
            depformer_max_period: 100,
            depformer_low_rank_embeddings: 0,
            extra_heads_num_heads: 1,
            inference_pre_text_silence_frames: 0,
            inference_post_text_silence_frames: 0,
            delay_count: 3,
            inference_prompt_token_count: 0,
            depformer_weight_schedule_count: 0,
            delays: [0; 64],
            inference_prompt_tokens: [0; 64],
            depformer_weight_schedule: [0; 64],
            causal: true,
            cross_attention: false,
            demux_second_stream: false,
            depformer_multi_linear: false,
            depformer_weights_per_step: false,
        })
        .unwrap()
    }

    fn model(dtype: SerializedType, shape: [u64; 4], payload: &[u8], resident: bool) -> Data {
        let metadata = TensorMetadata::new(TensorMetadataInput {
            tensor_type: dtype,
            dimension_count: 2,
            dimensions: shape,
            data_offset: 0,
            file_offset: 0,
            data_size: payload.len() as u64,
            file_index: 0,
            storage: resident.then(|| TensorBinding::new(0, 0, payload.len() as u64)),
        });
        let tensor = if resident {
            TensorInput::with_bytes(TEXT_EMBEDDING_TENSOR, metadata, payload)
        } else {
            TensorInput::new(TEXT_EMBEDDING_TENSOR, metadata)
        };
        Data::try_from_moshi_lm(MoshiLmDataInput {
            hparams: hparams(),
            tensors: &[tensor],
        })
        .unwrap()
    }

    #[test]
    fn valid_binding_dispatches_known_row_through_public_stage() {
        let mut payload = [0_u8; 4 * 8 * 4];
        for (index, value) in payload.chunks_exact_mut(4).enumerate() {
            value.copy_from_slice(&(index as f32).to_ne_bytes());
        }
        let data = model(SerializedType::F32, [4, 8, 1, 1], &payload, true);
        let binding =
            MoshiTextEmbeddingBinding::try_from_lm(data.moshi_lm_binding_input()).unwrap();
        let mut destination = [0.0_f32; 4];
        let index = 2;
        let operation = binding
            .row(&index, &mut destination)
            .unwrap()
            .into_kernel_operation();
        let mut kernel = Kernel::new();
        let run = NativeRowRun::new(operation, &mut kernel);
        let mut executor = SpeechPredictorMoshiNativeRowExecutor::new();
        assert_eq!(executor.process_row(&run), Ok(()));
        assert_eq!(destination, [8.0, 9.0, 10.0, 11.0]);
        assert!(matches!(
            executor.state(),
            &super::super::executor::sm::MoshiNativeRowExecutorStates::StateReady
        ));
        assert_eq!(executor.context().last_error(), ExecutorError::None);
    }

    #[test]
    fn native_row_operation_cannot_be_reused_silently() {
        let mut payload = [0_u8; 4 * 8 * 4];
        for (index, value) in payload.chunks_exact_mut(4).enumerate() {
            value.copy_from_slice(&(index as f32).to_ne_bytes());
        }
        let data = model(SerializedType::F32, [4, 8, 1, 1], &payload, true);
        let binding =
            MoshiTextEmbeddingBinding::try_from_lm(data.moshi_lm_binding_input()).unwrap();
        let mut destination = [0.0_f32; 4];
        let index = 2;
        let operation = binding
            .row(&index, &mut destination)
            .unwrap()
            .into_kernel_operation();
        let mut kernel = Kernel::new();
        let run = NativeRowRun::new(operation, &mut kernel);
        let mut executor = SpeechPredictorMoshiNativeRowExecutor::new();
        assert_eq!(executor.process_row(&run), Ok(()));
        assert_eq!(
            executor.process_row(&run),
            Err(ExecutorError::NativeRow(GetRowsError::UnexpectedEvent))
        );
        assert!(matches!(
            executor.state(),
            &super::super::executor::sm::MoshiNativeRowExecutorStates::StateError
        ));
        assert_eq!(destination, [8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn valid_binding_dispatches_known_row() {
        let mut payload = [0_u8; 4 * 8 * 4];
        for (index, value) in payload.chunks_exact_mut(4).enumerate() {
            value.copy_from_slice(&(index as f32).to_ne_bytes());
        }
        let data = model(SerializedType::F32, [4, 8, 1, 1], &payload, true);
        let binding =
            MoshiTextEmbeddingBinding::try_from_lm(data.moshi_lm_binding_input()).unwrap();
        let mut destination = [0.0_f32; 4];
        let index = 2;
        let operation = binding
            .row(&index, &mut destination)
            .unwrap()
            .into_kernel_operation();
        let mut kernel = Kernel::new();
        let run = NativeRowRun::new(operation, &mut kernel);
        assert_eq!(process_native_row(&run), Ok(()));
        assert_eq!(destination, [8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn invalid_native_row_publishes_kernel_error_without_writing_destination() {
        let mut payload = [0_u8; 4 * 8 * 4];
        for (index, value) in payload.chunks_exact_mut(4).enumerate() {
            value.copy_from_slice(&(index as f32).to_ne_bytes());
        }
        let data = model(SerializedType::F32, [4, 8, 1, 1], &payload, true);
        let binding =
            MoshiTextEmbeddingBinding::try_from_lm(data.moshi_lm_binding_input()).unwrap();
        let sentinel = -123.25_f32;
        let mut destination = [sentinel; 4];
        let index = -1;
        let operation = binding
            .row(&index, &mut destination)
            .unwrap()
            .into_kernel_operation();
        let mut kernel = Kernel::new();
        let run = NativeRowRun::new(operation, &mut kernel);
        assert_eq!(
            process_native_row(&run),
            Err(ExecutorError::NativeRow(GetRowsError::IndexOutOfBounds))
        );
        assert_eq!(destination, [sentinel; 4]);
    }

    #[test]
    fn old_audio_tensor_is_rejected_as_missing_text_tensor() {
        let payload = [0_u8; 4 * 8 * 4];
        let metadata = TensorMetadata::new(TensorMetadataInput {
            tensor_type: SerializedType::F32,
            dimension_count: 2,
            dimensions: [4, 8, 1, 1],
            data_offset: 0,
            file_offset: 0,
            data_size: payload.len() as u64,
            file_index: 0,
            storage: Some(TensorBinding::new(0, 0, payload.len() as u64)),
        });
        let tensor = TensorInput::with_bytes(b"lm.emb.0.weight", metadata, &payload);
        let data = Data::try_from_moshi_lm(MoshiLmDataInput {
            hparams: hparams(),
            tensors: &[tensor],
        })
        .unwrap();
        assert!(matches!(
            MoshiTextEmbeddingBinding::try_from_lm(data.moshi_lm_binding_input()),
            Err(MoshiTextEmbeddingBindingError::MissingTensor)
        ));
    }

    #[test]
    fn nonresident_tensor_is_rejected() {
        let payload = [0_u8; 4 * 8 * 4];
        let data = model(SerializedType::F32, [4, 8, 1, 1], &payload, false);
        assert!(matches!(
            MoshiTextEmbeddingBinding::try_from_lm(data.moshi_lm_binding_input()),
            Err(MoshiTextEmbeddingBindingError::NonResidentTensor)
        ));
    }

    #[test]
    fn wrong_dtype_and_shape_are_rejected() {
        let f16 = model(SerializedType::F16, [4, 8, 1, 1], &[0_u8; 4 * 8 * 2], true);
        assert!(matches!(
            MoshiTextEmbeddingBinding::try_from_lm(f16.moshi_lm_binding_input()),
            Err(MoshiTextEmbeddingBindingError::WrongDType(
                SerializedType::F16
            ))
        ));
        let shape = model(SerializedType::F32, [4, 7, 1, 1], &[0_u8; 4 * 7 * 4], true);
        assert!(matches!(
            MoshiTextEmbeddingBinding::try_from_lm(shape.moshi_lm_binding_input()),
            Err(MoshiTextEmbeddingBindingError::WrongShape { .. })
        ));
    }

    #[test]
    fn short_storage_is_rejected() {
        let payload = [0_u8; 4 * 8 * 4 - 1];
        let metadata = TensorMetadata::new(TensorMetadataInput {
            tensor_type: SerializedType::F32,
            dimension_count: 2,
            dimensions: [4, 8, 1, 1],
            data_offset: 0,
            file_offset: 0,
            data_size: (4 * 8 * 4) as u64,
            file_index: 0,
            storage: Some(TensorBinding::new(0, 0, (4 * 8 * 4) as u64)),
        });
        let result = Data::try_from_moshi_lm(MoshiLmDataInput {
            hparams: hparams(),
            tensors: &[TensorInput::with_bytes(
                TEXT_EMBEDDING_TENSOR,
                metadata,
                &payload,
            )],
        });
        assert!(matches!(result, Err(DataError::InvalidTensor)));
    }

    #[test]
    fn short_destination_is_rejected_without_dispatch() {
        let data = model(SerializedType::F32, [4, 8, 1, 1], &[0_u8; 4 * 8 * 4], true);
        let binding =
            MoshiTextEmbeddingBinding::try_from_lm(data.moshi_lm_binding_input()).unwrap();
        let index = 0;
        let mut destination = [0.0_f32; 3];
        assert!(matches!(
            binding.row(&index, &mut destination),
            Err(MoshiTextEmbeddingBindingError::DestinationLengthMismatch)
        ));
    }
}
