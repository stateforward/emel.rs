//! Safe, pre-dispatch Mimi model binding.
//!
//! This module owns only metadata validation and prepared runtime inputs. It
//! does not retain a GGUF loader, borrowed loader callback values, model file
//! bytes, arena slices, or codec state.

#![allow(
    clippy::struct_field_names,
    reason = "arena field names are the fixed public capacity contract"
)]

use core::fmt;

use emel_gguf::Loader;
use emel_gguf::event::{
    ParseDone, QueryError, ReadF32, ReadSigned, ReadStringInto, TensorDescriptor, WithTensor,
};
use emel_model::bridge::{MimiBindingInput, MimiHParamsError, MoshiComponent, TensorMetadata};
use emel_tensor::dtype::SerializedType;

/// The runtime operand class selected by the future Mimi facade.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeVariant {
    /// Use ordinary binary32 tensor operands.
    F32,
    /// Use native binary16 tensor operands.
    F16,
    /// Use the serialized Q8 tensor operand class.
    Q8,
}

impl RuntimeVariant {
    fn accepts(self, tensor_type: SerializedType, projection: bool) -> bool {
        match self {
            Self::F32 => tensor_type == SerializedType::F32,
            Self::F16 => tensor_type == SerializedType::F16,
            Self::Q8 if projection => tensor_type == SerializedType::Q8_0,
            Self::Q8 => matches!(tensor_type, SerializedType::F32 | SerializedType::F16),
        }
    }
}

/// Arena capacity supplied by the future facade before dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArenaCapacities {
    prepared_floats: usize,
    state_floats: usize,
    workspace_floats: usize,
    frame_floats: usize,
}

impl ArenaCapacities {
    /// Creates caller-owned capacities without allocating.
    #[must_use]
    pub const fn new(
        prepared_floats: usize,
        state_floats: usize,
        workspace_floats: usize,
        frame_floats: usize,
    ) -> Self {
        Self {
            prepared_floats,
            state_floats,
            workspace_floats,
            frame_floats,
        }
    }

    /// Returns the prepared-weight arena capacity.
    #[must_use]
    pub const fn prepared_floats(self) -> usize {
        self.prepared_floats
    }

    /// Returns the persistent-state arena capacity.
    #[must_use]
    pub const fn state_floats(self) -> usize {
        self.state_floats
    }

    /// Returns the workspace arena capacity.
    #[must_use]
    pub const fn workspace_floats(self) -> usize {
        self.workspace_floats
    }

    /// Returns the one-frame arena capacity.
    #[must_use]
    pub const fn frame_floats(self) -> usize {
        self.frame_floats
    }
}

/// Arena category used by a typed capacity failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArenaKind {
    /// Prepared-weight arena.
    Prepared,
    /// Persistent-state arena.
    State,
    /// Per-operation workspace arena.
    Workspace,
    /// One-frame arena.
    Frame,
}

/// Model metadata key used in a typed loader mismatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetadataKey {
    /// `general.architecture`.
    Architecture,
    /// `moshi.component`.
    Component,
    /// `moshi.mimi.sample_rate`.
    SampleRate,
    /// `moshi.mimi.frame_rate`.
    FrameRate,
    /// `moshi.mimi.n_q`.
    Nq,
    /// `moshi.mimi.card`.
    Card,
    /// `moshi.mimi.dim`.
    Dim,
    /// `moshi.mimi.semantic_n_q`.
    SemanticNq,
    /// `moshi.mimi.codebook_dim`.
    CodebookDim,
    /// `moshi.mimi.transformer.num_layers`.
    TransformerNumLayers,
    /// `moshi.mimi.transformer.num_heads`.
    TransformerNumHeads,
    /// `moshi.mimi.transformer.context`.
    TransformerContext,
    /// `moshi.mimi.transformer.max_period`.
    TransformerMaxPeriod,
}

impl MetadataKey {
    const fn bytes(self) -> &'static [u8] {
        match self {
            Self::Architecture => b"general.architecture",
            Self::Component => b"moshi.component",
            Self::SampleRate => b"moshi.mimi.sample_rate",
            Self::FrameRate => b"moshi.mimi.frame_rate",
            Self::Nq => b"moshi.mimi.n_q",
            Self::Card => b"moshi.mimi.card",
            Self::Dim => b"moshi.mimi.dim",
            Self::SemanticNq => b"moshi.mimi.semantic_n_q",
            Self::CodebookDim => b"moshi.mimi.codebook_dim",
            Self::TransformerNumLayers => b"moshi.mimi.transformer.num_layers",
            Self::TransformerNumHeads => b"moshi.mimi.transformer.num_heads",
            Self::TransformerContext => b"moshi.mimi.transformer.context",
            Self::TransformerMaxPeriod => b"moshi.mimi.transformer.max_period",
        }
    }
}

/// Mimi tensor family required by the model contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorFamily {
    /// `SEANet` encoder family.
    Encoder,
    /// Encoder transformer family.
    EncoderTransformer,
    /// Downsampler family.
    Downsample,
    /// Residual vector quantizer family.
    Quantizer,
    /// Upsampler family.
    Upsample,
    /// Decoder transformer family.
    DecoderTransformer,
    /// `SEANet` decoder family.
    Decoder,
}

impl TensorFamily {
    fn from_name(name: &[u8]) -> Option<Self> {
        let families = [
            (b"mimi.encoder.model.".as_slice(), Self::Encoder),
            (b"mimi.encoder_transformer.transformer.layers.".as_slice(), Self::EncoderTransformer),
            (b"mimi.downsample.conv.conv.conv.weight".as_slice(), Self::Downsample),
            (b"mimi.quantizer.rvq_first.".as_slice(), Self::Quantizer),
            (b"mimi.quantizer.rvq_rest.".as_slice(), Self::Quantizer),
            (b"mimi.upsample.convtr.convtr.convtr.weight".as_slice(), Self::Upsample),
            (b"mimi.decoder_transformer.transformer.layers.".as_slice(), Self::DecoderTransformer),
            (b"mimi.decoder.model.".as_slice(), Self::Decoder),
        ];
        families.iter().find_map(|(prefix, family)| name.starts_with(prefix).then_some(*family))
    }

    const fn index(self) -> usize {
        match self {
            Self::Encoder => 0,
            Self::EncoderTransformer => 1,
            Self::Downsample => 2,
            Self::Quantizer => 3,
            Self::Upsample => 4,
            Self::DecoderTransformer => 5,
            Self::Decoder => 6,
        }
    }
}

/// Failure while mapping public loader outcomes to a Mimi binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BindingError {
    /// The loader returned a typed query failure.
    LoaderQuery(QueryError),
    /// A required metadata key was absent.
    MissingMetadata(MetadataKey),
    /// A metadata value did not match the model-owned value.
    MetadataMismatch(MetadataKey),
    /// A metadata integer could not be represented by the model contract.
    MetadataRange(MetadataKey),
    /// The model architecture is not `moshi`.
    WrongArchitecture,
    /// The model component is not `mimi`.
    WrongComponent,
    /// The model-owned Mimi metadata is invalid.
    InvalidHParams(MimiHParamsError),
    /// The frame geometry is not the fixed 1,920-sample Mimi frame.
    InvalidFrameGeometry,
    /// Parsed and model-owned tensor counts differ.
    TensorCountMismatch,
    /// A populated model tensor record is malformed.
    InvalidTensor(u32),
    /// A tensor does not belong to one of the seven Mimi families.
    UnknownTensorFamily(u32),
    /// A required Mimi tensor family is absent.
    MissingTensorFamily(TensorFamily),
    /// A loader tensor name differs from model-owned storage.
    TensorNameMismatch(u32),
    /// Loader and model tensor shapes differ.
    TensorShapeMismatch(u32),
    /// Loader and model tensor dtypes differ or are not the selected class.
    TensorDtypeMismatch(u32),
    /// Loader and model tensor storage ranges differ.
    TensorStorageMismatch(u32),
    /// A selected runtime class is not supported by a tensor.
    UnsupportedRuntime(u32),
    /// A caller-owned arena is too small or empty.
    ArenaCapacity(ArenaKind),
}

impl fmt::Display for BindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoaderQuery(error) => error.fmt(formatter),
            Self::MissingMetadata(key) => write!(formatter, "missing Mimi metadata: {key:?}"),
            Self::MetadataMismatch(key) => write!(formatter, "Mimi metadata mismatch: {key:?}"),
            Self::MetadataRange(key) => write!(formatter, "Mimi metadata out of range: {key:?}"),
            Self::WrongArchitecture => formatter.write_str("model architecture is not moshi"),
            Self::WrongComponent => formatter.write_str("model component is not mimi"),
            Self::InvalidHParams(error) => error.fmt(formatter),
            Self::InvalidFrameGeometry => formatter.write_str("Mimi frame geometry is invalid"),
            Self::TensorCountMismatch => formatter.write_str("Mimi tensor count mismatch"),
            Self::InvalidTensor(index) => write!(formatter, "invalid Mimi tensor {index}"),
            Self::UnknownTensorFamily(index) => {
                write!(formatter, "unknown Mimi tensor family at {index}")
            }
            Self::MissingTensorFamily(family) => {
                write!(formatter, "missing Mimi tensor family: {family:?}")
            }
            Self::TensorNameMismatch(index) => {
                write!(formatter, "Mimi tensor name mismatch at {index}")
            }
            Self::TensorShapeMismatch(index) => {
                write!(formatter, "Mimi tensor shape mismatch at {index}")
            }
            Self::TensorDtypeMismatch(index) => {
                write!(formatter, "Mimi tensor dtype mismatch at {index}")
            }
            Self::TensorStorageMismatch(index) => {
                write!(formatter, "Mimi tensor storage mismatch at {index}")
            }
            Self::UnsupportedRuntime(index) => {
                write!(formatter, "Mimi runtime class is unsupported at {index}")
            }
            Self::ArenaCapacity(kind) => write!(formatter, "Mimi arena is too small: {kind:?}"),
        }
    }
}

impl std::error::Error for BindingError {}

/// Typed future-facing runtime input owned by the speech Mimi boundary.
#[derive(Clone, Copy, Debug)]
pub struct CodecRuntime<'a> {
    model: MimiBindingInput<'a>,
    variant: RuntimeVariant,
    arenas: ArenaCapacities,
    frame_samples: u32,
    n_q: u32,
}

impl<'a> CodecRuntime<'a> {
    /// Returns the immutable model input for a future facade.
    #[must_use]
    pub const fn model(self) -> MimiBindingInput<'a> {
        self.model
    }

    /// Returns the explicit runtime operand class.
    #[must_use]
    pub const fn variant(self) -> RuntimeVariant {
        self.variant
    }

    /// Returns caller-provided arena capacities.
    #[must_use]
    pub const fn arenas(self) -> ArenaCapacities {
        self.arenas
    }

    /// Returns the fixed frame sample count.
    #[must_use]
    pub const fn frame_samples(self) -> u32 {
        self.frame_samples
    }

    /// Returns the model's total quantizer level count.
    #[must_use]
    pub const fn n_q(self) -> u32 {
        self.n_q
    }
}

/// Immutable prepared binding returned by [`MimiBindingFactory`].
#[derive(Clone, Copy, Debug)]
pub struct PreparedMimiBinding<'a> {
    runtime: CodecRuntime<'a>,
}

impl<'a> PreparedMimiBinding<'a> {
    /// Returns the typed runtime input for a future Mimi facade.
    #[must_use]
    pub const fn codec_runtime(self) -> CodecRuntime<'a> {
        self.runtime
    }
}

/// Speech-owned Mimi binding/factory adapter.
#[derive(Clone, Copy, Debug, Default)]
pub struct MimiBindingFactory;

impl MimiBindingFactory {
    /// Creates the stateless factory.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Validates public GGUF outcomes against immutable model-owned inputs.
    ///
    /// All loader interaction is synchronous through its public
    /// [`Loader::process_event`] wrapper. The returned binding retains only an
    /// immutable model view and copied scalar capacities.
    ///
    /// # Errors
    ///
    /// Returns a typed metadata, tensor, runtime, loader-query, or arena
    /// validation error.
    pub fn prepare<'a>(
        &self,
        loader: &mut Loader,
        parsed: ParseDone,
        model: MimiBindingInput<'a>,
        arenas: ArenaCapacities,
        variant: RuntimeVariant,
    ) -> Result<PreparedMimiBinding<'a>, BindingError> {
        validate_model_metadata(loader, model)?;
        let frame_samples = model
            .hparams()
            .frame_samples()
            .filter(|samples| *samples == 1_920)
            .ok_or(BindingError::InvalidFrameGeometry)?;
        validate_arenas(arenas, model, frame_samples)?;
        let count = validate_model_tensors(model, variant)?;
        if parsed.tensor_count() != count {
            return Err(BindingError::TensorCountMismatch);
        }
        validate_loader_tensors(loader, model, count)?;
        Ok(PreparedMimiBinding {
            runtime: CodecRuntime {
                model,
                variant,
                arenas,
                frame_samples,
                n_q: u32::try_from(model.hparams().n_q())
                    .map_err(|_| BindingError::InvalidHParams(MimiHParamsError::NonPositive))?,
            },
        })
    }
}

/// Convenience wrapper for the speech-owned factory.
///
/// # Errors
///
/// Returns a typed metadata, tensor, runtime, loader-query, or arena
/// validation error.
pub fn prepare_mimi<'a>(
    loader: &mut Loader,
    parsed: ParseDone,
    model: MimiBindingInput<'a>,
    arenas: ArenaCapacities,
    variant: RuntimeVariant,
) -> Result<PreparedMimiBinding<'a>, BindingError> {
    MimiBindingFactory::new().prepare(loader, parsed, model, arenas, variant)
}

fn validate_model_metadata(
    loader: &mut Loader,
    model: MimiBindingInput<'_>,
) -> Result<(), BindingError> {
    if model.architecture_name() != b"moshi" {
        return Err(BindingError::WrongArchitecture);
    }
    if model.component() != MoshiComponent::Mimi {
        return Err(BindingError::WrongComponent);
    }
    let hparams = *model.hparams();
    hparams.validate().map_err(BindingError::InvalidHParams)?;
    let mut architecture = [0_u8; 64];
    let architecture_length = read_string(loader, MetadataKey::Architecture, &mut architecture)?
        .ok_or(BindingError::MissingMetadata(MetadataKey::Architecture))?;
    if architecture[..architecture_length] != b"moshi"[..] {
        return Err(BindingError::MetadataMismatch(MetadataKey::Architecture));
    }
    let mut component = [0_u8; 16];
    let component_length = read_string(loader, MetadataKey::Component, &mut component)?
        .ok_or(BindingError::MissingMetadata(MetadataKey::Component))?;
    if component[..component_length] != b"mimi"[..] {
        return Err(BindingError::MetadataMismatch(MetadataKey::Component));
    }
    compare_signed(
        loader,
        MetadataKey::SampleRate,
        i64::from(hparams.sample_rate()),
    )?;
    compare_float(loader, MetadataKey::FrameRate, hparams.frame_rate())?;
    compare_signed(loader, MetadataKey::Nq, i64::from(hparams.n_q()))?;
    compare_signed(loader, MetadataKey::Card, i64::from(hparams.card()))?;
    compare_signed(loader, MetadataKey::Dim, i64::from(hparams.dim()))?;
    compare_signed(
        loader,
        MetadataKey::SemanticNq,
        i64::from(hparams.semantic_n_q()),
    )?;
    compare_signed(
        loader,
        MetadataKey::CodebookDim,
        i64::from(hparams.codebook_dim()),
    )?;
    compare_signed(
        loader,
        MetadataKey::TransformerNumLayers,
        i64::from(hparams.transformer_num_layers()),
    )?;
    compare_signed(
        loader,
        MetadataKey::TransformerNumHeads,
        i64::from(hparams.transformer_num_heads()),
    )?;
    compare_signed(
        loader,
        MetadataKey::TransformerContext,
        i64::from(hparams.transformer_context()),
    )?;
    compare_signed(
        loader,
        MetadataKey::TransformerMaxPeriod,
        i64::from(hparams.transformer_max_period()),
    )?;
    Ok(())
}

fn validate_arenas(
    arenas: ArenaCapacities,
    model: MimiBindingInput<'_>,
    frame_samples: u32,
) -> Result<(), BindingError> {
    let dimension = usize::try_from(model.hparams().dim())
        .map_err(|_| BindingError::InvalidHParams(MimiHParamsError::NonPositive))?;
    if arenas.prepared_floats == 0 || arenas.prepared_floats < dimension {
        return Err(BindingError::ArenaCapacity(ArenaKind::Prepared));
    }
    if arenas.state_floats == 0 || arenas.state_floats < dimension {
        return Err(BindingError::ArenaCapacity(ArenaKind::State));
    }
    if arenas.workspace_floats == 0 || arenas.workspace_floats < dimension {
        return Err(BindingError::ArenaCapacity(ArenaKind::Workspace));
    }
    if arenas.frame_floats < usize::try_from(frame_samples).unwrap_or(usize::MAX) {
        return Err(BindingError::ArenaCapacity(ArenaKind::Frame));
    }
    Ok(())
}

fn validate_model_tensors(
    model: MimiBindingInput<'_>,
    variant: RuntimeVariant,
) -> Result<u32, BindingError> {
    let mut families = [false; 7];
    let mut saw_q8 = false;
    let count = model.tensor_count();
    if count == 0 {
        return Err(BindingError::TensorCountMismatch);
    }
    for index in 0..count {
        let tensor = model
            .tensor(index)
            .ok_or(BindingError::InvalidTensor(index))?;
        let metadata = tensor
            .metadata()
            .ok_or(BindingError::InvalidTensor(index))?;
        validate_tensor_metadata(
            index,
            tensor.name(),
            metadata,
            variant,
            &mut families,
            &mut saw_q8,
        )?;
    }
    if variant == RuntimeVariant::Q8 && !saw_q8 {
        return Err(BindingError::UnsupportedRuntime(0));
    }
    for family in [
        TensorFamily::Encoder,
        TensorFamily::EncoderTransformer,
        TensorFamily::Downsample,
        TensorFamily::Quantizer,
        TensorFamily::Upsample,
        TensorFamily::DecoderTransformer,
        TensorFamily::Decoder,
    ] {
        if !families[family.index()] {
            return Err(BindingError::MissingTensorFamily(family));
        }
    }
    Ok(count)
}

fn validate_tensor_metadata(
    index: u32,
    name: &[u8],
    metadata: TensorMetadata,
    variant: RuntimeVariant,
    families: &mut [bool; 7],
    saw_q8: &mut bool,
) -> Result<(), BindingError> {
    let family = TensorFamily::from_name(name).ok_or(BindingError::UnknownTensorFamily(index))?;
    families[family.index()] = true;
    let dimensions = metadata.dimensions();
    let dimension_count = usize::try_from(metadata.dimension_count())
        .map_err(|_| BindingError::InvalidTensor(index))?;
    if !(1..=4).contains(&dimension_count)
        || dimensions[..dimension_count].contains(&0)
        || dimensions[dimension_count..]
            .iter()
            .any(|dimension| *dimension != 1)
    {
        return Err(BindingError::InvalidTensor(index));
    }
    let Some(storage) = metadata.storage() else {
        return Err(BindingError::TensorStorageMismatch(index));
    };
    if metadata.data_size() == 0 || storage.length() != metadata.data_size() {
        return Err(BindingError::TensorStorageMismatch(index));
    }
    if storage.offset() != metadata.file_offset() || storage.split_index() != metadata.file_index()
    {
        return Err(BindingError::TensorStorageMismatch(index));
    }
    storage
        .offset()
        .checked_add(storage.length())
        .ok_or(BindingError::TensorStorageMismatch(index))?;
    let projection = name.ends_with(b"in_projs.0.weight")
        || name.ends_with(b"out_projs.0.weight")
        || name.ends_with(b"linear1.weight")
        || name.ends_with(b"linear2.weight")
        || name.ends_with(b"input_proj.weight")
        || name.ends_with(b"output_proj.weight");
    if !variant.accepts(metadata.tensor_type(), projection) {
        return Err(BindingError::UnsupportedRuntime(index));
    }
    if metadata.tensor_type() == SerializedType::Q8_0 {
        *saw_q8 = true;
    }
    Ok(())
}

fn validate_loader_tensors(
    loader: &mut Loader,
    model: MimiBindingInput<'_>,
    count: u32,
) -> Result<(), BindingError> {
    for index in 0..count {
        let model_tensor = model
            .tensor(index)
            .ok_or(BindingError::InvalidTensor(index))?;
        let model_metadata = model_tensor
            .metadata()
            .ok_or(BindingError::InvalidTensor(index))?;
        let model_name = model_tensor.name();
        let observed = loader
            .process_event(WithTensor::new(
                index,
                |name: &[u8], descriptor: TensorDescriptor, bytes: &[u8]| {
                    tensor_mismatch(model_name, model_metadata, name, descriptor, bytes.len())
                },
            ))
            .map_err(BindingError::LoaderQuery)?;
        match observed {
            Some(None) => {}
            Some(Some(mismatch)) => {
                return Err(match mismatch {
                    TensorMismatch::Name => BindingError::TensorNameMismatch(index),
                    TensorMismatch::Shape => BindingError::TensorShapeMismatch(index),
                    TensorMismatch::Dtype => BindingError::TensorDtypeMismatch(index),
                    TensorMismatch::Storage => BindingError::TensorStorageMismatch(index),
                });
            }
            None => return Err(BindingError::TensorCountMismatch),
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TensorMismatch {
    Name,
    Shape,
    Dtype,
    Storage,
}

fn tensor_mismatch(
    model_name: &[u8],
    model_metadata: TensorMetadata,
    name: &[u8],
    descriptor: TensorDescriptor,
    bytes_len: usize,
) -> Option<TensorMismatch> {
    let Some(storage) = model_metadata.storage() else {
        return Some(TensorMismatch::Storage);
    };
    if model_name != name {
        return Some(TensorMismatch::Name);
    }
    if model_metadata.tensor_type() != descriptor.tensor_type() {
        return Some(TensorMismatch::Dtype);
    }
    if model_metadata.dimension_count() != descriptor.dimension_count()
        || model_metadata.dimensions() != descriptor.dimensions()
    {
        return Some(TensorMismatch::Shape);
    }
    if model_metadata.data_offset() != descriptor.data_offset()
        || model_metadata.data_size() != descriptor.data_size()
    {
        return Some(TensorMismatch::Storage);
    }
    if storage.split_index() != descriptor.file_index()
        || storage.offset() != descriptor.file_offset()
        || storage.length() != descriptor.data_size()
    {
        return Some(TensorMismatch::Storage);
    }
    if u64::try_from(bytes_len).ok() != Some(descriptor.data_size()) {
        return Some(TensorMismatch::Storage);
    }
    None
}

fn read_string<const N: usize>(
    loader: &mut Loader,
    key: MetadataKey,
    destination: &mut [u8; N],
) -> Result<Option<usize>, BindingError> {
    loader
        .process_event(ReadStringInto::new(key.bytes(), destination))
        .map_err(BindingError::LoaderQuery)
}

fn compare_signed(
    loader: &mut Loader,
    key: MetadataKey,
    expected: i64,
) -> Result<(), BindingError> {
    let observed = loader
        .process_event(ReadSigned::new(key.bytes()))
        .map_err(BindingError::LoaderQuery)?
        .ok_or(BindingError::MissingMetadata(key))?;
    if observed != expected {
        return Err(BindingError::MetadataMismatch(key));
    }
    Ok(())
}

fn compare_float(loader: &mut Loader, key: MetadataKey, expected: f32) -> Result<(), BindingError> {
    let observed = loader
        .process_event(ReadF32::new(key.bytes()))
        .map_err(BindingError::LoaderQuery)?
        .ok_or(BindingError::MissingMetadata(key))?;
    if observed.to_bits() != expected.to_bits() {
        return Err(BindingError::MetadataMismatch(key));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use emel_model::bridge::{
        MimiDataInput, MimiHParams, MimiHParamsInput, TensorBinding, TensorInput,
    };

    fn hparams() -> MimiHParams {
        MimiHParams::try_new(MimiHParamsInput {
            sample_rate: 24_000,
            frame_rate: 12.5,
            n_q: 2,
            card: 32,
            dim: 16,
            semantic_n_q: 1,
            codebook_dim: 8,
            transformer_num_layers: 2,
            transformer_num_heads: 2,
            transformer_context: 8,
            transformer_max_period: 1_000,
        })
        .unwrap()
    }

    fn tensor(name: &'static [u8], tensor_type: SerializedType) -> TensorInput<'static> {
        TensorInput::new(
            name,
            TensorMetadata::new(emel_model::bridge::TensorMetadataInput {
                tensor_type,
                dimension_count: 1,
                dimensions: [16, 1, 1, 1],
                data_offset: 0,
                file_offset: 0,
                data_size: 64,
                file_index: 0,
                storage: Some(TensorBinding::new(0, 0, 64)),
            }),
        )
    }

    fn model() -> emel_model::bridge::Data {
        let tensors = [
            tensor(b"mimi.encoder.weight", SerializedType::F32),
            tensor(b"mimi.encoder_transformer.weight", SerializedType::F32),
            tensor(b"mimi.downsample.weight", SerializedType::F32),
            tensor(b"mimi.quantizer.weight", SerializedType::F32),
            tensor(b"mimi.upsample.weight", SerializedType::F32),
            tensor(b"mimi.decoder_transformer.weight", SerializedType::F32),
            tensor(b"mimi.decoder.weight", SerializedType::F32),
        ];
        emel_model::bridge::Data::try_from_mimi(MimiDataInput {
            hparams: hparams(),
            tensors: &tensors,
        })
        .unwrap()
    }

    #[test]
    fn rejects_unparsed_loader_through_typed_outcome() {
        let data = model();
        let mut loader = Loader::new();
        let result = prepare_mimi(
            &mut loader,
            ParseDone::default(),
            data.mimi_binding_input(),
            ArenaCapacities::new(16, 16, 16, 1_920),
            RuntimeVariant::F32,
        );
        assert!(matches!(
            result,
            Err(BindingError::LoaderQuery(QueryError::NotParsed))
        ));
    }

    #[test]
    fn model_input_is_immutable_and_exposes_prepared_scalar_contract() {
        let data = model();
        let input = data.mimi_binding_input();
        assert_eq!(input.architecture_name(), b"moshi");
        assert_eq!(input.component(), MoshiComponent::Mimi);
        assert_eq!(input.hparams().frame_samples(), Some(1_920));
        assert_eq!(input.tensor_count(), 7);
        assert_eq!(input.tensor(0).unwrap().name(), b"mimi.encoder.weight");
    }
}
