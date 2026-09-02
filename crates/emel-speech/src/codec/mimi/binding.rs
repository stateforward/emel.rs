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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RoleKind {
    ConvWeight,
    ConvBias,
    ConvTransposeWeight,
    ConvTransposeBias,
    Residual1Weight,
    Residual1Bias,
    Residual3Weight,
    Residual3Bias,
    Norm1Weight,
    Norm1Bias,
    AttentionInputProjection,
    AttentionOutputProjection,
    LayerScale1,
    Norm2Weight,
    Norm2Bias,
    Linear1,
    Linear2,
    LayerScale2,
    QuantizerInputProjection,
    QuantizerOutputProjection,
    Codebook,
    SingleWeight,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TensorRole {
    family: TensorFamily,
    kind: RoleKind,
    module: u16,
    split: u8,
    level: u16,
}

fn parse_decimal_segment(input: &[u8]) -> Option<(u16, &[u8])> {
    let split = input.iter().position(|byte| *byte == b'.')?;
    let (digits, rest) = input.split_at(split);
    let rest = rest.get(1..)?;
    // The pinned converter emits decimal indices with no leading zeroes. An
    // indexed name is therefore an exact slot identity, not a loose numeric
    // prefix; accepting `layers.01` would admit a name the runtime never
    // binds.
    if digits.is_empty()
        || (digits.len() > 1 && digits[0] == b'0')
        || !digits.iter().all(u8::is_ascii_digit)
    {
        return None;
    }
    let mut value = 0_u16;
    for byte in digits {
        value = value
            .checked_mul(10)?
            .checked_add(u16::from(*byte - b'0'))?;
    }
    Some((value, rest))
}

impl TensorRole {
    fn parse_seanet(name: &[u8], family: TensorFamily, prefix: &[u8]) -> Option<Self> {
        let rest = name.strip_prefix(prefix)?;
        let (module, suffix) = parse_decimal_segment(rest)?;
        let kind = match suffix {
            b"conv.conv.weight" => RoleKind::ConvWeight,
            b"conv.conv.bias" => RoleKind::ConvBias,
            b"convtr.convtr.weight" => RoleKind::ConvTransposeWeight,
            b"convtr.convtr.bias" => RoleKind::ConvTransposeBias,
            b"block.1.conv.conv.weight" => RoleKind::Residual1Weight,
            b"block.1.conv.conv.bias" => RoleKind::Residual1Bias,
            b"block.3.conv.conv.weight" => RoleKind::Residual3Weight,
            b"block.3.conv.conv.bias" => RoleKind::Residual3Bias,
            _ => return None,
        };
        Some(Self {
            family,
            kind,
            module,
            split: 0,
            level: 0,
        })
    }
    fn parse_transformer(name: &[u8], family: TensorFamily, prefix: &[u8]) -> Option<Self> {
        let rest = name.strip_prefix(prefix)?;
        let (module, suffix) = parse_decimal_segment(rest)?;
        let kind = match suffix {
            b"norm1.weight" => RoleKind::Norm1Weight,
            b"norm1.bias" => RoleKind::Norm1Bias,
            b"self_attn.in_projs.0.weight" => RoleKind::AttentionInputProjection,
            b"self_attn.out_projs.0.weight" => RoleKind::AttentionOutputProjection,
            b"layer_scale_1.scale" => RoleKind::LayerScale1,
            b"norm2.weight" => RoleKind::Norm2Weight,
            b"norm2.bias" => RoleKind::Norm2Bias,
            b"linear1.weight" => RoleKind::Linear1,
            b"linear2.weight" => RoleKind::Linear2,
            b"layer_scale_2.scale" => RoleKind::LayerScale2,
            _ => return None,
        };
        Some(Self {
            family,
            kind,
            module,
            split: 0,
            level: 0,
        })
    }
    fn parse(name: &[u8]) -> Option<Self> {
        if let Some(role) = Self::parse_seanet(name, TensorFamily::Encoder, b"mimi.encoder.model.")
        {
            return Some(role);
        }
        if let Some(role) = Self::parse_seanet(name, TensorFamily::Decoder, b"mimi.decoder.model.")
        {
            return Some(role);
        }
        if let Some(role) = Self::parse_transformer(
            name,
            TensorFamily::EncoderTransformer,
            b"mimi.encoder_transformer.transformer.layers.",
        ) {
            return Some(role);
        }
        if let Some(role) = Self::parse_transformer(
            name,
            TensorFamily::DecoderTransformer,
            b"mimi.decoder_transformer.transformer.layers.",
        ) {
            return Some(role);
        }
        if name == b"mimi.downsample.conv.conv.conv.weight"
            || name == b"mimi.upsample.convtr.convtr.convtr.weight"
        {
            return Some(Self {
                family: if name == b"mimi.downsample.conv.conv.conv.weight" {
                    TensorFamily::Downsample
                } else {
                    TensorFamily::Upsample
                },
                kind: RoleKind::SingleWeight,
                module: 0,
                split: 0,
                level: 0,
            });
        }
        let rest = name.strip_prefix(b"mimi.quantizer.")?;
        let (split, rest) = if let Some(rest) = rest.strip_prefix(b"rvq_first") {
            (0, rest)
        } else {
            (1, rest.strip_prefix(b"rvq_rest")?)
        };
        if rest == b".input_proj.weight" {
            return Some(Self {
                family: TensorFamily::Quantizer,
                kind: RoleKind::QuantizerInputProjection,
                module: 0,
                split,
                level: 0,
            });
        }
        if rest == b".output_proj.weight" {
            return Some(Self {
                family: TensorFamily::Quantizer,
                kind: RoleKind::QuantizerOutputProjection,
                module: 0,
                split,
                level: 0,
            });
        }
        let rest = rest.strip_prefix(b".vq.layers.")?;
        let (level, suffix) = parse_decimal_segment(rest)?;
        (suffix == b"_codebook.embedding").then_some(Self {
            family: TensorFamily::Quantizer,
            kind: RoleKind::Codebook,
            module: 0,
            split,
            level,
        })
    }
}
fn role_is_reachable(role: TensorRole, hparams: &emel_model::bridge::MimiHParams) -> bool {
    match role.family {
        TensorFamily::Encoder => match role.module {
            0 | 3 | 6 | 9 | 12 | 14 => {
                matches!(role.kind, RoleKind::ConvWeight | RoleKind::ConvBias)
            }
            1 | 4 | 7 | 10 => matches!(
                role.kind,
                RoleKind::Residual1Weight
                    | RoleKind::Residual1Bias
                    | RoleKind::Residual3Weight
                    | RoleKind::Residual3Bias
            ),
            _ => false,
        },
        TensorFamily::Decoder => match role.module {
            0 | 14 => matches!(role.kind, RoleKind::ConvWeight | RoleKind::ConvBias),
            2 | 5 | 8 | 11 => matches!(
                role.kind,
                RoleKind::ConvTransposeWeight | RoleKind::ConvTransposeBias
            ),
            3 | 6 | 9 | 12 => matches!(
                role.kind,
                RoleKind::Residual1Weight
                    | RoleKind::Residual1Bias
                    | RoleKind::Residual3Weight
                    | RoleKind::Residual3Bias
            ),
            _ => false,
        },
        TensorFamily::EncoderTransformer | TensorFamily::DecoderTransformer => {
            role.module < u16::try_from(hparams.transformer_num_layers()).unwrap_or(0)
        }
        TensorFamily::Downsample | TensorFamily::Upsample => {
            role.module == 0 && role.kind == RoleKind::SingleWeight
        }
        TensorFamily::Quantizer => {
            if role.module != 0 {
                return false;
            }
            match role.kind {
                RoleKind::QuantizerInputProjection | RoleKind::QuantizerOutputProjection => {
                    role.split < 2
                }
                RoleKind::Codebook => {
                    let levels = if role.split == 0 {
                        hparams.semantic_n_q()
                    } else if role.split == 1 {
                        hparams.n_q() - hparams.semantic_n_q()
                    } else {
                        return false;
                    };
                    i32::from(role.level) < levels
                }
                _ => false,
            }
        }
    }
}

impl RuntimeVariant {
    fn accepts(self, role: TensorRole, tensor_type: SerializedType) -> bool {
        // Decoder upsample weights are canonicalized into F32 before any
        // dispatch. This keeps the native stage's weight view allocation-free.
        if role.family == TensorFamily::Upsample && role.kind == RoleKind::SingleWeight {
            return tensor_type == SerializedType::F32;
        }
        let is_float = matches!(tensor_type, SerializedType::F32 | SerializedType::F16);
        let transformer_projection = matches!(
            role.kind,
            RoleKind::AttentionInputProjection
                | RoleKind::AttentionOutputProjection
                | RoleKind::Linear1
                | RoleKind::Linear2
        );
        let quantizer_projection = matches!(
            role.kind,
            RoleKind::QuantizerInputProjection | RoleKind::QuantizerOutputProjection
        );
        let projection = transformer_projection || quantizer_projection;
        let non_transposed_convolution = matches!(
            role.kind,
            RoleKind::ConvWeight | RoleKind::Residual1Weight | RoleKind::Residual3Weight
        ) || (role.family == TensorFamily::Downsample
            && role.kind == RoleKind::SingleWeight);
        match self {
            Self::F32 => tensor_type == SerializedType::F32,
            Self::F16 => {
                if non_transposed_convolution || quantizer_projection {
                    tensor_type == SerializedType::F16
                } else {
                    is_float
                }
            }
            Self::Q8 => {
                if projection {
                    tensor_type == SerializedType::Q8_0
                } else {
                    is_float
                }
            }
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

/// Canonical F32 decoder upsample weights and fixed convolution geometry.
#[derive(Clone, Copy, Debug)]
pub struct UpsampleBinding<'a> {
    bytes: &'a [u8],
    dim: usize,
    taps: usize,
}

impl UpsampleBinding<'_> {
    #[must_use]
    pub const fn dim(self) -> usize {
        self.dim
    }
    #[must_use]
    pub const fn taps(self) -> usize {
        self.taps
    }
    #[must_use]
    pub fn weight(self, tap: usize, channel: usize) -> Option<f32> {
        if tap >= self.taps || channel >= self.dim {
            return None;
        }
        let index = tap.checked_mul(self.dim)?.checked_add(channel)?;
        let start = index.checked_mul(4)?;
        let bytes = self.bytes.get(start..start.checked_add(4)?)?;
        Some(f32::from_ne_bytes(bytes.try_into().ok()?))
    }
}

#[cfg(test)]
impl<'a> UpsampleBinding<'a> {
    pub(crate) const fn for_test(bytes: &'a [u8], dim: usize, taps: usize) -> Self {
        Self { bytes, dim, taps }
    }
}

/// Typed future-facing runtime input owned by the speech Mimi boundary.
#[derive(Clone, Copy, Debug)]
pub struct CodecRuntime<'a> {
    model: MimiBindingInput<'a>,
    variant: RuntimeVariant,
    arenas: ArenaCapacities,
    frame_samples: u32,
    n_q: u32,
    upsample: UpsampleBinding<'a>,
}

impl<'a> CodecRuntime<'a> {
    #[must_use]
    pub const fn model(self) -> MimiBindingInput<'a> {
        self.model
    }
    #[must_use]
    pub const fn variant(self) -> RuntimeVariant {
        self.variant
    }
    #[must_use]
    pub const fn arenas(self) -> ArenaCapacities {
        self.arenas
    }
    #[must_use]
    pub const fn frame_samples(self) -> u32 {
        self.frame_samples
    }
    #[must_use]
    pub const fn n_q(self) -> u32 {
        self.n_q
    }
    #[must_use]
    pub const fn upsample(self) -> UpsampleBinding<'a> {
        self.upsample
    }
}

#[cfg(test)]
impl<'a> CodecRuntime<'a> {
    pub(crate) const fn for_test(
        model: MimiBindingInput<'a>,
        arenas: ArenaCapacities,
        upsample: UpsampleBinding<'a>,
    ) -> Self {
        Self {
            model,
            variant: RuntimeVariant::F32,
            arenas,
            frame_samples: 1_920,
            n_q: 1,
            upsample,
        }
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
        let count = validate_model_tensors(model, variant)?;
        validate_arenas(arenas, model, frame_samples)?;
        if parsed.tensor_count() != count {
            return Err(BindingError::TensorCountMismatch);
        }
        validate_loader_tensors(loader, model, count)?;
        let upsample = role_tensor(
            model,
            TensorRole {
                family: TensorFamily::Upsample,
                kind: RoleKind::SingleWeight,
                module: 0,
                split: 0,
                level: 0,
            },
        )?;
        let bytes = model
            .tensor(upsample.0)
            .and_then(emel_model::bridge::TensorView::byte_view)
            .ok_or(BindingError::TensorStorageMismatch(upsample.0))?;
        let dimension = usize::try_from(model.hparams().dim())
            .map_err(|_| BindingError::InvalidHParams(MimiHParamsError::NonPositive))?;
        let taps = usize::try_from(upsample.1.dimensions()[0])
            .map_err(|_| BindingError::TensorShapeMismatch(upsample.0))?;
        let n_q = u32::try_from(model.hparams().n_q())
            .map_err(|_| BindingError::InvalidHParams(MimiHParamsError::NonPositive))?;
        Ok(PreparedMimiBinding {
            runtime: CodecRuntime {
                model,
                variant,
                arenas,
                frame_samples,
                n_q,
                upsample: UpsampleBinding {
                    bytes,
                    dim: dimension,
                    taps,
                },
            },
        })
    }
}

/// Convenience wrapper for the speech-owned factory.
///
/// # Errors
///
/// Returns a typed metadata, tensor, runtime, loader-query, or arena validation
/// error when the model or loader does not satisfy the Mimi contract.
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

fn validate_model_tensors(
    model: MimiBindingInput<'_>,
    variant: RuntimeVariant,
) -> Result<u32, BindingError> {
    let (count, seen) = validate_model_tensor_records(model, variant)?;
    for family in [
        TensorFamily::Encoder,
        TensorFamily::EncoderTransformer,
        TensorFamily::Downsample,
        TensorFamily::Quantizer,
        TensorFamily::Upsample,
        TensorFamily::DecoderTransformer,
        TensorFamily::Decoder,
    ] {
        if !seen[family.index()] {
            return Err(BindingError::MissingTensorFamily(family));
        }
    }
    let h = model.hparams();
    let dim = u64::try_from(h.dim())
        .map_err(|_| BindingError::InvalidHParams(MimiHParamsError::NonPositive))?;
    let codebook_dim = u64::try_from(h.codebook_dim())
        .map_err(|_| BindingError::InvalidHParams(MimiHParamsError::NonPositive))?;
    let card = u64::try_from(h.card())
        .map_err(|_| BindingError::InvalidHParams(MimiHParamsError::NonPositive))?;
    let layers = u16::try_from(h.transformer_num_layers())
        .map_err(|_| BindingError::InvalidHParams(MimiHParamsError::NonPositive))?;
    if layers == 0 || layers > 16 || dim > (1 << 16) || codebook_dim > (1 << 16) || card > (1 << 16)
    {
        return Err(BindingError::InvalidHParams(MimiHParamsError::NonPositive));
    }
    validate_transformer_families(model, dim, layers, variant)?;
    validate_seanet_family(model, TensorFamily::Encoder, dim, 1_920, 2, false)?;
    // The fixed encoder stride chain produces two transformer tokens from a
    // 1,920-sample frame; downsample then reduces those two tokens to one.
    validate_downsample_upsample(model, dim, 2)?;
    validate_quantizer(model, *h, dim, codebook_dim, card, variant)?;
    Ok(count)
}

fn validate_model_tensor_records(
    model: MimiBindingInput<'_>,
    variant: RuntimeVariant,
) -> Result<(u32, [bool; 7]), BindingError> {
    let count = model.tensor_count();
    if count == 0 {
        return Err(BindingError::TensorCountMismatch);
    }
    let mut seen = [false; 7];
    for index in 0..count {
        let tensor = model
            .tensor(index)
            .ok_or(BindingError::InvalidTensor(index))?;
        let metadata = tensor
            .metadata()
            .ok_or(BindingError::InvalidTensor(index))?;
        // The pinned prepare walk reads every bound tensor payload. A metadata
        // record with no resident bytes is therefore not execution-bindable,
        // even when its declared shape and storage range look valid.
        if tensor.byte_view().is_none() {
            return Err(BindingError::TensorStorageMismatch(index));
        }
        let role =
            TensorRole::parse(tensor.name()).ok_or(BindingError::UnknownTensorFamily(index))?;
        let h = model.hparams();
        if !role_is_reachable(role, h) {
            return Err(BindingError::TensorCountMismatch);
        }
        // Every reachable C++ bind slot is a unique exact tensor name. A
        // duplicate role would otherwise be silently shadowed by role_tensor,
        // allowing an unreachable extra record to pass the family walk.
        for prior in 0..index {
            let prior_tensor = model
                .tensor(prior)
                .ok_or(BindingError::InvalidTensor(prior))?;
            let prior_role = TensorRole::parse(prior_tensor.name())
                .ok_or(BindingError::UnknownTensorFamily(prior))?;
            if prior_role == role {
                return Err(BindingError::TensorCountMismatch);
            }
        }
        seen[role.family.index()] = true;
        validate_tensor_metadata(index, role, metadata, variant)?;
    }
    Ok((count, seen))
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

fn validate_transformer_families(
    model: MimiBindingInput<'_>,
    dim: u64,
    layers: u16,
    variant: RuntimeVariant,
) -> Result<(), BindingError> {
    for family in [
        TensorFamily::EncoderTransformer,
        TensorFamily::DecoderTransformer,
    ] {
        let mut mlp_dim = 0_u64;
        for module in 0..layers {
            validate_transformer_module(model, family, module, dim, &mut mlp_dim, variant)?;
        }
        if mlp_dim == 0 || mlp_dim > (1 << 16) {
            return Err(BindingError::TensorShapeMismatch(0));
        }
    }
    Ok(())
}

fn validate_transformer_module(
    model: MimiBindingInput<'_>,
    family: TensorFamily,
    module: u16,
    dim: u64,
    mlp_dim: &mut u64,
    variant: RuntimeVariant,
) -> Result<(), BindingError> {
    for kind in [
        RoleKind::Norm1Weight,
        RoleKind::Norm1Bias,
        RoleKind::Norm2Weight,
        RoleKind::Norm2Bias,
        RoleKind::LayerScale1,
        RoleKind::LayerScale2,
    ] {
        let (index, metadata) = role_tensor(
            model,
            TensorRole {
                family,
                kind,
                module,
                split: 0,
                level: 0,
            },
        )?;
        expect_elements(index, metadata, dim)?;
    }
    let (index, metadata) = role_tensor(
        model,
        TensorRole {
            family,
            kind: RoleKind::AttentionInputProjection,
            module,
            split: 0,
            level: 0,
        },
    )?;
    expect_shape(index, metadata, 2, [dim, dim * 3, 1, 1])?;
    let (index, metadata) = role_tensor(
        model,
        TensorRole {
            family,
            kind: RoleKind::AttentionOutputProjection,
            module,
            split: 0,
            level: 0,
        },
    )?;
    expect_elements(index, metadata, dim * dim)?;
    let (index, metadata) = role_tensor(
        model,
        TensorRole {
            family,
            kind: RoleKind::Linear1,
            module,
            split: 0,
            level: 0,
        },
    )?;
    let dimensions = metadata.dimensions();
    if metadata.dimension_count() != 2 || dimensions[0] != dim || dimensions[1] == 0 {
        return Err(BindingError::TensorShapeMismatch(index));
    }
    if module == 0 {
        *mlp_dim = dimensions[1];
    } else if dimensions[1] != *mlp_dim {
        return Err(BindingError::TensorShapeMismatch(index));
    }
    let (index, metadata) = role_tensor(
        model,
        TensorRole {
            family,
            kind: RoleKind::Linear2,
            module,
            split: 0,
            level: 0,
        },
    )?;
    expect_elements(
        index,
        metadata,
        dim.checked_mul(*mlp_dim)
            .ok_or(BindingError::TensorShapeMismatch(index))?,
    )?;
    if variant == RuntimeVariant::Q8 && (!dim.is_multiple_of(32) || !(*mlp_dim).is_multiple_of(32))
    {
        return Err(BindingError::UnsupportedRuntime(index));
    }
    Ok(())
}

fn validate_downsample_upsample(
    model: MimiBindingInput<'_>,
    dim: u64,
    input_length: u64,
) -> Result<(), BindingError> {
    let down = role_tensor(
        model,
        TensorRole {
            family: TensorFamily::Downsample,
            kind: RoleKind::SingleWeight,
            module: 0,
            split: 0,
            level: 0,
        },
    )?;
    let (_, down_out) = resolve_conv(down.0, down.1, dim, 2)?;
    if down_out != dim
        || input_length == 0
        || !input_length.is_multiple_of(2)
        || input_length / 2 != 1
    {
        return Err(BindingError::TensorShapeMismatch(down.0));
    }
    let up = role_tensor(
        model,
        TensorRole {
            family: TensorFamily::Upsample,
            kind: RoleKind::SingleWeight,
            module: 0,
            split: 0,
            level: 0,
        },
    )?;
    let taps = up.1.dimensions()[0];
    if metadata_elements(up.1)
        != Some(
            taps.checked_mul(dim)
                .ok_or(BindingError::TensorShapeMismatch(up.0))?,
        )
        || !(2..=(1 << 16)).contains(&taps)
    {
        return Err(BindingError::TensorShapeMismatch(up.0));
    }
    Ok(())
}

fn validate_quantizer(
    model: MimiBindingInput<'_>,
    h: emel_model::bridge::MimiHParams,
    dim: u64,
    codebook_dim: u64,
    card: u64,
    variant: RuntimeVariant,
) -> Result<(), BindingError> {
    for split in 0..2_u8 {
        let input = role_tensor(
            model,
            TensorRole {
                family: TensorFamily::Quantizer,
                kind: RoleKind::QuantizerInputProjection,
                module: 0,
                split,
                level: 0,
            },
        )?;
        expect_projection_shape(input.0, input.1, dim, codebook_dim)?;
        let output = role_tensor(
            model,
            TensorRole {
                family: TensorFamily::Quantizer,
                kind: RoleKind::QuantizerOutputProjection,
                module: 0,
                split,
                level: 0,
            },
        )?;
        expect_projection_shape(output.0, output.1, codebook_dim, dim)?;
        if variant == RuntimeVariant::Q8
            && (!dim.is_multiple_of(32) || !codebook_dim.is_multiple_of(32))
        {
            return Err(BindingError::UnsupportedRuntime(input.0));
        }
        let levels = if split == 0 {
            h.semantic_n_q()
        } else {
            h.n_q() - h.semantic_n_q()
        };
        let levels = u16::try_from(levels).map_err(|_| {
            BindingError::InvalidHParams(MimiHParamsError::InvalidCodebookPartition)
        })?;
        if levels == 0 || levels > 32 {
            return Err(BindingError::InvalidHParams(
                MimiHParamsError::InvalidCodebookPartition,
            ));
        }
        for level in 0..levels {
            let codebook = role_tensor(
                model,
                TensorRole {
                    family: TensorFamily::Quantizer,
                    kind: RoleKind::Codebook,
                    module: 0,
                    split,
                    level,
                },
            )?;
            expect_shape(codebook.0, codebook.1, 2, [codebook_dim, card, 1, 1])?;
        }
    }
    Ok(())
}

fn metadata_elements(metadata: TensorMetadata) -> Option<u64> {
    let count = usize::try_from(metadata.dimension_count()).ok()?;
    if !(1..=4).contains(&count) {
        return None;
    }
    metadata.dimensions()[..count]
        .iter()
        .copied()
        .try_fold(1_u64, u64::checked_mul)
}

fn expect_elements(
    index: u32,
    metadata: TensorMetadata,
    expected: u64,
) -> Result<(), BindingError> {
    if metadata_elements(metadata) != Some(expected) {
        return Err(BindingError::TensorShapeMismatch(index));
    }
    Ok(())
}

fn resolve_conv(
    index: u32,
    metadata: TensorMetadata,
    input: u64,
    stride: u64,
) -> Result<(u64, u64), BindingError> {
    let rank = usize::try_from(metadata.dimension_count())
        .map_err(|_| BindingError::TensorShapeMismatch(index))?;
    if !(1..=3).contains(&rank) || input == 0 {
        return Err(BindingError::TensorShapeMismatch(index));
    }
    let taps = metadata.dimensions()[0];
    let elements = metadata_elements(metadata).ok_or(BindingError::TensorShapeMismatch(index))?;
    let divisor = taps
        .checked_mul(input)
        .ok_or(BindingError::TensorShapeMismatch(index))?;
    let output = elements
        .checked_div(divisor)
        .filter(|_| elements % divisor == 0)
        .ok_or(BindingError::TensorShapeMismatch(index))?;
    if taps == 0 || output == 0 || taps < stride || taps > (1 << 16) || output > (1 << 16) {
        return Err(BindingError::TensorShapeMismatch(index));
    }
    Ok((taps, output))
}

fn validate_seanet_family(
    model: MimiBindingInput<'_>,
    family: TensorFamily,
    dim: u64,
    input_length: u64,
    expected_length: u64,
    decoder: bool,
) -> Result<(), BindingError> {
    let modules: &[(u16, u8, u64)] = if decoder {
        &[
            (0, 0, 1),
            (2, 1, 8),
            (3, 2, 1),
            (5, 1, 6),
            (6, 2, 1),
            (8, 1, 5),
            (9, 2, 1),
            (11, 1, 4),
            (12, 2, 1),
            (14, 0, 1),
        ]
    } else {
        &[
            (0, 0, 1),
            (1, 2, 1),
            (3, 0, 4),
            (4, 2, 1),
            (6, 0, 5),
            (7, 2, 1),
            (9, 0, 6),
            (10, 2, 1),
            (12, 0, 8),
            (14, 0, 1),
        ]
    };
    let mut channels = if decoder { dim } else { 1 };
    let mut length = input_length;
    for &(module, kind, stride) in modules {
        channels = validate_seanet_module(model, family, channels, module, kind, stride)?;
        if kind == 0 {
            if !length.is_multiple_of(stride) {
                return Err(BindingError::TensorShapeMismatch(0));
            }
            length /= stride;
        } else if kind == 1 {
            length = length
                .checked_mul(stride)
                .ok_or(BindingError::TensorShapeMismatch(0))?;
        }
    }
    if channels != if decoder { 1 } else { dim } || length != expected_length {
        return Err(BindingError::TensorShapeMismatch(0));
    }
    Ok(())
}

fn validate_seanet_module(
    model: MimiBindingInput<'_>,
    family: TensorFamily,
    channels: u64,
    module: u16,
    kind: u8,
    stride: u64,
) -> Result<u64, BindingError> {
    if kind == 2 {
        validate_residual_module(model, family, channels, module)
    } else {
        validate_convolution_module(model, family, channels, module, kind, stride)
    }
}

fn validate_residual_module(
    model: MimiBindingInput<'_>,
    family: TensorFamily,
    channels: u64,
    module: u16,
) -> Result<u64, BindingError> {
    let first = role_tensor(
        model,
        TensorRole {
            family,
            kind: RoleKind::Residual1Weight,
            module,
            split: 0,
            level: 0,
        },
    )?;
    let (_, half) = resolve_conv(first.0, first.1, channels, 1)?;
    if let Some(first_bias) = optional_role_tensor(
        model,
        TensorRole {
            family,
            kind: RoleKind::Residual1Bias,
            module,
            split: 0,
            level: 0,
        },
    )? {
        expect_elements(first_bias.0, first_bias.1, half)?;
    }
    let second = role_tensor(
        model,
        TensorRole {
            family,
            kind: RoleKind::Residual3Weight,
            module,
            split: 0,
            level: 0,
        },
    )?;
    let (taps, output) = resolve_conv(second.0, second.1, half, 1)?;
    if taps != 1 || output != channels {
        return Err(BindingError::TensorShapeMismatch(second.0));
    }
    if let Some(second_bias) = optional_role_tensor(
        model,
        TensorRole {
            family,
            kind: RoleKind::Residual3Bias,
            module,
            split: 0,
            level: 0,
        },
    )? {
        expect_elements(second_bias.0, second_bias.1, channels)?;
    }
    Ok(channels)
}

fn validate_convolution_module(
    model: MimiBindingInput<'_>,
    family: TensorFamily,
    channels: u64,
    module: u16,
    kind: u8,
    stride: u64,
) -> Result<u64, BindingError> {
    let role_kind = if kind == 1 {
        RoleKind::ConvTransposeWeight
    } else {
        RoleKind::ConvWeight
    };
    let weight = role_tensor(
        model,
        TensorRole {
            family,
            kind: role_kind,
            module,
            split: 0,
            level: 0,
        },
    )?;
    let (_, output) = resolve_conv(weight.0, weight.1, channels, stride)?;
    let bias_kind = if kind == 1 {
        RoleKind::ConvTransposeBias
    } else {
        RoleKind::ConvBias
    };
    if let Some(bias) = optional_role_tensor(
        model,
        TensorRole {
            family,
            kind: bias_kind,
            module,
            split: 0,
            level: 0,
        },
    )? {
        expect_elements(bias.0, bias.1, output)?;
    }
    Ok(output)
}

fn optional_role_tensor(
    model: MimiBindingInput<'_>,
    expected: TensorRole,
) -> Result<Option<(u32, TensorMetadata)>, BindingError> {
    for index in 0..model.tensor_count() {
        let tensor = model
            .tensor(index)
            .ok_or(BindingError::InvalidTensor(index))?;
        let role =
            TensorRole::parse(tensor.name()).ok_or(BindingError::UnknownTensorFamily(index))?;
        if role == expected {
            return Ok(Some((
                index,
                tensor
                    .metadata()
                    .ok_or(BindingError::InvalidTensor(index))?,
            )));
        }
    }
    Ok(None)
}
fn role_tensor(
    model: MimiBindingInput<'_>,
    expected: TensorRole,
) -> Result<(u32, TensorMetadata), BindingError> {
    for index in 0..model.tensor_count() {
        let tensor = model
            .tensor(index)
            .ok_or(BindingError::InvalidTensor(index))?;
        let role =
            TensorRole::parse(tensor.name()).ok_or(BindingError::UnknownTensorFamily(index))?;
        if role == expected {
            return Ok((
                index,
                tensor
                    .metadata()
                    .ok_or(BindingError::InvalidTensor(index))?,
            ));
        }
    }
    Err(BindingError::TensorCountMismatch)
}

fn expect_shape(
    index: u32,
    metadata: TensorMetadata,
    count: u32,
    expected: [u64; 4],
) -> Result<(), BindingError> {
    if metadata.dimension_count() != count || metadata.dimensions() != expected {
        return Err(BindingError::TensorShapeMismatch(index));
    }
    Ok(())
}
fn expect_projection_shape(
    index: u32,
    metadata: TensorMetadata,
    input: u64,
    output: u64,
) -> Result<(), BindingError> {
    // The pinned prepare_linear and prepare_raw_q8_0 helpers consume these
    // projections by element count; GGUF may collapse trailing singleton
    // dimensions or retain a source-specific rank.
    expect_elements(
        index,
        metadata,
        input
            .checked_mul(output)
            .ok_or(BindingError::TensorShapeMismatch(index))?,
    )
}

fn validate_arenas(
    arenas: ArenaCapacities,
    model: MimiBindingInput<'_>,
    frame_samples: u32,
) -> Result<(), BindingError> {
    let dimension = usize::try_from(model.hparams().dim())
        .map_err(|_| BindingError::InvalidHParams(MimiHParamsError::NonPositive))?;
    let upsample = role_tensor(
        model,
        TensorRole {
            family: TensorFamily::Upsample,
            kind: RoleKind::SingleWeight,
            module: 0,
            split: 0,
            level: 0,
        },
    )?;
    let taps = usize::try_from(upsample.1.dimensions()[0])
        .map_err(|_| BindingError::TensorShapeMismatch(upsample.0))?;
    let state_required = dimension
        .checked_mul(taps)
        .ok_or(BindingError::ArenaCapacity(ArenaKind::State))?;
    let workspace_required = dimension
        .checked_add(state_required)
        .ok_or(BindingError::ArenaCapacity(ArenaKind::Workspace))?;
    let frame_required = usize::try_from(frame_samples)
        .unwrap_or(usize::MAX)
        .max(dimension.saturating_mul(2));
    if arenas.prepared_floats < state_required {
        return Err(BindingError::ArenaCapacity(ArenaKind::Prepared));
    }
    if arenas.state_floats < state_required {
        return Err(BindingError::ArenaCapacity(ArenaKind::State));
    }
    if arenas.workspace_floats < workspace_required {
        return Err(BindingError::ArenaCapacity(ArenaKind::Workspace));
    }
    if arenas.frame_floats < frame_required {
        return Err(BindingError::ArenaCapacity(ArenaKind::Frame));
    }
    Ok(())
}

fn validate_tensor_metadata(
    index: u32,
    role: TensorRole,
    metadata: TensorMetadata,
    variant: RuntimeVariant,
) -> Result<(), BindingError> {
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
    let serialized_size = metadata
        .tensor_type()
        .data_size(dimensions, metadata.dimension_count())
        .ok();
    let Some(storage) = metadata.storage() else {
        return Err(BindingError::TensorStorageMismatch(index));
    };
    if serialized_size != Some(metadata.data_size())
        || metadata.data_size() == 0
        || storage.length() != metadata.data_size()
        || storage.offset() != metadata.file_offset()
        || storage.split_index() != metadata.file_index()
        || storage.offset().checked_add(storage.length()).is_none()
        || metadata
            .data_offset()
            .checked_add(metadata.data_size())
            .is_none()
    {
        return Err(BindingError::TensorStorageMismatch(index));
    }
    if !variant.accepts(role, metadata.tensor_type()) {
        return Err(BindingError::UnsupportedRuntime(index));
    }
    if metadata.tensor_type() == SerializedType::Q8_0 && !dimensions[0].is_multiple_of(32) {
        return Err(BindingError::UnsupportedRuntime(index));
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

#[allow(
    clippy::suspicious_operation_groupings,
    reason = "storage.split_index is the model-owned copy of the descriptor split-file index"
)]
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
        || storage.split_index() != descriptor.file_index()
        || storage.offset() != descriptor.file_offset()
        || storage.length() != descriptor.data_size()
        || u64::try_from(bytes_len).ok() != Some(descriptor.data_size())
    {
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

    fn resident_tensor(
        name: &'static [u8],
        dimensions: [u64; 4],
        rank: u32,
    ) -> TensorInput<'static> {
        static BYTES: [u8; 8192] = [0; 8192];
        let elements = dimensions[..usize::try_from(rank).unwrap()]
            .iter()
            .product::<u64>();
        let byte_len = usize::try_from(elements * 4).unwrap();
        let bytes = &BYTES[..byte_len];
        TensorInput::with_bytes(
            name,
            TensorMetadata::new(emel_model::bridge::TensorMetadataInput {
                tensor_type: SerializedType::F32,
                dimension_count: rank,
                dimensions,
                data_offset: 0,
                file_offset: 0,
                data_size: elements * 4,
                file_index: 0,
                storage: Some(TensorBinding::new(0, 0, elements * 4)),
            }),
            bytes,
        )
    }

    #[test]
    fn accepts_only_exact_pinned_role_names() {
        assert_eq!(
            TensorRole::parse(b"mimi.encoder.model.0.conv.conv.weight"),
            Some(TensorRole {
                family: TensorFamily::Encoder,
                kind: RoleKind::ConvWeight,
                module: 0,
                split: 0,
                level: 0,
            })
        );
        assert!(TensorRole::parse(b"mimi.encoder.model.00.conv.conv.weight").is_none());
        assert!(TensorRole::parse(b"mimi.encoder.model.0.conv.conv.weight.extra").is_none());
        assert!(TensorRole::parse(b"mimi.encoderish.model.0.conv.conv.weight").is_none());
    }

    #[test]
    fn rejects_unknown_family_record() {
        let tensors = [resident_tensor(
            b"mimi.encoderish.model.0.conv.conv.weight",
            [16, 1, 1, 1],
            1,
        )];
        let data = emel_model::bridge::Data::try_from_mimi(MimiDataInput {
            hparams: hparams(),
            tensors: &tensors,
        })
        .unwrap();
        assert_eq!(
            validate_model_tensor_records(data.mimi_binding_input(), RuntimeVariant::F32),
            Err(BindingError::UnknownTensorFamily(0))
        );
    }

    #[test]
    fn rejects_missing_exact_tensor_role() {
        let tensors = [resident_tensor(
            b"mimi.encoder.model.0.conv.conv.weight",
            [7, 1, 4, 1],
            3,
        )];
        let data = emel_model::bridge::Data::try_from_mimi(MimiDataInput {
            hparams: hparams(),
            tensors: &tensors,
        })
        .unwrap();
        assert_eq!(
            role_tensor(
                data.mimi_binding_input(),
                TensorRole {
                    family: TensorFamily::Encoder,
                    kind: RoleKind::ConvBias,
                    module: 0,
                    split: 0,
                    level: 0,
                },
            ),
            Err(BindingError::TensorCountMismatch)
        );
    }

    #[test]
    fn rejects_wrong_rank_and_shape_for_exact_role() {
        let metadata =
            resident_tensor(b"mimi.encoder.model.0.conv.conv.weight", [7, 1, 4, 1], 3).metadata();
        assert_eq!(
            expect_shape(0, metadata, 2, [7, 4, 1, 1]),
            Err(BindingError::TensorShapeMismatch(0))
        );
        assert_eq!(expect_shape(0, metadata, 3, [7, 1, 4, 1]), Ok(()));
    }

    #[test]
    fn accepts_valid_exact_metadata_for_pinned_role() {
        let tensor = resident_tensor(
            b"mimi.quantizer.rvq_first.vq.layers.0._codebook.embedding",
            [8, 32, 1, 1],
            2,
        );
        let metadata = tensor.metadata();
        let role = TensorRole::parse(tensor.name());
        assert_eq!(
            role,
            Some(TensorRole {
                family: TensorFamily::Quantizer,
                kind: RoleKind::Codebook,
                module: 0,
                split: 0,
                level: 0,
            })
        );
        assert_eq!(
            validate_tensor_metadata(0, role.unwrap(), metadata, RuntimeVariant::F32),
            Ok(())
        );
        assert_eq!(expect_shape(0, metadata, 2, [8, 32, 1, 1]), Ok(()));
        assert_eq!(metadata.tensor_type(), SerializedType::F32);
    }

    fn role(kind: RoleKind, family: TensorFamily) -> TensorRole {
        TensorRole {
            family,
            kind,
            module: 0,
            split: 0,
            level: 0,
        }
    }

    #[test]
    fn runtime_variants_match_pinned_float_and_quantized_classes() {
        let conv = role(RoleKind::ConvWeight, TensorFamily::Encoder);
        let transposed = role(RoleKind::ConvTransposeWeight, TensorFamily::Decoder);
        let vector = role(RoleKind::Norm1Weight, TensorFamily::EncoderTransformer);
        let transformer_projection = role(
            RoleKind::AttentionInputProjection,
            TensorFamily::EncoderTransformer,
        );
        let rvq_projection = role(RoleKind::QuantizerInputProjection, TensorFamily::Quantizer);

        assert!(!RuntimeVariant::F32.accepts(conv, SerializedType::F16));
        assert!(!RuntimeVariant::F32.accepts(conv, SerializedType::Q8_0));
        assert!(RuntimeVariant::F16.accepts(conv, SerializedType::F16));
        assert!(!RuntimeVariant::F16.accepts(conv, SerializedType::F32));
        assert!(RuntimeVariant::F16.accepts(transposed, SerializedType::F32));
        assert!(RuntimeVariant::F16.accepts(transposed, SerializedType::F16));
        assert!(RuntimeVariant::F16.accepts(vector, SerializedType::F32));
        assert!(RuntimeVariant::F16.accepts(vector, SerializedType::F16));
        assert!(!RuntimeVariant::F16.accepts(vector, SerializedType::Q8_0));
        assert!(RuntimeVariant::F16.accepts(rvq_projection, SerializedType::F16));
        assert!(!RuntimeVariant::F16.accepts(rvq_projection, SerializedType::F32));
        assert!(RuntimeVariant::Q8.accepts(transformer_projection, SerializedType::Q8_0));
        assert!(RuntimeVariant::Q8.accepts(rvq_projection, SerializedType::Q8_0));
        assert!(!RuntimeVariant::Q8.accepts(transformer_projection, SerializedType::F32));
        assert!(RuntimeVariant::Q8.accepts(conv, SerializedType::F32));
        assert!(RuntimeVariant::Q8.accepts(conv, SerializedType::F16));
        assert!(!RuntimeVariant::Q8.accepts(conv, SerializedType::Q8_0));
    }

    #[test]
    fn rejects_metadata_only_tensor_before_prepare_walk() {
        let data = model();
        assert_eq!(
            validate_model_tensor_records(data.mimi_binding_input(), RuntimeVariant::F32),
            Err(BindingError::TensorStorageMismatch(0))
        );
    }

    #[test]
    fn accepts_source_optional_seanet_bias() {
        let weight = resident_tensor(b"mimi.encoder.model.0.conv.conv.weight", [7, 1, 4, 1], 3);
        let data = emel_model::bridge::Data::try_from_mimi(MimiDataInput {
            hparams: hparams(),
            tensors: &[weight],
        })
        .unwrap();
        assert_eq!(
            validate_convolution_module(
                data.mimi_binding_input(),
                TensorFamily::Encoder,
                1,
                0,
                0,
                1,
            ),
            Ok(4)
        );
    }

    #[test]
    fn validates_downsample_and_depthwise_upsample_geometry() {
        let down = resident_tensor(b"mimi.downsample.conv.conv.conv.weight", [2, 16, 16, 1], 3);
        let up = resident_tensor(
            b"mimi.upsample.convtr.convtr.convtr.weight",
            [2, 1, 16, 1],
            3,
        );
        let data = emel_model::bridge::Data::try_from_mimi(MimiDataInput {
            hparams: hparams(),
            tensors: &[down, up],
        })
        .unwrap();
        assert_eq!(
            validate_downsample_upsample(data.mimi_binding_input(), 16, 2),
            Ok(())
        );
    }

    #[test]
    fn projection_shape_uses_source_element_count_not_synthetic_rank() {
        let matrix = resident_tensor(
            b"mimi.quantizer.rvq_first.input_proj.weight",
            [16, 8, 1, 1],
            2,
        );
        let metadata = matrix.metadata();
        assert_eq!(expect_projection_shape(0, metadata, 16, 8), Ok(()));

        let collapsed = resident_tensor(
            b"mimi.quantizer.rvq_first.output_proj.weight",
            [128, 1, 1, 1],
            1,
        );
        assert_eq!(
            expect_projection_shape(1, collapsed.metadata(), 8, 16),
            Ok(())
        );
        let malformed = resident_tensor(
            b"mimi.quantizer.rvq_rest.input_proj.weight",
            [15, 8, 1, 1],
            2,
        );
        assert_eq!(
            expect_projection_shape(2, malformed.metadata(), 16, 8),
            Err(BindingError::TensorShapeMismatch(2))
        );
    }
    #[test]
    fn rejects_upsample_prepared_capacity_below_weight_count() {
        let up = resident_tensor(
            b"mimi.upsample.convtr.convtr.convtr.weight",
            [2, 1, 16, 1],
            3,
        );
        let model = emel_model::bridge::Data::try_from_mimi(MimiDataInput {
            hparams: hparams(),
            tensors: &[up],
        })
        .unwrap();
        assert_eq!(
            validate_arenas(
                ArenaCapacities::new(16, 32, 48, 1_920),
                model.mimi_binding_input(),
                1_920,
            ),
            Err(BindingError::ArenaCapacity(ArenaKind::Prepared))
        );
    }
}
