//! Public `OmniEmbed` actor events and immutable outcomes.

use core::fmt;
use core::marker::PhantomData;

use crate::catalog::event::{ModelIdentity, TensorDescriptor};

use super::OmniEmbed;

mod sealed {
    pub trait Sealed {}
}

pub trait Event: sealed::Sealed {
    type Output;
    fn dispatch(self, actor: &mut OmniEmbed) -> Self::Output;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidRequest,
    ModelInvalid,
    Capacity,
    Busy,
    StorageUnavailable,
    UnexpectedEvent,
    Internal,
}

/// Construction-time failure while binding one GGUF source to its model catalog.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoadError {
    Hparams(super::hparams::Error),
    Gguf(emel_gguf::event::QueryError),
    Catalog(crate::catalog::event::Error),
    Capacity,
    Internal,
}

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Hparams(_) => "OmniEmbed hyperparameter loading failed",
            Self::Gguf(_) => "OmniEmbed GGUF observation failed",
            Self::Catalog(_) => "OmniEmbed catalog binding failed",
            Self::Capacity => "OmniEmbed construction capacity is unavailable",
            Self::Internal => "internal OmniEmbed construction error",
        })
    }
}

impl std::error::Error for LoadError {}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidRequest => "invalid OmniEmbed request",
            Self::ModelInvalid => "OmniEmbed model contract is invalid",
            Self::Capacity => "OmniEmbed storage capacity is unavailable",
            Self::Busy => "OmniEmbed contract is busy",
            Self::StorageUnavailable => "OmniEmbed storage is unavailable",
            Self::UnexpectedEvent => "unexpected OmniEmbed event",
            Self::Internal => "internal OmniEmbed error",
        })
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Parameters {
    pub embedding_length: i32,
    pub image_encoder_length: i32,
    pub audio_encoder_length: i32,
    pub matryoshka_dimension_count: u32,
    pub matryoshka_dimensions: [i32; super::MAX_MATRYOSHKA_DIMENSIONS],
    pub image_encoder: EncoderName,
    pub audio_encoder: EncoderName,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            embedding_length: 0,
            image_encoder_length: 0,
            audio_encoder_length: 0,
            matryoshka_dimension_count: 0,
            matryoshka_dimensions: [0; super::MAX_MATRYOSHKA_DIMENSIONS],
            image_encoder: EncoderName::Missing,
            audio_encoder: EncoderName::Missing,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EncoderName {
    #[default]
    Missing,
    MobileNetV4Medium,
    EfficientAtMn20As,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Family {
    TextEncoder,
    TextProjection,
    ImageEncoder,
    ImageProjection,
    AudioEncoder,
    AudioProjection,
}

impl Family {
    pub const ALL: [Self; 6] = [
        Self::TextEncoder,
        Self::TextProjection,
        Self::ImageEncoder,
        Self::ImageProjection,
        Self::AudioEncoder,
        Self::AudioProjection,
    ];

    #[must_use]
    pub const fn prefix(self) -> &'static [u8] {
        match self {
            Self::TextEncoder => b"text_encoder.",
            Self::TextProjection => b"text_projection.",
            Self::ImageEncoder => b"image_encoder.",
            Self::ImageProjection => b"image_projection.",
            Self::AudioEncoder => b"audio_encoder.",
            Self::AudioProjection => b"audio_projection.",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FamilyDescriptor {
    family: Family,
    tensor_count: u32,
    first: TensorDescriptor,
}

impl FamilyDescriptor {
    #[must_use]
    pub const fn family(self) -> Family {
        self.family
    }
    #[must_use]
    pub const fn prefix(self) -> &'static [u8] {
        self.family.prefix()
    }
    #[must_use]
    pub const fn tensor_count(self) -> u32 {
        self.tensor_count
    }
    #[must_use]
    pub const fn first(self) -> TensorDescriptor {
        self.first
    }

    pub(super) const fn new(family: Family, tensor_count: u32, first: TensorDescriptor) -> Self {
        Self {
            family,
            tensor_count,
            first,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContractDescriptor {
    model: ModelIdentity,
    parameters: Parameters,
    families: [FamilyDescriptor; 6],
}

impl ContractDescriptor {
    #[must_use]
    pub const fn model(self) -> ModelIdentity {
        self.model
    }
    #[must_use]
    pub const fn embedding_length(self) -> i32 {
        self.parameters.embedding_length
    }
    #[must_use]
    pub const fn image_encoder_length(self) -> i32 {
        self.parameters.image_encoder_length
    }
    #[must_use]
    pub const fn audio_encoder_length(self) -> i32 {
        self.parameters.audio_encoder_length
    }
    #[must_use]
    pub const fn matryoshka_dimension_count(self) -> u32 {
        self.parameters.matryoshka_dimension_count
    }
    #[must_use]
    pub const fn matryoshka_dimensions(self) -> [i32; super::MAX_MATRYOSHKA_DIMENSIONS] {
        self.parameters.matryoshka_dimensions
    }
    #[must_use]
    pub const fn image_size(self) -> i32 {
        super::IMAGE_SIZE
    }
    #[must_use]
    pub const fn image_mean(self) -> [f32; 3] {
        super::IMAGE_MEAN
    }
    #[must_use]
    pub const fn image_std(self) -> [f32; 3] {
        super::IMAGE_STD
    }
    #[must_use]
    pub const fn audio_sample_rate(self) -> i32 {
        super::AUDIO_SAMPLE_RATE
    }
    #[must_use]
    pub const fn audio_n_fft(self) -> i32 {
        super::AUDIO_N_FFT
    }
    #[must_use]
    pub const fn audio_win_length(self) -> i32 {
        super::AUDIO_WIN_LENGTH
    }
    #[must_use]
    pub const fn audio_hop_size(self) -> i32 {
        super::AUDIO_HOP_SIZE
    }
    #[must_use]
    pub const fn audio_num_mel_bins(self) -> i32 {
        super::AUDIO_NUM_MEL_BINS
    }
    #[must_use]
    pub const fn audio_high_frequency(self) -> f32 {
        super::AUDIO_HIGH_FREQUENCY
    }
    #[must_use]
    pub const fn audio_preemphasis(self) -> f32 {
        super::AUDIO_PREEMPHASIS
    }
    #[must_use]
    pub const fn audio_log_offset(self) -> f32 {
        super::AUDIO_LOG_OFFSET
    }
    #[must_use]
    pub const fn audio_normalize_bias(self) -> f32 {
        super::AUDIO_NORMALIZE_BIAS
    }
    #[must_use]
    pub const fn audio_normalize_scale(self) -> f32 {
        super::AUDIO_NORMALIZE_SCALE
    }
    #[must_use]
    pub const fn family(self, family: Family) -> FamilyDescriptor {
        self.families[family as usize]
    }

    pub(super) const fn new(
        model: ModelIdentity,
        parameters: Parameters,
        families: &[FamilyDescriptor; 6],
    ) -> Self {
        Self {
            model,
            parameters,
            families: *families,
        }
    }
}

#[derive(Debug)]
pub struct Storage {
    pub(super) names: Box<[u8]>,
    pub(super) observation_names: Vec<u8>,
    pub(super) observations: Vec<ObservationRecord>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ObservationRecord {
    pub(super) name_offset: usize,
    pub(super) name_length: usize,
    pub(super) tensor: TensorDescriptor,
    pub(super) payload_present: bool,
}

impl Storage {
    /// Allocates the actor-owned first-name arena before dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Capacity`] when the arena cannot be allocated.
    pub fn with_name_capacity(capacity: usize) -> Result<Self, Error> {
        let mut names = Vec::new();
        names
            .try_reserve_exact(capacity)
            .map_err(|_| Error::Capacity)?;
        names.resize(capacity, 0);
        Ok(Self {
            names: names.into_boxed_slice(),
            observation_names: Vec::new(),
            observations: Vec::new(),
        })
    }

    #[must_use]
    pub const fn name_capacity(&self) -> usize {
        self.names.len()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ContractBegin;

impl ContractBegin {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ContractVisit;
impl ContractVisit {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ContractReset;
impl ContractReset {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[derive(Debug, Default)]
pub struct StorageRelease;
impl StorageRelease {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Invokes a callback with the first usable name in one tensor family.
///
/// The callback is synchronous and must not retain the borrowed name or
/// re-enter this actor.
pub struct WithFirstName<F, R> {
    pub(super) family: Family,
    pub(super) callback: F,
    pub(super) result: Result<Option<R>, Error>,
    marker: PhantomData<fn() -> R>,
}

impl<F, R> WithFirstName<F, R> {
    #[must_use]
    pub const fn new(family: Family, callback: F) -> Self {
        Self {
            family,
            callback,
            result: Err(Error::Internal),
            marker: PhantomData,
        }
    }
}

impl<F, R> fmt::Debug for WithFirstName<F, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WithFirstName")
            .field("family", &self.family)
            .finish_non_exhaustive()
    }
}

macro_rules! impl_event {
    ($event:ty, $output:ty, $method:ident) => {
        impl sealed::Sealed for $event {}
        impl Event for $event {
            type Output = $output;
            fn dispatch(self, actor: &mut OmniEmbed) -> Self::Output {
                actor.$method(self)
            }
        }
    };
}

impl_event!(ContractBegin, Result<(), Error>, contract_begin);
impl_event!(ContractVisit, Result<ContractDescriptor, Error>, contract_visit);
impl_event!(ContractReset, Result<(), Error>, contract_reset);
impl_event!(StorageRelease, Result<Storage, Error>, storage_release);

impl<F, R> sealed::Sealed for WithFirstName<F, R> where F: for<'name> FnMut(&'name [u8]) -> R {}
impl<F, R> Event for WithFirstName<F, R>
where
    F: for<'name> FnMut(&'name [u8]) -> R,
{
    type Output = Result<Option<R>, Error>;
    fn dispatch(mut self, actor: &mut OmniEmbed) -> Self::Output {
        actor.with_first_name(&mut self);
        self.result
    }
}
