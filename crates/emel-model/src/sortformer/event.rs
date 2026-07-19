//! Public Sortformer actor events and immutable outcomes.

use core::fmt;
use core::marker::PhantomData;

use crate::catalog::event::{ModelIdentity, TensorDescriptor};

use super::Sortformer;

mod sealed {
    pub trait Sealed {}
}

pub trait Event: sealed::Sealed {
    type Output;
    fn dispatch(self, actor: &mut Sortformer) -> Self::Output;
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
            Self::Hparams(_) => "Sortformer hyperparameter loading failed",
            Self::Gguf(_) => "Sortformer GGUF observation failed",
            Self::Catalog(_) => "Sortformer catalog binding failed",
            Self::Capacity => "Sortformer construction capacity is unavailable",
            Self::Internal => "internal Sortformer construction error",
        })
    }
}

impl std::error::Error for LoadError {}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidRequest => "invalid Sortformer request",
            Self::ModelInvalid => "Sortformer model contract is invalid",
            Self::Capacity => "Sortformer storage capacity is unavailable",
            Self::Busy => "Sortformer contract is busy",
            Self::StorageUnavailable => "Sortformer storage is unavailable",
            Self::UnexpectedEvent => "unexpected Sortformer event",
            Self::Internal => "internal Sortformer error",
        })
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Parameters {
    pub original_tensor_count: i32,
    pub tensor_count: i32,
    pub skipped_tensor_count: i32,
    pub sample_rate: i32,
    pub speaker_count: i32,
    pub chunk_len: i32,
    pub chunk_right_context: i32,
    pub fifo_len: i32,
    pub spkcache_update_period: i32,
    pub spkcache_len: i32,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            original_tensor_count: 0,
            tensor_count: 0,
            skipped_tensor_count: 0,
            sample_rate: super::SAMPLE_RATE,
            speaker_count: super::SPEAKER_COUNT,
            chunk_len: super::CHUNK_LEN,
            chunk_right_context: super::CHUNK_RIGHT_CONTEXT,
            fifo_len: super::FIFO_LEN,
            spkcache_update_period: super::SPKCACHE_UPDATE_PERIOD,
            spkcache_len: super::SPKCACHE_LEN,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Family {
    FeatureExtractor,
    Encoder,
    Modules,
    TransformerEncoder,
}

impl Family {
    pub const ALL: [Self; 4] = [
        Self::FeatureExtractor,
        Self::Encoder,
        Self::Modules,
        Self::TransformerEncoder,
    ];

    #[must_use]
    pub const fn prefix(self) -> &'static [u8] {
        match self {
            Self::FeatureExtractor => b"prep.",
            Self::Encoder => b"enc.",
            Self::Modules => b"mods.",
            Self::TransformerEncoder => b"te.",
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
    families: [FamilyDescriptor; 4],
}

impl ContractDescriptor {
    #[must_use]
    pub const fn model(self) -> ModelIdentity {
        self.model
    }
    #[must_use]
    pub const fn sample_rate(self) -> i32 {
        self.parameters.sample_rate
    }
    #[must_use]
    pub const fn speaker_count(self) -> i32 {
        self.parameters.speaker_count
    }
    #[must_use]
    pub const fn frame_shift_ms(self) -> i32 {
        super::FRAME_SHIFT_MS
    }
    #[must_use]
    pub const fn chunk_len(self) -> i32 {
        self.parameters.chunk_len
    }
    #[must_use]
    pub const fn chunk_right_context(self) -> i32 {
        self.parameters.chunk_right_context
    }
    #[must_use]
    pub const fn fifo_len(self) -> i32 {
        self.parameters.fifo_len
    }
    #[must_use]
    pub const fn spkcache_update_period(self) -> i32 {
        self.parameters.spkcache_update_period
    }
    #[must_use]
    pub const fn spkcache_len(self) -> i32 {
        self.parameters.spkcache_len
    }
    #[must_use]
    pub const fn family(self, family: Family) -> FamilyDescriptor {
        self.families[family as usize]
    }

    pub(super) const fn new(
        model: ModelIdentity,
        parameters: Parameters,
        families: &[FamilyDescriptor; 4],
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
            fn dispatch(self, actor: &mut Sortformer) -> Self::Output {
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
    fn dispatch(mut self, actor: &mut Sortformer) -> Self::Output {
        actor.with_first_name(&mut self);
        self.result
    }
}
