//! Sortformer metadata contract from the pinned `emel.cpp` model domain.

#![allow(clippy::large_stack_frames)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::large_stack_arrays)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::manual_range_contains)]

use crate::catalog::Catalog;
use crate::catalog::event::{
    DescribeTensor, FindTensor, ModelIdentity, PrefixScanSummary, ScanPrefix, TensorDescriptor,
    ValidateTensorShape,
};
use crate::loader::hparams::{Accessor, Error};
use core::fmt;
use emel_gguf::Loader;

pub const ARCHITECTURE_NAME: &[u8] = b"sortformer";

pub const ENCODER_DIM: usize = 512;
pub const HIDDEN_DIM: usize = 192;
pub const PAIR_HIDDEN_DIM: usize = 384;
pub const SPEAKER_COUNT: usize = 4;
pub const MODULE_TENSOR_COUNT: usize = 8;
pub const MODULE_TENSOR_RANKS: [usize; MODULE_TENSOR_COUNT] = [1, 2, 1, 2, 1, 2, 1, 2];

pub const MODULE_TENSOR_NAMES: [&[u8]; MODULE_TENSOR_COUNT] = [
    b"mods.ep.b",
    b"mods.ep.w",
    b"mods.fh2h.b",
    b"mods.fh2h.w",
    b"mods.h2s.b",
    b"mods.h2s.w",
    b"mods.sh2s.b",
    b"mods.sh2s.w",
];

pub const MODULE_TENSOR_DIMS: [[i64; 4]; MODULE_TENSOR_COUNT] = [
    [192, 0, 0, 0],
    [512, 192, 0, 0],
    [192, 0, 0, 0],
    [192, 192, 0, 0],
    [4, 0, 0, 0],
    [384, 4, 0, 0],
    [4, 0, 0, 0],
    [192, 4, 0, 0],
];

pub const TRANSFORMER_LAYER_COUNT: usize = 18;
pub const TRANSFORMER_LAYER_TENSOR_COUNT: usize = 16;
pub const TRANSFORMER_HIDDEN_DIM: usize = 192;
pub const TRANSFORMER_INNER_DIM: usize = 768;
pub const TRANSFORMER_HEAD_COUNT: usize = 8;
pub const TRANSFORMER_HEAD_DIM: usize = 24;
pub const TRANSFORMER_MAX_FRAME_COUNT: usize = 188;

pub const TRANSFORMER_TENSOR_SUFFIXES: [&[u8]; TRANSFORMER_LAYER_TENSOR_COUNT] = [
    b"sa.k.b", b"sa.k.w", b"sa.o.b", b"sa.o.w", b"sa.q.b", b"sa.q.w", b"sa.v.b", b"sa.v.w",
    b"ln1.b", b"ln1.w", b"ln2.b", b"ln2.w", b"ff.di.b", b"ff.di.w", b"ff.do.b", b"ff.do.w",
];

pub const TRANSFORMER_TENSOR_RANKS: [usize; TRANSFORMER_LAYER_TENSOR_COUNT] =
    [1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2];

pub const TRANSFORMER_TENSOR_DIMS: [[i64; 4]; TRANSFORMER_LAYER_TENSOR_COUNT] = [
    [192, 0, 0, 0],
    [192, 192, 0, 0],
    [192, 0, 0, 0],
    [192, 192, 0, 0],
    [192, 0, 0, 0],
    [192, 192, 0, 0],
    [192, 0, 0, 0],
    [192, 192, 0, 0],
    [192, 0, 0, 0],
    [192, 0, 0, 0],
    [192, 0, 0, 0],
    [192, 0, 0, 0],
    [768, 0, 0, 0],
    [192, 768, 0, 0],
    [192, 0, 0, 0],
    [768, 192, 0, 0],
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransformerTensorRole {
    KeyBias,
    KeyWeight,
    OutputBias,
    OutputWeight,
    QueryBias,
    QueryWeight,
    ValueBias,
    ValueWeight,
    Norm1Bias,
    Norm1Weight,
    Norm2Bias,
    Norm2Weight,
    FeedForwardInputBias,
    FeedForwardInputWeight,
    FeedForwardOutputBias,
    FeedForwardOutputWeight,
}

pub const TRANSFORMER_TENSOR_ROLES: [TransformerTensorRole; TRANSFORMER_LAYER_TENSOR_COUNT] = [
    TransformerTensorRole::KeyBias,
    TransformerTensorRole::KeyWeight,
    TransformerTensorRole::OutputBias,
    TransformerTensorRole::OutputWeight,
    TransformerTensorRole::QueryBias,
    TransformerTensorRole::QueryWeight,
    TransformerTensorRole::ValueBias,
    TransformerTensorRole::ValueWeight,
    TransformerTensorRole::Norm1Bias,
    TransformerTensorRole::Norm1Weight,
    TransformerTensorRole::Norm2Bias,
    TransformerTensorRole::Norm2Weight,
    TransformerTensorRole::FeedForwardInputBias,
    TransformerTensorRole::FeedForwardInputWeight,
    TransformerTensorRole::FeedForwardOutputBias,
    TransformerTensorRole::FeedForwardOutputWeight,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformerLayerContract {
    tensors: [TensorDescriptor; TRANSFORMER_LAYER_TENSOR_COUNT],
}

impl TransformerLayerContract {
    #[must_use]
    pub const fn tensor(self, index: usize) -> Option<TensorDescriptor> {
        if index < TRANSFORMER_LAYER_TENSOR_COUNT {
            Some(self.tensors[index])
        } else {
            None
        }
    }

    #[must_use]
    pub const fn tensor_count(self) -> usize {
        TRANSFORMER_LAYER_TENSOR_COUNT
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformerContract {
    layers: [TransformerLayerContract; TRANSFORMER_LAYER_COUNT],
    tensor_count: u32,
}

impl TransformerContract {
    #[must_use]
    pub const fn layer(self, index: usize) -> Option<TransformerLayerContract> {
        if index < TRANSFORMER_LAYER_COUNT {
            Some(self.layers[index])
        } else {
            None
        }
    }

    #[must_use]
    pub const fn layer_count(self) -> usize {
        TRANSFORMER_LAYER_COUNT
    }

    #[must_use]
    pub const fn tensor_count(self) -> u32 {
        self.tensor_count
    }
}

/// Opaque, source-ordered views of the eight required module tensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModulesContract {
    tensors: [TensorDescriptor; MODULE_TENSOR_COUNT],
}

impl ModulesContract {
    #[must_use]
    pub const fn tensor(self, index: usize) -> Option<TensorDescriptor> {
        if index < MODULE_TENSOR_COUNT {
            Some(self.tensors[index])
        } else {
            None
        }
    }

    #[must_use]
    pub const fn tensor_count(self) -> usize {
        MODULE_TENSOR_COUNT
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContractError {
    InvalidRequest,
    ModelInvalid,
    StorageUnavailable,
    WrongModelIdentity,
    StaleModelIdentity,
    MissingFamily(Family),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Family {
    Prep,
    Encoder,
    Modules,
    Output,
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidRequest => "invalid sortformer contract request",
            Self::ModelInvalid => "sortformer tensor contract is invalid",
            Self::StorageUnavailable => "sortformer catalog storage is unavailable",
            Self::WrongModelIdentity => "sortformer model identity belongs to another catalog",
            Self::StaleModelIdentity => "sortformer model identity is stale",
            Self::MissingFamily(_) => "sortformer tensor family is missing or unbound",
        })
    }
}
impl std::error::Error for ContractError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContractDescriptor {
    model: ModelIdentity,
    prep: PrefixScanSummary,
    encoder: PrefixScanSummary,
    modules: PrefixScanSummary,
    output: PrefixScanSummary,
    modules_contract: ModulesContract,
    transformer_contract: TransformerContract,
}

impl ContractDescriptor {
    #[must_use]
    pub const fn model(self) -> ModelIdentity {
        self.model
    }
    #[must_use]
    pub const fn prep(self) -> PrefixScanSummary {
        self.prep
    }
    #[must_use]
    pub const fn encoder(self) -> PrefixScanSummary {
        self.encoder
    }
    #[must_use]
    pub const fn modules(self) -> PrefixScanSummary {
        self.modules
    }
    #[must_use]
    pub const fn output(self) -> PrefixScanSummary {
        self.output
    }

    #[must_use]
    pub const fn modules_contract(self) -> ModulesContract {
        self.modules_contract
    }

    #[must_use]
    pub const fn transformer_contract(self) -> TransformerContract {
        self.transformer_contract
    }
}

#[derive(Debug)]
pub struct Storage {
    contract: Option<ContractDescriptor>,
}

impl Storage {
    #[must_use]
    pub const fn new() -> Self {
        Self { contract: None }
    }
    pub const fn clear(&mut self) {
        self.contract = None;
    }
}

impl Default for Storage {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct Sortformer {
    catalog: Catalog,
    storage: Storage,
    contract: Option<ContractDescriptor>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContractState {
    Empty,
    Ready,
}

impl Sortformer {
    #[must_use]
    pub const fn new(catalog: Catalog, storage: Storage) -> Self {
        Self {
            catalog,
            storage,
            contract: None,
        }
    }

    /// Builds and publishes the complete source-ordered tensor contract.
    ///
    /// # Errors
    ///
    /// Returns a typed contract error when the catalog identity is wrong or
    /// stale, a required family/tensor is absent, or a tensor shape is invalid.
    ///
    /// # Panics
    ///
    /// Panics only if an internal invariant is violated after the preceding
    /// validation proves every required descriptor is present.
    pub fn build_contract(
        &mut self,
        model: ModelIdentity,
    ) -> Result<ContractDescriptor, ContractError> {
        let mut scan = |prefix| self.catalog.process_event(ScanPrefix::new(model, prefix));
        let prep = scan(b"prep.").map_err(map_catalog_error)?;
        let encoder = scan(b"enc.").map_err(map_catalog_error)?;
        let modules = scan(b"mods.").map_err(map_catalog_error)?;
        let output = scan(b"te.").map_err(map_catalog_error)?;
        let mut module_tensors = [None; MODULE_TENSOR_COUNT];
        let mut transformer_layers =
            [[None; TRANSFORMER_LAYER_TENSOR_COUNT]; TRANSFORMER_LAYER_COUNT];
        let family = [
            (Family::Prep, prep),
            (Family::Encoder, encoder),
            (Family::Modules, modules),
            (Family::Output, output),
        ];
        if let Some((family, _summary)) = family.into_iter().find(|(_, summary)| {
            summary.count() == 0 || !summary.all_bound() || !summary.all_geometry_valid()
        }) {
            return Err(ContractError::MissingFamily(family));
        }
        for index in 0..MODULE_TENSOR_COUNT {
            let dimensions = &MODULE_TENSOR_DIMS[index][..MODULE_TENSOR_RANKS[index]];
            let valid = self
                .catalog
                .process_event(ValidateTensorShape::new(
                    model,
                    MODULE_TENSOR_NAMES[index],
                    dimensions,
                ))
                .map_err(map_catalog_error)?;
            if !valid {
                return Err(ContractError::MissingFamily(Family::Modules));
            }
            let tensor_id = self
                .catalog
                .process_event(FindTensor::new(model, MODULE_TENSOR_NAMES[index]))
                .map_err(map_catalog_error)?
                .map(TensorDescriptor::tensor_id)
                .ok_or(ContractError::MissingFamily(Family::Modules))?;
            module_tensors[index] = Some(
                self.catalog
                    .process_event(DescribeTensor::new(tensor_id))
                    .map_err(map_catalog_error)?,
            );
        }
        for (layer, transformer_layer) in transformer_layers.iter_mut().enumerate() {
            for tensor_index in 0..TRANSFORMER_LAYER_TENSOR_COUNT {
                let mut name = [0_u8; 32];
                let prefix = b"te.l";
                let mut cursor = 0;
                name[..prefix.len()].copy_from_slice(prefix);
                cursor += prefix.len();
                let layer_value =
                    u32::try_from(layer).map_err(|_| ContractError::InvalidRequest)?;
                if layer_value >= 10 {
                    name[cursor] = b'0'
                        + u8::try_from(layer_value / 10)
                            .map_err(|_| ContractError::InvalidRequest)?;
                    cursor += 1;
                }
                name[cursor] = b'0'
                    + u8::try_from(layer_value % 10).map_err(|_| ContractError::InvalidRequest)?;
                cursor += 1;
                name[cursor] = b'.';
                cursor += 1;
                let suffix = TRANSFORMER_TENSOR_SUFFIXES[tensor_index];
                if cursor + suffix.len() > name.len() {
                    return Err(ContractError::InvalidRequest);
                }
                name[cursor..cursor + suffix.len()].copy_from_slice(suffix);
                let name = &name[..cursor + suffix.len()];
                let rank = TRANSFORMER_TENSOR_RANKS[tensor_index];
                let dimensions = &TRANSFORMER_TENSOR_DIMS[tensor_index][..rank];
                let valid = self
                    .catalog
                    .process_event(ValidateTensorShape::new(model, name, dimensions))
                    .map_err(map_catalog_error)?;
                if !valid {
                    return Err(ContractError::MissingFamily(Family::Encoder));
                }
                let tensor_id = self
                    .catalog
                    .process_event(FindTensor::new(model, name))
                    .map_err(map_catalog_error)?
                    .map(TensorDescriptor::tensor_id)
                    .ok_or(ContractError::MissingFamily(Family::Encoder))?;
                transformer_layer[tensor_index] = Some(
                    self.catalog
                        .process_event(DescribeTensor::new(tensor_id))
                        .map_err(map_catalog_error)?,
                );
            }
        }
        let modules_contract = ModulesContract {
            tensors: module_tensors.map(|tensor| {
                tensor.expect("every exact module tensor was validated and described")
            }),
        };
        let transformer_contract = TransformerContract {
            layers: transformer_layers.map(|layer| TransformerLayerContract {
                tensors: layer.map(|tensor| {
                    tensor.expect("every transformer tensor was validated and described")
                }),
            }),
            tensor_count: u32::try_from(TRANSFORMER_LAYER_COUNT * TRANSFORMER_LAYER_TENSOR_COUNT)
                .map_err(|_| ContractError::InvalidRequest)?,
        };
        let descriptor = ContractDescriptor {
            model,
            prep,
            encoder,
            modules,
            output,
            modules_contract,
            transformer_contract,
        };
        self.storage.contract = Some(descriptor);
        self.contract = Some(descriptor);
        Ok(descriptor)
    }

    /// Clears the completed contract for caller-side reuse.
    pub const fn reset(&mut self) {
        self.storage.clear();
        self.contract = None;
    }

    /// Returns the last completed contract, if any.
    #[must_use]
    pub const fn contract(&self) -> Option<ContractDescriptor> {
        self.contract
    }

    #[must_use]
    pub const fn state(&self) -> ContractState {
        match self.contract {
            Some(_) => ContractState::Ready,
            None => ContractState::Empty,
        }
    }
}

fn map_catalog_error(error: crate::catalog::event::Error) -> ContractError {
    match error {
        crate::catalog::event::Error::WrongModelIdentity => ContractError::WrongModelIdentity,
        crate::catalog::event::Error::StaleModelIdentity => ContractError::StaleModelIdentity,
        crate::catalog::event::Error::StorageUnavailable => ContractError::StorageUnavailable,
        _ => ContractError::InvalidRequest,
    }
}

#[must_use]
pub fn is_execution_architecture(name: &[u8]) -> bool {
    name == ARCHITECTURE_NAME
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Parameters {
    pub sample_rate: i32,
    pub speaker_count: i32,
    pub chunk_len: i32,
    pub chunk_right_context: i32,
    pub fifo_len: i32,
    pub spkcache_update_period: i32,
    pub spkcache_len: i32,
    pub original_tensor_count: i32,
    pub tensor_count: i32,
    pub skipped_tensor_count: i32,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            speaker_count: 4,
            chunk_len: 188,
            chunk_right_context: 1,
            fifo_len: 0,
            spkcache_update_period: 188,
            spkcache_len: 188,
            original_tensor_count: 0,
            tensor_count: 0,
            skipped_tensor_count: 0,
        }
    }
}

impl Parameters {
    /// Validates the source-fixed tensor-count relationship.
    #[must_use]
    pub const fn counts_valid(self) -> bool {
        self.original_tensor_count > 0
            && self.tensor_count > 0
            && self.skipped_tensor_count >= 0
            && self.tensor_count <= self.original_tensor_count
            && self.skipped_tensor_count <= self.original_tensor_count - self.tensor_count
    }
}

/// Loads and validates the source-fixed Sortformer metadata contract.
pub fn load_hparams(loader: &mut Loader) -> Result<Parameters, Error> {
    let mut accessor = Accessor::new();
    let mut p = Parameters::default();
    Accessor::require_string(loader, b"sortformer.source.format", b"nemo")?;
    Accessor::require_string(loader, b"sortformer.tensor_name_scheme", b"compact_v1")?;
    Accessor::require_string(loader, b"sortformer.outtype", b"f32")?;
    accessor.assign_i32(
        loader,
        b"sortformer.original_tensor_count",
        &mut p.original_tensor_count,
    )?;
    accessor.assign_i32(loader, b"sortformer.tensor_count", &mut p.tensor_count)?;
    accessor.assign_i32(
        loader,
        b"sortformer.skipped_tensor_count",
        &mut p.skipped_tensor_count,
    )?;
    let sample_rate_primary = accessor.assign_i32(
        loader,
        b"sortformer.config.preprocessor.sample_rate",
        &mut p.sample_rate,
    );
    if sample_rate_primary.is_err() {
        p.sample_rate = 16000;
        accessor.assign_i32(loader, b"sortformer.config.sample_rate", &mut p.sample_rate)?;
    }
    let speaker_primary = accessor.assign_i32(
        loader,
        b"sortformer.config.sortformer_modules.num_spks",
        &mut p.speaker_count,
    );
    if speaker_primary.is_err() {
        p.speaker_count = 4;
        let speaker_alias = accessor.assign_i32(
            loader,
            b"sortformer.config.sortformer_modules.num_speakers",
            &mut p.speaker_count,
        );
        if speaker_alias.is_err() {
            p.speaker_count = 4;
            accessor.assign_i32(loader, b"sortformer.config.num_spks", &mut p.speaker_count)?;
        }
    }
    accessor.assign_i32(
        loader,
        b"sortformer.config.sortformer_modules.chunk_len",
        &mut p.chunk_len,
    )?;
    accessor.assign_i32(
        loader,
        b"sortformer.config.sortformer_modules.chunk_right_context",
        &mut p.chunk_right_context,
    )?;
    accessor.assign_i32(
        loader,
        b"sortformer.config.sortformer_modules.fifo_len",
        &mut p.fifo_len,
    )?;
    accessor.assign_i32(
        loader,
        b"sortformer.config.sortformer_modules.spkcache_update_period",
        &mut p.spkcache_update_period,
    )?;
    accessor.assign_i32(
        loader,
        b"sortformer.config.sortformer_modules.spkcache_len",
        &mut p.spkcache_len,
    )?;
    if !p.counts_valid()
        || p.sample_rate != 16000
        || p.speaker_count != 4
        || p.chunk_len != 188
        || p.chunk_right_context != 1
        || p.fifo_len != 0
        || p.spkcache_update_period != 188
        || p.spkcache_len != 188
    {
        return Err(Error {
            operation: crate::loader::hparams::Operation::OptionalI32,
            kind: crate::loader::hparams::ErrorKind::Range,
        });
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::event::{BindStorage, SealModel, Storage as CatalogStorage, TensorInput};
    use allocation_counter::measure;

    #[test]
    fn loads_source_alias_hparams_through_typed_gguf_queries() {
        use crate::loader::test_gguf::{Fixture, I32, U32};
        let mut loader = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"sortformer.source.format", b"nemo")
                .string(b"sortformer.tensor_name_scheme", b"compact_v1")
                .string(b"sortformer.outtype", b"f32")
                .scalar(
                    b"sortformer.original_tensor_count",
                    U32,
                    10_u32.to_le_bytes(),
                )
                .scalar(b"sortformer.tensor_count", U32, 8_u32.to_le_bytes())
                .scalar(b"sortformer.skipped_tensor_count", I32, 2_i32.to_le_bytes())
                .scalar(
                    b"sortformer.config.sample_rate",
                    U32,
                    16_000_u32.to_le_bytes(),
                )
                .scalar(
                    b"sortformer.config.sortformer_modules.num_speakers",
                    U32,
                    4_u32.to_le_bytes(),
                )
                .scalar(
                    b"sortformer.config.sortformer_modules.chunk_len",
                    U32,
                    188_u32.to_le_bytes(),
                )
                .scalar(
                    b"sortformer.config.sortformer_modules.chunk_right_context",
                    U32,
                    1_u32.to_le_bytes(),
                )
                .scalar(
                    b"sortformer.config.sortformer_modules.fifo_len",
                    U32,
                    0_u32.to_le_bytes(),
                )
                .scalar(
                    b"sortformer.config.sortformer_modules.spkcache_update_period",
                    U32,
                    188_u32.to_le_bytes(),
                )
                .scalar(
                    b"sortformer.config.sortformer_modules.spkcache_len",
                    U32,
                    188_u32.to_le_bytes(),
                )
                .build(),
        );
        let parameters = load_hparams(&mut loader).unwrap();
        assert_eq!(
            (parameters.sample_rate, parameters.speaker_count),
            (16_000, 4)
        );
    }

    #[test]
    fn missing_required_tensor_counts_are_rejected() {
        use crate::loader::test_gguf::{Fixture, U32};
        let mut loader = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"sortformer.source.format", b"nemo")
                .string(b"sortformer.tensor_name_scheme", b"compact_v1")
                .string(b"sortformer.outtype", b"f32")
                .scalar(b"sortformer.tensor_count", U32, 8_u32.to_le_bytes())
                .build(),
        );
        assert!(load_hparams(&mut loader).is_err());
    }

    fn catalog_with(names: &[&[u8]]) -> (Catalog, ModelIdentity) {
        let name_bytes = names.iter().map(|name| name.len()).sum();
        let mut storage =
            CatalogStorage::with_capacity(names.len(), name_bytes, names.len()).unwrap();
        for (index, name) in names.iter().enumerate() {
            let (dimension_count, dimensions) = if (2..=9).contains(&index) {
                let module_index = index - 2;
                let rank = MODULE_TENSOR_RANKS[module_index];
                let mut dimensions = [1, 1, 1, 1];
                dimensions[..rank].copy_from_slice(&MODULE_TENSOR_DIMS[module_index][..rank]);
                (i32::try_from(rank).unwrap(), dimensions)
            } else if name.starts_with(b"te.l") {
                let mut dot_count = 0_usize;
                let suffix_start = name
                    .iter()
                    .enumerate()
                    .find_map(|(position, byte)| {
                        if *byte == b'.' {
                            dot_count += 1;
                        }
                        (dot_count == 2).then_some(position + 1)
                    })
                    .unwrap_or(name.len());
                let suffix = &name[suffix_start..];
                let tensor_index = TRANSFORMER_TENSOR_SUFFIXES
                    .iter()
                    .position(|candidate| *candidate == suffix)
                    .unwrap_or(0);
                let rank = TRANSFORMER_TENSOR_RANKS[tensor_index];
                let mut dimensions = [1, 1, 1, 1];
                dimensions[..rank].copy_from_slice(&TRANSFORMER_TENSOR_DIMS[tensor_index][..rank]);
                (i32::try_from(rank).unwrap(), dimensions)
            } else {
                (2, [1, 1, 1, 1])
            };
            storage
                .push_tensor(TensorInput::new(
                    name,
                    0,
                    dimension_count,
                    dimensions,
                    16,
                    true,
                ))
                .unwrap();
        }
        let mut catalog = Catalog::try_new().unwrap();
        catalog.process_event(BindStorage::new(storage)).unwrap();
        let model = catalog.process_event(SealModel::new()).unwrap();
        (catalog, model)
    }

    fn complete_names() -> Vec<&'static [u8]> {
        let mut names: Vec<&'static [u8]> = vec![
            b"prep.x".as_slice(),
            b"enc.x".as_slice(),
            MODULE_TENSOR_NAMES[0],
            MODULE_TENSOR_NAMES[1],
            MODULE_TENSOR_NAMES[2],
            MODULE_TENSOR_NAMES[3],
            MODULE_TENSOR_NAMES[4],
            MODULE_TENSOR_NAMES[5],
            MODULE_TENSOR_NAMES[6],
            MODULE_TENSOR_NAMES[7],
            b"te.x",
        ];
        for layer in 0..TRANSFORMER_LAYER_COUNT {
            for suffix in TRANSFORMER_TENSOR_SUFFIXES {
                let name = format!("te.l{layer}.{}", std::str::from_utf8(suffix).unwrap());
                names.push(Box::leak(name.into_bytes().into_boxed_slice()));
            }
        }
        names
    }

    #[test]
    fn defaults_match_pinned_execution_constants() {
        let p = Parameters::default();
        assert_eq!(
            (p.sample_rate, p.speaker_count, p.chunk_len),
            (16000, 4, 188)
        );
    }

    #[test]
    fn count_validation_matches_source_relationships() {
        let mut parameters = Parameters {
            original_tensor_count: 10,
            tensor_count: 8,
            skipped_tensor_count: 2,
            ..Parameters::default()
        };
        assert!(parameters.counts_valid());
        parameters.skipped_tensor_count = 3;
        assert!(!parameters.counts_valid());
        parameters.skipped_tensor_count = 2;
        parameters.tensor_count = 11;
        assert!(!parameters.counts_valid());
    }

    #[test]
    fn module_tensor_inventory_matches_pinned_source() {
        assert_eq!(MODULE_TENSOR_NAMES.len(), MODULE_TENSOR_COUNT);
        assert_eq!(MODULE_TENSOR_DIMS[0], [192, 0, 0, 0]);
        assert_eq!(MODULE_TENSOR_DIMS[1], [512, 192, 0, 0]);
        assert_eq!(MODULE_TENSOR_DIMS[5], [384, 4, 0, 0]);
        assert_eq!(TRANSFORMER_LAYER_COUNT, 18);
        assert_eq!(TRANSFORMER_TENSOR_SUFFIXES[0], b"sa.k.b");
        assert_eq!(TRANSFORMER_TENSOR_SUFFIXES[15], b"ff.do.w");
        assert_eq!(TRANSFORMER_TENSOR_RANKS[13], 2);
        assert_eq!(TRANSFORMER_TENSOR_DIMS[13], [192, 768, 0, 0]);
        assert_eq!(TRANSFORMER_TENSOR_ROLES[0], TransformerTensorRole::KeyBias);
        assert_eq!(
            TRANSFORMER_TENSOR_ROLES[15],
            TransformerTensorRole::FeedForwardOutputWeight
        );
    }

    #[test]
    fn contract_scans_all_pinned_tensor_families() {
        let names = complete_names();
        let (catalog, model) = catalog_with(&names);
        let mut actor = Sortformer::new(catalog, Storage::new());
        let descriptor = actor.build_contract(model).unwrap();
        assert_eq!(descriptor.prep().count(), 1);
        assert_eq!(descriptor.encoder().count(), 1);
        assert_eq!(descriptor.modules().count(), 8);
        assert_eq!(descriptor.output().count(), 289);
        let modules = descriptor.modules_contract();
        assert_eq!(modules.tensor_count(), MODULE_TENSOR_COUNT);
        let first = modules.tensor(0).unwrap();
        assert_eq!(first.dimension_count(), 1);
        assert_eq!(first.dimensions()[0], HIDDEN_DIM as i64);
        let projection = modules.tensor(1).unwrap();
        assert_eq!(projection.dimension_count(), 2);
        assert_eq!(
            projection.dimensions()[..2],
            [ENCODER_DIM as i64, HIDDEN_DIM as i64]
        );
        assert!(modules.tensor(MODULE_TENSOR_COUNT).is_none());
        let transformer = descriptor.transformer_contract();
        assert_eq!(transformer.layer_count(), 18);
        assert_eq!(transformer.tensor_count(), 288);
        let first_layer = transformer.layer(0).unwrap();
        assert_eq!(first_layer.tensor_count(), 16);
        assert_eq!(first_layer.tensor(0).unwrap().dimension_count(), 1);
        assert_eq!(first_layer.tensor(1).unwrap().dimension_count(), 2);
        assert!(transformer.layer(18).is_none());
        assert_eq!(actor.contract(), Some(descriptor));
        actor.reset();
        assert_eq!(actor.contract(), None);
        assert_eq!(actor.state(), ContractState::Empty);
    }

    #[test]
    fn contract_rejects_missing_family_and_preserves_explicit_identity_errors() {
        let (catalog, model) = catalog_with(&[
            b"prep.x",
            b"enc.x",
            MODULE_TENSOR_NAMES[0],
            MODULE_TENSOR_NAMES[1],
            MODULE_TENSOR_NAMES[2],
            MODULE_TENSOR_NAMES[3],
            MODULE_TENSOR_NAMES[4],
            MODULE_TENSOR_NAMES[5],
            MODULE_TENSOR_NAMES[6],
            MODULE_TENSOR_NAMES[7],
        ]);
        let mut actor = Sortformer::new(catalog, Storage::new());
        assert_eq!(
            actor.build_contract(model),
            Err(ContractError::MissingFamily(Family::Output))
        );

        let names = complete_names();
        let (catalog, _) = catalog_with(&names);
        let (_, other_model) = catalog_with(&names);
        let mut actor = Sortformer::new(catalog, Storage::new());
        assert_eq!(
            actor.build_contract(other_model),
            Err(ContractError::WrongModelIdentity)
        );
    }

    #[test]
    fn contract_rejects_named_family_with_invalid_geometry() {
        let names = [
            b"prep.x".as_slice(),
            b"enc.x",
            MODULE_TENSOR_NAMES[0],
            MODULE_TENSOR_NAMES[1],
            MODULE_TENSOR_NAMES[2],
            MODULE_TENSOR_NAMES[3],
            MODULE_TENSOR_NAMES[4],
            MODULE_TENSOR_NAMES[5],
            MODULE_TENSOR_NAMES[6],
            MODULE_TENSOR_NAMES[7],
            b"te.x",
        ];
        let name_bytes = names.iter().map(|name| name.len()).sum();
        let mut storage =
            CatalogStorage::with_capacity(names.len(), name_bytes, names.len()).unwrap();
        for (index, name) in names.iter().enumerate() {
            let dimensions = if index == 2 {
                [0, 1, 1, 1]
            } else if index >= 2 && index <= 9 {
                let module_index = index - 2;
                let mut dimensions = [1, 1, 1, 1];
                let rank = MODULE_TENSOR_RANKS[module_index];
                dimensions[..rank].copy_from_slice(&MODULE_TENSOR_DIMS[module_index][..rank]);
                dimensions
            } else {
                [1, 1, 1, 1]
            };
            storage
                .push_tensor(TensorInput::new(name, 0, 2, dimensions, 16, true))
                .unwrap();
        }
        let mut catalog = Catalog::try_new().unwrap();
        catalog.process_event(BindStorage::new(storage)).unwrap();
        let model = catalog.process_event(SealModel::new()).unwrap();
        let mut actor = Sortformer::new(catalog, Storage::new());
        assert_eq!(
            actor.build_contract(model),
            Err(ContractError::MissingFamily(Family::Modules))
        );
    }

    #[test]
    fn contract_scans_are_allocation_free_after_construction() {
        let names = complete_names();
        let (catalog, model) = catalog_with(&names);
        let mut actor = Sortformer::new(catalog, Storage::new());
        let outcome = std::cell::RefCell::new(None);
        let allocation = measure(|| {
            outcome.replace(Some(actor.build_contract(model)));
        });
        assert_eq!(allocation.count_total, 0);
        assert!(outcome.into_inner().unwrap().is_ok());
    }

    #[test]
    fn full_transformer_contract_replay_is_allocation_free() {
        let names = complete_names();
        let (catalog, model) = catalog_with(&names);
        let mut actor = Sortformer::new(catalog, Storage::new());
        actor.build_contract(model).unwrap();
        let allocation = measure(|| {
            actor.build_contract(model).unwrap();
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(
            actor
                .contract()
                .unwrap()
                .transformer_contract()
                .tensor_count(),
            288
        );
    }

    #[test]
    fn rejected_contract_scans_are_allocation_free_and_do_not_publish_state() {
        let names = [
            b"prep.x".as_slice(),
            b"enc.x",
            MODULE_TENSOR_NAMES[0],
            MODULE_TENSOR_NAMES[1],
            MODULE_TENSOR_NAMES[2],
            MODULE_TENSOR_NAMES[3],
            MODULE_TENSOR_NAMES[4],
            MODULE_TENSOR_NAMES[5],
            MODULE_TENSOR_NAMES[6],
            MODULE_TENSOR_NAMES[7],
        ];
        let (catalog, model) = catalog_with(&names);
        let mut actor = Sortformer::new(catalog, Storage::new());
        let outcome = std::cell::RefCell::new(None);
        let allocation = measure(|| {
            outcome.replace(Some(actor.build_contract(model)));
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(
            outcome.into_inner().unwrap(),
            Err(ContractError::MissingFamily(Family::Output))
        );
        assert_eq!(actor.state(), ContractState::Empty);
        assert_eq!(actor.contract(), None);
    }

    #[test]
    fn reset_is_allocation_free_and_rejected_contracts_remain_recoverable() {
        let names = complete_names();
        let (catalog, model) = catalog_with(&names);
        let mut actor = Sortformer::new(catalog, Storage::new());
        actor.build_contract(model).unwrap();
        let allocation = measure(|| actor.reset());
        assert_eq!(allocation.count_total, 0);
        assert_eq!(actor.state(), ContractState::Empty);
        assert_eq!(actor.contract(), None);
    }
}
