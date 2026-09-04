//! Typed model-loader request and outcome contracts.
use super::super::data::Data;
use emel_io::loader::event::StrategyKind;
use emel_io::mmap::event::MmapSource;

/// Model-loader error classes from the pinned C++ contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    #[default]
    None,
    InvalidRequest,
    ParseFailed,
    BackendError,
    ModelInvalid,
    InternalError,
    Untracked,
    IoStrategyUnavailable,
    /// A mapped residency release failed and remains owned by the tensor actor.
    MappedReleaseFailed(emel_io::mmap::event::Error),
    Unknown(u32),
}

impl core::fmt::Display for Error {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "model loader error: {self:?}")
    }
}

impl std::error::Error for Error {}

/// Statistics published by one completed load.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadStats {
    pub(crate) bytes_total: u64,
    pub(crate) bytes_done: u64,
    pub(crate) used_mmap: bool,
    pub(crate) used_strategy: StrategyKind,
}

impl LoadStats {
    #[must_use]
    pub const fn bytes_total(self) -> u64 {
        self.bytes_total
    }
    #[must_use]
    pub const fn bytes_done(self) -> u64 {
        self.bytes_done
    }
    #[must_use]
    pub const fn used_mmap(self) -> bool {
        self.used_mmap
    }
    #[must_use]
    pub const fn used_strategy(self) -> StrategyKind {
        self.used_strategy
    }
}

impl Default for LoadStats {
    fn default() -> Self {
        Self {
            bytes_total: 0,
            bytes_done: 0,
            used_mmap: false,
            used_strategy: StrategyKind::None,
        }
    }
}

/// Error outcome published synchronously to the caller callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadError {
    pub(crate) error: Error,
    pub(crate) requested_strategy: StrategyKind,
    pub(crate) used_strategy: StrategyKind,
}

impl LoadError {
    pub(crate) const fn new(
        error: Error,
        requested_strategy: StrategyKind,
        used_strategy: StrategyKind,
    ) -> Self {
        Self {
            error,
            requested_strategy,
            used_strategy,
        }
    }
    #[must_use]
    pub const fn error(self) -> Error {
        self.error
    }
    #[must_use]
    pub const fn requested_strategy(self) -> StrategyKind {
        self.requested_strategy
    }
    #[must_use]
    pub const fn used_strategy(self) -> StrategyKind {
        self.used_strategy
    }
}

/// Borrowed source information passed to parser and tensor actors.
#[derive(Clone, Copy, Debug)]
pub struct Source<'a> {
    pub(crate) model_path: &'a str,
    pub(crate) file_image: Option<&'a [u8]>,
    pub(crate) mapped_files: Option<&'a [MmapSource]>,
}

impl<'a> Source<'a> {
    /// Creates a source from a path and optional caller-owned file image.
    #[must_use]
    pub const fn new(model_path: &'a str, file_image: Option<&'a [u8]>) -> Self {
        Self {
            model_path,
            file_image,
            mapped_files: None,
        }
    }
    /// Adds caller-owned, safely opened split-file mapping capabilities.
    #[must_use]
    pub const fn with_mapped_files(mut self, files: &'a [MmapSource]) -> Self {
        self.mapped_files = Some(files);
        self
    }

    /// Returns the caller-owned split-file mapping capabilities.
    #[must_use]
    pub const fn mapped_files(self) -> Option<&'a [MmapSource]> {
        self.mapped_files
    }
    #[must_use]
    pub const fn model_path(self) -> &'a str {
        self.model_path
    }
    #[must_use]
    pub const fn file_image(self) -> Option<&'a [u8]> {
        self.file_image
    }
}

/// Callback signatures are higher-ranked so callbacks cannot retain a request.
pub type ParseModel = for<'a> fn(&mut Data, Source<'a>) -> Error;
pub type ModelCheck = fn(&mut Data) -> Error;
pub type DoneCallback = for<'a> fn(&LoadRequest<'a>, LoadStats);
pub type ErrorCallback = for<'a> fn(&LoadRequest<'a>, LoadError);

/// Caller-owned model load request.
#[derive(Debug)]
pub struct LoadRequest<'a> {
    pub(crate) model: &'a mut Data,
    pub(crate) source: Source<'a>,
    pub(crate) parse_model: Option<ParseModel>,
    pub(crate) vocab_only: bool,
    pub(crate) check_tensors: bool,
    pub(crate) validate_architecture: bool,
    pub(crate) io_strategy: StrategyKind,
    pub(crate) map_layers: Option<ModelCheck>,
    pub(crate) validate_structure: Option<ModelCheck>,
    pub(crate) validate_architecture_impl: Option<ModelCheck>,
    pub(crate) on_done: Option<DoneCallback>,
    pub(crate) on_error: Option<ErrorCallback>,
}

impl<'a> LoadRequest<'a> {
    /// Creates a request carrying caller-owned model and source data.
    #[must_use]
    pub const fn new(model: &'a mut Data, source: Source<'a>) -> Self {
        Self {
            model,
            source,
            parse_model: None,
            vocab_only: false,
            check_tensors: true,
            validate_architecture: true,
            io_strategy: StrategyKind::None,
            map_layers: None,
            validate_structure: None,
            validate_architecture_impl: None,
            on_done: None,
            on_error: None,
        }
    }

    #[must_use]
    pub const fn with_parser(mut self, parser: ParseModel) -> Self {
        self.parse_model = Some(parser);
        self
    }
    #[must_use]
    pub const fn vocab_only(mut self, value: bool) -> Self {
        self.vocab_only = value;
        self
    }
    #[must_use]
    pub const fn check_tensors(mut self, value: bool) -> Self {
        self.check_tensors = value;
        self
    }
    #[must_use]
    pub const fn validate_architecture(mut self, value: bool) -> Self {
        self.validate_architecture = value;
        self
    }
    #[must_use]
    pub const fn io_strategy(mut self, value: StrategyKind) -> Self {
        self.io_strategy = value;
        self
    }
    #[must_use]
    pub const fn map_layers(mut self, callback: ModelCheck) -> Self {
        self.map_layers = Some(callback);
        self
    }
    #[must_use]
    pub const fn validate_structure(mut self, callback: ModelCheck) -> Self {
        self.validate_structure = Some(callback);
        self
    }
    #[must_use]
    pub const fn validate_architecture_with(mut self, callback: ModelCheck) -> Self {
        self.validate_architecture_impl = Some(callback);
        self
    }
    #[must_use]
    pub const fn on_done(mut self, callback: DoneCallback) -> Self {
        self.on_done = Some(callback);
        self
    }
    #[must_use]
    pub const fn on_error(mut self, callback: ErrorCallback) -> Self {
        self.on_error = Some(callback);
        self
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.parse_model.is_some()
            && ((!self.source.model_path.is_empty()
                || self
                    .source
                    .file_image
                    .is_some_and(|bytes| !bytes.is_empty()))
                || self
                    .source
                    .mapped_files
                    .is_some_and(|files| !files.is_empty()))
    }

    pub(crate) fn tensor_capacity_valid(&self) -> bool {
        usize::try_from(self.model.n_tensors)
            .is_ok_and(|count| count <= super::super::data::MAX_TENSORS)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LoadStatus {
    pub(crate) error: Error,
    pub(crate) stats: LoadStats,
}

impl LoadStatus {
    pub(crate) const fn error(error: Error) -> Self {
        Self {
            error,
            stats: LoadStats {
                bytes_total: 0,
                bytes_done: 0,
                used_mmap: false,
                used_strategy: StrategyKind::None,
            },
        }
    }

    pub(crate) const fn success(stats: LoadStats) -> Self {
        Self {
            error: Error::None,
            stats,
        }
    }
}
