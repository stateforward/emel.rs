//! Typed model-loader request and outcome contracts.

use super::super::data::Data;
use emel_io::loader::event::StrategyKind;

/// Model-loader error classes from the pinned C++ contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub(crate) enum Error {
    #[default]
    None,
    InvalidRequest,
    ParseFailed,
    BackendError,
    ModelInvalid,
    InternalError,
    Untracked,
    IoStrategyUnavailable,
    Unknown(u32),
}

/// Statistics published by one completed load.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LoadStats {
    pub(crate) bytes_total: u64,
    pub(crate) bytes_done: u64,
    pub(crate) used_mmap: bool,
    pub(crate) used_strategy: StrategyKind,
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
pub(crate) struct LoadError {
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
}

/// Borrowed source information passed to parser and tensor actors.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Source<'a> {
    pub(crate) model_path: &'a str,
    pub(crate) file_image: Option<&'a [u8]>,
}

/// Callback signatures are higher-ranked so callbacks cannot retain a request.
pub(crate) type ParseModel = for<'a> fn(&mut Data, Source<'a>) -> Error;
pub(crate) type ModelCheck = fn(&mut Data) -> Error;
pub(crate) type DoneCallback = for<'a> fn(&LoadRequest<'a>, LoadStats);
pub(crate) type ErrorCallback = for<'a> fn(&LoadRequest<'a>, LoadError);

/// Caller-owned model load request.
pub(crate) struct LoadRequest<'a> {
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
    pub(crate) const fn new(model: &'a mut Data, source: Source<'a>) -> Self {
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

    pub(crate) fn is_valid(&self) -> bool {
        self.parse_model.is_some()
            && (!self.source.model_path.is_empty()
                || self
                    .source
                    .file_image
                    .is_some_and(|bytes| !bytes.is_empty()))
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
