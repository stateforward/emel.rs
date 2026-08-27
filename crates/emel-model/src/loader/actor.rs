//! Owning model-loader actor and static tensor dependency boundary.

use std::cell::{Cell, RefCell};
use std::fmt;

use super::event::{Error, LoadRequest, LoadStats};
use super::sm::{EventLoadRuntime, ModelLoaderContext, ModelLoaderEvents, ModelLoaderStateMachine};
use crate::data::Data;

/// Replaceable tensor residency actor used by [`ModelLoader`].
pub(crate) trait TensorLoader {
    fn load(
        &mut self,
        model: &mut Data,
        source: super::event::Source<'_>,
        strategy: emel_io::loader::event::StrategyKind,
    ) -> Result<LoadStats, Error>;
}

/// Default dependency. It explicitly reports the unavailable actor path.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NoTensorLoader;

impl TensorLoader for NoTensorLoader {
    fn load(
        &mut self,
        _model: &mut Data,
        _source: super::event::Source<'_>,
        _strategy: emel_io::loader::event::StrategyKind,
    ) -> Result<LoadStats, Error> {
        Err(Error::IoStrategyUnavailable)
    }
}

/// Single-writer, run-to-completion model loader.
pub(crate) struct ModelLoader<T = NoTensorLoader>
where
    T: TensorLoader,
{
    machine: ModelLoaderStateMachine<ModelLoaderContext<T>>,
}

impl ModelLoader<NoTensorLoader> {
    pub(crate) const fn new() -> Self {
        Self::with_tensor_loader(NoTensorLoader)
    }
}

impl<T: TensorLoader> ModelLoader<T> {
    pub(crate) const fn with_tensor_loader(tensor_loader: T) -> Self {
        Self {
            machine: ModelLoaderStateMachine::new(ModelLoaderContext::new(tensor_loader)),
        }
    }

    /// Processes one caller-owned request synchronously.
    pub(crate) fn process_event(&mut self, request: LoadRequest<'_>) -> Result<LoadStats, Error> {
        let request = RefCell::new(request);
        let status = Cell::new(super::event::LoadStatus::default());
        let outcome = Cell::new(Err(Error::InternalError));
        self.machine
            .process_event(ModelLoaderEvents::Load(EventLoadRuntime {
                request: &request,
                status: &status,
                outcome: &outcome,
            }))
            .map_err(|_| Error::InternalError)?;
        outcome.get()
    }

    #[cfg(test)]
    pub(crate) fn is_ready(&self) -> bool {
        self.machine.is(&super::sm::ModelLoaderStates::Ready)
    }
}

impl Default for ModelLoader<NoTensorLoader> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: TensorLoader> fmt::Debug for ModelLoader<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelLoader")
            .finish_non_exhaustive()
    }
}
