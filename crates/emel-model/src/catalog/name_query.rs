//! Generic allocation-free canonical-name query child actor.

use super::actor::Context;
use super::event::{Error, NameId, WithTensorName};
use super::sm::{CatalogNameQueryStateMachine, CatalogNameQueryStateMachineContext};

pub(super) trait NameOperation {
    fn name_id(&self) -> NameId;
    fn apply(&mut self, name: &[u8]);
    fn error(&mut self, error: Error);
}

pub(super) struct NameRequest<'data, O> {
    sealed: bool,
    context: &'data Context,
    operation: &'data mut O,
}

#[derive(Clone, Copy, Debug, Default)]
struct NameContext;

pub(super) fn process<'data, O: NameOperation + 'data>(
    sealed: bool,
    context: &'data Context,
    operation: &'data mut O,
) {
    let mut request = NameRequest {
        sealed,
        context,
        operation,
    };
    let mut machine = CatalogNameQueryStateMachine::new(NameContext);
    if machine.process_event(&mut request).is_err() {
        request.operation.error(Error::Internal);
    }
}

impl CatalogNameQueryStateMachineContext for NameContext {
    fn guard_storage_unavailable<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(!request.sealed)
    }

    fn guard_wrong<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.sealed && request.operation.name_id().owner != request.context.owner())
    }

    fn guard_stale<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        let identity = request.operation.name_id();
        Ok(request.sealed
            && identity.owner == request.context.owner()
            && (identity.generation != request.context.generation()
                || !request.context.ordinal_valid(identity.ordinal)))
    }

    fn guard_valid<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        let identity = request.operation.name_id();
        Ok(request.sealed
            && identity.owner == request.context.owner()
            && identity.generation == request.context.generation()
            && request.context.ordinal_valid(identity.ordinal))
    }

    fn effect_storage_unavailable<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(Error::StorageUnavailable);
        Ok(())
    }

    fn effect_wrong<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(Error::WrongNameIdentity);
        Ok(())
    }

    fn effect_stale<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(Error::StaleNameIdentity);
        Ok(())
    }

    fn effect_apply<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        let identity = request.operation.name_id();
        let name = request
            .context
            .name(identity.ordinal)
            .expect("name query guard validates the canonical range");
        request.operation.apply(name);
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

impl<F, R> NameOperation for WithTensorName<F, R>
where
    F: for<'name> FnMut(&'name [u8]) -> R,
{
    fn name_id(&self) -> NameId {
        self.name_id
    }

    fn apply(&mut self, name: &[u8]) {
        self.result = Ok(Some((self.callback)(name)));
    }

    fn error(&mut self, error: Error) {
        self.result = Err(error);
    }
}
