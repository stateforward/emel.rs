//! Allocation-free first-family-name query child actor.

#![allow(clippy::struct_excessive_bools)]

use super::actor::Context;
use super::event::{Error, Family, WithFirstName};
use super::sm::{SortformerNameQueryStateMachine, SortformerNameQueryStateMachineContext};

pub(super) trait NameOperation {
    fn family(&self) -> Family;
    fn apply(&mut self, name: &[u8]);
    fn error(&mut self, error: Error);
}

pub(super) struct NameRequest<'data, O> {
    empty: bool,
    scanning: bool,
    ready: bool,
    storage_available: bool,
    context: &'data Context,
    operation: &'data mut O,
}

#[derive(Clone, Copy, Debug, Default)]
struct QueryContext;

pub(super) fn process<'data, O: NameOperation + 'data>(
    empty: bool,
    scanning: bool,
    ready: bool,
    context: &'data Context,
    operation: &'data mut O,
) {
    let mut request = NameRequest {
        empty,
        scanning,
        ready,
        storage_available: context.storage_available(),
        context,
        operation,
    };
    let mut machine = SortformerNameQueryStateMachine::new(QueryContext);
    machine
        .process_event(&mut request)
        .expect("typed name query has an explicit transition");
}

impl SortformerNameQueryStateMachineContext for QueryContext {
    fn guard_storage_unavailable<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.empty && !request.storage_available)
    }
    fn guard_invalid<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.empty && request.storage_available)
    }
    fn guard_busy<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.scanning)
    }
    fn guard_feature<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.ready && request.operation.family() == Family::FeatureExtractor)
    }
    fn guard_encoder<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.ready && request.operation.family() == Family::Encoder)
    }
    fn guard_modules<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.ready && request.operation.family() == Family::Modules)
    }
    fn guard_transformer<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.ready && request.operation.family() == Family::TransformerEncoder)
    }
    fn effect_storage_unavailable<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(Error::StorageUnavailable);
        Ok(())
    }
    fn effect_invalid<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(Error::InvalidRequest);
        Ok(())
    }
    fn effect_busy<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(Error::Busy);
        Ok(())
    }
    fn effect_apply_feature<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply(request.context.first_name(Family::FeatureExtractor));
        Ok(())
    }
    fn effect_apply_encoder<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply(request.context.first_name(Family::Encoder));
        Ok(())
    }
    fn effect_apply_modules<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply(request.context.first_name(Family::Modules));
        Ok(())
    }
    fn effect_apply_transformer<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply(request.context.first_name(Family::TransformerEncoder));
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

impl<F, R> NameOperation for WithFirstName<F, R>
where
    F: for<'name> FnMut(&'name [u8]) -> R,
{
    fn family(&self) -> Family {
        self.family
    }
    fn apply(&mut self, name: &[u8]) {
        self.result = Ok(Some((self.callback)(name)));
    }
    fn error(&mut self, error: Error) {
        self.result = Err(error);
    }
}
