//! Allocation-free first-family-name query child actor.

#![allow(clippy::struct_excessive_bools)]

use super::actor::Context;
use super::event::{Error, Family, WithFirstName};
use super::sm::{OmniEmbedNameQueryStateMachine, OmniEmbedNameQueryStateMachineContext};

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
    let mut machine = OmniEmbedNameQueryStateMachine::new(QueryContext);
    machine
        .process_event(&mut request)
        .expect("typed name query has an explicit transition");
}

impl OmniEmbedNameQueryStateMachineContext for QueryContext {
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
    fn guard_text_encoder<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.ready && request.operation.family() == Family::TextEncoder)
    }
    fn guard_text_projection<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.ready && request.operation.family() == Family::TextProjection)
    }
    fn guard_image_encoder<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.ready && request.operation.family() == Family::ImageEncoder)
    }
    fn guard_image_projection<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.ready && request.operation.family() == Family::ImageProjection)
    }
    fn guard_audio_encoder<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.ready && request.operation.family() == Family::AudioEncoder)
    }
    fn guard_audio_projection<'query, 'data: 'query, O: NameOperation + 'data>(
        &self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.ready && request.operation.family() == Family::AudioProjection)
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
    fn effect_apply_text_encoder<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply(request.context.first_name(Family::TextEncoder));
        Ok(())
    }
    fn effect_apply_text_projection<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply(request.context.first_name(Family::TextProjection));
        Ok(())
    }
    fn effect_apply_image_encoder<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply(request.context.first_name(Family::ImageEncoder));
        Ok(())
    }
    fn effect_apply_image_projection<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply(request.context.first_name(Family::ImageProjection));
        Ok(())
    }
    fn effect_apply_audio_encoder<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply(request.context.first_name(Family::AudioEncoder));
        Ok(())
    }
    fn effect_apply_audio_projection<'query, 'data: 'query, O: NameOperation + 'data>(
        &mut self,
        request: &'query mut NameRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply(request.context.first_name(Family::AudioProjection));
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
