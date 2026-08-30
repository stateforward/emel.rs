//! Allocation-free semantic tensor query child actor.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::needless_lifetimes,
    reason = "SML-generated generic callback signatures mirror event lifetimes"
)]

use crate::event::{QueryError, Storage, TensorDescriptor, WithTensor};

mod sm;

use sm::{TensorQueryStateMachine, TensorQueryStateMachineContext};

pub trait Operation {
    fn index(&self) -> u32;
    fn missing(&mut self);
    fn error(&mut self, error: QueryError);
    fn apply(&mut self, name: &[u8], descriptor: TensorDescriptor, data: &[u8]);
}

pub struct TensorRequest<'data, O> {
    parsed: bool,
    storage: Option<&'data Storage>,
    tensor_count: u32,
    operation: &'data mut O,
}

#[derive(Clone, Copy, Debug, Default)]
struct TensorContext;

pub fn process<'data, O: Operation + 'data>(
    parsed: bool,
    storage: Option<&'data Storage>,
    tensor_count: u32,
    operation: &'data mut O,
) {
    let mut request = TensorRequest {
        parsed,
        storage,
        tensor_count,
        operation,
    };
    let mut machine = TensorQueryStateMachine::new(TensorContext);
    let dispatch_result = machine.process_event(&mut request);
    record_dispatch_result(&dispatch_result, request.operation);
}

fn record_dispatch_result<T, E, O: Operation>(result: &Result<T, E>, operation: &mut O) {
    if result.is_err() {
        operation.error(QueryError::Internal);
    }
}

fn selected<'data, O: Operation>(
    request: &TensorRequest<'data, O>,
) -> Option<&'data super::TensorInfo> {
    let index = usize::try_from(request.operation.index()).ok()?;
    request
        .storage?
        .tensors
        .get(..usize::try_from(request.tensor_count).ok()?)?
        .get(index)
}

fn range(bytes: &[u8], offset: u64, length: u64) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    let length = usize::try_from(length).ok()?;
    bytes.get(start..start.checked_add(length)?)
}

fn selected_ranges_valid<O: Operation>(request: &TensorRequest<'_, O>) -> bool {
    let Some(tensor) = selected(request) else {
        return false;
    };
    let Some(storage) = request.storage else {
        return false;
    };
    range(
        &storage.source,
        u64::from(tensor.name_offset),
        u64::from(tensor.name_length),
    )
    .is_some()
        && range(&storage.source, tensor.file_offset, tensor.data_size).is_some()
}

impl TensorQueryStateMachineContext for TensorContext {
    fn guard_not_parsed<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut TensorRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(!request.parsed)
    }

    fn guard_missing<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut TensorRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.parsed && request.operation.index() >= request.tensor_count)
    }

    fn guard_malformed<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut TensorRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.parsed
            && request.operation.index() < request.tensor_count
            && !selected_ranges_valid(request))
    }

    fn guard_ready<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut TensorRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.parsed
            && request.operation.index() < request.tensor_count
            && selected_ranges_valid(request))
    }

    fn effect_not_parsed<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut TensorRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(QueryError::NotParsed);
        Ok(())
    }

    fn effect_missing<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut TensorRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.missing();
        Ok(())
    }

    fn effect_malformed<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut TensorRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(QueryError::Malformed);
        Ok(())
    }

    fn effect_apply<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut TensorRequest<'data, O>,
    ) -> Result<(), ()> {
        let tensor = selected(request).expect("ready guard validates tensor index");
        let image = &request
            .storage
            .expect("ready guard validates bound source")
            .source;
        let name = range(
            image,
            u64::from(tensor.name_offset),
            u64::from(tensor.name_length),
        )
        .expect("ready guard validates tensor name range");
        let data = range(image, tensor.file_offset, tensor.data_size)
            .expect("ready guard validates tensor data range");
        let descriptor = TensorDescriptor::new(
            tensor.tensor_type,
            tensor.dimension_count,
            tensor.dimensions,
            tensor.data_offset,
            tensor.data_section_offset,
            tensor.file_offset,
            tensor.alignment,
            tensor.data_size,
            tensor.file_index,
        );
        request.operation.apply(name, descriptor, data);
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

impl<F, R> Operation for WithTensor<F, R>
where
    F: for<'value> FnMut(&'value [u8], TensorDescriptor, &'value [u8]) -> R,
{
    fn index(&self) -> u32 {
        self.index
    }

    fn missing(&mut self) {
        self.result = Ok(None);
    }

    fn error(&mut self, error: QueryError) {
        self.result = Err(error);
    }

    fn apply(&mut self, name: &[u8], descriptor: TensorDescriptor, data: &[u8]) {
        self.result = Ok(Some((self.callback)(name, descriptor, data)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unexpected_events_report_typed_error_and_recovery() {
        let mut context = TensorContext;
        assert_eq!(context.effect_unexpected(), Err(()));

        let mut event = WithTensor::new(
            0,
            |_name: &[u8], _descriptor: TensorDescriptor, _data: &[u8]| (),
        );
        record_dispatch_result(&Err::<(), ()>(()), &mut event);
        assert_eq!(event.result, Err(QueryError::Internal));
        process(false, None, 0, &mut event);
        assert_eq!(event.result, Err(QueryError::NotParsed));
    }
}
