//! Allocation-free indexed metadata descriptor child actor.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::needless_lifetimes,
    reason = "SML-generated generic callback signatures mirror event lifetimes"
)]

use super::detail::{
    TYPE_ARRAY, TYPE_BOOL, TYPE_FLOAT32, TYPE_FLOAT64, TYPE_INT8, TYPE_INT16, TYPE_INT32,
    TYPE_INT64, TYPE_STRING, TYPE_UINT8, TYPE_UINT16, TYPE_UINT32, TYPE_UINT64,
};
use crate::event::{
    ElementKind, MetadataDescriptor, MetadataKind, QueryError, Storage, WithMetadataDescriptor,
};

mod sm;

use sm::{MetadataDescriptorStateMachine, MetadataDescriptorStateMachineContext};

pub trait Operation {
    fn index(&self) -> u32;
    fn missing(&mut self);
    fn error(&mut self, error: QueryError);
    fn apply(&mut self, key: &[u8], descriptor: MetadataDescriptor);
}

pub struct MetadataRequest<'data, O> {
    parsed: bool,
    storage: Option<&'data Storage>,
    metadata_count: u32,
    operation: &'data mut O,
}

#[derive(Clone, Copy, Debug, Default)]
struct MetadataContext;

pub fn process<'data, O: Operation + 'data>(
    parsed: bool,
    storage: Option<&'data Storage>,
    metadata_count: u32,
    operation: &'data mut O,
) {
    let mut request = MetadataRequest {
        parsed,
        storage,
        metadata_count,
        operation,
    };
    let mut machine = MetadataDescriptorStateMachine::new(MetadataContext);
    let dispatch_result = machine.process_event(&mut request);
    record_dispatch_result(&dispatch_result, request.operation);
}

fn record_dispatch_result<T, E, O: Operation>(result: &Result<T, E>, operation: &mut O) {
    if result.is_err() {
        operation.error(QueryError::Internal);
    }
}

fn range(bytes: &[u8], offset: u32, length: u32) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    let length = usize::try_from(length).ok()?;
    bytes.get(start..start.checked_add(length)?)
}

fn selected<'data, O: Operation>(
    request: &MetadataRequest<'data, O>,
) -> Option<&'data super::KvEntry> {
    let index = usize::try_from(request.operation.index()).ok()?;
    request
        .storage?
        .kv_entries
        .get(..usize::try_from(request.metadata_count).ok()?)?
        .get(index)
}

fn selected_key<'data, O: Operation>(request: &MetadataRequest<'data, O>) -> Option<&'data [u8]> {
    let entry = selected(request)?;
    range(
        &request.storage?.kv_arena,
        entry.key_offset,
        entry.key_length,
    )
}

fn selected_value<'data, O: Operation>(request: &MetadataRequest<'data, O>) -> Option<&'data [u8]> {
    let entry = selected(request)?;
    range(
        &request.storage?.kv_arena,
        entry.value_offset,
        entry.value_length,
    )
}

fn array_schema<O: Operation>(request: &MetadataRequest<'_, O>) -> Option<(u32, u64)> {
    let value = selected_value(request)?;
    let element_wire_type = u32::from_le_bytes(value.get(..4)?.try_into().ok()?);
    let length = u64::from_le_bytes(value.get(4..12)?.try_into().ok()?);
    Some((element_wire_type, length))
}

fn selected_type<O: Operation>(request: &MetadataRequest<'_, O>, wire_type: u32) -> bool {
    selected(request).is_some_and(|entry| entry.value_type == wire_type)
        && selected_key(request).is_some()
        && selected_value(request).is_some()
}

fn selected_array_type<O: Operation>(
    request: &MetadataRequest<'_, O>,
    element_wire_type: u32,
) -> bool {
    selected_type(request, TYPE_ARRAY)
        && array_schema(request).is_some_and(|(kind, _)| kind == element_wire_type)
}

fn apply_scalar<O: Operation>(request: &mut MetadataRequest<'_, O>, kind: MetadataKind) {
    let key = selected_key(request).expect("scalar guard validates metadata key");
    request
        .operation
        .apply(key, MetadataDescriptor::scalar(kind));
}

fn apply_array<O: Operation>(request: &mut MetadataRequest<'_, O>, kind: ElementKind) {
    let key = selected_key(request).expect("array guard validates metadata key");
    let (_, length) = array_schema(request).expect("array guard validates array schema");
    request
        .operation
        .apply(key, MetadataDescriptor::array(kind, length));
}

impl MetadataDescriptorStateMachineContext for MetadataContext {
    fn guard_not_parsed<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut MetadataRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(!request.parsed)
    }

    fn guard_missing<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut MetadataRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.parsed && request.operation.index() >= request.metadata_count)
    }

    fn guard_malformed<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut MetadataRequest<'data, O>,
    ) -> Result<bool, ()> {
        let known_scalar = selected(request).is_some_and(|entry| {
            matches!(
                entry.value_type,
                TYPE_UINT8
                    | TYPE_INT8
                    | TYPE_UINT16
                    | TYPE_INT16
                    | TYPE_UINT32
                    | TYPE_INT32
                    | TYPE_FLOAT32
                    | TYPE_BOOL
                    | TYPE_STRING
                    | TYPE_UINT64
                    | TYPE_INT64
                    | TYPE_FLOAT64
            )
        });
        let known_array = array_schema(request).is_some_and(|(kind, _)| {
            matches!(
                kind,
                TYPE_UINT8
                    | TYPE_INT8
                    | TYPE_UINT16
                    | TYPE_INT16
                    | TYPE_UINT32
                    | TYPE_INT32
                    | TYPE_FLOAT32
                    | TYPE_BOOL
                    | TYPE_STRING
                    | TYPE_UINT64
                    | TYPE_INT64
                    | TYPE_FLOAT64
            )
        });
        Ok(request.parsed
            && request.operation.index() < request.metadata_count
            && (selected_key(request).is_none()
                || selected_value(request).is_none()
                || selected(request).is_none_or(|entry| {
                    if entry.value_type == TYPE_ARRAY {
                        !known_array
                    } else {
                        !known_scalar
                    }
                })))
    }

    fn guard_uint8<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_type(r, TYPE_UINT8))
    }
    fn guard_int8<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_type(r, TYPE_INT8))
    }
    fn guard_uint16<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_type(r, TYPE_UINT16))
    }
    fn guard_int16<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_type(r, TYPE_INT16))
    }
    fn guard_uint32<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_type(r, TYPE_UINT32))
    }
    fn guard_int32<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_type(r, TYPE_INT32))
    }
    fn guard_float32<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_type(r, TYPE_FLOAT32))
    }
    fn guard_bool<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_type(r, TYPE_BOOL))
    }
    fn guard_string<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_type(r, TYPE_STRING))
    }
    fn guard_uint64<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_type(r, TYPE_UINT64))
    }
    fn guard_int64<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_type(r, TYPE_INT64))
    }
    fn guard_float64<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_type(r, TYPE_FLOAT64))
    }

    fn guard_array_uint8<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_array_type(r, TYPE_UINT8))
    }
    fn guard_array_int8<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_array_type(r, TYPE_INT8))
    }
    fn guard_array_uint16<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_array_type(r, TYPE_UINT16))
    }
    fn guard_array_int16<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_array_type(r, TYPE_INT16))
    }
    fn guard_array_uint32<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_array_type(r, TYPE_UINT32))
    }
    fn guard_array_int32<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_array_type(r, TYPE_INT32))
    }
    fn guard_array_float32<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_array_type(r, TYPE_FLOAT32))
    }
    fn guard_array_bool<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_array_type(r, TYPE_BOOL))
    }
    fn guard_array_string<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_array_type(r, TYPE_STRING))
    }
    fn guard_array_uint64<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_array_type(r, TYPE_UINT64))
    }
    fn guard_array_int64<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_array_type(r, TYPE_INT64))
    }
    fn guard_array_float64<'q, 'd: 'q, O: Operation + 'd>(
        &self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<bool, ()> {
        Ok(selected_array_type(r, TYPE_FLOAT64))
    }

    fn effect_not_parsed<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        r.operation.error(QueryError::NotParsed);
        Ok(())
    }
    fn effect_missing<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        r.operation.missing();
        Ok(())
    }
    fn effect_malformed<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        r.operation.error(QueryError::Malformed);
        Ok(())
    }

    fn effect_uint8<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_scalar(r, MetadataKind::Uint8);
        Ok(())
    }
    fn effect_int8<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_scalar(r, MetadataKind::Int8);
        Ok(())
    }
    fn effect_uint16<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_scalar(r, MetadataKind::Uint16);
        Ok(())
    }
    fn effect_int16<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_scalar(r, MetadataKind::Int16);
        Ok(())
    }
    fn effect_uint32<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_scalar(r, MetadataKind::Uint32);
        Ok(())
    }
    fn effect_int32<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_scalar(r, MetadataKind::Int32);
        Ok(())
    }
    fn effect_float32<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_scalar(r, MetadataKind::Float32);
        Ok(())
    }
    fn effect_bool<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_scalar(r, MetadataKind::Bool);
        Ok(())
    }
    fn effect_string<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_scalar(r, MetadataKind::String);
        Ok(())
    }
    fn effect_uint64<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_scalar(r, MetadataKind::Uint64);
        Ok(())
    }
    fn effect_int64<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_scalar(r, MetadataKind::Int64);
        Ok(())
    }
    fn effect_float64<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_scalar(r, MetadataKind::Float64);
        Ok(())
    }

    fn effect_array_uint8<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_array(r, ElementKind::Uint8);
        Ok(())
    }
    fn effect_array_int8<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_array(r, ElementKind::Int8);
        Ok(())
    }
    fn effect_array_uint16<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_array(r, ElementKind::Uint16);
        Ok(())
    }
    fn effect_array_int16<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_array(r, ElementKind::Int16);
        Ok(())
    }
    fn effect_array_uint32<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_array(r, ElementKind::Uint32);
        Ok(())
    }
    fn effect_array_int32<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_array(r, ElementKind::Int32);
        Ok(())
    }
    fn effect_array_float32<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_array(r, ElementKind::Float32);
        Ok(())
    }
    fn effect_array_bool<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_array(r, ElementKind::Bool);
        Ok(())
    }
    fn effect_array_string<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_array(r, ElementKind::String);
        Ok(())
    }
    fn effect_array_uint64<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_array(r, ElementKind::Uint64);
        Ok(())
    }
    fn effect_array_int64<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_array(r, ElementKind::Int64);
        Ok(())
    }
    fn effect_array_float64<'q, 'd: 'q, O: Operation + 'd>(
        &mut self,
        r: &'q mut MetadataRequest<'d, O>,
    ) -> Result<(), ()> {
        apply_array(r, ElementKind::Float64);
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

impl<F, R> Operation for WithMetadataDescriptor<F, R>
where
    F: for<'value> FnMut(&'value [u8], MetadataDescriptor) -> R,
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
    fn apply(&mut self, key: &[u8], descriptor: MetadataDescriptor) {
        self.result = Ok(Some((self.callback)(key, descriptor)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unexpected_events_report_typed_error_and_recovery() {
        let mut context = MetadataContext;
        assert_eq!(context.effect_unexpected(), Err(()));

        let mut event =
            WithMetadataDescriptor::new(0, |_key: &[u8], _descriptor: MetadataDescriptor| ());
        record_dispatch_result(&Err::<(), ()>(()), &mut event);
        assert_eq!(event.result, Err(QueryError::Internal));
        process(false, None, 0, &mut event);
        assert_eq!(event.result, Err(QueryError::NotParsed));
    }
}
