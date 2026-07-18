//! Allocation-free typed metadata query child actor.

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
    ElementKind, QueryError, ReadArrayLength, ReadBool, ReadBoolArrayElement, ReadF32,
    ReadF32ArrayElement, ReadF64, ReadF64ArrayElement, ReadSigned, ReadSignedArrayElement,
    ReadStringArrayMetrics, ReadUnsigned, ReadUnsignedArrayElement, ReadUnsignedArrayMetrics,
    Storage, StringArrayMetrics, UnsignedArrayMetrics, VisitF32Array, VisitStringArray,
    VisitUnsignedArray, WithByteArray, WithString, WithStringArrayElement,
};

mod sm;

use sm::{MetadataQueryStateMachine, MetadataQueryStateMachineContext};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Decision {
    Accept,
    TypeMismatch,
    IndexOutOfBounds,
    Range,
}

#[derive(Clone, Copy)]
pub struct Entry<'a> {
    kind: u32,
    bytes: &'a [u8],
    string_array_bytes: u64,
    validated: bool,
}

#[derive(Clone, Copy)]
pub struct Array<'a> {
    kind: u32,
    count: u64,
    payload: &'a [u8],
    string_array_bytes: u64,
}

pub trait Operation {
    fn key(&self) -> &[u8];
    fn decision(&self, entry: Entry<'_>) -> Decision;
    fn missing(&mut self);
    fn error(&mut self, error: QueryError);

    fn apply_uint8(&mut self, _value: u8) {
        self.error(QueryError::Internal);
    }
    fn apply_int8(&mut self, _value: i8) {
        self.error(QueryError::Internal);
    }
    fn apply_uint16(&mut self, _value: u16) {
        self.error(QueryError::Internal);
    }
    fn apply_int16(&mut self, _value: i16) {
        self.error(QueryError::Internal);
    }
    fn apply_uint32(&mut self, _value: u32) {
        self.error(QueryError::Internal);
    }
    fn apply_int32(&mut self, _value: i32) {
        self.error(QueryError::Internal);
    }
    fn apply_float32(&mut self, _value: f32) {
        self.error(QueryError::Internal);
    }
    fn apply_bool(&mut self, _value: bool) {
        self.error(QueryError::Internal);
    }
    fn apply_string(&mut self, _value: &[u8]) {
        self.error(QueryError::Internal);
    }
    fn apply_uint64(&mut self, _value: u64) {
        self.error(QueryError::Internal);
    }
    fn apply_int64(&mut self, _value: i64) {
        self.error(QueryError::Internal);
    }
    fn apply_float64(&mut self, _value: f64) {
        self.error(QueryError::Internal);
    }
    fn apply_array_uint8(&mut self, _array: Array<'_>) {
        self.error(QueryError::Internal);
    }
    fn apply_array_int8(&mut self, _array: Array<'_>) {
        self.error(QueryError::Internal);
    }
    fn apply_array_uint16(&mut self, _array: Array<'_>) {
        self.error(QueryError::Internal);
    }
    fn apply_array_int16(&mut self, _array: Array<'_>) {
        self.error(QueryError::Internal);
    }
    fn apply_array_uint32(&mut self, _array: Array<'_>) {
        self.error(QueryError::Internal);
    }
    fn apply_array_int32(&mut self, _array: Array<'_>) {
        self.error(QueryError::Internal);
    }
    fn apply_array_float32(&mut self, _array: Array<'_>) {
        self.error(QueryError::Internal);
    }
    fn apply_array_bool(&mut self, _array: Array<'_>) {
        self.error(QueryError::Internal);
    }
    fn apply_array_string(&mut self, _array: Array<'_>) {
        self.error(QueryError::Internal);
    }
    fn apply_array_uint64(&mut self, _array: Array<'_>) {
        self.error(QueryError::Internal);
    }
    fn apply_array_int64(&mut self, _array: Array<'_>) {
        self.error(QueryError::Internal);
    }
    fn apply_array_float64(&mut self, _array: Array<'_>) {
        self.error(QueryError::Internal);
    }
}

pub struct QueryRequest<'data, O> {
    parsed: bool,
    storage: Option<&'data Storage>,
    entry_count: u32,
    operation: &'data mut O,
}

#[derive(Clone, Copy, Debug, Default)]
struct QueryContext;

pub fn process<'data, O: Operation + 'data>(
    parsed: bool,
    storage: Option<&'data Storage>,
    entry_count: u32,
    operation: &'data mut O,
) {
    let mut request = QueryRequest {
        parsed,
        storage,
        entry_count,
        operation,
    };
    let mut machine = MetadataQueryStateMachine::new(QueryContext);
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

fn lookup<'a>(
    storage: Option<&'a Storage>,
    entry_count: u32,
    target_key: &[u8],
) -> Result<Option<Entry<'a>>, ()> {
    let storage = storage.ok_or(())?;
    let count = usize::try_from(entry_count).map_err(|_| ())?;
    let entries = storage.kv_entries.get(..count).ok_or(())?;
    for entry in entries {
        let entry_key = range(&storage.kv_arena, entry.key_offset, entry.key_length).ok_or(())?;
        if entry_key == target_key {
            let bytes =
                range(&storage.kv_arena, entry.value_offset, entry.value_length).ok_or(())?;
            return Ok(Some(Entry {
                kind: entry.value_type,
                bytes,
                string_array_bytes: entry.string_array_bytes,
                validated: entry.validated,
            }));
        }
    }
    Ok(None)
}

fn lookup_request<'data, O: Operation>(
    request: &QueryRequest<'data, O>,
) -> Result<Option<Entry<'data>>, ()> {
    lookup(
        request.storage,
        request.entry_count,
        request.operation.key(),
    )
}

fn read_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes(bytes[..2].try_into().expect("validated u16 metadata"))
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes[..4].try_into().expect("validated u32 metadata"))
}

fn read_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().expect("validated u64 metadata"))
}

const I64_MAX_U64: u64 = 0x7fff_ffff_ffff_ffff;

const fn scalar_size(kind: u32) -> Option<usize> {
    match kind {
        TYPE_UINT8 | TYPE_INT8 | TYPE_BOOL => Some(1),
        TYPE_UINT16 | TYPE_INT16 => Some(2),
        TYPE_UINT32 | TYPE_INT32 | TYPE_FLOAT32 => Some(4),
        TYPE_UINT64 | TYPE_INT64 | TYPE_FLOAT64 => Some(8),
        _ => None,
    }
}

fn array(entry: Entry<'_>) -> Option<Array<'_>> {
    if entry.kind != TYPE_ARRAY || entry.bytes.len() < 12 {
        return None;
    }
    Some(Array {
        kind: read_u32(entry.bytes),
        count: read_u64(&entry.bytes[4..]),
        payload: &entry.bytes[12..],
        string_array_bytes: entry.string_array_bytes,
    })
}

const fn valid_entry(entry: Entry<'_>) -> bool {
    entry.validated
}

fn decision<'data, O: Operation + 'data>(request: &QueryRequest<'data, O>) -> Option<Decision> {
    lookup_request(request)
        .ok()
        .flatten()
        .filter(|entry| valid_entry(*entry))
        .map(|entry| request.operation.decision(entry))
}

fn accepted_kind<'data, O: Operation + 'data>(request: &QueryRequest<'data, O>, kind: u32) -> bool {
    lookup_request(request).is_ok_and(|entry| {
        entry.is_some_and(|entry| {
            valid_entry(entry)
                && entry.kind == kind
                && request.operation.decision(entry) == Decision::Accept
        })
    })
}

fn accepted_array_kind<'data, O: Operation + 'data>(
    request: &QueryRequest<'data, O>,
    kind: u32,
) -> bool {
    lookup_request(request).is_ok_and(|entry| {
        entry.is_some_and(|entry| {
            valid_entry(entry)
                && array(entry).is_some_and(|array| array.kind == kind)
                && request.operation.decision(entry) == Decision::Accept
        })
    })
}

impl MetadataQueryStateMachineContext for QueryContext {
    fn guard_not_parsed<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(!request.parsed)
    }

    fn guard_missing<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.parsed && lookup_request(request).is_ok_and(|entry| entry.is_none()))
    }

    fn guard_malformed<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.parsed
            && match lookup_request(request) {
                Err(()) => true,
                Ok(Some(entry)) => !valid_entry(entry),
                Ok(None) => false,
            })
    }

    fn guard_type_mismatch<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(decision(request) == Some(Decision::TypeMismatch))
    }

    fn guard_index_out_of_bounds<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(decision(request) == Some(Decision::IndexOutOfBounds))
    }

    fn guard_range<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(decision(request) == Some(Decision::Range))
    }

    fn guard_uint8<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_kind(request, TYPE_UINT8))
    }
    fn guard_int8<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_kind(request, TYPE_INT8))
    }
    fn guard_uint16<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_kind(request, TYPE_UINT16))
    }
    fn guard_int16<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_kind(request, TYPE_INT16))
    }
    fn guard_uint32<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_kind(request, TYPE_UINT32))
    }
    fn guard_int32<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_kind(request, TYPE_INT32))
    }
    fn guard_float32<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_kind(request, TYPE_FLOAT32))
    }
    fn guard_bool<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_kind(request, TYPE_BOOL))
    }
    fn guard_string<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_kind(request, TYPE_STRING))
    }
    fn guard_uint64<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_kind(request, TYPE_UINT64))
    }
    fn guard_int64<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_kind(request, TYPE_INT64))
    }
    fn guard_float64<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_kind(request, TYPE_FLOAT64))
    }

    fn guard_array_uint8<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_array_kind(request, TYPE_UINT8))
    }
    fn guard_array_int8<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_array_kind(request, TYPE_INT8))
    }
    fn guard_array_uint16<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_array_kind(request, TYPE_UINT16))
    }
    fn guard_array_int16<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_array_kind(request, TYPE_INT16))
    }
    fn guard_array_uint32<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_array_kind(request, TYPE_UINT32))
    }
    fn guard_array_int32<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_array_kind(request, TYPE_INT32))
    }
    fn guard_array_float32<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_array_kind(request, TYPE_FLOAT32))
    }
    fn guard_array_bool<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_array_kind(request, TYPE_BOOL))
    }
    fn guard_array_string<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_array_kind(request, TYPE_STRING))
    }
    fn guard_array_uint64<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_array_kind(request, TYPE_UINT64))
    }
    fn guard_array_int64<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_array_kind(request, TYPE_INT64))
    }
    fn guard_array_float64<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(accepted_array_kind(request, TYPE_FLOAT64))
    }

    fn effect_not_parsed<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(QueryError::NotParsed);
        Ok(())
    }
    fn effect_missing<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.missing();
        Ok(())
    }
    fn effect_malformed<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(QueryError::Malformed);
        Ok(())
    }
    fn effect_type_mismatch<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(QueryError::TypeMismatch);
        Ok(())
    }
    fn effect_index_out_of_bounds<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(QueryError::IndexOutOfBounds);
        Ok(())
    }
    fn effect_range<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(QueryError::Range);
        Ok(())
    }

    fn effect_uint8<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.apply_uint8(entry(request).bytes[0]);
        Ok(())
    }
    fn effect_int8<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply_int8(i8::from_le_bytes([entry(request).bytes[0]]));
        Ok(())
    }
    fn effect_uint16<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply_uint16(read_u16(entry(request).bytes));
        Ok(())
    }
    fn effect_int16<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.apply_int16(i16::from_le_bytes(
            read_u16(entry(request).bytes).to_le_bytes(),
        ));
        Ok(())
    }
    fn effect_uint32<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply_uint32(read_u32(entry(request).bytes));
        Ok(())
    }
    fn effect_int32<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.apply_int32(i32::from_le_bytes(
            read_u32(entry(request).bytes).to_le_bytes(),
        ));
        Ok(())
    }
    fn effect_float32<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply_float32(f32::from_bits(read_u32(entry(request).bytes)));
        Ok(())
    }
    fn effect_bool<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.apply_bool(entry(request).bytes[0] != 0);
        Ok(())
    }
    fn effect_string<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        let value = string(entry(request));
        request.operation.apply_string(value);
        Ok(())
    }
    fn effect_uint64<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply_uint64(read_u64(entry(request).bytes));
        Ok(())
    }
    fn effect_int64<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.apply_int64(i64::from_le_bytes(
            read_u64(entry(request).bytes).to_le_bytes(),
        ));
        Ok(())
    }
    fn effect_float64<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request
            .operation
            .apply_float64(f64::from_bits(read_u64(entry(request).bytes)));
        Ok(())
    }

    fn effect_array_uint8<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        let value = array_entry(request);
        request.operation.apply_array_uint8(value);
        Ok(())
    }
    fn effect_array_int8<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        let value = array_entry(request);
        request.operation.apply_array_int8(value);
        Ok(())
    }
    fn effect_array_uint16<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        let value = array_entry(request);
        request.operation.apply_array_uint16(value);
        Ok(())
    }
    fn effect_array_int16<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        let value = array_entry(request);
        request.operation.apply_array_int16(value);
        Ok(())
    }
    fn effect_array_uint32<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        let value = array_entry(request);
        request.operation.apply_array_uint32(value);
        Ok(())
    }
    fn effect_array_int32<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        let value = array_entry(request);
        request.operation.apply_array_int32(value);
        Ok(())
    }
    fn effect_array_float32<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        let value = array_entry(request);
        request.operation.apply_array_float32(value);
        Ok(())
    }
    fn effect_array_bool<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        let value = array_entry(request);
        request.operation.apply_array_bool(value);
        Ok(())
    }
    fn effect_array_string<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        let value = array_entry(request);
        request.operation.apply_array_string(value);
        Ok(())
    }
    fn effect_array_uint64<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        let value = array_entry(request);
        request.operation.apply_array_uint64(value);
        Ok(())
    }
    fn effect_array_int64<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        let value = array_entry(request);
        request.operation.apply_array_int64(value);
        Ok(())
    }
    fn effect_array_float64<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        let value = array_entry(request);
        request.operation.apply_array_float64(value);
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

fn entry<'data, O: Operation>(request: &QueryRequest<'data, O>) -> Entry<'data> {
    lookup_request(request)
        .expect("accepted query has valid storage")
        .expect("accepted query has matching key")
}

fn array_entry<'data, O: Operation>(request: &QueryRequest<'data, O>) -> Array<'data> {
    array(entry(request)).expect("accepted array query has valid header")
}

fn string(entry: Entry<'_>) -> &[u8] {
    let length = usize::try_from(read_u64(entry.bytes)).expect("validated string length");
    &entry.bytes[8..8 + length]
}

const fn integer_decision(kind: u32) -> Decision {
    match kind {
        TYPE_UINT8 | TYPE_INT8 | TYPE_UINT16 | TYPE_INT16 | TYPE_UINT32 | TYPE_INT32
        | TYPE_UINT64 | TYPE_INT64 => Decision::Accept,
        _ => Decision::TypeMismatch,
    }
}

fn array_integer_decision(entry: Entry<'_>, index: u64, signed: bool) -> Decision {
    let Some(array) = array(entry) else {
        return Decision::TypeMismatch;
    };
    if integer_decision(array.kind) != Decision::Accept {
        return Decision::TypeMismatch;
    }
    if index >= array.count {
        return Decision::IndexOutOfBounds;
    }
    if signed && array.kind == TYPE_UINT64 && read_array_u64(array, index) > I64_MAX_U64 {
        return Decision::Range;
    }
    Decision::Accept
}

fn read_array_u64(array: Array<'_>, index: u64) -> u64 {
    let size = scalar_size(array.kind).expect("validated scalar array kind");
    let offset = usize::try_from(index).expect("validated array index") * size;
    let bytes = &array.payload[offset..offset + size];
    match array.kind {
        TYPE_UINT8 => u64::from(bytes[0]),
        TYPE_INT8 => u64::from_le_bytes(i64::from(i8::from_le_bytes([bytes[0]])).to_le_bytes()),
        TYPE_UINT16 | TYPE_INT16 => u64::from(read_u16(bytes)),
        TYPE_UINT32 | TYPE_INT32 => u64::from(read_u32(bytes)),
        TYPE_UINT64 | TYPE_INT64 => read_u64(bytes),
        _ => unreachable!("integer decision guarantees integer kind"),
    }
}

fn array_element_bytes(array: Array<'_>, index: u64, size: usize) -> &[u8] {
    let offset = usize::try_from(index).expect("validated array index") * size;
    &array.payload[offset..offset + size]
}

fn for_each_array_value<T>(
    array: Array<'_>,
    size: usize,
    mut decode: impl FnMut(&[u8]) -> T,
    mut consume: impl FnMut(u32, T),
) {
    for index in 0..array.count {
        consume(
            u32::try_from(index).expect("GGUF array count fits u32"),
            decode(array_element_bytes(array, index, size)),
        );
    }
}

fn unsigned_metrics(
    array: Array<'_>,
    size: usize,
    decode: impl FnMut(&[u8]) -> u64,
) -> UnsignedArrayMetrics {
    let mut maximum = 0_u64;
    for_each_array_value(array, size, decode, |_, value| maximum = maximum.max(value));
    UnsignedArrayMetrics::new(array.count, maximum)
}

fn decode_u8(bytes: &[u8]) -> u64 {
    u64::from(bytes[0])
}
fn decode_i8_raw(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(i64::from(i8::from_le_bytes([bytes[0]])).to_le_bytes())
}
fn decode_u16_raw(bytes: &[u8]) -> u64 {
    u64::from(read_u16(bytes))
}
fn decode_u32_raw(bytes: &[u8]) -> u64 {
    u64::from(read_u32(bytes))
}
fn decode_u64_raw(bytes: &[u8]) -> u64 {
    read_u64(bytes)
}
fn decode_f32(bytes: &[u8]) -> f32 {
    f32::from_bits(read_u32(bytes))
}
#[allow(
    clippy::cast_possible_truncation,
    reason = "GGUF float64 arrays expose f32 coercion"
)]
fn decode_f64_as_f32(bytes: &[u8]) -> f32 {
    f64::from_bits(read_u64(bytes)) as f32
}

macro_rules! result_methods {
    ($value:ty) => {
        fn missing(&mut self) {
            self.result = Ok(None);
        }
        fn error(&mut self, error: QueryError) {
            self.result = Err(error);
        }
    };
}

impl Operation for ReadUnsigned<'_> {
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        integer_decision(entry.kind)
    }
    result_methods!(u64);
    fn apply_uint8(&mut self, value: u8) {
        self.result = Ok(Some(u64::from(value)));
    }
    fn apply_int8(&mut self, value: i8) {
        self.result = Ok(Some(u64::from_le_bytes(i64::from(value).to_le_bytes())));
    }
    fn apply_uint16(&mut self, value: u16) {
        self.result = Ok(Some(u64::from(value)));
    }
    fn apply_int16(&mut self, value: i16) {
        self.result = Ok(Some(u64::from(u16::from_le_bytes(value.to_le_bytes()))));
    }
    fn apply_uint32(&mut self, value: u32) {
        self.result = Ok(Some(u64::from(value)));
    }
    fn apply_int32(&mut self, value: i32) {
        self.result = Ok(Some(u64::from(u32::from_le_bytes(value.to_le_bytes()))));
    }
    fn apply_uint64(&mut self, value: u64) {
        self.result = Ok(Some(value));
    }
    fn apply_int64(&mut self, value: i64) {
        self.result = Ok(Some(u64::from_le_bytes(value.to_le_bytes())));
    }
}

impl Operation for ReadSigned<'_> {
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        if entry.kind == TYPE_UINT64 && read_u64(entry.bytes) > I64_MAX_U64 {
            Decision::Range
        } else {
            integer_decision(entry.kind)
        }
    }
    result_methods!(i64);
    fn apply_uint8(&mut self, value: u8) {
        self.result = Ok(Some(i64::from(value)));
    }
    fn apply_int8(&mut self, value: i8) {
        self.result = Ok(Some(i64::from(value)));
    }
    fn apply_uint16(&mut self, value: u16) {
        self.result = Ok(Some(i64::from(value)));
    }
    fn apply_int16(&mut self, value: i16) {
        self.result = Ok(Some(i64::from(value)));
    }
    fn apply_uint32(&mut self, value: u32) {
        self.result = Ok(Some(i64::from(value)));
    }
    fn apply_int32(&mut self, value: i32) {
        self.result = Ok(Some(i64::from(value)));
    }
    fn apply_uint64(&mut self, value: u64) {
        self.result = Ok(Some(
            i64::try_from(value).expect("range guard validates signed conversion"),
        ));
    }
    fn apply_int64(&mut self, value: i64) {
        self.result = Ok(Some(value));
    }
}

impl Operation for ReadF32<'_> {
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        if matches!(entry.kind, TYPE_FLOAT32 | TYPE_FLOAT64) {
            Decision::Accept
        } else {
            Decision::TypeMismatch
        }
    }
    result_methods!(f32);
    fn apply_float32(&mut self, value: f32) {
        self.result = Ok(Some(value));
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "emel.cpp intentionally exposes both GGUF float widths as f32"
    )]
    fn apply_float64(&mut self, value: f64) {
        self.result = Ok(Some(value as f32));
    }
}

impl Operation for ReadF64<'_> {
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        if matches!(entry.kind, TYPE_FLOAT32 | TYPE_FLOAT64) {
            Decision::Accept
        } else {
            Decision::TypeMismatch
        }
    }
    result_methods!(f64);
    fn apply_float32(&mut self, value: f32) {
        self.result = Ok(Some(f64::from(value)));
    }
    fn apply_float64(&mut self, value: f64) {
        self.result = Ok(Some(value));
    }
}

impl Operation for ReadBool<'_> {
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        if entry.kind == TYPE_BOOL {
            Decision::Accept
        } else {
            Decision::TypeMismatch
        }
    }
    result_methods!(bool);
    fn apply_bool(&mut self, value: bool) {
        self.result = Ok(Some(value));
    }
}

impl<F, R> Operation for WithString<'_, F, R>
where
    F: for<'value> FnMut(&'value [u8]) -> R,
{
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        if entry.kind == TYPE_STRING {
            Decision::Accept
        } else {
            Decision::TypeMismatch
        }
    }
    fn missing(&mut self) {
        self.result = Ok(None);
    }
    fn error(&mut self, error: QueryError) {
        self.result = Err(error);
    }
    fn apply_string(&mut self, value: &[u8]) {
        self.result = Ok(Some((self.callback)(value)));
    }
}

impl Operation for ReadArrayLength<'_> {
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        let Some(array) = array(entry) else {
            return Decision::TypeMismatch;
        };
        if element_kind(self.element_kind) == array.kind {
            Decision::Accept
        } else {
            Decision::TypeMismatch
        }
    }
    result_methods!(u64);
    fn apply_array_uint8(&mut self, value: Array<'_>) {
        self.result = Ok(Some(value.count));
    }
    fn apply_array_int8(&mut self, value: Array<'_>) {
        self.result = Ok(Some(value.count));
    }
    fn apply_array_uint16(&mut self, value: Array<'_>) {
        self.result = Ok(Some(value.count));
    }
    fn apply_array_int16(&mut self, value: Array<'_>) {
        self.result = Ok(Some(value.count));
    }
    fn apply_array_uint32(&mut self, value: Array<'_>) {
        self.result = Ok(Some(value.count));
    }
    fn apply_array_int32(&mut self, value: Array<'_>) {
        self.result = Ok(Some(value.count));
    }
    fn apply_array_float32(&mut self, value: Array<'_>) {
        self.result = Ok(Some(value.count));
    }
    fn apply_array_bool(&mut self, value: Array<'_>) {
        self.result = Ok(Some(value.count));
    }
    fn apply_array_string(&mut self, value: Array<'_>) {
        self.result = Ok(Some(value.count));
    }
    fn apply_array_uint64(&mut self, value: Array<'_>) {
        self.result = Ok(Some(value.count));
    }
    fn apply_array_int64(&mut self, value: Array<'_>) {
        self.result = Ok(Some(value.count));
    }
    fn apply_array_float64(&mut self, value: Array<'_>) {
        self.result = Ok(Some(value.count));
    }
}

impl Operation for ReadStringArrayMetrics<'_> {
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        if array(entry).is_some_and(|value| value.kind == TYPE_STRING) {
            Decision::Accept
        } else {
            Decision::TypeMismatch
        }
    }
    result_methods!(StringArrayMetrics);
    fn apply_array_string(&mut self, value: Array<'_>) {
        self.result = Ok(Some(StringArrayMetrics::new(
            value.count,
            value.string_array_bytes,
        )));
    }
}

const fn element_kind(kind: ElementKind) -> u32 {
    match kind {
        ElementKind::Uint8 => TYPE_UINT8,
        ElementKind::Int8 => TYPE_INT8,
        ElementKind::Uint16 => TYPE_UINT16,
        ElementKind::Int16 => TYPE_INT16,
        ElementKind::Uint32 => TYPE_UINT32,
        ElementKind::Int32 => TYPE_INT32,
        ElementKind::Float32 => TYPE_FLOAT32,
        ElementKind::Bool => TYPE_BOOL,
        ElementKind::String => TYPE_STRING,
        ElementKind::Uint64 => TYPE_UINT64,
        ElementKind::Int64 => TYPE_INT64,
        ElementKind::Float64 => TYPE_FLOAT64,
    }
}

impl Operation for ReadUnsignedArrayElement<'_> {
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        array_integer_decision(entry, self.index, false)
    }
    result_methods!(u64);
    fn apply_array_uint8(&mut self, value: Array<'_>) {
        self.result = Ok(Some(u64::from(
            array_element_bytes(value, self.index, 1)[0],
        )));
    }
    fn apply_array_int8(&mut self, value: Array<'_>) {
        let item = i8::from_le_bytes([array_element_bytes(value, self.index, 1)[0]]);
        self.result = Ok(Some(u64::from_le_bytes(i64::from(item).to_le_bytes())));
    }
    fn apply_array_uint16(&mut self, value: Array<'_>) {
        self.result = Ok(Some(u64::from(read_u16(array_element_bytes(
            value, self.index, 2,
        )))));
    }
    fn apply_array_int16(&mut self, value: Array<'_>) {
        self.result = Ok(Some(u64::from(read_u16(array_element_bytes(
            value, self.index, 2,
        )))));
    }
    fn apply_array_uint32(&mut self, value: Array<'_>) {
        self.result = Ok(Some(u64::from(read_u32(array_element_bytes(
            value, self.index, 4,
        )))));
    }
    fn apply_array_int32(&mut self, value: Array<'_>) {
        self.result = Ok(Some(u64::from(read_u32(array_element_bytes(
            value, self.index, 4,
        )))));
    }
    fn apply_array_uint64(&mut self, value: Array<'_>) {
        self.result = Ok(Some(read_u64(array_element_bytes(value, self.index, 8))));
    }
    fn apply_array_int64(&mut self, value: Array<'_>) {
        self.result = Ok(Some(read_u64(array_element_bytes(value, self.index, 8))));
    }
}

impl Operation for ReadUnsignedArrayMetrics<'_> {
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        array(entry).map_or(Decision::TypeMismatch, |value| integer_decision(value.kind))
    }
    result_methods!(UnsignedArrayMetrics);
    fn apply_array_uint8(&mut self, v: Array<'_>) {
        self.result = Ok(Some(unsigned_metrics(v, 1, decode_u8)));
    }
    fn apply_array_int8(&mut self, v: Array<'_>) {
        self.result = Ok(Some(unsigned_metrics(v, 1, decode_i8_raw)));
    }
    fn apply_array_uint16(&mut self, v: Array<'_>) {
        self.result = Ok(Some(unsigned_metrics(v, 2, decode_u16_raw)));
    }
    fn apply_array_int16(&mut self, v: Array<'_>) {
        self.result = Ok(Some(unsigned_metrics(v, 2, decode_u16_raw)));
    }
    fn apply_array_uint32(&mut self, v: Array<'_>) {
        self.result = Ok(Some(unsigned_metrics(v, 4, decode_u32_raw)));
    }
    fn apply_array_int32(&mut self, v: Array<'_>) {
        self.result = Ok(Some(unsigned_metrics(v, 4, decode_u32_raw)));
    }
    fn apply_array_uint64(&mut self, v: Array<'_>) {
        self.result = Ok(Some(unsigned_metrics(v, 8, decode_u64_raw)));
    }
    fn apply_array_int64(&mut self, v: Array<'_>) {
        self.result = Ok(Some(unsigned_metrics(v, 8, decode_u64_raw)));
    }
}

fn publish_unsigned_visit<F>(
    array: Array<'_>,
    size: usize,
    decode: impl FnMut(&[u8]) -> u64,
    visitor: &mut F,
) where
    F: FnMut(u32, u64),
{
    for_each_array_value(array, size, decode, visitor);
}

impl<F> Operation for VisitUnsignedArray<'_, F>
where
    F: FnMut(u32, u64),
{
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        array(entry).map_or(Decision::TypeMismatch, |value| integer_decision(value.kind))
    }
    fn missing(&mut self) {
        self.result = Ok(None);
    }
    fn error(&mut self, error: QueryError) {
        self.result = Err(error);
    }
    fn apply_array_uint8(&mut self, v: Array<'_>) {
        publish_unsigned_visit(v, 1, decode_u8, &mut self.visitor);
        self.result = Ok(Some(v.count));
    }
    fn apply_array_int8(&mut self, v: Array<'_>) {
        publish_unsigned_visit(v, 1, decode_i8_raw, &mut self.visitor);
        self.result = Ok(Some(v.count));
    }
    fn apply_array_uint16(&mut self, v: Array<'_>) {
        publish_unsigned_visit(v, 2, decode_u16_raw, &mut self.visitor);
        self.result = Ok(Some(v.count));
    }
    fn apply_array_int16(&mut self, v: Array<'_>) {
        publish_unsigned_visit(v, 2, decode_u16_raw, &mut self.visitor);
        self.result = Ok(Some(v.count));
    }
    fn apply_array_uint32(&mut self, v: Array<'_>) {
        publish_unsigned_visit(v, 4, decode_u32_raw, &mut self.visitor);
        self.result = Ok(Some(v.count));
    }
    fn apply_array_int32(&mut self, v: Array<'_>) {
        publish_unsigned_visit(v, 4, decode_u32_raw, &mut self.visitor);
        self.result = Ok(Some(v.count));
    }
    fn apply_array_uint64(&mut self, v: Array<'_>) {
        publish_unsigned_visit(v, 8, decode_u64_raw, &mut self.visitor);
        self.result = Ok(Some(v.count));
    }
    fn apply_array_int64(&mut self, v: Array<'_>) {
        publish_unsigned_visit(v, 8, decode_u64_raw, &mut self.visitor);
        self.result = Ok(Some(v.count));
    }
}

impl<F> Operation for VisitF32Array<'_, F>
where
    F: FnMut(u32, f32),
{
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        if array(entry).is_some_and(|value| matches!(value.kind, TYPE_FLOAT32 | TYPE_FLOAT64)) {
            Decision::Accept
        } else {
            Decision::TypeMismatch
        }
    }
    fn missing(&mut self) {
        self.result = Ok(None);
    }
    fn error(&mut self, error: QueryError) {
        self.result = Err(error);
    }
    fn apply_array_float32(&mut self, v: Array<'_>) {
        for_each_array_value(v, 4, decode_f32, &mut self.visitor);
        self.result = Ok(Some(v.count));
    }
    fn apply_array_float64(&mut self, v: Array<'_>) {
        for_each_array_value(v, 8, decode_f64_as_f32, &mut self.visitor);
        self.result = Ok(Some(v.count));
    }
}

impl Operation for ReadSignedArrayElement<'_> {
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        array_integer_decision(entry, self.index, true)
    }
    result_methods!(i64);
    fn apply_array_uint8(&mut self, value: Array<'_>) {
        self.result = Ok(Some(i64::from(
            array_element_bytes(value, self.index, 1)[0],
        )));
    }
    fn apply_array_int8(&mut self, value: Array<'_>) {
        let item = i8::from_le_bytes([array_element_bytes(value, self.index, 1)[0]]);
        self.result = Ok(Some(i64::from(item)));
    }
    fn apply_array_uint16(&mut self, value: Array<'_>) {
        self.result = Ok(Some(i64::from(read_u16(array_element_bytes(
            value, self.index, 2,
        )))));
    }
    fn apply_array_int16(&mut self, value: Array<'_>) {
        let item =
            i16::from_le_bytes(read_u16(array_element_bytes(value, self.index, 2)).to_le_bytes());
        self.result = Ok(Some(i64::from(item)));
    }
    fn apply_array_uint32(&mut self, value: Array<'_>) {
        self.result = Ok(Some(i64::from(read_u32(array_element_bytes(
            value, self.index, 4,
        )))));
    }
    fn apply_array_int32(&mut self, value: Array<'_>) {
        let item =
            i32::from_le_bytes(read_u32(array_element_bytes(value, self.index, 4)).to_le_bytes());
        self.result = Ok(Some(i64::from(item)));
    }
    fn apply_array_uint64(&mut self, value: Array<'_>) {
        self.result = Ok(Some(
            i64::try_from(read_u64(array_element_bytes(value, self.index, 8)))
                .expect("range guard validates signed conversion"),
        ));
    }
    fn apply_array_int64(&mut self, value: Array<'_>) {
        self.result = Ok(Some(i64::from_le_bytes(
            read_u64(array_element_bytes(value, self.index, 8)).to_le_bytes(),
        )));
    }
}

impl Operation for ReadF32ArrayElement<'_> {
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        let Some(value) = array(entry) else {
            return Decision::TypeMismatch;
        };
        if !matches!(value.kind, TYPE_FLOAT32 | TYPE_FLOAT64) {
            Decision::TypeMismatch
        } else if self.index >= value.count {
            Decision::IndexOutOfBounds
        } else {
            Decision::Accept
        }
    }
    result_methods!(f32);
    fn apply_array_float32(&mut self, value: Array<'_>) {
        let offset = usize::try_from(self.index).expect("valid index") * 4;
        self.result = Ok(Some(f32::from_bits(read_u32(&value.payload[offset..]))));
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "emel.cpp intentionally exposes both GGUF float widths as f32"
    )]
    fn apply_array_float64(&mut self, value: Array<'_>) {
        let offset = usize::try_from(self.index).expect("valid index") * 8;
        self.result = Ok(Some(
            f64::from_bits(read_u64(&value.payload[offset..])) as f32
        ));
    }
}

impl Operation for ReadF64ArrayElement<'_> {
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        let Some(value) = array(entry) else {
            return Decision::TypeMismatch;
        };
        if !matches!(value.kind, TYPE_FLOAT32 | TYPE_FLOAT64) {
            Decision::TypeMismatch
        } else if self.index >= value.count {
            Decision::IndexOutOfBounds
        } else {
            Decision::Accept
        }
    }
    result_methods!(f64);
    fn apply_array_float32(&mut self, value: Array<'_>) {
        let offset = usize::try_from(self.index).expect("valid index") * 4;
        self.result = Ok(Some(f64::from(f32::from_bits(read_u32(
            &value.payload[offset..],
        )))));
    }
    fn apply_array_float64(&mut self, value: Array<'_>) {
        let offset = usize::try_from(self.index).expect("valid index") * 8;
        self.result = Ok(Some(f64::from_bits(read_u64(&value.payload[offset..]))));
    }
}

impl Operation for ReadBoolArrayElement<'_> {
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        let Some(value) = array(entry) else {
            return Decision::TypeMismatch;
        };
        if value.kind != TYPE_BOOL {
            Decision::TypeMismatch
        } else if self.index >= value.count {
            Decision::IndexOutOfBounds
        } else {
            Decision::Accept
        }
    }
    result_methods!(bool);
    fn apply_array_bool(&mut self, value: Array<'_>) {
        self.result = Ok(Some(
            value.payload[usize::try_from(self.index).expect("valid index")] != 0,
        ));
    }
}

impl<F, R> Operation for WithStringArrayElement<'_, F, R>
where
    F: for<'value> FnMut(&'value [u8]) -> R,
{
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        let Some(value) = array(entry) else {
            return Decision::TypeMismatch;
        };
        if value.kind != TYPE_STRING {
            Decision::TypeMismatch
        } else if self.index >= value.count {
            Decision::IndexOutOfBounds
        } else {
            Decision::Accept
        }
    }
    fn missing(&mut self) {
        self.result = Ok(None);
    }
    fn error(&mut self, error: QueryError) {
        self.result = Err(error);
    }
    fn apply_array_string(&mut self, value: Array<'_>) {
        let item = string_array_element(value, self.index);
        self.result = Ok(Some((self.callback)(item)));
    }
}

impl<F> Operation for VisitStringArray<'_, F>
where
    F: for<'value> FnMut(u32, &'value [u8]),
{
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        if array(entry).is_some_and(|value| value.kind == TYPE_STRING) {
            Decision::Accept
        } else {
            Decision::TypeMismatch
        }
    }
    fn missing(&mut self) {
        self.result = Ok(None);
    }
    fn error(&mut self, error: QueryError) {
        self.result = Err(error);
    }
    fn apply_array_string(&mut self, value: Array<'_>) {
        let mut cursor = 0_usize;
        for index in 0..value.count {
            let length = usize::try_from(read_u64(&value.payload[cursor..]))
                .expect("validated string length");
            cursor += 8;
            let end = cursor + length;
            (self.visitor)(
                u32::try_from(index).expect("GGUF array count fits u32"),
                &value.payload[cursor..end],
            );
            cursor = end;
        }
        self.result = Ok(Some(value.count));
    }
}

fn string_array_element(array: Array<'_>, index: u64) -> &[u8] {
    let mut cursor = 0_usize;
    for current in 0..=index {
        let length =
            usize::try_from(read_u64(&array.payload[cursor..])).expect("validated string length");
        cursor += 8;
        if current == index {
            return &array.payload[cursor..cursor + length];
        }
        cursor += length;
    }
    unreachable!("validated string-array index")
}

impl<F, R> Operation for WithByteArray<'_, F, R>
where
    F: for<'value> FnMut(&'value [u8]) -> R,
{
    fn key(&self) -> &[u8] {
        self.key
    }
    fn decision(&self, entry: Entry<'_>) -> Decision {
        if array(entry).is_some_and(|value| matches!(value.kind, TYPE_UINT8 | TYPE_INT8)) {
            Decision::Accept
        } else {
            Decision::TypeMismatch
        }
    }
    fn missing(&mut self) {
        self.result = Ok(None);
    }
    fn error(&mut self, error: QueryError) {
        self.result = Err(error);
    }
    fn apply_array_uint8(&mut self, value: Array<'_>) {
        self.result = Ok(Some((self.callback)(value.payload)));
    }
    fn apply_array_int8(&mut self, value: Array<'_>) {
        self.result = Ok(Some((self.callback)(value.payload)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DefensiveOperation {
        errors: u32,
    }

    impl Operation for DefensiveOperation {
        fn key(&self) -> &[u8] {
            b"key"
        }

        fn decision(&self, _entry: Entry<'_>) -> Decision {
            Decision::Accept
        }

        fn missing(&mut self) {}

        fn error(&mut self, _error: QueryError) {
            self.errors += 1;
        }
    }

    #[test]
    fn default_handlers_report_internal_errors() {
        let mut operation = DefensiveOperation { errors: 0 };
        let array = Array {
            kind: TYPE_UINT8,
            count: 0,
            payload: &[],
            string_array_bytes: 0,
        };
        operation.apply_uint8(0);
        operation.apply_int8(0);
        operation.apply_uint16(0);
        operation.apply_int16(0);
        operation.apply_uint32(0);
        operation.apply_int32(0);
        operation.apply_float32(0.0);
        operation.apply_bool(false);
        operation.apply_string(&[]);
        operation.apply_uint64(0);
        operation.apply_int64(0);
        operation.apply_float64(0.0);
        operation.apply_array_uint8(array);
        operation.apply_array_int8(array);
        operation.apply_array_uint16(array);
        operation.apply_array_int16(array);
        operation.apply_array_uint32(array);
        operation.apply_array_int32(array);
        operation.apply_array_float32(array);
        operation.apply_array_bool(array);
        operation.apply_array_string(array);
        operation.apply_array_uint64(array);
        operation.apply_array_int64(array);
        operation.apply_array_float64(array);
        assert_eq!(operation.errors, 24);
    }

    #[test]
    fn missing_actor_storage_is_malformed() {
        let mut event = ReadUnsigned::new(b"key");
        process(true, None, 1, &mut event);
        assert_eq!(event.result, Err(QueryError::Malformed));
    }

    #[test]
    fn unvalidated_actor_owned_entry_is_malformed() {
        let storage = Storage {
            source: std::sync::Arc::from([]),
            kv_arena: vec![b'k', 7],
            kv_entries: vec![super::super::KvEntry {
                key_offset: 0,
                key_length: 1,
                value_offset: 1,
                value_length: 1,
                value_type: TYPE_UINT8,
                string_array_bytes: 0,
                validated: false,
            }],
            tensors: Vec::new(),
        };
        let mut event = ReadUnsigned::new(b"k");

        process(true, Some(&storage), 1, &mut event);

        assert_eq!(event.result, Err(QueryError::Malformed));
    }

    #[test]
    fn unexpected_events_report_typed_error_and_recovery() {
        let mut context = QueryContext;
        assert_eq!(context.effect_unexpected(), Err(()));

        let mut event = ReadUnsigned::new(b"key");
        record_dispatch_result(&Err::<(), ()>(()), &mut event);
        assert_eq!(event.result, Err(QueryError::Internal));
        process(false, None, 0, &mut event);
        assert_eq!(event.result, Err(QueryError::NotParsed));
    }
}
