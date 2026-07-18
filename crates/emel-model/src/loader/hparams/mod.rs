//! Architecture-neutral typed GGUF hyperparameter access.

use core::cell::{Cell, RefCell};

use emel_gguf::Loader;
use emel_gguf::event::{
    ElementKind, QueryError, ReadArrayLength, ReadF32, ReadUnsigned, ReadUnsignedArrayElement,
    ReadUnsignedArrayMetrics, UnsignedArrayMetrics, VisitBoolArray, VisitUnsignedArray,
};

mod sm;

use sm::{HparamAccessStateMachine, HparamAccessStateMachineContext};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operation {
    OptionalI32,
    OptionalI32OrFirstArray,
    RequiredFirstNonzeroArray,
    OptionalF32,
    RequiredFlagArray,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    Missing,
    Malformed,
    WrongKind,
    Query,
    Capacity,
    Count,
    Range,
    Unexpected,
    Internal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Error {
    pub operation: Operation,
    pub kind: ErrorKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScanOutcome {
    Pending,
    Value(u64),
    Missing,
    Query(QueryError),
    Arithmetic,
}

#[derive(Clone, Copy)]
struct Runtime<'a> {
    operation: Operation,
    gguf: &'a RefCell<&'a mut Loader>,
    key: &'a [u8],
    unsigned: &'a Cell<Result<Option<u64>, QueryError>>,
    float: &'a Cell<Result<Option<f32>, QueryError>>,
    array_length: &'a Cell<Result<Option<u64>, QueryError>>,
    unsigned_metrics: &'a Cell<Result<Option<UnsignedArrayMetrics>, QueryError>>,
    array_visit: &'a Cell<Result<Option<u64>, QueryError>>,
    scan: &'a Cell<ScanOutcome>,
    result: &'a Cell<Result<u32, ErrorKind>>,
    i32_value: &'a Cell<i32>,
    f32_value: &'a Cell<f32>,
    flags: &'a RefCell<&'a mut [u8]>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Context;

macro_rules! raw_hparam_guard {
    ($name:ident, $field:ident, $pattern:pat) => {
        fn $name(&self, event: &Runtime<'_>) -> Result<bool, ()> {
            Ok(matches!(event.$field.get(), $pattern))
        }
    };
}

pub struct Accessor {
    machine: HparamAccessStateMachine<Context>,
}

impl Accessor {
    pub const fn new() -> Self {
        Self {
            machine: HparamAccessStateMachine::new(Context),
        }
    }

    pub fn assign_i32(
        &mut self,
        gguf: &mut Loader,
        key: &[u8],
        field: &mut i32,
    ) -> Result<(), Error> {
        self.dispatch_i32(gguf, Operation::OptionalI32, key, field)
    }

    pub fn assign_i32_or_first_array_value(
        &mut self,
        gguf: &mut Loader,
        key: &[u8],
        field: &mut i32,
    ) -> Result<(), Error> {
        self.dispatch_i32(gguf, Operation::OptionalI32OrFirstArray, key, field)
    }

    pub fn assign_first_nonzero_i32_from_array(
        &mut self,
        gguf: &mut Loader,
        key: &[u8],
        field: &mut i32,
    ) -> Result<(), Error> {
        self.dispatch_i32(gguf, Operation::RequiredFirstNonzeroArray, key, field)
    }

    pub fn assign_f32(
        &mut self,
        gguf: &mut Loader,
        key: &[u8],
        field: &mut f32,
    ) -> Result<(), Error> {
        let value = Cell::new(*field);
        self.dispatch(
            gguf,
            Operation::OptionalF32,
            key,
            &Cell::new(0),
            &value,
            &mut [],
        )?;
        *field = value.get();
        Ok(())
    }

    pub fn copy_flag_array(
        &mut self,
        gguf: &mut Loader,
        key: &[u8],
        destination: &mut [u8],
    ) -> Result<u32, Error> {
        self.dispatch(
            gguf,
            Operation::RequiredFlagArray,
            key,
            &Cell::new(0),
            &Cell::new(0.0),
            destination,
        )
    }

    fn dispatch_i32(
        &mut self,
        gguf: &mut Loader,
        operation: Operation,
        key: &[u8],
        field: &mut i32,
    ) -> Result<(), Error> {
        let value = Cell::new(*field);
        self.dispatch(gguf, operation, key, &value, &Cell::new(0.0), &mut [])?;
        *field = value.get();
        Ok(())
    }

    fn dispatch(
        &mut self,
        gguf: &mut Loader,
        operation: Operation,
        key: &[u8],
        i32_value: &Cell<i32>,
        f32_value: &Cell<f32>,
        flags: &mut [u8],
    ) -> Result<u32, Error> {
        let gguf = RefCell::new(gguf);
        let unsigned = Cell::new(Err(QueryError::Internal));
        let float = Cell::new(Err(QueryError::Internal));
        let array_length = Cell::new(Err(QueryError::Internal));
        let unsigned_metrics = Cell::new(Err(QueryError::Internal));
        let array_visit = Cell::new(Err(QueryError::Internal));
        let scan = Cell::new(ScanOutcome::Pending);
        let result = Cell::new(Err(ErrorKind::Internal));
        let flags = RefCell::new(flags);
        let runtime = Runtime {
            operation,
            gguf: &gguf,
            key,
            unsigned: &unsigned,
            float: &float,
            array_length: &array_length,
            unsigned_metrics: &unsigned_metrics,
            array_visit: &array_visit,
            scan: &scan,
            result: &result,
            i32_value,
            f32_value,
            flags: &flags,
        };
        let dispatch = self
            .machine
            .process_event(sm::HparamAccessEvents::Request(runtime));
        if dispatch.is_err() {
            return Err(Error {
                operation,
                kind: ErrorKind::Internal,
            });
        }
        result.get().map_err(|kind| Error { operation, kind })
    }
}

impl Default for Accessor {
    fn default() -> Self {
        Self::new()
    }
}

impl HparamAccessStateMachineContext for Context {
    fn guard_optional_i32(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.operation == Operation::OptionalI32)
    }
    fn guard_optional_i32_or_array(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.operation == Operation::OptionalI32OrFirstArray)
    }
    fn guard_required_nonzero(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.operation == Operation::RequiredFirstNonzeroArray)
    }
    fn guard_optional_f32(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.operation == Operation::OptionalF32)
    }
    fn guard_required_flags(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.operation == Operation::RequiredFlagArray)
    }

    fn effect_read_scalar(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.unsigned.set(
            event
                .gguf
                .borrow_mut()
                .process_event(ReadUnsigned::new(event.key)),
        );
        Ok(())
    }
    fn effect_read_first(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.unsigned.set(
            event
                .gguf
                .borrow_mut()
                .process_event(ReadUnsignedArrayElement::new(event.key, 0)),
        );
        Ok(())
    }
    fn effect_read_float(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.float.set(
            event
                .gguf
                .borrow_mut()
                .process_event(ReadF32::new(event.key)),
        );
        Ok(())
    }

    fn guard_unsigned_value_in_range(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event
            .unsigned
            .get()
            .is_ok_and(|value| value.is_some_and(|value| i32::try_from(value).is_ok())))
    }
    fn guard_unsigned_value_out_of_range(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event
            .unsigned
            .get()
            .is_ok_and(|value| value.is_some_and(|value| i32::try_from(value).is_err())))
    }
    raw_hparam_guard!(guard_unsigned_missing, unsigned, Ok(None));
    raw_hparam_guard!(
        guard_unsigned_wrong_kind,
        unsigned,
        Err(QueryError::TypeMismatch)
    );
    raw_hparam_guard!(
        guard_unsigned_count,
        unsigned,
        Err(QueryError::IndexOutOfBounds)
    );
    raw_hparam_guard!(
        guard_unsigned_malformed,
        unsigned,
        Err(QueryError::Malformed)
    );
    raw_hparam_guard!(guard_unsigned_query_range, unsigned, Err(QueryError::Range));
    raw_hparam_guard!(guard_unsigned_query, unsigned, Err(QueryError::NotParsed));
    raw_hparam_guard!(guard_unsigned_internal, unsigned, Err(QueryError::Internal));
    fn effect_assign_i32(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        let value = event.unsigned.get().ok().flatten().ok_or(())?;
        event.i32_value.set(i32::try_from(value).map_err(|_| ())?);
        event.result.set(Ok(1));
        Ok(())
    }
    fn effect_preserve(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Ok(0));
        Ok(())
    }
    fn effect_range(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(ErrorKind::Range));
        Ok(())
    }
    fn effect_wrong_kind(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(ErrorKind::WrongKind));
        Ok(())
    }
    fn effect_count(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(ErrorKind::Count));
        Ok(())
    }
    fn effect_malformed(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(ErrorKind::Malformed));
        Ok(())
    }
    fn effect_query(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(ErrorKind::Query));
        Ok(())
    }
    fn effect_internal(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(ErrorKind::Internal));
        Ok(())
    }

    fn guard_float_value(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.float.get().is_ok_and(|value| value.is_some()))
    }
    raw_hparam_guard!(guard_float_missing, float, Ok(None));
    raw_hparam_guard!(guard_float_wrong_kind, float, Err(QueryError::TypeMismatch));
    raw_hparam_guard!(guard_float_malformed, float, Err(QueryError::Malformed));
    raw_hparam_guard!(guard_float_range, float, Err(QueryError::Range));
    raw_hparam_guard!(
        guard_float_query,
        float,
        Err(QueryError::NotParsed | QueryError::IndexOutOfBounds)
    );
    raw_hparam_guard!(guard_float_internal, float, Err(QueryError::Internal));
    fn effect_assign_f32(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event
            .f32_value
            .set(event.float.get().ok().flatten().ok_or(())?);
        event.result.set(Ok(1));
        Ok(())
    }
    fn effect_scan_nonzero(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        let mut index = 0_u64;
        loop {
            match event
                .gguf
                .borrow_mut()
                .process_event(ReadUnsignedArrayElement::new(event.key, index))
            {
                Ok(Some(value)) if value != 0 => {
                    event.scan.set(ScanOutcome::Value(value));
                    return Ok(());
                }
                Ok(Some(_)) => {}
                Ok(None) => {
                    event.scan.set(ScanOutcome::Missing);
                    return Ok(());
                }
                Err(error) => {
                    event.scan.set(ScanOutcome::Query(error));
                    return Ok(());
                }
            }
            let Some(next) = index.checked_add(1) else {
                event.scan.set(ScanOutcome::Arithmetic);
                return Ok(());
            };
            index = next;
        }
    }

    fn effect_read_bool_flag_length(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.array_length.set(
            event
                .gguf
                .borrow_mut()
                .process_event(ReadArrayLength::new(event.key, ElementKind::Bool)),
        );
        Ok(())
    }
    fn effect_read_integer_flag_metrics(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.unsigned_metrics.set(
            event
                .gguf
                .borrow_mut()
                .process_event(ReadUnsignedArrayMetrics::new(event.key)),
        );
        Ok(())
    }
    fn effect_visit_bool_flags(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        let mut flags = event.flags.borrow_mut();
        event
            .array_visit
            .set(event.gguf.borrow_mut().process_event(VisitBoolArray::new(
                event.key,
                |index: u32, value: bool| {
                    flags[index as usize] = u8::from(value);
                },
            )));
        Ok(())
    }
    fn effect_visit_integer_flags(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        let mut flags = event.flags.borrow_mut();
        event.array_visit.set(
            event
                .gguf
                .borrow_mut()
                .process_event(VisitUnsignedArray::new(
                    event.key,
                    |index: u32, value: u64| {
                        flags[index as usize] = u8::from(value != 0);
                    },
                )),
        );
        Ok(())
    }

    fn guard_bool_length_fits(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.array_length.get().is_ok_and(|value| {
            value.is_some_and(|count| count <= event.flags.borrow().len() as u64)
        }))
    }
    fn guard_bool_length_exceeds(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.array_length.get().is_ok_and(|value| {
            value.is_some_and(|count| count > event.flags.borrow().len() as u64)
        }))
    }
    raw_hparam_guard!(guard_bool_length_missing, array_length, Ok(None));
    raw_hparam_guard!(
        guard_bool_length_wrong_kind,
        array_length,
        Err(QueryError::TypeMismatch)
    );
    raw_hparam_guard!(
        guard_bool_length_malformed,
        array_length,
        Err(QueryError::Malformed)
    );
    raw_hparam_guard!(
        guard_bool_length_range,
        array_length,
        Err(QueryError::Range)
    );
    raw_hparam_guard!(
        guard_bool_length_query,
        array_length,
        Err(QueryError::NotParsed | QueryError::IndexOutOfBounds)
    );
    raw_hparam_guard!(
        guard_bool_length_internal,
        array_length,
        Err(QueryError::Internal)
    );
    fn guard_integer_metrics_fit(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.unsigned_metrics.get().is_ok_and(|value| {
            value
                .is_some_and(|metrics| metrics.element_count() <= event.flags.borrow().len() as u64)
        }))
    }
    fn guard_integer_metrics_exceed(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event.unsigned_metrics.get().is_ok_and(|value| {
            value.is_some_and(|metrics| metrics.element_count() > event.flags.borrow().len() as u64)
        }))
    }
    raw_hparam_guard!(guard_integer_metrics_missing, unsigned_metrics, Ok(None));
    raw_hparam_guard!(
        guard_integer_metrics_wrong_kind,
        unsigned_metrics,
        Err(QueryError::TypeMismatch)
    );
    raw_hparam_guard!(
        guard_integer_metrics_malformed,
        unsigned_metrics,
        Err(QueryError::Malformed)
    );
    raw_hparam_guard!(
        guard_integer_metrics_range,
        unsigned_metrics,
        Err(QueryError::Range)
    );
    raw_hparam_guard!(
        guard_integer_metrics_query,
        unsigned_metrics,
        Err(QueryError::NotParsed | QueryError::IndexOutOfBounds)
    );
    raw_hparam_guard!(
        guard_integer_metrics_internal,
        unsigned_metrics,
        Err(QueryError::Internal)
    );
    fn guard_array_visit_done(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event
            .array_visit
            .get()
            .is_ok_and(|value| value.is_some_and(|count| u32::try_from(count).is_ok())))
    }
    fn guard_array_visit_count_range(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(event
            .array_visit
            .get()
            .is_ok_and(|value| value.is_some_and(|count| u32::try_from(count).is_err())))
    }
    raw_hparam_guard!(guard_array_visit_missing, array_visit, Ok(None));
    raw_hparam_guard!(
        guard_array_visit_wrong_kind,
        array_visit,
        Err(QueryError::TypeMismatch)
    );
    raw_hparam_guard!(
        guard_array_visit_count,
        array_visit,
        Err(QueryError::IndexOutOfBounds)
    );
    raw_hparam_guard!(
        guard_array_visit_malformed,
        array_visit,
        Err(QueryError::Malformed)
    );
    raw_hparam_guard!(guard_array_visit_range, array_visit, Err(QueryError::Range));
    raw_hparam_guard!(
        guard_array_visit_query,
        array_visit,
        Err(QueryError::NotParsed)
    );
    raw_hparam_guard!(
        guard_array_visit_internal,
        array_visit,
        Err(QueryError::Internal)
    );

    fn guard_scan_value_in_range(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.scan.get(), ScanOutcome::Value(value) if i32::try_from(value).is_ok()))
    }
    fn guard_scan_value_out_of_range(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.scan.get(), ScanOutcome::Value(value) if i32::try_from(value).is_err()))
    }
    raw_hparam_guard!(guard_scan_missing, scan, ScanOutcome::Missing);
    raw_hparam_guard!(
        guard_scan_count,
        scan,
        ScanOutcome::Query(QueryError::IndexOutOfBounds)
    );
    raw_hparam_guard!(
        guard_scan_wrong_kind,
        scan,
        ScanOutcome::Query(QueryError::TypeMismatch)
    );
    raw_hparam_guard!(
        guard_scan_malformed,
        scan,
        ScanOutcome::Query(QueryError::Malformed)
    );
    raw_hparam_guard!(
        guard_scan_range,
        scan,
        ScanOutcome::Query(QueryError::Range)
    );
    raw_hparam_guard!(
        guard_scan_query,
        scan,
        ScanOutcome::Query(QueryError::NotParsed)
    );
    raw_hparam_guard!(
        guard_scan_internal,
        scan,
        ScanOutcome::Query(QueryError::Internal) | ScanOutcome::Arithmetic
    );
    fn effect_assign_scanned_i32(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        let ScanOutcome::Value(value) = event.scan.get() else {
            return Err(());
        };
        event.i32_value.set(i32::try_from(value).map_err(|_| ())?);
        event.result.set(Ok(1));
        Ok(())
    }
    fn effect_publish_array_count(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        let count = event.array_visit.get().ok().flatten().ok_or(())?;
        #[allow(clippy::cast_possible_truncation)]
        event.result.set(Ok(count as u32));
        Ok(())
    }
    fn effect_scan_missing(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(ErrorKind::Missing));
        Ok(())
    }
    fn effect_scan_count(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(ErrorKind::Count));
        Ok(())
    }
    fn effect_array_capacity(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(ErrorKind::Capacity));
        Ok(())
    }
    fn effect_scan_wrong_kind(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(ErrorKind::WrongKind));
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
