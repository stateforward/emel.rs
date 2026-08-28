//! Synchronous tensor-state capture orchestration.
//!
//! The C++ owner receives pointers in its request. Rust represents those
//! capabilities with caller-owned `RefCell` handles. The request and runtime
//! event are immutable and cloneable; mutable borrows are made only by the
//! bounded execution action.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    dead_code,
    missing_docs
)]

use core::cell::{Cell, RefCell};
use core::fmt;

use sml::sml;

/// Policy boundary for tensor-state capture.
pub trait TensorViewPolicy {
    /// Tensor-machine type owned by the caller.
    type TensorMachine;
    /// Snapshot type written by capture.
    type TensorState;

    /// Largest accepted tensor identifier plus one.
    const MAX_TENSORS: i32;
    /// Policy code for no error.
    const NONE_ERROR_CODE: i32;
    /// Policy code for an invalid request.
    const INVALID_REQUEST_ERROR_CODE: i32;
    /// Policy code for an internal operation failure.
    const INTERNAL_ERROR_CODE: i32;

    /// Captures one tensor state synchronously.
    fn capture_tensor_state(
        tensor_machine: &mut Self::TensorMachine,
        tensor_id: i32,
        state_out: &mut Self::TensorState,
        error_out: &mut i32,
    ) -> bool;
}

/// Caller-owned single-writer capability for a tensor-view pointer.
pub struct TensorViewHandle<T> {
    value: RefCell<T>,
}

impl<T> TensorViewHandle<T> {
    /// Wraps a caller-owned value before dispatch.
    #[must_use]
    pub const fn new(value: T) -> Self {
        Self {
            value: RefCell::new(value),
        }
    }

    /// Borrows the wrapped value immutably.
    #[must_use]
    pub fn borrow(&self) -> core::cell::Ref<'_, T> {
        self.value.borrow()
    }
}

impl<T> fmt::Debug for TensorViewHandle<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorViewHandle")
            .finish_non_exhaustive()
    }
}

/// Immutable tensor-view capture request.
pub struct CaptureTensorView<'a, P: TensorViewPolicy> {
    tensor_machine: Option<&'a TensorViewHandle<P::TensorMachine>>,
    tensor_id: i32,
    state_out: Option<&'a TensorViewHandle<P::TensorState>>,
    error_out: Option<&'a Cell<i32>>,
}

impl<P: TensorViewPolicy> Copy for CaptureTensorView<'_, P> {}

impl<P: TensorViewPolicy> Clone for CaptureTensorView<'_, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: TensorViewPolicy> fmt::Debug for CaptureTensorView<'_, P> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CaptureTensorView")
            .field("tensor_machine", &self.tensor_machine.is_some())
            .field("tensor_id", &self.tensor_id)
            .field("state_out", &self.state_out.is_some())
            .field("error_out", &self.error_out.is_some())
            .finish()
    }
}

impl<'a, P: TensorViewPolicy> CaptureTensorView<'a, P> {
    /// Creates a request. Null C++ pointers are represented by None.
    #[must_use]
    pub const fn new(
        tensor_machine: Option<&'a TensorViewHandle<P::TensorMachine>>,
        tensor_id: i32,
        state_out: Option<&'a TensorViewHandle<P::TensorState>>,
        error_out: Option<&'a Cell<i32>>,
    ) -> Self {
        Self {
            tensor_machine,
            tensor_id,
            state_out,
            error_out,
        }
    }
}

/// Dispatch-local status corresponding to C++ `runtime_status`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RuntimeStatus {
    err: i32,
    ok: bool,
    accepted: bool,
}

impl RuntimeStatus {
    const fn new<P: TensorViewPolicy>() -> Self {
        Self {
            err: P::NONE_ERROR_CODE,
            ok: false,
            accepted: false,
        }
    }
}

/// Cloneable internal event carrying only non-owning handles and outcome slots.
#[derive(Debug)]
pub struct DetailCaptureTensorViewRuntime<'request, 'dispatch, P: TensorViewPolicy> {
    request: CaptureTensorView<'request, P>,
    status: &'dispatch Cell<RuntimeStatus>,
    error_code_out: &'dispatch Cell<i32>,
}

impl<P: TensorViewPolicy> Copy for DetailCaptureTensorViewRuntime<'_, '_, P> {}

impl<P: TensorViewPolicy> Clone for DetailCaptureTensorViewRuntime<'_, '_, P> {
    fn clone(&self) -> Self {
        *self
    }
}

sml! {
    TensorView<'request, 'dispatch, P>
    where
        P: TensorViewPolicy,
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        "capture_tensor_view_request_decision"_s <= *"ready"_s + event<DetailCaptureTensorViewRuntime<'request, 'dispatch, P>> / begin_capture_tensor_view,
        "capture_tensor_view_exec"_s <= "capture_tensor_view_request_decision"_s + completion<DetailCaptureTensorViewRuntime>(DetailCaptureTensorViewRuntime<'request, 'dispatch, P>) [capture_tensor_view_request_valid],
        "errored"_s <= "capture_tensor_view_request_decision"_s + completion<DetailCaptureTensorViewRuntime>(DetailCaptureTensorViewRuntime<'request, 'dispatch, P>) [capture_tensor_view_request_invalid] / mark_invalid_request,
        "capture_tensor_view_result_decision"_s <= "capture_tensor_view_exec"_s + completion<DetailCaptureTensorViewRuntime>(DetailCaptureTensorViewRuntime<'request, 'dispatch, P>) / exec_capture_tensor_view,
        "done"_s <= "capture_tensor_view_result_decision"_s + completion<DetailCaptureTensorViewRuntime>(DetailCaptureTensorViewRuntime<'request, 'dispatch, P>) [operation_succeeded],
        "errored"_s <= "capture_tensor_view_result_decision"_s + completion<DetailCaptureTensorViewRuntime>(DetailCaptureTensorViewRuntime<'request, 'dispatch, P>) [operation_failed_with_error] / mark_error_from_operation,
        "errored"_s <= "capture_tensor_view_result_decision"_s + completion<DetailCaptureTensorViewRuntime>(DetailCaptureTensorViewRuntime<'request, 'dispatch, P>) [operation_failed_without_error] / mark_internal_error,

        "ready"_s <= "done"_s + completion<DetailCaptureTensorViewRuntime>(DetailCaptureTensorViewRuntime<'request, 'dispatch, P>) / publish_done,
        "ready"_s <= "errored"_s + completion<DetailCaptureTensorViewRuntime>(DetailCaptureTensorViewRuntime<'request, 'dispatch, P>) / publish_error,

        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "ready"_s <= "capture_tensor_view_request_decision"_s + unexpected_event<DetailCaptureTensorViewRuntime>(DetailCaptureTensorViewRuntime<'request, 'dispatch, P>) / on_unexpected_runtime,
        "ready"_s <= "capture_tensor_view_exec"_s + unexpected_event<DetailCaptureTensorViewRuntime>(DetailCaptureTensorViewRuntime<'request, 'dispatch, P>) / on_unexpected_runtime,
        "ready"_s <= "capture_tensor_view_result_decision"_s + unexpected_event<DetailCaptureTensorViewRuntime>(DetailCaptureTensorViewRuntime<'request, 'dispatch, P>) / on_unexpected_runtime,
        "ready"_s <= "done"_s + unexpected_event<DetailCaptureTensorViewRuntime>(DetailCaptureTensorViewRuntime<'request, 'dispatch, P>) / on_unexpected_runtime,
        "ready"_s <= "errored"_s + unexpected_event<DetailCaptureTensorViewRuntime>(DetailCaptureTensorViewRuntime<'request, 'dispatch, P>) / on_unexpected_runtime,
    }
}

/// Empty actor-owned context. Request-specific data is call-scoped.
#[derive(Debug, Default)]
pub struct TensorViewContext;

impl TensorViewStateMachineContext for TensorViewContext {
    fn begin_capture_tensor_view<'request, 'dispatch, P: TensorViewPolicy>(
        &mut self,
        event: &DetailCaptureTensorViewRuntime<'request, 'dispatch, P>,
    ) -> Result<(), ()>
    where
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        event.status.set(RuntimeStatus::new::<P>());
        event.error_code_out.set(P::NONE_ERROR_CODE);
        Ok(())
    }

    fn capture_tensor_view_request_invalid<'request, 'dispatch, P: TensorViewPolicy>(
        &self,
        event: &DetailCaptureTensorViewRuntime<'request, 'dispatch, P>,
    ) -> Result<bool, ()>
    where
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        Ok(!capture_tensor_view_request_valid(event))
    }

    fn capture_tensor_view_request_valid<'request, 'dispatch, P: TensorViewPolicy>(
        &self,
        event: &DetailCaptureTensorViewRuntime<'request, 'dispatch, P>,
    ) -> Result<bool, ()>
    where
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        Ok(capture_tensor_view_request_valid(event))
    }

    fn exec_capture_tensor_view<'request, 'dispatch, P: TensorViewPolicy>(
        &mut self,
        event: &DetailCaptureTensorViewRuntime<'request, 'dispatch, P>,
    ) -> Result<(), ()>
    where
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        let request = event.request;
        let mut tensor_error = P::NONE_ERROR_CODE;
        let accepted = {
            let mut tensor_machine = request
                .tensor_machine
                .expect("request-valid guard selected a tensor machine")
                .value
                .borrow_mut();
            let mut state_out = request
                .state_out
                .expect("request-valid guard selected a state output")
                .value
                .borrow_mut();
            P::capture_tensor_state(
                &mut *tensor_machine,
                request.tensor_id,
                &mut *state_out,
                &mut tensor_error,
            )
        };
        event.status.set(RuntimeStatus {
            err: tensor_error,
            ok: false,
            accepted,
        });
        Ok(())
    }

    fn mark_error_from_operation<'request, 'dispatch, P: TensorViewPolicy>(
        &mut self,
        event: &DetailCaptureTensorViewRuntime<'request, 'dispatch, P>,
    ) -> Result<(), ()>
    where
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        event.error_code_out.set(event.status.get().err);
        Ok(())
    }

    fn mark_internal_error<'request, 'dispatch, P: TensorViewPolicy>(
        &mut self,
        event: &DetailCaptureTensorViewRuntime<'request, 'dispatch, P>,
    ) -> Result<(), ()>
    where
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        event.status.set(RuntimeStatus {
            err: P::INTERNAL_ERROR_CODE,
            ok: false,
            accepted: false,
        });
        event.error_code_out.set(P::INTERNAL_ERROR_CODE);
        Ok(())
    }

    fn on_unexpected_runtime<'request, 'dispatch, P: TensorViewPolicy>(
        &mut self,
        event: &DetailCaptureTensorViewRuntime<'request, 'dispatch, P>,
    ) -> Result<(), ()>
    where
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        let mut status = event.status.get();
        status.err = P::INTERNAL_ERROR_CODE;
        status.ok = false;
        event.status.set(status);
        event.error_code_out.set(P::INTERNAL_ERROR_CODE);
        Ok(())
    }

    fn mark_invalid_request<'request, 'dispatch, P: TensorViewPolicy>(
        &mut self,
        event: &DetailCaptureTensorViewRuntime<'request, 'dispatch, P>,
    ) -> Result<(), ()>
    where
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        mark_invalid(event);
        Ok(())
    }

    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        Ok(())
    }

    fn operation_failed_with_error<'request, 'dispatch, P: TensorViewPolicy>(
        &self,
        event: &DetailCaptureTensorViewRuntime<'request, 'dispatch, P>,
    ) -> Result<bool, ()>
    where
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        let status = event.status.get();
        Ok(status.err != P::NONE_ERROR_CODE)
    }

    fn operation_failed_without_error<'request, 'dispatch, P: TensorViewPolicy>(
        &self,
        event: &DetailCaptureTensorViewRuntime<'request, 'dispatch, P>,
    ) -> Result<bool, ()>
    where
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        let status = event.status.get();
        Ok(!status.accepted && status.err == P::NONE_ERROR_CODE)
    }

    fn operation_succeeded<'request, 'dispatch, P: TensorViewPolicy>(
        &self,
        event: &DetailCaptureTensorViewRuntime<'request, 'dispatch, P>,
    ) -> Result<bool, ()>
    where
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        let status = event.status.get();
        let _ = status.ok;
        Ok(status.accepted && status.err == P::NONE_ERROR_CODE)
    }

    fn publish_done<'request, 'dispatch, P: TensorViewPolicy>(
        &mut self,
        event: &DetailCaptureTensorViewRuntime<'request, 'dispatch, P>,
    ) -> Result<(), ()>
    where
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        let mut status = event.status.get();
        status.err = P::NONE_ERROR_CODE;
        status.ok = true;
        event.status.set(status);
        event.error_code_out.set(P::NONE_ERROR_CODE);
        Ok(())
    }

    fn publish_error<'request, 'dispatch, P: TensorViewPolicy>(
        &mut self,
        event: &DetailCaptureTensorViewRuntime<'request, 'dispatch, P>,
    ) -> Result<(), ()>
    where
        P::TensorMachine: 'request,
        P::TensorState: 'request,
    {
        let mut status = event.status.get();
        status.ok = false;
        event.status.set(status);
        event.error_code_out.set(status.err);
        Ok(())
    }
}

const fn capture_tensor_view_request_valid<P: TensorViewPolicy>(
    event: &DetailCaptureTensorViewRuntime<'_, '_, P>,
) -> bool {
    event.request.tensor_machine.is_some()
        && event.request.state_out.is_some()
        && event.request.tensor_id >= 0
        && event.request.tensor_id < P::MAX_TENSORS
}

fn mark_invalid<P: TensorViewPolicy>(event: &DetailCaptureTensorViewRuntime<'_, '_, P>) {
    event.status.set(RuntimeStatus {
        err: P::INVALID_REQUEST_ERROR_CODE,
        ok: false,
        accepted: false,
    });
    event.error_code_out.set(P::INVALID_REQUEST_ERROR_CODE);
}

/// Synchronous single-writer tensor-view actor.
pub struct TensorViewActor {
    machine: TensorViewStateMachine<TensorViewContext>,
}

impl fmt::Debug for TensorViewActor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorViewActor")
            .finish_non_exhaustive()
    }
}

impl Default for TensorViewActor {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorViewActor {
    /// Constructs an actor in ready.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: TensorViewStateMachine::new(TensorViewContext),
        }
    }

    /// Processes one request to completion.
    pub fn process_event<P: TensorViewPolicy>(
        &mut self,
        request: CaptureTensorView<'_, P>,
    ) -> Result<(), TensorViewDispatchError> {
        let error_sink = Cell::new(P::NONE_ERROR_CODE);
        let status = Cell::new(RuntimeStatus::new::<P>());
        let error_code_out = request.error_out.unwrap_or(&error_sink);
        let runtime = DetailCaptureTensorViewRuntime {
            request,
            status: &status,
            error_code_out,
        };
        self.machine
            .process_event(TensorViewEvents::DetailCaptureTensorViewRuntime(runtime))
            .map_err(|_| TensorViewDispatchError::Internal)?;
        if self.machine.is(&TensorViewStates::Ready) {
            let final_status = status.get();
            if final_status.ok {
                Ok(())
            } else {
                Err(TensorViewDispatchError::from_code(
                    final_status.err,
                    P::INVALID_REQUEST_ERROR_CODE,
                    P::INTERNAL_ERROR_CODE,
                ))
            }
        } else {
            Err(TensorViewDispatchError::Internal)
        }
    }

    /// Returns whether the generated actor is in ready.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&TensorViewStates::Ready)
    }
}

/// Public outcome classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorViewDispatchError {
    /// Request validation failed.
    InvalidRequest,
    /// Policy returned its classified internal error code.
    InternalError,
    /// Policy returned an unclassified error code.
    Operation(i32),
    /// Generated dispatch failed.
    Internal,
}

impl TensorViewDispatchError {
    const fn from_code(code: i32, invalid: i32, internal: i32) -> Self {
        if code == invalid {
            Self::InvalidRequest
        } else if code == internal {
            Self::InternalError
        } else {
            Self::Operation(code)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default, Eq, PartialEq)]
    struct Machine {
        calls: u32,
        mode: u8,
    }

    #[derive(Debug, Default, Eq, PartialEq)]
    struct State {
        tensor_id: i32,
        calls: u32,
    }

    struct Policy;

    impl TensorViewPolicy for Policy {
        type TensorMachine = Machine;
        type TensorState = State;

        const MAX_TENSORS: i32 = 4;
        const NONE_ERROR_CODE: i32 = 0;
        const INVALID_REQUEST_ERROR_CODE: i32 = 1;
        const INTERNAL_ERROR_CODE: i32 = 2;

        fn capture_tensor_state(
            tensor_machine: &mut Self::TensorMachine,
            tensor_id: i32,
            state_out: &mut Self::TensorState,
            error_out: &mut i32,
        ) -> bool {
            tensor_machine.calls += 1;
            match tensor_machine.mode {
                0 => {
                    state_out.tensor_id = tensor_id;
                    state_out.calls += 1;
                    true
                }
                1 => {
                    *error_out = Self::INVALID_REQUEST_ERROR_CODE;
                    false
                }
                _ => {
                    *error_out = Self::INTERNAL_ERROR_CODE;
                    false
                }
            }
        }
    }

    #[test]
    fn valid_capture_publishes_snapshot_and_restores_ready() {
        let machine = TensorViewHandle::new(Machine::default());
        let state = TensorViewHandle::new(State::default());
        let error = Cell::new(-1);
        let mut actor = TensorViewActor::new();

        assert_eq!(
            actor.process_event(CaptureTensorView::<Policy>::new(
                Some(&machine),
                3,
                Some(&state),
                Some(&error),
            )),
            Ok(())
        );
        assert!(actor.is_ready());
        assert_eq!(machine.borrow().calls, 1);
        assert_eq!(state.borrow().tensor_id, 3);
        assert_eq!(error.get(), 0);
    }

    #[test]
    fn invalid_request_is_published_without_operation() {
        let state = TensorViewHandle::new(State::default());
        let error = Cell::new(-1);
        let mut actor = TensorViewActor::new();

        assert_eq!(
            actor.process_event(CaptureTensorView::<Policy>::new(
                None,
                0,
                Some(&state),
                Some(&error),
            )),
            Err(TensorViewDispatchError::InvalidRequest)
        );
        assert!(actor.is_ready());
        assert_eq!(error.get(), 1);
    }

    #[test]
    fn classified_operation_errors_are_published() {
        for (mode, expected) in [
            (1, TensorViewDispatchError::InvalidRequest),
            (2, TensorViewDispatchError::InternalError),
        ] {
            let machine = TensorViewHandle::new(Machine {
                mode,
                ..Machine::default()
            });
            let state = TensorViewHandle::new(State::default());
            let error = Cell::new(-1);
            let mut actor = TensorViewActor::new();

            assert_eq!(
                actor.process_event(CaptureTensorView::<Policy>::new(
                    Some(&machine),
                    0,
                    Some(&state),
                    Some(&error),
                )),
                Err(expected)
            );
            assert!(actor.is_ready());
            assert_eq!(error.get(), i32::from(mode));
        }
    }

    #[test]
    fn unclassified_operation_error_is_preserved() {
        struct OtherPolicy;
        impl TensorViewPolicy for OtherPolicy {
            type TensorMachine = Machine;
            type TensorState = State;
            const MAX_TENSORS: i32 = 4;
            const NONE_ERROR_CODE: i32 = 0;
            const INVALID_REQUEST_ERROR_CODE: i32 = 1;
            const INTERNAL_ERROR_CODE: i32 = 2;

            fn capture_tensor_state(
                _: &mut Self::TensorMachine,
                _: i32,
                _: &mut Self::TensorState,
                error_out: &mut i32,
            ) -> bool {
                *error_out = 7;
                false
            }
        }

        let machine = TensorViewHandle::new(Machine::default());
        let state = TensorViewHandle::new(State::default());
        let error = Cell::new(-1);
        let mut actor = TensorViewActor::new();

        assert_eq!(
            actor.process_event(CaptureTensorView::<OtherPolicy>::new(
                Some(&machine),
                0,
                Some(&state),
                Some(&error),
            )),
            Err(TensorViewDispatchError::Operation(7))
        );
        assert_eq!(error.get(), 7);
        assert!(actor.is_ready());
    }

    #[test]
    fn unexpected_event_from_transient_state_returns_to_ready() {
        let machine = TensorViewHandle::new(Machine::default());
        let state = TensorViewHandle::new(State::default());
        let status = Cell::new(RuntimeStatus::new::<Policy>());
        let error_code_out = Cell::new(-1);
        let request = CaptureTensorView::<Policy>::new(
            Some(&machine),
            0,
            Some(&state),
            Some(&error_code_out),
        );
        let runtime = DetailCaptureTensorViewRuntime {
            request,
            status: &status,
            error_code_out: &error_code_out,
        };
        let mut machine = TensorViewStateMachine::new_with_state(
            TensorViewContext,
            TensorViewStates::CaptureTensorViewExec,
        );

        assert!(machine.process_event(runtime).is_ok());
        assert!(machine.is(&TensorViewStates::Ready));
        assert_eq!(status.get().err, Policy::INTERNAL_ERROR_CODE);
        assert!(!status.get().ok);
        assert_eq!(error_code_out.get(), Policy::INTERNAL_ERROR_CODE);
    }
}
