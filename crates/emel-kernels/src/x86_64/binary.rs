//! Safe x86-64 AVX2 F32 binary kernels for the pinned target contract.
//!
//! This bounded slice mirrors the pinned `emel.cpp` equal-element-count binary routes
//! for add, subtract, multiply, and divide from commit
//! `843a117386ef17dc5a50549bbfc821074c2141d6`. The public API uses one typed
//! event per operation, so an action never selects a runtime operation.
//! Pulp's typed V3 capability is resolved before dispatch. The source row-
//! broadcast add/mul variants are owned by the sibling
//! `X86BroadcastF32Kernel` and exposed through the target router, so this
//! equal-count actor does not duplicate that data-plane implementation.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::derive_partial_eq_without_eq)]
// `sml!` emits public machine event enums for private runtime payloads. This
// is generated visibility noise and does not widen the actor API.
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{Simd, WithSimd, x86::Arch};
use sml::sml;

/// Errors returned by the x86 binary actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum X86BinaryF32Error {
    /// The target did not expose the required AVX2 capability.
    BackendUnavailable,
    /// The source and destination do not have equal, non-empty dense F32 lengths.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for X86BinaryF32Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("x86 AVX2 backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid x86 binary F32 shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected x86 binary F32 event"),
            Self::Internal => formatter.write_str("internal x86 binary F32 dispatch error"),
        }
    }
}

impl std::error::Error for X86BinaryF32Error {}

/// Result returned after a binary event reaches run-to-completion.
pub type X86BinaryF32Result = Result<(), X86BinaryF32Error>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_GUARD_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";
const PINNED_ACTION_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";
const PINNED_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/x86_64/guards.hpp:405-434";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/x86_64/actions.hpp:1692-1865,2240-2292";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/x86_64/sm.hpp:46-143";
const BROADCAST_ROUTE_OWNER: &str = "X86BroadcastF32Kernel via X86Kernel::X86BroadcastAdd/Mul";

/// Typed addition request over equal-length dense F32 slices.
#[derive(Debug)]
pub struct OpX86BinaryAdd<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
}

impl<'a> OpX86BinaryAdd<'a> {
    /// Creates an addition request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32]) -> Self {
        Self { lhs, rhs }
    }
}

/// Typed subtraction request over equal-length dense F32 slices.
#[derive(Debug)]
pub struct OpX86BinarySub<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
}

impl<'a> OpX86BinarySub<'a> {
    /// Creates a subtraction request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32]) -> Self {
        Self { lhs, rhs }
    }
}

/// Typed multiplication request over equal-length dense F32 slices.
#[derive(Debug)]
pub struct OpX86BinaryMul<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
}

impl<'a> OpX86BinaryMul<'a> {
    /// Creates a multiplication request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32]) -> Self {
        Self { lhs, rhs }
    }
}

/// Typed division request over equal-length dense F32 slices.
#[derive(Debug)]
pub struct OpX86BinaryDiv<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
}

impl<'a> OpX86BinaryDiv<'a> {
    /// Creates a division request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32]) -> Self {
        Self { lhs, rhs }
    }
}

/// Explicitly reports an event outside the x86 binary API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedX86BinaryF32;

struct BinaryRuntime<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    output: &'a mut [f32],
    result: &'a Cell<X86BinaryF32Result>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<X86BinaryF32Result>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    backend: pulp::x86::V3,
}

sml! {
    X86BinaryF32Machine<'dispatch> {
        "ready"_s <= *"ready"_s + Add(BinaryRuntime<'dispatch>) [guard_add_ready] / effect_add,
        "ready"_s <= "ready"_s + Add(BinaryRuntime<'dispatch>) [guard_add_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Sub(BinaryRuntime<'dispatch>) [guard_sub_ready] / effect_sub,
        "ready"_s <= "ready"_s + Sub(BinaryRuntime<'dispatch>) [guard_sub_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Mul(BinaryRuntime<'dispatch>) [guard_mul_ready] / effect_mul,
        "ready"_s <= "ready"_s + Mul(BinaryRuntime<'dispatch>) [guard_mul_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Div(BinaryRuntime<'dispatch>) [guard_div_ready] / effect_div,
        "ready"_s <= "ready"_s + Div(BinaryRuntime<'dispatch>) [guard_div_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion x86 binary actor.
pub struct X86BinaryF32Kernel {
    machine: X86BinaryF32MachineStateMachine<Context>,
}

impl fmt::Debug for X86BinaryF32Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("X86BinaryF32Kernel")
            .finish_non_exhaustive()
    }
}

impl X86BinaryF32Kernel {
    /// Resolves AVX2 before allowing dispatch; scalar is never substituted.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::V3(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: X86BinaryF32MachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: X86BinaryF32Event>(
        &mut self,
        event: E,
        output: &mut [f32],
    ) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated actor is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&X86BinaryF32MachineStates::Ready)
    }

    fn add(&mut self, lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> X86BinaryF32Result {
        let result = Cell::new(Err(X86BinaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(X86BinaryF32MachineEvents::Add(BinaryRuntime {
                lhs,
                rhs,
                output,
                result: &result,
            }))
            .map_err(|_| X86BinaryF32Error::Internal)?;
        result.get()
    }

    fn sub(&mut self, lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> X86BinaryF32Result {
        let result = Cell::new(Err(X86BinaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(X86BinaryF32MachineEvents::Sub(BinaryRuntime {
                lhs,
                rhs,
                output,
                result: &result,
            }))
            .map_err(|_| X86BinaryF32Error::Internal)?;
        result.get()
    }

    fn mul(&mut self, lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> X86BinaryF32Result {
        let result = Cell::new(Err(X86BinaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(X86BinaryF32MachineEvents::Mul(BinaryRuntime {
                lhs,
                rhs,
                output,
                result: &result,
            }))
            .map_err(|_| X86BinaryF32Error::Internal)?;
        result.get()
    }

    fn div(&mut self, lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> X86BinaryF32Result {
        let result = Cell::new(Err(X86BinaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(X86BinaryF32MachineEvents::Div(BinaryRuntime {
                lhs,
                rhs,
                output,
                result: &result,
            }))
            .map_err(|_| X86BinaryF32Error::Internal)?;
        result.get()
    }
}

/// Event implemented by the x86 binary actor.
pub trait X86BinaryF32Event: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86BinaryF32Kernel, output: &mut [f32]) -> Self::Output;
}

impl X86BinaryF32Event for OpX86BinaryAdd<'_> {
    type Output = X86BinaryF32Result;

    fn dispatch(self, actor: &mut X86BinaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.add(self.lhs, self.rhs, output)
    }
}

impl X86BinaryF32Event for OpX86BinarySub<'_> {
    type Output = X86BinaryF32Result;

    fn dispatch(self, actor: &mut X86BinaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.sub(self.lhs, self.rhs, output)
    }
}

impl X86BinaryF32Event for OpX86BinaryMul<'_> {
    type Output = X86BinaryF32Result;

    fn dispatch(self, actor: &mut X86BinaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.mul(self.lhs, self.rhs, output)
    }
}

impl X86BinaryF32Event for OpX86BinaryDiv<'_> {
    type Output = X86BinaryF32Result;

    fn dispatch(self, actor: &mut X86BinaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.div(self.lhs, self.rhs, output)
    }
}

impl X86BinaryF32Event for UnexpectedX86BinaryF32 {
    type Output = X86BinaryF32Result;

    fn dispatch(self, actor: &mut X86BinaryF32Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(X86BinaryF32Error::UnexpectedEvent));
        actor
            .machine
            .process_event(X86BinaryF32MachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| X86BinaryF32Error::Internal)?;
        result.get()
    }
}

impl X86BinaryF32MachineStateMachineContext for Context {
    fn guard_add_ready(&self, event: &BinaryRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_add_invalid(&self, event: &BinaryRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn guard_sub_ready(&self, event: &BinaryRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_sub_invalid(&self, event: &BinaryRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn guard_mul_ready(&self, event: &BinaryRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_mul_invalid(&self, event: &BinaryRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn guard_div_ready(&self, event: &BinaryRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_div_invalid(&self, event: &BinaryRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn effect_add(&mut self, event: BinaryRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            AddOperation {
                lhs: event.lhs,
                rhs: event.rhs,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_sub(&mut self, event: BinaryRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            SubOperation {
                lhs: event.lhs,
                rhs: event.rhs,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mul(&mut self, event: BinaryRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            MulOperation {
                lhs: event.lhs,
                rhs: event.rhs,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_div(&mut self, event: BinaryRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            DivOperation {
                lhs: event.lhs,
                rhs: event.rhs,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_invalid(&mut self, event: BinaryRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(X86BinaryF32Error::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(X86BinaryF32Error::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn request_valid(event: &BinaryRuntime<'_>) -> bool {
    !event.lhs.is_empty()
        && event.lhs.len() == event.rhs.len()
        && event.lhs.len() == event.output.len()
}

struct AddOperation<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for AddOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self { lhs, rhs, output } = self;
        let (lhs_vectors, lhs_tail) = S::as_simd_f32s(lhs);
        let (rhs_vectors, rhs_tail) = S::as_simd_f32s(rhs);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(output);
        for ((output, lhs), rhs) in output_vectors.iter_mut().zip(lhs_vectors).zip(rhs_vectors) {
            *output = simd.add_f32s(*lhs, *rhs);
        }
        for ((output, lhs), rhs) in output_tail.iter_mut().zip(lhs_tail).zip(rhs_tail) {
            *output = *lhs + *rhs;
        }
    }
}

struct SubOperation<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for SubOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self { lhs, rhs, output } = self;
        let (lhs_vectors, lhs_tail) = S::as_simd_f32s(lhs);
        let (rhs_vectors, rhs_tail) = S::as_simd_f32s(rhs);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(output);
        for ((output, lhs), rhs) in output_vectors.iter_mut().zip(lhs_vectors).zip(rhs_vectors) {
            *output = simd.sub_f32s(*lhs, *rhs);
        }
        for ((output, lhs), rhs) in output_tail.iter_mut().zip(lhs_tail).zip(rhs_tail) {
            *output = *lhs - *rhs;
        }
    }
}

struct MulOperation<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for MulOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self { lhs, rhs, output } = self;
        let (lhs_vectors, lhs_tail) = S::as_simd_f32s(lhs);
        let (rhs_vectors, rhs_tail) = S::as_simd_f32s(rhs);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(output);
        for ((output, lhs), rhs) in output_vectors.iter_mut().zip(lhs_vectors).zip(rhs_vectors) {
            *output = simd.mul_f32s(*lhs, *rhs);
        }
        for ((output, lhs), rhs) in output_tail.iter_mut().zip(lhs_tail).zip(rhs_tail) {
            *output = *lhs * *rhs;
        }
    }
}

struct DivOperation<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for DivOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self { lhs, rhs, output } = self;
        let (lhs_vectors, lhs_tail) = S::as_simd_f32s(lhs);
        let (rhs_vectors, rhs_tail) = S::as_simd_f32s(rhs);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(output);
        for ((output, lhs), rhs) in output_vectors.iter_mut().zip(lhs_vectors).zip(rhs_vectors) {
            *output = simd.div_f32s(*lhs, *rhs);
        }
        for ((output, lhs), rhs) in output_tail.iter_mut().zip(lhs_tail).zip(rhs_tail) {
            *output = *lhs / *rhs;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    use allocation_counter::measure;

    #[test]
    fn pinned_source_identity_is_exact() {
        assert_eq!(
            PINNED_EMEL_CPP_COMMIT,
            "843a117386ef17dc5a50549bbfc821074c2141d6"
        );
        assert_eq!(
            PINNED_GUARD_BLOB,
            "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf"
        );
        assert_eq!(
            PINNED_ACTION_BLOB,
            "d45558f5eb96950f43c16a09d768cb4f382d6d61"
        );
        assert_eq!(PINNED_SM_BLOB, "0b4d635ebbd0fbd52dbca8a2345547fb571205c8");
        assert_eq!(
            PINNED_GUARD_SPAN,
            "src/emel/kernel/x86_64/guards.hpp:405-434"
        );
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/x86_64/actions.hpp:1692-1865,2240-2292"
        );
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/x86_64/sm.hpp:46-143"
        );
        assert_eq!(
            BROADCAST_ROUTE_OWNER,
            "X86BroadcastF32Kernel via X86Kernel::X86BroadcastAdd/Mul"
        );
    }

    #[test]
    fn all_four_binary_operations_match_pinned_formulas() {
        let Some(mut actor) = X86BinaryF32Kernel::try_new() else {
            return;
        };
        let lhs = [8.0_f32, -9.0, 6.0, 4.0, -2.0, 3.0, 7.0, 11.0, 5.0];
        let rhs = [2.0_f32, 3.0, -2.0, 4.0, 2.0, -3.0, 7.0, 11.0, 5.0];
        let mut output = [0.0_f32; 9];
        assert_eq!(
            actor.process_event(OpX86BinaryAdd::new(&lhs, &rhs), &mut output),
            Ok(())
        );
        assert_eq!(output, [10.0, -6.0, 4.0, 8.0, 0.0, 0.0, 14.0, 22.0, 10.0]);
        assert_eq!(
            actor.process_event(OpX86BinarySub::new(&lhs, &rhs), &mut output),
            Ok(())
        );
        assert_eq!(output, [6.0, -12.0, 8.0, 0.0, -4.0, 6.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            actor.process_event(OpX86BinaryMul::new(&lhs, &rhs), &mut output),
            Ok(())
        );
        assert_eq!(
            output,
            [16.0, -27.0, -12.0, 16.0, -4.0, -9.0, 49.0, 121.0, 25.0]
        );
        assert_eq!(
            actor.process_event(OpX86BinaryDiv::new(&lhs, &rhs), &mut output),
            Ok(())
        );
        assert_eq!(output, [4.0, -3.0, -3.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
        assert!(actor.is_ready());
    }

    #[test]
    fn invalid_shape_is_explicit_and_does_not_mutate_output() {
        let Some(mut actor) = X86BinaryF32Kernel::try_new() else {
            return;
        };
        let mut output = [9.0_f32; 3];
        assert_eq!(
            actor.process_event(OpX86BinaryAdd::new(&[1.0], &[2.0, 3.0]), &mut output),
            Err(X86BinaryF32Error::InvalidShape)
        );
        assert_eq!(output, [9.0; 3]);
        assert_eq!(
            actor.process_event(OpX86BinaryMul::new(&[], &[]), &mut output),
            Err(X86BinaryF32Error::InvalidShape)
        );
        assert_eq!(output, [9.0; 3]);
    }

    #[test]
    fn unexpected_event_is_typed_and_machine_stays_ready() {
        let Some(mut actor) = X86BinaryF32Kernel::try_new() else {
            return;
        };
        assert_eq!(
            actor.process_event(UnexpectedX86BinaryF32, &mut []),
            Err(X86BinaryF32Error::UnexpectedEvent)
        );
        assert!(actor.is_ready());
    }

    #[test]
    fn dispatch_is_allocation_free_after_construction() {
        let Some(mut actor) = X86BinaryF32Kernel::try_new() else {
            return;
        };
        let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let rhs = [9.0_f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut output = [0.0_f32; 9];
        let allocation = measure(|| {
            for _ in 0..64 {
                assert_eq!(
                    actor.process_event(OpX86BinaryAdd::new(&lhs, &rhs), &mut output),
                    Ok(())
                );
            }
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(output, [10.0; 9]);
    }
}
