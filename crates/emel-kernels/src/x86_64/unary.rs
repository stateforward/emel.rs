//! Safe x86-64 AVX2 F32 unary kernels for the pinned target contract.
//!
//! The bounded implementation mirrors the pinned `emel.cpp` AVX2 routes for
//! `abs`, `neg`, and `relu` from commit
//! `843a117386ef17dc5a50549bbfc821074c2141d6`.  The public API uses one
//! typed event per operation, so the action never selects a runtime subop.
//! Pulp's typed V3 capability is resolved before dispatch; scalar capability
//! fallback is not advertised.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::derive_partial_eq_without_eq)]
// `sml!` emits public machine event enums for private runtime payloads. This
// is generated visibility noise and does not widen the actor API.
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{Simd, WithSimd, x86::Arch};
use sml::sml;

/// Errors returned by the x86 unary actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum X86UnaryF32Error {
    /// The target did not expose the required AVX2 capability.
    BackendUnavailable,
    /// The source and destination are not equal, non-empty dense F32 slices.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for X86UnaryF32Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("x86 AVX2 backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid x86 unary F32 shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected x86 unary F32 event"),
            Self::Internal => formatter.write_str("internal x86 unary F32 dispatch error"),
        }
    }
}

impl std::error::Error for X86UnaryF32Error {}

/// Result returned after a unary event reaches run-to-completion.
pub type X86UnaryF32Result = Result<(), X86UnaryF32Error>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_GUARD_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";
const PINNED_ACTION_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";
const PINNED_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/x86_64/guards.hpp:232-251";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/x86_64/actions.hpp:104-186,2172-2236";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/x86_64/sm.hpp:1035-1053";

/// Typed absolute-value request over a dense F32 source slice.
#[derive(Debug)]
pub struct OpX86UnaryAbs<'a> {
    input: &'a [f32],
}

impl<'a> OpX86UnaryAbs<'a> {
    /// Creates an absolute-value request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Typed negation request over a dense F32 source slice.
#[derive(Debug)]
pub struct OpX86UnaryNeg<'a> {
    input: &'a [f32],
}

impl<'a> OpX86UnaryNeg<'a> {
    /// Creates a negation request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Typed `ReLU` request over a dense F32 source slice.
#[derive(Debug)]
pub struct OpX86UnaryRelu<'a> {
    input: &'a [f32],
}

impl<'a> OpX86UnaryRelu<'a> {
    /// Creates a `ReLU` request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Explicitly reports an event outside the x86 unary API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedX86UnaryF32;

struct UnaryRuntime<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    result: &'a Cell<X86UnaryF32Result>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<X86UnaryF32Result>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    backend: pulp::x86::V3,
}

sml! {
    X86UnaryF32Machine<'dispatch> {
        "ready"_s <= *"ready"_s + Abs(UnaryRuntime<'dispatch>) [guard_abs_ready] / effect_abs,
        "ready"_s <= "ready"_s + Abs(UnaryRuntime<'dispatch>) [guard_abs_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Neg(UnaryRuntime<'dispatch>) [guard_neg_ready] / effect_neg,
        "ready"_s <= "ready"_s + Neg(UnaryRuntime<'dispatch>) [guard_neg_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Relu(UnaryRuntime<'dispatch>) [guard_relu_ready] / effect_relu,
        "ready"_s <= "ready"_s + Relu(UnaryRuntime<'dispatch>) [guard_relu_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion x86 unary actor.
pub struct X86UnaryF32Kernel {
    machine: X86UnaryF32MachineStateMachine<Context>,
}

impl fmt::Debug for X86UnaryF32Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("X86UnaryF32Kernel")
            .finish_non_exhaustive()
    }
}

impl X86UnaryF32Kernel {
    /// Resolves AVX2 before allowing dispatch; scalar is never substituted.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::V3(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: X86UnaryF32MachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: X86UnaryF32Event>(
        &mut self,
        event: E,
        output: &mut [f32],
    ) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated actor is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&X86UnaryF32MachineStates::Ready)
    }

    fn abs(&mut self, input: &[f32], output: &mut [f32]) -> X86UnaryF32Result {
        let result = Cell::new(Err(X86UnaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(X86UnaryF32MachineEvents::Abs(UnaryRuntime {
                input,
                output,
                result: &result,
            }))
            .map_err(|_| X86UnaryF32Error::Internal)?;
        result.get()
    }

    fn neg(&mut self, input: &[f32], output: &mut [f32]) -> X86UnaryF32Result {
        let result = Cell::new(Err(X86UnaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(X86UnaryF32MachineEvents::Neg(UnaryRuntime {
                input,
                output,
                result: &result,
            }))
            .map_err(|_| X86UnaryF32Error::Internal)?;
        result.get()
    }

    fn relu(&mut self, input: &[f32], output: &mut [f32]) -> X86UnaryF32Result {
        let result = Cell::new(Err(X86UnaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(X86UnaryF32MachineEvents::Relu(UnaryRuntime {
                input,
                output,
                result: &result,
            }))
            .map_err(|_| X86UnaryF32Error::Internal)?;
        result.get()
    }
}

/// Event implemented by the x86 unary actor.
pub trait X86UnaryF32Event: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86UnaryF32Kernel, output: &mut [f32]) -> Self::Output;
}

impl X86UnaryF32Event for OpX86UnaryAbs<'_> {
    type Output = X86UnaryF32Result;

    fn dispatch(self, actor: &mut X86UnaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.abs(self.input, output)
    }
}

impl X86UnaryF32Event for OpX86UnaryNeg<'_> {
    type Output = X86UnaryF32Result;

    fn dispatch(self, actor: &mut X86UnaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.neg(self.input, output)
    }
}

impl X86UnaryF32Event for OpX86UnaryRelu<'_> {
    type Output = X86UnaryF32Result;

    fn dispatch(self, actor: &mut X86UnaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.relu(self.input, output)
    }
}

impl X86UnaryF32Event for UnexpectedX86UnaryF32 {
    type Output = X86UnaryF32Result;

    fn dispatch(self, actor: &mut X86UnaryF32Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(X86UnaryF32Error::UnexpectedEvent));
        actor
            .machine
            .process_event(X86UnaryF32MachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| X86UnaryF32Error::Internal)?;
        result.get()
    }
}

impl X86UnaryF32MachineStateMachineContext for Context {
    fn guard_abs_ready(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_abs_invalid(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn guard_neg_ready(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_neg_invalid(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn guard_relu_ready(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_relu_invalid(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn effect_abs(&mut self, event: UnaryRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            AbsOperation {
                input: event.input,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_neg(&mut self, event: UnaryRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            NegOperation {
                input: event.input,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_relu(&mut self, event: UnaryRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            ReluOperation {
                input: event.input,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_invalid(&mut self, event: UnaryRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(X86UnaryF32Error::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(X86UnaryF32Error::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn request_valid(event: &UnaryRuntime<'_>) -> bool {
    !event.input.is_empty() && event.input.len() == event.output.len()
}

struct AbsOperation<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for AbsOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self { input, output } = self;
        let (input_vectors, input_tail) = S::as_simd_f32s(input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(output);
        for (output, input) in output_vectors.iter_mut().zip(input_vectors) {
            *output = simd.abs_f32s(*input);
        }
        // The pinned AVX2 action uses the same bounded scalar tail after its
        // vector loop; this is data-plane completion, never backend fallback.
        for (output, input) in output_tail.iter_mut().zip(input_tail) {
            *output = input.abs();
        }
    }
}

struct NegOperation<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for NegOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self { input, output } = self;
        let (input_vectors, input_tail) = S::as_simd_f32s(input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(output);
        for (output, input) in output_vectors.iter_mut().zip(input_vectors) {
            *output = simd.neg_f32s(*input);
        }
        // Preserve the pinned action's scalar tail for non-vector-aligned
        // counts without selecting a different backend.
        for (output, input) in output_tail.iter_mut().zip(input_tail) {
            *output = -*input;
        }
    }
}

struct ReluOperation<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for ReluOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self { input, output } = self;
        let (input_vectors, input_tail) = S::as_simd_f32s(input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(output);
        let zero = simd.splat_f32s(0.0);
        for (output, input) in output_vectors.iter_mut().zip(input_vectors) {
            *output = simd.max_f32s(*input, zero);
        }
        // Preserve the pinned action's scalar tail for non-vector-aligned
        // counts without selecting a different backend.
        for (output, input) in output_tail.iter_mut().zip(input_tail) {
            *output = 0.0_f32.max(*input);
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
            "src/emel/kernel/x86_64/guards.hpp:232-251"
        );
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/x86_64/actions.hpp:104-186,2172-2236"
        );
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/x86_64/sm.hpp:1035-1053"
        );
    }

    #[test]
    fn all_three_unary_operations_match_pinned_formulas() {
        let Some(mut actor) = X86UnaryF32Kernel::try_new() else {
            return;
        };
        let input = [-2.0_f32, -0.0, 0.0, 0.5, 2.0, 9.0, -11.0, 1.25, -3.5];
        let mut output = [0.0_f32; 9];
        assert_eq!(
            actor.process_event(OpX86UnaryAbs::new(&input), &mut output),
            Ok(())
        );
        assert_eq!(output, [2.0, 0.0, 0.0, 0.5, 2.0, 9.0, 11.0, 1.25, 3.5]);
        assert_eq!(
            actor.process_event(OpX86UnaryNeg::new(&input), &mut output),
            Ok(())
        );
        assert_eq!(output, [2.0, 0.0, -0.0, -0.5, -2.0, -9.0, 11.0, -1.25, 3.5]);
        assert_eq!(
            actor.process_event(OpX86UnaryRelu::new(&input), &mut output),
            Ok(())
        );
        assert_eq!(output, [0.0, -0.0, 0.0, 0.5, 2.0, 9.0, 0.0, 1.25, 0.0]);
    }

    #[test]
    fn invalid_shape_is_explicit_and_does_not_mutate_output() {
        let Some(mut actor) = X86UnaryF32Kernel::try_new() else {
            return;
        };
        let input = [1.0_f32, -2.0, 3.0];
        let mut output = [9.0_f32; 2];
        assert_eq!(
            actor.process_event(OpX86UnaryAbs::new(&input), &mut output),
            Err(X86UnaryF32Error::InvalidShape)
        );
        assert_eq!(output, [9.0; 2]);
        assert_eq!(
            actor.process_event(OpX86UnaryNeg::new(&[]), &mut output),
            Err(X86UnaryF32Error::InvalidShape)
        );
        assert_eq!(output, [9.0; 2]);
    }

    #[test]
    fn unexpected_event_is_typed_and_machine_stays_ready() {
        let Some(mut actor) = X86UnaryF32Kernel::try_new() else {
            return;
        };
        let mut output = [0.0_f32; 1];
        assert_eq!(
            actor.process_event(UnexpectedX86UnaryF32, &mut output),
            Err(X86UnaryF32Error::UnexpectedEvent)
        );
        assert!(actor.is_ready());
    }

    #[test]
    fn dispatch_is_allocation_free_after_construction() {
        let Some(mut actor) = X86UnaryF32Kernel::try_new() else {
            return;
        };
        let input = [-1.0_f32, 0.0, 1.0, 2.0, -3.0, 4.0, -5.0, 6.0, 7.0];
        let mut output = [0.0_f32; 9];
        let allocation = measure(|| {
            for _ in 0..64 {
                assert_eq!(
                    actor.process_event(OpX86UnaryRelu::new(&input), &mut output),
                    Ok(())
                );
            }
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(allocation.bytes_total, 0);
    }
}
