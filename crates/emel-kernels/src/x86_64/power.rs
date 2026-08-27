//! Safe x86-64 AVX2 F32 square and square-root kernels.
//!
//! This bounded target slice mirrors the pinned `emel.cpp` `op_sqr` and
//! `op_sqrt` routes from commit
//! `843a117386ef17dc5a50549bbfc821074c2141d6`. Pulp supplies the safe SIMD
//! boundary; its typed V3 capability is resolved before dispatch and no scalar
//! backend is substituted when AVX2 is unavailable.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::derive_partial_eq_without_eq)]
// `sml!` emits public machine event enums for private runtime payloads. This
// is generated visibility noise and does not widen the actor API.
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{Simd, WithSimd, x86::Arch};
use sml::sml;

/// Errors returned by the x86 power actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum X86PowerF32Error {
    /// The target did not expose the required AVX2 capability.
    BackendUnavailable,
    /// The source and destination are not equal, non-empty dense F32 slices.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for X86PowerF32Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("x86 AVX2 backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid x86 power F32 shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected x86 power F32 event"),
            Self::Internal => formatter.write_str("internal x86 power F32 dispatch error"),
        }
    }
}

impl std::error::Error for X86PowerF32Error {}

/// Result returned after a power event reaches run-to-completion.
pub type X86PowerF32Result = Result<(), X86PowerF32Error>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_GUARD_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";
const PINNED_ACTION_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";
const PINNED_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/x86_64/guards.hpp:10-18";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/x86_64/actions.hpp:1812-1865,2240-2292";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/x86_64/sm.hpp:146-173";

/// Typed square request over equal-length dense F32 slices.
#[derive(Debug)]
pub struct OpX86PowerSqr<'a> {
    input: &'a [f32],
}

impl<'a> OpX86PowerSqr<'a> {
    /// Creates a square request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Typed square-root request over equal-length dense F32 slices.
#[derive(Debug)]
pub struct OpX86PowerSqrt<'a> {
    input: &'a [f32],
}

impl<'a> OpX86PowerSqrt<'a> {
    /// Creates a square-root request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Explicitly reports an event outside the x86 power API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedX86PowerF32;

struct PowerRuntime<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    result: &'a Cell<X86PowerF32Result>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<X86PowerF32Result>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    backend: pulp::x86::V3,
}

sml! {
    X86PowerF32Machine<'dispatch> {
        "ready"_s <= *"ready"_s + Sqr(PowerRuntime<'dispatch>)
            [guard_sqr_ready] / effect_sqr,
        "ready"_s <= "ready"_s + Sqr(PowerRuntime<'dispatch>)
            [guard_sqr_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Sqrt(PowerRuntime<'dispatch>)
            [guard_sqrt_ready] / effect_sqrt,
        "ready"_s <= "ready"_s + Sqrt(PowerRuntime<'dispatch>)
            [guard_sqrt_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion x86 power actor.
pub struct X86PowerF32Kernel {
    machine: X86PowerF32MachineStateMachine<Context>,
}

impl fmt::Debug for X86PowerF32Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("X86PowerF32Kernel")
            .finish_non_exhaustive()
    }
}

impl X86PowerF32Kernel {
    /// Resolves AVX2 before allowing dispatch; scalar is never substituted.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::V3(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: X86PowerF32MachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: X86PowerF32Event>(
        &mut self,
        event: E,
        output: &mut [f32],
    ) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated actor is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&X86PowerF32MachineStates::Ready)
    }

    fn sqr(&mut self, input: &[f32], output: &mut [f32]) -> X86PowerF32Result {
        let result = Cell::new(Err(X86PowerF32Error::UnexpectedEvent));
        self.machine
            .process_event(X86PowerF32MachineEvents::Sqr(PowerRuntime {
                input,
                output,
                result: &result,
            }))
            .map_err(|_| X86PowerF32Error::Internal)?;
        result.get()
    }

    fn sqrt(&mut self, input: &[f32], output: &mut [f32]) -> X86PowerF32Result {
        let result = Cell::new(Err(X86PowerF32Error::UnexpectedEvent));
        self.machine
            .process_event(X86PowerF32MachineEvents::Sqrt(PowerRuntime {
                input,
                output,
                result: &result,
            }))
            .map_err(|_| X86PowerF32Error::Internal)?;
        result.get()
    }
}

/// Event implemented by the x86 power actor.
pub trait X86PowerF32Event: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86PowerF32Kernel, output: &mut [f32]) -> Self::Output;
}

impl X86PowerF32Event for OpX86PowerSqr<'_> {
    type Output = X86PowerF32Result;

    fn dispatch(self, actor: &mut X86PowerF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.sqr(self.input, output)
    }
}

impl X86PowerF32Event for OpX86PowerSqrt<'_> {
    type Output = X86PowerF32Result;

    fn dispatch(self, actor: &mut X86PowerF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.sqrt(self.input, output)
    }
}

impl X86PowerF32Event for UnexpectedX86PowerF32 {
    type Output = X86PowerF32Result;

    fn dispatch(self, actor: &mut X86PowerF32Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(X86PowerF32Error::UnexpectedEvent));
        actor
            .machine
            .process_event(X86PowerF32MachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| X86PowerF32Error::Internal)?;
        result.get()
    }
}

impl X86PowerF32MachineStateMachineContext for Context {
    fn guard_sqr_ready(&self, event: &PowerRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_sqr_invalid(&self, event: &PowerRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn guard_sqrt_ready(&self, event: &PowerRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_sqrt_invalid(&self, event: &PowerRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn effect_sqr(&mut self, event: PowerRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            SqrOperation {
                input: event.input,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_sqrt(&mut self, event: PowerRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            SqrtOperation {
                input: event.input,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_invalid(&mut self, event: PowerRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(X86PowerF32Error::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(X86PowerF32Error::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn request_valid(event: &PowerRuntime<'_>) -> bool {
    !event.input.is_empty() && event.input.len() == event.output.len()
}

struct SqrOperation<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for SqrOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self { input, output } = self;
        let (input_vectors, input_tail) = S::as_simd_f32s(input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(output);
        for (output, input) in output_vectors.iter_mut().zip(input_vectors) {
            *output = simd.mul_f32s(*input, *input);
        }
        for (output, input) in output_tail.iter_mut().zip(input_tail) {
            *output = *input * *input;
        }
    }
}

struct SqrtOperation<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for SqrtOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self { input, output } = self;
        let (input_vectors, input_tail) = S::as_simd_f32s(input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(output);
        for (output, input) in output_vectors.iter_mut().zip(input_vectors) {
            *output = simd.sqrt_f32s(*input);
        }
        for (output, input) in output_tail.iter_mut().zip(input_tail) {
            *output = input.sqrt();
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
        assert_eq!(PINNED_GUARD_SPAN, "src/emel/kernel/x86_64/guards.hpp:10-18");
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/x86_64/actions.hpp:1812-1865,2240-2292"
        );
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/x86_64/sm.hpp:146-173"
        );
    }

    #[test]
    fn square_and_square_root_match_pinned_formulas() {
        let Some(mut actor) = X86PowerF32Kernel::try_new() else {
            return;
        };
        let input = [0.0_f32, 0.25, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0];
        let mut output = [0.0_f32; 9];
        assert_eq!(
            actor.process_event(OpX86PowerSqr::new(&input), &mut output),
            Ok(())
        );
        assert_eq!(
            output,
            [0.0, 0.0625, 1.0, 16.0, 81.0, 256.0, 625.0, 1296.0, 2401.0]
        );
        let squared = output;
        assert_eq!(
            actor.process_event(OpX86PowerSqrt::new(&squared), &mut output),
            Ok(())
        );
        assert_eq!(output, input);
    }

    #[test]
    fn invalid_shape_is_explicit_and_does_not_mutate_output() {
        let Some(mut actor) = X86PowerF32Kernel::try_new() else {
            return;
        };
        let input = [1.0_f32, 4.0, 9.0];
        let mut output = [7.0_f32; 2];
        assert_eq!(
            actor.process_event(OpX86PowerSqr::new(&input), &mut output),
            Err(X86PowerF32Error::InvalidShape)
        );
        assert_eq!(output, [7.0; 2]);
        assert_eq!(
            actor.process_event(OpX86PowerSqrt::new(&[]), &mut output),
            Err(X86PowerF32Error::InvalidShape)
        );
        assert_eq!(output, [7.0; 2]);
    }

    #[test]
    fn unexpected_event_is_typed_and_machine_stays_ready() {
        let Some(mut actor) = X86PowerF32Kernel::try_new() else {
            return;
        };
        let mut output = [0.0_f32; 1];
        assert_eq!(
            actor.process_event(UnexpectedX86PowerF32, &mut output),
            Err(X86PowerF32Error::UnexpectedEvent)
        );
        assert!(actor.is_ready());
    }

    #[test]
    fn dispatch_is_allocation_free_after_construction() {
        let Some(mut actor) = X86PowerF32Kernel::try_new() else {
            return;
        };
        let input = [1.0_f32, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0];
        let mut output = [0.0_f32; 9];
        let allocation = measure(|| {
            for _ in 0..64 {
                assert_eq!(
                    actor.process_event(OpX86PowerSqr::new(&input), &mut output),
                    Ok(())
                );
                assert_eq!(
                    actor.process_event(OpX86PowerSqrt::new(&input), &mut output),
                    Ok(())
                );
            }
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(allocation.bytes_total, 0);
    }
}
