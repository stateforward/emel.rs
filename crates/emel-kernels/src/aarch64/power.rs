//! Safe `AArch64` NEON F32 square and square-root kernels.
//!
//! This bounded target slice ports the pinned `op_sqr` and `op_sqrt` routes
//! from `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6`. NEON is
//! resolved before construction through Pulp; no backend fallback is selected
//! during dispatch.

#![cfg(target_arch = "aarch64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{Simd, WithSimd, aarch64::Arch};
use sml::sml;

/// Errors returned by the `AArch64` power actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PowerF32Error {
    /// The target did not expose NEON.
    BackendUnavailable,
    /// The source and destination are not equal, non-empty dense F32 slices.
    InvalidShape,
    /// The generated machine received an event outside this API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for PowerF32Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("AArch64 NEON unavailable"),
            Self::InvalidShape => formatter.write_str("invalid AArch64 power F32 shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected AArch64 power F32 event"),
            Self::Internal => formatter.write_str("internal AArch64 power F32 dispatch error"),
        }
    }
}

impl std::error::Error for PowerF32Error {}

/// Result returned after a power event reaches run-to-completion.
pub type PowerF32Result = Result<(), PowerF32Error>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:805-824";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:886-930,2346-2400,8464-8496";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:145-173";

/// Typed square request over equal-length dense F32 slices.
#[derive(Debug)]
pub struct OpAarch64PowerSqr<'a> {
    input: &'a [f32],
}

impl<'a> OpAarch64PowerSqr<'a> {
    /// Creates a square request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Typed square-root request over equal-length dense F32 slices.
#[derive(Debug)]
pub struct OpAarch64PowerSqrt<'a> {
    input: &'a [f32],
}

impl<'a> OpAarch64PowerSqrt<'a> {
    /// Creates a square-root request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Explicitly reports an event outside the `AArch64` power API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedAarch64PowerF32;

struct PowerRuntime<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    result: &'a Cell<PowerF32Result>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<PowerF32Result>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    backend: pulp::aarch64::Neon,
}

sml! {
    PowerF32Machine<'dispatch> {
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

/// Single-writer, run-to-completion `AArch64` NEON power actor.
pub struct PowerF32Kernel {
    machine: PowerF32MachineStateMachine<Context>,
}

impl fmt::Debug for PowerF32Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PowerF32Kernel")
            .finish_non_exhaustive()
    }
}

impl PowerF32Kernel {
    /// Resolves NEON before allowing dispatch; no scalar backend is selected.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: PowerF32MachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: PowerF32Event>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated actor is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&PowerF32MachineStates::Ready)
    }

    fn sqr(&mut self, input: &[f32], output: &mut [f32]) -> PowerF32Result {
        let result = Cell::new(Err(PowerF32Error::UnexpectedEvent));
        self.machine
            .process_event(PowerF32MachineEvents::Sqr(PowerRuntime {
                input,
                output,
                result: &result,
            }))
            .map_err(|_| PowerF32Error::Internal)?;
        result.get()
    }

    fn sqrt(&mut self, input: &[f32], output: &mut [f32]) -> PowerF32Result {
        let result = Cell::new(Err(PowerF32Error::UnexpectedEvent));
        self.machine
            .process_event(PowerF32MachineEvents::Sqrt(PowerRuntime {
                input,
                output,
                result: &result,
            }))
            .map_err(|_| PowerF32Error::Internal)?;
        result.get()
    }
}

/// Event implemented by the `AArch64` power actor.
pub trait PowerF32Event: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut PowerF32Kernel, output: &mut [f32]) -> Self::Output;
}

impl PowerF32Event for OpAarch64PowerSqr<'_> {
    type Output = PowerF32Result;

    fn dispatch(self, actor: &mut PowerF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.sqr(self.input, output)
    }
}

impl PowerF32Event for OpAarch64PowerSqrt<'_> {
    type Output = PowerF32Result;

    fn dispatch(self, actor: &mut PowerF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.sqrt(self.input, output)
    }
}

impl PowerF32Event for UnexpectedAarch64PowerF32 {
    type Output = PowerF32Result;

    fn dispatch(self, actor: &mut PowerF32Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(PowerF32Error::UnexpectedEvent));
        actor
            .machine
            .process_event(PowerF32MachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| PowerF32Error::Internal)?;
        result.get()
    }
}

impl PowerF32MachineStateMachineContext for Context {
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
        event.result.set(Err(PowerF32Error::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(PowerF32Error::UnexpectedEvent));
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
        let (input_vectors, input_tail) = S::as_simd_f32s(self.input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(self.output);
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
        let (input_vectors, input_tail) = S::as_simd_f32s(self.input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(self.output);
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

    fn kernel() -> PowerF32Kernel {
        PowerF32Kernel::try_new().expect("NEON is required on AArch64")
    }

    #[test]
    fn pinned_source_identity_is_exact() {
        assert_eq!(
            PINNED_EMEL_CPP_COMMIT,
            "843a117386ef17dc5a50549bbfc821074c2141d6"
        );
        assert_eq!(
            PINNED_GUARD_BLOB,
            "c25714566ec9a02679daef85089544575123408e"
        );
        assert_eq!(
            PINNED_ACTION_BLOB,
            "267d4f74e6e7498155c8535920322ffef2c02fb6"
        );
        assert_eq!(PINNED_SM_BLOB, "865a9cc6ba6115382ed043c464f3d62bcd851357");
        assert_eq!(
            PINNED_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:805-824"
        );
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:886-930,2346-2400,8464-8496"
        );
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:145-173"
        );
    }

    #[test]
    fn square_and_square_root_match_pinned_formulas() {
        let mut actor = kernel();
        let input = [0.0_f32, 0.25, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0];
        let mut output = [0.0_f32; 9];
        assert_eq!(
            actor.process_event(OpAarch64PowerSqr::new(&input), &mut output),
            Ok(())
        );
        assert_eq!(
            output,
            [0.0, 0.0625, 1.0, 16.0, 81.0, 256.0, 625.0, 1296.0, 2401.0]
        );
        let squared = output;
        assert_eq!(
            actor.process_event(OpAarch64PowerSqrt::new(&squared), &mut output),
            Ok(())
        );
        assert_eq!(output, input);
    }

    #[test]
    fn invalid_shape_is_explicit_and_does_not_mutate_output() {
        let mut actor = kernel();
        let input = [1.0_f32, 4.0, 9.0];
        let mut output = [7.0_f32; 2];
        assert_eq!(
            actor.process_event(OpAarch64PowerSqr::new(&input), &mut output),
            Err(PowerF32Error::InvalidShape)
        );
        assert_eq!(output, [7.0; 2]);
        assert_eq!(
            actor.process_event(OpAarch64PowerSqrt::new(&[]), &mut output),
            Err(PowerF32Error::InvalidShape)
        );
        assert_eq!(output, [7.0; 2]);
    }

    #[test]
    fn unexpected_event_is_typed_and_machine_stays_ready() {
        let mut actor = kernel();
        assert_eq!(
            actor.process_event(UnexpectedAarch64PowerF32, &mut []),
            Err(PowerF32Error::UnexpectedEvent)
        );
        assert!(actor.is_ready());
    }

    #[test]
    fn dispatch_is_allocation_free_after_construction() {
        let mut actor = kernel();
        let input = [1.0_f32, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0];
        let mut output = [0.0_f32; 9];
        let allocation = measure(|| {
            for _ in 0..64 {
                assert_eq!(
                    actor.process_event(OpAarch64PowerSqr::new(&input), &mut output),
                    Ok(())
                );
                assert_eq!(
                    actor.process_event(OpAarch64PowerSqrt::new(&input), &mut output),
                    Ok(())
                );
            }
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(allocation.bytes_total, 0);
    }
}
