//! Safe `AArch64` NEON F32 duplication kernel.
//!
//! This bounded target slice ports the pinned `op_dup` route from
//! `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6`. NEON is resolved
//! before construction through Pulp; no scalar backend fallback is selected
//! during dispatch.

#![cfg(target_arch = "aarch64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{Simd, WithSimd, aarch64::Arch};
use sml::sml;

/// Errors returned by the `AArch64` duplication actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DupF32Error {
    /// The target did not expose NEON.
    BackendUnavailable,
    /// The source and destination are not equal, non-empty dense F32 slices.
    InvalidShape,
    /// The generated machine received an event outside this API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for DupF32Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("AArch64 NEON unavailable"),
            Self::InvalidShape => formatter.write_str("invalid AArch64 dup F32 shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected AArch64 dup F32 event"),
            Self::Internal => formatter.write_str("internal AArch64 dup F32 dispatch error"),
        }
    }
}

impl std::error::Error for DupF32Error {}

/// Result returned after a duplication event reaches run-to-completion.
pub type DupF32Result = Result<(), DupF32Error>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:320-329";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:887-930,2217-2236,8448-8478";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:30-43";

/// Typed duplication request over equal-length dense F32 slices.
#[derive(Debug)]
pub struct OpAarch64DupF32<'a> {
    input: &'a [f32],
}

impl<'a> OpAarch64DupF32<'a> {
    /// Creates a duplication request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Explicitly reports an event outside the `AArch64` duplication API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedAarch64DupF32;

struct DupRuntime<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    result: &'a Cell<DupF32Result>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<DupF32Result>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    backend: pulp::aarch64::Neon,
}

sml! {
    DupF32Machine<'dispatch> {
        "ready"_s <= *"ready"_s + Dup(DupRuntime<'dispatch>)
            [guard_ready] / effect_dup,
        "ready"_s <= "ready"_s + Dup(DupRuntime<'dispatch>)
            [guard_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` NEON duplication actor.
pub struct DupF32Kernel {
    machine: DupF32MachineStateMachine<Context>,
}

impl fmt::Debug for DupF32Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DupF32Kernel")
            .finish_non_exhaustive()
    }
}

impl DupF32Kernel {
    /// Resolves NEON before allowing dispatch; no scalar backend is selected.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: DupF32MachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: DupF32Event>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated actor is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&DupF32MachineStates::Ready)
    }

    fn dup(&mut self, input: &[f32], output: &mut [f32]) -> DupF32Result {
        let result = Cell::new(Err(DupF32Error::UnexpectedEvent));
        self.machine
            .process_event(DupF32MachineEvents::Dup(DupRuntime {
                input,
                output,
                result: &result,
            }))
            .map_err(|_| DupF32Error::Internal)?;
        result.get()
    }
}

/// Event implemented by the `AArch64` duplication actor.
pub trait DupF32Event: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut DupF32Kernel, output: &mut [f32]) -> Self::Output;
}

impl DupF32Event for OpAarch64DupF32<'_> {
    type Output = DupF32Result;

    fn dispatch(self, actor: &mut DupF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.dup(self.input, output)
    }
}

impl DupF32Event for UnexpectedAarch64DupF32 {
    type Output = DupF32Result;

    fn dispatch(self, actor: &mut DupF32Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(DupF32Error::UnexpectedEvent));
        actor
            .machine
            .process_event(DupF32MachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| DupF32Error::Internal)?;
        result.get()
    }
}

impl DupF32MachineStateMachineContext for Context {
    fn guard_ready(&self, event: &DupRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_invalid(&self, event: &DupRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn effect_dup(&mut self, event: DupRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            DupOperation {
                input: event.input,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_invalid(&mut self, event: DupRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(DupF32Error::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(DupF32Error::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn request_valid(event: &DupRuntime<'_>) -> bool {
    !event.input.is_empty() && event.input.len() == event.output.len()
}

struct DupOperation<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for DupOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let (input_vectors, input_tail) = S::as_simd_f32s(self.input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(self.output);
        for (output, input) in output_vectors.iter_mut().zip(input_vectors) {
            *output = *input;
        }
        for (output, input) in output_tail.iter_mut().zip(input_tail) {
            *output = *input;
        }
        let _ = simd;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    use allocation_counter::measure;

    fn kernel() -> DupF32Kernel {
        DupF32Kernel::try_new().expect("NEON is required on AArch64")
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
            "src/emel/kernel/aarch64/guards.hpp:320-329"
        );
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:887-930,2217-2236,8448-8478"
        );
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:30-43"
        );
    }

    #[test]
    fn duplication_matches_pinned_neon_copy_for_vector_and_tail() {
        let input = [0.0_f32, -1.5, 2.25, 4.0, -8.0, 16.0, 32.0, -64.0, 128.0];
        let mut output = [7.0_f32; 9];
        let mut actor = kernel();
        assert_eq!(
            actor.process_event(OpAarch64DupF32::new(&input), &mut output),
            Ok(())
        );
        assert_eq!(output, input);
        assert!(actor.is_ready());
    }

    #[test]
    fn invalid_shape_is_explicit_and_does_not_mutate_output() {
        let mut actor = kernel();
        let mut output = [7.0_f32; 2];
        assert_eq!(
            actor.process_event(OpAarch64DupF32::new(&[1.0, 2.0, 3.0]), &mut output),
            Err(DupF32Error::InvalidShape)
        );
        assert_eq!(output, [7.0; 2]);
        assert_eq!(
            actor.process_event(OpAarch64DupF32::new(&[]), &mut output),
            Err(DupF32Error::InvalidShape)
        );
        assert_eq!(output, [7.0; 2]);
    }

    #[test]
    fn unexpected_event_is_typed_and_actor_recovers() {
        let mut actor = kernel();
        assert_eq!(
            actor.process_event(UnexpectedAarch64DupF32, &mut []),
            Err(DupF32Error::UnexpectedEvent)
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
                    actor.process_event(OpAarch64DupF32::new(&input), &mut output),
                    Ok(())
                );
            }
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(allocation.bytes_total, 0);
    }
}
