//! Safe x86-64 AVX2 F32 duplication for the pinned target contract.
//!
//! This is the bounded target implementation of `op_dup` from pinned
//! `emel.cpp` commit `843a117386ef17dc5a50549bbfc821074c2141d6`. The public
//! event describes immutable input; the caller-provided destination remains
//! outside the event and is borrowed only for this synchronous dispatch.
//! Pulp's typed V3 capability is resolved during construction, before the
//! actor can accept a dispatch.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::derive_partial_eq_without_eq)]
// `sml!` emits public machine event enums for private runtime payloads. This
// is generated visibility noise and does not widen the actor API.
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{Simd, WithSimd, x86::Arch};
use sml::sml;

/// Errors returned by the x86 duplication actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum X86DupF32Error {
    /// The target did not expose AVX2.
    BackendUnavailable,
    /// The source and destination are not equal, non-empty dense F32 slices.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for X86DupF32Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("x86 AVX2 backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid x86 duplication F32 shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected x86 duplication F32 event"),
            Self::Internal => formatter.write_str("internal x86 duplication F32 dispatch error"),
        }
    }
}

impl std::error::Error for X86DupF32Error {}

/// Result returned after a duplication event reaches run-to-completion.
pub type X86DupF32Result = Result<(), X86DupF32Error>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_GUARD_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";
const PINNED_ACTION_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";
const PINNED_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/x86_64/guards.hpp:10-18,141-166";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/x86_64/actions.hpp:1664-1688,2240-2292";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/x86_64/sm.hpp:30-44";

/// A typed duplication request over a dense F32 source slice.
#[derive(Debug)]
pub struct OpX86DupF32<'a> {
    input: &'a [f32],
}

impl<'a> OpX86DupF32<'a> {
    /// Creates a duplication request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Explicitly reports an event outside the x86 duplication API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedX86DupF32;

struct DupRuntime<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    result: &'a Cell<X86DupF32Result>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<X86DupF32Result>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    backend: pulp::x86::V3,
}

sml! {
    X86DupF32Machine<'dispatch> {
        "ready"_s <= *"ready"_s + Dup(DupRuntime<'dispatch>)
            [guard_dup_ready] / effect_dup,
        "ready"_s <= "ready"_s + Dup(DupRuntime<'dispatch>)
            [guard_dup_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion x86 duplication actor.
pub struct X86DupF32Kernel {
    machine: X86DupF32MachineStateMachine<Context>,
}

impl fmt::Debug for X86DupF32Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("X86DupF32Kernel")
            .finish_non_exhaustive()
    }
}

impl X86DupF32Kernel {
    /// Resolves AVX2 before allowing dispatch; scalar is never substituted.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::V3(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: X86DupF32MachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: X86DupF32Event>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated actor is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&X86DupF32MachineStates::Ready)
    }

    fn dup(&mut self, input: &[f32], output: &mut [f32]) -> X86DupF32Result {
        let result = Cell::new(Err(X86DupF32Error::UnexpectedEvent));
        self.machine
            .process_event(X86DupF32MachineEvents::Dup(DupRuntime {
                input,
                output,
                result: &result,
            }))
            .map_err(|_| X86DupF32Error::Internal)?;
        result.get()
    }
}

/// Event implemented by the x86 duplication actor.
pub trait X86DupF32Event: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86DupF32Kernel, output: &mut [f32]) -> Self::Output;
}

impl X86DupF32Event for OpX86DupF32<'_> {
    type Output = X86DupF32Result;

    fn dispatch(self, actor: &mut X86DupF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.dup(self.input, output)
    }
}

impl X86DupF32Event for UnexpectedX86DupF32 {
    type Output = X86DupF32Result;

    fn dispatch(self, actor: &mut X86DupF32Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(X86DupF32Error::UnexpectedEvent));
        actor
            .machine
            .process_event(X86DupF32MachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| X86DupF32Error::Internal)?;
        result.get()
    }
}

impl X86DupF32MachineStateMachineContext for Context {
    fn guard_dup_ready(&self, event: &DupRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_dup_invalid(&self, event: &DupRuntime<'_>) -> Result<bool, ()> {
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
        event.result.set(Err(X86DupF32Error::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(X86DupF32Error::UnexpectedEvent));
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

    fn with_simd<S: Simd>(self, _simd: S) {
        let Self { input, output } = self;
        let (input_vectors, input_tail) = S::as_simd_f32s(input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(output);
        for (output, input) in output_vectors.iter_mut().zip(input_vectors) {
            *output = *input;
        }
        for (output, input) in output_tail.iter_mut().zip(input_tail) {
            *output = *input;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    use allocation_counter::measure;

    #[test]
    fn duplicates_dense_f32_values() {
        let input = [1.0_f32, -2.5, 4.0, 8.25, -9.0, 0.5, 3.0, 7.0, 11.0];
        let mut output = [0.0_f32; 9];
        let Some(mut kernel) = X86DupF32Kernel::try_new() else {
            return;
        };
        assert!(kernel.is_ready());
        assert_eq!(
            kernel.process_event(OpX86DupF32::new(&input), &mut output),
            Ok(())
        );
        assert_eq!(output, input);
    }

    #[test]
    fn invalid_shape_rejects_without_mutation() {
        let input = [1.0_f32, 2.0, 3.0];
        let mut output = [9.0_f32, 9.0];
        let before = output;
        let Some(mut kernel) = X86DupF32Kernel::try_new() else {
            return;
        };
        assert_eq!(
            kernel.process_event(OpX86DupF32::new(&input), &mut output),
            Err(X86DupF32Error::InvalidShape)
        );
        assert_eq!(output, before);
        let mut empty = [];
        assert_eq!(
            kernel.process_event(OpX86DupF32::new(&[]), &mut empty),
            Err(X86DupF32Error::InvalidShape)
        );
    }

    #[test]
    fn explicit_unexpected_is_typed() {
        let Some(mut kernel) = X86DupF32Kernel::try_new() else {
            return;
        };
        let mut output = [0.0_f32; 1];
        assert_eq!(
            kernel.process_event(UnexpectedX86DupF32, &mut output),
            Err(X86DupF32Error::UnexpectedEvent)
        );
    }

    #[test]
    fn dispatch_is_allocation_free() {
        let input = [1.0_f32; 17];
        let mut output = [0.0_f32; 17];
        let Some(mut kernel) = X86DupF32Kernel::try_new() else {
            return;
        };
        let allocation = measure(|| {
            for _ in 0..256 {
                assert_eq!(
                    kernel.process_event(OpX86DupF32::new(&input), &mut output),
                    Ok(())
                );
            }
        });
        assert_eq!(allocation.count_total, 0);
    }

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
            "src/emel/kernel/x86_64/guards.hpp:10-18,141-166"
        );
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/x86_64/actions.hpp:1664-1688,2240-2292"
        );
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/x86_64/sm.hpp:30-44"
        );
    }
}
