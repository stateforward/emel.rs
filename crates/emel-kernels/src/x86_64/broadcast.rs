//! Safe x86-64 AVX2 F32 row-broadcast kernels for the pinned target contract.
//!
//! The source contract is `emel.cpp` commit
//! `843a117386ef17dc5a50549bbfc821074c2141d6`. The row variant is selected
//! by `detail.hpp:3804-3819` and executed by `detail.hpp:3823-3834`; x86
//! dispatch names the add and multiply rows at `sm.hpp:56-63` and
//! `sm.hpp:121-128`. This actor requires Pulp V3 and never substitutes a
//! scalar backend.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{Simd, WithSimd, x86::Arch};
use sml::sml;

/// Errors returned by the x86 row-broadcast actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum X86BroadcastF32Error {
    /// The request is not a non-empty, multi-row dense F32 broadcast.
    InvalidShape,
    /// The generated machine received an event outside this API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for X86BroadcastF32Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid x86 broadcast F32 shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected x86 broadcast F32 event"),
            Self::Internal => formatter.write_str("internal x86 broadcast F32 dispatch error"),
        }
    }
}

impl std::error::Error for X86BroadcastF32Error {}

/// Result returned after a row-broadcast event reaches run-to-completion.
pub type X86BroadcastF32Result = Result<(), X86BroadcastF32Error>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const PINNED_GUARD_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";
const PINNED_ACTION_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";
const PINNED_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const PINNED_DETAIL_SPAN: &str = "src/emel/kernel/detail.hpp:3804-3834";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/x86_64/guards.hpp:405-434";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/x86_64/actions.hpp:2651-2658";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/x86_64/sm.hpp:56-63,121-128";

/// Adds one dense F32 row to every dense F32 input row.
#[derive(Debug)]
pub struct OpX86BroadcastAdd<'a> {
    input: &'a [f32],
    row: &'a [f32],
}

impl<'a> OpX86BroadcastAdd<'a> {
    /// Creates an addition request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], row: &'a [f32]) -> Self {
        Self { input, row }
    }
}

/// Multiplies every dense F32 input row by one dense F32 row of weights.
#[derive(Debug)]
pub struct OpX86BroadcastMul<'a> {
    input: &'a [f32],
    row: &'a [f32],
}

impl<'a> OpX86BroadcastMul<'a> {
    /// Creates a multiplication request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], row: &'a [f32]) -> Self {
        Self { input, row }
    }
}

/// Explicitly reports an event outside this broadcast API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedX86BroadcastF32;

struct BroadcastRuntime<'a> {
    input: &'a [f32],
    row: &'a [f32],
    output: &'a mut [f32],
    result: &'a Cell<X86BroadcastF32Result>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<X86BroadcastF32Result>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    backend: pulp::x86::V3,
}

sml! {
    X86BroadcastF32Machine<'dispatch> {
        "ready"_s <= *"ready"_s + Add(BroadcastRuntime<'dispatch>)
            [guard_add_ready] / effect_add,
        "ready"_s <= "ready"_s + Add(BroadcastRuntime<'dispatch>)
            [guard_add_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Mul(BroadcastRuntime<'dispatch>)
            [guard_mul_ready] / effect_mul,
        "ready"_s <= "ready"_s + Mul(BroadcastRuntime<'dispatch>)
            [guard_mul_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion x86 row-broadcast actor.
pub struct X86BroadcastF32Kernel {
    machine: X86BroadcastF32MachineStateMachine<Context>,
}

impl fmt::Debug for X86BroadcastF32Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("X86BroadcastF32Kernel")
            .finish_non_exhaustive()
    }
}

impl X86BroadcastF32Kernel {
    /// Resolves AVX2 before allowing dispatch; scalar is never substituted.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::V3(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: X86BroadcastF32MachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: X86BroadcastF32Event>(
        &mut self,
        event: E,
        output: &mut [f32],
    ) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated actor is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&X86BroadcastF32MachineStates::Ready)
    }

    fn add(&mut self, input: &[f32], row: &[f32], output: &mut [f32]) -> X86BroadcastF32Result {
        let result = Cell::new(Err(X86BroadcastF32Error::UnexpectedEvent));
        self.machine
            .process_event(X86BroadcastF32MachineEvents::Add(BroadcastRuntime {
                input,
                row,
                output,
                result: &result,
            }))
            .map_err(|_| X86BroadcastF32Error::Internal)?;
        result.get()
    }

    fn mul(&mut self, input: &[f32], row: &[f32], output: &mut [f32]) -> X86BroadcastF32Result {
        let result = Cell::new(Err(X86BroadcastF32Error::UnexpectedEvent));
        self.machine
            .process_event(X86BroadcastF32MachineEvents::Mul(BroadcastRuntime {
                input,
                row,
                output,
                result: &result,
            }))
            .map_err(|_| X86BroadcastF32Error::Internal)?;
        result.get()
    }
}

/// Event implemented by the x86 row-broadcast actor.
pub trait X86BroadcastF32Event: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86BroadcastF32Kernel, output: &mut [f32]) -> Self::Output;
}

impl X86BroadcastF32Event for OpX86BroadcastAdd<'_> {
    type Output = X86BroadcastF32Result;

    fn dispatch(self, actor: &mut X86BroadcastF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.add(self.input, self.row, output)
    }
}

impl X86BroadcastF32Event for OpX86BroadcastMul<'_> {
    type Output = X86BroadcastF32Result;

    fn dispatch(self, actor: &mut X86BroadcastF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.mul(self.input, self.row, output)
    }
}

impl X86BroadcastF32Event for UnexpectedX86BroadcastF32 {
    type Output = X86BroadcastF32Result;

    fn dispatch(self, actor: &mut X86BroadcastF32Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(X86BroadcastF32Error::UnexpectedEvent));
        actor
            .machine
            .process_event(X86BroadcastF32MachineEvents::Unexpected(
                UnexpectedRuntime { result: &result },
            ))
            .map_err(|_| X86BroadcastF32Error::Internal)?;
        result.get()
    }
}

impl X86BroadcastF32MachineStateMachineContext for Context {
    fn guard_add_ready(&self, event: &BroadcastRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_add_invalid(&self, event: &BroadcastRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn guard_mul_ready(&self, event: &BroadcastRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_mul_invalid(&self, event: &BroadcastRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn effect_add(&mut self, event: BroadcastRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            AddBroadcastOperation {
                input: event.input,
                row: event.row,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mul(&mut self, event: BroadcastRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            MulBroadcastOperation {
                input: event.input,
                row: event.row,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_invalid(&mut self, event: BroadcastRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(X86BroadcastF32Error::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(X86BroadcastF32Error::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn request_valid(event: &BroadcastRuntime<'_>) -> bool {
    !event.row.is_empty()
        && event.input.len() > event.row.len()
        && event.input.len().is_multiple_of(event.row.len())
        && event.input.len() == event.output.len()
}

struct AddBroadcastOperation<'a> {
    input: &'a [f32],
    row: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for AddBroadcastOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let columns = self.row.len();
        for (input_row, output_row) in self
            .input
            .chunks_exact(columns)
            .zip(self.output.chunks_exact_mut(columns))
        {
            let (input_vectors, input_tail) = S::as_simd_f32s(input_row);
            let (row_vectors, row_tail) = S::as_simd_f32s(self.row);
            let (output_vectors, output_tail) = S::as_mut_simd_f32s(output_row);
            for ((output, input), row_value) in output_vectors
                .iter_mut()
                .zip(input_vectors)
                .zip(row_vectors)
            {
                *output = simd.add_f32s(*input, *row_value);
            }
            for ((output, input), row_value) in output_tail.iter_mut().zip(input_tail).zip(row_tail)
            {
                *output = *input + *row_value;
            }
        }
    }
}

struct MulBroadcastOperation<'a> {
    input: &'a [f32],
    row: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for MulBroadcastOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let columns = self.row.len();
        for (input_row, output_row) in self
            .input
            .chunks_exact(columns)
            .zip(self.output.chunks_exact_mut(columns))
        {
            let (input_vectors, input_tail) = S::as_simd_f32s(input_row);
            let (row_vectors, row_tail) = S::as_simd_f32s(self.row);
            let (output_vectors, output_tail) = S::as_mut_simd_f32s(output_row);
            for ((output, input), row_value) in output_vectors
                .iter_mut()
                .zip(input_vectors)
                .zip(row_vectors)
            {
                *output = simd.mul_f32s(*input, *row_value);
            }
            for ((output, input), row_value) in output_tail.iter_mut().zip(input_tail).zip(row_tail)
            {
                *output = *input * *row_value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_source_identity_is_exact() {
        assert_eq!(
            PINNED_EMEL_CPP_COMMIT,
            "843a117386ef17dc5a50549bbfc821074c2141d6"
        );
        assert_eq!(
            PINNED_DETAIL_BLOB,
            "c8a82643eabfe8f2d7883e655955f455794511b0"
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
        assert_eq!(PINNED_DETAIL_SPAN, "src/emel/kernel/detail.hpp:3804-3834");
        assert_eq!(
            PINNED_GUARD_SPAN,
            "src/emel/kernel/x86_64/guards.hpp:405-434"
        );
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/x86_64/actions.hpp:2651-2658"
        );
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/x86_64/sm.hpp:56-63,121-128"
        );
    }
}
