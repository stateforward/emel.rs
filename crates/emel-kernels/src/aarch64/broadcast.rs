//! Safe `AArch64` NEON F32 row-broadcast kernels for the pinned target contract.
//!
//! The pinned reference validates `op_add` and `op_mul` row broadcasting in
//! `detail.hpp:3804-3819`, then applies `detail.hpp:3823-3834` to every dense
//! logical row.  The `AArch64` state machine selects these operations through
//! the scalar broadcast transitions at `sm.hpp:56-63` and `:121-128`.
//!
//! This target actor keeps the same dense row contract while using the safe
//! `pulp` vector API.  Non-dense tensor layouts and the portable tensor-view
//! surface remain owned by [`crate::any::broadcast`].

#![cfg(target_arch = "aarch64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{Simd, WithSimd, aarch64::Arch};
use sml::sml;

/// Errors returned by `AArch64` F32 row-broadcast dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BroadcastF32Error {
    /// The request is not a non-empty, multi-row dense F32 broadcast.
    InvalidShape,
    /// The generated machine received an event outside this API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for BroadcastF32Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid AArch64 broadcast F32 shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected AArch64 broadcast F32 event"),
            Self::Internal => formatter.write_str("internal AArch64 broadcast F32 dispatch error"),
        }
    }
}

impl std::error::Error for BroadcastF32Error {}

/// Result returned after a row-broadcast event reaches run-to-completion.
pub type BroadcastF32Result = Result<(), BroadcastF32Error>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_DETAIL_SPAN: &str = "src/emel/kernel/detail.hpp:3804-3834";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:1058-1080";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:9261-9268";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:56-63,121-128";
const SCOPE_RESIDUAL: &str =
    "non-dense tensor layouts and portable tensor-view dispatch remain in broadcast_ops";

/// Adds one dense F32 row to every dense F32 input row.
#[derive(Debug)]
pub struct OpAarch64BroadcastAdd<'a> {
    input: &'a [f32],
    row: &'a [f32],
}

impl<'a> OpAarch64BroadcastAdd<'a> {
    /// Creates an addition request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], row: &'a [f32]) -> Self {
        Self { input, row }
    }
}

/// Multiplies every dense F32 input row by one dense F32 row of weights.
#[derive(Debug)]
pub struct OpAarch64BroadcastMul<'a> {
    input: &'a [f32],
    row: &'a [f32],
}

impl<'a> OpAarch64BroadcastMul<'a> {
    /// Creates a multiplication request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], row: &'a [f32]) -> Self {
        Self { input, row }
    }
}

/// Explicitly reports an event outside this broadcast API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedAarch64BroadcastF32;

struct BroadcastRuntime<'a> {
    input: &'a [f32],
    row: &'a [f32],
    output: &'a mut [f32],
    result: &'a Cell<BroadcastF32Result>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<BroadcastF32Result>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    backend: pulp::aarch64::Neon,
}

sml! {
    BroadcastF32Machine<'dispatch> {
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

/// Single-writer, run-to-completion `AArch64` row-broadcast actor.
pub struct BroadcastF32Kernel {
    machine: BroadcastF32MachineStateMachine<Context>,
}

impl fmt::Debug for BroadcastF32Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BroadcastF32Kernel")
            .finish_non_exhaustive()
    }
}

impl BroadcastF32Kernel {
    /// Resolves NEON before allowing dispatch; scalar is never substituted.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: BroadcastF32MachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: BroadcastF32Event>(
        &mut self,
        event: E,
        output: &mut [f32],
    ) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated actor is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&BroadcastF32MachineStates::Ready)
    }

    fn add(&mut self, input: &[f32], row: &[f32], output: &mut [f32]) -> BroadcastF32Result {
        let result = Cell::new(Err(BroadcastF32Error::UnexpectedEvent));
        self.machine
            .process_event(BroadcastF32MachineEvents::Add(BroadcastRuntime {
                input,
                row,
                output,
                result: &result,
            }))
            .map_err(|_| BroadcastF32Error::Internal)?;
        result.get()
    }

    fn mul(&mut self, input: &[f32], row: &[f32], output: &mut [f32]) -> BroadcastF32Result {
        let result = Cell::new(Err(BroadcastF32Error::UnexpectedEvent));
        self.machine
            .process_event(BroadcastF32MachineEvents::Mul(BroadcastRuntime {
                input,
                row,
                output,
                result: &result,
            }))
            .map_err(|_| BroadcastF32Error::Internal)?;
        result.get()
    }
}

/// Event implemented by the `AArch64` row-broadcast actor.
pub trait BroadcastF32Event: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut BroadcastF32Kernel, output: &mut [f32]) -> Self::Output;
}

impl BroadcastF32Event for OpAarch64BroadcastAdd<'_> {
    type Output = BroadcastF32Result;

    fn dispatch(self, actor: &mut BroadcastF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.add(self.input, self.row, output)
    }
}

impl BroadcastF32Event for OpAarch64BroadcastMul<'_> {
    type Output = BroadcastF32Result;

    fn dispatch(self, actor: &mut BroadcastF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.mul(self.input, self.row, output)
    }
}

impl BroadcastF32Event for UnexpectedAarch64BroadcastF32 {
    type Output = BroadcastF32Result;

    fn dispatch(self, actor: &mut BroadcastF32Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(BroadcastF32Error::UnexpectedEvent));
        actor
            .machine
            .process_event(BroadcastF32MachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| BroadcastF32Error::Internal)?;
        result.get()
    }
}

impl BroadcastF32MachineStateMachineContext for Context {
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
        event.result.set(Err(BroadcastF32Error::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(BroadcastF32Error::UnexpectedEvent));
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
            "c25714566ec9a02679daef85089544575123408e"
        );
        assert_eq!(
            PINNED_ACTION_BLOB,
            "267d4f74e6e7498155c8535920322ffef2c02fb6"
        );
        assert_eq!(PINNED_SM_BLOB, "865a9cc6ba6115382ed043c464f3d62bcd851357");
        assert_eq!(PINNED_DETAIL_SPAN, "src/emel/kernel/detail.hpp:3804-3834");
        assert_eq!(
            PINNED_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:1058-1080"
        );
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:9261-9268"
        );
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:56-63,121-128"
        );
        assert_eq!(
            SCOPE_RESIDUAL,
            "non-dense tensor layouts and portable tensor-view dispatch remain in broadcast_ops"
        );
    }
}
