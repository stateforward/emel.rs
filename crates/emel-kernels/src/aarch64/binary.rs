//! Safe `AArch64` NEON F32 binary kernels for the pinned target contract.
//!
//! This bounded slice ports the pinned equal-element-count NEON routes for
//! add, subtract, multiply, and divide. The source row-broadcast add/mul
//! variants are owned by the sibling `BroadcastF32Kernel` and
//! exposed through the target router, so this equal-count actor does not
//! duplicate that data-plane implementation. Non-dense layouts and scalar
//! fallback remain explicit residuals.

#![cfg(target_arch = "aarch64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{Simd, WithSimd, aarch64::Arch};
use sml::sml;

/// Errors returned by the `AArch64` binary actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinaryF32Error {
    /// The target did not expose NEON.
    BackendUnavailable,
    /// The requests are not equal, non-empty dense F32 slices.
    InvalidShape,
    /// The generated machine received an event outside this API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for BinaryF32Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("AArch64 NEON unavailable"),
            Self::InvalidShape => formatter.write_str("invalid AArch64 binary F32 shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected AArch64 binary F32 event"),
            Self::Internal => formatter.write_str("internal AArch64 binary F32 dispatch error"),
        }
    }
}

impl std::error::Error for BinaryF32Error {}

/// Result returned after a binary event reaches run-to-completion.
pub type BinaryF32Result = Result<(), BinaryF32Error>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:320-329,1065-1080";
const PINNED_ACTION_SPAN: &str =
    "src/emel/kernel/aarch64/actions.hpp:2239-2339,8444-8468,8576-8593";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:46-143";
const BROADCAST_ROUTE_OWNER: &str = "BroadcastF32Kernel via Kernel::BroadcastAdd/Mul";
const SCOPE_RESIDUAL: &str = "non-dense and scalar routes remain residuals; this slice requires equal non-empty dense F32 lengths";

/// Typed addition request over equal-length dense F32 slices.
#[derive(Debug)]
pub struct OpAarch64BinaryAdd<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
}

impl<'a> OpAarch64BinaryAdd<'a> {
    /// Creates an addition request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32]) -> Self {
        Self { lhs, rhs }
    }
}

/// Typed subtraction request over equal-length dense F32 slices.
#[derive(Debug)]
pub struct OpAarch64BinarySub<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
}

impl<'a> OpAarch64BinarySub<'a> {
    /// Creates a subtraction request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32]) -> Self {
        Self { lhs, rhs }
    }
}

/// Typed multiplication request over equal-length dense F32 slices.
#[derive(Debug)]
pub struct OpAarch64BinaryMul<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
}

impl<'a> OpAarch64BinaryMul<'a> {
    /// Creates a multiplication request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32]) -> Self {
        Self { lhs, rhs }
    }
}

/// Typed division request over equal-length dense F32 slices.
#[derive(Debug)]
pub struct OpAarch64BinaryDiv<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
}

impl<'a> OpAarch64BinaryDiv<'a> {
    /// Creates a division request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32]) -> Self {
        Self { lhs, rhs }
    }
}

/// Explicitly reports an event outside this binary API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedAarch64BinaryF32;

struct BinaryRuntime<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    output: &'a mut [f32],
    result: &'a Cell<BinaryF32Result>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<BinaryF32Result>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    backend: pulp::aarch64::Neon,
}

sml! {
    BinaryF32Machine<'dispatch> {
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

/// Single-writer, run-to-completion `AArch64` binary actor.
pub struct BinaryF32Kernel {
    machine: BinaryF32MachineStateMachine<Context>,
}

impl fmt::Debug for BinaryF32Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BinaryF32Kernel")
            .finish_non_exhaustive()
    }
}

impl BinaryF32Kernel {
    /// Resolves NEON before allowing dispatch; scalar is never substituted.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: BinaryF32MachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: BinaryF32Event>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated actor is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&BinaryF32MachineStates::Ready)
    }

    fn add(&mut self, lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> BinaryF32Result {
        let result = Cell::new(Err(BinaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(BinaryF32MachineEvents::Add(BinaryRuntime {
                lhs,
                rhs,
                output,
                result: &result,
            }))
            .map_err(|_| BinaryF32Error::Internal)?;
        result.get()
    }

    fn sub(&mut self, lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> BinaryF32Result {
        let result = Cell::new(Err(BinaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(BinaryF32MachineEvents::Sub(BinaryRuntime {
                lhs,
                rhs,
                output,
                result: &result,
            }))
            .map_err(|_| BinaryF32Error::Internal)?;
        result.get()
    }

    fn mul(&mut self, lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> BinaryF32Result {
        let result = Cell::new(Err(BinaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(BinaryF32MachineEvents::Mul(BinaryRuntime {
                lhs,
                rhs,
                output,
                result: &result,
            }))
            .map_err(|_| BinaryF32Error::Internal)?;
        result.get()
    }

    fn div(&mut self, lhs: &[f32], rhs: &[f32], output: &mut [f32]) -> BinaryF32Result {
        let result = Cell::new(Err(BinaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(BinaryF32MachineEvents::Div(BinaryRuntime {
                lhs,
                rhs,
                output,
                result: &result,
            }))
            .map_err(|_| BinaryF32Error::Internal)?;
        result.get()
    }
}

/// Event implemented by the `AArch64` binary actor.
pub trait BinaryF32Event: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut BinaryF32Kernel, output: &mut [f32]) -> Self::Output;
}

impl BinaryF32Event for OpAarch64BinaryAdd<'_> {
    type Output = BinaryF32Result;
    fn dispatch(self, actor: &mut BinaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.add(self.lhs, self.rhs, output)
    }
}

impl BinaryF32Event for OpAarch64BinarySub<'_> {
    type Output = BinaryF32Result;
    fn dispatch(self, actor: &mut BinaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.sub(self.lhs, self.rhs, output)
    }
}

impl BinaryF32Event for OpAarch64BinaryMul<'_> {
    type Output = BinaryF32Result;
    fn dispatch(self, actor: &mut BinaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.mul(self.lhs, self.rhs, output)
    }
}

impl BinaryF32Event for OpAarch64BinaryDiv<'_> {
    type Output = BinaryF32Result;
    fn dispatch(self, actor: &mut BinaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.div(self.lhs, self.rhs, output)
    }
}

impl BinaryF32Event for UnexpectedAarch64BinaryF32 {
    type Output = BinaryF32Result;
    fn dispatch(self, actor: &mut BinaryF32Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(BinaryF32Error::UnexpectedEvent));
        actor
            .machine
            .process_event(BinaryF32MachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| BinaryF32Error::Internal)?;
        result.get()
    }
}

impl BinaryF32MachineStateMachineContext for Context {
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
        event.result.set(Err(BinaryF32Error::InvalidShape));
        Ok(())
    }
    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(BinaryF32Error::UnexpectedEvent));
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

macro_rules! binary_operation {
    ($name:ident, $method:ident, $op:tt) => {
        struct $name<'a> { lhs: &'a [f32], rhs: &'a [f32], output: &'a mut [f32] }
        impl WithSimd for $name<'_> {
            type Output = ();
            fn with_simd<S: Simd>(self, simd: S) {
                let (lhs_vectors, lhs_tail) = S::as_simd_f32s(self.lhs);
                let (rhs_vectors, rhs_tail) = S::as_simd_f32s(self.rhs);
                let (output_vectors, output_tail) = S::as_mut_simd_f32s(self.output);
                for ((output, lhs), rhs) in output_vectors.iter_mut().zip(lhs_vectors).zip(rhs_vectors) { *output = simd.$method(*lhs, *rhs); }
                for ((output, lhs), rhs) in output_tail.iter_mut().zip(lhs_tail).zip(rhs_tail) { *output = *lhs $op *rhs; }
            }
        }
    };
}

binary_operation!(AddOperation, add_f32s, +);
binary_operation!(SubOperation, sub_f32s, -);
binary_operation!(MulOperation, mul_f32s, *);
binary_operation!(DivOperation, div_f32s, /);

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
            "src/emel/kernel/aarch64/guards.hpp:320-329,1065-1080"
        );
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:2239-2339,8444-8468,8576-8593"
        );
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:46-143"
        );
        assert_eq!(
            SCOPE_RESIDUAL,
            "non-dense and scalar routes remain residuals; this slice requires equal non-empty dense F32 lengths"
        );
        assert_eq!(
            BROADCAST_ROUTE_OWNER,
            "BroadcastF32Kernel via Kernel::BroadcastAdd/Mul"
        );
    }
}
