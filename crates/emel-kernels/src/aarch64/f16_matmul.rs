//! Safe `AArch64` F16 matrix multiplication for the pinned FP16 route.
//!
//! The reference selects this route only when NEON FP16 arithmetic is
//! available.  The crate-level safety contract forbids an EMEL-owned native
//! intrinsic boundary, so the route uses the crate's maintained IEEE-754
//! binary16 conversion helpers while retaining the reference's four
//! eight-lane accumulators, FP16 reductions, and F64 tail accumulation.  The
//! selection remains in the owning target state machine; this actor only
//! executes an already-selected F16 algorithm.

#![cfg(target_arch = "aarch64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use crate::any::f16_matmul::{F16MatmulError, F16MatmulResult, F16View};
use crate::any::quant::{fp16_to_f32, fp32_to_fp16};
use crate::detail::f16_matmul::{OpScalarMulMatF16, UnexpectedF16Matmul};

/// Reports whether the target has the capabilities required by the pinned
/// `AArch64` F16 vector guard.
#[must_use]
pub fn f16_vector_supported() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
        && std::arch::is_aarch64_feature_detected!("fp16")
}

struct Runtime<'dispatch> {
    event: OpScalarMulMatF16<'dispatch>,
    result: &'dispatch Cell<F16MatmulResult>,
}

struct UnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<F16MatmulResult>,
}

struct Context {
    backend_available: bool,
}

sml! {
    Aarch64F16MatmulMachine<'dispatch> {
        "ready"_s <= *"ready"_s + MulMatF16(Runtime<'dispatch>)
            [guard_f16_ready] / effect_f16_execute,
        "ready"_s <= "ready"_s + MulMatF16(Runtime<'dispatch>)
            [guard_f16_invalid] / effect_f16_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer actor for the guarded `AArch64` F16 vector matrix route.
pub struct Aarch64F16MatmulKernel {
    machine: Aarch64F16MatmulMachineStateMachine<Context>,
}

impl fmt::Debug for Aarch64F16MatmulKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Aarch64F16MatmulKernel")
            .finish_non_exhaustive()
    }
}

impl Aarch64F16MatmulKernel {
    /// Constructs the actor after resolving the target capability.
    pub const fn new(backend_available: bool) -> Self {
        Self {
            machine: Aarch64F16MatmulMachineStateMachine::new(Context { backend_available }),
        }
    }

    /// Dispatches one target-selected F16 request synchronously.
    pub(crate) fn process_event<E: Aarch64F16MatmulEvent>(&mut self, event: E) -> E::Output {
        let output = event.dispatch(self);
        assert!(
            self.machine.is(&Aarch64F16MatmulMachineStates::Ready),
            "AArch64 F16 matrix machine must return to ready after dispatch"
        );
        output
    }

    fn matmul(&mut self, event: OpScalarMulMatF16<'_>) -> F16MatmulResult {
        let result = Cell::new(Err(F16MatmulError::Internal));
        self.machine
            .process_event(Aarch64F16MatmulMachineEvents::MulMatF16(Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| F16MatmulError::Internal)?;
        result.get()
    }

    fn unexpected(&mut self, _event: UnexpectedF16Matmul) -> F16MatmulResult {
        let result = Cell::new(Err(F16MatmulError::Internal));
        self.machine
            .process_event(Aarch64F16MatmulMachineEvents::Unexpected(
                UnexpectedRuntime { result: &result },
            ))
            .map_err(|_| F16MatmulError::Internal)?;
        result.get()
    }
}

/// Typed events accepted by the target F16 actor.
pub trait Aarch64F16MatmulEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    fn dispatch(self, actor: &mut Aarch64F16MatmulKernel) -> Self::Output;
}

impl Aarch64F16MatmulEvent for OpScalarMulMatF16<'_> {
    type Output = F16MatmulResult;

    fn dispatch(self, actor: &mut Aarch64F16MatmulKernel) -> Self::Output {
        actor.matmul(self)
    }
}

impl Aarch64F16MatmulEvent for UnexpectedF16Matmul {
    type Output = F16MatmulResult;

    fn dispatch(self, actor: &mut Aarch64F16MatmulKernel) -> Self::Output {
        actor.unexpected(self)
    }
}

impl Aarch64F16MatmulMachineStateMachineContext for Context {
    fn guard_f16_ready(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && event.event.is_valid())
    }

    fn guard_f16_invalid(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(!event.event.is_valid())
    }

    fn effect_f16_execute(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        let (lhs, rhs, mut destination) = event.event.into_parts();
        let lhs_shape = lhs.layout().ne();
        let rhs_shape = rhs.layout().ne();
        let k = usize::try_from(lhs_shape[0]).expect("guard-proven k fits usize");
        let rows = usize::try_from(lhs_shape[1]).expect("guard-proven rows fit usize");
        let columns = usize::try_from(rhs_shape[1]).expect("guard-proven n fits usize");
        let output = destination.as_mut_slice();

        let mut column = 0;
        while column < columns {
            let mut row = 0;
            while row < rows {
                let value = dot_f16_vector(lhs, rhs, row, column, k);
                output[row + rows * column] = value;
                row += 1;
            }
            column += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_f16_invalid(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(F16MatmulError::InvalidView));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(F16MatmulError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

#[inline]
fn f16_fma(lhs: u16, rhs: u16, accumulator: u16) -> u16 {
    fp32_to_fp16(fp16_to_f32(lhs).mul_add(fp16_to_f32(rhs), fp16_to_f32(accumulator)))
}

#[inline]
fn f16_add(lhs: u16, rhs: u16) -> u16 {
    fp32_to_fp16(fp16_to_f32(lhs) + fp16_to_f32(rhs))
}

#[inline]
#[allow(clippy::cast_possible_truncation)]
fn dot_f16_vector(lhs: F16View<'_>, rhs: F16View<'_>, row: usize, column: usize, k: usize) -> f32 {
    let mut sums = [[0_u16; 8]; 4];
    let lhs_start = row * k;
    let rhs_start = column * k;
    let vector_count = k & !31;
    let mut offset = 0;
    while offset < vector_count {
        let mut group = 0;
        while group < 4 {
            let mut lane = 0;
            while lane < 8 {
                let index = offset + group * 8 + lane;
                sums[group][lane] = f16_fma(
                    lhs.read_bits(lhs_start + index),
                    rhs.read_bits(rhs_start + index),
                    sums[group][lane],
                );
                lane += 1;
            }
            group += 1;
        }
        offset += 32;
    }

    let mut lane = 0;
    while lane < 8 {
        sums[0][lane] = f16_add(sums[0][lane], sums[2][lane]);
        sums[1][lane] = f16_add(sums[1][lane], sums[3][lane]);
        sums[0][lane] = f16_add(sums[0][lane], sums[1][lane]);
        lane += 1;
    }

    let low0 = fp16_to_f32(sums[0][0]) + fp16_to_f32(sums[0][4]);
    let low1 = fp16_to_f32(sums[0][1]) + fp16_to_f32(sums[0][5]);
    let low2 = fp16_to_f32(sums[0][2]) + fp16_to_f32(sums[0][6]);
    let low3 = fp16_to_f32(sums[0][3]) + fp16_to_f32(sums[0][7]);
    let mut accumulator = f64::from((low0 + low1) + (low2 + low3));
    while offset < k {
        let lhs_value = fp16_to_f32(lhs.read_bits(lhs_start + offset));
        let rhs_value = fp16_to_f32(rhs.read_bits(rhs_start + offset));
        accumulator += f64::from(lhs_value * rhs_value);
        offset += 1;
    }
    accumulator as f32
}

#[cfg(test)]
mod tests {
    use super::dot_f16_vector;
    use crate::any::f16_matmul::F16View;
    use crate::any::quant::fp32_to_fp16;
    use crate::any::tensor_view::{DType, Layout};

    #[test]
    fn vector_route_preserves_pinned_f16_accumulation_order() {
        let layout = Layout::new(DType::F16, [32, 1, 1, 1], [2, 64, 64, 64]);
        let lhs = [fp32_to_fp16(1.0); 32];
        let rhs = [fp32_to_fp16(0.0005); 32];
        let lhs = F16View::new(&lhs, layout);
        let rhs = F16View::new(&rhs, layout);

        assert_eq!(dot_f16_vector(lhs, rhs, 0, 0, 32).to_bits(), 0x3c83_2000);
    }
}
