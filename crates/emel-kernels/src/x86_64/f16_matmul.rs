//! Safe x86-64 F16 matrix multiplication for the pinned F16C/FMA route.
//!
//! The reference selects this algorithm when AVX2, F16C, and FMA are
//! available. The actor keeps the reference's four eight-lane F32 accumulators
//! and double-precision scalar tail while using Pulp's safe SIMD interface;
//! no EMEL-owned intrinsic or unsafe boundary is required.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::redundant_pub_crate)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{Simd, WithSimd};
use sml::sml;

use crate::any::f16_matmul::{F16MatmulError, F16MatmulResult, F16View};
use crate::detail::f16_matmul::{OpScalarMulMatF16, UnexpectedF16Matmul};

/// Reports whether the pinned x86 F16 vector route is available.
#[must_use]
pub fn f16_vector_supported() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("f16c")
        && std::arch::is_x86_feature_detected!("fma")
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
    backend: pulp::x86::V3,
}

sml! {
    X86F16MatmulMachine<'dispatch> {
        "ready"_s <= *"ready"_s + MulMatF16(Runtime<'dispatch>)
            [guard_f16_ready] / effect_f16_execute,
        "ready"_s <= "ready"_s + MulMatF16(Runtime<'dispatch>)
            [guard_f16_invalid] / effect_f16_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer actor for the guarded x86-64 F16 vector matrix route.
pub struct X86F16MatmulKernel {
    machine: X86F16MatmulMachineStateMachine<Context>,
}

impl fmt::Debug for X86F16MatmulKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("X86F16MatmulKernel")
            .finish_non_exhaustive()
    }
}

impl X86F16MatmulKernel {
    /// Constructs the actor after resolving the x86 SIMD backend.
    pub(crate) fn try_new() -> Option<Self> {
        let pulp::x86::Arch::V3(backend) = pulp::x86::Arch::new() else {
            return None;
        };
        Some(Self {
            machine: X86F16MatmulMachineStateMachine::new(Context {
                backend_available: f16_vector_supported(),
                backend,
            }),
        })
    }

    /// Dispatches one target-selected F16 request synchronously.
    pub(crate) fn process_event<E: X86F16MatmulEvent>(&mut self, event: E) -> E::Output {
        let output = event.dispatch(self);
        assert!(
            self.machine.is(&X86F16MatmulMachineStates::Ready),
            "x86 F16 matrix machine must return to ready after dispatch"
        );
        output
    }

    fn matmul(&mut self, event: OpScalarMulMatF16<'_>) -> F16MatmulResult {
        let result = Cell::new(Err(F16MatmulError::Internal));
        self.machine
            .process_event(X86F16MatmulMachineEvents::MulMatF16(Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| F16MatmulError::Internal)?;
        result.get()
    }

    fn unexpected(&mut self, _event: UnexpectedF16Matmul) -> F16MatmulResult {
        let result = Cell::new(Err(F16MatmulError::Internal));
        self.machine
            .process_event(X86F16MatmulMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| F16MatmulError::Internal)?;
        result.get()
    }
}

/// Typed events accepted by the target F16 actor.
pub trait X86F16MatmulEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    fn dispatch(self, actor: &mut X86F16MatmulKernel) -> Self::Output;
}

impl X86F16MatmulEvent for OpScalarMulMatF16<'_> {
    type Output = F16MatmulResult;

    fn dispatch(self, actor: &mut X86F16MatmulKernel) -> Self::Output {
        actor.matmul(self)
    }
}

impl X86F16MatmulEvent for UnexpectedF16Matmul {
    type Output = F16MatmulResult;

    fn dispatch(self, actor: &mut X86F16MatmulKernel) -> Self::Output {
        actor.unexpected(self)
    }
}

impl X86F16MatmulMachineStateMachineContext for Context {
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
                output[row + rows * column] =
                    dot_f16_vector(self.backend, lhs, rhs, row, column, k);
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
#[allow(clippy::cast_possible_truncation)]
fn dot_f16_vector(
    backend: pulp::x86::V3,
    lhs: F16View<'_>,
    rhs: F16View<'_>,
    row: usize,
    column: usize,
    k: usize,
) -> f32 {
    let lhs_start = row * k;
    let rhs_start = column * k;
    let vector_count = k & !31;
    let mut offset = 0;
    let mut total = 0.0_f64;
    while offset < vector_count {
        let mut lhs_values = [0.0_f32; 32];
        let mut rhs_values = [0.0_f32; 32];
        let mut lane = 0;
        while lane < 32 {
            lhs_values[lane] = lhs.read(lhs_start + offset + lane);
            rhs_values[lane] = rhs.read(rhs_start + offset + lane);
            lane += 1;
        }
        total += f64::from(dot_f32_block(backend, &lhs_values, &rhs_values));
        offset += 32;
    }
    while offset < k {
        total += f64::from(lhs.read(lhs_start + offset) * rhs.read(rhs_start + offset));
        offset += 1;
    }
    total as f32
}

struct F32Block<'a> {
    lhs: &'a [f32; 32],
    rhs: &'a [f32; 32],
    output: &'a mut f32,
}

#[allow(clippy::suboptimal_flops)]
impl WithSimd for F32Block<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self { lhs, rhs, output } = self;
        let zero = simd.splat_f32s(0.0);
        let mut sums = [zero; 4];
        let mut tail_sums = [0.0_f32; 4];
        let mut group = 0;
        while group < 4 {
            let lhs_group = &lhs[group * 8..(group + 1) * 8];
            let rhs_group = &rhs[group * 8..(group + 1) * 8];
            let (lhs_vectors, lhs_tail) = S::as_simd_f32s(lhs_group);
            let (rhs_vectors, rhs_tail) = S::as_simd_f32s(rhs_group);
            for (lhs_vector, rhs_vector) in lhs_vectors.iter().zip(rhs_vectors) {
                sums[group] = simd.mul_add_f32s(*lhs_vector, *rhs_vector, sums[group]);
            }
            for (lhs_value, rhs_value) in lhs_tail.iter().zip(rhs_tail) {
                tail_sums[group] += *lhs_value * *rhs_value;
            }
            group += 1;
        }
        sums[0] = simd.add_f32s(sums[0], sums[2]);
        sums[1] = simd.add_f32s(sums[1], sums[3]);
        sums[0] = simd.add_f32s(sums[0], sums[1]);

        let mut lanes = [0.0_f32; 8];
        simd.partial_store_f32s(&mut lanes, sums[0]);
        let pair0 = lanes[0] + lanes[4];
        let pair1 = lanes[1] + lanes[5];
        let pair2 = lanes[2] + lanes[6];
        let pair3 = lanes[3] + lanes[7];
        let vector_sum = (pair0 + pair1) + (pair2 + pair3);
        let tail_sum = (tail_sums[0] + tail_sums[1]) + (tail_sums[2] + tail_sums[3]);
        *output = vector_sum + tail_sum;
    }
}

#[inline]
fn dot_f32_block(backend: pulp::x86::V3, lhs: &[f32; 32], rhs: &[f32; 32]) -> f32 {
    let mut output = 0.0_f32;
    Simd::vectorize(
        backend,
        F32Block {
            lhs,
            rhs,
            output: &mut output,
        },
    );
    output
}
