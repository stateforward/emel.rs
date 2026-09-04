//! Safe `AArch64` NEON F32 GEMV for the pinned kernel contract.
//!
//! The source-backed boundary is deliberately narrow: `src0` is a dense
//! `[k, m]` F32 matrix, `src1` is one dense `[k]` F32 vector, and the output is
//! one dense `[m]` F32 vector. This is the exact single-RHS route selected by
//! `can_run_neon_mul_mat_f32_vector_request`; arbitrary dense matrix routes,
//! other dtypes, and non-contiguous layouts remain explicit residuals. The
//! execution uses the pinned four-row NEON route when possible, then applies
//! the same per-row route to the remaining rows.
//!
//! Capability detection happens during construction. The dispatch action
//! receives a typed `pulp::aarch64::Neon` backend and therefore never performs
//! runtime backend selection or silently falls back to scalar code.

#![cfg(target_arch = "aarch64")]
#![allow(clippy::derive_partial_eq_without_eq)]
// `sml!` emits public machine event enums for private runtime payloads.  This
// is generated visibility noise and does not widen the actor's public API.
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{
    Simd, WithSimd,
    aarch64::{Arch, Neon},
};
use sml::sml;

/// Errors returned by the `AArch64` F32 GEMV actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32GemvError {
    /// The target did not expose the required NEON capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the pinned route.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for F32GemvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("F32 GEMV NEON backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid F32 GEMV shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected F32 GEMV event"),
            Self::Internal => formatter.write_str("internal F32 GEMV dispatch error"),
        }
    }
}

impl std::error::Error for F32GemvError {}

/// A pinned `AArch64` F32 single-RHS matrix-vector request.
///
/// `lhs` has logical shape `[k,m]` with `k`-fastest storage.  `rhs` has
/// logical shape `[1,k]`, and `output` has logical shape `[1,m]`.
#[derive(Debug)]
pub struct OpF32Gemv<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> OpF32Gemv<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// Explicitly reports an event outside the GEMV API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedF32Gemv;

/// Result returned by target F32 GEMV events.
pub type F32GemvResult = Result<(), F32GemvError>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:397-405";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:1210-1231";
const PINNED_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:7019-7042";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:515-518";
const SCOPE_RESIDUAL: &str = "F32 GEMV only (src0[k,m], src1[1,k], dst[1,m]); dense matrix and other dtypes remain unimplemented";

/// Event implemented by the target F32 GEMV actor.
pub trait F32GemvEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut F32GemvKernel, output: &mut [f32]) -> Self::Output;
}

struct GemvRuntime<'a> {
    event: OpF32Gemv<'a>,
    output: &'a mut [f32],
    result: &'a Cell<F32GemvResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<F32GemvResult>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    backend: Neon,
}

sml! {
    F32GemvMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Gemv(GemvRuntime<'dispatch>) [guard_gemv_ready] / effect_gemv_execute,
        "ready"_s <= "ready"_s + Gemv(GemvRuntime<'dispatch>) [guard_gemv_invalid] / effect_gemv_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` F32 GEMV actor.
pub struct F32GemvKernel {
    machine: F32GemvMachineStateMachine<Context>,
}

impl fmt::Debug for F32GemvKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("F32GemvKernel")
            .finish_non_exhaustive()
    }
}

impl F32GemvKernel {
    /// Resolves NEON before allowing dispatch; scalar is never substituted.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: F32GemvMachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: F32GemvEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&F32GemvMachineStates::Ready)
    }

    fn gemv(&mut self, event: OpF32Gemv<'_>, output: &mut [f32]) -> F32GemvResult {
        let result = Cell::new(Err(F32GemvError::UnexpectedEvent));
        self.machine
            .process_event(F32GemvMachineEvents::Gemv(GemvRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| F32GemvError::Internal)?;
        result.get()
    }
}

impl F32GemvEvent for OpF32Gemv<'_> {
    type Output = F32GemvResult;

    fn dispatch(self, actor: &mut F32GemvKernel, output: &mut [f32]) -> Self::Output {
        actor.gemv(self, output)
    }
}

impl F32GemvEvent for UnexpectedF32Gemv {
    type Output = F32GemvResult;

    fn dispatch(self, actor: &mut F32GemvKernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(F32GemvError::UnexpectedEvent));
        actor
            .machine
            .process_event(F32GemvMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| F32GemvError::Internal)?;
        result.get()
    }
}

impl F32GemvMachineStateMachineContext for Context {
    fn guard_gemv_ready(&self, event: &GemvRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && gemv_request_valid(event))
    }

    fn guard_gemv_invalid(&self, event: &GemvRuntime<'_>) -> Result<bool, ()> {
        Ok(!gemv_request_valid(event))
    }

    fn effect_gemv_execute(&mut self, event: GemvRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            GemvOperation {
                lhs: event.event.lhs,
                rhs: event.event.rhs,
                output: event.output,
                m: event.event.m,
                k: event.event.k,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_gemv_invalid(&mut self, event: GemvRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(F32GemvError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(F32GemvError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn gemv_request_valid(event: &GemvRuntime<'_>) -> bool {
    let request = &event.event;
    let Some(lhs_len) = request.k.checked_mul(request.m) else {
        return false;
    };
    request.m > 0
        && request.k > 0
        && request.lhs.len() == lhs_len
        && request.rhs.len() == request.k
        && event.output.len() == request.m
}

struct GemvOperation<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    output: &'a mut [f32],
    m: usize,
    k: usize,
}

impl WithSimd for GemvOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self {
            lhs,
            rhs,
            output,
            m,
            k,
        } = self;

        let mut row_index = 0;
        while row_index + 4 <= m {
            execute_four_rows(simd, lhs, rhs, output, row_index, k);
            row_index += 4;
        }
        while row_index < m {
            output[row_index] =
                execute_one_row(simd, &lhs[row_index * k..(row_index + 1) * k], rhs);
            row_index += 1;
        }
    }
}

fn execute_four_rows<S: Simd>(
    simd: S,
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    row_index: usize,
    k: usize,
) {
    let rows = [
        &lhs[row_index * k..(row_index + 1) * k],
        &lhs[(row_index + 1) * k..(row_index + 2) * k],
        &lhs[(row_index + 2) * k..(row_index + 3) * k],
        &lhs[(row_index + 3) * k..(row_index + 4) * k],
    ];
    let vector_count = (k / 16) * 16;
    let (rhs_prefix, rhs_tail) = rhs.split_at(vector_count);
    let (rhs_vectors, _) = S::as_simd_f32s(rhs_prefix);
    let (rhs_groups, _) = rhs_vectors.as_chunks::<4>();
    let mut values = [0.0_f32; 4];

    for (row_index, row) in rows.into_iter().enumerate() {
        let (row_prefix, _) = row.split_at(vector_count);
        let (row_vectors, _) = S::as_simd_f32s(row_prefix);
        let (row_groups, _) = row_vectors.as_chunks::<4>();
        for (vectors, rhs_vectors) in row_groups.iter().zip(rhs_groups) {
            let mut sums = [simd.splat_f32s(0.0); 4];
            for (sum, (lhs_vector, rhs_vector)) in
                sums.iter_mut().zip(vectors.iter().zip(rhs_vectors))
            {
                *sum = simd.mul_add_f32s(*lhs_vector, *rhs_vector, *sum);
            }
            let combined = simd.add_f32s(
                simd.add_f32s(sums[0], sums[2]),
                simd.add_f32s(sums[1], sums[3]),
            );
            values[row_index] += simd.reduce_sum_f32s(combined);
        }
        for (lhs_value, rhs_value) in row[vector_count..].iter().zip(rhs_tail) {
            values[row_index] += lhs_value * rhs_value;
        }
    }
    output[row_index..row_index + 4].copy_from_slice(&values);
}
fn execute_one_row<S: Simd>(simd: S, row: &[f32], rhs: &[f32]) -> f32 {
    let vector_count = (row.len() / 16) * 16;
    let (row_prefix, row_tail) = row.split_at(vector_count);
    let (rhs_prefix, rhs_tail) = rhs.split_at(vector_count);
    let (row_vectors, _) = S::as_simd_f32s(row_prefix);
    let (rhs_vectors, _) = S::as_simd_f32s(rhs_prefix);
    let mut sums = [simd.splat_f32s(0.0); 4];
    let (row_groups, _) = row_vectors.as_chunks::<4>();
    let (rhs_groups, _) = rhs_vectors.as_chunks::<4>();
    for (vectors, rhs_vector) in row_groups.iter().zip(rhs_groups) {
        for (sum, (lhs_vector, rhs_vector)) in sums.iter_mut().zip(vectors.iter().zip(rhs_vector)) {
            *sum = simd.mul_add_f32s(*lhs_vector, *rhs_vector, *sum);
        }
    }
    let combined = simd.add_f32s(
        simd.add_f32s(sums[0], sums[2]),
        simd.add_f32s(sums[1], sums[3]),
    );
    let mut value = simd.reduce_sum_f32s(combined);
    for (lhs_value, rhs_value) in row_tail.iter().zip(rhs_tail) {
        value += lhs_value * rhs_value;
    }
    value
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::{F32GemvError, F32GemvEvent, F32GemvKernel, OpF32Gemv, UnexpectedF32Gemv};
    use allocation_counter::measure;

    #[test]
    fn target_capability_is_resolved_before_dispatch() {
        let capability = F32GemvKernel::try_new();
        assert!(capability.is_some());
    }

    #[test]
    fn pinned_source_identity_is_explicit() {
        assert_eq!(
            super::PINNED_EMEL_CPP_COMMIT,
            "843a117386ef17dc5a50549bbfc821074c2141d6"
        );
        assert_eq!(
            super::PINNED_ACTION_BLOB,
            "267d4f74e6e7498155c8535920322ffef2c02fb6"
        );
        assert_eq!(
            super::PINNED_GUARD_BLOB,
            "c25714566ec9a02679daef85089544575123408e"
        );
        assert_eq!(
            super::PINNED_SM_BLOB,
            "865a9cc6ba6115382ed043c464f3d62bcd851357"
        );
        assert_eq!(
            super::PINNED_GUARD_SPAN,
            "src/emel/kernel/aarch64/guards.hpp:397-405"
        );
        assert_eq!(
            super::PINNED_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:1210-1231"
        );
        assert_eq!(
            super::PINNED_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:7019-7042"
        );
        assert_eq!(
            super::PINNED_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:515-518"
        );
        assert_eq!(
            super::SCOPE_RESIDUAL,
            "F32 GEMV only (src0[k,m], src1[1,k], dst[1,m]); dense matrix and other dtypes remain unimplemented"
        );
    }

    #[test]
    fn four_row_route_matches_pinned_pairing_and_scalar_tail() {
        let Some(mut kernel) = F32GemvKernel::try_new() else {
            return;
        };
        let k = 17;
        let lhs = (0_u8..68_u8)
            .map(|index| f32::from(index) - 20.0)
            .collect::<Vec<_>>();
        let rhs = (0_u8..17_u8)
            .map(|index| f32::from(index) / 4.0 - 2.0)
            .collect::<Vec<_>>();
        let mut output = [f32::NAN; 4];
        assert_eq!(
            kernel.process_event(OpF32Gemv::new(&lhs, &rhs, 4, k), &mut output),
            Ok(())
        );
        let expected = (0..4)
            .map(|row| {
                lhs[row * k..(row + 1) * k]
                    .iter()
                    .zip(&rhs)
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_eq!(output.to_vec(), expected);
        assert!(kernel.is_ready());
    }

    #[test]
    fn four_row_reduction_preserves_pinned_cancellation_order() {
        let Some(mut kernel) = F32GemvKernel::try_new() else {
            return;
        };
        let mut lhs = [0.0_f32; 4 * 32];
        for row in 0..4 {
            lhs[row * 32..row * 32 + 8].fill(1.0e20);
            lhs[row * 32 + 8..row * 32 + 16].fill(-1.0e20);
            lhs[row * 32 + 16..row * 32 + 32].fill(1.0);
        }
        let rhs = [1.0_f32; 32];
        let mut output = [f32::NAN; 4];
        assert_eq!(
            kernel.process_event(OpF32Gemv::new(&lhs, &rhs, 4, 32), &mut output),
            Ok(())
        );
        assert_eq!(output, [16.0; 4]);
    }

    #[test]
    fn gemv_orientation_matches_pinned_layout() {
        let Some(mut kernel) = F32GemvKernel::try_new() else {
            return;
        };
        let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = [7.0, 8.0, 9.0];
        let mut output = [f32::NAN; 2];
        assert_eq!(
            kernel.process_event(OpF32Gemv::new(&lhs, &rhs, 2, 3), &mut output),
            Ok(())
        );
        assert!(kernel.is_ready());
        assert_eq!(output, [50.0, 122.0]);
    }

    #[test]
    fn gemv_vector_prefix_and_scalar_tail_match_reference_partition() {
        let Some(mut kernel) = F32GemvKernel::try_new() else {
            return;
        };
        let lhs = [1.0; 17];
        let rhs = [2.0; 17];
        let mut output = [f32::NAN; 1];
        assert_eq!(
            kernel.process_event(OpF32Gemv::new(&lhs, &rhs, 1, 17), &mut output),
            Ok(())
        );
        assert_eq!(output, [34.0]);
    }

    #[test]
    fn invalid_shape_does_not_mutate_output() {
        let Some(mut kernel) = F32GemvKernel::try_new() else {
            return;
        };
        let mut output = [3.0; 2];
        assert_eq!(
            kernel.process_event(OpF32Gemv::new(&[1.0, 2.0], &[3.0], 2, 2), &mut output),
            Err(F32GemvError::InvalidShape)
        );
        assert_eq!(output, [3.0; 2]);
    }

    #[test]
    fn unexpected_event_is_explicit() {
        let Some(mut kernel) = F32GemvKernel::try_new() else {
            return;
        };
        assert_eq!(
            kernel.process_event(UnexpectedF32Gemv, &mut []),
            Err(F32GemvError::UnexpectedEvent)
        );
    }

    #[test]
    fn event_trait_remains_dispatch_surface() {
        let Some(mut kernel) = F32GemvKernel::try_new() else {
            return;
        };
        let mut output = [0.0; 1];
        assert_eq!(
            F32GemvEvent::dispatch(
                OpF32Gemv::new(&[2.0], &[3.0], 1, 1),
                &mut kernel,
                &mut output
            ),
            Ok(())
        );
        assert_eq!(output, [6.0]);
    }

    #[test]
    fn dispatch_is_allocation_free_after_construction() {
        let Some(mut kernel) = F32GemvKernel::try_new() else {
            return;
        };
        let lhs = [2.0];
        let rhs = [3.0];
        let mut output = [0.0];
        let info = measure(|| {
            assert_eq!(
                kernel.process_event(OpF32Gemv::new(&lhs, &rhs, 1, 1), &mut output),
                Ok(())
            );
        });
        assert_eq!(info.count_current, 0);
        assert_eq!(output, [6.0]);
    }
}
