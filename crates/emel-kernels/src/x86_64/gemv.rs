//! Safe x86 F32 FMA GEMV for the pinned `op_mul_mat` vector route.
//!
//! This target slice ports the pinned `src1.ne[0] == 1` route from
//! `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6`. The maintained
//! dense FMA matrix actor owns the `n != 1` route; this actor owns only the
//! single-right-hand-side vector route. AVX2/FMA capability is resolved by
//! Pulp before construction, and no scalar backend is selected.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{Simd, WithSimd, x86::Arch};
use sml::sml;

/// Errors returned by the x86 F32 GEMV actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum X86F32GemvError {
    /// The target did not expose the required AVX2/FMA capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the pinned route.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for X86F32GemvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("x86 F32 GEMV backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid x86 F32 GEMV shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected x86 F32 GEMV event"),
            Self::Internal => formatter.write_str("internal x86 F32 GEMV dispatch error"),
        }
    }
}

impl std::error::Error for X86F32GemvError {}

/// Result returned after a target event reaches run-to-completion.
pub type X86F32GemvResult = Result<(), X86F32GemvError>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_GUARD_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";
const PINNED_ACTION_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";
const PINNED_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/x86_64/guards.hpp:32-41";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/x86_64/actions.hpp:313-325,2117-2168";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/x86_64/sm.hpp:395-403";
const MATRIX_ROUTE_OWNER: &str = "F32FmaKernel via X86Kernel::X86Fma";
const SCOPE_RESIDUAL: &str = "quantized, non-dense, and other dtypes remain residuals; n!=1 dense F32 matrix is owned by the FMA route";

/// A dense F32 single-right-hand-side request using the pinned ggml layout.
///
/// `lhs` has logical shape `[k,m]` with `k`-fastest storage, `rhs` has shape
/// `[1,k]`, and `output` has shape `[1,m]`.
#[derive(Debug)]
pub struct OpX86F32Gemv<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    m: usize,
    k: usize,
}

impl<'a> OpX86F32Gemv<'a> {
    /// Creates a request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32], m: usize, k: usize) -> Self {
        Self { lhs, rhs, m, k }
    }
}

/// Explicitly reports an event outside this target API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedX86F32Gemv;

struct GemvRuntime<'a> {
    event: OpX86F32Gemv<'a>,
    output: &'a mut [f32],
    result: &'a Cell<X86F32GemvResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<X86F32GemvResult>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    backend: pulp::x86::V3,
}

sml! {
    X86F32GemvMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Gemv(GemvRuntime<'dispatch>)
            [guard_gemv_ready] / effect_gemv_execute,
        "ready"_s <= "ready"_s + Gemv(GemvRuntime<'dispatch>)
            [guard_gemv_invalid] / effect_gemv_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion x86 F32 GEMV actor.
pub struct X86F32GemvKernel {
    machine: X86F32GemvMachineStateMachine<Context>,
}

impl fmt::Debug for X86F32GemvKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("X86F32GemvKernel")
            .finish_non_exhaustive()
    }
}

impl X86F32GemvKernel {
    /// Resolves AVX2/FMA before allowing dispatch; scalar is never substituted.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::V3(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: X86F32GemvMachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: X86F32GemvEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Returns whether the generated actor is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&X86F32GemvMachineStates::Ready)
    }

    fn gemv(&mut self, event: OpX86F32Gemv<'_>, output: &mut [f32]) -> X86F32GemvResult {
        let result = Cell::new(Err(X86F32GemvError::UnexpectedEvent));
        self.machine
            .process_event(X86F32GemvMachineEvents::Gemv(GemvRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| X86F32GemvError::Internal)?;
        result.get()
    }
}

/// Executes an already-guarded dense F32 GEMV request without actor
/// allocation or routing. The owning target state machine chooses this path.
pub(crate) fn run_gemv(
    backend: pulp::x86::V3,
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
) {
    Simd::vectorize(
        backend,
        GemvOperation {
            lhs,
            rhs,
            output,
            m,
            k,
        },
    );
}

/// Event implemented by the x86 F32 GEMV actor.
pub trait X86F32GemvEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut X86F32GemvKernel, output: &mut [f32]) -> Self::Output;
}

impl X86F32GemvEvent for OpX86F32Gemv<'_> {
    type Output = X86F32GemvResult;

    fn dispatch(self, actor: &mut X86F32GemvKernel, output: &mut [f32]) -> Self::Output {
        actor.gemv(self, output)
    }
}

impl X86F32GemvEvent for UnexpectedX86F32Gemv {
    type Output = X86F32GemvResult;

    fn dispatch(self, actor: &mut X86F32GemvKernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(X86F32GemvError::UnexpectedEvent));
        actor
            .machine
            .process_event(X86F32GemvMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| X86F32GemvError::Internal)?;
        result.get()
    }
}

impl X86F32GemvMachineStateMachineContext for Context {
    fn guard_gemv_ready(&self, event: &GemvRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && gemv_request_valid(event))
    }

    fn guard_gemv_invalid(&self, event: &GemvRuntime<'_>) -> Result<bool, ()> {
        Ok(!gemv_request_valid(event))
    }

    fn effect_gemv_execute(&mut self, event: GemvRuntime<'_>) -> Result<(), ()> {
        run_gemv(
            self.backend,
            event.event.lhs,
            event.event.rhs,
            event.output,
            event.event.m,
            event.event.k,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_gemv_invalid(&mut self, event: GemvRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(X86F32GemvError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(X86F32GemvError::UnexpectedEvent));
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

        for (row_index, output_value) in output.iter_mut().enumerate().take(m) {
            let row = &lhs[row_index * k..(row_index + 1) * k];
            let full_depth = (k / 32) * 32;
            let (full_row, vector_tail_row) = row.split_at(full_depth);
            let (full_rhs, vector_tail_rhs) = rhs.split_at(full_depth);
            let (row_vectors, _) = S::as_simd_f32s(full_row);
            let (rhs_vectors, _) = S::as_simd_f32s(full_rhs);
            let mut sums = [simd.splat_f32s(0.0); 4];
            let (row_groups, _) = row_vectors.as_chunks::<4>();
            let (rhs_groups, _) = rhs_vectors.as_chunks::<4>();
            for (vectors, rhs_group) in row_groups.iter().zip(rhs_groups) {
                for (sum, (lhs_vector, rhs_vector)) in
                    sums.iter_mut().zip(vectors.iter().zip(rhs_group))
                {
                    *sum = simd.mul_add_f32s(*lhs_vector, *rhs_vector, *sum);
                }
            }

            let vector_tail_depth = (vector_tail_row.len() / 8) * 8;
            let (vector_tail_row, scalar_tail_row) = vector_tail_row.split_at(vector_tail_depth);
            let (vector_tail_rhs, scalar_tail_rhs) = vector_tail_rhs.split_at(vector_tail_depth);
            let (tail_row_vectors, _) = S::as_simd_f32s(vector_tail_row);
            let (tail_rhs_vectors, _) = S::as_simd_f32s(vector_tail_rhs);
            for (lhs_vector, rhs_vector) in tail_row_vectors.iter().zip(tail_rhs_vectors) {
                sums[0] = simd.mul_add_f32s(*lhs_vector, *rhs_vector, sums[0]);
            }

            // Keep the reduction tree identical to the pinned AVX2 action:
            // the first two accumulators are reduced before the second pair,
            // then the two pair results are combined.  In particular, do not
            // reassociate these four sums into crossed pairs; the rounding
            // difference is observable for large-magnitude cancellation.
            let left_pair = simd.add_f32s(sums[0], sums[1]);
            let right_pair = simd.add_f32s(sums[2], sums[3]);
            let combined = simd.add_f32s(left_pair, right_pair);
            let mut total = simd.reduce_sum_f32s(combined);
            for (lhs_value, rhs_value) in scalar_tail_row.iter().zip(scalar_tail_rhs) {
                total += lhs_value * rhs_value;
            }
            *output_value = total;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    use allocation_counter::measure;

    fn kernel() -> Option<X86F32GemvKernel> {
        X86F32GemvKernel::try_new()
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
        assert_eq!(PINNED_GUARD_SPAN, "src/emel/kernel/x86_64/guards.hpp:32-41");
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/x86_64/actions.hpp:313-325,2117-2168"
        );
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/x86_64/sm.hpp:395-403"
        );
        assert_eq!(
            SCOPE_RESIDUAL,
            "quantized, non-dense, and other dtypes remain residuals; n!=1 dense F32 matrix is owned by the FMA route"
        );
        assert_eq!(MATRIX_ROUTE_OWNER, "F32FmaKernel via X86Kernel::X86Fma");
    }

    #[test]
    fn vector_route_matches_pinned_orientation_and_formula() {
        let Some(mut actor) = kernel() else {
            return;
        };
        let lhs = [1.0_f32; 35]
            .into_iter()
            .chain([2.0; 35])
            .collect::<Vec<_>>();
        let rhs = [1.0_f32; 35];
        let mut output = [f32::NAN; 2];
        assert_eq!(
            actor.process_event(OpX86F32Gemv::new(&lhs, &rhs, 2, 35), &mut output),
            Ok(())
        );
        assert_eq!(output, [35.0, 70.0]);
        assert!(actor.is_ready());
    }

    #[test]
    fn reduction_pairing_preserves_large_magnitude_cancellation() {
        let Some(mut actor) = kernel() else {
            return;
        };
        let mut lhs = [0.0_f32; 32];
        lhs[..8].fill(1.0e20);
        lhs[8..16].fill(-1.0e20);
        lhs[16..].fill(1.0);
        let rhs = [1.0_f32; 32];
        let mut output = [f32::NAN; 1];

        assert_eq!(
            actor.process_event(OpX86F32Gemv::new(&lhs, &rhs, 1, 32), &mut output),
            Ok(())
        );
        // The pinned `(s0 + s1) + (s2 + s3)` tree yields 16.  A crossed
        // `(s0 + s2) + (s1 + s3)` tree loses the sixteen unit terms to the
        // 1e20 cancellation and yields 0 instead.
        assert_eq!(output, [16.0]);
    }

    #[test]
    fn invalid_shape_is_explicit_and_does_not_mutate_output() {
        let Some(mut actor) = kernel() else {
            return;
        };
        let mut output = [7.0_f32; 2];
        assert_eq!(
            actor.process_event(OpX86F32Gemv::new(&[1.0], &[], 2, 0), &mut output),
            Err(X86F32GemvError::InvalidShape)
        );
        assert_eq!(output, [7.0; 2]);
        assert_eq!(
            actor.process_event(OpX86F32Gemv::new(&[1.0, 2.0], &[1.0], 2, 1), &mut output),
            Err(X86F32GemvError::InvalidShape)
        );
        assert_eq!(output, [7.0; 2]);
    }

    #[test]
    fn unexpected_event_is_typed_and_machine_stays_ready() {
        let Some(mut actor) = kernel() else {
            return;
        };
        assert_eq!(
            actor.process_event(UnexpectedX86F32Gemv, &mut []),
            Err(X86F32GemvError::UnexpectedEvent)
        );
        assert!(actor.is_ready());
    }

    #[test]
    fn dispatch_is_allocation_free_after_construction() {
        let Some(mut actor) = kernel() else {
            return;
        };
        let lhs = [1.0_f32; 9];
        let rhs = [2.0_f32; 9];
        let mut output = [0.0_f32; 1];
        let allocation = measure(|| {
            for _ in 0..64 {
                assert_eq!(
                    actor.process_event(OpX86F32Gemv::new(&lhs, &rhs, 1, 9), &mut output),
                    Ok(())
                );
            }
        });
        assert_eq!(allocation.count_total, 0);
        assert_eq!(allocation.bytes_total, 0);
    }
}
