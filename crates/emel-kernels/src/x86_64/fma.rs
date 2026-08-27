//! Safe x86 target-dispatched dense F32 matrix multiplication.
//!
//! This is the bounded target slice for the pinned `emel.cpp` F32 matmul
//! route. The source identity is `843a117386ef17dc5a50549bbfc821074c2141d6`;
//! the x86 eligibility contract is `src/emel/kernel/x86_64/guards.hpp:21-50`
//! and the tiled FMA implementation is
//! `src/emel/kernel/x86_64/actions.hpp:1993-2168`.
//!
//! This actor owns only the non-vector matrix route (`src1.ne[0] != 1`); the
//! pinned reference sends a single-right-hand-side request to the sibling GEMV
//! route with a different accumulation order. `pulp` is used only as a safe
//! SIMD boundary. The capability is resolved by
//! [`F32FmaKernel::try_new`] before any event is dispatched. A scalar
//! `pulp` backend is never accepted by this actor, so an unavailable FMA/Neon
//! target is reported at construction rather than silently substituted.

#![allow(clippy::derive_partial_eq_without_eq)]
// `sml!` emits public machine event enums even for this private runtime
// event payload. Keep the generated implementation details out of the public
// actor API while suppressing only that macro-generated visibility warning.
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

#[cfg(target_arch = "x86_64")]
use pulp::{Simd, WithSimd, x86::Arch};
use sml::sml;

/// Errors returned by the target F32 FMA actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32FmaError {
    /// The target did not expose the required FMA or Neon capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the operation.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for F32FmaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("F32 FMA backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid F32 matrix shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected F32 FMA event"),
            Self::Internal => formatter.write_str("internal F32 FMA dispatch error"),
        }
    }
}

impl std::error::Error for F32FmaError {}

/// A dense F32 matrix multiplication request using the pinned ggml layout.
///
/// `lhs` has logical shape `[k,m]`, `rhs` has logical shape `[n,k]`, and
/// `output` has logical shape `[n,m]`; dimension zero is the contiguous
/// dimension in each slice. The target route is deliberately limited to
/// non-empty, exactly sized, dense buffers.
#[derive(Debug)]
pub struct OpF32Fma<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    m: usize,
    n: usize,
    k: usize,
}

impl<'a> OpF32Fma<'a> {
    /// Creates a target F32 matmul request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32], m: usize, n: usize, k: usize) -> Self {
        Self { lhs, rhs, m, n, k }
    }
}

/// Explicitly reports an event outside the target F32 API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedF32Fma;

/// Result returned by target F32 events.
pub type F32FmaResult = Result<(), F32FmaError>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_X86_GUARD_SPAN: &str = "src/emel/kernel/x86_64/guards.hpp:21-50";
const PINNED_X86_ACTION_SPAN: &str = "src/emel/kernel/x86_64/actions.hpp:1993-2168";
const PINNED_X86_DETAIL_SPAN: &str = "src/emel/kernel/x86_64/actions.hpp:299-323";
const VECTOR_ROUTE_OWNER: &str = "X86F32GemvKernel via X86Kernel::X86Gemv";

/// Event implemented by the target F32 actor.
pub trait F32FmaEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut F32FmaKernel, output: &mut [f32]) -> Self::Output;
}

struct FmaRuntime<'a> {
    event: OpF32Fma<'a>,
    output: &'a mut [f32],
    result: &'a Cell<F32FmaResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<F32FmaResult>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    #[cfg(target_arch = "x86_64")]
    backend: pulp::x86::V3,
}

sml! {
    F32FmaMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Fma(FmaRuntime<'dispatch>) [guard_fma_ready] / effect_fma_execute,
        "ready"_s <= "ready"_s + Fma(FmaRuntime<'dispatch>) [guard_fma_invalid] / effect_fma_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion target F32 FMA actor.
pub struct F32FmaKernel {
    machine: F32FmaMachineStateMachine<Context>,
}

impl fmt::Debug for F32FmaKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("F32FmaKernel")
            .finish_non_exhaustive()
    }
}

impl F32FmaKernel {
    /// Resolves the target capability before allowing event dispatch.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        #[cfg(target_arch = "x86_64")]
        {
            let Arch::V3(backend) = Arch::new() else {
                return None;
            };
            Some(Self {
                machine: F32FmaMachineStateMachine::new(Context {
                    backend_available: true,
                    backend,
                }),
            })
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            None
        }
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: F32FmaEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Returns whether the generated actor is in its RTC-ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&F32FmaMachineStates::Ready)
    }

    fn fma(&mut self, event: OpF32Fma<'_>, output: &mut [f32]) -> F32FmaResult {
        let result = Cell::new(Err(F32FmaError::UnexpectedEvent));
        self.machine
            .process_event(F32FmaMachineEvents::Fma(FmaRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| F32FmaError::Internal)?;
        result.get()
    }
}

/// Executes an already-guarded dense F32 matrix request without actor
/// allocation or routing. The owning target state machine chooses this path.
pub(crate) fn run_fma(
    backend: pulp::x86::V3,
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    Simd::vectorize(
        backend,
        FmaOperation {
            lhs,
            rhs,
            output,
            m,
            n,
            k,
        },
    );
}

impl F32FmaEvent for OpF32Fma<'_> {
    type Output = F32FmaResult;

    fn dispatch(self, actor: &mut F32FmaKernel, output: &mut [f32]) -> Self::Output {
        actor.fma(self, output)
    }
}

impl F32FmaEvent for UnexpectedF32Fma {
    type Output = F32FmaResult;

    fn dispatch(self, actor: &mut F32FmaKernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(F32FmaError::UnexpectedEvent));
        actor
            .machine
            .process_event(F32FmaMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| F32FmaError::Internal)?;
        result.get()
    }
}

impl F32FmaMachineStateMachineContext for Context {
    fn guard_fma_ready(&self, event: &FmaRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && fma_request_valid(event))
    }

    fn guard_fma_invalid(&self, event: &FmaRuntime<'_>) -> Result<bool, ()> {
        Ok(!fma_request_valid(event))
    }

    fn effect_fma_execute(&mut self, event: FmaRuntime<'_>) -> Result<(), ()> {
        let result = event.result;
        run_fma(
            self.backend,
            event.event.lhs,
            event.event.rhs,
            event.output,
            event.event.m,
            event.event.n,
            event.event.k,
        );
        result.set(Ok(()));
        Ok(())
    }

    fn effect_fma_invalid(&mut self, event: FmaRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(F32FmaError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(F32FmaError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn fma_request_valid(event: &FmaRuntime<'_>) -> bool {
    let request = &event.event;
    let Some(lhs_len) = request.k.checked_mul(request.m) else {
        return false;
    };
    let Some(rhs_len) = request.n.checked_mul(request.k) else {
        return false;
    };
    let Some(output_len) = request.n.checked_mul(request.m) else {
        return false;
    };
    request.m > 0
        && request.n > 0
        && request.k > 0
        // The pinned x86 FMA guard excludes n == 1; that shape is routed to
        // the vector/GEMV action with a different reduction order.
        && request.n != 1
        && request.lhs.len() == lhs_len
        && request.rhs.len() == rhs_len
        && event.output.len() == output_len
}

struct FmaOperation<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    output: &'a mut [f32],
    m: usize,
    n: usize,
    k: usize,
}

impl WithSimd for FmaOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self {
            lhs,
            rhs,
            output,
            m,
            n,
            k,
        } = self;

        for row in output.chunks_exact_mut(n).take(m) {
            row.fill(0.0);
        }

        for depth in 0..k {
            let rhs_row = &rhs[depth * n..(depth + 1) * n];
            for row_index in 0..m {
                let lhs_value = lhs[row_index * k + depth];
                let output_row = &mut output[row_index * n..(row_index + 1) * n];
                let (rhs_vectors, rhs_tail) = S::as_simd_f32s(rhs_row);
                let (output_vectors, output_tail) = S::as_mut_simd_f32s(output_row);
                let scalar = simd.splat_f32s(lhs_value);
                for (output_vector, rhs_vector) in output_vectors.iter_mut().zip(rhs_vectors) {
                    *output_vector = simd.mul_add_f32s(scalar, *rhs_vector, *output_vector);
                }
                for (output_value, rhs_value) in output_tail.iter_mut().zip(rhs_tail) {
                    let product = lhs_value * *rhs_value;
                    *output_value += product;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::{F32FmaError, F32FmaEvent, F32FmaKernel, OpF32Fma, UnexpectedF32Fma};
    use allocation_counter::measure;

    #[test]
    fn target_capability_is_resolved_before_dispatch() {
        let capability = F32FmaKernel::try_new();
        #[cfg(not(target_arch = "x86_64"))]
        assert!(capability.is_none());
        #[cfg(target_arch = "x86_64")]
        let _ = capability;
    }

    #[test]
    fn pinned_source_identity_is_explicit() {
        assert_eq!(
            super::PINNED_EMEL_CPP_COMMIT,
            "843a117386ef17dc5a50549bbfc821074c2141d6"
        );
        assert_eq!(
            super::PINNED_X86_GUARD_SPAN,
            "src/emel/kernel/x86_64/guards.hpp:21-50"
        );
        assert_eq!(
            super::PINNED_X86_ACTION_SPAN,
            "src/emel/kernel/x86_64/actions.hpp:1993-2168"
        );
        assert_eq!(
            super::PINNED_X86_DETAIL_SPAN,
            "src/emel/kernel/x86_64/actions.hpp:299-323"
        );
        assert_eq!(
            super::VECTOR_ROUTE_OWNER,
            "X86F32GemvKernel via X86Kernel::X86Gemv"
        );
    }

    #[test]
    fn dense_f32_orientation_matches_pinned_layout() {
        let Some(mut kernel) = F32FmaKernel::try_new() else {
            return;
        };
        // lhs[k,m] has k-fastest storage: m=0 is [1, 2, 3], m=1 is [4, 5, 6].
        let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // rhs[n,k] has n-fastest storage: k=0 is [7, 8], k=1 is [9, 10], k=2 is [11, 12].
        let rhs = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut output = [f32::NAN; 4];
        let result = kernel.process_event(OpF32Fma::new(&lhs, &rhs, 2, 2, 3), &mut output);
        assert_eq!(result, Ok(()));
        assert!(kernel.is_ready());
        assert_eq!(output, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn invalid_shape_does_not_mutate_output() {
        let Some(mut kernel) = F32FmaKernel::try_new() else {
            return;
        };
        let mut output = [3.0; 4];
        let result = kernel.process_event(
            OpF32Fma::new(&[1.0, 2.0], &[3.0, 4.0], 2, 2, 2),
            &mut output,
        );
        assert_eq!(result, Err(F32FmaError::InvalidShape));
        assert_eq!(output, [3.0; 4]);
    }

    #[test]
    fn vector_shape_is_reserved_for_the_pinned_gemv_route() {
        let Some(mut kernel) = F32FmaKernel::try_new() else {
            return;
        };
        let mut output = [3.0; 2];
        let result = kernel.process_event(
            OpF32Fma::new(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0], 2, 1, 2),
            &mut output,
        );
        assert_eq!(result, Err(F32FmaError::InvalidShape));
        assert_eq!(output, [3.0; 2]);
    }

    #[test]
    fn scalar_tail_preserves_separate_multiply_then_add_rounding() {
        let Some(mut kernel) = F32FmaKernel::try_new() else {
            return;
        };
        let a = f32::from_bits(0x3f80_0001);
        let b = f32::from_bits(0x3f80_0001);
        let c = f32::from_bits(0x3f80_0003);
        let lhs = [a, 1.0];
        // rhs[n,k] has n-fastest storage: depth 0 is [b, 0], depth 1 is [c, 0].
        let rhs = [b, 0.0, c, 0.0];
        let mut output = [f32::NAN; 2];
        let result = kernel.process_event(OpF32Fma::new(&lhs, &rhs, 1, 2, 2), &mut output);
        assert_eq!(result, Ok(()));
        assert_eq!(output[0].to_bits(), 0x4000_0002);
        assert_eq!(output[1].to_bits(), 0);
    }

    #[test]
    fn unexpected_event_is_explicit() {
        let Some(mut kernel) = F32FmaKernel::try_new() else {
            return;
        };
        assert_eq!(
            kernel.process_event(UnexpectedF32Fma, &mut []),
            Err(F32FmaError::UnexpectedEvent)
        );
    }

    #[test]
    fn event_trait_remains_dispatch_surface() {
        let Some(mut kernel) = F32FmaKernel::try_new() else {
            return;
        };
        let mut output = [0.0; 2];
        let event = OpF32Fma::new(&[2.0], &[3.0, 4.0], 1, 2, 1);
        assert_eq!(
            F32FmaEvent::dispatch(event, &mut kernel, &mut output),
            Ok(())
        );
        assert_eq!(output, [6.0, 8.0]);
    }

    #[test]
    fn dispatch_is_allocation_free_after_construction() {
        let Some(mut kernel) = F32FmaKernel::try_new() else {
            return;
        };
        let lhs = [2.0];
        let rhs = [3.0, 4.0];
        let mut output = [0.0; 2];
        let info = measure(|| {
            assert_eq!(
                kernel.process_event(OpF32Fma::new(&lhs, &rhs, 1, 2, 1), &mut output),
                Ok(())
            );
        });
        assert_eq!(info.count_total, 0);
        assert_eq!(output, [6.0, 8.0]);
    }
}
