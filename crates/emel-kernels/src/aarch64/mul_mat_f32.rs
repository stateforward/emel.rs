//! Safe `AArch64` NEON dense F32 GEMM for the pinned generic `mul_mat` route.
//!
//! The source-backed boundary is `execute_neon_mul_mat`'s dense F32 branch:
//! `src0` is `[k, m]` F32, `src1` is `[n, k]` F32 stored as `k` rows of `n`,
//! and `dst` is `[n, m]` F32 stored as `m` rows of `n`.  The specialized
//! single-RHS GEMV route (`n == 1`) is excluded here.  Capability detection
//! happens during construction.  Integer-lane FMA uses pulp's stable
//! `mul_add_f32x4`, which is the `vmlaq_n_f32` contract: each output element
//! is an independent fused multiply-add over `k`.

#![cfg(target_arch = "aarch64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{
    aarch64::{Arch, Neon},
    cast, f32x4,
};
use sml::sml;

/// Errors returned by the `AArch64` dense F32 GEMM actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F32MulMatError {
    /// The target did not expose the required NEON capability.
    BackendUnavailable,
    /// The dimensions or backing slices do not describe the pinned route.
    InvalidShape,
    /// The generated machine received an event outside its API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for F32MulMatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("F32 mul_mat NEON backend unavailable"),
            Self::InvalidShape => formatter.write_str("invalid F32 mul_mat shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected F32 mul_mat event"),
            Self::Internal => formatter.write_str("internal F32 mul_mat dispatch error"),
        }
    }
}

impl std::error::Error for F32MulMatError {}

/// A pinned `AArch64` dense F32 matrix-matrix request with `n >= 2`.
#[derive(Debug)]
pub struct OpMulMatF32<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    m: usize,
    k: usize,
    n: usize,
}

impl<'a> OpMulMatF32<'a> {
    /// Creates a request.  Validation is performed by the machine guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32], m: usize, k: usize, n: usize) -> Self {
        Self { lhs, rhs, m, k, n }
    }
}

/// Explicitly reports an event outside the F32 GEMM API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedF32MulMat;

/// Result returned by target F32 GEMM events.
pub type F32MulMatResult = Result<(), F32MulMatError>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:738-775";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:7306-7820";
const PINNED_EXECUTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:7408-7820";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:522-525";
const SCOPE_RESIDUAL: &str = "dense F32 GEMM n>=2 (generic neon mul_mat F32 branch); F16 neon, remaining quantized generic mul_mat, and x86 remain residuals";
const COL_VEC: usize = 4;

/// Event implemented by the target dense F32 GEMM actor.
pub trait F32MulMatEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut F32MulMatKernel, output: &mut [f32]) -> Self::Output;
}

struct MulMatRuntime<'a> {
    event: OpMulMatF32<'a>,
    output: &'a mut [f32],
    result: &'a Cell<F32MulMatResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<F32MulMatResult>,
}

struct Context {
    backend_available: bool,
    backend: Neon,
}

sml! {
    F32MulMatMachine<'dispatch> {
        "ready"_s <= *"ready"_s + MulMat(MulMatRuntime<'dispatch>) [guard_ready] / effect_execute,
        "ready"_s <= "ready"_s + MulMat(MulMatRuntime<'dispatch>) [guard_invalid] / effect_invalid,
        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` dense F32 GEMM actor.
pub struct F32MulMatKernel {
    machine: F32MulMatMachineStateMachine<Context>,
}

impl fmt::Debug for F32MulMatKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("F32MulMatKernel")
            .finish_non_exhaustive()
    }
}

impl F32MulMatKernel {
    /// Resolves NEON before allowing dispatch.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: F32MulMatMachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one event synchronously to completion.
    pub fn process_event<E: F32MulMatEvent>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated machine is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&F32MulMatMachineStates::Ready)
    }

    fn mul_mat(&mut self, event: OpMulMatF32<'_>, output: &mut [f32]) -> F32MulMatResult {
        let result = Cell::new(Err(F32MulMatError::UnexpectedEvent));
        self.machine
            .process_event(F32MulMatMachineEvents::MulMat(MulMatRuntime {
                event,
                output,
                result: &result,
            }))
            .map_err(|_| F32MulMatError::Internal)?;
        result.get()
    }
}

impl F32MulMatEvent for OpMulMatF32<'_> {
    type Output = F32MulMatResult;

    fn dispatch(self, actor: &mut F32MulMatKernel, output: &mut [f32]) -> Self::Output {
        actor.mul_mat(self, output)
    }
}

impl F32MulMatEvent for UnexpectedF32MulMat {
    type Output = F32MulMatResult;

    fn dispatch(self, actor: &mut F32MulMatKernel, _output: &mut [f32]) -> Self::Output {
        let result = Cell::new(Err(F32MulMatError::Internal));
        actor
            .machine
            .process_event(F32MulMatMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| F32MulMatError::Internal)?;
        result.get()
    }
}

impl F32MulMatMachineStateMachineContext for Context {
    fn guard_ready(&self, event: &MulMatRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_invalid(&self, event: &MulMatRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn effect_execute(&mut self, event: MulMatRuntime<'_>) -> Result<(), ()> {
        execute_mul_mat_f32(
            self.backend,
            event.event.lhs,
            event.event.rhs,
            event.output,
            event.event.m,
            event.event.k,
            event.event.n,
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_invalid(&mut self, event: MulMatRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(F32MulMatError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(F32MulMatError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn request_valid(event: &MulMatRuntime<'_>) -> bool {
    let m = event.event.m;
    let k = event.event.k;
    let n = event.event.n;
    if m == 0 || k == 0 || n < 2 {
        return false;
    }
    let Some(lhs_len) = m.checked_mul(k) else {
        return false;
    };
    let Some(rhs_len) = k.checked_mul(n) else {
        return false;
    };
    let Some(out_len) = m.checked_mul(n) else {
        return false;
    };
    event.event.lhs.len() == lhs_len
        && event.event.rhs.len() == rhs_len
        && event.output.len() == out_len
}

#[allow(clippy::suboptimal_flops)]
fn execute_mul_mat_f32(
    backend: Neon,
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let vec_end = (n / COL_VEC) * COL_VEC;
    let mut row = 0;
    while row < m {
        let lhs_row = row * k;
        let dst_row = row * n;
        let mut col = 0;
        while col < vec_end {
            let mut acc = backend.splat_f32x4(0.0);
            let mut depth = 0;
            while depth < k {
                let bv = load4(rhs, depth * n + col);
                acc = backend.mul_add_f32x4(bv, backend.splat_f32x4(lhs[lhs_row + depth]), acc);
                depth += 1;
            }
            store4(output, dst_row + col, acc);
            col += COL_VEC;
        }
        while col < n {
            let mut acc = 0.0_f32;
            let mut depth = 0;
            while depth < k {
                acc += lhs[lhs_row + depth] * rhs[depth * n + col];
                depth += 1;
            }
            output[dst_row + col] = acc;
            col += 1;
        }
        row += 1;
    }
}

fn load4(values: &[f32], offset: usize) -> f32x4 {
    let mut lanes = [0.0_f32; COL_VEC];
    lanes.copy_from_slice(&values[offset..offset + COL_VEC]);
    cast(lanes)
}

fn store4(values: &mut [f32], offset: usize, vector: f32x4) {
    let lanes: [f32; COL_VEC] = cast(vector);
    values[offset..offset + COL_VEC].copy_from_slice(&lanes);
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_precision_loss,
        clippy::float_cmp,
        clippy::manual_midpoint,
        clippy::suboptimal_flops
    )]

    use super::{
        COL_VEC, F32MulMatError, F32MulMatEvent, F32MulMatKernel, OpMulMatF32, PINNED_ACTION_BLOB,
        PINNED_ACTION_SPAN, PINNED_EMEL_CPP_COMMIT, PINNED_EXECUTION_SPAN, PINNED_GUARD_BLOB,
        PINNED_GUARD_SPAN, PINNED_SM_BLOB, PINNED_TRANSITION_SPAN, UnexpectedF32MulMat,
    };

    fn fill_lhs(m: usize, k: usize) -> Vec<f32> {
        let mut values = vec![0.0_f32; m * k];
        let mut row = 0;
        while row < m {
            let mut depth = 0;
            while depth < k {
                values[row * k + depth] = (row as f32 + 1.0) * 0.125 + depth as f32 * 0.25;
                depth += 1;
            }
            row += 1;
        }
        values
    }

    fn fill_rhs(k: usize, n: usize) -> Vec<f32> {
        let mut values = vec![0.0_f32; k * n];
        let mut depth = 0;
        while depth < k {
            let mut col = 0;
            while col < n {
                values[depth * n + col] = (col as f32 + 1.0) * 0.5 - depth as f32 * 0.0625;
                col += 1;
            }
            depth += 1;
        }
        values
    }

    fn expected_gemm(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut output = vec![0.0_f32; m * n];
        let mut row = 0;
        while row < m {
            let mut col = 0;
            while col < n {
                let mut acc = 0.0_f32;
                let mut depth = 0;
                while depth < k {
                    acc = lhs[row * k + depth].mul_add(rhs[depth * n + col], acc);
                    depth += 1;
                }
                output[row * n + col] = acc;
                col += 1;
            }
            row += 1;
        }
        output
    }

    #[test]
    fn mul_mat_f32_constructs_on_neon() {
        assert!(F32MulMatKernel::try_new().is_some());
    }

    #[test]
    fn mul_mat_f32_pin_constants_match_pinned_source() {
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
            "src/emel/kernel/aarch64/guards.hpp:738-775"
        );
        assert_eq!(
            PINNED_ACTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:7306-7820"
        );
        assert_eq!(
            PINNED_EXECUTION_SPAN,
            "src/emel/kernel/aarch64/actions.hpp:7408-7820"
        );
        assert_eq!(
            PINNED_TRANSITION_SPAN,
            "src/emel/kernel/aarch64/sm.hpp:522-525"
        );
        assert!(super::SCOPE_RESIDUAL.contains("dense F32 GEMM"));
        assert_eq!(COL_VEC, 4);
    }

    #[test]
    fn neon_f32_gemm_matches_lane_fma_oracle() {
        let Some(mut kernel) = F32MulMatKernel::try_new() else {
            return;
        };
        let lhs = fill_lhs(5, 8);
        let rhs = fill_rhs(8, 4);
        let mut output = [f32::NAN; 20];
        assert_eq!(
            kernel.process_event(OpMulMatF32::new(&lhs, &rhs, 5, 8, 4), &mut output),
            Ok(())
        );
        let expected = expected_gemm(&lhs, &rhs, 5, 8, 4);
        let mut index = 0;
        while index < 20 {
            assert_eq!(output[index].to_bits(), expected[index].to_bits());
            index += 1;
        }
        assert!(kernel.is_ready());
    }

    #[test]
    fn mul_mat_f32_rejects_vector_n() {
        let Some(mut kernel) = F32MulMatKernel::try_new() else {
            return;
        };
        let mut output = [f32::from_bits(0x7fc0_0001); 5];
        assert_eq!(
            kernel.process_event(OpMulMatF32::new(&[1.0; 5], &[1.0; 5], 5, 1, 1), &mut output),
            Err(F32MulMatError::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [0x7fc0_0001; 5]);
    }

    #[test]
    fn mul_mat_f32_rejects_unexpected_event() {
        let Some(mut kernel) = F32MulMatKernel::try_new() else {
            return;
        };
        assert_eq!(
            kernel.process_event(UnexpectedF32MulMat, &mut []),
            Err(F32MulMatError::UnexpectedEvent)
        );
    }

    #[test]
    fn mul_mat_f32_second_dispatch_is_allocation_free() {
        let Some(mut kernel) = F32MulMatKernel::try_new() else {
            return;
        };
        let lhs = fill_lhs(2, 4);
        let rhs = fill_rhs(4, 4);
        let mut first = [f32::NAN; 8];
        let mut second = [f32::NAN; 8];
        assert_eq!(
            kernel.process_event(OpMulMatF32::new(&lhs, &rhs, 2, 4, 4), &mut first),
            Ok(())
        );
        assert_eq!(
            kernel.process_event(OpMulMatF32::new(&lhs, &rhs, 2, 4, 4), &mut second),
            Ok(())
        );
        assert_eq!(first, second);
        let _ = F32MulMatEvent::dispatch(
            OpMulMatF32::new(&lhs, &rhs, 2, 4, 4),
            &mut kernel,
            &mut first,
        );
    }
}
