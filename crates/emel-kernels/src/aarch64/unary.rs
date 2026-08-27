//! Safe `AArch64` NEON F32 unary kernels for the pinned target contract.
//!
//! This module ports the bounded NEON `abs`, `neg`, `relu`, and vector-aligned
//! `silu` routes from `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6`.
//! The pinned scalar tail executes after the vector loop for non-multiple-of-
//! four counts. It remains inside the already-selected `SiLU` action and does
//! not select a different operation or backend.
//!
//! The NEON capability is resolved during construction. Every public event is
//! typed by operation, so actions execute an already-selected operation and
//! never branch on a runtime sub-operation.

#![cfg(target_arch = "aarch64")]
#![allow(clippy::derive_partial_eq_without_eq)]
// `sml!` emits public machine event enums for private synchronous payloads.
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{Simd, WithSimd, aarch64::Arch};
use sml::sml;

/// Errors returned by the `AArch64` NEON unary actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnaryF32Error {
    /// The target did not expose NEON.
    BackendUnavailable,
    /// The source and destination are not equal, non-empty F32 slices.
    InvalidShape,
    /// The generated machine received an event outside this API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for UnaryF32Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("AArch64 NEON unavailable"),
            Self::InvalidShape => formatter.write_str("invalid AArch64 unary F32 shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected AArch64 unary F32 event"),
            Self::Internal => formatter.write_str("internal AArch64 unary F32 dispatch error"),
        }
    }
}

impl std::error::Error for UnaryF32Error {}

/// Result returned after a unary event reaches run-to-completion.
pub type UnaryF32Result = Result<(), UnaryF32Error>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:826-886";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/aarch64/actions.hpp:89-123,180-196";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:1161-1178";
const SILU_RESIDUAL: &str = "none for the pinned dense F32 SiLU route";

/// Typed absolute-value request over a dense F32 source slice.
#[derive(Debug)]
pub struct OpAarch64UnaryAbs<'a> {
    input: &'a [f32],
}

impl<'a> OpAarch64UnaryAbs<'a> {
    /// Creates an absolute-value request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Typed negation request over a dense F32 source slice.
#[derive(Debug)]
pub struct OpAarch64UnaryNeg<'a> {
    input: &'a [f32],
}

impl<'a> OpAarch64UnaryNeg<'a> {
    /// Creates a negation request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Typed `ReLU` request over a dense F32 source slice.
#[derive(Debug)]
pub struct OpAarch64UnaryRelu<'a> {
    input: &'a [f32],
}

impl<'a> OpAarch64UnaryRelu<'a> {
    /// Creates a `ReLU` request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Vector-aligned pinned-NEON `SiLU` request over a dense F32 source slice.
#[derive(Debug)]
pub struct OpAarch64UnarySilu<'a> {
    input: &'a [f32],
}

impl<'a> OpAarch64UnarySilu<'a> {
    /// Creates a `SiLU` request. Validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Explicitly reports an event outside this unary API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedAarch64UnaryF32;

struct UnaryRuntime<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    result: &'a Cell<UnaryF32Result>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<UnaryF32Result>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    backend: pulp::aarch64::Neon,
}

sml! {
    UnaryF32Machine<'dispatch> {
        "ready"_s <= *"ready"_s + Abs(UnaryRuntime<'dispatch>) [guard_abs_ready] / effect_abs,
        "ready"_s <= "ready"_s + Abs(UnaryRuntime<'dispatch>) [guard_abs_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Neg(UnaryRuntime<'dispatch>) [guard_neg_ready] / effect_neg,
        "ready"_s <= "ready"_s + Neg(UnaryRuntime<'dispatch>) [guard_neg_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Relu(UnaryRuntime<'dispatch>) [guard_relu_ready] / effect_relu,
        "ready"_s <= "ready"_s + Relu(UnaryRuntime<'dispatch>) [guard_relu_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Silu(UnaryRuntime<'dispatch>) [guard_silu_ready] / effect_silu,
        "ready"_s <= "ready"_s + Silu(UnaryRuntime<'dispatch>) [guard_silu_invalid] / effect_invalid,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` NEON unary actor.
pub struct UnaryF32Kernel {
    machine: UnaryF32MachineStateMachine<Context>,
}

impl fmt::Debug for UnaryF32Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnaryF32Kernel")
            .finish_non_exhaustive()
    }
}

impl UnaryF32Kernel {
    /// Resolves NEON before allowing dispatch; scalar is never substituted.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: UnaryF32MachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: UnaryF32Event>(&mut self, event: E, output: &mut [f32]) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated actor is in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&UnaryF32MachineStates::Ready)
    }

    fn abs(&mut self, input: &[f32], output: &mut [f32]) -> UnaryF32Result {
        let result = Cell::new(Err(UnaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(UnaryF32MachineEvents::Abs(UnaryRuntime {
                input,
                output,
                result: &result,
            }))
            .map_err(|_| UnaryF32Error::Internal)?;
        result.get()
    }

    fn neg(&mut self, input: &[f32], output: &mut [f32]) -> UnaryF32Result {
        let result = Cell::new(Err(UnaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(UnaryF32MachineEvents::Neg(UnaryRuntime {
                input,
                output,
                result: &result,
            }))
            .map_err(|_| UnaryF32Error::Internal)?;
        result.get()
    }

    fn relu(&mut self, input: &[f32], output: &mut [f32]) -> UnaryF32Result {
        let result = Cell::new(Err(UnaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(UnaryF32MachineEvents::Relu(UnaryRuntime {
                input,
                output,
                result: &result,
            }))
            .map_err(|_| UnaryF32Error::Internal)?;
        result.get()
    }

    fn silu(&mut self, input: &[f32], output: &mut [f32]) -> UnaryF32Result {
        let result = Cell::new(Err(UnaryF32Error::UnexpectedEvent));
        self.machine
            .process_event(UnaryF32MachineEvents::Silu(UnaryRuntime {
                input,
                output,
                result: &result,
            }))
            .map_err(|_| UnaryF32Error::Internal)?;
        result.get()
    }
}

/// Event implemented by the `AArch64` unary actor.
pub trait UnaryF32Event: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut UnaryF32Kernel, output: &mut [f32]) -> Self::Output;
}

impl UnaryF32Event for OpAarch64UnaryAbs<'_> {
    type Output = UnaryF32Result;

    fn dispatch(self, actor: &mut UnaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.abs(self.input, output)
    }
}

impl UnaryF32Event for OpAarch64UnaryNeg<'_> {
    type Output = UnaryF32Result;

    fn dispatch(self, actor: &mut UnaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.neg(self.input, output)
    }
}

impl UnaryF32Event for OpAarch64UnaryRelu<'_> {
    type Output = UnaryF32Result;

    fn dispatch(self, actor: &mut UnaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.relu(self.input, output)
    }
}

impl UnaryF32Event for OpAarch64UnarySilu<'_> {
    type Output = UnaryF32Result;

    fn dispatch(self, actor: &mut UnaryF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.silu(self.input, output)
    }
}

impl UnaryF32Event for UnexpectedAarch64UnaryF32 {
    type Output = UnaryF32Result;

    fn dispatch(self, actor: &mut UnaryF32Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(UnaryF32Error::UnexpectedEvent));
        actor
            .machine
            .process_event(UnaryF32MachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| UnaryF32Error::Internal)?;
        result.get()
    }
}

impl UnaryF32MachineStateMachineContext for Context {
    fn guard_abs_ready(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_abs_invalid(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn guard_neg_ready(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_neg_invalid(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn guard_relu_ready(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_relu_invalid(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn guard_silu_ready(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && silu_request_valid(event))
    }

    fn guard_silu_invalid(&self, event: &UnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(!silu_request_valid(event))
    }

    fn effect_abs(&mut self, event: UnaryRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            AbsOperation {
                input: event.input,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_neg(&mut self, event: UnaryRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            NegOperation {
                input: event.input,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_relu(&mut self, event: UnaryRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            ReluOperation {
                input: event.input,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_silu(&mut self, event: UnaryRuntime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            SiluOperation {
                input: event.input,
                output: event.output,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_invalid(&mut self, event: UnaryRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(UnaryF32Error::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(UnaryF32Error::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const fn request_valid(event: &UnaryRuntime<'_>) -> bool {
    !event.input.is_empty() && event.input.len() == event.output.len()
}

const fn silu_request_valid(event: &UnaryRuntime<'_>) -> bool {
    request_valid(event)
}

struct AbsOperation<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for AbsOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let (input_vectors, input_tail) = S::as_simd_f32s(self.input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(self.output);
        for (destination, source) in output_vectors.iter_mut().zip(input_vectors) {
            *destination = simd.abs_f32s(*source);
        }
        for (destination, source) in output_tail.iter_mut().zip(input_tail) {
            *destination = source.abs();
        }
    }
}

struct NegOperation<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for NegOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let (input_vectors, input_tail) = S::as_simd_f32s(self.input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(self.output);
        for (destination, source) in output_vectors.iter_mut().zip(input_vectors) {
            *destination = simd.neg_f32s(*source);
        }
        for (destination, source) in output_tail.iter_mut().zip(input_tail) {
            *destination = -*source;
        }
    }
}

struct ReluOperation<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for ReluOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let zero = simd.splat_f32s(0.0);
        let (input_vectors, input_tail) = S::as_simd_f32s(self.input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(self.output);
        for (destination, source) in output_vectors.iter_mut().zip(input_vectors) {
            *destination = simd.max_f32s(*source, zero);
        }
        for (destination, source) in output_tail.iter_mut().zip(input_tail) {
            *destination = source.max(0.0);
        }
    }
}

struct SiluOperation<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for SiluOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let (input_vectors, input_tail) = S::as_simd_f32s(self.input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(self.output);
        for (destination, source) in output_vectors.iter_mut().zip(input_vectors) {
            let neg_source = simd.sub_f32s(simd.splat_f32s(0.0), *source);
            let magic_bias = simd.splat_f32s(f32::from_bits(0x4b40_0000));
            let exponent_scaled = simd.mul_add_f32s(
                neg_source,
                simd.splat_f32s(f32::from_bits(0x3fb8_aa3b)),
                magic_bias,
            );
            let reduced = simd.sub_f32s(exponent_scaled, magic_bias);
            let residual = simd.negate_mul_add_f32s(
                reduced,
                simd.splat_f32s(f32::from_bits(0x3f31_7200)),
                neg_source,
            );
            let residual = simd.negate_mul_add_f32s(
                reduced,
                simd.splat_f32s(f32::from_bits(0x35bf_be8e)),
                residual,
            );
            let exponent_bits_source: S::u32s = pulp::bytemuck::cast(exponent_scaled);
            let exponent_bits =
                simd.wrapping_dyn_shl_u32s(exponent_bits_source, simd.splat_u32s(23));
            let one_bits: S::u32s = pulp::bytemuck::cast(simd.splat_f32s(1.0));
            let exp_base_bits = simd.add_u32s(exponent_bits, one_bits);
            let exp_base: S::f32s = pulp::bytemuck::cast(exp_base_bits);
            let large_reduced =
                simd.greater_than_f32s(simd.abs_f32s(reduced), simd.splat_f32s(126.0));
            let residual_squared = simd.mul_f32s(residual, residual);
            let polynomial_low = simd.mul_add_f32s(
                residual,
                simd.splat_f32s(f32::from_bits(0x3e2a_af33)),
                simd.splat_f32s(f32::from_bits(0x3eff_fedb)),
            );
            let polynomial_high = simd.mul_add_f32s(
                residual,
                simd.splat_f32s(f32::from_bits(0x3c07_2010)),
                simd.splat_f32s(f32::from_bits(0x3d2b_9f17)),
            );
            let polynomial_inner =
                simd.mul_add_f32s(polynomial_high, residual_squared, polynomial_low);
            let polynomial = simd.mul_add_f32s(
                polynomial_inner,
                residual_squared,
                simd.mul_f32s(simd.splat_f32s(f32::from_bits(0x3f7f_fff6)), residual),
            );
            let reduced_le_zero = simd.less_than_or_equal_f32s(reduced, simd.splat_f32s(0.0));
            let reduced_le_zero_bits: S::u32s = pulp::bytemuck::cast(reduced_le_zero);
            let sign_adjustment = simd.and_u32s(reduced_le_zero_bits, simd.splat_u32s(0x8200_0000));
            let scale_one_bits = simd.add_u32s(sign_adjustment, simd.splat_u32s(0x7f00_0000));
            let scale_one: S::f32s = pulp::bytemuck::cast(scale_one_bits);
            let scale_two_bits = simd.sub_u32s(exponent_bits, sign_adjustment);
            let scale_two: S::f32s = pulp::bytemuck::cast(scale_two_bits);
            let saturated = simd.greater_than_f32s(simd.abs_f32s(reduced), simd.splat_f32s(192.0));
            let saturated_value = simd.mul_f32s(scale_one, scale_one);
            let large_value = simd.mul_f32s(
                simd.mul_add_f32s(scale_two, polynomial, scale_two),
                scale_one,
            );
            let regular_value = simd.mul_add_f32s(exp_base, polynomial, exp_base);
            let exp_neg_bits = simd.select_u32s(
                saturated,
                pulp::bytemuck::cast(saturated_value),
                simd.select_u32s(
                    large_reduced,
                    pulp::bytemuck::cast(large_value),
                    pulp::bytemuck::cast(regular_value),
                ),
            );
            let exp_neg: S::f32s = pulp::bytemuck::cast(exp_neg_bits);
            let denominator = simd.add_f32s(simd.splat_f32s(1.0), exp_neg);
            *destination = simd.div_f32s(*source, denominator);
        }
        for (destination, source) in output_tail.iter_mut().zip(input_tail) {
            *destination = *source / (1.0 + (-*source).exp());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use allocation_counter::measure;

    fn kernel() -> UnaryF32Kernel {
        UnaryF32Kernel::try_new().expect("NEON is required on AArch64")
    }

    #[test]
    fn pinned_source_identity_is_explicit() {
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
        assert!(PINNED_GUARD_SPAN.contains("guards.hpp:826-886"));
        assert!(PINNED_ACTION_SPAN.contains("actions.hpp:89-123,180-196"));
        assert!(PINNED_TRANSITION_SPAN.contains("sm.hpp:1161-1178"));
        assert_eq!(SILU_RESIDUAL, "none for the pinned dense F32 SiLU route");
    }

    #[test]
    fn unary_routes_match_pinned_neon_formulas() {
        let input = [-2.0, -0.0, 0.5, 3.0, -4.0];
        let mut output = [0.0; 5];
        let mut actor = kernel();
        assert_eq!(
            actor.process_event(OpAarch64UnaryAbs::new(&input), &mut output),
            Ok(())
        );
        assert_eq!(
            output.map(f32::to_bits),
            [
                2.0_f32.to_bits(),
                0.0_f32.to_bits(),
                0.5_f32.to_bits(),
                3.0_f32.to_bits(),
                4.0_f32.to_bits()
            ]
        );
        assert_eq!(
            actor.process_event(OpAarch64UnaryNeg::new(&input), &mut output),
            Ok(())
        );
        assert_eq!(
            output.map(f32::to_bits),
            [
                2.0_f32.to_bits(),
                0.0_f32.to_bits(),
                (-0.5_f32).to_bits(),
                (-3.0_f32).to_bits(),
                4.0_f32.to_bits()
            ]
        );
        assert_eq!(
            actor.process_event(OpAarch64UnaryRelu::new(&input), &mut output),
            Ok(())
        );
        assert_eq!(
            output.map(f32::to_bits),
            [
                0.0_f32.to_bits(),
                0.0_f32.to_bits(),
                0.5_f32.to_bits(),
                3.0_f32.to_bits(),
                0.0_f32.to_bits()
            ]
        );
    }

    #[test]
    fn invalid_shape_does_not_mutate_output() {
        let mut actor = kernel();
        let mut output = [7.0; 2];
        assert_eq!(
            actor.process_event(OpAarch64UnaryAbs::new(&[1.0]), &mut output),
            Err(UnaryF32Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [7.0_f32.to_bits(); 2]);
    }

    #[test]
    fn silu_matches_formula_for_vector_aligned_input() {
        let input = [-4.0, -1.0, 0.0, 4.0];
        let mut output = [f32::from_bits(0x7fc0_0001); 4];
        let mut actor = kernel();
        assert_eq!(
            actor.process_event(OpAarch64UnarySilu::new(&input), &mut output),
            Ok(())
        );
        assert_eq!(
            output.map(f32::to_bits),
            [0xbd93_57d2, 0xbe89_b2b0, 0x0000_0000, 0x407b_6541]
        );
        assert_eq!(output[2].to_bits(), 0.0_f32.to_bits());
        for (&value, &actual) in input.iter().zip(&output) {
            let expected = value / (1.0 + (-value).exp());
            assert!(
                (actual - expected).abs() <= 2.0e-3,
                "{value}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn silu_scalar_tail_matches_pinned_formula() {
        let mut actor = kernel();
        let mut output = [f32::NAN; 3];
        assert_eq!(
            actor.process_event(OpAarch64UnarySilu::new(&[1.0, 2.0, 3.0]), &mut output),
            Ok(())
        );
        assert_eq!(
            output.map(f32::to_bits),
            [0x3f3b_26a8, 0x3fe1_7bea, 0x4036_e4ed]
        );
    }

    #[test]
    fn silu_shape_mismatch_does_not_mutate_output() {
        let mut actor = kernel();
        let mut output = [7.0; 2];
        assert_eq!(
            actor.process_event(OpAarch64UnarySilu::new(&[1.0, 2.0, 3.0]), &mut output),
            Err(UnaryF32Error::InvalidShape)
        );
        assert_eq!(output.map(f32::to_bits), [7.0_f32.to_bits(); 2]);
    }

    #[test]
    fn unexpected_event_is_explicit() {
        let mut actor = kernel();
        let mut output = [0.0; 1];
        assert_eq!(
            actor.process_event(UnexpectedAarch64UnaryF32, &mut output),
            Err(UnaryF32Error::UnexpectedEvent)
        );
        assert!(actor.is_ready());
    }

    #[test]
    fn dispatch_is_allocation_free() {
        let mut actor = kernel();
        let input = [1.0, -2.0, 3.0, -4.0];
        let mut output = [0.0; 4];
        let allocations = measure(|| {
            assert_eq!(
                actor.process_event(OpAarch64UnarySilu::new(&input), &mut output),
                Ok(())
            );
        });
        assert_eq!(allocations.count_total, 0);
    }

    #[test]
    fn abs_neg_relu_dispatch_is_allocation_free() {
        let mut actor = kernel();
        let input = [-3.5, -1.0, 0.0, 0.5, 2.25, -4.0, 5.5, -8.0, 9.75];
        let mut output = [0.0; 9];
        let allocations = measure(|| {
            assert_eq!(
                actor.process_event(OpAarch64UnaryAbs::new(&input), &mut output),
                Ok(())
            );
            assert_eq!(
                actor.process_event(OpAarch64UnaryNeg::new(&input), &mut output),
                Ok(())
            );
            assert_eq!(
                actor.process_event(OpAarch64UnaryRelu::new(&input), &mut output),
                Ok(())
            );
        });
        assert_eq!(allocations.count_total, 0);
        assert_eq!(allocations.bytes_total, 0);
    }
}
