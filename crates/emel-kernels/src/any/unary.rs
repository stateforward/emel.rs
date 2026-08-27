//! Explicit portable F32 unary operation family.
//!
//! The pinned reference declares these operations in
//! `src/emel/kernel/events.hpp` and implements their scalar formulas in
//! `src/emel/kernel/detail.hpp:2392-2411,3271-3325`. Each operation below has
//! its own event and SML transition rows. Formula selection is compile-time;
//! dispatch actions only execute a guard-proven bounded element loop.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]
#![allow(clippy::imprecise_flops, clippy::unreadable_literal)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::tensor_view::{TensorView, TensorViewMut};

/// Errors returned by unary dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnaryError {
    /// A source or destination view failed validation.
    InvalidView,
    /// Source and destination logical shapes differ.
    ShapeMismatch,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for UnaryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid unary tensor view"),
            Self::ShapeMismatch => formatter.write_str("unary tensor shapes differ"),
            Self::UnexpectedEvent => formatter.write_str("unexpected unary event"),
            Self::Internal => formatter.write_str("internal unary dispatch error"),
        }
    }
}

impl std::error::Error for UnaryError {}

/// Result returned after an operation reaches run-to-completion.
pub type UnaryResult = Result<(), UnaryError>;

macro_rules! define_unary_event {
    ($event:ident, $runtime:ident) => {
        #[derive(Debug)]
        pub struct $event<'a> {
            input: TensorView<'a>,
            output: TensorViewMut<'a>,
        }

        impl<'a> $event<'a> {
            /// Creates an event; view and shape validation occurs in a guard.
            #[must_use]
            pub const fn new(input: TensorView<'a>, output: TensorViewMut<'a>) -> Self {
                Self { input, output }
            }
        }

        struct $runtime<'a> {
            event: $event<'a>,
            result: &'a Cell<UnaryResult>,
        }
    };
}

define_unary_event!(OpAbs, AbsRuntime);
define_unary_event!(OpSgn, SgnRuntime);
define_unary_event!(OpNeg, NegRuntime);
define_unary_event!(OpStep, StepRuntime);
define_unary_event!(OpTanh, TanhRuntime);
define_unary_event!(OpElu, EluRuntime);
define_unary_event!(OpRelu, ReluRuntime);
define_unary_event!(OpSigmoid, SigmoidRuntime);
define_unary_event!(OpGelu, GeluRuntime);
define_unary_event!(OpGeluQuick, GeluQuickRuntime);
define_unary_event!(OpSilu, SiluRuntime);
define_unary_event!(OpHardswish, HardswishRuntime);
define_unary_event!(OpHardsigmoid, HardsigmoidRuntime);
define_unary_event!(OpExp, ExpRuntime);
define_unary_event!(OpExpm1, Expm1Runtime);
define_unary_event!(OpSoftplus, SoftplusRuntime);
define_unary_event!(OpGeluErf, GeluErfRuntime);
define_unary_event!(OpFloor, FloorRuntime);
define_unary_event!(OpCeil, CeilRuntime);
define_unary_event!(OpRound, RoundRuntime);
define_unary_event!(OpTrunc, TruncRuntime);

/// Generic `op_unary` suboperation code from the pinned kernel event contract.
///
/// The portable parity lane accepts only the eight suboperations selected by
/// the pinned backend guards. The remaining wire variants stay representable
/// so their rejection remains explicit rather than silently falling back.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnarySubOp {
    /// Absolute value (wire code 0).
    Abs = 0,
    /// Sign (wire code 1; not accepted by the pinned backend lane).
    Sgn = 1,
    /// Negation (wire code 2).
    Neg = 2,
    /// Step (wire code 3; not accepted by the pinned backend lane).
    Step = 3,
    /// Hyperbolic tangent (wire code 4).
    Tanh = 4,
    /// ELU (wire code 5).
    Elu = 5,
    /// `ReLU` (wire code 6).
    Relu = 6,
    /// Sigmoid (wire code 7; not accepted by the pinned backend lane).
    Sigmoid = 7,
    /// GELU (wire code 8).
    Gelu = 8,
    /// Quick GELU (wire code 9; not accepted by the pinned backend lane).
    GeluQuick = 9,
    /// `SiLU` (wire code 10).
    Silu = 10,
    /// Hardswish (wire code 11; not accepted by the pinned backend lane).
    Hardswish = 11,
    /// Hardsigmoid (wire code 12; not accepted by the pinned backend lane).
    Hardsigmoid = 12,
    /// Exponential (wire code 13).
    Exp = 13,
    /// Exponential minus one (wire code 14; not accepted by the pinned backend lane).
    Expm1 = 14,
    /// Softplus (wire code 15; not accepted by the pinned backend lane).
    Softplus = 15,
    /// GELU-erf (wire code 16; not accepted by the pinned backend lane).
    GeluErf = 16,
    /// XIELU (wire code 17; not accepted by the pinned backend lane).
    Xielu = 17,
    /// Floor (wire code 18; not accepted by the pinned backend lane).
    Floor = 18,
    /// Ceiling (wire code 19; not accepted by the pinned backend lane).
    Ceil = 19,
    /// Round (wire code 20; not accepted by the pinned backend lane).
    Round = 20,
    /// Truncate (wire code 21; not accepted by the pinned backend lane).
    Trunc = 21,
}

/// Generic portable F32 unary request corresponding to pinned `op_unary`.
#[derive(Debug)]
pub struct OpUnary<'a> {
    input: TensorView<'a>,
    output: TensorViewMut<'a>,
    subop: UnarySubOp,
}

impl<'a> OpUnary<'a> {
    /// Creates a generic unary request. Validation occurs in an SML guard.
    #[must_use]
    pub const fn new(subop: UnarySubOp, input: TensorView<'a>, output: TensorViewMut<'a>) -> Self {
        Self {
            input,
            output,
            subop,
        }
    }

    /// Returns the requested wire suboperation.
    #[must_use]
    pub const fn subop(&self) -> UnarySubOp {
        self.subop
    }
}

struct OpUnaryRuntime<'a> {
    event: OpUnary<'a>,
    result: &'a Cell<UnaryResult>,
}

/// Explicitly reports an event outside the maintained unary family.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedUnary;

struct UnexpectedRuntime<'a> {
    result: &'a Cell<UnaryResult>,
}

#[derive(Default)]
struct Context;

sml! {
    UnaryMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Abs(AbsRuntime<'dispatch>) [guard_abs_valid] / effect_abs,
        "ready"_s <= "ready"_s + Abs(AbsRuntime<'dispatch>) [guard_abs_shape] / effect_abs_shape,
        "ready"_s <= "ready"_s + Abs(AbsRuntime<'dispatch>) [guard_abs_view] / effect_abs_view,

        "ready"_s <= "ready"_s + Sgn(SgnRuntime<'dispatch>) [guard_sgn_valid] / effect_sgn,
        "ready"_s <= "ready"_s + Sgn(SgnRuntime<'dispatch>) [guard_sgn_shape] / effect_sgn_shape,
        "ready"_s <= "ready"_s + Sgn(SgnRuntime<'dispatch>) [guard_sgn_view] / effect_sgn_view,

        "ready"_s <= "ready"_s + Neg(NegRuntime<'dispatch>) [guard_neg_valid] / effect_neg,
        "ready"_s <= "ready"_s + Neg(NegRuntime<'dispatch>) [guard_neg_shape] / effect_neg_shape,
        "ready"_s <= "ready"_s + Neg(NegRuntime<'dispatch>) [guard_neg_view] / effect_neg_view,

        "ready"_s <= "ready"_s + Step(StepRuntime<'dispatch>) [guard_step_valid] / effect_step,
        "ready"_s <= "ready"_s + Step(StepRuntime<'dispatch>) [guard_step_shape] / effect_step_shape,
        "ready"_s <= "ready"_s + Step(StepRuntime<'dispatch>) [guard_step_view] / effect_step_view,

        "ready"_s <= "ready"_s + Tanh(TanhRuntime<'dispatch>) [guard_tanh_valid] / effect_tanh,
        "ready"_s <= "ready"_s + Tanh(TanhRuntime<'dispatch>) [guard_tanh_shape] / effect_tanh_shape,
        "ready"_s <= "ready"_s + Tanh(TanhRuntime<'dispatch>) [guard_tanh_view] / effect_tanh_view,

        "ready"_s <= "ready"_s + Elu(EluRuntime<'dispatch>) [guard_elu_valid] / effect_elu,
        "ready"_s <= "ready"_s + Elu(EluRuntime<'dispatch>) [guard_elu_shape] / effect_elu_shape,
        "ready"_s <= "ready"_s + Elu(EluRuntime<'dispatch>) [guard_elu_view] / effect_elu_view,

        "ready"_s <= "ready"_s + Relu(ReluRuntime<'dispatch>) [guard_relu_valid] / effect_relu,
        "ready"_s <= "ready"_s + Relu(ReluRuntime<'dispatch>) [guard_relu_shape] / effect_relu_shape,
        "ready"_s <= "ready"_s + Relu(ReluRuntime<'dispatch>) [guard_relu_view] / effect_relu_view,

        "ready"_s <= "ready"_s + Sigmoid(SigmoidRuntime<'dispatch>) [guard_sigmoid_valid] / effect_sigmoid,
        "ready"_s <= "ready"_s + Sigmoid(SigmoidRuntime<'dispatch>) [guard_sigmoid_shape] / effect_sigmoid_shape,
        "ready"_s <= "ready"_s + Sigmoid(SigmoidRuntime<'dispatch>) [guard_sigmoid_view] / effect_sigmoid_view,

        "ready"_s <= "ready"_s + Gelu(GeluRuntime<'dispatch>) [guard_gelu_valid] / effect_gelu,
        "ready"_s <= "ready"_s + Gelu(GeluRuntime<'dispatch>) [guard_gelu_shape] / effect_gelu_shape,
        "ready"_s <= "ready"_s + Gelu(GeluRuntime<'dispatch>) [guard_gelu_view] / effect_gelu_view,

        "ready"_s <= "ready"_s + GeluQuick(GeluQuickRuntime<'dispatch>) [guard_gelu_quick_valid] / effect_gelu_quick,
        "ready"_s <= "ready"_s + GeluQuick(GeluQuickRuntime<'dispatch>) [guard_gelu_quick_shape] / effect_gelu_quick_shape,
        "ready"_s <= "ready"_s + GeluQuick(GeluQuickRuntime<'dispatch>) [guard_gelu_quick_view] / effect_gelu_quick_view,

        "ready"_s <= "ready"_s + Silu(SiluRuntime<'dispatch>) [guard_silu_valid] / effect_silu,
        "ready"_s <= "ready"_s + Silu(SiluRuntime<'dispatch>) [guard_silu_shape] / effect_silu_shape,
        "ready"_s <= "ready"_s + Silu(SiluRuntime<'dispatch>) [guard_silu_view] / effect_silu_view,

        "ready"_s <= "ready"_s + Hardswish(HardswishRuntime<'dispatch>) [guard_hardswish_valid] / effect_hardswish,
        "ready"_s <= "ready"_s + Hardswish(HardswishRuntime<'dispatch>) [guard_hardswish_shape] / effect_hardswish_shape,
        "ready"_s <= "ready"_s + Hardswish(HardswishRuntime<'dispatch>) [guard_hardswish_view] / effect_hardswish_view,

        "ready"_s <= "ready"_s + Hardsigmoid(HardsigmoidRuntime<'dispatch>) [guard_hardsigmoid_valid] / effect_hardsigmoid,
        "ready"_s <= "ready"_s + Hardsigmoid(HardsigmoidRuntime<'dispatch>) [guard_hardsigmoid_shape] / effect_hardsigmoid_shape,
        "ready"_s <= "ready"_s + Hardsigmoid(HardsigmoidRuntime<'dispatch>) [guard_hardsigmoid_view] / effect_hardsigmoid_view,

        "ready"_s <= "ready"_s + Exp(ExpRuntime<'dispatch>) [guard_exp_valid] / effect_exp,
        "ready"_s <= "ready"_s + Exp(ExpRuntime<'dispatch>) [guard_exp_shape] / effect_exp_shape,
        "ready"_s <= "ready"_s + Exp(ExpRuntime<'dispatch>) [guard_exp_view] / effect_exp_view,

        "ready"_s <= "ready"_s + Expm1(Expm1Runtime<'dispatch>) [guard_expm1_valid] / effect_expm1,
        "ready"_s <= "ready"_s + Expm1(Expm1Runtime<'dispatch>) [guard_expm1_shape] / effect_expm1_shape,
        "ready"_s <= "ready"_s + Expm1(Expm1Runtime<'dispatch>) [guard_expm1_view] / effect_expm1_view,

        "ready"_s <= "ready"_s + Softplus(SoftplusRuntime<'dispatch>) [guard_softplus_valid] / effect_softplus,
        "ready"_s <= "ready"_s + Softplus(SoftplusRuntime<'dispatch>) [guard_softplus_shape] / effect_softplus_shape,
        "ready"_s <= "ready"_s + Softplus(SoftplusRuntime<'dispatch>) [guard_softplus_view] / effect_softplus_view,

        "ready"_s <= "ready"_s + GeluErf(GeluErfRuntime<'dispatch>) [guard_gelu_erf_valid] / effect_gelu_erf,
        "ready"_s <= "ready"_s + GeluErf(GeluErfRuntime<'dispatch>) [guard_gelu_erf_shape] / effect_gelu_erf_shape,
        "ready"_s <= "ready"_s + GeluErf(GeluErfRuntime<'dispatch>) [guard_gelu_erf_view] / effect_gelu_erf_view,

        "ready"_s <= "ready"_s + Floor(FloorRuntime<'dispatch>) [guard_floor_valid] / effect_floor,
        "ready"_s <= "ready"_s + Floor(FloorRuntime<'dispatch>) [guard_floor_shape] / effect_floor_shape,
        "ready"_s <= "ready"_s + Floor(FloorRuntime<'dispatch>) [guard_floor_view] / effect_floor_view,

        "ready"_s <= "ready"_s + Ceil(CeilRuntime<'dispatch>) [guard_ceil_valid] / effect_ceil,
        "ready"_s <= "ready"_s + Ceil(CeilRuntime<'dispatch>) [guard_ceil_shape] / effect_ceil_shape,
        "ready"_s <= "ready"_s + Ceil(CeilRuntime<'dispatch>) [guard_ceil_view] / effect_ceil_view,

        "ready"_s <= "ready"_s + Round(RoundRuntime<'dispatch>) [guard_round_valid] / effect_round,
        "ready"_s <= "ready"_s + Round(RoundRuntime<'dispatch>) [guard_round_shape] / effect_round_shape,
        "ready"_s <= "ready"_s + Round(RoundRuntime<'dispatch>) [guard_round_view] / effect_round_view,

        "ready"_s <= "ready"_s + Trunc(TruncRuntime<'dispatch>) [guard_trunc_valid] / effect_trunc,
        "ready"_s <= "ready"_s + Trunc(TruncRuntime<'dispatch>) [guard_trunc_shape] / effect_trunc_shape,
        "ready"_s <= "ready"_s + Trunc(TruncRuntime<'dispatch>) [guard_trunc_view] / effect_trunc_view,

        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_abs_dense] / effect_op_unary_abs_dense,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_abs] / effect_op_unary_abs,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_neg_dense] / effect_op_unary_neg_dense,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_neg] / effect_op_unary_neg,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_tanh_dense] / effect_op_unary_tanh_dense,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_tanh] / effect_op_unary_tanh,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_elu_dense] / effect_op_unary_elu_dense,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_elu] / effect_op_unary_elu,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_relu_dense] / effect_op_unary_relu_dense,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_relu] / effect_op_unary_relu,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_gelu_dense] / effect_op_unary_gelu_dense,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_gelu] / effect_op_unary_gelu,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_silu_dense] / effect_op_unary_silu_dense,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_silu] / effect_op_unary_silu,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_exp_dense] / effect_op_unary_exp_dense,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_exp] / effect_op_unary_exp,
        "ready"_s <= "ready"_s + OpUnary(OpUnaryRuntime<'dispatch>) [guard_op_unary_invalid] / effect_op_unary_invalid,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>) / effect_unexpected_event,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion actor for the unary family.
pub struct UnaryKernel {
    machine: UnaryMachineStateMachine<Context>,
}

impl fmt::Debug for UnaryKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnaryKernel")
            .finish_non_exhaustive()
    }
}

impl Default for UnaryKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl UnaryKernel {
    /// Constructs an independent unary actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: UnaryMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: UnaryEvent>(&mut self, event: E) -> E::Output {
        let output = event.dispatch(self);
        let _ = self.machine.is(&UnaryMachineStates::Ready);
        output
    }
}

pub trait UnaryEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut UnaryKernel) -> Self::Output;
}

macro_rules! impl_unary_event {
    ($event:ident, $method:ident) => {
        impl UnaryEvent for $event<'_> {
            type Output = UnaryResult;

            fn dispatch(self, actor: &mut UnaryKernel) -> Self::Output {
                actor.$method(self)
            }
        }
    };
}

impl_unary_event!(OpAbs, abs);
impl_unary_event!(OpSgn, sgn);
impl_unary_event!(OpNeg, neg);
impl_unary_event!(OpStep, step);
impl_unary_event!(OpTanh, tanh);
impl_unary_event!(OpElu, elu);
impl_unary_event!(OpRelu, relu);
impl_unary_event!(OpSigmoid, sigmoid);
impl_unary_event!(OpGelu, gelu);
impl_unary_event!(OpGeluQuick, gelu_quick);
impl_unary_event!(OpSilu, silu);
impl_unary_event!(OpHardswish, hardswish);
impl_unary_event!(OpHardsigmoid, hardsigmoid);
impl_unary_event!(OpExp, exp);
impl_unary_event!(OpExpm1, expm1);
impl_unary_event!(OpSoftplus, softplus);
impl_unary_event!(OpGeluErf, gelu_erf);
impl_unary_event!(OpFloor, floor);
impl_unary_event!(OpCeil, ceil);
impl_unary_event!(OpRound, round);
impl_unary_event!(OpTrunc, trunc);

impl UnaryEvent for OpUnary<'_> {
    type Output = UnaryResult;

    fn dispatch(self, actor: &mut UnaryKernel) -> Self::Output {
        actor.op_unary(self)
    }
}

impl UnaryEvent for UnexpectedUnary {
    type Output = UnaryResult;

    fn dispatch(self, actor: &mut UnaryKernel) -> Self::Output {
        actor.unexpected(self)
    }
}

macro_rules! unary_method {
    ($method:ident, $event:ident, $runtime:ident, $machine_event:ident) => {
        impl UnaryKernel {
            fn $method(&mut self, event: $event<'_>) -> UnaryResult {
                let result = Cell::new(Err(UnaryError::UnexpectedEvent));
                self.machine
                    .process_event(UnaryMachineEvents::$machine_event($runtime {
                        event,
                        result: &result,
                    }))
                    .map_err(|_| UnaryError::Internal)?;
                result.get()
            }
        }
    };
}

unary_method!(abs, OpAbs, AbsRuntime, Abs);
unary_method!(sgn, OpSgn, SgnRuntime, Sgn);
unary_method!(neg, OpNeg, NegRuntime, Neg);
unary_method!(step, OpStep, StepRuntime, Step);
unary_method!(tanh, OpTanh, TanhRuntime, Tanh);
unary_method!(elu, OpElu, EluRuntime, Elu);
unary_method!(relu, OpRelu, ReluRuntime, Relu);
unary_method!(sigmoid, OpSigmoid, SigmoidRuntime, Sigmoid);
unary_method!(gelu, OpGelu, GeluRuntime, Gelu);
unary_method!(gelu_quick, OpGeluQuick, GeluQuickRuntime, GeluQuick);
unary_method!(silu, OpSilu, SiluRuntime, Silu);
unary_method!(hardswish, OpHardswish, HardswishRuntime, Hardswish);
unary_method!(hardsigmoid, OpHardsigmoid, HardsigmoidRuntime, Hardsigmoid);
unary_method!(exp, OpExp, ExpRuntime, Exp);
unary_method!(expm1, OpExpm1, Expm1Runtime, Expm1);
unary_method!(softplus, OpSoftplus, SoftplusRuntime, Softplus);
unary_method!(gelu_erf, OpGeluErf, GeluErfRuntime, GeluErf);
unary_method!(floor, OpFloor, FloorRuntime, Floor);
unary_method!(ceil, OpCeil, CeilRuntime, Ceil);
unary_method!(round, OpRound, RoundRuntime, Round);
unary_method!(trunc, OpTrunc, TruncRuntime, Trunc);

impl UnaryKernel {
    fn op_unary(&mut self, event: OpUnary<'_>) -> UnaryResult {
        let result = Cell::new(Err(UnaryError::UnexpectedEvent));
        self.machine
            .process_event(UnaryMachineEvents::OpUnary(OpUnaryRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| UnaryError::Internal)?;
        result.get()
    }

    fn unexpected(&mut self, _event: UnexpectedUnary) -> UnaryResult {
        let result = Cell::new(Err(UnaryError::Internal));
        self.machine
            .process_event(UnaryMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| UnaryError::Internal)?;
        result.get()
    }
}

fn views_valid(input: &TensorView<'_>, output: &TensorViewMut<'_>) -> bool {
    input.validate().is_ok() && output.validate().is_ok()
}

fn shape_valid(input: &TensorView<'_>, output: &TensorViewMut<'_>) -> bool {
    input.layout().element_count() == output.layout().element_count()
}

fn dense_views_valid(input: &TensorView<'_>, output: &TensorViewMut<'_>) -> bool {
    views_valid(input, output)
        && shape_valid(input, output)
        && input.is_dense_f32()
        && output.is_dense_f32()
}

trait UnaryFormula {
    fn apply(value: f32) -> f32;
}

struct AbsFormula;
impl UnaryFormula for AbsFormula {
    fn apply(value: f32) -> f32 {
        value.abs()
    }
}
struct SgnFormula;
impl UnaryFormula for SgnFormula {
    fn apply(value: f32) -> f32 {
        value.signum()
    }
}
struct NegFormula;
impl UnaryFormula for NegFormula {
    fn apply(value: f32) -> f32 {
        -value
    }
}
struct StepFormula;
impl UnaryFormula for StepFormula {
    fn apply(value: f32) -> f32 {
        if value > 0.0 { 1.0 } else { 0.0 }
    }
}
struct TanhFormula;
impl UnaryFormula for TanhFormula {
    fn apply(value: f32) -> f32 {
        value.tanh()
    }
}
struct EluFormula;
impl UnaryFormula for EluFormula {
    fn apply(value: f32) -> f32 {
        if value >= 0.0 { value } else { value.exp_m1() }
    }
}
struct ReluFormula;
impl UnaryFormula for ReluFormula {
    fn apply(value: f32) -> f32 {
        value.max(0.0)
    }
}
struct SigmoidFormula;
impl UnaryFormula for SigmoidFormula {
    fn apply(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }
}
struct GeluFormula;
impl UnaryFormula for GeluFormula {
    // Preserve the pinned approximation's separate products and rounding.
    #[allow(clippy::suboptimal_flops)]
    fn apply(value: f32) -> f32 {
        if value <= -10.0 {
            return 0.0;
        }
        if value >= 10.0 {
            return value;
        }
        let quantized = fp16_to_f32(fp32_to_fp16(value));
        let approximation = 0.5
            * quantized
            * (1.0
                + (0.797_884_6 * (quantized + 0.044_715 * quantized * quantized * quantized))
                    .tanh());
        fp16_to_f32(fp32_to_fp16(approximation))
    }
}

/// Converts F32 to binary16 with the same round-to-nearest-even arithmetic as
/// the pinned reference helper. This is a scalar numeric detail, not routing.
#[allow(clippy::cast_possible_truncation)]
fn fp32_to_fp16(value: f32) -> u16 {
    let scale_to_inf = f32::from_bits(0x7780_0000);
    // `0x1p-110f` from the pinned reference (not the fp16 decode scale
    // `0x1p-112f` used below).
    let scale_to_zero = f32::from_bits(0x0880_0000);
    let mut base = (value.abs() * scale_to_inf) * scale_to_zero;
    let word = value.to_bits();
    let doubled = word.wrapping_add(word);
    let sign = word & 0x8000_0000;
    let mut bias = doubled & 0xff00_0000;
    if bias < 0x7100_0000 {
        bias = 0x7100_0000;
    }
    base += f32::from_bits((bias >> 1).wrapping_add(0x0780_0000));
    let base_bits = base.to_bits();
    let exponent = (base_bits >> 13) & 0x0000_7c00;
    let mantissa = base_bits & 0x0000_0fff;
    let nonsign = exponent + mantissa;
    ((sign >> 16)
        | if doubled > 0xff00_0000 {
            0x7e00
        } else {
            nonsign
        }) as u16
}

fn fp16_to_f32(bits16: u16) -> f32 {
    let word = u32::from(bits16) << 16;
    let sign = word & 0x8000_0000;
    let doubled = word.wrapping_add(word);
    let exp_offset = 0xe0_u32 << 23;
    let normalized =
        f32::from_bits((doubled >> 4).wrapping_add(exp_offset)) * f32::from_bits(0x0780_0000);
    let denormalized = f32::from_bits((doubled >> 17) | (126_u32 << 23)) - 0.5;
    let result = if doubled < 1_u32 << 27 {
        sign | denormalized.to_bits()
    } else {
        sign | normalized.to_bits()
    };
    f32::from_bits(result)
}
struct GeluQuickFormula;
impl UnaryFormula for GeluQuickFormula {
    // Preserve the pinned approximation's separate products and rounding.
    #[allow(clippy::suboptimal_flops)]
    fn apply(value: f32) -> f32 {
        0.5 * value * (1.0 + (0.797_884_6 * value * (1.0 + 0.044_715 * value * value)).tanh())
    }
}
struct SiluFormula;
impl UnaryFormula for SiluFormula {
    fn apply(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }
}
struct HardswishFormula;
impl UnaryFormula for HardswishFormula {
    fn apply(value: f32) -> f32 {
        value * (value + 3.0).clamp(0.0, 6.0) / 6.0
    }
}
struct HardsigmoidFormula;
impl UnaryFormula for HardsigmoidFormula {
    fn apply(value: f32) -> f32 {
        (value + 3.0).clamp(0.0, 6.0) / 6.0
    }
}
struct ExpFormula;
impl UnaryFormula for ExpFormula {
    fn apply(value: f32) -> f32 {
        value.exp()
    }
}
struct Expm1Formula;
impl UnaryFormula for Expm1Formula {
    fn apply(value: f32) -> f32 {
        value.exp_m1()
    }
}
struct SoftplusFormula;
impl UnaryFormula for SoftplusFormula {
    fn apply(value: f32) -> f32 {
        value.exp().ln_1p()
    }
}
struct GeluErfFormula;
impl UnaryFormula for GeluErfFormula {
    // Preserve the pinned polynomial's separate products and rounding.
    #[allow(clippy::suboptimal_flops)]
    fn apply(value: f32) -> f32 {
        // Abramowitz-Stegun erf approximation; the branch is numerical sign
        // handling within a selected formula, not runtime operation routing.
        let sign = if value < 0.0 { -1.0 } else { 1.0 };
        let x = value.abs() / core::f32::consts::SQRT_2;
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let polynomial =
            (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
                + 0.254_829_6)
                * t;
        let erf = sign * (1.0 - polynomial * (-x * x).exp());
        0.5 * value * (1.0 + erf)
    }
}
struct FloorFormula;
impl UnaryFormula for FloorFormula {
    fn apply(value: f32) -> f32 {
        value.floor()
    }
}
struct CeilFormula;
impl UnaryFormula for CeilFormula {
    fn apply(value: f32) -> f32 {
        value.ceil()
    }
}
struct RoundFormula;
impl UnaryFormula for RoundFormula {
    fn apply(value: f32) -> f32 {
        value.round()
    }
}
struct TruncFormula;
impl UnaryFormula for TruncFormula {
    fn apply(value: f32) -> f32 {
        value.trunc()
    }
}

fn execute<F: UnaryFormula>(
    input: &TensorView<'_>,
    output: &mut TensorViewMut<'_>,
    result: &Cell<UnaryResult>,
) {
    let mut ordinal = 0;
    while ordinal < input.count() {
        output.write(ordinal, F::apply(input.read(ordinal)));
        ordinal += 1;
    }
    result.set(Ok(()));
}

fn execute_dense<F: UnaryFormula>(
    input: &TensorView<'_>,
    output: &mut TensorViewMut<'_>,
    result: &Cell<UnaryResult>,
) {
    let input = input.as_slice();
    let output = output.as_mut_slice();
    let mut ordinal = 0;
    while ordinal < input.len() {
        output[ordinal] = F::apply(input[ordinal]);
        ordinal += 1;
    }
    result.set(Ok(()));
}

macro_rules! impl_unary_context {
    ($valid:ident, $shape:ident, $view:ident, $effect:ident, $effect_shape:ident, $effect_view:ident, $runtime:ident, $formula:ty) => {
        fn $valid(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(views_valid(&event.event.input, &event.event.output)
                && shape_valid(&event.event.input, &event.event.output))
        }
        fn $shape(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(views_valid(&event.event.input, &event.event.output)
                && !shape_valid(&event.event.input, &event.event.output))
        }
        fn $view(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(!views_valid(&event.event.input, &event.event.output))
        }
        fn $effect(&mut self, mut event: $runtime<'_>) -> Result<(), ()> {
            execute::<$formula>(&event.event.input, &mut event.event.output, event.result);
            Ok(())
        }
        fn $effect_shape(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event.result.set(Err(UnaryError::ShapeMismatch));
            Ok(())
        }
        fn $effect_view(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event.result.set(Err(UnaryError::InvalidView));
            Ok(())
        }
    };
}

impl UnaryMachineStateMachineContext for Context {
    impl_unary_context!(
        guard_abs_valid,
        guard_abs_shape,
        guard_abs_view,
        effect_abs,
        effect_abs_shape,
        effect_abs_view,
        AbsRuntime,
        AbsFormula
    );
    impl_unary_context!(
        guard_sgn_valid,
        guard_sgn_shape,
        guard_sgn_view,
        effect_sgn,
        effect_sgn_shape,
        effect_sgn_view,
        SgnRuntime,
        SgnFormula
    );
    impl_unary_context!(
        guard_neg_valid,
        guard_neg_shape,
        guard_neg_view,
        effect_neg,
        effect_neg_shape,
        effect_neg_view,
        NegRuntime,
        NegFormula
    );
    impl_unary_context!(
        guard_step_valid,
        guard_step_shape,
        guard_step_view,
        effect_step,
        effect_step_shape,
        effect_step_view,
        StepRuntime,
        StepFormula
    );
    impl_unary_context!(
        guard_tanh_valid,
        guard_tanh_shape,
        guard_tanh_view,
        effect_tanh,
        effect_tanh_shape,
        effect_tanh_view,
        TanhRuntime,
        TanhFormula
    );
    impl_unary_context!(
        guard_elu_valid,
        guard_elu_shape,
        guard_elu_view,
        effect_elu,
        effect_elu_shape,
        effect_elu_view,
        EluRuntime,
        EluFormula
    );
    impl_unary_context!(
        guard_relu_valid,
        guard_relu_shape,
        guard_relu_view,
        effect_relu,
        effect_relu_shape,
        effect_relu_view,
        ReluRuntime,
        ReluFormula
    );
    impl_unary_context!(
        guard_sigmoid_valid,
        guard_sigmoid_shape,
        guard_sigmoid_view,
        effect_sigmoid,
        effect_sigmoid_shape,
        effect_sigmoid_view,
        SigmoidRuntime,
        SigmoidFormula
    );
    impl_unary_context!(
        guard_gelu_valid,
        guard_gelu_shape,
        guard_gelu_view,
        effect_gelu,
        effect_gelu_shape,
        effect_gelu_view,
        GeluRuntime,
        GeluFormula
    );
    impl_unary_context!(
        guard_gelu_quick_valid,
        guard_gelu_quick_shape,
        guard_gelu_quick_view,
        effect_gelu_quick,
        effect_gelu_quick_shape,
        effect_gelu_quick_view,
        GeluQuickRuntime,
        GeluQuickFormula
    );
    impl_unary_context!(
        guard_silu_valid,
        guard_silu_shape,
        guard_silu_view,
        effect_silu,
        effect_silu_shape,
        effect_silu_view,
        SiluRuntime,
        SiluFormula
    );
    impl_unary_context!(
        guard_hardswish_valid,
        guard_hardswish_shape,
        guard_hardswish_view,
        effect_hardswish,
        effect_hardswish_shape,
        effect_hardswish_view,
        HardswishRuntime,
        HardswishFormula
    );
    impl_unary_context!(
        guard_hardsigmoid_valid,
        guard_hardsigmoid_shape,
        guard_hardsigmoid_view,
        effect_hardsigmoid,
        effect_hardsigmoid_shape,
        effect_hardsigmoid_view,
        HardsigmoidRuntime,
        HardsigmoidFormula
    );
    impl_unary_context!(
        guard_exp_valid,
        guard_exp_shape,
        guard_exp_view,
        effect_exp,
        effect_exp_shape,
        effect_exp_view,
        ExpRuntime,
        ExpFormula
    );
    impl_unary_context!(
        guard_expm1_valid,
        guard_expm1_shape,
        guard_expm1_view,
        effect_expm1,
        effect_expm1_shape,
        effect_expm1_view,
        Expm1Runtime,
        Expm1Formula
    );
    impl_unary_context!(
        guard_softplus_valid,
        guard_softplus_shape,
        guard_softplus_view,
        effect_softplus,
        effect_softplus_shape,
        effect_softplus_view,
        SoftplusRuntime,
        SoftplusFormula
    );
    impl_unary_context!(
        guard_gelu_erf_valid,
        guard_gelu_erf_shape,
        guard_gelu_erf_view,
        effect_gelu_erf,
        effect_gelu_erf_shape,
        effect_gelu_erf_view,
        GeluErfRuntime,
        GeluErfFormula
    );
    impl_unary_context!(
        guard_floor_valid,
        guard_floor_shape,
        guard_floor_view,
        effect_floor,
        effect_floor_shape,
        effect_floor_view,
        FloorRuntime,
        FloorFormula
    );
    impl_unary_context!(
        guard_ceil_valid,
        guard_ceil_shape,
        guard_ceil_view,
        effect_ceil,
        effect_ceil_shape,
        effect_ceil_view,
        CeilRuntime,
        CeilFormula
    );
    impl_unary_context!(
        guard_round_valid,
        guard_round_shape,
        guard_round_view,
        effect_round,
        effect_round_shape,
        effect_round_view,
        RoundRuntime,
        RoundFormula
    );
    impl_unary_context!(
        guard_trunc_valid,
        guard_trunc_shape,
        guard_trunc_view,
        effect_trunc,
        effect_trunc_shape,
        effect_trunc_view,
        TruncRuntime,
        TruncFormula
    );

    fn guard_op_unary_abs(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Abs
            && views_valid(&event.event.input, &event.event.output)
            && shape_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_abs_dense(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Abs
            && dense_views_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_neg(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Neg
            && views_valid(&event.event.input, &event.event.output)
            && shape_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_neg_dense(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Neg
            && dense_views_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_tanh(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Tanh
            && views_valid(&event.event.input, &event.event.output)
            && shape_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_tanh_dense(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Tanh
            && dense_views_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_elu(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Elu
            && views_valid(&event.event.input, &event.event.output)
            && shape_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_elu_dense(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Elu
            && dense_views_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_relu(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Relu
            && views_valid(&event.event.input, &event.event.output)
            && shape_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_relu_dense(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Relu
            && dense_views_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_gelu(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Gelu
            && views_valid(&event.event.input, &event.event.output)
            && shape_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_gelu_dense(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Gelu
            && dense_views_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_silu(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Silu
            && views_valid(&event.event.input, &event.event.output)
            && shape_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_silu_dense(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Silu
            && dense_views_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_exp(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Exp
            && views_valid(&event.event.input, &event.event.output)
            && shape_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_exp_dense(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.subop == UnarySubOp::Exp
            && dense_views_valid(&event.event.input, &event.event.output))
    }
    fn guard_op_unary_invalid(&self, event: &OpUnaryRuntime<'_>) -> Result<bool, ()> {
        Ok(!matches!(
            event.event.subop,
            UnarySubOp::Abs
                | UnarySubOp::Neg
                | UnarySubOp::Tanh
                | UnarySubOp::Elu
                | UnarySubOp::Relu
                | UnarySubOp::Gelu
                | UnarySubOp::Silu
                | UnarySubOp::Exp
        ) || !views_valid(&event.event.input, &event.event.output)
            || !shape_valid(&event.event.input, &event.event.output))
    }

    fn effect_op_unary_abs(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute::<AbsFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_abs_dense(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute_dense::<AbsFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_neg(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute::<NegFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_neg_dense(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute_dense::<NegFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_tanh(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute::<TanhFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_tanh_dense(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute_dense::<TanhFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_elu(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute::<EluFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_elu_dense(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute_dense::<EluFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_relu(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute::<ReluFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_relu_dense(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute_dense::<ReluFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_gelu(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute::<GeluFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_gelu_dense(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute_dense::<GeluFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_silu(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute::<SiluFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_silu_dense(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute_dense::<SiluFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_exp(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute::<ExpFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_exp_dense(&mut self, mut event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        execute_dense::<ExpFormula>(&event.event.input, &mut event.event.output, event.result);
        Ok(())
    }
    fn effect_op_unary_invalid(&mut self, event: OpUnaryRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(UnaryError::UnexpectedEvent));
        Ok(())
    }

    fn effect_unexpected_event(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(UnaryError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}
