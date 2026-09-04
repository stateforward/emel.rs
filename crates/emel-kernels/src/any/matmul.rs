//! Safe F32 matrix multiplication with explicit SML dispatch.
//!
//! The pinned `emel.cpp` source is commit
//! `843a117386ef17dc5a50549bbfc821074c2141d6`. Its generic
//! `detail.hpp:3357-3570` execution uses source layout `[k,m]`, right-hand
//! layout `[n,k]`, and destination layout `[n,m]`. The common shape contract
//! is at `detail.hpp:3844-3875`, and the backend transition is
//! `x86_64/sm.hpp:416-423`.
//!
//! This module ports the safe F32 path plus the maintained native packed
//! `Q4_0`, `Q4_1`, `Q5_0`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, and `Q6_K` regular routes and the pinned native packed
//! argmax families. Guards validate the complete view and exact rank/layout
//! contract before one bounded data-plane action. Each packed family has an
//! explicit SML route and uses actor-owned reusable Q8 scratch; unsupported
//! reference families remain explicit residuals rather than being silently
//! substituted.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::tensor_view::{TensorView, TensorViewMut};

pub use super::tensor_view::PackedRowView as QuantizedView;

const Q4_0_CODE: u8 = 2;
const Q4_1_CODE: u8 = 3;
const Q5_0_CODE: u8 = 6;
const Q8_0_CODE: u8 = 8;
const Q2_K_CODE: u8 = 10;
const Q3_K_CODE: u8 = 11;
const Q4_K_CODE: u8 = 12;
const Q6_K_CODE: u8 = 14;
const QK_32: usize = 32;
const QK_K: usize = 256;
const Q4_0_BYTES: usize = 18;
const Q4_1_BYTES: usize = 20;
const Q5_0_BYTES: usize = 22;
const Q8_0_BYTES: usize = 34;
const Q2_K_BYTES: usize = 84;
const Q3_K_BYTES: usize = 110;
const Q4_K_BYTES: usize = 144;
const Q6_K_BYTES: usize = 210;
const MAX_Q8_0_BLOCKS: usize = 1024;
const MAX_Q8_K_BLOCKS: usize = 128;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Q8_0Scratch {
    pub(crate) d: u16,
    pub(crate) qs: [i8; QK_32],
}

impl Q8_0Scratch {
    pub(crate) const ZERO: Self = Self {
        d: 0,
        qs: [0; QK_32],
    };
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Q8KScratch {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK_K],
    pub(crate) bsums: [i16; QK_K / 16],
}

impl Q8KScratch {
    pub(crate) const ZERO: Self = Self {
        d: 0.0,
        qs: [0; QK_K],
        bsums: [0; QK_K / 16],
    };
}

/// Errors returned by F32 matrix multiplication dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatmulError {
    /// One or more operands failed dtype, shape, stride, or bounds validation.
    InvalidView,
    /// Operands do not satisfy the pinned matrix dimensions or rank contract.
    ShapeMismatch,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for MatmulError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid matrix tensor view"),
            Self::ShapeMismatch => formatter.write_str("matrix tensor shapes differ"),
            Self::UnexpectedEvent => formatter.write_str("unexpected matrix event"),
            Self::Internal => formatter.write_str("internal matrix dispatch error"),
        }
    }
}

impl std::error::Error for MatmulError {}

/// F32 matrix multiplication operands using the pinned ggml orientation.
#[derive(Debug)]
pub struct OpMulMat<'a> {
    /// Left operand with logical shape `[k,m,1,1]`.
    lhs: TensorView<'a>,
    /// Right operand with logical shape `[n,k,1,1]`.
    rhs: TensorView<'a>,
    /// Destination with logical shape `[n,m,1,1]`.
    destination: TensorViewMut<'a>,
}

impl<'a> OpMulMat<'a> {
    /// Creates an event. View and shape validation occurs in machine guards.
    #[must_use]
    pub const fn new(
        lhs: TensorView<'a>,
        rhs: TensorView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            lhs,
            rhs,
            destination,
        }
    }

    /// Reports whether all three operands are dense contiguous F32 views.
    pub(crate) fn target_dense_views_valid(&self) -> bool {
        self.lhs.validate().is_ok()
            && self.rhs.validate().is_ok()
            && self.destination.validate().is_ok()
            && self.lhs.is_dense_f32()
            && self.rhs.is_dense_f32()
            && self.destination.is_dense_f32()
    }

    /// Reports whether the complete view contract is valid, independent of
    /// the dense-layout requirement of a target vector or matrix action.
    pub(crate) fn target_views_valid(&self) -> bool {
        views_valid(self)
    }

    /// Returns the logical number of right-hand-side columns.
    pub(crate) const fn target_rhs_columns(&self) -> u64 {
        self.rhs.layout().ne()[0]
    }

    /// Reports whether dimensions match the pinned dense matmul contract.
    pub(crate) const fn target_shape_valid(&self) -> bool {
        shape_valid(self)
    }

    /// Returns the operands for a same-RTC target child dispatch.
    pub(crate) const fn into_parts(self) -> (TensorView<'a>, TensorView<'a>, TensorViewMut<'a>) {
        (self.lhs, self.rhs, self.destination)
    }
}

/// Explicitly reports an event outside the maintained F32 matrix API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedMatmul;

/// F32 matrix-vector argmax with the pinned `[k,m] · [k]` orientation.
#[derive(Debug)]
pub struct OpMulMatArgmax<'a> {
    /// Matrix operand with logical shape `[k,m,1,1]`.
    lhs: TensorView<'a>,
    /// Vector operand with logical shape `[k,1,1,1]`.
    rhs: TensorView<'a>,
    /// Scalar destination receiving the best score.
    destination: TensorViewMut<'a>,
}

impl<'a> OpMulMatArgmax<'a> {
    /// Creates an argmax event. Validation occurs in machine guards.
    #[must_use]
    pub const fn new(
        lhs: TensorView<'a>,
        rhs: TensorView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            lhs,
            rhs,
            destination,
        }
    }
}

/// `Q4_0` packed matrix multiplied by a dense F32 matrix.
///
/// This is the pinned native packed `run_mul_mat` path: the left operand is
/// stored in `Q4_0` blocks while the right operand remains dense F32 and is
/// quantized into actor-owned `Q8_0` scratch one output column at a time.
#[derive(Debug)]
pub struct OpMulMatQ4_0<'a> {
    lhs: QuantizedView<'a>,
    rhs: TensorView<'a>,
    destination: TensorViewMut<'a>,
}

impl<'a> OpMulMatQ4_0<'a> {
    /// Creates a packed matmul event. Validation occurs in machine guards.
    #[must_use]
    pub const fn new(
        lhs: QuantizedView<'a>,
        rhs: TensorView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            lhs,
            rhs,
            destination,
        }
    }

    /// Returns the validated operands to a target-owned packed action.
    pub(crate) const fn into_parts(self) -> (QuantizedView<'a>, TensorView<'a>, TensorViewMut<'a>) {
        (self.lhs, self.rhs, self.destination)
    }
}

/// `Q4_1` packed matrix multiplied by a dense F32 matrix.
///
/// This follows the pinned native packed `run_mul_mat` branch at
/// `detail.hpp:3405-3432`: the left operand is stored in `Q4_1` blocks, the
/// right operand is quantized into actor-owned `Q8_0` scratch per output
/// column, and the destination remains dense F32.
#[derive(Debug)]
pub struct OpMulMatQ4_1<'a> {
    lhs: QuantizedView<'a>,
    rhs: TensorView<'a>,
    destination: TensorViewMut<'a>,
}

impl<'a> OpMulMatQ4_1<'a> {
    /// Creates a packed matmul event. Validation occurs in machine guards.
    #[must_use]
    pub const fn new(
        lhs: QuantizedView<'a>,
        rhs: TensorView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            lhs,
            rhs,
            destination,
        }
    }

    /// Returns the validated operands to a target-owned packed action.
    pub(crate) const fn into_parts(self) -> (QuantizedView<'a>, TensorView<'a>, TensorViewMut<'a>) {
        (self.lhs, self.rhs, self.destination)
    }
}

/// `Q5_0` packed matrix multiplied by a dense F32 matrix.
///
/// This follows the pinned native packed `run_mul_mat` branch at
/// `detail.hpp:3436-3464`: the left operand is stored in `Q5_0` blocks, the
/// right operand is quantized into actor-owned `Q8_0` scratch per output
/// column, and the destination remains dense F32.
#[derive(Debug)]
pub struct OpMulMatQ5_0<'a> {
    lhs: QuantizedView<'a>,
    rhs: TensorView<'a>,
    destination: TensorViewMut<'a>,
}

/// `Q8_0` packed matrix multiplied by a dense F32 matrix.
///
/// This follows the pinned native packed `run_mul_mat` branch at
/// `detail.hpp:3466-3494`: both the left-hand blocks and the reusable
/// right-hand quantization use the `Q8_0` format; the destination remains
/// dense F32.
#[derive(Debug)]
pub struct OpMulMatQ8_0<'a> {
    lhs: QuantizedView<'a>,
    rhs: TensorView<'a>,
    destination: TensorViewMut<'a>,
}

impl<'a> OpMulMatQ8_0<'a> {
    /// Creates a packed matmul event. Validation occurs in machine guards.
    #[must_use]
    pub const fn new(
        lhs: QuantizedView<'a>,
        rhs: TensorView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            lhs,
            rhs,
            destination,
        }
    }

    /// Returns the validated operands to a target-owned packed action.
    pub(crate) const fn into_parts(self) -> (QuantizedView<'a>, TensorView<'a>, TensorViewMut<'a>) {
        (self.lhs, self.rhs, self.destination)
    }
}

/// `Q2_K` packed matrix multiplied by a dense F32 matrix.
///
/// This follows the pinned native packed `run_mul_mat` `Q2_K` branch: the left
/// operand remains packed, while each dense RHS column is quantized into the
/// actor-owned `Q8_K` scratch before the packed dot product.
#[derive(Debug)]
pub struct OpMulMatQ2K<'a> {
    lhs: QuantizedView<'a>,
    rhs: TensorView<'a>,
    destination: TensorViewMut<'a>,
}

impl<'a> OpMulMatQ2K<'a> {
    /// Creates a packed matmul event. Validation occurs in machine guards.
    #[must_use]
    pub const fn new(
        lhs: QuantizedView<'a>,
        rhs: TensorView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            lhs,
            rhs,
            destination,
        }
    }

    /// Returns the validated operands to a target-owned packed action.
    pub(crate) const fn into_parts(self) -> (QuantizedView<'a>, TensorView<'a>, TensorViewMut<'a>) {
        (self.lhs, self.rhs, self.destination)
    }
}

/// `Q3_K` packed matrix multiplied by a dense F32 matrix.
///
/// This follows the pinned native packed `run_mul_mat` `Q3_K` branch: the left
/// operand remains packed, while each dense RHS column is quantized into the
/// actor-owned `Q8_K` scratch before the packed dot product.
#[derive(Debug)]
pub struct OpMulMatQ3K<'a> {
    lhs: QuantizedView<'a>,
    rhs: TensorView<'a>,
    destination: TensorViewMut<'a>,
}

/// `Q4_K` packed matrix multiplied by a dense F32 matrix.
///
/// This follows the pinned native packed `run_mul_mat` `Q4_K` branch at
/// `detail.hpp:3517-3528`: the left operand remains packed, while each dense
/// RHS column is quantized into actor-owned `Q8_K` scratch before the packed
/// dot product.
#[derive(Debug)]
pub struct OpMulMatQ4K<'a> {
    lhs: QuantizedView<'a>,
    rhs: TensorView<'a>,
    destination: TensorViewMut<'a>,
}

/// `Q6_K` packed matrix multiplied by a dense F32 matrix.
///
/// This follows the pinned native packed `run_mul_mat` `Q6_K` branch at
/// `detail.hpp:3496-3537`: the left operand remains packed, while each dense
/// RHS column is quantized into actor-owned `Q8_K` scratch before the packed
/// scalar dot product.
#[derive(Debug)]
pub struct OpMulMatQ6K<'a> {
    lhs: QuantizedView<'a>,
    rhs: TensorView<'a>,
    destination: TensorViewMut<'a>,
}

impl<'a> OpMulMatQ4K<'a> {
    /// Creates a packed matmul event. Validation occurs in machine guards.
    #[must_use]
    pub const fn new(
        lhs: QuantizedView<'a>,
        rhs: TensorView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            lhs,
            rhs,
            destination,
        }
    }

    /// Returns the validated operands to a target-owned packed action.
    pub(crate) const fn into_parts(self) -> (QuantizedView<'a>, TensorView<'a>, TensorViewMut<'a>) {
        (self.lhs, self.rhs, self.destination)
    }
}

impl<'a> OpMulMatQ6K<'a> {
    /// Creates a packed matmul event. Validation occurs in machine guards.
    #[must_use]
    pub const fn new(
        lhs: QuantizedView<'a>,
        rhs: TensorView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            lhs,
            rhs,
            destination,
        }
    }

    /// Returns the validated operands to a target-owned packed action.
    pub(crate) const fn into_parts(self) -> (QuantizedView<'a>, TensorView<'a>, TensorViewMut<'a>) {
        (self.lhs, self.rhs, self.destination)
    }
}

impl<'a> OpMulMatQ3K<'a> {
    /// Creates a packed matmul event. Validation occurs in machine guards.
    #[must_use]
    pub const fn new(
        lhs: QuantizedView<'a>,
        rhs: TensorView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            lhs,
            rhs,
            destination,
        }
    }

    /// Returns the validated operands to a target-owned packed action.
    pub(crate) const fn into_parts(self) -> (QuantizedView<'a>, TensorView<'a>, TensorViewMut<'a>) {
        (self.lhs, self.rhs, self.destination)
    }
}

impl<'a> OpMulMatQ5_0<'a> {
    /// Creates a packed matmul event. Validation occurs in machine guards.
    #[must_use]
    pub const fn new(
        lhs: QuantizedView<'a>,
        rhs: TensorView<'a>,
        destination: TensorViewMut<'a>,
    ) -> Self {
        Self {
            lhs,
            rhs,
            destination,
        }
    }

    /// Returns the validated operands to a target-owned packed action.
    pub(crate) const fn into_parts(self) -> (QuantizedView<'a>, TensorView<'a>, TensorViewMut<'a>) {
        (self.lhs, self.rhs, self.destination)
    }
}

macro_rules! define_quantized_argmax_event {
    ($name:ident, $code:expr, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub struct $name<'a> {
            lhs: QuantizedView<'a>,
            rhs: TensorView<'a>,
            destination: TensorViewMut<'a>,
        }

        impl<'a> $name<'a> {
            /// Creates a packed argmax event. Validation occurs in actor guards.
            #[must_use]
            pub const fn new(
                lhs: QuantizedView<'a>,
                rhs: TensorView<'a>,
                destination: TensorViewMut<'a>,
            ) -> Self {
                Self {
                    lhs,
                    rhs,
                    destination,
                }
            }

            /// Returns the validated operands to a target-owned packed action.
            pub(crate) const fn into_parts(
                self,
            ) -> (QuantizedView<'a>, TensorView<'a>, TensorViewMut<'a>) {
                (self.lhs, self.rhs, self.destination)
            }
        }

        const _: u8 = $code;
    };
}

define_quantized_argmax_event!(
    OpMulMatArgmaxQ4_0,
    Q4_0_CODE,
    "`Q4_0` packed matrix-vector argmax using a reusable `Q8_0` RHS."
);
define_quantized_argmax_event!(
    OpMulMatArgmaxQ4_1,
    Q4_1_CODE,
    "`Q4_1` packed matrix-vector argmax using a reusable `Q8_0` RHS."
);
define_quantized_argmax_event!(
    OpMulMatArgmaxQ5_0,
    Q5_0_CODE,
    "`Q5_0` packed matrix-vector argmax using a reusable `Q8_0` RHS."
);
define_quantized_argmax_event!(
    OpMulMatArgmaxQ8_0,
    Q8_0_CODE,
    "`Q8_0` packed matrix-vector argmax using a reusable `Q8_0` RHS."
);
define_quantized_argmax_event!(
    OpMulMatArgmaxQ2K,
    Q2_K_CODE,
    "`Q2_K` packed matrix-vector argmax using a reusable `Q8_K` RHS."
);
define_quantized_argmax_event!(
    OpMulMatArgmaxQ3K,
    Q3_K_CODE,
    "`Q3_K` packed matrix-vector argmax using a reusable `Q8_K` RHS."
);

define_quantized_argmax_event!(
    OpMulMatArgmaxQ4K,
    Q4_K_CODE,
    "`Q4_K` packed matrix-vector argmax using a reusable `Q8_K` RHS."
);
define_quantized_argmax_event!(
    OpMulMatArgmaxQ6K,
    Q6_K_CODE,
    "`Q6_K` packed matrix-vector argmax using a reusable `Q8_K` RHS."
);

/// Result returned by a regular matrix multiplication actor.
pub type MatmulResult = Result<(), MatmulError>;

#[derive(Debug)]
struct MatmulRuntime<'a> {
    event: OpMulMat<'a>,
    result: &'a Cell<MatmulResult>,
}

#[derive(Debug)]
struct MatmulQ4Runtime<'a> {
    event: OpMulMatQ4_0<'a>,
    result: &'a Cell<MatmulResult>,
}

#[derive(Debug)]
struct MatmulQ4_1Runtime<'a> {
    event: OpMulMatQ4_1<'a>,
    result: &'a Cell<MatmulResult>,
}

#[derive(Debug)]
struct MatmulQ5Runtime<'a> {
    event: OpMulMatQ5_0<'a>,
    result: &'a Cell<MatmulResult>,
}

#[derive(Debug)]
struct MatmulQ8Runtime<'a> {
    event: OpMulMatQ8_0<'a>,
    result: &'a Cell<MatmulResult>,
}

#[derive(Debug)]
struct MatmulQ2KRuntime<'a> {
    event: OpMulMatQ2K<'a>,
    result: &'a Cell<MatmulResult>,
}

#[derive(Debug)]
struct MatmulQ3KRuntime<'a> {
    event: OpMulMatQ3K<'a>,
    result: &'a Cell<MatmulResult>,
}

#[derive(Debug)]
struct MatmulQ4KRuntime<'a> {
    event: OpMulMatQ4K<'a>,
    result: &'a Cell<MatmulResult>,
}

#[derive(Debug)]
struct MatmulQ6KRuntime<'a> {
    event: OpMulMatQ6K<'a>,
    result: &'a Cell<MatmulResult>,
}

#[derive(Debug)]
struct MatmulArgmaxRuntime<'a> {
    event: OpMulMatArgmax<'a>,
    result: &'a Cell<MatmulArgmaxResult>,
}

macro_rules! define_quantized_runtime {
    ($runtime:ident, $event:ident) => {
        #[derive(Debug)]
        struct $runtime<'a> {
            event: $event<'a>,
            result: &'a Cell<MatmulArgmaxResult>,
        }
    };
}

define_quantized_runtime!(MatmulArgmaxQ4_0Runtime, OpMulMatArgmaxQ4_0);
define_quantized_runtime!(MatmulArgmaxQ4_1Runtime, OpMulMatArgmaxQ4_1);
define_quantized_runtime!(MatmulArgmaxQ5_0Runtime, OpMulMatArgmaxQ5_0);
define_quantized_runtime!(MatmulArgmaxQ8_0Runtime, OpMulMatArgmaxQ8_0);
define_quantized_runtime!(MatmulArgmaxQ2KRuntime, OpMulMatArgmaxQ2K);
define_quantized_runtime!(MatmulArgmaxQ3KRuntime, OpMulMatArgmaxQ3K);
define_quantized_runtime!(MatmulArgmaxQ4KRuntime, OpMulMatArgmaxQ4K);
define_quantized_runtime!(MatmulArgmaxQ6KRuntime, OpMulMatArgmaxQ6K);

struct UnexpectedRuntime<'a> {
    result: &'a Cell<MatmulResult>,
}

struct MatmulContext {
    q8_0: Box<[Q8_0Scratch]>,
    q8_k: Box<[Q8KScratch]>,
}

impl MatmulContext {
    fn new() -> Self {
        Self {
            q8_0: vec![Q8_0Scratch::ZERO; MAX_Q8_0_BLOCKS].into_boxed_slice(),
            q8_k: vec![Q8KScratch::ZERO; MAX_Q8_K_BLOCKS].into_boxed_slice(),
        }
    }
}

struct MatmulArgmaxContext {
    q8_0: Box<[Q8_0Scratch]>,
    q8_k: Box<[Q8KScratch]>,
}

impl MatmulArgmaxContext {
    fn new() -> Self {
        Self {
            q8_0: vec![Q8_0Scratch::ZERO; MAX_Q8_0_BLOCKS].into_boxed_slice(),
            q8_k: vec![Q8KScratch::ZERO; MAX_Q8_K_BLOCKS].into_boxed_slice(),
        }
    }
}

sml! {
    MatmulMachine<'dispatch> {
        "ready"_s <= *"ready"_s + MulMat(MatmulRuntime<'dispatch>)
            [guard_mul_mat_valid] / effect_mul_mat_execute,
        "ready"_s <= "ready"_s + MulMat(MatmulRuntime<'dispatch>)
            [guard_mul_mat_shape_mismatch] / effect_mul_mat_shape_reject,
        "ready"_s <= "ready"_s + MulMat(MatmulRuntime<'dispatch>)
            [guard_mul_mat_invalid_view] / effect_mul_mat_view_reject,

        "ready"_s <= "ready"_s + MulMatQ4(MatmulQ4Runtime<'dispatch>)
            [guard_mul_mat_q4_valid] / effect_mul_mat_q4_execute,
        "ready"_s <= "ready"_s + MulMatQ4(MatmulQ4Runtime<'dispatch>)
            [guard_mul_mat_q4_shape_mismatch] / effect_mul_mat_q4_shape_reject,
        "ready"_s <= "ready"_s + MulMatQ4(MatmulQ4Runtime<'dispatch>)
            [guard_mul_mat_q4_invalid_view] / effect_mul_mat_q4_view_reject,

        "ready"_s <= "ready"_s + MulMatQ4_1(MatmulQ4_1Runtime<'dispatch>)
            [guard_mul_mat_q4_1_valid] / effect_mul_mat_q4_1_execute,
        "ready"_s <= "ready"_s + MulMatQ4_1(MatmulQ4_1Runtime<'dispatch>)
            [guard_mul_mat_q4_1_shape_mismatch] / effect_mul_mat_q4_1_shape_reject,
        "ready"_s <= "ready"_s + MulMatQ4_1(MatmulQ4_1Runtime<'dispatch>)
            [guard_mul_mat_q4_1_invalid_view] / effect_mul_mat_q4_1_view_reject,

        "ready"_s <= "ready"_s + MulMatQ5(MatmulQ5Runtime<'dispatch>)
            [guard_mul_mat_q5_valid] / effect_mul_mat_q5_execute,
        "ready"_s <= "ready"_s + MulMatQ5(MatmulQ5Runtime<'dispatch>)
            [guard_mul_mat_q5_shape_mismatch] / effect_mul_mat_q5_shape_reject,
        "ready"_s <= "ready"_s + MulMatQ5(MatmulQ5Runtime<'dispatch>)
            [guard_mul_mat_q5_invalid_view] / effect_mul_mat_q5_view_reject,

        "ready"_s <= "ready"_s + MulMatQ8(MatmulQ8Runtime<'dispatch>)
            [guard_mul_mat_q8_valid] / effect_mul_mat_q8_execute,
        "ready"_s <= "ready"_s + MulMatQ8(MatmulQ8Runtime<'dispatch>)
            [guard_mul_mat_q8_shape_mismatch] / effect_mul_mat_q8_shape_reject,
        "ready"_s <= "ready"_s + MulMatQ8(MatmulQ8Runtime<'dispatch>)
            [guard_mul_mat_q8_invalid_view] / effect_mul_mat_q8_view_reject,

        "ready"_s <= "ready"_s + MulMatQ2K(MatmulQ2KRuntime<'dispatch>)
            [guard_mul_mat_q2_k_valid] / effect_mul_mat_q2_k_execute,
        "ready"_s <= "ready"_s + MulMatQ2K(MatmulQ2KRuntime<'dispatch>)
            [guard_mul_mat_q2_k_shape_mismatch] / effect_mul_mat_q2_k_shape_reject,
        "ready"_s <= "ready"_s + MulMatQ2K(MatmulQ2KRuntime<'dispatch>)
            [guard_mul_mat_q2_k_invalid_view] / effect_mul_mat_q2_k_view_reject,

        "ready"_s <= "ready"_s + MulMatQ3K(MatmulQ3KRuntime<'dispatch>)
            [guard_mul_mat_q3_k_valid] / effect_mul_mat_q3_k_execute,
        "ready"_s <= "ready"_s + MulMatQ3K(MatmulQ3KRuntime<'dispatch>)
            [guard_mul_mat_q3_k_shape_mismatch] / effect_mul_mat_q3_k_shape_reject,
        "ready"_s <= "ready"_s + MulMatQ3K(MatmulQ3KRuntime<'dispatch>)
            [guard_mul_mat_q3_k_invalid_view] / effect_mul_mat_q3_k_view_reject,

        "ready"_s <= "ready"_s + MulMatQ4K(MatmulQ4KRuntime<'dispatch>)
            [guard_mul_mat_q4_k_valid] / effect_mul_mat_q4_k_execute,
        "ready"_s <= "ready"_s + MulMatQ4K(MatmulQ4KRuntime<'dispatch>)
            [guard_mul_mat_q4_k_shape_mismatch] / effect_mul_mat_q4_k_shape_reject,
        "ready"_s <= "ready"_s + MulMatQ4K(MatmulQ4KRuntime<'dispatch>)
            [guard_mul_mat_q4_k_invalid_view] / effect_mul_mat_q4_k_view_reject,

        "ready"_s <= "ready"_s + MulMatQ6K(MatmulQ6KRuntime<'dispatch>)
            [guard_mul_mat_q6_k_valid] / effect_mul_mat_q6_k_execute,
        "ready"_s <= "ready"_s + MulMatQ6K(MatmulQ6KRuntime<'dispatch>)
            [guard_mul_mat_q6_k_shape_mismatch] / effect_mul_mat_q6_k_shape_reject,
        "ready"_s <= "ready"_s + MulMatQ6K(MatmulQ6KRuntime<'dispatch>)
            [guard_mul_mat_q6_k_invalid_view] / effect_mul_mat_q6_k_view_reject,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_mul_mat_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }

    MatmulArgmaxMachine<'dispatch> {
        "ready"_s <= *"ready"_s + MulMatArgmax(MatmulArgmaxRuntime<'dispatch>)
            [guard_mul_mat_argmax_valid] / effect_mul_mat_argmax_execute,
        "ready"_s <= "ready"_s + MulMatArgmax(MatmulArgmaxRuntime<'dispatch>)
            [guard_mul_mat_argmax_shape_mismatch] / effect_mul_mat_argmax_shape_reject,
        "ready"_s <= "ready"_s + MulMatArgmax(MatmulArgmaxRuntime<'dispatch>)
        [guard_mul_mat_argmax_invalid_view] / effect_mul_mat_argmax_view_reject,

        "ready"_s <= "ready"_s + MulMatArgmaxQ4_0(MatmulArgmaxQ4_0Runtime<'dispatch>)
            [guard_mul_mat_argmax_q4_0_valid] / effect_mul_mat_argmax_q4_0_execute,
        "ready"_s <= "ready"_s + MulMatArgmaxQ4_0(MatmulArgmaxQ4_0Runtime<'dispatch>)
            [guard_mul_mat_argmax_q4_0_shape_mismatch] / effect_mul_mat_argmax_shape_reject_q4_0,
        "ready"_s <= "ready"_s + MulMatArgmaxQ4_0(MatmulArgmaxQ4_0Runtime<'dispatch>)
            [guard_mul_mat_argmax_q4_0_invalid_view] / effect_mul_mat_argmax_view_reject_q4_0,

        "ready"_s <= "ready"_s + MulMatArgmaxQ4_1(MatmulArgmaxQ4_1Runtime<'dispatch>)
            [guard_mul_mat_argmax_q4_1_valid] / effect_mul_mat_argmax_q4_1_execute,
        "ready"_s <= "ready"_s + MulMatArgmaxQ4_1(MatmulArgmaxQ4_1Runtime<'dispatch>)
            [guard_mul_mat_argmax_q4_1_shape_mismatch] / effect_mul_mat_argmax_shape_reject_q4_1,
        "ready"_s <= "ready"_s + MulMatArgmaxQ4_1(MatmulArgmaxQ4_1Runtime<'dispatch>)
            [guard_mul_mat_argmax_q4_1_invalid_view] / effect_mul_mat_argmax_view_reject_q4_1,

        "ready"_s <= "ready"_s + MulMatArgmaxQ5_0(MatmulArgmaxQ5_0Runtime<'dispatch>)
            [guard_mul_mat_argmax_q5_0_valid] / effect_mul_mat_argmax_q5_0_execute,
        "ready"_s <= "ready"_s + MulMatArgmaxQ5_0(MatmulArgmaxQ5_0Runtime<'dispatch>)
            [guard_mul_mat_argmax_q5_0_shape_mismatch] / effect_mul_mat_argmax_shape_reject_q5_0,
        "ready"_s <= "ready"_s + MulMatArgmaxQ5_0(MatmulArgmaxQ5_0Runtime<'dispatch>)
            [guard_mul_mat_argmax_q5_0_invalid_view] / effect_mul_mat_argmax_view_reject_q5_0,

        "ready"_s <= "ready"_s + MulMatArgmaxQ8_0(MatmulArgmaxQ8_0Runtime<'dispatch>)
            [guard_mul_mat_argmax_q8_0_valid] / effect_mul_mat_argmax_q8_0_execute,
        "ready"_s <= "ready"_s + MulMatArgmaxQ8_0(MatmulArgmaxQ8_0Runtime<'dispatch>)
            [guard_mul_mat_argmax_q8_0_shape_mismatch] / effect_mul_mat_argmax_shape_reject_q8_0,
        "ready"_s <= "ready"_s + MulMatArgmaxQ8_0(MatmulArgmaxQ8_0Runtime<'dispatch>)
            [guard_mul_mat_argmax_q8_0_invalid_view] / effect_mul_mat_argmax_view_reject_q8_0,

        "ready"_s <= "ready"_s + MulMatArgmaxQ2K(MatmulArgmaxQ2KRuntime<'dispatch>)
            [guard_mul_mat_argmax_q2_k_valid] / effect_mul_mat_argmax_q2_k_execute,
        "ready"_s <= "ready"_s + MulMatArgmaxQ2K(MatmulArgmaxQ2KRuntime<'dispatch>)
            [guard_mul_mat_argmax_q2_k_shape_mismatch] / effect_mul_mat_argmax_shape_reject_q2_k,
        "ready"_s <= "ready"_s + MulMatArgmaxQ2K(MatmulArgmaxQ2KRuntime<'dispatch>)
            [guard_mul_mat_argmax_q2_k_invalid_view] / effect_mul_mat_argmax_view_reject_q2_k,

        "ready"_s <= "ready"_s + MulMatArgmaxQ3K(MatmulArgmaxQ3KRuntime<'dispatch>)
            [guard_mul_mat_argmax_q3_k_valid] / effect_mul_mat_argmax_q3_k_execute,
        "ready"_s <= "ready"_s + MulMatArgmaxQ3K(MatmulArgmaxQ3KRuntime<'dispatch>)
            [guard_mul_mat_argmax_q3_k_shape_mismatch] / effect_mul_mat_argmax_shape_reject_q3_k,
        "ready"_s <= "ready"_s + MulMatArgmaxQ3K(MatmulArgmaxQ3KRuntime<'dispatch>)
            [guard_mul_mat_argmax_q3_k_invalid_view] / effect_mul_mat_argmax_view_reject_q3_k,

        "ready"_s <= "ready"_s + MulMatArgmaxQ4K(MatmulArgmaxQ4KRuntime<'dispatch>)
            [guard_mul_mat_argmax_q4_k_valid] / effect_mul_mat_argmax_q4_k_execute,
        "ready"_s <= "ready"_s + MulMatArgmaxQ4K(MatmulArgmaxQ4KRuntime<'dispatch>)
            [guard_mul_mat_argmax_q4_k_shape_mismatch] / effect_mul_mat_argmax_shape_reject_q4_k,
        "ready"_s <= "ready"_s + MulMatArgmaxQ4K(MatmulArgmaxQ4KRuntime<'dispatch>)
            [guard_mul_mat_argmax_q4_k_invalid_view] / effect_mul_mat_argmax_view_reject_q4_k,

        "ready"_s <= "ready"_s + MulMatArgmaxQ6K(MatmulArgmaxQ6KRuntime<'dispatch>)
            [guard_mul_mat_argmax_q6_k_valid] / effect_mul_mat_argmax_q6_k_execute,
        "ready"_s <= "ready"_s + MulMatArgmaxQ6K(MatmulArgmaxQ6KRuntime<'dispatch>)
            [guard_mul_mat_argmax_q6_k_shape_mismatch] / effect_mul_mat_argmax_shape_reject_q6_k,
        "ready"_s <= "ready"_s + MulMatArgmaxQ6K(MatmulArgmaxQ6KRuntime<'dispatch>)
            [guard_mul_mat_argmax_q6_k_invalid_view] / effect_mul_mat_argmax_view_reject_q6_k,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_mul_mat_argmax_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, run-to-completion actor for the maintained F32 matrix path.
pub struct MatmulKernel {
    machine: MatmulMachineStateMachine<MatmulContext>,
}

impl fmt::Debug for MatmulKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MatmulKernel")
            .finish_non_exhaustive()
    }
}

impl Default for MatmulKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl MatmulKernel {
    /// Constructs an independent matrix actor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: MatmulMachineStateMachine::new(MatmulContext::new()),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: MatmulEvent>(&mut self, event: E) -> E::Output {
        let output = event.dispatch(self);
        let _ = self.machine.is(&MatmulMachineStates::Ready);
        output
    }

    fn mul_mat(&mut self, event: OpMulMat<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(MatmulMachineEvents::MulMat(MatmulRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    fn mul_mat_q4_0(&mut self, event: OpMulMatQ4_0<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(MatmulMachineEvents::MulMatQ4(MatmulQ4Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    fn mul_mat_q4_1(&mut self, event: OpMulMatQ4_1<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(MatmulMachineEvents::MulMatQ4_1(MatmulQ4_1Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    fn mul_mat_q5(&mut self, event: OpMulMatQ5_0<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(MatmulMachineEvents::MulMatQ5(MatmulQ5Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    fn mul_mat_q8_0(&mut self, event: OpMulMatQ8_0<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(MatmulMachineEvents::MulMatQ8(MatmulQ8Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    fn mul_mat_q2_k(&mut self, event: OpMulMatQ2K<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(MatmulMachineEvents::MulMatQ2K(MatmulQ2KRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    fn mul_mat_q3_k(&mut self, event: OpMulMatQ3K<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(MatmulMachineEvents::MulMatQ3K(MatmulQ3KRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    fn mul_mat_q4_k(&mut self, event: OpMulMatQ4K<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(MatmulMachineEvents::MulMatQ4K(MatmulQ4KRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    fn mul_mat_q6_k(&mut self, event: OpMulMatQ6K<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(MatmulMachineEvents::MulMatQ6K(MatmulQ6KRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    fn unexpected(&mut self, _event: UnexpectedMatmul) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::Internal));
        self.machine
            .process_event(MatmulMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }
}

/// Single-writer, run-to-completion actor for F32 matrix-vector argmax.
pub struct MatmulArgmaxKernel {
    machine: MatmulArgmaxMachineStateMachine<MatmulArgmaxContext>,
}

impl fmt::Debug for MatmulArgmaxKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MatmulArgmaxKernel")
            .finish_non_exhaustive()
    }
}

impl Default for MatmulArgmaxKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl MatmulArgmaxKernel {
    /// Constructs an independent matrix-vector argmax actor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: MatmulArgmaxMachineStateMachine::new(MatmulArgmaxContext::new()),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: MatmulArgmaxEvent>(&mut self, event: E) -> E::Output {
        let output = event.dispatch(self);
        let _ = self.machine.is(&MatmulArgmaxMachineStates::Ready);
        output
    }

    fn argmax(&mut self, event: OpMulMatArgmax<'_>) -> MatmulArgmaxResult {
        let result = Cell::new(Err(MatmulError::UnexpectedEvent));
        self.machine
            .process_event(MatmulArgmaxMachineEvents::MulMatArgmax(
                MatmulArgmaxRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    fn unexpected(&mut self, _event: UnexpectedMatmul) -> MatmulArgmaxResult {
        let result = Cell::new(Err(MatmulError::Internal));
        self.machine
            .process_event(MatmulArgmaxMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get().map(|()| 0)
    }
}

/// Result returned by the F32 matrix-vector argmax actor.
pub type MatmulArgmaxResult = Result<i32, MatmulError>;

/// Typed event accepted by [`MatmulArgmaxKernel`].
pub trait MatmulArgmaxEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut MatmulArgmaxKernel) -> Self::Output;
}

impl MatmulArgmaxEvent for OpMulMatArgmax<'_> {
    type Output = MatmulArgmaxResult;

    fn dispatch(self, actor: &mut MatmulArgmaxKernel) -> Self::Output {
        actor.argmax(self)
    }
}

macro_rules! impl_quantized_argmax_event {
    ($event:ident, $runtime:ident, $variant:ident) => {
        impl MatmulArgmaxEvent for $event<'_> {
            type Output = MatmulArgmaxResult;

            fn dispatch(self, actor: &mut MatmulArgmaxKernel) -> Self::Output {
                let result = Cell::new(Err(MatmulError::UnexpectedEvent));
                actor
                    .machine
                    .process_event(MatmulArgmaxMachineEvents::$variant($runtime {
                        event: self,
                        result: &result,
                    }))
                    .map_err(|_| MatmulError::Internal)?;
                result.get()
            }
        }
    };
}

impl_quantized_argmax_event!(
    OpMulMatArgmaxQ4_0,
    MatmulArgmaxQ4_0Runtime,
    MulMatArgmaxQ4_0
);
impl_quantized_argmax_event!(
    OpMulMatArgmaxQ4_1,
    MatmulArgmaxQ4_1Runtime,
    MulMatArgmaxQ4_1
);
impl_quantized_argmax_event!(
    OpMulMatArgmaxQ5_0,
    MatmulArgmaxQ5_0Runtime,
    MulMatArgmaxQ5_0
);
impl_quantized_argmax_event!(
    OpMulMatArgmaxQ8_0,
    MatmulArgmaxQ8_0Runtime,
    MulMatArgmaxQ8_0
);
impl_quantized_argmax_event!(OpMulMatArgmaxQ2K, MatmulArgmaxQ2KRuntime, MulMatArgmaxQ2K);
impl_quantized_argmax_event!(OpMulMatArgmaxQ3K, MatmulArgmaxQ3KRuntime, MulMatArgmaxQ3K);

impl_quantized_argmax_event!(OpMulMatArgmaxQ4K, MatmulArgmaxQ4KRuntime, MulMatArgmaxQ4K);
impl_quantized_argmax_event!(OpMulMatArgmaxQ6K, MatmulArgmaxQ6KRuntime, MulMatArgmaxQ6K);

impl MatmulArgmaxEvent for UnexpectedMatmul {
    type Output = MatmulArgmaxResult;

    fn dispatch(self, actor: &mut MatmulArgmaxKernel) -> Self::Output {
        actor.unexpected(self)
    }
}

/// Typed event accepted by [`MatmulKernel`].
pub trait MatmulEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut MatmulKernel) -> Self::Output;
}

impl MatmulEvent for OpMulMat<'_> {
    type Output = MatmulResult;

    fn dispatch(self, actor: &mut MatmulKernel) -> Self::Output {
        actor.mul_mat(self)
    }
}

impl MatmulEvent for OpMulMatQ4_0<'_> {
    type Output = MatmulResult;

    fn dispatch(self, actor: &mut MatmulKernel) -> Self::Output {
        actor.mul_mat_q4_0(self)
    }
}

impl MatmulEvent for OpMulMatQ4_1<'_> {
    type Output = MatmulResult;

    fn dispatch(self, actor: &mut MatmulKernel) -> Self::Output {
        actor.mul_mat_q4_1(self)
    }
}

impl MatmulEvent for OpMulMatQ5_0<'_> {
    type Output = MatmulResult;

    fn dispatch(self, actor: &mut MatmulKernel) -> Self::Output {
        actor.mul_mat_q5(self)
    }
}

impl MatmulEvent for OpMulMatQ8_0<'_> {
    type Output = MatmulResult;

    fn dispatch(self, actor: &mut MatmulKernel) -> Self::Output {
        actor.mul_mat_q8_0(self)
    }
}

impl MatmulEvent for OpMulMatQ2K<'_> {
    type Output = MatmulResult;

    fn dispatch(self, actor: &mut MatmulKernel) -> Self::Output {
        actor.mul_mat_q2_k(self)
    }
}

impl MatmulEvent for OpMulMatQ3K<'_> {
    type Output = MatmulResult;

    fn dispatch(self, actor: &mut MatmulKernel) -> Self::Output {
        actor.mul_mat_q3_k(self)
    }
}

impl MatmulEvent for OpMulMatQ4K<'_> {
    type Output = MatmulResult;

    fn dispatch(self, actor: &mut MatmulKernel) -> Self::Output {
        actor.mul_mat_q4_k(self)
    }
}

impl MatmulEvent for OpMulMatQ6K<'_> {
    type Output = MatmulResult;

    fn dispatch(self, actor: &mut MatmulKernel) -> Self::Output {
        actor.mul_mat_q6_k(self)
    }
}

impl MatmulEvent for UnexpectedMatmul {
    type Output = MatmulResult;

    fn dispatch(self, actor: &mut MatmulKernel) -> Self::Output {
        actor.unexpected(self)
    }
}

fn views_valid(event: &OpMulMat<'_>) -> bool {
    event.lhs.validate().is_ok()
        && event.rhs.validate().is_ok()
        && event.destination.validate().is_ok()
}

fn q4_views_valid(event: &OpMulMatQ4_0<'_>) -> bool {
    event.lhs.validate(Q4_0_CODE, QK_32, Q4_0_BYTES)
        && usize::try_from(event.lhs.layout().ne()[0] / QK_32 as u64)
            .is_ok_and(|blocks| blocks <= MAX_Q8_0_BLOCKS)
        && event.rhs.validate().is_ok()
        && event.destination.validate().is_ok()
        && event.rhs.layout().is_dense_contiguous()
        && event.destination.layout().is_dense_contiguous()
}

fn q4_1_views_valid(event: &OpMulMatQ4_1<'_>) -> bool {
    event.lhs.validate(Q4_1_CODE, QK_32, Q4_1_BYTES)
        && usize::try_from(event.lhs.layout().ne()[0] / QK_32 as u64)
            .is_ok_and(|blocks| blocks <= MAX_Q8_0_BLOCKS)
        && event.rhs.validate().is_ok()
        && event.destination.validate().is_ok()
        && event.rhs.layout().is_dense_contiguous()
        && event.destination.layout().is_dense_contiguous()
}

fn q5_views_valid(event: &OpMulMatQ5_0<'_>) -> bool {
    event.lhs.validate(Q5_0_CODE, QK_32, Q5_0_BYTES)
        && usize::try_from(event.lhs.layout().ne()[0] / QK_32 as u64)
            .is_ok_and(|blocks| blocks <= MAX_Q8_0_BLOCKS)
        && event.rhs.validate().is_ok()
        && event.destination.validate().is_ok()
        && event.rhs.layout().is_dense_contiguous()
        && event.destination.layout().is_dense_contiguous()
}

fn q8_views_valid(event: &OpMulMatQ8_0<'_>) -> bool {
    event.lhs.validate(Q8_0_CODE, QK_32, Q8_0_BYTES)
        && usize::try_from(event.lhs.layout().ne()[0] / QK_32 as u64)
            .is_ok_and(|blocks| blocks <= MAX_Q8_0_BLOCKS)
        && event.rhs.validate().is_ok()
        && event.destination.validate().is_ok()
        && event.rhs.layout().is_dense_contiguous()
        && event.destination.layout().is_dense_contiguous()
}

fn q2_k_views_valid(event: &OpMulMatQ2K<'_>) -> bool {
    quantized_views_valid(
        &event.lhs,
        &event.rhs,
        &event.destination,
        Q2_K_CODE,
        QK_K,
        Q2_K_BYTES,
    )
}

fn q3_k_views_valid(event: &OpMulMatQ3K<'_>) -> bool {
    quantized_views_valid(
        &event.lhs,
        &event.rhs,
        &event.destination,
        Q3_K_CODE,
        QK_K,
        Q3_K_BYTES,
    )
}

fn q4_k_views_valid(event: &OpMulMatQ4K<'_>) -> bool {
    quantized_views_valid(
        &event.lhs,
        &event.rhs,
        &event.destination,
        Q4_K_CODE,
        QK_K,
        Q4_K_BYTES,
    )
}

fn q6_k_views_valid(event: &OpMulMatQ6K<'_>) -> bool {
    quantized_views_valid(
        &event.lhs,
        &event.rhs,
        &event.destination,
        Q6_K_CODE,
        QK_K,
        Q6_K_BYTES,
    )
}

const fn q4_shape_valid(event: &OpMulMatQ4_0<'_>) -> bool {
    let lhs = event.lhs.layout().ne();
    let rhs = event.rhs.layout().ne();
    let destination = event.destination.layout().ne();
    lhs[0] > 0
        && lhs[1] > 0
        && rhs[0] > 0
        && rhs[1] == lhs[0]
        && destination[0] == rhs[0]
        && destination[1] == lhs[1]
        && lhs[2] == 1
        && lhs[3] == 1
        && rhs[2] == 1
        && rhs[3] == 1
        && destination[2] == 1
        && destination[3] == 1
}

const fn q4_1_shape_valid(event: &OpMulMatQ4_1<'_>) -> bool {
    let lhs = event.lhs.layout().ne();
    let rhs = event.rhs.layout().ne();
    let destination = event.destination.layout().ne();
    lhs[0] > 0
        && lhs[1] > 0
        && rhs[0] > 0
        && rhs[1] == lhs[0]
        && destination[0] == rhs[0]
        && destination[1] == lhs[1]
        && lhs[2] == 1
        && lhs[3] == 1
        && rhs[2] == 1
        && rhs[3] == 1
        && destination[2] == 1
        && destination[3] == 1
}

const fn q5_shape_valid(event: &OpMulMatQ5_0<'_>) -> bool {
    let lhs = event.lhs.layout().ne();
    let rhs = event.rhs.layout().ne();
    let destination = event.destination.layout().ne();
    lhs[0] > 0
        && lhs[1] > 0
        && rhs[0] > 0
        && rhs[1] == lhs[0]
        && destination[0] == rhs[0]
        && destination[1] == lhs[1]
        && lhs[2] == 1
        && lhs[3] == 1
        && rhs[2] == 1
        && rhs[3] == 1
        && destination[2] == 1
        && destination[3] == 1
}

const fn q8_shape_valid(event: &OpMulMatQ8_0<'_>) -> bool {
    let lhs = event.lhs.layout().ne();
    let rhs = event.rhs.layout().ne();
    let destination = event.destination.layout().ne();
    lhs[0] > 0
        && lhs[1] > 0
        && rhs[0] > 0
        && rhs[1] == lhs[0]
        && destination[0] == rhs[0]
        && destination[1] == lhs[1]
        && lhs[2] == 1
        && lhs[3] == 1
        && rhs[2] == 1
        && rhs[3] == 1
        && destination[2] == 1
        && destination[3] == 1
}

const fn q2_k_shape_valid(event: &OpMulMatQ2K<'_>) -> bool {
    let lhs = event.lhs.layout().ne();
    let rhs = event.rhs.layout().ne();
    let destination = event.destination.layout().ne();
    lhs[0] > 0
        && lhs[1] > 0
        && rhs[0] > 0
        && rhs[1] == lhs[0]
        && destination[0] == rhs[0]
        && destination[1] == lhs[1]
        && lhs[2] == 1
        && lhs[3] == 1
        && rhs[2] == 1
        && rhs[3] == 1
        && destination[2] == 1
        && destination[3] == 1
}

const fn q3_k_shape_valid(event: &OpMulMatQ3K<'_>) -> bool {
    let lhs = event.lhs.layout().ne();
    let rhs = event.rhs.layout().ne();
    let destination = event.destination.layout().ne();
    lhs[0] > 0
        && lhs[1] > 0
        && rhs[0] > 0
        && rhs[1] == lhs[0]
        && destination[0] == rhs[0]
        && destination[1] == lhs[1]
        && lhs[2] == 1
        && lhs[3] == 1
        && rhs[2] == 1
        && rhs[3] == 1
        && destination[2] == 1
        && destination[3] == 1
}

const fn q4_k_shape_valid(event: &OpMulMatQ4K<'_>) -> bool {
    let lhs = event.lhs.layout().ne();
    let rhs = event.rhs.layout().ne();
    let destination = event.destination.layout().ne();
    lhs[0] > 0
        && lhs[1] > 0
        && rhs[0] > 0
        && rhs[1] == lhs[0]
        && destination[0] == rhs[0]
        && destination[1] == lhs[1]
        && lhs[2] == 1
        && lhs[3] == 1
        && rhs[2] == 1
        && rhs[3] == 1
        && destination[2] == 1
        && destination[3] == 1
}

const fn q6_k_shape_valid(event: &OpMulMatQ6K<'_>) -> bool {
    let lhs = event.lhs.layout().ne();
    let rhs = event.rhs.layout().ne();
    let destination = event.destination.layout().ne();
    lhs[0] > 0
        && lhs[1] > 0
        && rhs[0] > 0
        && rhs[1] == lhs[0]
        && destination[0] == rhs[0]
        && destination[1] == lhs[1]
        && lhs[2] == 1
        && lhs[3] == 1
        && rhs[2] == 1
        && rhs[3] == 1
        && destination[2] == 1
        && destination[3] == 1
}

const fn shape_valid(event: &OpMulMat<'_>) -> bool {
    let lhs = event.lhs.layout().ne();
    let rhs = event.rhs.layout().ne();
    let destination = event.destination.layout().ne();
    lhs[0] > 0
        && lhs[1] > 0
        && rhs[0] > 0
        && rhs[1] > 0
        && rhs[1] == lhs[0]
        && destination[0] == rhs[0]
        && destination[1] == lhs[1]
        && lhs[2] == 1
        && lhs[3] == 1
        && rhs[2] == 1
        && rhs[3] == 1
        && destination[2] == 1
        && destination[3] == 1
}

fn argmax_views_valid(event: &OpMulMatArgmax<'_>) -> bool {
    event.lhs.validate().is_ok()
        && event.rhs.validate().is_ok()
        && event.destination.validate().is_ok()
        && event.lhs.layout().is_dense_contiguous()
        && event.rhs.layout().is_dense_contiguous()
        && event.destination.layout().is_dense_contiguous()
}

macro_rules! impl_quantized_argmax_routes {
    (
        $valid_guard:ident,
        $shape_guard:ident,
        $view_guard:ident,
        $execute:ident,
        $shape_reject:ident,
        $view_reject:ident,
        $runtime:ident,
        $code:expr,
        $block_values:expr,
        $block_bytes:expr,
        $runner:ident,
        $scratch:ident
    ) => {
        fn $valid_guard(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(quantized_views_valid(
                &event.event.lhs,
                &event.event.rhs,
                &event.event.destination,
                $code,
                $block_values,
                $block_bytes,
            ) && quantized_shape_valid(
                &event.event.lhs,
                &event.event.rhs,
                &event.event.destination,
            ))
        }

        fn $shape_guard(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(quantized_views_valid(
                &event.event.lhs,
                &event.event.rhs,
                &event.event.destination,
                $code,
                $block_values,
                $block_bytes,
            ) && !quantized_shape_valid(
                &event.event.lhs,
                &event.event.rhs,
                &event.event.destination,
            ))
        }

        fn $view_guard(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(!quantized_views_valid(
                &event.event.lhs,
                &event.event.rhs,
                &event.event.destination,
                $code,
                $block_values,
                $block_bytes,
            ))
        }

        fn $execute(&mut self, mut event: $runtime<'_>) -> Result<(), ()> {
            let index = $runner(
                &event.event.lhs,
                &event.event.rhs,
                &mut event.event.destination,
                &mut self.$scratch,
            );
            event.result.set(Ok(index));
            Ok(())
        }

        fn $shape_reject(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event.result.set(Err(MatmulError::ShapeMismatch));
            Ok(())
        }

        fn $view_reject(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event.result.set(Err(MatmulError::InvalidView));
            Ok(())
        }
    };
}

const fn argmax_shape_valid(event: &OpMulMatArgmax<'_>) -> bool {
    let lhs = event.lhs.layout().ne();
    let rhs = event.rhs.layout().ne();
    let destination = event.destination.layout().ne();
    lhs[0] > 0
        && lhs[1] > 0
        && lhs[2] == 1
        && lhs[3] == 1
        && rhs[0] == 1
        && rhs[1] == lhs[0]
        && rhs[2] == 1
        && rhs[3] == 1
        && destination[0] == 1
        && destination[1] == 1
        && destination[2] == 1
        && destination[3] == 1
}

fn quantized_views_valid(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &TensorViewMut<'_>,
    code: u8,
    block_values: usize,
    block_bytes: usize,
) -> bool {
    let max_blocks = if block_values == QK_32 {
        MAX_Q8_0_BLOCKS
    } else {
        MAX_Q8_K_BLOCKS
    };
    lhs.validate(code, block_values, block_bytes)
        && usize::try_from(lhs.layout().ne()[0] / block_values as u64)
            .is_ok_and(|blocks| blocks <= max_blocks)
        && rhs.validate().is_ok()
        && destination.validate().is_ok()
        && rhs.layout().is_dense_contiguous()
        && destination.layout().is_dense_contiguous()
}

const fn quantized_shape_valid(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &TensorViewMut<'_>,
) -> bool {
    let lhs_shape = lhs.layout().ne();
    let rhs_shape = rhs.layout().ne();
    let destination_shape = destination.layout().ne();
    lhs_shape[0] > 0
        && lhs_shape[1] > 0
        && lhs_shape[2] == 1
        && lhs_shape[3] == 1
        && rhs_shape[0] == 1
        && rhs_shape[1] == lhs_shape[0]
        && rhs_shape[2] == 1
        && rhs_shape[3] == 1
        && destination_shape[0] == 1
        && destination_shape[1] == 1
        && destination_shape[2] == 1
        && destination_shape[3] == 1
}

macro_rules! impl_target_matmul_validation {
    ($event:ident, $views:ident, $shape:ident) => {
        impl<'a> $event<'a> {
            /// Reports whether the complete packed view contract is valid.
            pub(crate) fn target_views_valid(&self) -> bool {
                $views(self)
            }

            /// Reports whether dimensions match the packed matmul contract.
            pub(crate) const fn target_shape_valid(&self) -> bool {
                $shape(self)
            }
        }
    };
}

impl_target_matmul_validation!(OpMulMatQ4_0, q4_views_valid, q4_shape_valid);
impl_target_matmul_validation!(OpMulMatQ4_1, q4_1_views_valid, q4_1_shape_valid);
impl_target_matmul_validation!(OpMulMatQ5_0, q5_views_valid, q5_shape_valid);
impl_target_matmul_validation!(OpMulMatQ8_0, q8_views_valid, q8_shape_valid);
impl_target_matmul_validation!(OpMulMatQ2K, q2_k_views_valid, q2_k_shape_valid);
impl_target_matmul_validation!(OpMulMatQ3K, q3_k_views_valid, q3_k_shape_valid);
impl_target_matmul_validation!(OpMulMatQ4K, q4_k_views_valid, q4_k_shape_valid);
impl_target_matmul_validation!(OpMulMatQ6K, q6_k_views_valid, q6_k_shape_valid);

macro_rules! impl_target_matmul_argmax_validation {
    ($event:ident, $code:expr, $block_values:expr, $block_bytes:expr) => {
        impl<'a> $event<'a> {
            /// Reports whether the complete packed argmax view contract is valid.
            pub(crate) fn target_views_valid(&self) -> bool {
                quantized_views_valid(
                    &self.lhs,
                    &self.rhs,
                    &self.destination,
                    $code,
                    $block_values,
                    $block_bytes,
                )
            }

            /// Reports whether dimensions match the packed argmax contract.
            pub(crate) const fn target_shape_valid(&self) -> bool {
                quantized_shape_valid(&self.lhs, &self.rhs, &self.destination)
            }
        }
    };
}

impl_target_matmul_argmax_validation!(OpMulMatArgmaxQ4_0, Q4_0_CODE, QK_32, Q4_0_BYTES);
impl_target_matmul_argmax_validation!(OpMulMatArgmaxQ4_1, Q4_1_CODE, QK_32, Q4_1_BYTES);
impl_target_matmul_argmax_validation!(OpMulMatArgmaxQ5_0, Q5_0_CODE, QK_32, Q5_0_BYTES);
impl_target_matmul_argmax_validation!(OpMulMatArgmaxQ8_0, Q8_0_CODE, QK_32, Q8_0_BYTES);
impl_target_matmul_argmax_validation!(OpMulMatArgmaxQ2K, Q2_K_CODE, QK_K, Q2_K_BYTES);
impl_target_matmul_argmax_validation!(OpMulMatArgmaxQ3K, Q3_K_CODE, QK_K, Q3_K_BYTES);
impl_target_matmul_argmax_validation!(OpMulMatArgmaxQ4K, Q4_K_CODE, QK_K, Q4_K_BYTES);
impl_target_matmul_argmax_validation!(OpMulMatArgmaxQ6K, Q6_K_CODE, QK_K, Q6_K_BYTES);

#[inline]
fn f32_to_f16(value: f32) -> u16 {
    crate::any::quant::fp32_to_fp16(value)
}

#[allow(clippy::cast_possible_truncation)]
pub(crate) fn quantize_q8_0(rhs: &TensorView<'_>, scratch: &mut [Q8_0Scratch], k: usize) {
    let blocks = k / QK_32;
    let mut block = 0;
    while block < blocks {
        let base = block * QK_32;
        let mut amax = 0.0_f32;
        let mut index = 0;
        while index < QK_32 {
            amax = amax.max(rhs.read(base + index).abs());
            index += 1;
        }
        let scale = amax / 127.0;
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        scratch[block].d = f32_to_f16(scale);
        index = 0;
        while index < QK_32 {
            let quantized = (rhs.read(base + index) * inverse)
                .round()
                .clamp(-127.0, 127.0);
            scratch[block].qs[index] = quantized as i8;
            index += 1;
        }
        block += 1;
    }
}

/// Quantizes one logical RHS column using the pinned `nb=[4, 4*n, ...]`
/// traversal from `detail.hpp:3392-3394` without materializing a dense copy.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn quantize_q8_0_strided(
    rhs: &TensorView<'_>,
    scratch: &mut [Q8_0Scratch],
    k: usize,
    columns: usize,
    column: usize,
) {
    let blocks = k / QK_32;
    let mut block = 0;
    while block < blocks {
        let base = block * QK_32;
        let mut amax = 0.0_f32;
        let mut index = 0;
        while index < QK_32 {
            amax = amax.max(rhs.read(column + columns * (base + index)).abs());
            index += 1;
        }
        let scale = amax / 127.0;
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        scratch[block].d = f32_to_f16(scale);
        index = 0;
        while index < QK_32 {
            let quantized = (rhs.read(column + columns * (base + index)) * inverse)
                .round()
                .clamp(-127.0, 127.0);
            scratch[block].qs[index] = quantized as i8;
            index += 1;
        }
        block += 1;
    }
}

/// Ports the pinned `nearest_int` bit-level rounding used by `Q8_K` quantization.
///
/// The reference deliberately differs from `std::round` on halfway values;
/// retaining its float-bit arithmetic keeps the packed RHS contract exact
/// without relying on a native unchecked conversion.
#[inline]
fn nearest_int(value: f32) -> i32 {
    let biased = value + 12_582_912.0_f32;
    let bits = biased.to_bits() & 0x007f_ffff;
    i32::try_from(bits).expect("nearest_int mantissa fits i32") - 0x0040_0000
}

#[allow(clippy::cast_possible_truncation)]
pub(crate) fn quantize_q8_k(rhs: &TensorView<'_>, scratch: &mut [Q8KScratch], k: usize) {
    let blocks = k / QK_K;
    let mut block = 0;
    while block < blocks {
        let base = block * QK_K;
        let mut maximum = 0.0_f32;
        let mut absolute = 0.0_f32;
        let mut index = 0;
        while index < QK_K {
            let value = rhs.read(base + index);
            if value.abs() > absolute {
                absolute = value.abs();
                maximum = value;
            }
            index += 1;
        }
        if absolute == 0.0 {
            scratch[block] = Q8KScratch::ZERO;
            block += 1;
            continue;
        }
        let inverse = -127.0 / maximum;
        scratch[block].d = 1.0 / inverse;
        index = 0;
        while index < QK_K {
            let quantized = nearest_int(inverse * rhs.read(base + index)).min(127);
            scratch[block].qs[index] = i8::try_from(quantized).expect("Q8_K value fits i8");
            index += 1;
        }
        let mut group = 0;
        while group < QK_K / 16 {
            let mut sum = 0_i16;
            index = 0;
            while index < 16 {
                sum += i16::from(scratch[block].qs[group * 16 + index]);
                index += 1;
            }
            scratch[block].bsums[group] = sum;
            group += 1;
        }
        block += 1;
    }
}

#[allow(clippy::cast_possible_truncation)]
pub(crate) fn quantize_q8_k_strided(
    rhs: &TensorView<'_>,
    scratch: &mut [Q8KScratch],
    k: usize,
    columns: usize,
    column: usize,
) {
    let blocks = k / QK_K;
    let mut block = 0;
    while block < blocks {
        let base = block * QK_K;
        let mut maximum = 0.0_f32;
        let mut absolute = 0.0_f32;
        let mut index = 0;
        while index < QK_K {
            let value = rhs.read(column + columns * (base + index));
            if value.abs() > absolute {
                absolute = value.abs();
                maximum = value;
            }
            index += 1;
        }
        if absolute == 0.0 {
            scratch[block] = Q8KScratch::ZERO;
            block += 1;
            continue;
        }
        let inverse = -127.0 / maximum;
        scratch[block].d = 1.0 / inverse;
        index = 0;
        while index < QK_K {
            let quantized =
                nearest_int(inverse * rhs.read(column + columns * (base + index))).min(127);
            scratch[block].qs[index] = i8::try_from(quantized).expect("Q8_K value fits i8");
            index += 1;
        }
        let mut group = 0;
        while group < QK_K / 16 {
            let mut sum = 0_i16;
            index = 0;
            while index < 16 {
                sum += i16::from(scratch[block].qs[group * 16 + index]);
                index += 1;
            }
            scratch[block].bsums[group] = sum;
            group += 1;
        }
        block += 1;
    }
}

#[inline]
const fn packed_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

/// Computes the pinned scalar `q4_0` by `q8_0` row dot product.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn dot_q4_0(row: &[u8], rhs: &[Q8_0Scratch], blocks: usize) -> f32 {
    let mut sum = 0.0_f32;
    let mut block = 0;
    while block < blocks {
        let offset = block * Q4_0_BYTES;
        let mut integer = 0_i32;
        let mut index = 0;
        while index < QK_32 / 2 {
            let packed = row[offset + 2 + index];
            integer += (i32::from(packed & 0x0f) - 8) * i32::from(rhs[block].qs[index]);
            integer += (i32::from(packed >> 4) - 8) * i32::from(rhs[block].qs[index + QK_32 / 2]);
            index += 1;
        }
        sum += integer as f32
            * (crate::any::quant::fp16_to_f32(packed_u16(row, offset))
                * crate::any::quant::fp16_to_f32(rhs[block].d));
        block += 1;
    }
    sum
}

/// Computes the pinned scalar `q4_1` by `q8_0` row dot product.
///
/// The affine minimum term follows `detail.hpp:3155-3178` exactly: the
/// packed integer sum is scaled by `d`, while the RHS integer sum is scaled
/// by `m`, and both are finally multiplied by the RHS scale.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn dot_q4_1(row: &[u8], rhs: &[Q8_0Scratch], blocks: usize) -> f32 {
    let mut sum = 0.0_f32;
    let mut block = 0;
    while block < blocks {
        let offset = block * Q4_1_BYTES;
        let mut integer = 0_i32;
        let mut rhs_sum = 0_i32;
        let mut index = 0;
        while index < QK_32 / 2 {
            let packed = row[offset + 4 + index];
            let rhs_low = i32::from(rhs[block].qs[index]);
            let rhs_high = i32::from(rhs[block].qs[index + QK_32 / 2]);
            integer += i32::from(packed & 0x0f) * rhs_low;
            integer += i32::from(packed >> 4) * rhs_high;
            rhs_sum += rhs_low + rhs_high;
            index += 1;
        }
        let rhs_scale = crate::any::quant::fp16_to_f32(rhs[block].d);
        sum += rhs_scale
            * (crate::any::quant::fp16_to_f32(packed_u16(row, offset)) * integer as f32
                + crate::any::quant::fp16_to_f32(packed_u16(row, offset + 2)) * rhs_sum as f32);
        block += 1;
    }
    sum
}

// Preserve the packed q5 scalar product order; fusion changes result bits.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn dot_q5_0(row: &[u8], rhs: &[Q8_0Scratch], blocks: usize) -> f32 {
    let mut sum = 0.0_f32;
    let mut block = 0;
    while block < blocks {
        let offset = block * Q5_0_BYTES;
        let high = u32::from_le_bytes([
            row[offset + 2],
            row[offset + 3],
            row[offset + 4],
            row[offset + 5],
        ]);
        let mut integer = 0_i32;
        let mut index = 0;
        while index < QK_32 / 2 {
            let low_high = (((high >> index) & 1) as u8) << 4;
            let high_high = (((high >> (index + QK_32 / 2)) & 1) as u8) << 4;
            let lhs_low = i32::from((row[offset + 6 + index] & 0x0f) | low_high) - 16;
            let lhs_high = i32::from((row[offset + 6 + index] >> 4) | high_high) - 16;
            integer += lhs_low * i32::from(rhs[block].qs[index]);
            integer += lhs_high * i32::from(rhs[block].qs[index + QK_32 / 2]);
            index += 1;
        }
        sum += integer as f32
            * (crate::any::quant::fp16_to_f32(packed_u16(row, offset))
                * crate::any::quant::fp16_to_f32(rhs[block].d));
        block += 1;
    }
    sum
}

// Preserve the packed q8 scalar product order; fusion changes result bits.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn dot_q8_0(row: &[u8], rhs: &[Q8_0Scratch], blocks: usize) -> f32 {
    let mut sum = 0.0_f32;
    let mut block = 0;
    while block < blocks {
        let offset = block * Q8_0_BYTES;
        let mut integer = 0_i32;
        let mut index = 0;
        while index < QK_32 {
            integer += i32::from(i8::from_ne_bytes([row[offset + 2 + index]]))
                * i32::from(rhs[block].qs[index]);
            index += 1;
        }
        sum += integer as f32
            * (crate::any::quant::fp16_to_f32(packed_u16(row, offset))
                * crate::any::quant::fp16_to_f32(rhs[block].d));
        block += 1;
    }
    sum
}

// Preserve the packed q2 scalar product order; fusion changes result bits.
#[allow(
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    clippy::similar_names
)]
fn dot_q2_k(row: &[u8], rhs: &[Q8KScratch], blocks: usize) -> f32 {
    let mut total = 0.0_f32;
    let mut block = 0;
    while block < blocks {
        let offset = block * Q2_K_BYTES;
        let scales = &row[offset..offset + 16];
        let q = &row[offset + 16..offset + 80];
        let d = crate::any::quant::fp16_to_f32(packed_u16(row, offset + 80));
        let dmin = crate::any::quant::fp16_to_f32(packed_u16(row, offset + 82));
        let mut sum_mins = 0_i32;
        let mut group = 0;
        while group < 16 {
            sum_mins += i32::from(rhs[block].bsums[group]) * i32::from(scales[group] >> 4);
            group += 1;
        }
        let mut sum = 0_i32;
        let mut scale_index = 0;
        let mut chunk = 0;
        while chunk < 2 {
            let mut shift = 0;
            while shift < 8 {
                let scale0 = i32::from(scales[scale_index] & 0x0f);
                let scale1 = i32::from(scales[scale_index + 1] & 0x0f);
                let mut low = 0_i32;
                let mut high = 0_i32;
                let qbase = chunk * 32;
                let mut index = 0;
                while index < 16 {
                    low += i32::from(rhs[block].qs[chunk * 128 + (shift / 2) * 32 + index])
                        * i32::from((q[qbase + index] >> shift) & 3);
                    high += i32::from(rhs[block].qs[chunk * 128 + (shift / 2) * 32 + 16 + index])
                        * i32::from((q[qbase + 16 + index] >> shift) & 3);
                    index += 1;
                }
                sum += scale0 * low + scale1 * high;
                scale_index += 2;
                shift += 2;
            }
            chunk += 1;
        }
        let d_all = rhs[block].d * d;
        let d_min = rhs[block].d * dmin;
        total += d_all * sum as f32 - d_min * sum_mins as f32;
        block += 1;
    }
    total
}

const fn unpack_q3_scales(bytes: &[u8]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut aux = [0_u32; 4];
    aux[0] = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    aux[1] = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    aux[2] = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let temporary = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((temporary >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((temporary >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((temporary & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((temporary >> 2) & KMASK1) << 4);
    let mut scales = [0_i8; 16];
    let mut index = 0;
    while index < 4 {
        let word = aux[index].to_le_bytes();
        scales[index * 4] = i8::from_ne_bytes([word[0]]).wrapping_sub(32);
        scales[index * 4 + 1] = i8::from_ne_bytes([word[1]]).wrapping_sub(32);
        scales[index * 4 + 2] = i8::from_ne_bytes([word[2]]).wrapping_sub(32);
        scales[index * 4 + 3] = i8::from_ne_bytes([word[3]]).wrapping_sub(32);
        index += 1;
    }
    scales
}

// Preserve the packed q3 scalar product order; lane fusion changes result bits.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn dot_q3_k(row: &[u8], rhs: &[Q8KScratch], blocks: usize) -> f32 {
    let mut lane_sums = [0.0_f32; 8];
    let mut block = 0;
    while block < blocks {
        let offset = block * Q3_K_BYTES;
        let hmask = &row[offset..offset + 32];
        let q = &row[offset + 32..offset + 96];
        let scales = unpack_q3_scales(&row[offset + 96..offset + 108]);
        let d = crate::any::quant::fp16_to_f32(packed_u16(row, offset + 108));
        let mut accumulators = [0_i32; 8];
        let mut group = 0;
        while group < 16 {
            let scale = i32::from(scales[group]);
            let qbase = (group / 8) * 32;
            let rhsbase = group * 16;
            let mask_shift = group / 2;
            let mask = 1_u8 << mask_shift;
            let mut index = 0;
            while index < 16 {
                let half = (group % 2) * 16;
                let qbyte = q[qbase + half + index];
                let quantized = i32::from((qbyte >> (((group / 2) % 4) * 2)) & 3)
                    - if hmask[half + index] & mask == 0 {
                        4
                    } else {
                        0
                    };
                accumulators[index & 7] +=
                    scale * quantized * i32::from(rhs[block].qs[rhsbase + index]);
                index += 1;
            }
            group += 1;
        }
        let scale = rhs[block].d * d;
        let mut lane = 0;
        while lane < 8 {
            lane_sums[lane] += scale * accumulators[lane] as f32;
            lane += 1;
        }
        block += 1;
    }
    lane_sums.into_iter().sum()
}

// Preserve the packed q4 scalar product order; lane fusion changes result bits.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn dot_q4_k(row: &[u8], rhs: &[Q8KScratch], blocks: usize) -> f32 {
    let mut lane_sums = [0.0_f32; 8];
    let mut sum = 0.0_f32;
    let mut block = 0;
    while block < blocks {
        let offset = block * Q4_K_BYTES;
        let mut unpacked_q4 = [0_i8; QK_K];
        let mut group = 0;
        while group < QK_K / 64 {
            let q4_offset = group * 32;
            let output_offset = group * 64;
            let mut lane = 0;
            while lane < 32 {
                let packed = row[offset + 16 + q4_offset + lane];
                unpacked_q4[output_offset + lane] = i8::from_ne_bytes([packed & 0x0f]);
                unpacked_q4[output_offset + 32 + lane] = i8::from_ne_bytes([packed >> 4]);
                lane += 1;
            }
            group += 1;
        }

        let scales = &row[offset + 4..offset + 16];
        let mut unpacked_scales = [0_u8; 12];
        let mut minimums = [0_u8; 8];
        let mut lane = 0;
        while lane < 4 {
            let scale_word0 = scales[lane];
            let scale_word1 = scales[4 + lane];
            let scale_word2 = scales[8 + lane];
            unpacked_scales[lane] = scale_word0 & 0x3f;
            unpacked_scales[4 + lane] = (scale_word2 & 0x0f) | (((scale_word0 >> 6) & 0x03) << 4);
            unpacked_scales[8 + lane] = scale_word1 & 0x3f;
            minimums[lane] = scale_word1 & 0x3f;
            minimums[4 + lane] = ((scale_word2 >> 4) & 0x0f) | (((scale_word1 >> 6) & 0x03) << 4);
            lane += 1;
        }

        let mut minimum_sum = 0_i32;
        group = 0;
        while group < QK_K / 16 {
            minimum_sum += i32::from(rhs[block].bsums[group]) * i32::from(minimums[group / 2]);
            group += 1;
        }

        let mut accumulators = [0_i32; 8];
        group = 0;
        while group < QK_K / 32 {
            let scale = i32::from(unpacked_scales[group]);
            let value_offset = group * 32;
            lane = 0;
            while lane < 32 {
                let lhs_value = i32::from(unpacked_q4[value_offset + lane]);
                let rhs_value = i32::from(rhs[block].qs[value_offset + lane]);
                accumulators[lane % 8] += scale * (lhs_value * rhs_value);
                lane += 1;
            }
            group += 1;
        }

        let lhs_scale = crate::any::quant::fp16_to_f32(packed_u16(row, offset));
        let lhs_minimum = crate::any::quant::fp16_to_f32(packed_u16(row, offset + 2));
        let d = lhs_scale * rhs[block].d;
        lane = 0;
        while lane < 8 {
            lane_sums[lane] += d * accumulators[lane] as f32;
            lane += 1;
        }
        let dmin = lhs_minimum * rhs[block].d;
        sum -= dmin * minimum_sum as f32;
        block += 1;
    }
    for lane in lane_sums {
        sum += lane;
    }
    sum
}

// Preserve the packed q6 scalar product order; fusion changes result bits.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn dot_q6_k(row: &[u8], rhs: &[Q8KScratch], blocks: usize) -> f32 {
    let mut lane_sums = [0.0_f32; 8];
    let mut block = 0;
    while block < blocks {
        let offset = block * Q6_K_BYTES;
        let ql = &row[offset..offset + 128];
        let qh = &row[offset + 128..offset + 192];
        let scales = &row[offset + 192..offset + 208];
        let d = crate::any::quant::fp16_to_f32(packed_u16(row, offset + 208));
        let mut decoded = [0_i8; QK_K];
        let mut chunk = 0;
        while chunk < 2 {
            let mut index = 0;
            while index < 32 {
                let high = qh[chunk * 32 + index];
                let q1 = i8::try_from((ql[chunk * 64 + index] & 0x0f) | ((high & 3) << 4))
                    .expect("q6 value fits i8")
                    - 32;
                let q2 =
                    i8::try_from((ql[chunk * 64 + index + 32] & 0x0f) | (((high >> 2) & 3) << 4))
                        .expect("q6 value fits i8")
                        - 32;
                let q3 =
                    i8::try_from(((ql[chunk * 64 + index] >> 4) & 0x0f) | (((high >> 4) & 3) << 4))
                        .expect("q6 value fits i8")
                        - 32;
                let q4 = i8::try_from(
                    ((ql[chunk * 64 + index + 32] >> 4) & 0x0f) | (((high >> 6) & 3) << 4),
                )
                .expect("q6 value fits i8")
                    - 32;
                decoded[chunk * 128 + index] = q1;
                decoded[chunk * 128 + index + 32] = q2;
                decoded[chunk * 128 + index + 64] = q3;
                decoded[chunk * 128 + index + 96] = q4;
                index += 1;
            }
            chunk += 1;
        }
        let mut sums = [0_i32; 8];
        let mut group = 0;
        while group < 16 {
            let scale = i32::from(i8::from_ne_bytes([scales[group]]));
            let base = group * 16;
            let mut lane = 0;
            while lane < 8 {
                sums[lane] +=
                    scale * i32::from(decoded[base + lane]) * i32::from(rhs[block].qs[base + lane]);
                lane += 1;
            }
            lane = 0;
            while lane < 8 {
                sums[lane] += scale
                    * i32::from(decoded[base + 8 + lane])
                    * i32::from(rhs[block].qs[base + 8 + lane]);
                lane += 1;
            }
            group += 1;
        }
        let scale = rhs[block].d * d;
        let mut lane = 0;
        while lane < 8 {
            lane_sums[lane] += scale * sums[lane] as f32;
            lane += 1;
        }
        block += 1;
    }
    let mut total = 0.0_f32;
    for lane in lane_sums {
        total += lane;
    }
    total
}

fn run_argmax_q4_0(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &mut TensorViewMut<'_>,
    scratch: &mut [Q8_0Scratch],
) -> i32 {
    let shape = lhs.layout().ne();
    let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
    let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
    let blocks = k / QK_32;
    quantize_q8_0(rhs, scratch, k);
    let row_bytes = blocks * Q4_0_BYTES;
    let mut best = f32::NEG_INFINITY;
    let mut best_index = 0_i32;
    let mut row = 0;
    while row < rows {
        let value = dot_q4_0(lhs.row(row, row_bytes), scratch, blocks);
        if value > best || row == 0 {
            best = value;
            best_index = i32::try_from(row).expect("guard-proven row fits i32");
        }
        row += 1;
    }
    destination.write(0, best);
    best_index
}

fn run_mul_mat_q4_0(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &mut TensorViewMut<'_>,
    scratch: &mut [Q8_0Scratch],
) {
    let shape = lhs.layout().ne();
    let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
    let rows = usize::try_from(shape[1]).expect("guard-proven m fits usize");
    let columns = usize::try_from(rhs.layout().ne()[0]).expect("guard-proven n fits usize");
    let blocks = k / QK_32;
    let row_bytes = blocks * Q4_0_BYTES;
    let mut column = 0;
    while column < columns {
        quantize_q8_0_strided(rhs, scratch, k, columns, column);
        let mut row = 0;
        while row < rows {
            let value = dot_q4_0(lhs.row(row, row_bytes), scratch, blocks);
            destination.write(column + columns * row, value);
            row += 1;
        }
        column += 1;
    }
}

fn run_mul_mat_q4_1(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &mut TensorViewMut<'_>,
    scratch: &mut [Q8_0Scratch],
) {
    let shape = lhs.layout().ne();
    let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
    let rows = usize::try_from(shape[1]).expect("guard-proven m fits usize");
    let columns = usize::try_from(rhs.layout().ne()[0]).expect("guard-proven n fits usize");
    let blocks = k / QK_32;
    let row_bytes = blocks * Q4_1_BYTES;
    let mut column = 0;
    while column < columns {
        quantize_q8_0_strided(rhs, scratch, k, columns, column);
        let mut row = 0;
        while row < rows {
            let value = dot_q4_1(lhs.row(row, row_bytes), scratch, blocks);
            destination.write(column + columns * row, value);
            row += 1;
        }
        column += 1;
    }
}

fn run_mul_mat_q5(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &mut TensorViewMut<'_>,
    scratch: &mut [Q8_0Scratch],
) {
    let shape = lhs.layout().ne();
    let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
    let rows = usize::try_from(shape[1]).expect("guard-proven m fits usize");
    let columns = usize::try_from(rhs.layout().ne()[0]).expect("guard-proven n fits usize");
    let blocks = k / QK_32;
    let row_bytes = blocks * Q5_0_BYTES;
    let mut column = 0;
    while column < columns {
        quantize_q8_0_strided(rhs, scratch, k, columns, column);
        let mut row = 0;
        while row < rows {
            let value = dot_q5_0(lhs.row(row, row_bytes), scratch, blocks);
            destination.write(column + columns * row, value);
            row += 1;
        }
        column += 1;
    }
}

fn run_mul_mat_q8_0(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &mut TensorViewMut<'_>,
    scratch: &mut [Q8_0Scratch],
) {
    let shape = lhs.layout().ne();
    let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
    let rows = usize::try_from(shape[1]).expect("guard-proven m fits usize");
    let columns = usize::try_from(rhs.layout().ne()[0]).expect("guard-proven n fits usize");
    let blocks = k / QK_32;
    let row_bytes = blocks * Q8_0_BYTES;
    let mut column = 0;
    while column < columns {
        quantize_q8_0_strided(rhs, scratch, k, columns, column);
        let mut row = 0;
        while row < rows {
            let value = dot_q8_0(lhs.row(row, row_bytes), scratch, blocks);
            destination.write(column + columns * row, value);
            row += 1;
        }
        column += 1;
    }
}

fn run_mul_mat_q2_k(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &mut TensorViewMut<'_>,
    scratch: &mut [Q8KScratch],
) {
    let shape = lhs.layout().ne();
    let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
    let rows = usize::try_from(shape[1]).expect("guard-proven m fits usize");
    let columns = usize::try_from(rhs.layout().ne()[0]).expect("guard-proven n fits usize");
    let blocks = k / QK_K;
    let row_bytes = blocks * Q2_K_BYTES;
    let mut column = 0;
    while column < columns {
        quantize_q8_k_strided(rhs, scratch, k, columns, column);
        let mut row = 0;
        while row < rows {
            let value = dot_q2_k(lhs.row(row, row_bytes), scratch, blocks);
            destination.write(column + columns * row, value);
            row += 1;
        }
        column += 1;
    }
}

fn run_mul_mat_q3_k(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &mut TensorViewMut<'_>,
    scratch: &mut [Q8KScratch],
) {
    let shape = lhs.layout().ne();
    let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
    let rows = usize::try_from(shape[1]).expect("guard-proven m fits usize");
    let columns = usize::try_from(rhs.layout().ne()[0]).expect("guard-proven n fits usize");
    let blocks = k / QK_K;
    let row_bytes = blocks * Q3_K_BYTES;
    let mut column = 0;
    while column < columns {
        quantize_q8_k_strided(rhs, scratch, k, columns, column);
        let mut row = 0;
        while row < rows {
            let value = dot_q3_k(lhs.row(row, row_bytes), scratch, blocks);
            destination.write(column + columns * row, value);
            row += 1;
        }
        column += 1;
    }
}

fn run_mul_mat_q4_k(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &mut TensorViewMut<'_>,
    scratch: &mut [Q8KScratch],
) {
    let shape = lhs.layout().ne();
    let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
    let rows = usize::try_from(shape[1]).expect("guard-proven m fits usize");
    let columns = usize::try_from(rhs.layout().ne()[0]).expect("guard-proven n fits usize");
    let blocks = k / QK_K;
    let row_bytes = blocks * Q4_K_BYTES;
    let mut column = 0;
    while column < columns {
        quantize_q8_k_strided(rhs, scratch, k, columns, column);
        let mut row = 0;
        while row < rows {
            let value = dot_q4_k(lhs.row(row, row_bytes), scratch, blocks);
            destination.write(column + columns * row, value);
            row += 1;
        }
        column += 1;
    }
}

fn run_mul_mat_q6_k(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &mut TensorViewMut<'_>,
    scratch: &mut [Q8KScratch],
) {
    let shape = lhs.layout().ne();
    let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
    let rows = usize::try_from(shape[1]).expect("guard-proven m fits usize");
    let columns = usize::try_from(rhs.layout().ne()[0]).expect("guard-proven n fits usize");
    let blocks = k / QK_K;
    let row_bytes = blocks * Q6_K_BYTES;
    let mut column = 0;
    while column < columns {
        quantize_q8_k_strided(rhs, scratch, k, columns, column);
        let mut row = 0;
        while row < rows {
            let value = dot_q6_k(lhs.row(row, row_bytes), scratch, blocks);
            destination.write(column + columns * row, value);
            row += 1;
        }
        column += 1;
    }
}

fn run_argmax_q4_1(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &mut TensorViewMut<'_>,
    scratch: &mut [Q8_0Scratch],
) -> i32 {
    let shape = lhs.layout().ne();
    let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
    let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
    let blocks = k / QK_32;
    quantize_q8_0(rhs, scratch, k);
    let row_bytes = blocks * Q4_1_BYTES;
    let mut best = f32::NEG_INFINITY;
    let mut best_index = 0_i32;
    let mut row = 0;
    while row < rows {
        let value = dot_q4_1(lhs.row(row, row_bytes), scratch, blocks);
        if value > best || row == 0 {
            best = value;
            best_index = i32::try_from(row).expect("guard-proven row fits i32");
        }
        row += 1;
    }
    destination.write(0, best);
    best_index
}

fn run_argmax_q5_0(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &mut TensorViewMut<'_>,
    scratch: &mut [Q8_0Scratch],
) -> i32 {
    let shape = lhs.layout().ne();
    let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
    let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
    let blocks = k / QK_32;
    quantize_q8_0(rhs, scratch, k);
    let row_bytes = blocks * Q5_0_BYTES;
    let mut best = f32::NEG_INFINITY;
    let mut best_index = 0_i32;
    let mut row = 0;
    while row < rows {
        let value = dot_q5_0(lhs.row(row, row_bytes), scratch, blocks);
        if value > best || row == 0 {
            best = value;
            best_index = i32::try_from(row).expect("guard-proven row fits i32");
        }
        row += 1;
    }
    destination.write(0, best);
    best_index
}

fn run_argmax_q8_0(
    lhs: &QuantizedView<'_>,
    rhs: &TensorView<'_>,
    destination: &mut TensorViewMut<'_>,
    scratch: &mut [Q8_0Scratch],
) -> i32 {
    let shape = lhs.layout().ne();
    let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
    let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
    let blocks = k / QK_32;
    quantize_q8_0(rhs, scratch, k);
    let row_bytes = blocks * Q8_0_BYTES;
    let mut best = f32::NEG_INFINITY;
    let mut best_index = 0_i32;
    let mut row = 0;
    while row < rows {
        let value = dot_q8_0(lhs.row(row, row_bytes), scratch, blocks);
        if value > best || row == 0 {
            best = value;
            best_index = i32::try_from(row).expect("guard-proven row fits i32");
        }
        row += 1;
    }
    destination.write(0, best);
    best_index
}

macro_rules! define_k_argmax_runner {
    ($name:ident, $bytes:expr, $dot:ident) => {
        fn $name(
            lhs: &QuantizedView<'_>,
            rhs: &TensorView<'_>,
            destination: &mut TensorViewMut<'_>,
            scratch: &mut [Q8KScratch],
        ) -> i32 {
            let shape = lhs.layout().ne();
            let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
            let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
            let blocks = k / QK_K;
            quantize_q8_k(rhs, scratch, k);
            let row_bytes = blocks * $bytes;
            let mut best = f32::NEG_INFINITY;
            let mut best_index = 0_i32;
            let mut row = 0;
            while row < rows {
                let value = $dot(lhs.row(row, row_bytes), scratch, blocks);
                if value > best || row == 0 {
                    best = value;
                    best_index = i32::try_from(row).expect("guard-proven row fits i32");
                }
                row += 1;
            }
            destination.write(0, best);
            best_index
        }
    };
}

define_k_argmax_runner!(run_argmax_q2_k, Q2_K_BYTES, dot_q2_k);
define_k_argmax_runner!(run_argmax_q3_k, Q3_K_BYTES, dot_q3_k);
define_k_argmax_runner!(run_argmax_q4_k, Q4_K_BYTES, dot_q4_k);
define_k_argmax_runner!(run_argmax_q6_k, Q6_K_BYTES, dot_q6_k);

impl MatmulMachineStateMachineContext for MatmulContext {
    fn guard_mul_mat_valid(&self, event: &MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event) && shape_valid(&event.event))
    }

    fn guard_mul_mat_shape_mismatch(&self, event: &MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event) && !shape_valid(&event.event))
    }

    fn guard_mul_mat_invalid_view(&self, event: &MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(!views_valid(&event.event))
    }

    fn guard_mul_mat_q4_valid(&self, event: &MatmulQ4Runtime<'_>) -> Result<bool, ()> {
        Ok(q4_views_valid(&event.event) && q4_shape_valid(&event.event))
    }

    fn guard_mul_mat_q4_shape_mismatch(&self, event: &MatmulQ4Runtime<'_>) -> Result<bool, ()> {
        Ok(q4_views_valid(&event.event) && !q4_shape_valid(&event.event))
    }

    fn guard_mul_mat_q4_invalid_view(&self, event: &MatmulQ4Runtime<'_>) -> Result<bool, ()> {
        Ok(!q4_views_valid(&event.event))
    }

    fn guard_mul_mat_q4_1_valid(&self, event: &MatmulQ4_1Runtime<'_>) -> Result<bool, ()> {
        Ok(q4_1_views_valid(&event.event) && q4_1_shape_valid(&event.event))
    }

    fn guard_mul_mat_q4_1_shape_mismatch(&self, event: &MatmulQ4_1Runtime<'_>) -> Result<bool, ()> {
        Ok(q4_1_views_valid(&event.event) && !q4_1_shape_valid(&event.event))
    }

    fn guard_mul_mat_q4_1_invalid_view(&self, event: &MatmulQ4_1Runtime<'_>) -> Result<bool, ()> {
        Ok(!q4_1_views_valid(&event.event))
    }

    fn guard_mul_mat_q5_valid(&self, event: &MatmulQ5Runtime<'_>) -> Result<bool, ()> {
        Ok(q5_views_valid(&event.event) && q5_shape_valid(&event.event))
    }

    fn guard_mul_mat_q5_shape_mismatch(&self, event: &MatmulQ5Runtime<'_>) -> Result<bool, ()> {
        Ok(q5_views_valid(&event.event) && !q5_shape_valid(&event.event))
    }

    fn guard_mul_mat_q5_invalid_view(&self, event: &MatmulQ5Runtime<'_>) -> Result<bool, ()> {
        Ok(!q5_views_valid(&event.event))
    }

    fn guard_mul_mat_q8_valid(&self, event: &MatmulQ8Runtime<'_>) -> Result<bool, ()> {
        Ok(q8_views_valid(&event.event) && q8_shape_valid(&event.event))
    }

    fn guard_mul_mat_q8_shape_mismatch(&self, event: &MatmulQ8Runtime<'_>) -> Result<bool, ()> {
        Ok(q8_views_valid(&event.event) && !q8_shape_valid(&event.event))
    }

    fn guard_mul_mat_q8_invalid_view(&self, event: &MatmulQ8Runtime<'_>) -> Result<bool, ()> {
        Ok(!q8_views_valid(&event.event))
    }

    fn guard_mul_mat_q2_k_valid(&self, event: &MatmulQ2KRuntime<'_>) -> Result<bool, ()> {
        Ok(q2_k_views_valid(&event.event) && q2_k_shape_valid(&event.event))
    }

    fn guard_mul_mat_q2_k_shape_mismatch(&self, event: &MatmulQ2KRuntime<'_>) -> Result<bool, ()> {
        Ok(q2_k_views_valid(&event.event) && !q2_k_shape_valid(&event.event))
    }

    fn guard_mul_mat_q2_k_invalid_view(&self, event: &MatmulQ2KRuntime<'_>) -> Result<bool, ()> {
        Ok(!q2_k_views_valid(&event.event))
    }

    fn guard_mul_mat_q3_k_valid(&self, event: &MatmulQ3KRuntime<'_>) -> Result<bool, ()> {
        Ok(q3_k_views_valid(&event.event) && q3_k_shape_valid(&event.event))
    }

    fn guard_mul_mat_q3_k_shape_mismatch(&self, event: &MatmulQ3KRuntime<'_>) -> Result<bool, ()> {
        Ok(q3_k_views_valid(&event.event) && !q3_k_shape_valid(&event.event))
    }

    fn guard_mul_mat_q3_k_invalid_view(&self, event: &MatmulQ3KRuntime<'_>) -> Result<bool, ()> {
        Ok(!q3_k_views_valid(&event.event))
    }

    fn guard_mul_mat_q4_k_valid(&self, event: &MatmulQ4KRuntime<'_>) -> Result<bool, ()> {
        Ok(q4_k_views_valid(&event.event) && q4_k_shape_valid(&event.event))
    }

    fn guard_mul_mat_q4_k_shape_mismatch(&self, event: &MatmulQ4KRuntime<'_>) -> Result<bool, ()> {
        Ok(q4_k_views_valid(&event.event) && !q4_k_shape_valid(&event.event))
    }

    fn guard_mul_mat_q4_k_invalid_view(&self, event: &MatmulQ4KRuntime<'_>) -> Result<bool, ()> {
        Ok(!q4_k_views_valid(&event.event))
    }

    fn guard_mul_mat_q6_k_valid(&self, event: &MatmulQ6KRuntime<'_>) -> Result<bool, ()> {
        Ok(q6_k_views_valid(&event.event) && q6_k_shape_valid(&event.event))
    }

    fn guard_mul_mat_q6_k_shape_mismatch(&self, event: &MatmulQ6KRuntime<'_>) -> Result<bool, ()> {
        Ok(q6_k_views_valid(&event.event) && !q6_k_shape_valid(&event.event))
    }

    fn guard_mul_mat_q6_k_invalid_view(&self, event: &MatmulQ6KRuntime<'_>) -> Result<bool, ()> {
        Ok(!q6_k_views_valid(&event.event))
    }

    fn effect_mul_mat_q4_execute(&mut self, event: MatmulQ4Runtime<'_>) -> Result<(), ()> {
        let OpMulMatQ4_0 {
            lhs,
            rhs,
            mut destination,
        } = event.event;
        run_mul_mat_q4_0(&lhs, &rhs, &mut destination, &mut self.q8_0);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mul_mat_q4_shape_reject(&mut self, event: MatmulQ4Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::ShapeMismatch));
        Ok(())
    }

    fn effect_mul_mat_q4_view_reject(&mut self, event: MatmulQ4Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::InvalidView));
        Ok(())
    }

    fn effect_mul_mat_q4_1_execute(&mut self, event: MatmulQ4_1Runtime<'_>) -> Result<(), ()> {
        let OpMulMatQ4_1 {
            lhs,
            rhs,
            mut destination,
        } = event.event;
        run_mul_mat_q4_1(&lhs, &rhs, &mut destination, &mut self.q8_0);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mul_mat_q4_1_shape_reject(&mut self, event: MatmulQ4_1Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::ShapeMismatch));
        Ok(())
    }

    fn effect_mul_mat_q4_1_view_reject(&mut self, event: MatmulQ4_1Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::InvalidView));
        Ok(())
    }

    fn effect_mul_mat_q5_execute(&mut self, event: MatmulQ5Runtime<'_>) -> Result<(), ()> {
        let OpMulMatQ5_0 {
            lhs,
            rhs,
            mut destination,
        } = event.event;
        run_mul_mat_q5(&lhs, &rhs, &mut destination, &mut self.q8_0);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mul_mat_q5_shape_reject(&mut self, event: MatmulQ5Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::ShapeMismatch));
        Ok(())
    }

    fn effect_mul_mat_q5_view_reject(&mut self, event: MatmulQ5Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::InvalidView));
        Ok(())
    }

    fn effect_mul_mat_q8_execute(&mut self, event: MatmulQ8Runtime<'_>) -> Result<(), ()> {
        let OpMulMatQ8_0 {
            lhs,
            rhs,
            mut destination,
        } = event.event;
        run_mul_mat_q8_0(&lhs, &rhs, &mut destination, &mut self.q8_0);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mul_mat_q8_shape_reject(&mut self, event: MatmulQ8Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::ShapeMismatch));
        Ok(())
    }

    fn effect_mul_mat_q8_view_reject(&mut self, event: MatmulQ8Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::InvalidView));
        Ok(())
    }

    fn effect_mul_mat_q2_k_execute(&mut self, event: MatmulQ2KRuntime<'_>) -> Result<(), ()> {
        let OpMulMatQ2K {
            lhs,
            rhs,
            mut destination,
        } = event.event;
        run_mul_mat_q2_k(&lhs, &rhs, &mut destination, &mut self.q8_k);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mul_mat_q2_k_shape_reject(&mut self, event: MatmulQ2KRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::ShapeMismatch));
        Ok(())
    }

    fn effect_mul_mat_q2_k_view_reject(&mut self, event: MatmulQ2KRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::InvalidView));
        Ok(())
    }

    fn effect_mul_mat_q3_k_execute(&mut self, event: MatmulQ3KRuntime<'_>) -> Result<(), ()> {
        let OpMulMatQ3K {
            lhs,
            rhs,
            mut destination,
        } = event.event;
        run_mul_mat_q3_k(&lhs, &rhs, &mut destination, &mut self.q8_k);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mul_mat_q3_k_shape_reject(&mut self, event: MatmulQ3KRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::ShapeMismatch));
        Ok(())
    }

    fn effect_mul_mat_q3_k_view_reject(&mut self, event: MatmulQ3KRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::InvalidView));
        Ok(())
    }

    fn effect_mul_mat_q4_k_execute(&mut self, event: MatmulQ4KRuntime<'_>) -> Result<(), ()> {
        let OpMulMatQ4K {
            lhs,
            rhs,
            mut destination,
        } = event.event;
        run_mul_mat_q4_k(&lhs, &rhs, &mut destination, &mut self.q8_k);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mul_mat_q4_k_shape_reject(&mut self, event: MatmulQ4KRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::ShapeMismatch));
        Ok(())
    }

    fn effect_mul_mat_q4_k_view_reject(&mut self, event: MatmulQ4KRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::InvalidView));
        Ok(())
    }

    fn effect_mul_mat_q6_k_execute(&mut self, event: MatmulQ6KRuntime<'_>) -> Result<(), ()> {
        let OpMulMatQ6K {
            lhs,
            rhs,
            mut destination,
        } = event.event;
        run_mul_mat_q6_k(&lhs, &rhs, &mut destination, &mut self.q8_k);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mul_mat_q6_k_shape_reject(&mut self, event: MatmulQ6KRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::ShapeMismatch));
        Ok(())
    }

    fn effect_mul_mat_q6_k_view_reject(&mut self, event: MatmulQ6KRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::InvalidView));
        Ok(())
    }

    // Preserve the pinned dense matmul accumulation order; fusion changes bits.
    #[allow(clippy::suboptimal_flops)]
    fn effect_mul_mat_execute(&mut self, mut event: MatmulRuntime<'_>) -> Result<(), ()> {
        let lhs_shape = event.event.lhs.layout().ne();
        let rhs_shape = event.event.rhs.layout().ne();
        let k = usize::try_from(lhs_shape[0]).expect("guard-proven k fits usize");
        let rows = usize::try_from(lhs_shape[1]).expect("guard-proven m fits usize");
        let columns = usize::try_from(rhs_shape[0]).expect("guard-proven n fits usize");
        let mut row = 0;
        while row < rows {
            let mut column = 0;
            while column < columns {
                let mut inner = 0;
                let mut accumulator = 0.0_f32;
                while inner < k {
                    let lhs_ordinal = inner
                        .checked_add(
                            k.checked_mul(row)
                                .expect("guard-proven lhs ordinal fits usize"),
                        )
                        .expect("guard-proven lhs ordinal fits usize");
                    let rhs_ordinal = column
                        .checked_add(
                            columns
                                .checked_mul(inner)
                                .expect("guard-proven rhs ordinal fits usize"),
                        )
                        .expect("guard-proven rhs ordinal fits usize");
                    accumulator +=
                        event.event.lhs.read(lhs_ordinal) * event.event.rhs.read(rhs_ordinal);
                    inner += 1;
                }
                let destination_ordinal = column
                    .checked_add(
                        columns
                            .checked_mul(row)
                            .expect("guard-proven destination ordinal fits usize"),
                    )
                    .expect("guard-proven destination ordinal fits usize");
                event
                    .event
                    .destination
                    .write(destination_ordinal, accumulator);
                column += 1;
            }
            row += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_mul_mat_shape_reject(&mut self, event: MatmulRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::ShapeMismatch));
        Ok(())
    }

    fn effect_mul_mat_view_reject(&mut self, event: MatmulRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::InvalidView));
        Ok(())
    }

    fn effect_mul_mat_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::UnexpectedEvent));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

impl MatmulArgmaxMachineStateMachineContext for MatmulArgmaxContext {
    fn guard_mul_mat_argmax_valid(&self, event: &MatmulArgmaxRuntime<'_>) -> Result<bool, ()> {
        Ok(argmax_views_valid(&event.event) && argmax_shape_valid(&event.event))
    }

    fn guard_mul_mat_argmax_shape_mismatch(
        &self,
        event: &MatmulArgmaxRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(argmax_views_valid(&event.event) && !argmax_shape_valid(&event.event))
    }

    fn guard_mul_mat_argmax_invalid_view(
        &self,
        event: &MatmulArgmaxRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!argmax_views_valid(&event.event))
    }

    impl_quantized_argmax_routes!(
        guard_mul_mat_argmax_q4_0_valid,
        guard_mul_mat_argmax_q4_0_shape_mismatch,
        guard_mul_mat_argmax_q4_0_invalid_view,
        effect_mul_mat_argmax_q4_0_execute,
        effect_mul_mat_argmax_shape_reject_q4_0,
        effect_mul_mat_argmax_view_reject_q4_0,
        MatmulArgmaxQ4_0Runtime,
        Q4_0_CODE,
        QK_32,
        Q4_0_BYTES,
        run_argmax_q4_0,
        q8_0
    );
    impl_quantized_argmax_routes!(
        guard_mul_mat_argmax_q4_1_valid,
        guard_mul_mat_argmax_q4_1_shape_mismatch,
        guard_mul_mat_argmax_q4_1_invalid_view,
        effect_mul_mat_argmax_q4_1_execute,
        effect_mul_mat_argmax_shape_reject_q4_1,
        effect_mul_mat_argmax_view_reject_q4_1,
        MatmulArgmaxQ4_1Runtime,
        Q4_1_CODE,
        QK_32,
        Q4_1_BYTES,
        run_argmax_q4_1,
        q8_0
    );

    impl_quantized_argmax_routes!(
        guard_mul_mat_argmax_q5_0_valid,
        guard_mul_mat_argmax_q5_0_shape_mismatch,
        guard_mul_mat_argmax_q5_0_invalid_view,
        effect_mul_mat_argmax_q5_0_execute,
        effect_mul_mat_argmax_shape_reject_q5_0,
        effect_mul_mat_argmax_view_reject_q5_0,
        MatmulArgmaxQ5_0Runtime,
        Q5_0_CODE,
        QK_32,
        Q5_0_BYTES,
        run_argmax_q5_0,
        q8_0
    );
    impl_quantized_argmax_routes!(
        guard_mul_mat_argmax_q8_0_valid,
        guard_mul_mat_argmax_q8_0_shape_mismatch,
        guard_mul_mat_argmax_q8_0_invalid_view,
        effect_mul_mat_argmax_q8_0_execute,
        effect_mul_mat_argmax_shape_reject_q8_0,
        effect_mul_mat_argmax_view_reject_q8_0,
        MatmulArgmaxQ8_0Runtime,
        Q8_0_CODE,
        QK_32,
        Q8_0_BYTES,
        run_argmax_q8_0,
        q8_0
    );
    impl_quantized_argmax_routes!(
        guard_mul_mat_argmax_q2_k_valid,
        guard_mul_mat_argmax_q2_k_shape_mismatch,
        guard_mul_mat_argmax_q2_k_invalid_view,
        effect_mul_mat_argmax_q2_k_execute,
        effect_mul_mat_argmax_shape_reject_q2_k,
        effect_mul_mat_argmax_view_reject_q2_k,
        MatmulArgmaxQ2KRuntime,
        Q2_K_CODE,
        QK_K,
        Q2_K_BYTES,
        run_argmax_q2_k,
        q8_k
    );
    impl_quantized_argmax_routes!(
        guard_mul_mat_argmax_q3_k_valid,
        guard_mul_mat_argmax_q3_k_shape_mismatch,
        guard_mul_mat_argmax_q3_k_invalid_view,
        effect_mul_mat_argmax_q3_k_execute,
        effect_mul_mat_argmax_shape_reject_q3_k,
        effect_mul_mat_argmax_view_reject_q3_k,
        MatmulArgmaxQ3KRuntime,
        Q3_K_CODE,
        QK_K,
        Q3_K_BYTES,
        run_argmax_q3_k,
        q8_k
    );
    impl_quantized_argmax_routes!(
        guard_mul_mat_argmax_q4_k_valid,
        guard_mul_mat_argmax_q4_k_shape_mismatch,
        guard_mul_mat_argmax_q4_k_invalid_view,
        effect_mul_mat_argmax_q4_k_execute,
        effect_mul_mat_argmax_shape_reject_q4_k,
        effect_mul_mat_argmax_view_reject_q4_k,
        MatmulArgmaxQ4KRuntime,
        Q4_K_CODE,
        QK_K,
        Q4_K_BYTES,
        run_argmax_q4_k,
        q8_k
    );
    impl_quantized_argmax_routes!(
        guard_mul_mat_argmax_q6_k_valid,
        guard_mul_mat_argmax_q6_k_shape_mismatch,
        guard_mul_mat_argmax_q6_k_invalid_view,
        effect_mul_mat_argmax_q6_k_execute,
        effect_mul_mat_argmax_shape_reject_q6_k,
        effect_mul_mat_argmax_view_reject_q6_k,
        MatmulArgmaxQ6KRuntime,
        Q6_K_CODE,
        QK_K,
        Q6_K_BYTES,
        run_argmax_q6_k,
        q8_k
    );

    // Preserve the pinned argmax accumulation order; fusion changes result bits.
    #[allow(clippy::suboptimal_flops)]
    fn effect_mul_mat_argmax_execute(
        &mut self,
        mut event: MatmulArgmaxRuntime<'_>,
    ) -> Result<(), ()> {
        let lhs_shape = event.event.lhs.layout().ne();
        let k = usize::try_from(lhs_shape[0]).expect("guard-proven k fits usize");
        let rows = usize::try_from(lhs_shape[1]).expect("guard-proven m fits usize");
        let mut best_index = 0_i32;
        let mut best_value = f32::NEG_INFINITY;
        let mut row = 0;
        while row < rows {
            let mut inner = 0;
            let mut accumulator = 0.0_f32;
            while inner < k {
                let lhs_ordinal = inner
                    .checked_add(
                        k.checked_mul(row)
                            .expect("guard-proven lhs ordinal fits usize"),
                    )
                    .expect("guard-proven lhs ordinal fits usize");
                accumulator += event.event.lhs.read(lhs_ordinal) * event.event.rhs.read(inner);
                inner += 1;
            }
            if accumulator > best_value || row == 0 {
                best_value = accumulator;
                best_index = i32::try_from(row).expect("guard-proven row fits i32");
            }
            row += 1;
        }
        event.event.destination.write(0, best_value);
        event.result.set(Ok(best_index));
        Ok(())
    }

    fn effect_mul_mat_argmax_shape_reject(
        &mut self,
        event: MatmulArgmaxRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(MatmulError::ShapeMismatch));
        Ok(())
    }

    fn effect_mul_mat_argmax_view_reject(
        &mut self,
        event: MatmulArgmaxRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(MatmulError::InvalidView));
        Ok(())
    }

    fn effect_mul_mat_argmax_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::UnexpectedEvent));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::f32_to_f16;

    #[test]
    fn pinned_fp32_to_fp16_preserves_finite_subnormal_boundary() {
        // This is the smallest finite scale boundary found by comparing the
        // current helper with detail.hpp:372-392. The pinned conversion rounds
        // it to the representable half subnormal 0x0001.
        assert_eq!(f32_to_f16(2.983_142_6e-8), 0x0001);
    }

    #[test]
    fn pinned_fp32_to_fp16_covers_sign_normal_and_subnormal_scales() {
        assert_eq!(f32_to_f16(-2.983_142_6e-8), 0x8001);
        assert_eq!(f32_to_f16(5.960_464_5e-8), 0x0001);
        assert_eq!(f32_to_f16(1.0e-7), 0x0002);
        assert_eq!(f32_to_f16(-1.0e-7), 0x8002);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-1.0), 0xbc00);
        assert_eq!(f32_to_f16(65_504.0), 0x7bff);
    }
}
