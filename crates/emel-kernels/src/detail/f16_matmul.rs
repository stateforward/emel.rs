//! Target-router events for the source-backed scalar F16 matrix route.
//!
//! The target routers keep the target event and guard surface explicit while
//! delegating validation and arithmetic to the maintained safe
//! [`crate::any::f16_matmul::F16MatmulKernel`] actor.

#![allow(clippy::derive_partial_eq_without_eq)]

use crate::any::f16_matmul::{F16MatmulResult as AnyF16MatmulResult, F16View, OpMulMatF16};
use crate::any::tensor_view::TensorViewMut;

/// Pinned reference revision for the target scalar F16 matrix lanes.
pub const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

/// Pinned shared F16 capability and execution spans.
pub const PINNED_DETAIL_SPAN: &str = "src/emel/kernel/detail.hpp:3878-3894,4199-4214";
/// Pinned generic detail source blob.
pub const PINNED_DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";

/// Target scalar F16 matrix multiplication request.
#[derive(Debug)]
pub struct OpScalarMulMatF16<'a> {
    lhs: F16View<'a>,
    rhs: F16View<'a>,
    destination: TensorViewMut<'a>,
}

impl<'a> OpScalarMulMatF16<'a> {
    /// Creates a request; target guards classify views and dimensions.
    #[must_use]
    pub const fn new(lhs: F16View<'a>, rhs: F16View<'a>, destination: TensorViewMut<'a>) -> Self {
        Self {
            lhs,
            rhs,
            destination,
        }
    }

    /// Returns whether this request satisfies the pinned scalar F16 contract.
    #[must_use]
    pub(crate) fn is_valid(&self) -> bool {
        if !self.lhs.validate()
            || !self.rhs.validate()
            || self.destination.validate().is_err()
            || !self.lhs.layout().is_dense_contiguous_f16()
            || !self.rhs.layout().is_dense_contiguous_f16()
            || !self.destination.layout().is_dense_contiguous()
        {
            return false;
        }
        let lhs = self.lhs.layout().ne();
        let rhs = self.rhs.layout().ne();
        let destination = self.destination.layout().ne();
        lhs[0] > 0
            && lhs[1] > 0
            && rhs[0] == lhs[0]
            && rhs[1] > 0
            && destination[0] == lhs[1]
            && destination[1] == rhs[1]
            && lhs[2] == 1
            && lhs[3] == 1
            && rhs[2] == 1
            && rhs[3] == 1
            && destination[2] == 1
            && destination[3] == 1
    }

    pub(crate) const fn into_reduction(self) -> OpMulMatF16<'a> {
        OpMulMatF16::new(self.lhs, self.rhs, self.destination)
    }

    /// Transfers validated target operands to a same-RTC target kernel.
    pub(crate) const fn into_parts(self) -> (F16View<'a>, F16View<'a>, TensorViewMut<'a>) {
        (self.lhs, self.rhs, self.destination)
    }
}

/// Short target-event alias for scalar F16 matrix multiplication.
pub type OpMulMatF16Scalar<'a> = OpScalarMulMatF16<'a>;

/// Result returned by a target scalar F16 matrix request.
pub type F16MatmulResult = AnyF16MatmulResult;

/// Explicit unexpected-event request for target scalar F16 matrix routing.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedF16Matmul;
