//! Target-router events for the source-backed scalar `log`, `sin`, and `cos`
//! operation slice.
//!
//! These requests retain the portable reduction actor's tensor-view contract.
//! Target routers own only the explicit target event and transition surface;
//! validation and formulas remain in [`crate::any::reductions::ReductionKernel`].

#![allow(clippy::derive_partial_eq_without_eq)]

use crate::any::reductions::{OpCos, OpLog, OpSin, ReductionResult};
use crate::any::tensor_view::{TensorView, TensorViewMut};

/// Pinned reference revision for the target scalar-trigonometry lanes.
pub const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

/// Pinned generic unary capability source span.
pub const PINNED_DETAIL_SPAN: &str =
    "src/emel/kernel/detail.hpp:2393-2412,3838-3841,5182-5187,5357-5362";
/// Pinned generic unary capability source blob.
pub const PINNED_DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";

/// A target scalar `log` request over tensor views.
#[derive(Debug)]
pub struct OpScalarLog<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpScalarLog<'a> {
    /// Creates a request; the target router's guards classify the views.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }

    /// Returns whether this request satisfies the valid F32/count unary contract.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.src.validate().is_ok()
            && self.dst.validate().is_ok()
            && self.src.count() == self.dst.count()
    }

    pub(crate) const fn into_reduction(self) -> OpLog<'a> {
        OpLog::new(self.src, self.dst)
    }
}

/// A target scalar `sin` request over tensor views.
#[derive(Debug)]
pub struct OpScalarSin<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpScalarSin<'a> {
    /// Creates a request; the target router's guards classify the views.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }

    /// Returns whether this request satisfies the valid F32/count unary contract.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.src.validate().is_ok()
            && self.dst.validate().is_ok()
            && self.src.count() == self.dst.count()
    }

    pub(crate) const fn into_reduction(self) -> OpSin<'a> {
        OpSin::new(self.src, self.dst)
    }
}

/// A target scalar `cos` request over tensor views.
#[derive(Debug)]
pub struct OpScalarCos<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpScalarCos<'a> {
    /// Creates a request; the target router's guards classify the views.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }

    /// Returns whether this request satisfies the valid F32/count unary contract.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.src.validate().is_ok()
            && self.dst.validate().is_ok()
            && self.src.count() == self.dst.count()
    }

    pub(crate) const fn into_reduction(self) -> OpCos<'a> {
        OpCos::new(self.src, self.dst)
    }
}

/// Result returned by every target scalar-trigonometry request.
pub type ScalarTrigResult = ReductionResult;

/// Explicit unexpected-event request for target scalar trigonometry.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedScalarTrig;
