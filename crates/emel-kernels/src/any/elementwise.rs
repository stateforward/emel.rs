//! Portable F32 add-one and contiguous accumulate kernels.
//!
//! The pinned kernel contract names `op_add1` and `op_acc` as separate SML
//! operations.  This module keeps those operations separate at the public
//! event and transition level.  The maintained `op_acc` slice accepts dense
//! F32 views and an element offset; the reference byte-stride view is kept out
//! of this safe slice until the tensor-view contract carries the required
//! proven subview metadata.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::tensor_view::{TensorView, TensorViewMut};

/// Errors returned by F32 elementwise dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElementwiseError {
    /// One or more views failed dtype, shape, stride, or bounds validation.
    InvalidView,
    /// Valid views do not have the required logical extents.
    ShapeMismatch,
    /// The accumulate update does not fit at its requested element offset.
    InvalidOffset,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for ElementwiseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid elementwise tensor view"),
            Self::ShapeMismatch => formatter.write_str("elementwise tensor shapes differ"),
            Self::InvalidOffset => formatter.write_str("accumulate update exceeds destination"),
            Self::UnexpectedEvent => formatter.write_str("unexpected elementwise event"),
            Self::Internal => formatter.write_str("internal elementwise dispatch error"),
        }
    }
}

impl std::error::Error for ElementwiseError {}

/// Adds one scalar to every element of a dense or strided F32 source view.
#[derive(Debug)]
pub struct OpAdd1<'a> {
    src: TensorView<'a>,
    scalar: f32,
    dst: TensorViewMut<'a>,
}

impl<'a> OpAdd1<'a> {
    /// Creates an add-one event. View and shape validation occurs in a guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, scalar: f32, dst: TensorViewMut<'a>) -> Self {
        Self { src, scalar, dst }
    }
}

/// Copies a base tensor and accumulates a contiguous update at an element
/// offset.
#[derive(Debug)]
pub struct OpAcc<'a> {
    base: TensorView<'a>,
    update: TensorView<'a>,
    dst: TensorViewMut<'a>,
    offset: usize,
}

impl<'a> OpAcc<'a> {
    /// Creates an accumulate event.
    ///
    /// `offset` is measured in F32 logical elements.  The maintained slice
    /// requires all three views to be dense contiguous F32 storage; this is
    /// the safe subset of the reference operation's byte-offset/stride form.
    #[must_use]
    pub const fn new(
        base: TensorView<'a>,
        update: TensorView<'a>,
        dst: TensorViewMut<'a>,
        offset: usize,
    ) -> Self {
        Self {
            base,
            update,
            dst,
            offset,
        }
    }
}

/// Result returned by a portable elementwise dispatch.
pub type ElementwiseResult = Result<(), ElementwiseError>;

struct Add1Runtime<'a> {
    src: TensorView<'a>,
    scalar: f32,
    dst: TensorViewMut<'a>,
    result: &'a Cell<ElementwiseResult>,
}

struct AccRuntime<'a> {
    base: TensorView<'a>,
    update: TensorView<'a>,
    dst: TensorViewMut<'a>,
    offset: usize,
    result: &'a Cell<ElementwiseResult>,
}

#[derive(Default)]
struct Context;

sml! {
    ElementwiseMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Add1(Add1Runtime<'dispatch>) [guard_add1_valid] / effect_add1_execute,
        "ready"_s <= "ready"_s + Add1(Add1Runtime<'dispatch>) [guard_add1_shape_invalid] / effect_add1_shape_mismatch,
        "ready"_s <= "ready"_s + Add1(Add1Runtime<'dispatch>) [guard_add1_view_invalid] / effect_add1_invalid_view,

        "ready"_s <= "ready"_s + Acc(AccRuntime<'dispatch>) [guard_acc_valid] / effect_acc_execute,
        "ready"_s <= "ready"_s + Acc(AccRuntime<'dispatch>) [guard_acc_shape_invalid] / effect_acc_shape_mismatch,
        "ready"_s <= "ready"_s + Acc(AccRuntime<'dispatch>) [guard_acc_offset_invalid] / effect_acc_invalid_offset,
        "ready"_s <= "ready"_s + Acc(AccRuntime<'dispatch>) [guard_acc_view_invalid] / effect_acc_invalid_view,

        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, run-to-completion actor for the maintained F32 operations.
pub struct ElementwiseKernel {
    machine: ElementwiseMachineStateMachine<Context>,
}

impl fmt::Debug for ElementwiseKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ElementwiseKernel")
            .finish_non_exhaustive()
    }
}

impl Default for ElementwiseKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementwiseKernel {
    /// Constructs an independent elementwise actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: ElementwiseMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: ElementwiseEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn add1(&mut self, event: OpAdd1<'_>) -> ElementwiseResult {
        let result = Cell::new(Err(ElementwiseError::UnexpectedEvent));
        self.machine
            .process_event(ElementwiseMachineEvents::Add1(Add1Runtime {
                src: event.src,
                scalar: event.scalar,
                dst: event.dst,
                result: &result,
            }))
            .map_err(|_| ElementwiseError::Internal)?;
        result.get()
    }

    fn acc(&mut self, event: OpAcc<'_>) -> ElementwiseResult {
        let result = Cell::new(Err(ElementwiseError::UnexpectedEvent));
        self.machine
            .process_event(ElementwiseMachineEvents::Acc(AccRuntime {
                base: event.base,
                update: event.update,
                dst: event.dst,
                offset: event.offset,
                result: &result,
            }))
            .map_err(|_| ElementwiseError::Internal)?;
        result.get()
    }
}

/// Trait implemented by every public elementwise event.
pub trait ElementwiseEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut ElementwiseKernel) -> Self::Output;
}

impl ElementwiseEvent for OpAdd1<'_> {
    type Output = ElementwiseResult;

    fn dispatch(self, actor: &mut ElementwiseKernel) -> Self::Output {
        actor.add1(self)
    }
}

impl ElementwiseEvent for OpAcc<'_> {
    type Output = ElementwiseResult;

    fn dispatch(self, actor: &mut ElementwiseKernel) -> Self::Output {
        actor.acc(self)
    }
}

fn valid_pair(src: &TensorView<'_>, dst: &TensorViewMut<'_>) -> bool {
    src.validate().is_ok() && dst.validate().is_ok()
}

fn same_shape(src: &TensorView<'_>, dst: &TensorViewMut<'_>) -> bool {
    src.layout().ne() == dst.layout().ne()
}

fn dense_view(view: &TensorView<'_>) -> bool {
    view.layout().is_dense_contiguous()
}

fn dense_mut_view(view: &TensorViewMut<'_>) -> bool {
    view.layout().is_dense_contiguous()
}

fn acc_views_valid(event: &AccRuntime<'_>) -> bool {
    valid_pair(&event.base, &event.dst)
        && event.update.validate().is_ok()
        && same_shape(&event.base, &event.dst)
        && dense_view(&event.base)
        && dense_view(&event.update)
        && dense_mut_view(&event.dst)
}

const fn acc_offset_valid(event: &AccRuntime<'_>) -> bool {
    event.offset <= event.dst.count()
        && event.update.count() <= event.dst.count().saturating_sub(event.offset)
}

impl ElementwiseMachineStateMachineContext for Context {
    fn guard_add1_valid(&self, event: &Add1Runtime<'_>) -> Result<bool, ()> {
        Ok(valid_pair(&event.src, &event.dst) && same_shape(&event.src, &event.dst))
    }

    fn guard_add1_shape_invalid(&self, event: &Add1Runtime<'_>) -> Result<bool, ()> {
        Ok(valid_pair(&event.src, &event.dst) && !same_shape(&event.src, &event.dst))
    }

    fn guard_add1_view_invalid(&self, event: &Add1Runtime<'_>) -> Result<bool, ()> {
        Ok(!valid_pair(&event.src, &event.dst))
    }

    fn guard_acc_valid(&self, event: &AccRuntime<'_>) -> Result<bool, ()> {
        Ok(acc_views_valid(event) && acc_offset_valid(event))
    }

    fn guard_acc_shape_invalid(&self, event: &AccRuntime<'_>) -> Result<bool, ()> {
        Ok(valid_pair(&event.base, &event.dst)
            && event.update.validate().is_ok()
            && !same_shape(&event.base, &event.dst))
    }

    fn guard_acc_offset_invalid(&self, event: &AccRuntime<'_>) -> Result<bool, ()> {
        Ok(acc_views_valid(event) && !acc_offset_valid(event))
    }

    fn guard_acc_view_invalid(&self, event: &AccRuntime<'_>) -> Result<bool, ()> {
        Ok(!acc_views_valid(event))
    }

    fn effect_add1_execute(&mut self, mut event: Add1Runtime<'_>) -> Result<(), ()> {
        let mut ordinal = 0;
        while ordinal < event.src.count() {
            event
                .dst
                .write(ordinal, event.src.read(ordinal) + event.scalar);
            ordinal += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_acc_execute(&mut self, mut event: AccRuntime<'_>) -> Result<(), ()> {
        let mut ordinal = 0;
        while ordinal < event.base.count() {
            event.dst.write(ordinal, event.base.read(ordinal));
            ordinal += 1;
        }
        ordinal = 0;
        while ordinal < event.update.count() {
            let destination = event.offset + ordinal;
            let value = event.dst.read(destination) + event.update.read(ordinal);
            event.dst.write(destination, value);
            ordinal += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_add1_shape_mismatch(&mut self, event: Add1Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(ElementwiseError::ShapeMismatch));
        Ok(())
    }

    fn effect_add1_invalid_view(&mut self, event: Add1Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(ElementwiseError::InvalidView));
        Ok(())
    }

    fn effect_acc_shape_mismatch(&mut self, event: AccRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ElementwiseError::ShapeMismatch));
        Ok(())
    }

    fn effect_acc_invalid_offset(&mut self, event: AccRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ElementwiseError::InvalidOffset));
        Ok(())
    }

    fn effect_acc_invalid_view(&mut self, event: AccRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ElementwiseError::InvalidView));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}
