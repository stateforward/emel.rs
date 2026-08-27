//! Safe F32 shape and layout operations from the pinned kernel contract.
//!
//! Source identity: emel.cpp commit
//! `843a117386ef17dc5a50549bbfc821074c2141d6`.
//!
//! The implementation follows `src/emel/kernel/detail.hpp` lines 1700-1778
//! (`tensor_element_count`, layout validation, dense-contiguous checks, and
//! logical offsets), `detail.hpp` lines 2342-2362 (`run_copy`), and the
//! explicit operation rows in `src/emel/kernel/x86_64/sm.hpp` lines 475-533.
//! `op_cpy` and `op_cont` materialize data into caller-owned destination
//! storage. `op_reshape`, `op_view`, `op_permute`, and `op_transpose` publish
//! validated metadata layouts without allocating or moving data.

#![allow(clippy::derive_partial_eq_without_eq)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::tensor_view::{Layout, TensorView, TensorViewMut};

/// Errors returned by shape and layout operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShapeError {
    /// A source or destination view failed dtype, shape, stride, or bounds validation.
    InvalidView,
    /// Logical element counts do not match the requested operation.
    ShapeMismatch,
    /// A destination layout does not satisfy the requested layout contract.
    InvalidLayout,
    /// The permutation is not a bijection over the four dimensions.
    InvalidPermutation,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid shape-operation view"),
            Self::ShapeMismatch => formatter.write_str("shape-operation element counts differ"),
            Self::InvalidLayout => formatter.write_str("invalid shape-operation layout"),
            Self::InvalidPermutation => formatter.write_str("invalid tensor permutation"),
            Self::UnexpectedEvent => formatter.write_str("unexpected shape-operation event"),
            Self::Internal => formatter.write_str("internal shape-operation dispatch error"),
        }
    }
}

impl std::error::Error for ShapeError {}

/// Copies one source view into a destination view with the same logical shape.
#[derive(Debug)]
pub struct OpCpy<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpCpy<'a> {
    /// Creates a copy event; view and shape validation occurs in a guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Copies a source view into a dense-contiguous destination view.
#[derive(Debug)]
pub struct OpCont<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpCont<'a> {
    /// Creates a contiguous-materialization event.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Validates and publishes a reshaped metadata layout without copying data.
#[derive(Debug)]
pub struct OpReshape<'a> {
    src: TensorView<'a>,
    target: Layout,
}

impl<'a> OpReshape<'a> {
    /// Creates a reshape event. The target must preserve the source count.
    #[must_use]
    pub const fn new(src: TensorView<'a>, target: Layout) -> Self {
        Self { src, target }
    }

    /// Returns the requested target metadata.
    #[must_use]
    pub const fn target(&self) -> Layout {
        self.target
    }
}

/// Validates and publishes a view metadata layout without copying data.
#[derive(Debug)]
pub struct OpView<'a> {
    src: TensorView<'a>,
    target: Layout,
}

impl<'a> OpView<'a> {
    /// Creates a view event. The target must preserve the source count.
    #[must_use]
    pub const fn new(src: TensorView<'a>, target: Layout) -> Self {
        Self { src, target }
    }

    /// Returns the requested target metadata.
    #[must_use]
    pub const fn target(&self) -> Layout {
        self.target
    }
}

/// Publishes a layout with dimensions and strides permuted by `axes`.
#[derive(Debug)]
pub struct OpPermute<'a> {
    src: TensorView<'a>,
    axes: [usize; 4],
    target: Layout,
}

impl<'a> OpPermute<'a> {
    /// Creates a permutation event. Axes are validated by an actor guard.
    #[must_use]
    pub fn new(src: TensorView<'a>, axes: [usize; 4]) -> Self {
        let target = src
            .layout()
            .permuted(axes)
            .unwrap_or_else(|| Layout::new(src.layout().dtype(), [0; 4], [0; 4]));
        Self { src, axes, target }
    }

    /// Returns the requested dimension permutation.
    #[must_use]
    pub const fn axes(&self) -> [usize; 4] {
        self.axes
    }

    /// Returns the derived target metadata.
    #[must_use]
    pub const fn target(&self) -> Layout {
        self.target
    }
}

/// Publishes a two-dimensional transpose of the source metadata.
#[derive(Debug)]
pub struct OpTranspose<'a> {
    src: TensorView<'a>,
    target: Layout,
}

impl<'a> OpTranspose<'a> {
    /// Creates a transpose event.
    #[must_use]
    pub const fn new(src: TensorView<'a>) -> Self {
        Self {
            target: src.layout().transposed(),
            src,
        }
    }

    /// Returns the derived target metadata.
    #[must_use]
    pub const fn target(&self) -> Layout {
        self.target
    }
}

/// Result returned by copy and contiguous-materialization events.
pub type CopyOutcome = Result<(), ShapeError>;
/// Result returned by metadata-only shape and layout events.
pub type LayoutOutcome = Result<Layout, ShapeError>;

#[derive(Debug)]
pub struct CpyRuntime<'a> {
    event: OpCpy<'a>,
    result: &'a Cell<CopyOutcome>,
}

#[derive(Debug)]
pub struct ContRuntime<'a> {
    event: OpCont<'a>,
    result: &'a Cell<CopyOutcome>,
}

#[derive(Debug)]
pub struct ReshapeRuntime<'a> {
    event: OpReshape<'a>,
    result: &'a Cell<LayoutOutcome>,
}

#[derive(Debug)]
pub struct ViewRuntime<'a> {
    event: OpView<'a>,
    result: &'a Cell<LayoutOutcome>,
}

#[derive(Debug)]
pub struct PermuteRuntime<'a> {
    event: OpPermute<'a>,
    result: &'a Cell<LayoutOutcome>,
}

#[derive(Debug)]
pub struct TransposeRuntime<'a> {
    event: OpTranspose<'a>,
    result: &'a Cell<LayoutOutcome>,
}

#[derive(Default)]
struct Context;

sml! {
    ShapeMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Cpy(CpyRuntime<'dispatch>) [guard_cpy_valid] / effect_cpy_execute,
        "ready"_s <= "ready"_s + Cpy(CpyRuntime<'dispatch>) [guard_cpy_shape_invalid] / effect_cpy_shape_reject,
        "ready"_s <= "ready"_s + Cpy(CpyRuntime<'dispatch>) [guard_cpy_view_invalid] / effect_cpy_view_reject,

        "ready"_s <= "ready"_s + Cont(ContRuntime<'dispatch>) [guard_cont_valid] / effect_cont_execute,
        "ready"_s <= "ready"_s + Cont(ContRuntime<'dispatch>) [guard_cont_shape_invalid] / effect_cont_shape_reject,
        "ready"_s <= "ready"_s + Cont(ContRuntime<'dispatch>) [guard_cont_layout_invalid] / effect_cont_layout_reject,
        "ready"_s <= "ready"_s + Cont(ContRuntime<'dispatch>) [guard_cont_view_invalid] / effect_cont_view_reject,

        "ready"_s <= "ready"_s + Reshape(ReshapeRuntime<'dispatch>) [guard_reshape_valid] / effect_reshape_publish,
        "ready"_s <= "ready"_s + Reshape(ReshapeRuntime<'dispatch>) [guard_reshape_shape_invalid] / effect_reshape_shape_reject,
        "ready"_s <= "ready"_s + Reshape(ReshapeRuntime<'dispatch>) [guard_reshape_layout_invalid] / effect_reshape_layout_reject,
        "ready"_s <= "ready"_s + Reshape(ReshapeRuntime<'dispatch>) [guard_reshape_view_invalid] / effect_reshape_view_reject,

        "ready"_s <= "ready"_s + View(ViewRuntime<'dispatch>) [guard_view_valid] / effect_view_publish,
        "ready"_s <= "ready"_s + View(ViewRuntime<'dispatch>) [guard_view_shape_invalid] / effect_view_shape_reject,
        "ready"_s <= "ready"_s + View(ViewRuntime<'dispatch>) [guard_view_layout_invalid] / effect_view_layout_reject,
        "ready"_s <= "ready"_s + View(ViewRuntime<'dispatch>) [guard_view_invalid] / effect_view_invalid_reject,

        "ready"_s <= "ready"_s + Permute(PermuteRuntime<'dispatch>) [guard_permute_valid] / effect_permute_publish,
        "ready"_s <= "ready"_s + Permute(PermuteRuntime<'dispatch>) [guard_permute_shape_invalid] / effect_permute_shape_reject,
        "ready"_s <= "ready"_s + Permute(PermuteRuntime<'dispatch>) [guard_permute_layout_invalid] / effect_permute_layout_reject,
        "ready"_s <= "ready"_s + Permute(PermuteRuntime<'dispatch>) [guard_permute_axes_invalid] / effect_permute_axes_reject,
        "ready"_s <= "ready"_s + Permute(PermuteRuntime<'dispatch>) [guard_permute_view_invalid] / effect_permute_view_reject,

        "ready"_s <= "ready"_s + Transpose(TransposeRuntime<'dispatch>) [guard_transpose_valid] / effect_transpose_publish,
        "ready"_s <= "ready"_s + Transpose(TransposeRuntime<'dispatch>) [guard_transpose_shape_invalid] / effect_transpose_shape_reject,
        "ready"_s <= "ready"_s + Transpose(TransposeRuntime<'dispatch>) [guard_transpose_layout_invalid] / effect_transpose_layout_reject,
        "ready"_s <= "ready"_s + Transpose(TransposeRuntime<'dispatch>) [guard_transpose_view_invalid] / effect_transpose_view_reject,

        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, RTC actor for shape and layout operations.
pub struct ShapeKernel {
    machine: ShapeMachineStateMachine<Context>,
}

impl fmt::Debug for ShapeKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ShapeKernel")
            .finish_non_exhaustive()
    }
}

impl Default for ShapeKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeKernel {
    /// Constructs an independent shape actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: ShapeMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: ShapeEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn cpy(&mut self, event: OpCpy<'_>) -> CopyOutcome {
        let result = Cell::new(Err(ShapeError::UnexpectedEvent));
        self.machine
            .process_event(ShapeMachineEvents::Cpy(CpyRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ShapeError::Internal)?;
        result.into_inner()
    }

    fn cont(&mut self, event: OpCont<'_>) -> CopyOutcome {
        let result = Cell::new(Err(ShapeError::UnexpectedEvent));
        self.machine
            .process_event(ShapeMachineEvents::Cont(ContRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ShapeError::Internal)?;
        result.into_inner()
    }

    fn reshape(&mut self, event: OpReshape<'_>) -> LayoutOutcome {
        let result = Cell::new(Err(ShapeError::UnexpectedEvent));
        self.machine
            .process_event(ShapeMachineEvents::Reshape(ReshapeRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ShapeError::Internal)?;
        result.into_inner()
    }

    fn view(&mut self, event: OpView<'_>) -> LayoutOutcome {
        let result = Cell::new(Err(ShapeError::UnexpectedEvent));
        self.machine
            .process_event(ShapeMachineEvents::View(ViewRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ShapeError::Internal)?;
        result.into_inner()
    }

    fn permute(&mut self, event: OpPermute<'_>) -> LayoutOutcome {
        let result = Cell::new(Err(ShapeError::UnexpectedEvent));
        self.machine
            .process_event(ShapeMachineEvents::Permute(PermuteRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ShapeError::Internal)?;
        result.into_inner()
    }

    fn transpose(&mut self, event: OpTranspose<'_>) -> LayoutOutcome {
        let result = Cell::new(Err(ShapeError::UnexpectedEvent));
        self.machine
            .process_event(ShapeMachineEvents::Transpose(TransposeRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ShapeError::Internal)?;
        result.into_inner()
    }
}

/// Typed event accepted by [`ShapeKernel`].
pub trait ShapeEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut ShapeKernel) -> Self::Output;
}

impl ShapeEvent for OpCpy<'_> {
    type Output = CopyOutcome;

    fn dispatch(self, actor: &mut ShapeKernel) -> Self::Output {
        actor.cpy(self)
    }
}

impl ShapeEvent for OpCont<'_> {
    type Output = CopyOutcome;

    fn dispatch(self, actor: &mut ShapeKernel) -> Self::Output {
        actor.cont(self)
    }
}

impl ShapeEvent for OpReshape<'_> {
    type Output = LayoutOutcome;

    fn dispatch(self, actor: &mut ShapeKernel) -> Self::Output {
        actor.reshape(self)
    }
}

impl ShapeEvent for OpView<'_> {
    type Output = LayoutOutcome;

    fn dispatch(self, actor: &mut ShapeKernel) -> Self::Output {
        actor.view(self)
    }
}

impl ShapeEvent for OpPermute<'_> {
    type Output = LayoutOutcome;

    fn dispatch(self, actor: &mut ShapeKernel) -> Self::Output {
        actor.permute(self)
    }
}

impl ShapeEvent for OpTranspose<'_> {
    type Output = LayoutOutcome;

    fn dispatch(self, actor: &mut ShapeKernel) -> Self::Output {
        actor.transpose(self)
    }
}

fn views_valid(src: &TensorView<'_>, dst: &TensorViewMut<'_>) -> bool {
    src.validate().is_ok() && dst.validate().is_ok()
}

const fn same_count(src: &TensorView<'_>, dst: &TensorViewMut<'_>) -> bool {
    src.count() == dst.count()
}

fn target_valid(src: &TensorView<'_>, target: Layout) -> bool {
    target.validate(src.storage_len()).is_ok()
}

fn target_count_matches(src: &TensorView<'_>, target: Layout) -> bool {
    src.count() == target.element_count().unwrap_or_default()
}

const fn axes_valid(axes: [usize; 4]) -> bool {
    let mut seen = 0u8;
    let mut index = 0;
    while index < 4 {
        let source_axis = axes[index];
        if source_axis >= 4 {
            return false;
        }
        let bit = 1u8 << source_axis;
        if seen & bit != 0 {
            return false;
        }
        seen |= bit;
        index += 1;
    }
    seen == 0b1111
}

const fn copy_view(src: &TensorView<'_>, dst: &mut TensorViewMut<'_>) {
    let mut ordinal = 0;
    while ordinal < src.count() {
        dst.write(ordinal, src.read(ordinal));
        ordinal += 1;
    }
}

impl ShapeMachineStateMachineContext for Context {
    fn guard_cpy_valid(&self, event: &CpyRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event.src, &event.event.dst)
            && same_count(&event.event.src, &event.event.dst))
    }

    fn guard_cpy_shape_invalid(&self, event: &CpyRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event.src, &event.event.dst)
            && !same_count(&event.event.src, &event.event.dst))
    }

    fn guard_cpy_view_invalid(&self, event: &CpyRuntime<'_>) -> Result<bool, ()> {
        Ok(!views_valid(&event.event.src, &event.event.dst))
    }

    fn effect_cpy_execute(&mut self, mut event: CpyRuntime<'_>) -> Result<(), ()> {
        copy_view(&event.event.src, &mut event.event.dst);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_cpy_shape_reject(&mut self, event: CpyRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::ShapeMismatch));
        Ok(())
    }

    fn effect_cpy_view_reject(&mut self, event: CpyRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::InvalidView));
        Ok(())
    }

    fn guard_cont_valid(&self, event: &ContRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event.src, &event.event.dst)
            && same_count(&event.event.src, &event.event.dst)
            && event.event.dst.layout().is_dense_contiguous())
    }

    fn guard_cont_shape_invalid(&self, event: &ContRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event.src, &event.event.dst)
            && !same_count(&event.event.src, &event.event.dst))
    }

    fn guard_cont_layout_invalid(&self, event: &ContRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event.src, &event.event.dst)
            && same_count(&event.event.src, &event.event.dst)
            && !event.event.dst.layout().is_dense_contiguous())
    }

    fn guard_cont_view_invalid(&self, event: &ContRuntime<'_>) -> Result<bool, ()> {
        Ok(!views_valid(&event.event.src, &event.event.dst))
    }

    fn effect_cont_execute(&mut self, mut event: ContRuntime<'_>) -> Result<(), ()> {
        copy_view(&event.event.src, &mut event.event.dst);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_cont_shape_reject(&mut self, event: ContRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::ShapeMismatch));
        Ok(())
    }

    fn effect_cont_layout_reject(&mut self, event: ContRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::InvalidLayout));
        Ok(())
    }

    fn effect_cont_view_reject(&mut self, event: ContRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::InvalidView));
        Ok(())
    }

    fn guard_reshape_valid(&self, event: &ReshapeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && target_valid(&event.event.src, event.event.target)
            && target_count_matches(&event.event.src, event.event.target))
    }

    fn guard_reshape_shape_invalid(&self, event: &ReshapeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && target_valid(&event.event.src, event.event.target)
            && !target_count_matches(&event.event.src, event.event.target))
    }

    fn guard_reshape_layout_invalid(&self, event: &ReshapeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && !target_valid(&event.event.src, event.event.target))
    }

    fn guard_reshape_view_invalid(&self, event: &ReshapeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_err())
    }

    fn effect_reshape_publish(&mut self, event: ReshapeRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(event.event.target));
        Ok(())
    }

    fn effect_reshape_shape_reject(&mut self, event: ReshapeRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::ShapeMismatch));
        Ok(())
    }

    fn effect_reshape_layout_reject(&mut self, event: ReshapeRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::InvalidLayout));
        Ok(())
    }

    fn effect_reshape_view_reject(&mut self, event: ReshapeRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::InvalidView));
        Ok(())
    }

    fn guard_view_valid(&self, event: &ViewRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && target_valid(&event.event.src, event.event.target)
            && target_count_matches(&event.event.src, event.event.target))
    }

    fn guard_view_shape_invalid(&self, event: &ViewRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && target_valid(&event.event.src, event.event.target)
            && !target_count_matches(&event.event.src, event.event.target))
    }

    fn guard_view_layout_invalid(&self, event: &ViewRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && !target_valid(&event.event.src, event.event.target))
    }

    fn guard_view_invalid(&self, event: &ViewRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_err())
    }

    fn effect_view_publish(&mut self, event: ViewRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(event.event.target));
        Ok(())
    }

    fn effect_view_shape_reject(&mut self, event: ViewRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::ShapeMismatch));
        Ok(())
    }

    fn effect_view_layout_reject(&mut self, event: ViewRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::InvalidLayout));
        Ok(())
    }

    fn effect_view_invalid_reject(&mut self, event: ViewRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::InvalidView));
        Ok(())
    }

    fn guard_permute_valid(&self, event: &PermuteRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && axes_valid(event.event.axes)
            && target_valid(&event.event.src, event.event.target)
            && target_count_matches(&event.event.src, event.event.target))
    }

    fn guard_permute_shape_invalid(&self, event: &PermuteRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && axes_valid(event.event.axes)
            && target_valid(&event.event.src, event.event.target)
            && !target_count_matches(&event.event.src, event.event.target))
    }

    fn guard_permute_layout_invalid(&self, event: &PermuteRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && axes_valid(event.event.axes)
            && !target_valid(&event.event.src, event.event.target))
    }

    fn guard_permute_axes_invalid(&self, event: &PermuteRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok() && !axes_valid(event.event.axes))
    }

    fn guard_permute_view_invalid(&self, event: &PermuteRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_err())
    }

    fn effect_permute_publish(&mut self, event: PermuteRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(event.event.target));
        Ok(())
    }

    fn effect_permute_shape_reject(&mut self, event: PermuteRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::ShapeMismatch));
        Ok(())
    }

    fn effect_permute_layout_reject(&mut self, event: PermuteRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::InvalidLayout));
        Ok(())
    }

    fn effect_permute_axes_reject(&mut self, event: PermuteRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::InvalidPermutation));
        Ok(())
    }

    fn effect_permute_view_reject(&mut self, event: PermuteRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::InvalidView));
        Ok(())
    }

    fn guard_transpose_valid(&self, event: &TransposeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && target_valid(&event.event.src, event.event.target)
            && target_count_matches(&event.event.src, event.event.target))
    }

    fn guard_transpose_shape_invalid(&self, event: &TransposeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && target_valid(&event.event.src, event.event.target)
            && !target_count_matches(&event.event.src, event.event.target))
    }

    fn guard_transpose_layout_invalid(&self, event: &TransposeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && !target_valid(&event.event.src, event.event.target))
    }

    fn guard_transpose_view_invalid(&self, event: &TransposeRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_err())
    }

    fn effect_transpose_publish(&mut self, event: TransposeRuntime<'_>) -> Result<(), ()> {
        event.result.set(Ok(event.event.target));
        Ok(())
    }

    fn effect_transpose_shape_reject(&mut self, event: TransposeRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::ShapeMismatch));
        Ok(())
    }

    fn effect_transpose_layout_reject(&mut self, event: TransposeRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::InvalidLayout));
        Ok(())
    }

    fn effect_transpose_view_reject(&mut self, event: TransposeRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ShapeError::InvalidView));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}
