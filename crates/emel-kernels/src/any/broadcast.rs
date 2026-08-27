//! Allocation-free F32 row-broadcast add and multiply with explicit SML
//! dispatch.
//!
//! The pinned emel.cpp source is commit
//! `843a117386ef17dc5a50549bbfc821074c2141d6`. Its
//! `src/emel/kernel/detail.hpp:3788-3836` contract requires `src0` and `dst`
//! to share shape, `src1` to be one row of `dst.ne[0]` values, and the output
//! to contain more than one such row. The source operation loop applies the
//! row values to every output row. The explicit x86-64 transition rows are
//! `src/emel/kernel/x86_64/sm.hpp:46-63` for `op_add` and
//! `:111-128` for `op_mul`.
//!
//! This module keeps the pinned row-major logical ordinal contract while using
//! safe `TensorView` and `TensorViewMut` access. Layouts must satisfy the
//! maintained F32 view validator before dispatch.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::{Cell, RefCell};
use core::fmt;

use sml::sml;

use super::tensor_view::{TensorView, TensorViewMut};

/// Errors returned by F32 row-broadcast dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BroadcastError {
    /// One or more views failed dtype, shape, stride, or bounds validation.
    InvalidView,
    /// The source row or output shape does not satisfy the broadcast contract.
    ShapeMismatch,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for BroadcastError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid broadcast tensor view"),
            Self::ShapeMismatch => formatter.write_str("broadcast tensor shapes differ"),
            Self::UnexpectedEvent => formatter.write_str("unexpected broadcast event"),
            Self::Internal => formatter.write_str("internal broadcast dispatch error"),
        }
    }
}

impl std::error::Error for BroadcastError {}

/// Adds one F32 row to every row of an F32 tensor.
#[derive(Debug)]
pub struct OpAddBroadcastRow<'a> {
    src0: TensorView<'a>,
    src1: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpAddBroadcastRow<'a> {
    /// Creates a row-broadcast addition event. Validation occurs in guards.
    #[must_use]
    pub const fn new(src0: TensorView<'a>, src1: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src0, src1, dst }
    }
}

/// Multiplies every F32 row by one F32 row of weights.
#[derive(Debug)]
pub struct OpMulBroadcastRow<'a> {
    src0: TensorView<'a>,
    src1: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> OpMulBroadcastRow<'a> {
    /// Creates a row-broadcast multiplication event. Validation occurs in
    /// guards.
    #[must_use]
    pub const fn new(src0: TensorView<'a>, src1: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src0, src1, dst }
    }
}

/// Result returned by [`OpAddBroadcastRow`] dispatch.
pub type BroadcastOutcome = Result<(), BroadcastError>;

/// Explicitly reports an event outside the maintained broadcast family.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedBroadcast;

#[derive(Clone, Copy, Debug)]
struct AddBroadcastRuntime<'dispatch, 'data> {
    src0: TensorView<'data>,
    src1: TensorView<'data>,
    dst: &'dispatch RefCell<TensorViewMut<'data>>,
    result: &'dispatch Cell<BroadcastOutcome>,
}

#[derive(Clone, Copy, Debug)]
struct MulBroadcastRuntime<'dispatch, 'data> {
    src0: TensorView<'data>,
    src1: TensorView<'data>,
    dst: &'dispatch RefCell<TensorViewMut<'data>>,
    result: &'dispatch Cell<BroadcastOutcome>,
}

#[derive(Clone, Copy)]
struct UnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<BroadcastOutcome>,
}

#[derive(Default)]
struct Context;

sml! {
    BroadcastMachine<'dispatch, 'data>
    where
        'data: 'dispatch,
    {
        "add_view_decision"_s <= *"ready"_s + AddBroadcast(AddBroadcastRuntime<'dispatch, 'data>),
        "add_shape_decision"_s <= "add_view_decision"_s
            + completion<AddBroadcast>(AddBroadcastRuntime<'dispatch, 'data>)
            [guard_add_views_valid],
        "ready"_s <= "add_view_decision"_s
            + completion<AddBroadcast>(AddBroadcastRuntime<'dispatch, 'data>)
            [guard_add_views_invalid] / effect_add_view_reject,
        "add_layout_decision"_s <= "add_shape_decision"_s
            + completion<AddBroadcast>(AddBroadcastRuntime<'dispatch, 'data>)
            [guard_add_shape_valid],
        "ready"_s <= "add_shape_decision"_s
            + completion<AddBroadcast>(AddBroadcastRuntime<'dispatch, 'data>)
            [guard_add_shape_invalid] / effect_add_shape_reject,
        "add_explicit_execution"_s <= "add_layout_decision"_s
            + completion<AddBroadcast>(AddBroadcastRuntime<'dispatch, 'data>)
            [guard_add_explicit],
        "add_implicit_execution"_s <= "add_layout_decision"_s
            + completion<AddBroadcast>(AddBroadcastRuntime<'dispatch, 'data>)
            [guard_add_implicit],
        "add_mixed_execution"_s <= "add_layout_decision"_s
            + completion<AddBroadcast>(AddBroadcastRuntime<'dispatch, 'data>)
            [guard_add_mixed],
        "ready"_s <= "add_explicit_execution"_s
            + completion<AddBroadcast>(AddBroadcastRuntime<'dispatch, 'data>) / effect_add_explicit,
        "ready"_s <= "add_implicit_execution"_s
            + completion<AddBroadcast>(AddBroadcastRuntime<'dispatch, 'data>) / effect_add_implicit,
        "ready"_s <= "add_mixed_execution"_s
            + completion<AddBroadcast>(AddBroadcastRuntime<'dispatch, 'data>) / effect_add_mixed,

        "mul_view_decision"_s <= "ready"_s + MulBroadcast(MulBroadcastRuntime<'dispatch, 'data>),
        "mul_shape_decision"_s <= "mul_view_decision"_s
            + completion<MulBroadcast>(MulBroadcastRuntime<'dispatch, 'data>)
            [guard_mul_views_valid],
        "ready"_s <= "mul_view_decision"_s
            + completion<MulBroadcast>(MulBroadcastRuntime<'dispatch, 'data>)
            [guard_mul_views_invalid] / effect_mul_view_reject,
        "mul_layout_decision"_s <= "mul_shape_decision"_s
            + completion<MulBroadcast>(MulBroadcastRuntime<'dispatch, 'data>)
            [guard_mul_shape_valid],
        "ready"_s <= "mul_shape_decision"_s
            + completion<MulBroadcast>(MulBroadcastRuntime<'dispatch, 'data>)
            [guard_mul_shape_invalid] / effect_mul_shape_reject,
        "mul_explicit_execution"_s <= "mul_layout_decision"_s
            + completion<MulBroadcast>(MulBroadcastRuntime<'dispatch, 'data>)
            [guard_mul_explicit],
        "mul_implicit_execution"_s <= "mul_layout_decision"_s
            + completion<MulBroadcast>(MulBroadcastRuntime<'dispatch, 'data>)
            [guard_mul_implicit],
        "mul_mixed_execution"_s <= "mul_layout_decision"_s
            + completion<MulBroadcast>(MulBroadcastRuntime<'dispatch, 'data>)
            [guard_mul_mixed],
        "ready"_s <= "mul_explicit_execution"_s
            + completion<MulBroadcast>(MulBroadcastRuntime<'dispatch, 'data>) / effect_mul_explicit,
        "ready"_s <= "mul_implicit_execution"_s
            + completion<MulBroadcast>(MulBroadcastRuntime<'dispatch, 'data>) / effect_mul_implicit,
        "ready"_s <= "mul_mixed_execution"_s
            + completion<MulBroadcast>(MulBroadcastRuntime<'dispatch, 'data>) / effect_mul_mixed,

        "unexpected_dispatch"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected_event,
        "ready"_s <= "unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "unexpected_dispatch"_s + unexpected_event<_> / effect_unexpected,
        "generic_unexpected_dispatch"_s <= "ready"_s + unexpected_event<_>
            / effect_unexpected,
        "ready"_s <= "generic_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "generic_unexpected_dispatch"_s + unexpected_event<_>
            / effect_unexpected,

        "ready"_s <= "add_view_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "add_shape_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "add_layout_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "add_explicit_execution"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "add_implicit_execution"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "add_mixed_execution"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "mul_view_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "mul_shape_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "mul_layout_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "mul_explicit_execution"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "mul_implicit_execution"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "mul_mixed_execution"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, run-to-completion actor for F32 row-broadcast operations.
pub struct BroadcastKernel {
    machine: BroadcastMachineStateMachine<Context>,
}

impl fmt::Debug for BroadcastKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BroadcastKernel")
            .finish_non_exhaustive()
    }
}

impl Default for BroadcastKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl BroadcastKernel {
    /// Constructs an independent row-broadcast actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: BroadcastMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    ///
    /// # Errors
    ///
    /// Returns [`BroadcastError`] when a view or row-broadcast shape invariant
    /// fails, or when generated dispatch cannot complete.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: BroadcastEvent>(&mut self, event: E) -> E::Output {
        let output = event.dispatch(self);
        assert!(self.machine.is(&BroadcastMachineStates::Ready));
        output
    }

    fn add_broadcast(&mut self, event: OpAddBroadcastRow<'_>) -> BroadcastOutcome {
        let output = RefCell::new(event.dst);
        let result = Cell::new(Err(BroadcastError::UnexpectedEvent));
        self.machine
            .process_event(BroadcastMachineEvents::AddBroadcast(AddBroadcastRuntime {
                src0: event.src0,
                src1: event.src1,
                dst: &output,
                result: &result,
            }))
            .map_err(|_| BroadcastError::Internal)?;
        assert!(self.machine.is(&BroadcastMachineStates::Ready));
        result.into_inner()
    }

    fn mul_broadcast(&mut self, event: OpMulBroadcastRow<'_>) -> BroadcastOutcome {
        let output = RefCell::new(event.dst);
        let result = Cell::new(Err(BroadcastError::UnexpectedEvent));
        self.machine
            .process_event(BroadcastMachineEvents::MulBroadcast(MulBroadcastRuntime {
                src0: event.src0,
                src1: event.src1,
                dst: &output,
                result: &result,
            }))
            .map_err(|_| BroadcastError::Internal)?;
        assert!(self.machine.is(&BroadcastMachineStates::Ready));
        result.into_inner()
    }

    fn unexpected(&mut self, _event: UnexpectedBroadcast) -> BroadcastOutcome {
        let result = Cell::new(Err(BroadcastError::UnexpectedEvent));
        self.machine
            .process_event(BroadcastMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| BroadcastError::Internal)?;
        result.into_inner()
    }
}

/// Typed event accepted by [`BroadcastKernel`].
pub trait BroadcastEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut BroadcastKernel) -> Self::Output;
}

impl BroadcastEvent for OpAddBroadcastRow<'_> {
    type Output = BroadcastOutcome;

    fn dispatch(self, actor: &mut BroadcastKernel) -> Self::Output {
        actor.add_broadcast(self)
    }
}

impl BroadcastEvent for OpMulBroadcastRow<'_> {
    type Output = BroadcastOutcome;

    fn dispatch(self, actor: &mut BroadcastKernel) -> Self::Output {
        actor.mul_broadcast(self)
    }
}

impl BroadcastEvent for UnexpectedBroadcast {
    type Output = BroadcastOutcome;

    fn dispatch(self, actor: &mut BroadcastKernel) -> Self::Output {
        actor.unexpected(self)
    }
}

fn views_valid(
    src0: &TensorView<'_>,
    src1: &TensorView<'_>,
    dst: &RefCell<TensorViewMut<'_>>,
) -> bool {
    let dst = dst.borrow();
    src0.validate_binary().is_ok()
        && src1.validate_binary().is_ok()
        && dst.validate_binary().is_ok()
}

fn shape_valid(
    src0_view: &TensorView<'_>,
    src1_view: &TensorView<'_>,
    dst_view: &TensorViewMut<'_>,
) -> bool {
    let src0 = src0_view.layout().ne();
    let src1 = src1_view.layout().ne();
    let dst = dst_view.layout().ne();
    src0 == dst
        && dst[0] > 0
        && src1[0] == dst[0]
        && src1[1] == 1
        && src1[2] == 1
        && src1[3] == 1
        && dst[1] > 0
        && dst_view.count() > src1_view.count()
}

fn execute_add_broadcast(
    src0: &TensorView<'_>,
    src1: &TensorView<'_>,
    dst: &RefCell<TensorViewMut<'_>>,
) {
    let columns = src1.count();
    let rows = dst.borrow().count() / columns;
    let mut row = 0;
    while row < rows {
        let mut column = 0;
        while column < columns {
            let ordinal = row * columns + column;
            let value = src0.read(ordinal) + src1.read(column);
            dst.borrow_mut().write(ordinal, value);
            column += 1;
        }
        row += 1;
    }
}

fn execute_mul_broadcast(
    src0: &TensorView<'_>,
    src1: &TensorView<'_>,
    dst: &RefCell<TensorViewMut<'_>>,
) {
    let columns = src1.count();
    let rows = dst.borrow().count() / columns;
    let mut row = 0;
    while row < rows {
        let mut column = 0;
        while column < columns {
            let ordinal = row * columns + column;
            let value = src0.read(ordinal) * src1.read(column);
            dst.borrow_mut().write(ordinal, value);
            column += 1;
        }
        row += 1;
    }
}

fn all_explicit<'data>(
    src0: &TensorView<'data>,
    src1: &TensorView<'data>,
    dst: &RefCell<TensorViewMut<'data>>,
) -> bool {
    let output = dst.borrow();
    src0.layout().nb()[0] != 0 && src1.layout().nb()[0] != 0 && output.layout().nb()[0] != 0
}

fn all_implicit<'data>(
    src0: &TensorView<'data>,
    src1: &TensorView<'data>,
    dst: &RefCell<TensorViewMut<'data>>,
) -> bool {
    let output = dst.borrow();
    src0.layout().nb()[0] == 0 && src1.layout().nb()[0] == 0 && output.layout().nb()[0] == 0
}

macro_rules! impl_broadcast_context {
    ($runtime:ident, $views_valid:ident, $views_invalid:ident, $shape_valid:ident,
        $shape_invalid:ident, $explicit:ident, $implicit:ident, $mixed:ident,
        $effect_explicit:ident, $effect_implicit:ident, $effect_mixed:ident,
        $shape_reject:ident, $view_reject:ident, $execute:path) => {
        fn $views_valid<'dispatch, 'data>(
            &self,
            event: &$runtime<'dispatch, 'data>,
        ) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(views_valid(&event.src0, &event.src1, event.dst))
        }

        fn $views_invalid<'dispatch, 'data>(
            &self,
            event: &$runtime<'dispatch, 'data>,
        ) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(!views_valid(&event.src0, &event.src1, event.dst))
        }

        fn $shape_valid<'dispatch, 'data>(
            &self,
            event: &$runtime<'dispatch, 'data>,
        ) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(shape_valid(&event.src0, &event.src1, &event.dst.borrow()))
        }

        fn $shape_invalid<'dispatch, 'data>(
            &self,
            event: &$runtime<'dispatch, 'data>,
        ) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(!shape_valid(&event.src0, &event.src1, &event.dst.borrow()))
        }

        fn $explicit<'dispatch, 'data>(
            &self,
            event: &$runtime<'dispatch, 'data>,
        ) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(all_explicit(&event.src0, &event.src1, event.dst))
        }

        fn $implicit<'dispatch, 'data>(
            &self,
            event: &$runtime<'dispatch, 'data>,
        ) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(all_implicit(&event.src0, &event.src1, event.dst))
        }

        fn $mixed<'dispatch, 'data>(&self, event: &$runtime<'dispatch, 'data>) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(!all_explicit(&event.src0, &event.src1, event.dst)
                && !all_implicit(&event.src0, &event.src1, event.dst))
        }

        fn $effect_explicit<'dispatch, 'data>(
            &mut self,
            event: $runtime<'dispatch, 'data>,
        ) -> Result<(), ()>
        where
            'data: 'dispatch,
        {
            $execute(&event.src0, &event.src1, event.dst);
            event.result.set(Ok(()));
            Ok(())
        }

        fn $effect_implicit<'dispatch, 'data>(
            &mut self,
            event: $runtime<'dispatch, 'data>,
        ) -> Result<(), ()>
        where
            'data: 'dispatch,
        {
            $execute(&event.src0, &event.src1, event.dst);
            event.result.set(Ok(()));
            Ok(())
        }

        fn $effect_mixed<'dispatch, 'data>(
            &mut self,
            event: $runtime<'dispatch, 'data>,
        ) -> Result<(), ()>
        where
            'data: 'dispatch,
        {
            $execute(&event.src0, &event.src1, event.dst);
            event.result.set(Ok(()));
            Ok(())
        }

        fn $shape_reject<'dispatch, 'data>(
            &mut self,
            event: $runtime<'dispatch, 'data>,
        ) -> Result<(), ()>
        where
            'data: 'dispatch,
        {
            event.result.set(Err(BroadcastError::ShapeMismatch));
            Ok(())
        }

        fn $view_reject<'dispatch, 'data>(
            &mut self,
            event: $runtime<'dispatch, 'data>,
        ) -> Result<(), ()>
        where
            'data: 'dispatch,
        {
            event.result.set(Err(BroadcastError::InvalidView));
            Ok(())
        }
    };
}

impl BroadcastMachineStateMachineContext for Context {
    impl_broadcast_context!(
        AddBroadcastRuntime,
        guard_add_views_valid,
        guard_add_views_invalid,
        guard_add_shape_valid,
        guard_add_shape_invalid,
        guard_add_explicit,
        guard_add_implicit,
        guard_add_mixed,
        effect_add_explicit,
        effect_add_implicit,
        effect_add_mixed,
        effect_add_shape_reject,
        effect_add_view_reject,
        execute_add_broadcast
    );
    impl_broadcast_context!(
        MulBroadcastRuntime,
        guard_mul_views_valid,
        guard_mul_views_invalid,
        guard_mul_shape_valid,
        guard_mul_shape_invalid,
        guard_mul_explicit,
        guard_mul_implicit,
        guard_mul_mixed,
        effect_mul_explicit,
        effect_mul_implicit,
        effect_mul_mixed,
        effect_mul_shape_reject,
        effect_mul_view_reject,
        execute_mul_broadcast
    );

    fn effect_unexpected_event(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(BroadcastError::UnexpectedEvent));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}
