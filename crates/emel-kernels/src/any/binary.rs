//! Explicit portable equal-count F32 binary operations.
//!
//! The pinned scalar reference applies `op_add`, `op_sub`, `op_mul`, and
//! `op_div` element-wise after checking all three tensor element counts. The
//! maintained slice covers that equal-count path for dense and validated
//! strided views. Row-broadcast variants remain a separate operation contract.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::{Cell, RefCell};
use core::fmt;

use sml::sml;

use super::tensor_view::{TensorView, TensorViewMut};

/// Errors returned by equal-count binary dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinaryError {
    /// A source or destination view failed validation.
    InvalidView,
    /// The three logical element counts differ.
    ShapeMismatch,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for BinaryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid binary tensor view"),
            Self::ShapeMismatch => formatter.write_str("binary tensor element counts differ"),
            Self::UnexpectedEvent => formatter.write_str("unexpected binary event"),
            Self::Internal => formatter.write_str("internal binary dispatch error"),
        }
    }
}

impl std::error::Error for BinaryError {}

/// Result returned after binary dispatch.
pub type BinaryResult = Result<(), BinaryError>;

macro_rules! define_binary_event {
    ($event:ident, $runtime:ident) => {
        #[derive(Debug)]
        pub struct $event<'a> {
            lhs: TensorView<'a>,
            rhs: TensorView<'a>,
            output: TensorViewMut<'a>,
        }

        impl<'a> $event<'a> {
            /// Creates an equal-count binary event; validation occurs in a guard.
            #[must_use]
            pub const fn new(
                lhs: TensorView<'a>,
                rhs: TensorView<'a>,
                output: TensorViewMut<'a>,
            ) -> Self {
                Self { lhs, rhs, output }
            }
        }

        #[derive(Clone, Copy)]
        struct $runtime<'dispatch, 'data> {
            lhs: TensorView<'data>,
            rhs: TensorView<'data>,
            output: &'dispatch RefCell<TensorViewMut<'data>>,
            result: &'dispatch Cell<BinaryResult>,
        }
    };
}

define_binary_event!(OpAdd, AddRuntime);
define_binary_event!(OpSub, SubRuntime);
define_binary_event!(OpMul, MulRuntime);
define_binary_event!(OpDiv, DivRuntime);

/// Explicitly reports an event outside the maintained binary family.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedBinary;

#[derive(Clone, Copy)]
struct UnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<BinaryResult>,
}

#[derive(Default)]
struct Context;

sml! {
    BinaryMachine<'dispatch, 'data>
    where
        'data: 'dispatch,
    {
        // Add: view, shape, layout, and execution decisions are separate
        // phase-level states. The completion payload is Copy because it only
        // carries immutable views and a RefCell capability for the output.
        "add_view_decision"_s <= *"ready"_s + Add(AddRuntime<'dispatch, 'data>),
        "add_shape_decision"_s <= "add_view_decision"_s + completion<Add>(AddRuntime<'dispatch, 'data>)
            [guard_add_views_valid],
        "ready"_s <= "add_view_decision"_s + completion<Add>(AddRuntime<'dispatch, 'data>)
            [guard_add_views_invalid] / effect_add_view_reject,
        "add_layout_decision"_s <= "add_shape_decision"_s + completion<Add>(AddRuntime<'dispatch, 'data>)
            [guard_add_shape_valid],
        "ready"_s <= "add_shape_decision"_s + completion<Add>(AddRuntime<'dispatch, 'data>)
            [guard_add_shape_invalid] / effect_add_shape_reject,
        "add_explicit_execution"_s <= "add_layout_decision"_s + completion<Add>(AddRuntime<'dispatch, 'data>)
            [guard_add_explicit],
        "add_implicit_execution"_s <= "add_layout_decision"_s + completion<Add>(AddRuntime<'dispatch, 'data>)
            [guard_add_implicit],
        "add_mixed_execution"_s <= "add_layout_decision"_s + completion<Add>(AddRuntime<'dispatch, 'data>)
            [guard_add_mixed],
        "ready"_s <= "add_explicit_execution"_s + completion<Add>(AddRuntime<'dispatch, 'data>)
            / effect_add_explicit,
        "ready"_s <= "add_implicit_execution"_s + completion<Add>(AddRuntime<'dispatch, 'data>)
            / effect_add_implicit,
        "ready"_s <= "add_mixed_execution"_s + completion<Add>(AddRuntime<'dispatch, 'data>)
            / effect_add_mixed,

        // Sub: the same explicit phase chain keeps validation and layout
        // selection out of actions while retaining borrowed output ownership.
        "sub_view_decision"_s <= "ready"_s + Sub(SubRuntime<'dispatch, 'data>),
        "sub_shape_decision"_s <= "sub_view_decision"_s + completion<Sub>(SubRuntime<'dispatch, 'data>)
            [guard_sub_views_valid],
        "ready"_s <= "sub_view_decision"_s + completion<Sub>(SubRuntime<'dispatch, 'data>)
            [guard_sub_views_invalid] / effect_sub_view_reject,
        "sub_layout_decision"_s <= "sub_shape_decision"_s + completion<Sub>(SubRuntime<'dispatch, 'data>)
            [guard_sub_shape_valid],
        "ready"_s <= "sub_shape_decision"_s + completion<Sub>(SubRuntime<'dispatch, 'data>)
            [guard_sub_shape_invalid] / effect_sub_shape_reject,
        "sub_explicit_execution"_s <= "sub_layout_decision"_s + completion<Sub>(SubRuntime<'dispatch, 'data>)
            [guard_sub_explicit],
        "sub_implicit_execution"_s <= "sub_layout_decision"_s + completion<Sub>(SubRuntime<'dispatch, 'data>)
            [guard_sub_implicit],
        "sub_mixed_execution"_s <= "sub_layout_decision"_s + completion<Sub>(SubRuntime<'dispatch, 'data>)
            [guard_sub_mixed],
        "ready"_s <= "sub_explicit_execution"_s + completion<Sub>(SubRuntime<'dispatch, 'data>)
            / effect_sub_explicit,
        "ready"_s <= "sub_implicit_execution"_s + completion<Sub>(SubRuntime<'dispatch, 'data>)
            / effect_sub_implicit,
        "ready"_s <= "sub_mixed_execution"_s + completion<Sub>(SubRuntime<'dispatch, 'data>)
            / effect_sub_mixed,

        // Mul: explicit decision states cover all guarded layout variants.
        "mul_view_decision"_s <= "ready"_s + Mul(MulRuntime<'dispatch, 'data>),
        "mul_shape_decision"_s <= "mul_view_decision"_s + completion<Mul>(MulRuntime<'dispatch, 'data>)
            [guard_mul_views_valid],
        "ready"_s <= "mul_view_decision"_s + completion<Mul>(MulRuntime<'dispatch, 'data>)
            [guard_mul_views_invalid] / effect_mul_view_reject,
        "mul_layout_decision"_s <= "mul_shape_decision"_s + completion<Mul>(MulRuntime<'dispatch, 'data>)
            [guard_mul_shape_valid],
        "ready"_s <= "mul_shape_decision"_s + completion<Mul>(MulRuntime<'dispatch, 'data>)
            [guard_mul_shape_invalid] / effect_mul_shape_reject,
        "mul_explicit_execution"_s <= "mul_layout_decision"_s + completion<Mul>(MulRuntime<'dispatch, 'data>)
            [guard_mul_explicit],
        "mul_implicit_execution"_s <= "mul_layout_decision"_s + completion<Mul>(MulRuntime<'dispatch, 'data>)
            [guard_mul_implicit],
        "mul_mixed_execution"_s <= "mul_layout_decision"_s + completion<Mul>(MulRuntime<'dispatch, 'data>)
            [guard_mul_mixed],
        "ready"_s <= "mul_explicit_execution"_s + completion<Mul>(MulRuntime<'dispatch, 'data>)
            / effect_mul_explicit,
        "ready"_s <= "mul_implicit_execution"_s + completion<Mul>(MulRuntime<'dispatch, 'data>)
            / effect_mul_implicit,
        "ready"_s <= "mul_mixed_execution"_s + completion<Mul>(MulRuntime<'dispatch, 'data>)
            / effect_mul_mixed,

        // Div: explicit decision states cover all guarded layout variants.
        "div_view_decision"_s <= "ready"_s + Div(DivRuntime<'dispatch, 'data>),
        "div_shape_decision"_s <= "div_view_decision"_s + completion<Div>(DivRuntime<'dispatch, 'data>)
            [guard_div_views_valid],
        "ready"_s <= "div_view_decision"_s + completion<Div>(DivRuntime<'dispatch, 'data>)
            [guard_div_views_invalid] / effect_div_view_reject,
        "div_layout_decision"_s <= "div_shape_decision"_s + completion<Div>(DivRuntime<'dispatch, 'data>)
            [guard_div_shape_valid],
        "ready"_s <= "div_shape_decision"_s + completion<Div>(DivRuntime<'dispatch, 'data>)
            [guard_div_shape_invalid] / effect_div_shape_reject,
        "div_explicit_execution"_s <= "div_layout_decision"_s + completion<Div>(DivRuntime<'dispatch, 'data>)
            [guard_div_explicit],
        "div_implicit_execution"_s <= "div_layout_decision"_s + completion<Div>(DivRuntime<'dispatch, 'data>)
            [guard_div_implicit],
        "div_mixed_execution"_s <= "div_layout_decision"_s + completion<Div>(DivRuntime<'dispatch, 'data>)
            [guard_div_mixed],
        "ready"_s <= "div_explicit_execution"_s + completion<Div>(DivRuntime<'dispatch, 'data>)
            / effect_div_explicit,
        "ready"_s <= "div_implicit_execution"_s + completion<Div>(DivRuntime<'dispatch, 'data>)
            / effect_div_implicit,
        "ready"_s <= "div_mixed_execution"_s + completion<Div>(DivRuntime<'dispatch, 'data>)
            / effect_div_mixed,

        // Unexpected routes use explicit transient states so their lifecycle
        // is visible to the generated machine before the completion join.
        "unexpected_dispatch"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected,
        "ready"_s <= "unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "unexpected_dispatch"_s + unexpected_event<_> / effect_generic_unexpected,
        "generic_unexpected_dispatch"_s <= "ready"_s + unexpected_event<_>
            / effect_generic_unexpected,
        "ready"_s <= "generic_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "generic_unexpected_dispatch"_s + unexpected_event<_>
            / effect_generic_unexpected,
        "ready"_s <= "add_view_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "add_shape_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "add_layout_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "add_explicit_execution"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "add_implicit_execution"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "add_mixed_execution"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "sub_view_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "sub_shape_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "sub_layout_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "sub_explicit_execution"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "sub_implicit_execution"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "sub_mixed_execution"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "mul_view_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "mul_shape_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "mul_layout_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "mul_explicit_execution"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "mul_implicit_execution"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "mul_mixed_execution"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "div_view_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "div_shape_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "div_layout_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "div_explicit_execution"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "div_implicit_execution"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "div_mixed_execution"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion actor for equal-count binary operations.
pub struct BinaryKernel {
    machine: BinaryMachineStateMachine<Context>,
}

impl fmt::Debug for BinaryKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BinaryKernel")
            .finish_non_exhaustive()
    }
}

impl Default for BinaryKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl BinaryKernel {
    /// Constructs an independent binary actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: BinaryMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    ///
    /// # Panics
    ///
    /// Panics if the generated binary machine does not return to `Ready`.
    pub fn process_event<E: BinaryEvent>(&mut self, event: E) -> E::Output {
        let result = event.dispatch(self);
        assert!(
            self.machine.is(&BinaryMachineStates::Ready),
            "binary machine must return to ready after dispatch"
        );
        result
    }

    /// Reports whether the generated binary machine is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&BinaryMachineStates::Ready)
    }
}

/// Events accepted by [`BinaryKernel`].
pub trait BinaryEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut BinaryKernel) -> Self::Output;
}

macro_rules! impl_binary_event {
    ($event:ident, $runtime:ident, $variant:ident) => {
        impl BinaryEvent for $event<'_> {
            type Output = BinaryResult;

            fn dispatch(self, actor: &mut BinaryKernel) -> Self::Output {
                let output = RefCell::new(self.output);
                let result = Cell::new(Err(BinaryError::UnexpectedEvent));
                actor
                    .machine
                    .process_event(BinaryMachineEvents::$variant($runtime {
                        lhs: self.lhs,
                        rhs: self.rhs,
                        output: &output,
                        result: &result,
                    }))
                    .map_err(|_| BinaryError::Internal)?;
                assert!(actor.is_ready());
                result.get()
            }
        }
    };
}

impl_binary_event!(OpAdd, AddRuntime, Add);
impl_binary_event!(OpSub, SubRuntime, Sub);
impl_binary_event!(OpMul, MulRuntime, Mul);
impl_binary_event!(OpDiv, DivRuntime, Div);

impl BinaryEvent for UnexpectedBinary {
    type Output = BinaryResult;

    fn dispatch(self, actor: &mut BinaryKernel) -> Self::Output {
        let result = Cell::new(Err(BinaryError::Internal));
        actor
            .machine
            .process_event(BinaryMachineEvents::Unexpected(UnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| BinaryError::Internal)?;
        assert!(actor.is_ready());
        result.get()
    }
}

fn views_valid(
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    output: &RefCell<TensorViewMut<'_>>,
) -> bool {
    let output = output.borrow();
    lhs.validate_binary().is_ok()
        && rhs.validate_binary().is_ok()
        && output.validate_binary().is_ok()
}

fn shape_valid(
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    output: &RefCell<TensorViewMut<'_>>,
) -> bool {
    let output = output.borrow();
    lhs.count() == rhs.count() && lhs.count() == output.count()
}

fn all_explicit(
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    output: &RefCell<TensorViewMut<'_>>,
) -> bool {
    let output = output.borrow();
    lhs.layout().nb()[0] != 0 && rhs.layout().nb()[0] != 0 && output.layout().nb()[0] != 0
}

fn all_implicit(
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    output: &RefCell<TensorViewMut<'_>>,
) -> bool {
    let output = output.borrow();
    lhs.layout().nb()[0] == 0 && rhs.layout().nb()[0] == 0 && output.layout().nb()[0] == 0
}

fn mixed_layout(
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    output: &RefCell<TensorViewMut<'_>>,
) -> bool {
    !all_explicit(lhs, rhs, output) && !all_implicit(lhs, rhs, output)
}

trait BinaryFormula {
    fn apply(lhs: f32, rhs: f32) -> f32;
}

struct AddFormula;
impl BinaryFormula for AddFormula {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}
struct SubFormula;
impl BinaryFormula for SubFormula {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs - rhs
    }
}
struct MulFormula;
impl BinaryFormula for MulFormula {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs * rhs
    }
}
struct DivFormula;
impl BinaryFormula for DivFormula {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs / rhs
    }
}

fn execute<F: BinaryFormula>(
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    output: &RefCell<TensorViewMut<'_>>,
    result: &Cell<BinaryResult>,
) {
    let mut output = output.borrow_mut();
    let mut ordinal = 0;
    while ordinal < output.count() {
        output.write(ordinal, F::apply(lhs.read(ordinal), rhs.read(ordinal)));
        ordinal += 1;
    }
    result.set(Ok(()));
}

macro_rules! impl_binary_context {
    ($views_valid:ident, $views_invalid:ident, $shape_valid:ident, $shape_invalid:ident,
        $explicit:ident, $implicit:ident, $mixed:ident, $effect_explicit:ident,
        $effect_implicit:ident, $effect_mixed:ident, $shape_reject:ident, $view_reject:ident,
        $runtime:ident, $formula:ty) => {
        fn $views_valid<'dispatch, 'data>(
            &self,
            event: &$runtime<'dispatch, 'data>,
        ) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(views_valid(&event.lhs, &event.rhs, event.output))
        }
        fn $views_invalid<'dispatch, 'data>(
            &self,
            event: &$runtime<'dispatch, 'data>,
        ) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(!views_valid(&event.lhs, &event.rhs, event.output))
        }
        fn $shape_valid<'dispatch, 'data>(
            &self,
            event: &$runtime<'dispatch, 'data>,
        ) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(shape_valid(&event.lhs, &event.rhs, event.output))
        }
        fn $shape_invalid<'dispatch, 'data>(
            &self,
            event: &$runtime<'dispatch, 'data>,
        ) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(!shape_valid(&event.lhs, &event.rhs, event.output))
        }
        fn $explicit<'dispatch, 'data>(
            &self,
            event: &$runtime<'dispatch, 'data>,
        ) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(all_explicit(&event.lhs, &event.rhs, event.output))
        }
        fn $implicit<'dispatch, 'data>(
            &self,
            event: &$runtime<'dispatch, 'data>,
        ) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(all_implicit(&event.lhs, &event.rhs, event.output))
        }
        fn $mixed<'dispatch, 'data>(&self, event: &$runtime<'dispatch, 'data>) -> Result<bool, ()>
        where
            'data: 'dispatch,
        {
            Ok(mixed_layout(&event.lhs, &event.rhs, event.output))
        }
        fn $effect_explicit<'dispatch, 'data>(
            &mut self,
            event: $runtime<'dispatch, 'data>,
        ) -> Result<(), ()>
        where
            'data: 'dispatch,
        {
            execute::<$formula>(&event.lhs, &event.rhs, event.output, event.result);
            Ok(())
        }
        fn $effect_implicit<'dispatch, 'data>(
            &mut self,
            event: $runtime<'dispatch, 'data>,
        ) -> Result<(), ()>
        where
            'data: 'dispatch,
        {
            execute::<$formula>(&event.lhs, &event.rhs, event.output, event.result);
            Ok(())
        }
        fn $effect_mixed<'dispatch, 'data>(
            &mut self,
            event: $runtime<'dispatch, 'data>,
        ) -> Result<(), ()>
        where
            'data: 'dispatch,
        {
            execute::<$formula>(&event.lhs, &event.rhs, event.output, event.result);
            Ok(())
        }
        fn $shape_reject<'dispatch, 'data>(
            &mut self,
            event: $runtime<'dispatch, 'data>,
        ) -> Result<(), ()>
        where
            'data: 'dispatch,
        {
            event.result.set(Err(BinaryError::ShapeMismatch));
            Ok(())
        }
        fn $view_reject<'dispatch, 'data>(
            &mut self,
            event: $runtime<'dispatch, 'data>,
        ) -> Result<(), ()>
        where
            'data: 'dispatch,
        {
            event.result.set(Err(BinaryError::InvalidView));
            Ok(())
        }
    };
}

impl BinaryMachineStateMachineContext for Context {
    impl_binary_context!(
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
        AddRuntime,
        AddFormula
    );
    impl_binary_context!(
        guard_sub_views_valid,
        guard_sub_views_invalid,
        guard_sub_shape_valid,
        guard_sub_shape_invalid,
        guard_sub_explicit,
        guard_sub_implicit,
        guard_sub_mixed,
        effect_sub_explicit,
        effect_sub_implicit,
        effect_sub_mixed,
        effect_sub_shape_reject,
        effect_sub_view_reject,
        SubRuntime,
        SubFormula
    );
    impl_binary_context!(
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
        MulRuntime,
        MulFormula
    );
    impl_binary_context!(
        guard_div_views_valid,
        guard_div_views_invalid,
        guard_div_shape_valid,
        guard_div_shape_invalid,
        guard_div_explicit,
        guard_div_implicit,
        guard_div_mixed,
        effect_div_explicit,
        effect_div_implicit,
        effect_div_mixed,
        effect_div_shape_reject,
        effect_div_view_reject,
        DivRuntime,
        DivFormula
    );

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(BinaryError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}
