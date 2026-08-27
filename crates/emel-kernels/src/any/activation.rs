//! Allocation-free F32 activation operations with explicit SML dispatch.
//!
//! The operation identities and state-machine rows follow the pinned
//! `emel.cpp` source at commit `843a117386ef17dc5a50549bbfc821074c2141d6`:
//! `src/emel/kernel/detail.hpp:45,55,73,93` lists `op_silu_back`, `op_scale`,
//! `op_clamp`, and `op_leaky_relu`; the x86-64 rows are in
//! `src/emel/kernel/x86_64/sm.hpp:296-303,456-463,671-678,881-888`.
//! The scalar scale loop is `detail.hpp:4262-4267`. Formula parity is checked
//! against the pinned ggml CPU implementation: `ggml-cpu/vec.h:1393-1406`
//! for `SiLU` backward and `vec.h:937` for leaky `ReLU`, with scale and clamp in
//! `ggml-cpu/ops.cpp:4374-4426` and `5448-5535`.
//!
//! Each operation has its own event, guard pair, and action. Dispatch only
//! borrows caller-owned tensor views; the actions perform bounded data-plane loops
//! and do not allocate, route, call another actor, or use `unsafe`.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::tensor_view::{TensorView, TensorViewMut};

/// Errors returned by an activation operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivationError {
    /// A source, gradient, or destination view failed validation.
    InvalidView,
    /// The views have different logical shapes.
    ShapeMismatch,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for ActivationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid activation-operation view"),
            Self::ShapeMismatch => formatter.write_str("activation-operation shapes differ"),
            Self::UnexpectedEvent => formatter.write_str("unexpected activation-operation event"),
            Self::Internal => formatter.write_str("internal activation-operation dispatch error"),
        }
    }
}

impl std::error::Error for ActivationError {}

/// Result returned after one activation event reaches run-to-completion.
pub type ActivationResult = Result<(), ActivationError>;

/// Scales each F32 element by `scale`.
#[derive(Debug)]
pub struct OpScale<'a> {
    input: TensorView<'a>,
    output: TensorViewMut<'a>,
    scale: f32,
}

impl<'a> OpScale<'a> {
    /// Creates a scale event. Shape validation occurs in the actor guard.
    #[must_use]
    pub const fn new(input: TensorView<'a>, output: TensorViewMut<'a>, scale: f32) -> Self {
        Self {
            input,
            output,
            scale,
        }
    }
}

/// Clamps each F32 element to the inclusive `[min, max]` interval.
#[derive(Debug)]
pub struct OpClamp<'a> {
    input: TensorView<'a>,
    output: TensorViewMut<'a>,
    min: f32,
    max: f32,
}

impl<'a> OpClamp<'a> {
    /// Creates a clamp event. Shape validation occurs in the actor guard.
    #[must_use]
    pub const fn new(input: TensorView<'a>, output: TensorViewMut<'a>, min: f32, max: f32) -> Self {
        Self {
            input,
            output,
            min,
            max,
        }
    }
}

/// Applies the derivative of `SiLU` to `input`, weighted by `gradient`.
#[derive(Debug)]
pub struct OpSiluBack<'a> {
    input: TensorView<'a>,
    gradient: TensorView<'a>,
    output: TensorViewMut<'a>,
}

impl<'a> OpSiluBack<'a> {
    /// Creates a SiLU-backward event. Shape validation occurs in the guard.
    #[must_use]
    pub const fn new(
        input: TensorView<'a>,
        gradient: TensorView<'a>,
        output: TensorViewMut<'a>,
    ) -> Self {
        Self {
            input,
            gradient,
            output,
        }
    }
}

/// Applies leaky `ReLU` with `negative_slope` to each F32 element.
#[derive(Debug)]
pub struct OpLeakyRelu<'a> {
    input: TensorView<'a>,
    output: TensorViewMut<'a>,
    negative_slope: f32,
}

impl<'a> OpLeakyRelu<'a> {
    /// Creates a leaky-ReLU event. Shape validation occurs in the actor guard.
    #[must_use]
    pub const fn new(
        input: TensorView<'a>,
        output: TensorViewMut<'a>,
        negative_slope: f32,
    ) -> Self {
        Self {
            input,
            output,
            negative_slope,
        }
    }
}

struct ScaleRuntime<'a> {
    event: OpScale<'a>,
    result: &'a Cell<ActivationResult>,
}

struct ClampRuntime<'a> {
    event: OpClamp<'a>,
    result: &'a Cell<ActivationResult>,
}

struct SiluBackRuntime<'a> {
    event: OpSiluBack<'a>,
    result: &'a Cell<ActivationResult>,
}

struct LeakyReluRuntime<'a> {
    event: OpLeakyRelu<'a>,
    result: &'a Cell<ActivationResult>,
}

#[derive(Default)]
struct Context;

sml! {
    ActivationMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Scale(ScaleRuntime<'dispatch>)
            [guard_scale_valid] / effect_scale_execute,
        "ready"_s <= "ready"_s + Scale(ScaleRuntime<'dispatch>)
            [guard_scale_shape_invalid] / effect_scale_shape_reject,
        "ready"_s <= "ready"_s + Scale(ScaleRuntime<'dispatch>)
            [guard_scale_view_invalid] / effect_scale_view_reject,

        "ready"_s <= "ready"_s + Clamp(ClampRuntime<'dispatch>)
            [guard_clamp_valid] / effect_clamp_execute,
        "ready"_s <= "ready"_s + Clamp(ClampRuntime<'dispatch>)
            [guard_clamp_shape_invalid] / effect_clamp_shape_reject,
        "ready"_s <= "ready"_s + Clamp(ClampRuntime<'dispatch>)
            [guard_clamp_view_invalid] / effect_clamp_view_reject,

        "ready"_s <= "ready"_s + SiluBack(SiluBackRuntime<'dispatch>)
            [guard_silu_back_valid] / effect_silu_back_execute,
        "ready"_s <= "ready"_s + SiluBack(SiluBackRuntime<'dispatch>)
            [guard_silu_back_shape_invalid] / effect_silu_back_shape_reject,
        "ready"_s <= "ready"_s + SiluBack(SiluBackRuntime<'dispatch>)
            [guard_silu_back_view_invalid] / effect_silu_back_view_reject,

        "ready"_s <= "ready"_s + LeakyRelu(LeakyReluRuntime<'dispatch>)
            [guard_leaky_relu_valid] / effect_leaky_relu_execute,
        "ready"_s <= "ready"_s + LeakyRelu(LeakyReluRuntime<'dispatch>)
            [guard_leaky_relu_shape_invalid] / effect_leaky_relu_shape_reject,
        "ready"_s <= "ready"_s + LeakyRelu(LeakyReluRuntime<'dispatch>)
            [guard_leaky_relu_view_invalid] / effect_leaky_relu_view_reject,

        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, run-to-completion actor for F32 activation operations.
pub struct ActivationKernel {
    machine: ActivationMachineStateMachine<Context>,
}

impl fmt::Debug for ActivationKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ActivationKernel")
            .finish_non_exhaustive()
    }
}

impl Default for ActivationKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivationKernel {
    /// Constructs an independent activation actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: ActivationMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    ///
    /// # Errors
    ///
    /// Returns the event's [`ActivationError`] when its slices are invalid or
    /// the generated machine cannot complete dispatch.
    pub fn process_event<E: ActivationEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn scale(&mut self, event: OpScale<'_>) -> ActivationResult {
        let result = Cell::new(Err(ActivationError::UnexpectedEvent));
        self.machine
            .process_event(ActivationMachineEvents::Scale(ScaleRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ActivationError::Internal)?;
        result.get()
    }

    fn clamp(&mut self, event: OpClamp<'_>) -> ActivationResult {
        let result = Cell::new(Err(ActivationError::UnexpectedEvent));
        self.machine
            .process_event(ActivationMachineEvents::Clamp(ClampRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ActivationError::Internal)?;
        result.get()
    }

    fn silu_back(&mut self, event: OpSiluBack<'_>) -> ActivationResult {
        let result = Cell::new(Err(ActivationError::UnexpectedEvent));
        self.machine
            .process_event(ActivationMachineEvents::SiluBack(SiluBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ActivationError::Internal)?;
        result.get()
    }

    fn leaky_relu(&mut self, event: OpLeakyRelu<'_>) -> ActivationResult {
        let result = Cell::new(Err(ActivationError::UnexpectedEvent));
        self.machine
            .process_event(ActivationMachineEvents::LeakyRelu(LeakyReluRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ActivationError::Internal)?;
        result.get()
    }
}

/// Typed event accepted by [`ActivationKernel`].
pub trait ActivationEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut ActivationKernel) -> Self::Output;
}

impl ActivationEvent for OpScale<'_> {
    type Output = ActivationResult;

    fn dispatch(self, actor: &mut ActivationKernel) -> Self::Output {
        actor.scale(self)
    }
}

impl ActivationEvent for OpClamp<'_> {
    type Output = ActivationResult;

    fn dispatch(self, actor: &mut ActivationKernel) -> Self::Output {
        actor.clamp(self)
    }
}

impl ActivationEvent for OpSiluBack<'_> {
    type Output = ActivationResult;

    fn dispatch(self, actor: &mut ActivationKernel) -> Self::Output {
        actor.silu_back(self)
    }
}

impl ActivationEvent for OpLeakyRelu<'_> {
    type Output = ActivationResult;

    fn dispatch(self, actor: &mut ActivationKernel) -> Self::Output {
        actor.leaky_relu(self)
    }
}

fn views_valid(input: &TensorView<'_>, output: &TensorViewMut<'_>) -> bool {
    input.validate().is_ok() && output.validate().is_ok()
}

fn same_shape(input: &TensorView<'_>, output: &TensorViewMut<'_>) -> bool {
    input.layout().ne() == output.layout().ne()
}

fn three_views_valid(
    input: &TensorView<'_>,
    gradient: &TensorView<'_>,
    output: &TensorViewMut<'_>,
) -> bool {
    input.validate().is_ok() && gradient.validate().is_ok() && output.validate().is_ok()
}

fn three_views_same_shape(
    input: &TensorView<'_>,
    gradient: &TensorView<'_>,
    output: &TensorViewMut<'_>,
) -> bool {
    input.layout().ne() == gradient.layout().ne() && input.layout().ne() == output.layout().ne()
}

impl ActivationMachineStateMachineContext for Context {
    fn guard_scale_valid(&self, event: &ScaleRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event.input, &event.event.output)
            && same_shape(&event.event.input, &event.event.output))
    }

    fn guard_scale_shape_invalid(&self, event: &ScaleRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event.input, &event.event.output)
            && !same_shape(&event.event.input, &event.event.output))
    }

    fn guard_scale_view_invalid(&self, event: &ScaleRuntime<'_>) -> Result<bool, ()> {
        Ok(!views_valid(&event.event.input, &event.event.output))
    }

    fn effect_scale_execute(&mut self, mut event: ScaleRuntime<'_>) -> Result<(), ()> {
        let mut index = 0;
        while index < event.event.input.count() {
            event
                .event
                .output
                .write(index, event.event.input.read(index) * event.event.scale);
            index += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_scale_shape_reject(&mut self, event: ScaleRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ActivationError::ShapeMismatch));
        Ok(())
    }

    fn effect_scale_view_reject(&mut self, event: ScaleRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ActivationError::InvalidView));
        Ok(())
    }

    fn guard_clamp_valid(&self, event: &ClampRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event.input, &event.event.output)
            && same_shape(&event.event.input, &event.event.output))
    }

    fn guard_clamp_shape_invalid(&self, event: &ClampRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event.input, &event.event.output)
            && !same_shape(&event.event.input, &event.event.output))
    }

    fn guard_clamp_view_invalid(&self, event: &ClampRuntime<'_>) -> Result<bool, ()> {
        Ok(!views_valid(&event.event.input, &event.event.output))
    }

    fn effect_clamp_execute(&mut self, mut event: ClampRuntime<'_>) -> Result<(), ()> {
        let mut index = 0;
        while index < event.event.input.count() {
            let value = event.event.input.read(index);
            // Match the pinned C++ `std::clamp` comparison contract. In
            // particular, both comparisons are false for NaN, so NaN passes
            // through instead of being canonicalized to a bound by
            // `f32::min`/`f32::max`.
            let clamped = if value < event.event.min {
                event.event.min
            } else if value > event.event.max {
                event.event.max
            } else {
                value
            };
            event.event.output.write(index, clamped);
            index += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_clamp_shape_reject(&mut self, event: ClampRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ActivationError::ShapeMismatch));
        Ok(())
    }

    fn effect_clamp_view_reject(&mut self, event: ClampRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ActivationError::InvalidView));
        Ok(())
    }

    fn guard_silu_back_valid(&self, event: &SiluBackRuntime<'_>) -> Result<bool, ()> {
        Ok(three_views_valid(
            &event.event.input,
            &event.event.gradient,
            &event.event.output,
        ) && three_views_same_shape(
            &event.event.input,
            &event.event.gradient,
            &event.event.output,
        ))
    }

    fn guard_silu_back_shape_invalid(&self, event: &SiluBackRuntime<'_>) -> Result<bool, ()> {
        Ok(three_views_valid(
            &event.event.input,
            &event.event.gradient,
            &event.event.output,
        ) && !three_views_same_shape(
            &event.event.input,
            &event.event.gradient,
            &event.event.output,
        ))
    }

    fn guard_silu_back_view_invalid(&self, event: &SiluBackRuntime<'_>) -> Result<bool, ()> {
        Ok(!three_views_valid(
            &event.event.input,
            &event.event.gradient,
            &event.event.output,
        ))
    }

    fn effect_silu_back_execute(&mut self, mut event: SiluBackRuntime<'_>) -> Result<(), ()> {
        let mut index = 0;
        while index < event.event.input.count() {
            let value = event.event.input.read(index);
            let sigmoid = 1.0 / (1.0 + (-value).exp());
            let derivative = sigmoid * (1.0 + value * (1.0 - sigmoid));
            event
                .event
                .output
                .write(index, event.event.gradient.read(index) * derivative);
            index += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_silu_back_shape_reject(&mut self, event: SiluBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ActivationError::ShapeMismatch));
        Ok(())
    }

    fn effect_silu_back_view_reject(&mut self, event: SiluBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ActivationError::InvalidView));
        Ok(())
    }

    fn guard_leaky_relu_valid(&self, event: &LeakyReluRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event.input, &event.event.output)
            && same_shape(&event.event.input, &event.event.output))
    }

    fn guard_leaky_relu_shape_invalid(&self, event: &LeakyReluRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event.input, &event.event.output)
            && !same_shape(&event.event.input, &event.event.output))
    }

    fn guard_leaky_relu_view_invalid(&self, event: &LeakyReluRuntime<'_>) -> Result<bool, ()> {
        Ok(!views_valid(&event.event.input, &event.event.output))
    }

    fn effect_leaky_relu_execute(&mut self, mut event: LeakyReluRuntime<'_>) -> Result<(), ()> {
        let mut index = 0;
        while index < event.event.input.count() {
            let value = event.event.input.read(index);
            let output = if value < 0.0 {
                value * event.event.negative_slope
            } else {
                value
            };
            event.event.output.write(index, output);
            index += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_leaky_relu_shape_reject(&mut self, event: LeakyReluRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ActivationError::ShapeMismatch));
        Ok(())
    }

    fn effect_leaky_relu_view_reject(&mut self, event: LeakyReluRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ActivationError::InvalidView));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}
