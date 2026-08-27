//! Callback custom/map kernels.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_map_custom1`, `op_map_custom2`, `op_map_custom3`, and `op_custom`
//! and names `exec_op_*` routes. Those routes invoke host function pointers
//! that this crate cannot own. Geometry-valid events therefore reject with
//! [`CustomError::NoCallback`] and write nothing.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by custom/map dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CustomError {
    /// Buffer lengths do not match.
    InvalidShape,
    /// No host callback is registered for this source-contract route.
    NoCallback,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for CustomError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid custom shape"),
            Self::NoCallback => formatter.write_str("custom kernel has no host callback"),
            Self::UnexpectedEvent => formatter.write_str("unexpected custom event"),
            Self::Internal => formatter.write_str("internal custom dispatch error"),
        }
    }
}

impl std::error::Error for CustomError {}

/// Result returned by custom/map dispatch.
pub type CustomResult = Result<(), CustomError>;

/// Unary custom map. C++ `op_map_custom1`.
#[derive(Debug)]
pub struct OpMapCustom1<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl<'a> OpMapCustom1<'a> {
    /// Creates a unary custom-map request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], output: &'a mut [f32]) -> Self {
        Self { input, output }
    }
    const fn valid(&self) -> bool {
        !self.input.is_empty() && self.input.len() == self.output.len()
    }
}

/// Binary custom map. C++ `op_map_custom2`.
#[derive(Debug)]
pub struct OpMapCustom2<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
    output: &'a mut [f32],
}

impl<'a> OpMapCustom2<'a> {
    /// Creates a binary custom-map request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(lhs: &'a [f32], rhs: &'a [f32], output: &'a mut [f32]) -> Self {
        Self { lhs, rhs, output }
    }
    const fn valid(&self) -> bool {
        !self.lhs.is_empty()
            && self.lhs.len() == self.rhs.len()
            && self.lhs.len() == self.output.len()
    }
}

/// Ternary custom map. C++ `op_map_custom3`.
#[derive(Debug)]
pub struct OpMapCustom3<'a> {
    first: &'a [f32],
    second: &'a [f32],
    third: &'a [f32],
    output: &'a mut [f32],
}

impl<'a> OpMapCustom3<'a> {
    /// Creates a ternary custom-map request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        first: &'a [f32],
        second: &'a [f32],
        third: &'a [f32],
        output: &'a mut [f32],
    ) -> Self {
        Self {
            first,
            second,
            third,
            output,
        }
    }
    const fn valid(&self) -> bool {
        !self.first.is_empty()
            && self.first.len() == self.second.len()
            && self.first.len() == self.third.len()
            && self.first.len() == self.output.len()
    }
}

/// Generic custom kernel. C++ `op_custom`.
#[derive(Debug)]
pub struct OpCustom<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl<'a> OpCustom<'a> {
    /// Creates a custom request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], output: &'a mut [f32]) -> Self {
        Self { input, output }
    }
    const fn valid(&self) -> bool {
        !self.input.is_empty() && self.input.len() == self.output.len()
    }
}

struct Map1Runtime<'a> {
    event: OpMapCustom1<'a>,
    result: &'a Cell<CustomResult>,
}
struct Map2Runtime<'a> {
    event: OpMapCustom2<'a>,
    result: &'a Cell<CustomResult>,
}
struct Map3Runtime<'a> {
    event: OpMapCustom3<'a>,
    result: &'a Cell<CustomResult>,
}
struct CustomRuntime<'a> {
    event: OpCustom<'a>,
    result: &'a Cell<CustomResult>,
}

#[derive(Default)]
struct Context;

sml! {
    CustomMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Map1(Map1Runtime<'dispatch>) [guard_map1_valid] / effect_map1_no_callback,
        "ready"_s <= "ready"_s + Map1(Map1Runtime<'dispatch>) [guard_map1_invalid] / effect_map1_reject,
        "ready"_s <= "ready"_s + Map2(Map2Runtime<'dispatch>) [guard_map2_valid] / effect_map2_no_callback,
        "ready"_s <= "ready"_s + Map2(Map2Runtime<'dispatch>) [guard_map2_invalid] / effect_map2_reject,
        "ready"_s <= "ready"_s + Map3(Map3Runtime<'dispatch>) [guard_map3_valid] / effect_map3_no_callback,
        "ready"_s <= "ready"_s + Map3(Map3Runtime<'dispatch>) [guard_map3_invalid] / effect_map3_reject,
        "ready"_s <= "ready"_s + Custom(CustomRuntime<'dispatch>) [guard_custom_valid] / effect_custom_no_callback,
        "ready"_s <= "ready"_s + Custom(CustomRuntime<'dispatch>) [guard_custom_invalid] / effect_custom_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for custom events.
pub trait CustomEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut CustomKernel) -> Self::Output;
}

/// Single-writer custom/map actor.
pub struct CustomKernel {
    machine: CustomMachineStateMachine<Context>,
}

impl fmt::Debug for CustomKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CustomKernel")
            .finish_non_exhaustive()
    }
}

impl Default for CustomKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomKernel {
    /// Constructs an independent custom actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: CustomMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed custom event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`CustomError::InvalidShape`] on length mismatch and
    /// [`CustomError::NoCallback`] when geometry is valid but no host
    /// callback exists.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: CustomEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn map1(&mut self, event: OpMapCustom1<'_>) -> CustomResult {
        let result = Cell::new(Err(CustomError::UnexpectedEvent));
        self.machine
            .process_event(CustomMachineEvents::Map1(Map1Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| CustomError::Internal)?;
        ready(self);
        result.get()
    }
    fn map2(&mut self, event: OpMapCustom2<'_>) -> CustomResult {
        let result = Cell::new(Err(CustomError::UnexpectedEvent));
        self.machine
            .process_event(CustomMachineEvents::Map2(Map2Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| CustomError::Internal)?;
        ready(self);
        result.get()
    }
    fn map3(&mut self, event: OpMapCustom3<'_>) -> CustomResult {
        let result = Cell::new(Err(CustomError::UnexpectedEvent));
        self.machine
            .process_event(CustomMachineEvents::Map3(Map3Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| CustomError::Internal)?;
        ready(self);
        result.get()
    }
    fn custom(&mut self, event: OpCustom<'_>) -> CustomResult {
        let result = Cell::new(Err(CustomError::UnexpectedEvent));
        self.machine
            .process_event(CustomMachineEvents::Custom(CustomRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| CustomError::Internal)?;
        ready(self);
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&CustomMachineStates::Ready)
    }
}

fn ready(kernel: &CustomKernel) {
    assert!(
        kernel.machine.is(&CustomMachineStates::Ready),
        "custom machine must return to ready after dispatch"
    );
}

impl CustomEvent for OpMapCustom1<'_> {
    type Output = CustomResult;
    fn dispatch(self, kernel: &mut CustomKernel) -> Self::Output {
        kernel.map1(self)
    }
}
impl CustomEvent for OpMapCustom2<'_> {
    type Output = CustomResult;
    fn dispatch(self, kernel: &mut CustomKernel) -> Self::Output {
        kernel.map2(self)
    }
}
impl CustomEvent for OpMapCustom3<'_> {
    type Output = CustomResult;
    fn dispatch(self, kernel: &mut CustomKernel) -> Self::Output {
        kernel.map3(self)
    }
}
impl CustomEvent for OpCustom<'_> {
    type Output = CustomResult;
    fn dispatch(self, kernel: &mut CustomKernel) -> Self::Output {
        kernel.custom(self)
    }
}

impl CustomMachineStateMachineContext for Context {
    fn guard_map1_valid(&self, event: &Map1Runtime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_map1_invalid(&self, event: &Map1Runtime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn guard_map2_valid(&self, event: &Map2Runtime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_map2_invalid(&self, event: &Map2Runtime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn guard_map3_valid(&self, event: &Map3Runtime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_map3_invalid(&self, event: &Map3Runtime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn guard_custom_valid(&self, event: &CustomRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_custom_invalid(&self, event: &CustomRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn effect_map1_no_callback(&mut self, event: Map1Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(CustomError::NoCallback));
        Ok(())
    }
    fn effect_map1_reject(&mut self, event: Map1Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(CustomError::InvalidShape));
        Ok(())
    }
    fn effect_map2_no_callback(&mut self, event: Map2Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(CustomError::NoCallback));
        Ok(())
    }
    fn effect_map2_reject(&mut self, event: Map2Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(CustomError::InvalidShape));
        Ok(())
    }
    fn effect_map3_no_callback(&mut self, event: Map3Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(CustomError::NoCallback));
        Ok(())
    }
    fn effect_map3_reject(&mut self, event: Map3Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(CustomError::InvalidShape));
        Ok(())
    }
    fn effect_custom_no_callback(&mut self, event: CustomRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(CustomError::NoCallback));
        Ok(())
    }
    fn effect_custom_reject(&mut self, event: CustomRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(CustomError::InvalidShape));
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{CustomError, CustomKernel, OpCustom, OpMapCustom1, OpMapCustom2, OpMapCustom3};
    use crate::Kernel;

    #[test]
    fn map_custom1_geometry_valid_has_no_callback() {
        let input = [1.0_f32, 2.0];
        let mut output = [9.0_f32; 2];
        let mut kernel = CustomKernel::new();
        assert_eq!(
            kernel.process_event(OpMapCustom1::new(&input, &mut output)),
            Err(CustomError::NoCallback)
        );
        assert_eq!(output[0].to_bits(), 9.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn public_kernel_custom_rejects_shape() {
        let input = [1.0_f32];
        let mut output = [9.0_f32; 2];
        assert_eq!(
            Kernel::new().process_event(OpCustom::new(&input, &mut output)),
            Err(CustomError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 9.0_f32.to_bits());
    }

    #[test]
    fn map_custom_family_rejects_callback_and_shape() {
        let lhs = [1.0_f32, 2.0];
        let rhs = [3.0_f32, 4.0];
        let third = [5.0_f32, 6.0];
        let mut output = [9.0_f32; 2];
        let mut kernel = CustomKernel::new();
        assert_eq!(
            kernel.process_event(OpMapCustom2::new(&lhs, &rhs, &mut output)),
            Err(CustomError::NoCallback)
        );
        assert_eq!(
            kernel.process_event(OpMapCustom3::new(&lhs, &rhs, &third, &mut output)),
            Err(CustomError::NoCallback)
        );
        assert_eq!(
            kernel.process_event(OpCustom::new(&lhs, &mut output)),
            Err(CustomError::NoCallback)
        );
        assert_eq!(output[0].to_bits(), 9.0_f32.to_bits());
        let short = [1.0_f32];
        assert_eq!(
            kernel.process_event(OpMapCustom1::new(&short, &mut output)),
            Err(CustomError::InvalidShape)
        );
        assert_eq!(
            kernel.process_event(OpMapCustom2::new(&short, &rhs, &mut output)),
            Err(CustomError::InvalidShape)
        );
        assert_eq!(
            kernel.process_event(OpMapCustom3::new(&short, &rhs, &third, &mut output)),
            Err(CustomError::InvalidShape)
        );
        assert_eq!(
            CustomError::NoCallback.to_string(),
            "custom kernel has no host callback"
        );
        assert_eq!(
            CustomError::InvalidShape.to_string(),
            "invalid custom shape"
        );
        assert_eq!(
            CustomError::UnexpectedEvent.to_string(),
            "unexpected custom event"
        );
        assert_eq!(
            CustomError::Internal.to_string(),
            "internal custom dispatch error"
        );
        let _ = format!("{:?}", CustomKernel::default());
        assert!(kernel.is_ready());
    }

    #[test]
    fn public_kernel_map_custom_family_has_no_callback() {
        let input = [1.0_f32];
        let rhs = [2.0_f32];
        let third = [3.0_f32];
        let mut output = [9.0_f32; 1];
        let mut kernel = Kernel::new();
        assert_eq!(
            kernel.process_event(OpMapCustom1::new(&input, &mut output)),
            Err(CustomError::NoCallback)
        );
        assert_eq!(
            kernel.process_event(OpMapCustom2::new(&input, &rhs, &mut output)),
            Err(CustomError::NoCallback)
        );
        assert_eq!(
            kernel.process_event(OpMapCustom3::new(&input, &rhs, &third, &mut output)),
            Err(CustomError::NoCallback)
        );
    }
}
