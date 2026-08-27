//! 1-D pad, reflect-pad, and triangular mask kernels over dense F32 buffers.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_pad`, `op_pad_reflect_1d`, and `op_tri` and routes them on the
//! `aarch64`/`x86_64` machines. Those machines name `exec_op_pad*` /
//! `exec_op_tri` actions, but the pinned action headers do not define them.
//! This actor is therefore a source-contract implementation.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by pad and triangular dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PadError {
    /// The pad widths or matrix geometry are invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for PadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid pad/tri shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected pad/tri event"),
            Self::Internal => formatter.write_str("internal pad/tri dispatch error"),
        }
    }
}

impl std::error::Error for PadError {}

/// Result returned by pad and triangular dispatch.
pub type PadResult = Result<(), PadError>;

/// Pads a dense F32 vector on both sides with a constant.
#[derive(Debug)]
pub struct OpPad<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    left: usize,
    right: usize,
    value: f32,
}

impl<'a> OpPad<'a> {
    /// Creates a constant-pad request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        input: &'a [f32],
        output: &'a mut [f32],
        left: usize,
        right: usize,
        value: f32,
    ) -> Self {
        Self {
            input,
            output,
            left,
            right,
            value,
        }
    }

    const fn valid(&self) -> bool {
        !self.input.is_empty()
            && self.output.len()
                == self
                    .input
                    .len()
                    .saturating_add(self.left)
                    .saturating_add(self.right)
    }
}

/// Reflect-pads a dense F32 vector on both sides.
#[derive(Debug)]
pub struct OpPadReflect1d<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    left: usize,
    right: usize,
}

impl<'a> OpPadReflect1d<'a> {
    /// Creates a reflect-pad request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], output: &'a mut [f32], left: usize, right: usize) -> Self {
        Self {
            input,
            output,
            left,
            right,
        }
    }

    const fn valid(&self) -> bool {
        self.input.len() > 1
            && self.left < self.input.len()
            && self.right < self.input.len()
            && self.output.len()
                == self
                    .input
                    .len()
                    .saturating_add(self.left)
                    .saturating_add(self.right)
    }
}

/// Keeps the lower triangle of a square matrix, including the diagonal.
#[derive(Debug)]
pub struct OpTri<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl<'a> OpTri<'a> {
    /// Creates a triangular-mask request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], output: &'a mut [f32]) -> Self {
        Self { input, output }
    }
}

const fn square_side(len: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let mut side = 1_usize;
    while side.saturating_mul(side) < len {
        side += 1;
    }
    if side.saturating_mul(side) == len {
        Some(side)
    } else {
        None
    }
}

const fn reflect_index(index: isize, len: usize) -> usize {
    let last = (len - 1).cast_signed();
    let period = last * 2;
    let mut wrapped = index % period;
    if wrapped < 0 {
        wrapped += period;
    }
    if wrapped > last {
        (period - wrapped).cast_unsigned()
    } else {
        wrapped.cast_unsigned()
    }
}

struct PadRuntime<'a> {
    event: OpPad<'a>,
    result: &'a Cell<PadResult>,
}

struct ReflectRuntime<'a> {
    event: OpPadReflect1d<'a>,
    result: &'a Cell<PadResult>,
}

struct TriRuntime<'a> {
    event: OpTri<'a>,
    result: &'a Cell<PadResult>,
}

#[derive(Default)]
struct Context;

sml! {
    PadMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Pad(PadRuntime<'dispatch>) [guard_pad_valid] / effect_pad,
        "ready"_s <= "ready"_s + Pad(PadRuntime<'dispatch>) [guard_pad_invalid] / effect_pad_reject,
        "ready"_s <= "ready"_s + Reflect(ReflectRuntime<'dispatch>) [guard_reflect_valid] / effect_reflect,
        "ready"_s <= "ready"_s + Reflect(ReflectRuntime<'dispatch>) [guard_reflect_invalid] / effect_reflect_reject,
        "ready"_s <= "ready"_s + Tri(TriRuntime<'dispatch>) [guard_tri_valid] / effect_tri,
        "ready"_s <= "ready"_s + Tri(TriRuntime<'dispatch>) [guard_tri_invalid] / effect_tri_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for pad-family events.
pub trait PadEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut PadKernel) -> Self::Output;
}

/// Single-writer pad and triangular actor.
pub struct PadKernel {
    machine: PadMachineStateMachine<Context>,
}

impl fmt::Debug for PadKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("PadKernel").finish_non_exhaustive()
    }
}

impl Default for PadKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl PadKernel {
    /// Constructs an independent pad actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: PadMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed pad-family event run-to-completion.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: PadEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn pad(&mut self, event: OpPad<'_>) -> PadResult {
        let result = Cell::new(Err(PadError::UnexpectedEvent));
        self.machine
            .process_event(PadMachineEvents::Pad(PadRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PadError::Internal)?;
        assert!(
            self.machine.is(&PadMachineStates::Ready),
            "pad machine must return to ready after dispatch"
        );
        result.get()
    }

    fn reflect(&mut self, event: OpPadReflect1d<'_>) -> PadResult {
        let result = Cell::new(Err(PadError::UnexpectedEvent));
        self.machine
            .process_event(PadMachineEvents::Reflect(ReflectRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PadError::Internal)?;
        assert!(
            self.machine.is(&PadMachineStates::Ready),
            "pad machine must return to ready after dispatch"
        );
        result.get()
    }

    fn tri(&mut self, event: OpTri<'_>) -> PadResult {
        let result = Cell::new(Err(PadError::UnexpectedEvent));
        self.machine
            .process_event(PadMachineEvents::Tri(TriRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PadError::Internal)?;
        assert!(
            self.machine.is(&PadMachineStates::Ready),
            "pad machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&PadMachineStates::Ready)
    }
}

impl PadEvent for OpPad<'_> {
    type Output = PadResult;
    fn dispatch(self, kernel: &mut PadKernel) -> Self::Output {
        kernel.pad(self)
    }
}

impl PadEvent for OpPadReflect1d<'_> {
    type Output = PadResult;
    fn dispatch(self, kernel: &mut PadKernel) -> Self::Output {
        kernel.reflect(self)
    }
}

impl PadEvent for OpTri<'_> {
    type Output = PadResult;
    fn dispatch(self, kernel: &mut PadKernel) -> Self::Output {
        kernel.tri(self)
    }
}

impl PadMachineStateMachineContext for Context {
    fn guard_pad_valid(&self, event: &PadRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_pad_invalid(&self, event: &PadRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_reflect_valid(&self, event: &ReflectRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_reflect_invalid(&self, event: &ReflectRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_tri_valid(&self, event: &TriRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.input.len() == event.event.output.len()
            && square_side(event.event.input.len()).is_some())
    }

    fn guard_tri_invalid(&self, event: &TriRuntime<'_>) -> Result<bool, ()> {
        Ok(!(event.event.input.len() == event.event.output.len()
            && square_side(event.event.input.len()).is_some()))
    }

    fn effect_pad(&mut self, event: PadRuntime<'_>) -> Result<(), ()> {
        let left = event.event.left;
        event.event.output[..left].fill(event.event.value);
        event.event.output[left..left + event.event.input.len()].copy_from_slice(event.event.input);
        event.event.output[left + event.event.input.len()..].fill(event.event.value);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_pad_reject(&mut self, event: PadRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(PadError::InvalidShape));
        Ok(())
    }

    fn effect_reflect(&mut self, event: ReflectRuntime<'_>) -> Result<(), ()> {
        let input_len = event.event.input.len();
        let left = event.event.left.cast_signed();
        for (out_index, slot) in event.event.output.iter_mut().enumerate() {
            let source = out_index.cast_signed() - left;
            *slot = event.event.input[reflect_index(source, input_len)];
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_reflect_reject(&mut self, event: ReflectRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(PadError::InvalidShape));
        Ok(())
    }

    fn effect_tri(&mut self, event: TriRuntime<'_>) -> Result<(), ()> {
        let Some(side) = square_side(event.event.input.len()) else {
            event.result.set(Err(PadError::InvalidShape));
            return Ok(());
        };
        for row in 0..side {
            for col in 0..side {
                let index = row * side + col;
                event.event.output[index] = if row >= col {
                    event.event.input[index]
                } else {
                    0.0
                };
            }
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_tri_reject(&mut self, event: TriRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(PadError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{OpPad, OpPadReflect1d, OpTri, PadError, PadKernel};
    use crate::Kernel;

    #[test]
    fn pad_writes_constant_shoulders() {
        let input = [1.0_f32, 2.0];
        let mut output = [0.0_f32; 5];
        let mut kernel = PadKernel::new();
        kernel
            .process_event(OpPad::new(&input, &mut output, 1, 2, 9.0))
            .unwrap();
        assert_eq!(output[0].to_bits(), 9.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 1.0_f32.to_bits());
        assert_eq!(output[2].to_bits(), 2.0_f32.to_bits());
        assert_eq!(output[3].to_bits(), 9.0_f32.to_bits());
        assert_eq!(output[4].to_bits(), 9.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn reflect_pad_mirrors_interior_samples() {
        let input = [1.0_f32, 2.0, 3.0];
        let mut output = [0.0_f32; 5];
        let mut kernel = PadKernel::new();
        kernel
            .process_event(OpPadReflect1d::new(&input, &mut output, 1, 1))
            .unwrap();
        assert_eq!(output[0].to_bits(), 2.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 1.0_f32.to_bits());
        assert_eq!(output[2].to_bits(), 2.0_f32.to_bits());
        assert_eq!(output[3].to_bits(), 3.0_f32.to_bits());
        assert_eq!(output[4].to_bits(), 2.0_f32.to_bits());
    }

    #[test]
    fn tri_clears_the_strict_upper_triangle() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        let mut kernel = PadKernel::new();
        kernel
            .process_event(OpTri::new(&input, &mut output))
            .unwrap();
        assert_eq!(output[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 0.0_f32.to_bits());
        assert_eq!(output[2].to_bits(), 3.0_f32.to_bits());
        assert_eq!(output[3].to_bits(), 4.0_f32.to_bits());
    }

    #[test]
    fn pad_rejects_length_mismatch_without_mutation() {
        let input = [1.0_f32];
        let mut output = [5.0_f32; 2];
        let mut kernel = PadKernel::new();
        assert_eq!(
            kernel.process_event(OpPad::new(&input, &mut output, 0, 0, 1.0)),
            Err(PadError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 5.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn public_kernel_dispatches_pad_family() {
        let input = [1.0_f32, 2.0];
        let mut output = [0.0_f32; 5];
        let mut kernel = Kernel::new();
        kernel
            .process_event(OpPad::new(&input, &mut output, 1, 2, 9.0))
            .unwrap();
        assert_eq!(output[0].to_bits(), 9.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 1.0_f32.to_bits());
        assert_eq!(output[2].to_bits(), 2.0_f32.to_bits());
        assert_eq!(output[3].to_bits(), 9.0_f32.to_bits());
        assert_eq!(output[4].to_bits(), 9.0_f32.to_bits());
        assert!(kernel.is_ready());

        let input = [1.0_f32, 2.0, 3.0];
        let mut reflected = [0.0_f32; 5];
        kernel
            .process_event(OpPadReflect1d::new(&input, &mut reflected, 1, 1))
            .unwrap();
        assert_eq!(reflected[0].to_bits(), 2.0_f32.to_bits());
        assert_eq!(reflected[1].to_bits(), 1.0_f32.to_bits());
        assert_eq!(reflected[2].to_bits(), 2.0_f32.to_bits());
        assert_eq!(reflected[3].to_bits(), 3.0_f32.to_bits());
        assert_eq!(reflected[4].to_bits(), 2.0_f32.to_bits());

        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut tri = [0.0_f32; 4];
        kernel.process_event(OpTri::new(&input, &mut tri)).unwrap();
        assert_eq!(tri[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(tri[1].to_bits(), 0.0_f32.to_bits());
        assert_eq!(tri[2].to_bits(), 3.0_f32.to_bits());
        assert_eq!(tri[3].to_bits(), 4.0_f32.to_bits());
        assert!(kernel.is_ready());
    }
}
