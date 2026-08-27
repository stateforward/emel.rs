//! Dense backward and scatter kernels for remaining C++ routes.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_soft_max_back`, `op_rms_norm_back`, `op_get_rows_back`, `op_set`,
//! and `op_set_rows` and names `exec_op_*` routes on the arch machines.
//! Those action/guard types are not defined in the pinned headers, so this
//! actor is source-contract.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by backward/scatter dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackwardError {
    /// Row, index, or buffer geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for BackwardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid backward shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected backward event"),
            Self::Internal => formatter.write_str("internal backward dispatch error"),
        }
    }
}

impl std::error::Error for BackwardError {}

/// Result returned by backward/scatter dispatch.
pub type BackwardResult = Result<(), BackwardError>;

#[allow(clippy::cast_precision_loss)]
const fn count_f32(count: usize) -> f32 {
    count as f32
}

const fn fits_i32(value: usize) -> bool {
    value <= i32::MAX.cast_unsigned() as usize
}

const fn row_index(indices: &[i32], row_count: usize) -> Option<usize> {
    let mut offset = 0_usize;
    while offset < indices.len() {
        let index = indices[offset];
        if index < 0 {
            return None;
        }
        let index_u = index.cast_unsigned() as usize;
        if index_u >= row_count {
            return None;
        }
        offset += 1;
    }
    Some(offset)
}

/// Softmax Jacobian applied to a dense F32 gradient.
#[derive(Debug)]
pub struct OpSoftMaxBack<'a> {
    values: &'a [f32],
    grad_out: &'a [f32],
    grad_in: &'a mut [f32],
    row: usize,
}

impl<'a> OpSoftMaxBack<'a> {
    /// Creates a softmax-backward request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        values: &'a [f32],
        grad_out: &'a [f32],
        grad_in: &'a mut [f32],
        row: usize,
    ) -> Self {
        Self {
            values,
            grad_out,
            grad_in,
            row,
        }
    }

    const fn valid(&self) -> bool {
        self.row > 0
            && self.values.len() == self.grad_out.len()
            && self.values.len() == self.grad_in.len()
            && self.values.len().is_multiple_of(self.row)
    }
}

/// RMS-normalization Jacobian applied to a dense F32 gradient.
#[derive(Debug)]
pub struct OpRmsNormBack<'a> {
    input: &'a [f32],
    grad_out: &'a [f32],
    grad_in: &'a mut [f32],
    row: usize,
    epsilon: f32,
}

impl<'a> OpRmsNormBack<'a> {
    /// Creates an RMS-norm backward request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        input: &'a [f32],
        grad_out: &'a [f32],
        grad_in: &'a mut [f32],
        row: usize,
        epsilon: f32,
    ) -> Self {
        Self {
            input,
            grad_out,
            grad_in,
            row,
            epsilon,
        }
    }

    const fn valid(&self) -> bool {
        self.row > 0
            && self.epsilon >= 0.0
            && self.input.len() == self.grad_out.len()
            && self.input.len() == self.grad_in.len()
            && self.input.len().is_multiple_of(self.row)
    }
}

/// Scatter-adds gathered row gradients back into the source table.
#[derive(Debug)]
pub struct OpGetRowsBack<'a> {
    grad_out: &'a [f32],
    indices: &'a [i32],
    grad_in: &'a mut [f32],
    row: usize,
}

impl<'a> OpGetRowsBack<'a> {
    /// Creates a get-rows backward request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        grad_out: &'a [f32],
        indices: &'a [i32],
        grad_in: &'a mut [f32],
        row: usize,
    ) -> Self {
        Self {
            grad_out,
            indices,
            grad_in,
            row,
        }
    }

    const fn valid(&self) -> bool {
        self.row > 0
            && fits_i32(self.row)
            && !self.indices.is_empty()
            && self.grad_out.len() == self.indices.len().saturating_mul(self.row)
            && self.grad_in.len().is_multiple_of(self.row)
            && row_index(self.indices, self.grad_in.len() / self.row).is_some()
    }
}

/// Overwrites destination rows selected by I32 indices.
#[derive(Debug)]
pub struct OpSetRows<'a> {
    source: &'a [f32],
    indices: &'a [i32],
    destination: &'a mut [f32],
    row: usize,
}

impl<'a> OpSetRows<'a> {
    /// Creates a set-rows request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        source: &'a [f32],
        indices: &'a [i32],
        destination: &'a mut [f32],
        row: usize,
    ) -> Self {
        Self {
            source,
            indices,
            destination,
            row,
        }
    }

    const fn valid(&self) -> bool {
        self.row > 0
            && fits_i32(self.row)
            && !self.indices.is_empty()
            && self.source.len() == self.indices.len().saturating_mul(self.row)
            && self.destination.len().is_multiple_of(self.row)
            && row_index(self.indices, self.destination.len() / self.row).is_some()
    }
}

/// Writes a dense F32 slice into a destination at an explicit offset.
#[derive(Debug)]
pub struct OpSet<'a> {
    source: &'a [f32],
    destination: &'a mut [f32],
    offset: usize,
}

impl<'a> OpSet<'a> {
    /// Creates a set request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(source: &'a [f32], destination: &'a mut [f32], offset: usize) -> Self {
        Self {
            source,
            destination,
            offset,
        }
    }

    const fn valid(&self) -> bool {
        !self.source.is_empty()
            && self.offset.saturating_add(self.source.len()) <= self.destination.len()
    }
}

struct SoftMaxBackRuntime<'a> {
    event: OpSoftMaxBack<'a>,
    result: &'a Cell<BackwardResult>,
}

struct RmsNormBackRuntime<'a> {
    event: OpRmsNormBack<'a>,
    result: &'a Cell<BackwardResult>,
}

struct GetRowsBackRuntime<'a> {
    event: OpGetRowsBack<'a>,
    result: &'a Cell<BackwardResult>,
}

struct SetRowsRuntime<'a> {
    event: OpSetRows<'a>,
    result: &'a Cell<BackwardResult>,
}

struct SetRuntime<'a> {
    event: OpSet<'a>,
    result: &'a Cell<BackwardResult>,
}

#[derive(Default)]
struct Context;

sml! {
    BackwardMachine<'dispatch> {
        "ready"_s <= *"ready"_s + SoftMaxBack(SoftMaxBackRuntime<'dispatch>) [guard_softmax_valid] / effect_softmax,
        "ready"_s <= "ready"_s + SoftMaxBack(SoftMaxBackRuntime<'dispatch>) [guard_softmax_invalid] / effect_softmax_reject,
        "ready"_s <= "ready"_s + RmsNormBack(RmsNormBackRuntime<'dispatch>) [guard_rms_valid] / effect_rms,
        "ready"_s <= "ready"_s + RmsNormBack(RmsNormBackRuntime<'dispatch>) [guard_rms_invalid] / effect_rms_reject,
        "ready"_s <= "ready"_s + GetRowsBack(GetRowsBackRuntime<'dispatch>) [guard_get_rows_valid] / effect_get_rows,
        "ready"_s <= "ready"_s + GetRowsBack(GetRowsBackRuntime<'dispatch>) [guard_get_rows_invalid] / effect_get_rows_reject,
        "ready"_s <= "ready"_s + SetRows(SetRowsRuntime<'dispatch>) [guard_set_rows_valid] / effect_set_rows,
        "ready"_s <= "ready"_s + SetRows(SetRowsRuntime<'dispatch>) [guard_set_rows_invalid] / effect_set_rows_reject,
        "ready"_s <= "ready"_s + Set(SetRuntime<'dispatch>) [guard_set_valid] / effect_set,
        "ready"_s <= "ready"_s + Set(SetRuntime<'dispatch>) [guard_set_invalid] / effect_set_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for backward/scatter events.
pub trait BackwardEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut BackwardKernel) -> Self::Output;
}

/// Single-writer backward/scatter actor.
pub struct BackwardKernel {
    machine: BackwardMachineStateMachine<Context>,
}

impl fmt::Debug for BackwardKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BackwardKernel")
            .finish_non_exhaustive()
    }
}

impl Default for BackwardKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardKernel {
    /// Constructs an independent backward actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: BackwardMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed backward/scatter event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`BackwardError::InvalidShape`] when the dense buffers cannot
    /// form the requested Jacobian or scatter geometry.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: BackwardEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn softmax(&mut self, event: OpSoftMaxBack<'_>) -> BackwardResult {
        let result = Cell::new(Err(BackwardError::UnexpectedEvent));
        self.machine
            .process_event(BackwardMachineEvents::SoftMaxBack(SoftMaxBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BackwardError::Internal)?;
        assert!(
            self.machine.is(&BackwardMachineStates::Ready),
            "backward machine must return to ready after dispatch"
        );
        result.get()
    }

    fn rms(&mut self, event: OpRmsNormBack<'_>) -> BackwardResult {
        let result = Cell::new(Err(BackwardError::UnexpectedEvent));
        self.machine
            .process_event(BackwardMachineEvents::RmsNormBack(RmsNormBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BackwardError::Internal)?;
        assert!(
            self.machine.is(&BackwardMachineStates::Ready),
            "backward machine must return to ready after dispatch"
        );
        result.get()
    }

    fn get_rows(&mut self, event: OpGetRowsBack<'_>) -> BackwardResult {
        let result = Cell::new(Err(BackwardError::UnexpectedEvent));
        self.machine
            .process_event(BackwardMachineEvents::GetRowsBack(GetRowsBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BackwardError::Internal)?;
        assert!(
            self.machine.is(&BackwardMachineStates::Ready),
            "backward machine must return to ready after dispatch"
        );
        result.get()
    }

    fn set_rows(&mut self, event: OpSetRows<'_>) -> BackwardResult {
        let result = Cell::new(Err(BackwardError::UnexpectedEvent));
        self.machine
            .process_event(BackwardMachineEvents::SetRows(SetRowsRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BackwardError::Internal)?;
        assert!(
            self.machine.is(&BackwardMachineStates::Ready),
            "backward machine must return to ready after dispatch"
        );
        result.get()
    }

    fn set(&mut self, event: OpSet<'_>) -> BackwardResult {
        let result = Cell::new(Err(BackwardError::UnexpectedEvent));
        self.machine
            .process_event(BackwardMachineEvents::Set(SetRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| BackwardError::Internal)?;
        assert!(
            self.machine.is(&BackwardMachineStates::Ready),
            "backward machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&BackwardMachineStates::Ready)
    }
}

impl BackwardEvent for OpSoftMaxBack<'_> {
    type Output = BackwardResult;
    fn dispatch(self, kernel: &mut BackwardKernel) -> Self::Output {
        kernel.softmax(self)
    }
}

impl BackwardEvent for OpRmsNormBack<'_> {
    type Output = BackwardResult;
    fn dispatch(self, kernel: &mut BackwardKernel) -> Self::Output {
        kernel.rms(self)
    }
}

impl BackwardEvent for OpGetRowsBack<'_> {
    type Output = BackwardResult;
    fn dispatch(self, kernel: &mut BackwardKernel) -> Self::Output {
        kernel.get_rows(self)
    }
}

impl BackwardEvent for OpSetRows<'_> {
    type Output = BackwardResult;
    fn dispatch(self, kernel: &mut BackwardKernel) -> Self::Output {
        kernel.set_rows(self)
    }
}

impl BackwardEvent for OpSet<'_> {
    type Output = BackwardResult;
    fn dispatch(self, kernel: &mut BackwardKernel) -> Self::Output {
        kernel.set(self)
    }
}

fn softmax_back_values(event: &mut OpSoftMaxBack<'_>) {
    let row = event.row;
    for (values, (grad_out, grad_in)) in event.values.chunks(row).zip(
        event
            .grad_out
            .chunks(row)
            .zip(event.grad_in.chunks_mut(row)),
    ) {
        let mut dot = 0.0_f32;
        for (value, grad) in values.iter().zip(grad_out.iter()) {
            dot = value.mul_add(*grad, dot);
        }
        for ((value, grad), slot) in values.iter().zip(grad_out.iter()).zip(grad_in.iter_mut()) {
            *slot = *value * (*grad - dot);
        }
    }
}

fn rms_norm_back_values(event: &mut OpRmsNormBack<'_>) {
    let row = event.row;
    let count = count_f32(row);
    for (input, (grad_out, grad_in)) in event.input.chunks(row).zip(
        event
            .grad_out
            .chunks(row)
            .zip(event.grad_in.chunks_mut(row)),
    ) {
        let mut sum_sq = 0.0_f32;
        let mut sum_xdy = 0.0_f32;
        for (sample, grad) in input.iter().zip(grad_out.iter()) {
            sum_sq = sample.mul_add(*sample, sum_sq);
            sum_xdy = sample.mul_add(*grad, sum_xdy);
        }
        let mean_ss = sum_sq / count + event.epsilon;
        let scale = mean_ss.sqrt().recip();
        let coeff = sum_xdy * scale / (count * mean_ss);
        for ((sample, grad), slot) in input.iter().zip(grad_out.iter()).zip(grad_in.iter_mut()) {
            *slot = grad.mul_add(scale, -sample * coeff);
        }
    }
}

fn get_rows_back_values(event: &mut OpGetRowsBack<'_>) {
    event.grad_in.fill(0.0_f32);
    let row = event.row;
    for (row_index, grad_out) in event.grad_out.chunks(row).enumerate() {
        let source = event.indices[row_index].cast_unsigned() as usize;
        let dest = &mut event.grad_in[source * row..source * row + row];
        for (slot, grad) in dest.iter_mut().zip(grad_out.iter()) {
            *slot += *grad;
        }
    }
}

fn set_rows_values(event: &mut OpSetRows<'_>) {
    let row = event.row;
    for (row_index, source) in event.source.chunks(row).enumerate() {
        let dest_row = event.indices[row_index].cast_unsigned() as usize;
        event.destination[dest_row * row..dest_row * row + row].copy_from_slice(source);
    }
}

fn set_values(event: &mut OpSet<'_>) {
    let end = event.offset + event.source.len();
    event.destination[event.offset..end].copy_from_slice(event.source);
}

impl BackwardMachineStateMachineContext for Context {
    fn guard_softmax_valid(&self, event: &SoftMaxBackRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_softmax_invalid(&self, event: &SoftMaxBackRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_rms_valid(&self, event: &RmsNormBackRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_rms_invalid(&self, event: &RmsNormBackRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_get_rows_valid(&self, event: &GetRowsBackRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_get_rows_invalid(&self, event: &GetRowsBackRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_set_rows_valid(&self, event: &SetRowsRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_set_rows_invalid(&self, event: &SetRowsRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_set_valid(&self, event: &SetRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_set_invalid(&self, event: &SetRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn effect_softmax(&mut self, mut event: SoftMaxBackRuntime<'_>) -> Result<(), ()> {
        softmax_back_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_softmax_reject(&mut self, event: SoftMaxBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(BackwardError::InvalidShape));
        Ok(())
    }

    fn effect_rms(&mut self, mut event: RmsNormBackRuntime<'_>) -> Result<(), ()> {
        rms_norm_back_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_rms_reject(&mut self, event: RmsNormBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(BackwardError::InvalidShape));
        Ok(())
    }

    fn effect_get_rows(&mut self, mut event: GetRowsBackRuntime<'_>) -> Result<(), ()> {
        get_rows_back_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_get_rows_reject(&mut self, event: GetRowsBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(BackwardError::InvalidShape));
        Ok(())
    }

    fn effect_set_rows(&mut self, mut event: SetRowsRuntime<'_>) -> Result<(), ()> {
        set_rows_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_set_rows_reject(&mut self, event: SetRowsRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(BackwardError::InvalidShape));
        Ok(())
    }

    fn effect_set(&mut self, mut event: SetRuntime<'_>) -> Result<(), ()> {
        set_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_set_reject(&mut self, event: SetRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(BackwardError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BackwardError, BackwardKernel, OpGetRowsBack, OpRmsNormBack, OpSet, OpSetRows,
        OpSoftMaxBack,
    };
    use crate::Kernel;

    #[test]
    fn softmax_back_uses_row_jacobian() {
        let values = [0.25_f32, 0.75];
        let grad_out = [1.0_f32, 0.0];
        let mut grad_in = [0.0_f32; 2];
        let mut kernel = BackwardKernel::new();
        kernel
            .process_event(OpSoftMaxBack::new(&values, &grad_out, &mut grad_in, 2))
            .unwrap();
        assert_eq!(grad_in[0].to_bits(), 0.1875_f32.to_bits());
        assert_eq!(grad_in[1].to_bits(), (-0.1875_f32).to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn get_rows_back_scatter_adds_and_set_rows_overwrites() {
        let grad_out = [1.0_f32, 2.0];
        let indices = [1_i32];
        let mut grad_in = [9.0_f32; 4];
        let mut kernel = BackwardKernel::new();
        kernel
            .process_event(OpGetRowsBack::new(&grad_out, &indices, &mut grad_in, 2))
            .unwrap();
        assert_eq!(grad_in[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(grad_in[2].to_bits(), 1.0_f32.to_bits());
        assert_eq!(grad_in[3].to_bits(), 2.0_f32.to_bits());
        let source = [3.0_f32, 4.0];
        kernel
            .process_event(OpSetRows::new(&source, &indices, &mut grad_in, 2))
            .unwrap();
        assert_eq!(grad_in[2].to_bits(), 3.0_f32.to_bits());
        assert_eq!(grad_in[3].to_bits(), 4.0_f32.to_bits());
    }

    #[test]
    fn set_writes_at_offset() {
        let source = [8.0_f32, 9.0];
        let mut destination = [1.0_f32, 2.0, 3.0];
        let mut kernel = BackwardKernel::new();
        kernel
            .process_event(OpSet::new(&source, &mut destination, 1))
            .unwrap();
        assert_eq!(destination[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(destination[1].to_bits(), 8.0_f32.to_bits());
        assert_eq!(destination[2].to_bits(), 9.0_f32.to_bits());
    }

    #[test]
    fn rms_norm_back_and_public_kernel_agree() {
        let input = [3.0_f32, 4.0];
        let grad_out = [1.0_f32, 0.0];
        let mut via_child = [0.0_f32; 2];
        let mut via_public = [0.0_f32; 2];
        BackwardKernel::new()
            .process_event(OpRmsNormBack::new(
                &input,
                &grad_out,
                &mut via_child,
                2,
                0.0,
            ))
            .unwrap();
        Kernel::new()
            .process_event(OpRmsNormBack::new(
                &input,
                &grad_out,
                &mut via_public,
                2,
                0.0,
            ))
            .unwrap();
        assert_eq!(via_child[0].to_bits(), via_public[0].to_bits());
        assert_eq!(via_child[1].to_bits(), via_public[1].to_bits());
    }

    #[test]
    fn softmax_back_rejects_without_mutation() {
        let values = [1.0_f32];
        let grad_out = [1.0_f32, 2.0];
        let mut grad_in = [5.0_f32; 2];
        let mut kernel = BackwardKernel::new();
        assert_eq!(
            kernel.process_event(OpSoftMaxBack::new(&values, &grad_out, &mut grad_in, 2)),
            Err(BackwardError::InvalidShape)
        );
        assert_eq!(grad_in[0].to_bits(), 5.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_softmax_set_and_rows() {
        let values = [0.25_f32, 0.75];
        let grad_out = [1.0_f32, 0.0];
        let mut grad_in = [0.0_f32; 2];
        let mut kernel = Kernel::new();
        kernel
            .process_event(OpSoftMaxBack::new(&values, &grad_out, &mut grad_in, 2))
            .unwrap();
        assert_eq!(grad_in[0].to_bits(), 0.1875_f32.to_bits());
        let source = [8.0_f32, 9.0];
        let mut destination = [1.0_f32, 2.0, 3.0];
        kernel
            .process_event(OpSet::new(&source, &mut destination, 1))
            .unwrap();
        assert_eq!(destination[2].to_bits(), 9.0_f32.to_bits());
        let grad_out = [1.0_f32, 2.0];
        let indices = [1_i32];
        let mut rows = [0.0_f32; 4];
        kernel
            .process_event(OpGetRowsBack::new(&grad_out, &indices, &mut rows, 2))
            .unwrap();
        kernel
            .process_event(OpSetRows::new(&source, &indices, &mut rows, 2))
            .unwrap();
        assert_eq!(rows[2].to_bits(), 8.0_f32.to_bits());
    }
}
