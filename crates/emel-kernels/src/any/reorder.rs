//! Roll, nearest upscale, argsort, and top-k kernels over dense buffers.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_roll`, `op_upscale`, `op_argsort`, and `op_top_k` and names
//! `exec_op_*` routes on the arch machines. Those action/guard types are not
//! defined in the pinned headers, so this actor is source-contract.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Sort direction for [`OpArgsort`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArgsortOrder {
    /// Increasing values. C++ `GGML_SORT_ORDER_ASC`.
    Asc,
    /// Decreasing values. C++ `GGML_SORT_ORDER_DESC`.
    Desc,
}

/// Errors returned by reorder dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReorderError {
    /// Shift, scale, row, or buffer geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for ReorderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid reorder shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected reorder event"),
            Self::Internal => formatter.write_str("internal reorder dispatch error"),
        }
    }
}

impl std::error::Error for ReorderError {}

/// Result returned by reorder dispatch.
pub type ReorderResult = Result<(), ReorderError>;

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
const fn index_i32(index: usize) -> i32 {
    index as i32
}

const fn fits_i32(index: usize) -> bool {
    index <= i32::MAX as usize
}

/// Cyclic shift of a dense F32 vector.
#[derive(Debug)]
pub struct OpRoll<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    shift: i32,
}

impl<'a> OpRoll<'a> {
    /// Creates a roll request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], output: &'a mut [f32], shift: i32) -> Self {
        Self {
            input,
            output,
            shift,
        }
    }

    const fn valid(&self) -> bool {
        !self.input.is_empty()
            && self.input.len() == self.output.len()
            && fits_i32(self.input.len())
    }
}

/// Nearest-neighbor 1-D upscale of a dense F32 vector.
#[derive(Debug)]
pub struct OpUpscale<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    scale: usize,
}

impl<'a> OpUpscale<'a> {
    /// Creates an upscale request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], output: &'a mut [f32], scale: usize) -> Self {
        Self {
            input,
            output,
            scale,
        }
    }

    const fn valid(&self) -> bool {
        self.scale >= 1
            && !self.input.is_empty()
            && self.output.len() == self.input.len().saturating_mul(self.scale)
    }
}

/// Row-wise argsort of a dense F32 buffer into I32 indices.
#[derive(Debug)]
pub struct OpArgsort<'a> {
    input: &'a [f32],
    output: &'a mut [i32],
    row: usize,
    order: ArgsortOrder,
}

impl<'a> OpArgsort<'a> {
    /// Creates an argsort request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        input: &'a [f32],
        output: &'a mut [i32],
        row: usize,
        order: ArgsortOrder,
    ) -> Self {
        Self {
            input,
            output,
            row,
            order,
        }
    }

    const fn valid(&self) -> bool {
        self.row > 0
            && fits_i32(self.row)
            && !self.input.is_empty()
            && self.input.len().is_multiple_of(self.row)
            && self.output.len() == self.input.len()
    }
}

/// Row-wise top-k of a dense F32 buffer into I32 indices.
#[derive(Debug)]
pub struct OpTopK<'a> {
    input: &'a [f32],
    output: &'a mut [i32],
    row: usize,
    k: usize,
}

impl<'a> OpTopK<'a> {
    /// Creates a top-k request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(input: &'a [f32], output: &'a mut [i32], row: usize, k: usize) -> Self {
        Self {
            input,
            output,
            row,
            k,
        }
    }

    const fn valid(&self) -> bool {
        self.row > 0
            && self.k > 0
            && self.k <= self.row
            && fits_i32(self.row)
            && !self.input.is_empty()
            && self.input.len().is_multiple_of(self.row)
            && self.output.len() == (self.input.len() / self.row).saturating_mul(self.k)
    }
}

struct RollRuntime<'a> {
    event: OpRoll<'a>,
    result: &'a Cell<ReorderResult>,
}

struct UpscaleRuntime<'a> {
    event: OpUpscale<'a>,
    result: &'a Cell<ReorderResult>,
}

struct ArgsortRuntime<'a> {
    event: OpArgsort<'a>,
    result: &'a Cell<ReorderResult>,
}

struct TopKRuntime<'a> {
    event: OpTopK<'a>,
    result: &'a Cell<ReorderResult>,
}

#[derive(Default)]
struct Context;

sml! {
    ReorderMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Roll(RollRuntime<'dispatch>) [guard_roll_valid] / effect_roll,
        "ready"_s <= "ready"_s + Roll(RollRuntime<'dispatch>) [guard_roll_invalid] / effect_roll_reject,
        "ready"_s <= "ready"_s + Upscale(UpscaleRuntime<'dispatch>) [guard_upscale_valid] / effect_upscale,
        "ready"_s <= "ready"_s + Upscale(UpscaleRuntime<'dispatch>) [guard_upscale_invalid] / effect_upscale_reject,
        "ready"_s <= "ready"_s + Argsort(ArgsortRuntime<'dispatch>) [guard_argsort_asc] / effect_argsort_asc,
        "ready"_s <= "ready"_s + Argsort(ArgsortRuntime<'dispatch>) [guard_argsort_desc] / effect_argsort_desc,
        "ready"_s <= "ready"_s + Argsort(ArgsortRuntime<'dispatch>) [guard_argsort_invalid] / effect_argsort_reject,
        "ready"_s <= "ready"_s + TopK(TopKRuntime<'dispatch>) [guard_topk_valid] / effect_topk,
        "ready"_s <= "ready"_s + TopK(TopKRuntime<'dispatch>) [guard_topk_invalid] / effect_topk_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for reorder-family events.
pub trait ReorderEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut ReorderKernel) -> Self::Output;
}

/// Single-writer roll/upscale/argsort/top-k actor.
pub struct ReorderKernel {
    machine: ReorderMachineStateMachine<Context>,
}

impl fmt::Debug for ReorderKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReorderKernel")
            .finish_non_exhaustive()
    }
}

impl Default for ReorderKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ReorderKernel {
    /// Constructs an independent reorder actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: ReorderMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed reorder-family event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`ReorderError::InvalidShape`] when the geometry cannot form a
    /// dense roll, upscale, argsort, or top-k window.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: ReorderEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn roll(&mut self, event: OpRoll<'_>) -> ReorderResult {
        dispatch_roll(self, event)
    }

    fn upscale(&mut self, event: OpUpscale<'_>) -> ReorderResult {
        dispatch_upscale(self, event)
    }

    fn argsort(&mut self, event: OpArgsort<'_>) -> ReorderResult {
        dispatch_argsort(self, event)
    }

    fn topk(&mut self, event: OpTopK<'_>) -> ReorderResult {
        dispatch_topk(self, event)
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&ReorderMachineStates::Ready)
    }
}

fn ready_after(kernel: &ReorderKernel) {
    assert!(
        kernel.machine.is(&ReorderMachineStates::Ready),
        "reorder machine must return to ready after dispatch"
    );
}

fn dispatch_roll(kernel: &mut ReorderKernel, event: OpRoll<'_>) -> ReorderResult {
    let result = Cell::new(Err(ReorderError::UnexpectedEvent));
    kernel
        .machine
        .process_event(ReorderMachineEvents::Roll(RollRuntime {
            event,
            result: &result,
        }))
        .map_err(|_| ReorderError::Internal)?;
    ready_after(kernel);
    result.get()
}

fn dispatch_upscale(kernel: &mut ReorderKernel, event: OpUpscale<'_>) -> ReorderResult {
    let result = Cell::new(Err(ReorderError::UnexpectedEvent));
    kernel
        .machine
        .process_event(ReorderMachineEvents::Upscale(UpscaleRuntime {
            event,
            result: &result,
        }))
        .map_err(|_| ReorderError::Internal)?;
    ready_after(kernel);
    result.get()
}

fn dispatch_argsort(kernel: &mut ReorderKernel, event: OpArgsort<'_>) -> ReorderResult {
    let result = Cell::new(Err(ReorderError::UnexpectedEvent));
    kernel
        .machine
        .process_event(ReorderMachineEvents::Argsort(ArgsortRuntime {
            event,
            result: &result,
        }))
        .map_err(|_| ReorderError::Internal)?;
    ready_after(kernel);
    result.get()
}

fn dispatch_topk(kernel: &mut ReorderKernel, event: OpTopK<'_>) -> ReorderResult {
    let result = Cell::new(Err(ReorderError::UnexpectedEvent));
    kernel
        .machine
        .process_event(ReorderMachineEvents::TopK(TopKRuntime {
            event,
            result: &result,
        }))
        .map_err(|_| ReorderError::Internal)?;
    ready_after(kernel);
    result.get()
}

impl ReorderEvent for OpRoll<'_> {
    type Output = ReorderResult;
    fn dispatch(self, kernel: &mut ReorderKernel) -> Self::Output {
        kernel.roll(self)
    }
}

impl ReorderEvent for OpUpscale<'_> {
    type Output = ReorderResult;
    fn dispatch(self, kernel: &mut ReorderKernel) -> Self::Output {
        kernel.upscale(self)
    }
}

impl ReorderEvent for OpArgsort<'_> {
    type Output = ReorderResult;
    fn dispatch(self, kernel: &mut ReorderKernel) -> Self::Output {
        kernel.argsort(self)
    }
}

impl ReorderEvent for OpTopK<'_> {
    type Output = ReorderResult;
    fn dispatch(self, kernel: &mut ReorderKernel) -> Self::Output {
        kernel.topk(self)
    }
}

const fn wrap_index(index: isize, len: usize) -> usize {
    let len_i = len.cast_signed();
    index.rem_euclid(len_i).cast_unsigned()
}

fn roll_values(event: &mut OpRoll<'_>) {
    let len = event.input.len();
    let shift = isize::try_from(event.shift).unwrap_or(0);
    for (out_index, slot) in event.output.iter_mut().enumerate() {
        let source = wrap_index(out_index.cast_signed() - shift, len);
        *slot = event.input[source];
    }
}

fn upscale_values(event: &mut OpUpscale<'_>) {
    for (out_index, slot) in event.output.iter_mut().enumerate() {
        *slot = event.input[out_index / event.scale];
    }
}

fn argsort_row_asc(input: &[f32], output: &mut [i32]) {
    fill_identity(output);
    insertion_sort(input, output, false_less);
}

fn argsort_row_desc(input: &[f32], output: &mut [i32]) {
    fill_identity(output);
    insertion_sort(input, output, true_less);
}

fn fill_identity(output: &mut [i32]) {
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = index_i32(index);
    }
}

const fn false_less(left: f32, right: f32) -> bool {
    left > right
}

const fn true_less(left: f32, right: f32) -> bool {
    left < right
}

fn insertion_sort(input: &[f32], output: &mut [i32], out_of_order: fn(f32, f32) -> bool) {
    for start in 1..output.len() {
        let mut current = start;
        while current > 0 {
            let left = output[current - 1];
            let right = output[current];
            let left_u = left.cast_unsigned() as usize;
            let right_u = right.cast_unsigned() as usize;
            if !out_of_order(input[left_u], input[right_u]) {
                break;
            }
            output[current - 1] = right;
            output[current] = left;
            current -= 1;
        }
    }
}

fn argsort_values_asc(event: &mut OpArgsort<'_>) {
    let row = event.row;
    for (row_index, out_row) in event.output.chunks_mut(row).enumerate() {
        let start = row_index * row;
        argsort_row_asc(&event.input[start..start + row], out_row);
    }
}

fn argsort_values_desc(event: &mut OpArgsort<'_>) {
    let row = event.row;
    for (row_index, out_row) in event.output.chunks_mut(row).enumerate() {
        let start = row_index * row;
        argsort_row_desc(&event.input[start..start + row], out_row);
    }
}

fn already_taken(taken: &[i32], candidate: i32) -> bool {
    taken.contains(&candidate)
}

fn topk_row(input: &[f32], output: &mut [i32]) {
    for slot_index in 0..output.len() {
        let mut best = 0_usize;
        let mut best_value = input[0];
        let mut found = !already_taken(&output[..slot_index], index_i32(0));
        for (index, sample) in input.iter().copied().enumerate() {
            let candidate = index_i32(index);
            if already_taken(&output[..slot_index], candidate) {
                continue;
            }
            if !found || sample > best_value {
                best = index;
                best_value = sample;
                found = true;
            }
        }
        output[slot_index] = index_i32(best);
    }
}

fn topk_values(event: &mut OpTopK<'_>) {
    let row = event.row;
    let k = event.k;
    for (row_index, out_row) in event.output.chunks_mut(k).enumerate() {
        let start = row_index * row;
        topk_row(&event.input[start..start + row], out_row);
    }
}

impl ReorderMachineStateMachineContext for Context {
    fn guard_roll_valid(&self, event: &RollRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_roll_invalid(&self, event: &RollRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_upscale_valid(&self, event: &UpscaleRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_upscale_invalid(&self, event: &UpscaleRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_argsort_asc(&self, event: &ArgsortRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid() && event.event.order == ArgsortOrder::Asc)
    }

    fn guard_argsort_desc(&self, event: &ArgsortRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid() && event.event.order == ArgsortOrder::Desc)
    }

    fn guard_argsort_invalid(&self, event: &ArgsortRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_topk_valid(&self, event: &TopKRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_topk_invalid(&self, event: &TopKRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn effect_roll(&mut self, mut event: RollRuntime<'_>) -> Result<(), ()> {
        roll_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_roll_reject(&mut self, event: RollRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ReorderError::InvalidShape));
        Ok(())
    }

    fn effect_upscale(&mut self, mut event: UpscaleRuntime<'_>) -> Result<(), ()> {
        upscale_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_upscale_reject(&mut self, event: UpscaleRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ReorderError::InvalidShape));
        Ok(())
    }

    fn effect_argsort_asc(&mut self, mut event: ArgsortRuntime<'_>) -> Result<(), ()> {
        argsort_values_asc(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_argsort_desc(&mut self, mut event: ArgsortRuntime<'_>) -> Result<(), ()> {
        argsort_values_desc(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_argsort_reject(&mut self, event: ArgsortRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ReorderError::InvalidShape));
        Ok(())
    }

    fn effect_topk(&mut self, mut event: TopKRuntime<'_>) -> Result<(), ()> {
        topk_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_topk_reject(&mut self, event: TopKRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ReorderError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ArgsortOrder, OpArgsort, OpRoll, OpTopK, OpUpscale, ReorderError, ReorderKernel};
    use crate::Kernel;

    #[test]
    fn roll_wraps_left_and_right() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        let mut kernel = ReorderKernel::new();
        kernel
            .process_event(OpRoll::new(&input, &mut output, 1))
            .unwrap();
        assert_eq!(output[0].to_bits(), 4.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 1.0_f32.to_bits());
        assert_eq!(output[2].to_bits(), 2.0_f32.to_bits());
        assert_eq!(output[3].to_bits(), 3.0_f32.to_bits());
        kernel
            .process_event(OpRoll::new(&input, &mut output, -1))
            .unwrap();
        assert_eq!(output[0].to_bits(), 2.0_f32.to_bits());
        assert_eq!(output[3].to_bits(), 1.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn upscale_repeats_nearest_samples() {
        let input = [1.0_f32, 2.0];
        let mut output = [0.0_f32; 4];
        let mut kernel = ReorderKernel::new();
        kernel
            .process_event(OpUpscale::new(&input, &mut output, 2))
            .unwrap();
        assert_eq!(output[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 1.0_f32.to_bits());
        assert_eq!(output[2].to_bits(), 2.0_f32.to_bits());
        assert_eq!(output[3].to_bits(), 2.0_f32.to_bits());
    }

    #[test]
    fn argsort_and_top_k_write_indices() {
        let input = [3.0_f32, 1.0, 2.0];
        let mut order = [0_i32; 3];
        let mut kernel = ReorderKernel::new();
        kernel
            .process_event(OpArgsort::new(&input, &mut order, 3, ArgsortOrder::Asc))
            .unwrap();
        assert_eq!(order, [1, 2, 0]);
        kernel
            .process_event(OpArgsort::new(&input, &mut order, 3, ArgsortOrder::Desc))
            .unwrap();
        assert_eq!(order, [0, 2, 1]);
        let mut top = [0_i32; 2];
        kernel
            .process_event(OpTopK::new(&input, &mut top, 3, 2))
            .unwrap();
        assert_eq!(top, [0, 2]);
    }

    #[test]
    fn public_kernel_dispatches_roll() {
        let input = [1.0_f32, 2.0, 3.0];
        let mut output = [0.0_f32; 3];
        let mut kernel = Kernel::new();
        kernel
            .process_event(OpRoll::new(&input, &mut output, 1))
            .unwrap();
        assert_eq!(output[0].to_bits(), 3.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 1.0_f32.to_bits());
        assert_eq!(output[2].to_bits(), 2.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn roll_rejects_length_mismatch_without_mutation() {
        let input = [1.0_f32];
        let mut output = [5.0_f32; 2];
        let mut kernel = ReorderKernel::new();
        assert_eq!(
            kernel.process_event(OpRoll::new(&input, &mut output, 0)),
            Err(ReorderError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 5.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn public_kernel_dispatches_upscale_argsort_and_top_k() {
        let input = [3.0_f32, 1.0, 2.0];
        let mut up = [0.0_f32; 6];
        let mut kernel = Kernel::new();
        kernel
            .process_event(OpUpscale::new(&input[..2], &mut up[..4], 2))
            .unwrap();
        assert_eq!(up[2].to_bits(), 1.0_f32.to_bits());
        let mut order = [0_i32; 3];
        kernel
            .process_event(OpArgsort::new(&input, &mut order, 3, ArgsortOrder::Asc))
            .unwrap();
        assert_eq!(order, [1, 2, 0]);
        let mut top = [0_i32; 2];
        kernel
            .process_event(OpTopK::new(&input, &mut top, 3, 2))
            .unwrap();
        assert_eq!(top, [0, 2]);
    }
}
