//! Dense indexed add and expert matmul kernels.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_add_id` and `op_mul_mat_id` and names `exec_op_*` routes on the arch
//! machines. Those action/guard types are not defined in the pinned headers,
//! so this actor is source-contract.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by indexed dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexedError {
    /// Row, expert, or index geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for IndexedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid indexed shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected indexed event"),
            Self::Internal => formatter.write_str("internal indexed dispatch error"),
        }
    }
}

impl std::error::Error for IndexedError {}

/// Result returned by indexed dispatch.
pub type IndexedResult = Result<(), IndexedError>;

const fn ids_in_range(ids: &[i32], table_rows: usize) -> bool {
    let mut offset = 0;
    while offset < ids.len() {
        let id = ids[offset];
        if id < 0 || (id.cast_unsigned() as usize) >= table_rows {
            return false;
        }
        offset += 1;
    }
    true
}

/// Adds table rows selected by I32 ids into a dense token buffer.
#[derive(Debug)]
pub struct OpAddId<'a> {
    tokens: &'a [f32],
    table: &'a [f32],
    ids: &'a [i32],
    output: &'a mut [f32],
    dim: usize,
}

impl<'a> OpAddId<'a> {
    /// Creates an add-id request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        tokens: &'a [f32],
        table: &'a [f32],
        ids: &'a [i32],
        output: &'a mut [f32],
        dim: usize,
    ) -> Self {
        Self {
            tokens,
            table,
            ids,
            output,
            dim,
        }
    }

    const fn valid(&self) -> bool {
        self.dim > 0
            && !self.ids.is_empty()
            && self.tokens.len() == self.ids.len().saturating_mul(self.dim)
            && self.output.len() == self.tokens.len()
            && self.table.len().is_multiple_of(self.dim)
            && !self.table.is_empty()
            && ids_in_range(self.ids, self.table.len() / self.dim)
    }
}

/// Geometry for [`OpMulMatId`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MulMatIdParams {
    /// Expert count.
    pub experts: usize,
    /// Input channels.
    pub n_in: usize,
    /// Output channels.
    pub n_out: usize,
}

/// Expert-selected dense GEMV: `out[t] = experts[ids[t]] * tokens[t]`.
#[derive(Debug)]
pub struct OpMulMatId<'a> {
    experts: &'a [f32],
    tokens: &'a [f32],
    ids: &'a [i32],
    output: &'a mut [f32],
    params: MulMatIdParams,
}

impl<'a> OpMulMatId<'a> {
    /// Creates a mul-mat-id request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        experts: &'a [f32],
        tokens: &'a [f32],
        ids: &'a [i32],
        output: &'a mut [f32],
        params: MulMatIdParams,
    ) -> Self {
        Self {
            experts,
            tokens,
            ids,
            output,
            params,
        }
    }

    const fn valid(&self) -> bool {
        self.params.experts > 0
            && self.params.n_in > 0
            && self.params.n_out > 0
            && !self.ids.is_empty()
            && self.tokens.len() == self.ids.len().saturating_mul(self.params.n_in)
            && self.output.len() == self.ids.len().saturating_mul(self.params.n_out)
            && self.experts.len()
                == self
                    .params
                    .experts
                    .saturating_mul(self.params.n_out)
                    .saturating_mul(self.params.n_in)
            && ids_in_range(self.ids, self.params.experts)
    }
}

struct AddIdRuntime<'a> {
    event: OpAddId<'a>,
    result: &'a Cell<IndexedResult>,
}

struct MulMatIdRuntime<'a> {
    event: OpMulMatId<'a>,
    result: &'a Cell<IndexedResult>,
}

#[derive(Default)]
struct Context;

sml! {
    IndexedMachine<'dispatch> {
        "ready"_s <= *"ready"_s + AddId(AddIdRuntime<'dispatch>) [guard_add_valid] / effect_add,
        "ready"_s <= "ready"_s + AddId(AddIdRuntime<'dispatch>) [guard_add_invalid] / effect_add_reject,
        "ready"_s <= "ready"_s + MulMatId(MulMatIdRuntime<'dispatch>) [guard_mul_valid] / effect_mul,
        "ready"_s <= "ready"_s + MulMatId(MulMatIdRuntime<'dispatch>) [guard_mul_invalid] / effect_mul_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for indexed events.
pub trait IndexedEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut IndexedKernel) -> Self::Output;
}

/// Single-writer indexed actor.
pub struct IndexedKernel {
    machine: IndexedMachineStateMachine<Context>,
}

impl fmt::Debug for IndexedKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IndexedKernel")
            .finish_non_exhaustive()
    }
}

impl Default for IndexedKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexedKernel {
    /// Constructs an independent indexed actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: IndexedMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed indexed event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`IndexedError::InvalidShape`] when the dense buffers cannot
    /// form the requested indexed add or expert GEMV.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: IndexedEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn add_id(&mut self, event: OpAddId<'_>) -> IndexedResult {
        let result = Cell::new(Err(IndexedError::UnexpectedEvent));
        self.machine
            .process_event(IndexedMachineEvents::AddId(AddIdRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| IndexedError::Internal)?;
        assert!(
            self.machine.is(&IndexedMachineStates::Ready),
            "indexed machine must return to ready after dispatch"
        );
        result.get()
    }

    fn mul_mat_id(&mut self, event: OpMulMatId<'_>) -> IndexedResult {
        let result = Cell::new(Err(IndexedError::UnexpectedEvent));
        self.machine
            .process_event(IndexedMachineEvents::MulMatId(MulMatIdRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| IndexedError::Internal)?;
        assert!(
            self.machine.is(&IndexedMachineStates::Ready),
            "indexed machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&IndexedMachineStates::Ready)
    }
}

impl IndexedEvent for OpAddId<'_> {
    type Output = IndexedResult;
    fn dispatch(self, kernel: &mut IndexedKernel) -> Self::Output {
        kernel.add_id(self)
    }
}

impl IndexedEvent for OpMulMatId<'_> {
    type Output = IndexedResult;
    fn dispatch(self, kernel: &mut IndexedKernel) -> Self::Output {
        kernel.mul_mat_id(self)
    }
}

fn add_id_values(event: &mut OpAddId<'_>) {
    let dim = event.dim;
    for (row, id) in event.ids.iter().copied().enumerate() {
        let table_row = (id.cast_unsigned() as usize) * dim;
        let token_row = row * dim;
        for col in 0..dim {
            event.output[token_row + col] =
                event.tokens[token_row + col] + event.table[table_row + col];
        }
    }
}

fn mul_mat_id_values(event: &mut OpMulMatId<'_>) {
    let n_in = event.params.n_in;
    let n_out = event.params.n_out;
    let expert_stride = n_out * n_in;
    for (token, id) in event.ids.iter().copied().enumerate() {
        let expert = (id.cast_unsigned() as usize) * expert_stride;
        let src = token * n_in;
        let dst = token * n_out;
        for out_ch in 0..n_out {
            let mut acc = 0.0_f32;
            let weight = expert + out_ch * n_in;
            for in_ch in 0..n_in {
                acc = event.experts[weight + in_ch].mul_add(event.tokens[src + in_ch], acc);
            }
            event.output[dst + out_ch] = acc;
        }
    }
}

impl IndexedMachineStateMachineContext for Context {
    fn guard_add_valid(&self, event: &AddIdRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_add_invalid(&self, event: &AddIdRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn guard_mul_valid(&self, event: &MulMatIdRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_mul_invalid(&self, event: &MulMatIdRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn effect_add(&mut self, mut event: AddIdRuntime<'_>) -> Result<(), ()> {
        add_id_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_add_reject(&mut self, event: AddIdRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(IndexedError::InvalidShape));
        Ok(())
    }
    fn effect_mul(&mut self, mut event: MulMatIdRuntime<'_>) -> Result<(), ()> {
        mul_mat_id_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_mul_reject(&mut self, event: MulMatIdRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(IndexedError::InvalidShape));
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{IndexedError, IndexedKernel, MulMatIdParams, OpAddId, OpMulMatId};
    use crate::Kernel;

    #[test]
    fn add_id_adds_selected_table_row() {
        let tokens = [1.0_f32, 2.0];
        let table = [10.0_f32, 20.0, 3.0, 4.0];
        let ids = [1_i32];
        let mut output = [0.0_f32; 2];
        IndexedKernel::new()
            .process_event(OpAddId::new(&tokens, &table, &ids, &mut output, 2))
            .unwrap();
        assert_eq!(output[0].to_bits(), 4.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 6.0_f32.to_bits());
    }

    #[test]
    fn mul_mat_id_selects_the_expert_matrix() {
        let experts = [2.0_f32, 0.0, 0.0, 3.0];
        let tokens = [4.0_f32, 5.0];
        let ids = [1_i32];
        let mut output = [0.0_f32; 1];
        Kernel::new()
            .process_event(OpMulMatId::new(
                &experts,
                &tokens,
                &ids,
                &mut output,
                MulMatIdParams {
                    experts: 2,
                    n_in: 2,
                    n_out: 1,
                },
            ))
            .unwrap();
        assert_eq!(output[0].to_bits(), 15.0_f32.to_bits());
    }

    #[test]
    fn add_id_rejects_without_mutation() {
        let tokens = [1.0_f32];
        let table = [1.0_f32];
        let ids = [3_i32];
        let mut output = [9.0_f32; 1];
        let mut kernel = IndexedKernel::new();
        assert_eq!(
            kernel.process_event(OpAddId::new(&tokens, &table, &ids, &mut output, 1)),
            Err(IndexedError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 9.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_add_id() {
        let tokens = [1.0_f32, 2.0];
        let table = [10.0_f32, 20.0, 3.0, 4.0];
        let ids = [1_i32];
        let mut output = [0.0_f32; 2];
        Kernel::new()
            .process_event(OpAddId::new(&tokens, &table, &ids, &mut output, 2))
            .unwrap();
        assert_eq!(output[0].to_bits(), 4.0_f32.to_bits());
    }
}
