//! Dense RWKV WKV kernels.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_rwkv_wkv6` and `op_rwkv_wkv7` and names `exec_op_*` routes on the arch
//! machines. Those action/guard types are not defined in the pinned headers,
//! so this actor is source-contract.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by RWKV dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RwkvError {
    /// Sequence or channel geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for RwkvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid rwkv shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected rwkv event"),
            Self::Internal => formatter.write_str("internal rwkv dispatch error"),
        }
    }
}

impl std::error::Error for RwkvError {}

/// Result returned by RWKV dispatch.
pub type RwkvResult = Result<(), RwkvError>;

/// Sequence/channel geometry for WKV.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RwkvParams {
    /// Time steps.
    pub sequence: usize,
    /// Channel count.
    pub channels: usize,
}

/// Linear WKV-6: `y = r * (u*k*v + h)` then `h = w*h + k*v`.
#[derive(Debug)]
pub struct OpRwkvWkv6<'a> {
    receptance: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    decay: &'a [f32],
    bonus: &'a [f32],
    output: &'a mut [f32],
    params: RwkvParams,
}

impl<'a> OpRwkvWkv6<'a> {
    /// Creates a WKV-6 request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        receptance: &'a [f32],
        key: &'a [f32],
        value: &'a [f32],
        decay: &'a [f32],
        bonus: &'a [f32],
        output: &'a mut [f32],
        params: RwkvParams,
    ) -> Self {
        Self {
            receptance,
            key,
            value,
            decay,
            bonus,
            output,
            params,
        }
    }

    const fn valid(&self) -> bool {
        seq_valid(
            self.params,
            self.receptance,
            self.key,
            self.value,
            self.decay,
            self.output,
        ) && self.bonus.len() == self.params.channels
    }
}

/// Linear WKV-7: `y = r * (h + k*v)` then `h = w*h + k*v`.
#[derive(Debug)]
pub struct OpRwkvWkv7<'a> {
    receptance: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    decay: &'a [f32],
    output: &'a mut [f32],
    params: RwkvParams,
}

impl<'a> OpRwkvWkv7<'a> {
    /// Creates a WKV-7 request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        receptance: &'a [f32],
        key: &'a [f32],
        value: &'a [f32],
        decay: &'a [f32],
        output: &'a mut [f32],
        params: RwkvParams,
    ) -> Self {
        Self {
            receptance,
            key,
            value,
            decay,
            output,
            params,
        }
    }

    const fn valid(&self) -> bool {
        seq_valid(
            self.params,
            self.receptance,
            self.key,
            self.value,
            self.decay,
            self.output,
        )
    }
}

const fn seq_valid(
    params: RwkvParams,
    receptance: &[f32],
    key: &[f32],
    value: &[f32],
    decay: &[f32],
    output: &[f32],
) -> bool {
    let seq_len = params.sequence.saturating_mul(params.channels);
    params.sequence > 0
        && params.channels > 0
        && receptance.len() == seq_len
        && key.len() == seq_len
        && value.len() == seq_len
        && decay.len() == seq_len
        && output.len() == seq_len
}

struct Wkv6Runtime<'a> {
    event: OpRwkvWkv6<'a>,
    result: &'a Cell<RwkvResult>,
}

struct Wkv7Runtime<'a> {
    event: OpRwkvWkv7<'a>,
    result: &'a Cell<RwkvResult>,
}

#[derive(Default)]
struct Context;

sml! {
    RwkvMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Wkv6(Wkv6Runtime<'dispatch>) [guard_wkv6_valid] / effect_wkv6,
        "ready"_s <= "ready"_s + Wkv6(Wkv6Runtime<'dispatch>) [guard_wkv6_invalid] / effect_wkv6_reject,
        "ready"_s <= "ready"_s + Wkv7(Wkv7Runtime<'dispatch>) [guard_wkv7_valid] / effect_wkv7,
        "ready"_s <= "ready"_s + Wkv7(Wkv7Runtime<'dispatch>) [guard_wkv7_invalid] / effect_wkv7_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for RWKV events.
pub trait RwkvEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut RwkvKernel) -> Self::Output;
}

/// Single-writer RWKV actor.
pub struct RwkvKernel {
    machine: RwkvMachineStateMachine<Context>,
}

impl fmt::Debug for RwkvKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("RwkvKernel").finish_non_exhaustive()
    }
}

impl Default for RwkvKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl RwkvKernel {
    /// Constructs an independent RWKV actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: RwkvMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed RWKV event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`RwkvError::InvalidShape`] when the dense buffers cannot form
    /// the requested WKV geometry.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: RwkvEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn wkv6(&mut self, event: OpRwkvWkv6<'_>) -> RwkvResult {
        let result = Cell::new(Err(RwkvError::UnexpectedEvent));
        self.machine
            .process_event(RwkvMachineEvents::Wkv6(Wkv6Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| RwkvError::Internal)?;
        assert!(
            self.machine.is(&RwkvMachineStates::Ready),
            "rwkv machine must return to ready after dispatch"
        );
        result.get()
    }

    fn wkv7(&mut self, event: OpRwkvWkv7<'_>) -> RwkvResult {
        let result = Cell::new(Err(RwkvError::UnexpectedEvent));
        self.machine
            .process_event(RwkvMachineEvents::Wkv7(Wkv7Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| RwkvError::Internal)?;
        assert!(
            self.machine.is(&RwkvMachineStates::Ready),
            "rwkv machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&RwkvMachineStates::Ready)
    }
}

impl RwkvEvent for OpRwkvWkv6<'_> {
    type Output = RwkvResult;
    fn dispatch(self, kernel: &mut RwkvKernel) -> Self::Output {
        kernel.wkv6(self)
    }
}

impl RwkvEvent for OpRwkvWkv7<'_> {
    type Output = RwkvResult;
    fn dispatch(self, kernel: &mut RwkvKernel) -> Self::Output {
        kernel.wkv7(self)
    }
}

fn wkv6_values(event: &mut OpRwkvWkv6<'_>) {
    let channels = event.params.channels;
    for channel in 0..channels {
        let mut hidden = 0.0_f32;
        for time in 0..event.params.sequence {
            let index = time * channels + channel;
            let kv = event.key[index] * event.value[index];
            event.output[index] =
                event.receptance[index] * event.bonus[channel].mul_add(kv, hidden);
            hidden = event.decay[index].mul_add(hidden, kv);
        }
    }
}

fn wkv7_values(event: &mut OpRwkvWkv7<'_>) {
    let channels = event.params.channels;
    for channel in 0..channels {
        let mut hidden = 0.0_f32;
        for time in 0..event.params.sequence {
            let index = time * channels + channel;
            let kv = event.key[index] * event.value[index];
            event.output[index] = event.receptance[index] * (hidden + kv);
            hidden = event.decay[index].mul_add(hidden, kv);
        }
    }
}

impl RwkvMachineStateMachineContext for Context {
    fn guard_wkv6_valid(&self, event: &Wkv6Runtime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_wkv6_invalid(&self, event: &Wkv6Runtime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn guard_wkv7_valid(&self, event: &Wkv7Runtime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_wkv7_invalid(&self, event: &Wkv7Runtime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn effect_wkv6(&mut self, mut event: Wkv6Runtime<'_>) -> Result<(), ()> {
        wkv6_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_wkv6_reject(&mut self, event: Wkv6Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(RwkvError::InvalidShape));
        Ok(())
    }
    fn effect_wkv7(&mut self, mut event: Wkv7Runtime<'_>) -> Result<(), ()> {
        wkv7_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_wkv7_reject(&mut self, event: Wkv7Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(RwkvError::InvalidShape));
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{OpRwkvWkv6, OpRwkvWkv7, RwkvError, RwkvKernel, RwkvParams};
    use crate::Kernel;

    fn unit() -> RwkvParams {
        RwkvParams {
            sequence: 1,
            channels: 1,
        }
    }

    #[test]
    fn wkv7_first_step_is_receptance_times_kv() {
        let receptance = [2.0_f32];
        let key = [3.0_f32];
        let value = [4.0_f32];
        let decay = [0.0_f32];
        let mut output = [0.0_f32; 1];
        RwkvKernel::new()
            .process_event(OpRwkvWkv7::new(
                &receptance,
                &key,
                &value,
                &decay,
                &mut output,
                unit(),
            ))
            .unwrap();
        assert_eq!(output[0].to_bits(), 24.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_wkv6() {
        let receptance = [1.0_f32];
        let key = [1.0_f32];
        let value = [2.0_f32];
        let decay = [0.0_f32];
        let bonus = [3.0_f32];
        let mut output = [0.0_f32; 1];
        Kernel::new()
            .process_event(OpRwkvWkv6::new(
                &receptance,
                &key,
                &value,
                &decay,
                &bonus,
                &mut output,
                unit(),
            ))
            .unwrap();
        assert_eq!(output[0].to_bits(), 6.0_f32.to_bits());
    }

    #[test]
    fn wkv7_rejects_without_mutation() {
        let receptance = [1.0_f32];
        let key = [1.0_f32];
        let value = [1.0_f32];
        let decay = [1.0_f32, 1.0];
        let mut output = [9.0_f32; 1];
        let mut kernel = RwkvKernel::new();
        assert_eq!(
            kernel.process_event(OpRwkvWkv7::new(
                &receptance,
                &key,
                &value,
                &decay,
                &mut output,
                unit()
            )),
            Err(RwkvError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 9.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_wkv7() {
        let receptance = [2.0_f32];
        let key = [3.0_f32];
        let value = [4.0_f32];
        let decay = [0.0_f32];
        let mut output = [0.0_f32; 1];
        Kernel::new()
            .process_event(OpRwkvWkv7::new(
                &receptance,
                &key,
                &value,
                &decay,
                &mut output,
                unit(),
            ))
            .unwrap();
        assert_eq!(output[0].to_bits(), 24.0_f32.to_bits());
    }
}
