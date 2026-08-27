//! Dense SSM convolution and scan kernels.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_ssm_conv` and `op_ssm_scan` and names `exec_op_ssm*` routes on the
//! arch machines. Those action/guard types are not defined in the pinned
//! headers, so this actor is source-contract.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by SSM dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SsmError {
    /// Sequence, channel, or kernel geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for SsmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid ssm shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected ssm event"),
            Self::Internal => formatter.write_str("internal ssm dispatch error"),
        }
    }
}

impl std::error::Error for SsmError {}

/// Result returned by SSM dispatch.
pub type SsmResult = Result<(), SsmError>;

/// Depthwise causal convolution over a dense `[sequence, channels]` buffer.
#[derive(Debug)]
pub struct OpSsmConv<'a> {
    input: &'a [f32],
    kernel: &'a [f32],
    output: &'a mut [f32],
    sequence: usize,
    channels: usize,
    kernel_len: usize,
}

impl<'a> OpSsmConv<'a> {
    /// Creates an SSM-conv request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        input: &'a [f32],
        kernel: &'a [f32],
        output: &'a mut [f32],
        sequence: usize,
        channels: usize,
        kernel_len: usize,
    ) -> Self {
        Self {
            input,
            kernel,
            output,
            sequence,
            channels,
            kernel_len,
        }
    }

    const fn valid(&self) -> bool {
        self.sequence > 0
            && self.channels > 0
            && self.kernel_len > 0
            && self.input.len() == self.sequence.saturating_mul(self.channels)
            && self.output.len() == self.input.len()
            && self.kernel.len() == self.kernel_len.saturating_mul(self.channels)
    }
}

/// Sequence/channel geometry for [`OpSsmScan`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsmScanParams {
    /// Time steps.
    pub sequence: usize,
    /// Channel count.
    pub channels: usize,
}

/// Selective discrete SSM scan over dense `[sequence, channels]` buffers.
#[derive(Debug)]
pub struct OpSsmScan<'a> {
    input: &'a [f32],
    decay: &'a [f32],
    inject: &'a [f32],
    mix: &'a [f32],
    skip: &'a [f32],
    output: &'a mut [f32],
    params: SsmScanParams,
}

impl<'a> OpSsmScan<'a> {
    /// Creates an SSM-scan request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        input: &'a [f32],
        decay: &'a [f32],
        inject: &'a [f32],
        mix: &'a [f32],
        skip: &'a [f32],
        output: &'a mut [f32],
        params: SsmScanParams,
    ) -> Self {
        Self {
            input,
            decay,
            inject,
            mix,
            skip,
            output,
            params,
        }
    }

    const fn valid(&self) -> bool {
        let seq_len = self.params.sequence.saturating_mul(self.params.channels);
        self.params.sequence > 0
            && self.params.channels > 0
            && self.input.len() == seq_len
            && self.decay.len() == seq_len
            && self.inject.len() == seq_len
            && self.mix.len() == seq_len
            && self.skip.len() == self.params.channels
            && self.output.len() == seq_len
    }
}

struct SsmConvRuntime<'a> {
    event: OpSsmConv<'a>,
    result: &'a Cell<SsmResult>,
}

struct SsmScanRuntime<'a> {
    event: OpSsmScan<'a>,
    result: &'a Cell<SsmResult>,
}

#[derive(Default)]
struct Context;

sml! {
    SsmMachine<'dispatch> {
        "ready"_s <= *"ready"_s + SsmConv(SsmConvRuntime<'dispatch>) [guard_conv_valid] / effect_conv,
        "ready"_s <= "ready"_s + SsmConv(SsmConvRuntime<'dispatch>) [guard_conv_invalid] / effect_conv_reject,
        "ready"_s <= "ready"_s + SsmScan(SsmScanRuntime<'dispatch>) [guard_scan_valid] / effect_scan,
        "ready"_s <= "ready"_s + SsmScan(SsmScanRuntime<'dispatch>) [guard_scan_invalid] / effect_scan_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for SSM events.
pub trait SsmEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut SsmKernel) -> Self::Output;
}

/// Single-writer SSM actor.
pub struct SsmKernel {
    machine: SsmMachineStateMachine<Context>,
}

impl fmt::Debug for SsmKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("SsmKernel").finish_non_exhaustive()
    }
}

impl Default for SsmKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl SsmKernel {
    /// Constructs an independent SSM actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: SsmMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed SSM event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`SsmError::InvalidShape`] when the dense buffers cannot form
    /// the requested convolution or scan geometry.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: SsmEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn conv(&mut self, event: OpSsmConv<'_>) -> SsmResult {
        let result = Cell::new(Err(SsmError::UnexpectedEvent));
        self.machine
            .process_event(SsmMachineEvents::SsmConv(SsmConvRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| SsmError::Internal)?;
        assert!(
            self.machine.is(&SsmMachineStates::Ready),
            "ssm machine must return to ready after dispatch"
        );
        result.get()
    }

    fn scan(&mut self, event: OpSsmScan<'_>) -> SsmResult {
        let result = Cell::new(Err(SsmError::UnexpectedEvent));
        self.machine
            .process_event(SsmMachineEvents::SsmScan(SsmScanRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| SsmError::Internal)?;
        assert!(
            self.machine.is(&SsmMachineStates::Ready),
            "ssm machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&SsmMachineStates::Ready)
    }
}

impl SsmEvent for OpSsmConv<'_> {
    type Output = SsmResult;
    fn dispatch(self, kernel: &mut SsmKernel) -> Self::Output {
        kernel.conv(self)
    }
}

impl SsmEvent for OpSsmScan<'_> {
    type Output = SsmResult;
    fn dispatch(self, kernel: &mut SsmKernel) -> Self::Output {
        kernel.scan(self)
    }
}

fn ssm_conv_values(event: &mut OpSsmConv<'_>) {
    let channels = event.channels;
    for time in 0..event.sequence {
        for channel in 0..channels {
            let mut acc = 0.0_f32;
            for tap in 0..event.kernel_len {
                if time >= tap {
                    let sample = event.input[(time - tap) * channels + channel];
                    let weight = event.kernel[tap * channels + channel];
                    acc = sample.mul_add(weight, acc);
                }
            }
            event.output[time * channels + channel] = acc;
        }
    }
}

fn ssm_scan_values(event: &mut OpSsmScan<'_>) {
    let channels = event.params.channels;
    for channel in 0..channels {
        let mut hidden = 0.0_f32;
        for time in 0..event.params.sequence {
            let index = time * channels + channel;
            hidden = event.decay[index].mul_add(hidden, event.inject[index] * event.input[index]);
            event.output[index] =
                event.mix[index].mul_add(hidden, event.skip[channel] * event.input[index]);
        }
    }
}

impl SsmMachineStateMachineContext for Context {
    fn guard_conv_valid(&self, event: &SsmConvRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_conv_invalid(&self, event: &SsmConvRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_scan_valid(&self, event: &SsmScanRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_scan_invalid(&self, event: &SsmScanRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn effect_conv(&mut self, mut event: SsmConvRuntime<'_>) -> Result<(), ()> {
        ssm_conv_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_conv_reject(&mut self, event: SsmConvRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(SsmError::InvalidShape));
        Ok(())
    }

    fn effect_scan(&mut self, mut event: SsmScanRuntime<'_>) -> Result<(), ()> {
        ssm_scan_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_scan_reject(&mut self, event: SsmScanRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(SsmError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{OpSsmConv, OpSsmScan, SsmError, SsmKernel, SsmScanParams};
    use crate::Kernel;

    #[test]
    fn ssm_conv_unit_kernel_copies_input() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let kernel = [1.0_f32, 1.0];
        let mut output = [0.0_f32; 4];
        let mut actor = SsmKernel::new();
        actor
            .process_event(OpSsmConv::new(&input, &kernel, &mut output, 2, 2, 1))
            .unwrap();
        assert_eq!(output[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(output[3].to_bits(), 4.0_f32.to_bits());
        assert!(actor.is_ready());
    }

    #[test]
    fn ssm_scan_passthrough_when_decay_is_zero() {
        let input = [2.0_f32, 3.0];
        let decay = [0.0_f32, 0.0];
        let inject = [1.0_f32, 1.0];
        let mix = [1.0_f32, 1.0];
        let skip = [0.0_f32, 0.0];
        let mut output = [0.0_f32; 2];
        SsmKernel::new()
            .process_event(OpSsmScan::new(
                &input,
                &decay,
                &inject,
                &mix,
                &skip,
                &mut output,
                SsmScanParams {
                    sequence: 1,
                    channels: 2,
                },
            ))
            .unwrap();
        assert_eq!(output[0].to_bits(), 2.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 3.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_ssm_conv() {
        let input = [5.0_f32];
        let kernel = [2.0_f32];
        let mut output = [0.0_f32; 1];
        Kernel::new()
            .process_event(OpSsmConv::new(&input, &kernel, &mut output, 1, 1, 1))
            .unwrap();
        assert_eq!(output[0].to_bits(), 10.0_f32.to_bits());
    }

    #[test]
    fn ssm_conv_rejects_without_mutation() {
        let input = [1.0_f32];
        let kernel = [1.0_f32, 1.0];
        let mut output = [7.0_f32; 1];
        let mut actor = SsmKernel::new();
        assert_eq!(
            actor.process_event(OpSsmConv::new(&input, &kernel, &mut output, 1, 1, 1)),
            Err(SsmError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 7.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_ssm_scan() {
        let input = [2.0_f32, 3.0];
        let decay = [0.0_f32, 0.0];
        let inject = [1.0_f32, 1.0];
        let mix = [1.0_f32, 1.0];
        let skip = [0.0_f32, 0.0];
        let mut output = [0.0_f32; 2];
        Kernel::new()
            .process_event(OpSsmScan::new(
                &input,
                &decay,
                &inject,
                &mix,
                &skip,
                &mut output,
                SsmScanParams {
                    sequence: 1,
                    channels: 2,
                },
            ))
            .unwrap();
        assert_eq!(output[1].to_bits(), 3.0_f32.to_bits());
    }
}
