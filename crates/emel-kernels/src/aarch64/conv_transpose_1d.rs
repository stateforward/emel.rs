//! Safe `AArch64` NEON F32 transposed-convolution kernel.
//!
//! This bounded target slice ports the pinned `op_conv_transpose_1d` NEON
//! route from `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6`. NEON is
//! resolved before construction through Pulp; this actor never substitutes a
//! scalar backend during dispatch.
//!
//! The public request intentionally models the dense F32 contract that the
//! target action can consume without exposing native tensor metadata. General
//! strided layouts, F16 weights, and the shared scalar transition remain
//! explicit residuals in the parity snapshot.

#![cfg(target_arch = "aarch64")]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use pulp::{Simd, WithSimd, aarch64::Arch};
use sml::sml;

use crate::any::quant::fp16_to_f32;

/// Errors returned by the `AArch64` NEON transposed-convolution actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConvTranspose1dF32Error {
    /// The target did not expose NEON.
    BackendUnavailable,
    /// The dense F32 buffers or convolution dimensions are invalid.
    InvalidShape,
    /// The generated machine received an event outside this API.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for ConvTranspose1dF32Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable => formatter.write_str("AArch64 NEON unavailable"),
            Self::InvalidShape => formatter.write_str("invalid AArch64 conv-transpose F32 shape"),
            Self::UnexpectedEvent => {
                formatter.write_str("unexpected AArch64 conv-transpose F32 event")
            }
            Self::Internal => {
                formatter.write_str("internal AArch64 conv-transpose F32 dispatch error")
            }
        }
    }
}

impl std::error::Error for ConvTranspose1dF32Error {}

/// Result returned after a transposed-convolution event reaches RTC.
pub type ConvTranspose1dF32Result = Result<(), ConvTranspose1dF32Error>;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:287-297,408-417,1026-1049";
const PINNED_ACTION_SPAN: &str =
    "src/emel/kernel/aarch64/actions.hpp:1252-1258,7074-7121,8770-8779";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:800-818";

/// Dense F32 dimensions for the pinned transposed-convolution route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConvTranspose1dF32Shape {
    /// Number of taps in each input/output-channel pair.
    pub kernel: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of input positions.
    pub input_length: usize,
    /// Input-to-output stride.
    pub stride: i32,
    /// Explicit padding; the pinned route requires zero.
    pub padding: i32,
    /// Tap dilation; the pinned route requires one.
    pub dilation: i32,
}

/// Typed dense F32-weight transposed-convolution request.
#[derive(Debug)]
pub struct OpAarch64ConvTranspose1dF32<'a> {
    weights: &'a [f32],
    input: &'a [f32],
    shape: ConvTranspose1dF32Shape,
}

/// Typed dense F16-weight transposed-convolution request with F32 input and
/// destination storage exactly as the pinned route consumes them.
#[derive(Debug)]
pub struct OpAarch64ConvTranspose1dF16<'a> {
    weights: &'a [u16],
    input: &'a [f32],
    shape: ConvTranspose1dF32Shape,
}

impl<'a> OpAarch64ConvTranspose1dF16<'a> {
    /// Creates a request. Buffer and dimension validation occurs in a guard.
    #[must_use]
    pub const fn new(weights: &'a [u16], input: &'a [f32], shape: ConvTranspose1dF32Shape) -> Self {
        Self {
            weights,
            input,
            shape,
        }
    }
}

impl<'a> OpAarch64ConvTranspose1dF32<'a> {
    /// Creates a request. Buffer and dimension validation occurs in a guard.
    #[must_use]
    pub const fn new(weights: &'a [f32], input: &'a [f32], shape: ConvTranspose1dF32Shape) -> Self {
        Self {
            weights,
            input,
            shape,
        }
    }
}

/// Explicitly reports an event outside the F32 actor's API.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedAarch64ConvTranspose1dF32;

struct Runtime<'a> {
    weights: &'a [f32],
    input: &'a [f32],
    output: &'a mut [f32],
    shape: ConvTranspose1dF32Shape,
    result: &'a Cell<ConvTranspose1dF32Result>,
}

struct F16WeightsRuntime<'a> {
    weights: &'a [u16],
    input: &'a [f32],
    output: &'a mut [f32],
    shape: ConvTranspose1dF32Shape,
    result: &'a Cell<ConvTranspose1dF32Result>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<ConvTranspose1dF32Result>,
}

#[derive(Debug)]
struct Context {
    backend_available: bool,
    backend: pulp::aarch64::Neon,
}

sml! {
    ConvTranspose1dF32Machine<'dispatch> {
        "ready"_s <= *"ready"_s + ConvTranspose1d(Runtime<'dispatch>)
            [guard_ready] / effect_conv_transpose,
        "ready"_s <= "ready"_s + ConvTranspose1d(Runtime<'dispatch>)
            [guard_invalid] / effect_invalid,
        "ready"_s <= "ready"_s + ConvTranspose1dF16Weights(F16WeightsRuntime<'dispatch>)
            [guard_f16_weights_ready] / effect_conv_transpose_f16_weights,
        "ready"_s <= "ready"_s + ConvTranspose1dF16Weights(F16WeightsRuntime<'dispatch>)
            [guard_f16_weights_invalid] / effect_conv_transpose_f16_weights_invalid,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion `AArch64` NEON transposed-convolution actor.
pub struct ConvTranspose1dF32Kernel {
    machine: ConvTranspose1dF32MachineStateMachine<Context>,
}

impl fmt::Debug for ConvTranspose1dF32Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConvTranspose1dF32Kernel")
            .finish_non_exhaustive()
    }
}

impl ConvTranspose1dF32Kernel {
    /// Resolves NEON before construction; no scalar backend is selected.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        let Arch::Neon(backend) = Arch::new() else {
            return None;
        };
        Some(Self {
            machine: ConvTranspose1dF32MachineStateMachine::new(Context {
                backend_available: true,
                backend,
            }),
        })
    }

    /// Dispatches one typed event synchronously to completion.
    pub fn process_event<E: ConvTranspose1dF32Event>(
        &mut self,
        event: E,
        output: &mut [f32],
    ) -> E::Output {
        event.dispatch(self, output)
    }

    /// Reports whether the generated actor remains in its ready state.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&ConvTranspose1dF32MachineStates::Ready)
    }

    fn conv_transpose(
        &mut self,
        weights: &[f32],
        input: &[f32],
        shape: ConvTranspose1dF32Shape,
        output: &mut [f32],
    ) -> ConvTranspose1dF32Result {
        let result = Cell::new(Err(ConvTranspose1dF32Error::UnexpectedEvent));
        self.machine
            .process_event(ConvTranspose1dF32MachineEvents::ConvTranspose1d(Runtime {
                weights,
                input,
                output,
                shape,
                result: &result,
            }))
            .map_err(|_| ConvTranspose1dF32Error::Internal)?;
        result.get()
    }

    fn conv_transpose_f16(
        &mut self,
        weights: &[u16],
        input: &[f32],
        shape: ConvTranspose1dF32Shape,
        output: &mut [f32],
    ) -> ConvTranspose1dF32Result {
        let result = Cell::new(Err(ConvTranspose1dF32Error::UnexpectedEvent));
        self.machine
            .process_event(ConvTranspose1dF32MachineEvents::ConvTranspose1dF16Weights(
                F16WeightsRuntime {
                    weights,
                    input,
                    output,
                    shape,
                    result: &result,
                },
            ))
            .map_err(|_| ConvTranspose1dF32Error::Internal)?;
        result.get()
    }
}

/// Event implemented by the `AArch64` transposed-convolution actor.
pub trait ConvTranspose1dF32Event: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut ConvTranspose1dF32Kernel, output: &mut [f32]) -> Self::Output;
}

impl ConvTranspose1dF32Event for OpAarch64ConvTranspose1dF32<'_> {
    type Output = ConvTranspose1dF32Result;

    fn dispatch(self, actor: &mut ConvTranspose1dF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.conv_transpose(self.weights, self.input, self.shape, output)
    }
}

impl ConvTranspose1dF32Event for OpAarch64ConvTranspose1dF16<'_> {
    type Output = ConvTranspose1dF32Result;

    fn dispatch(self, actor: &mut ConvTranspose1dF32Kernel, output: &mut [f32]) -> Self::Output {
        actor.conv_transpose_f16(self.weights, self.input, self.shape, output)
    }
}

impl ConvTranspose1dF32Event for UnexpectedAarch64ConvTranspose1dF32 {
    type Output = ConvTranspose1dF32Result;

    fn dispatch(self, actor: &mut ConvTranspose1dF32Kernel, output: &mut [f32]) -> Self::Output {
        let _ = output;
        let result = Cell::new(Err(ConvTranspose1dF32Error::UnexpectedEvent));
        actor
            .machine
            .process_event(ConvTranspose1dF32MachineEvents::Unexpected(
                UnexpectedRuntime { result: &result },
            ))
            .map_err(|_| ConvTranspose1dF32Error::Internal)?;
        result.get()
    }
}

impl ConvTranspose1dF32MachineStateMachineContext for Context {
    fn guard_ready(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid(event))
    }

    fn guard_invalid(&self, event: &Runtime<'_>) -> Result<bool, ()> {
        Ok(!request_valid(event))
    }

    fn guard_f16_weights_ready(&self, event: &F16WeightsRuntime<'_>) -> Result<bool, ()> {
        Ok(self.backend_available && request_valid_f16(event))
    }

    fn guard_f16_weights_invalid(&self, event: &F16WeightsRuntime<'_>) -> Result<bool, ()> {
        Ok(!request_valid_f16(event))
    }

    fn effect_conv_transpose(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        Simd::vectorize(
            self.backend,
            ConvTransposeOperation {
                weights: event.weights,
                input: event.input,
                output: event.output,
                shape: event.shape,
            },
        );
        event.result.set(Ok(()));
        Ok(())
    }

    #[allow(clippy::suboptimal_flops)]
    fn effect_conv_transpose_f16_weights(
        &mut self,
        event: F16WeightsRuntime<'_>,
    ) -> Result<(), ()> {
        let stride = usize::try_from(event.shape.stride).expect("guard-proven positive stride");
        let output_length = (event.shape.input_length - 1) * stride + event.shape.kernel;
        event.output.fill(0.0);

        let mut input_channel = 0;
        while input_channel < event.shape.in_channels {
            let mut input_index = 0;
            while input_index < event.shape.input_length {
                let input_value =
                    event.input[input_channel * event.shape.input_length + input_index];
                let mut output_channel = 0;
                while output_channel < event.shape.out_channels {
                    let weight_offset = (input_channel * event.shape.out_channels + output_channel)
                        * event.shape.kernel;
                    let output_offset = output_channel * output_length + input_index * stride;
                    let weight_row =
                        &event.weights[weight_offset..weight_offset + event.shape.kernel];
                    let output_row =
                        &mut event.output[output_offset..output_offset + event.shape.kernel];
                    for (output_value, weight_bits) in output_row.iter_mut().zip(weight_row) {
                        let accumulated = *output_value;
                        *output_value = accumulated + input_value * fp16_to_f32(*weight_bits);
                    }
                    output_channel += 1;
                }
                input_index += 1;
            }
            input_channel += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_conv_transpose_f16_weights_invalid(
        &mut self,
        event: F16WeightsRuntime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(ConvTranspose1dF32Error::InvalidShape));
        Ok(())
    }

    fn effect_invalid(&mut self, event: Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(ConvTranspose1dF32Error::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .set(Err(ConvTranspose1dF32Error::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

const MAX_CONV_GUARD_EXTENT: usize = 1usize << 31;

fn request_valid_for_lengths(
    weights_len: usize,
    input_len: usize,
    output_len: usize,
    shape: ConvTranspose1dF32Shape,
) -> bool {
    let Ok(stride) = usize::try_from(shape.stride) else {
        return false;
    };
    let Some(output_length) = shape
        .input_length
        .checked_sub(1)
        .and_then(|length| length.checked_mul(stride))
        .and_then(|length| length.checked_add(shape.kernel))
    else {
        return false;
    };
    shape.kernel > 0
        && shape.out_channels > 0
        && shape.in_channels > 0
        && shape.input_length > 0
        && shape.stride > 0
        && shape.padding == 0
        && shape.dilation == 1
        && shape.kernel <= MAX_CONV_GUARD_EXTENT
        && shape.out_channels <= MAX_CONV_GUARD_EXTENT
        && shape.in_channels <= MAX_CONV_GUARD_EXTENT
        && shape.input_length <= MAX_CONV_GUARD_EXTENT
        && stride <= MAX_CONV_GUARD_EXTENT
        && output_length <= MAX_CONV_GUARD_EXTENT
        && output_length
            .checked_mul(shape.out_channels)
            .is_some_and(|count| count <= MAX_CONV_GUARD_EXTENT)
        && shape
            .kernel
            .checked_mul(shape.out_channels)
            .and_then(|value| value.checked_mul(shape.in_channels))
            == Some(weights_len)
        && shape.input_length.checked_mul(shape.in_channels) == Some(input_len)
        && output_length.checked_mul(shape.out_channels) == Some(output_len)
}

fn request_valid(event: &Runtime<'_>) -> bool {
    request_valid_for_lengths(
        event.weights.len(),
        event.input.len(),
        event.output.len(),
        event.shape,
    )
}

fn request_valid_f16(event: &F16WeightsRuntime<'_>) -> bool {
    request_valid_for_lengths(
        event.weights.len(),
        event.input.len(),
        event.output.len(),
        event.shape,
    )
}

struct ConvTransposeOperation<'a> {
    weights: &'a [f32],
    input: &'a [f32],
    output: &'a mut [f32],
    shape: ConvTranspose1dF32Shape,
}

impl WithSimd for ConvTransposeOperation<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self {
            weights,
            input,
            output,
            shape,
        } = self;
        let stride = usize::try_from(shape.stride).expect("guard-proven positive stride");
        let output_length = (shape.input_length - 1) * stride + shape.kernel;
        output.fill(0.0);

        let mut input_channel = 0;
        while input_channel < shape.in_channels {
            let mut input_index = 0;
            while input_index < shape.input_length {
                let input_value = input[input_channel * shape.input_length + input_index];
                let input_vector = simd.splat_f32s(input_value);
                let mut output_channel = 0;
                while output_channel < shape.out_channels {
                    let weight_offset =
                        (input_channel * shape.out_channels + output_channel) * shape.kernel;
                    let output_offset = output_channel * output_length + input_index * stride;
                    let weight_row = &weights[weight_offset..weight_offset + shape.kernel];
                    let output_row = &mut output[output_offset..output_offset + shape.kernel];
                    let (weight_vectors, weight_tail) = S::as_simd_f32s(weight_row);
                    let (output_vectors, output_tail) = S::as_mut_simd_f32s(output_row);
                    for (output_vector, weight_vector) in
                        output_vectors.iter_mut().zip(weight_vectors)
                    {
                        *output_vector =
                            simd.mul_add_f32s(input_vector, *weight_vector, *output_vector);
                    }
                    for (output_value, weight_value) in output_tail.iter_mut().zip(weight_tail) {
                        *output_value = input_value.mul_add(*weight_value, *output_value);
                    }
                    output_channel += 1;
                }
                input_index += 1;
            }
            input_channel += 1;
        }
    }
}
