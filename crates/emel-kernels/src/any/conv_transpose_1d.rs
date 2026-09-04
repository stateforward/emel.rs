//! Allocation-free F32 1-D transposed convolution with explicit SML dispatch.
//!
//! The pinned `emel.cpp` source is commit
//! `843a117386ef17dc5a50549bbfc821074c2141d6`. Its
//! `src/emel/kernel/detail.hpp:4955-5009` guard reads integer parameters
//! `{s0, p0, d0}`, accepts only `s0 > 0`, `p0 == 0`, and `d0 == 1`, and
//! requires weights `[kernel, out_channels, in_channels]`, input
//! `[length, in_channels]`, and output `[out_length, out_channels]`, where
//! `out_length = (length - 1) * s0 + kernel`. The F32 scatter/accumulate loop
//! is `detail.hpp:5011-5064`. The x86-64 transition rows are
//! `src/emel/kernel/x86_64/sm.hpp:680-693`; the corresponding aarch64 rows
//! are `src/emel/kernel/aarch64/sm.hpp:801-818`. The pinned guard also has an
//! F16-weight transition, but this safe view API carries F32 storage only, so
//! that route remains an explicit residual rather than an implicit conversion.
//!
//! This module ports only the F32 weight route. Events carry caller-owned
//! [`TensorView`](crate::any::tensor_view::TensorView) and
//! [`TensorViewMut`](crate::any::tensor_view::TensorViewMut) values plus explicit,
//! `Copy` operation parameters. F16 weights remain a documented residual.

#![allow(clippy::derive_partial_eq_without_eq)]

use core::cell::{Cell, RefCell};
use core::fmt;

use sml::sml;

use super::tensor_view::{TensorView, TensorViewMut};
/// Executes Mimi's native depthwise stride-2 transposed-convolution leaf over
/// canonical F32 weight bytes.
///
/// The caller owns latent, output, overlap state, and workspace; this helper
/// performs no allocation and preserves the pinned channel-major overlap
/// arithmetic.
///
/// The serialized weight layout is `[taps, 1, channels]`, with each channel's
/// taps contiguous. One latent column emits two time-major samples and retains
/// `taps - 2` samples per channel for the next call.
#[allow(clippy::cast_possible_truncation)]
pub fn native_depthwise_stride2_f32(
    weights: &[u8],
    dim: usize,
    taps: usize,
    state: &mut [f32],
    latent: &[f32],
    frame: &mut [f32],
    workspace: &mut [f32],
) -> bool {
    const STRIDE: usize = 2;
    let Some(tail_per_channel) = taps.checked_sub(STRIDE) else {
        return false;
    };
    let Some(full_len) = dim.checked_mul(taps) else {
        return false;
    };
    let Some(tail_len) = dim.checked_mul(tail_per_channel) else {
        return false;
    };
    let Some(frame_len) = dim.checked_mul(STRIDE) else {
        return false;
    };
    let Some(weight_count) = full_len.checked_mul(4) else {
        return false;
    };
    if dim == 0
        || taps < STRIDE
        || weights.len() < weight_count
        || latent.len() < dim
        || frame.len() < frame_len
        || state.len() < tail_len
        || workspace.len() < dim.saturating_add(full_len)
    {
        return false;
    }
    let (input, full) = workspace.split_at_mut(dim);
    input[..dim].copy_from_slice(&latent[..dim]);
    full[..full_len].fill(0.0);
    for (channel, input_value) in input.iter().copied().enumerate().take(dim) {
        let base = channel * taps;
        let tail_base = channel * tail_per_channel;
        for tap in 0..taps {
            let offset = (base + tap) * 4;
            let weight = f32::from_le_bytes([
                weights[offset],
                weights[offset + 1],
                weights[offset + 2],
                weights[offset + 3],
            ]);
            full[base + tap] = input_value * weight;
        }
        for tap in 0..tail_per_channel {
            full[base + tap] += state[tail_base + tap];
        }
    }
    for channel in 0..dim {
        let base = channel * taps;
        let tail_base = channel * tail_per_channel;
        for tap in 0..tail_per_channel {
            state[tail_base + tap] = full[base + STRIDE + tap];
        }
        for time in 0..STRIDE {
            frame[time * dim + channel] = full[base + time];
        }
    }
    true
}

/// Parameters matching the pinned `op_params` integer slots `{s0, p0, d0}`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConvTranspose1dParams {
    /// Input-to-output stride. The maintained route requires this to be > 0.
    pub stride: i32,
    /// Explicit padding. The pinned route requires this to be zero.
    pub padding: i32,
    /// Tap dilation. The pinned route requires this to be one.
    pub dilation: i32,
}

/// Errors returned by F32 transposed-convolution dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConvTranspose1dError {
    /// One or more views failed dtype, shape, stride, or bounds validation.
    InvalidView,
    /// Valid views do not satisfy the pinned convolution dimensions or rank.
    ShapeMismatch,
    /// Parameters are missing or outside the maintained pinned route.
    InvalidParameters,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for ConvTranspose1dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid conv-transpose tensor view"),
            Self::ShapeMismatch => formatter.write_str("conv-transpose tensor shapes differ"),
            Self::InvalidParameters => formatter.write_str("invalid conv-transpose parameters"),
            Self::UnexpectedEvent => formatter.write_str("unexpected conv-transpose event"),
            Self::Internal => formatter.write_str("internal conv-transpose dispatch error"),
        }
    }
}

impl std::error::Error for ConvTranspose1dError {}

/// Applies F32 1-D transposed convolution with pinned weight/input layouts.
#[derive(Debug)]
pub struct OpConvTranspose1d<'a> {
    weights: TensorView<'a>,
    input: TensorView<'a>,
    output: TensorViewMut<'a>,
    params: ConvTranspose1dParams,
}

impl<'a> OpConvTranspose1d<'a> {
    /// Creates an event. View, shape, and parameter validation occurs in
    /// explicit machine guards.
    #[must_use]
    pub const fn new(
        weights: TensorView<'a>,
        input: TensorView<'a>,
        output: TensorViewMut<'a>,
        params: ConvTranspose1dParams,
    ) -> Self {
        Self {
            weights,
            input,
            output,
            params,
        }
    }
}

/// Explicitly reports an event outside the maintained transposed-convolution
/// operation family.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnexpectedConvTranspose1d;

/// Result returned after a transposed-convolution dispatch reaches RTC.
pub type ConvTranspose1dResult = Result<(), ConvTranspose1dError>;

/// Runtime payload used only within one synchronous machine dispatch.
#[derive(Clone, Copy, Debug)]
pub struct ConvTranspose1dRuntime<'dispatch, 'data> {
    weights: TensorView<'data>,
    input: TensorView<'data>,
    output: &'dispatch RefCell<TensorViewMut<'data>>,
    params: ConvTranspose1dParams,
    result: &'dispatch Cell<ConvTranspose1dResult>,
}

#[derive(Clone, Copy, Debug)]
pub struct UnexpectedRuntime<'dispatch> {
    result: &'dispatch Cell<ConvTranspose1dResult>,
}

#[derive(Default)]
struct Context;

sml! {
    ConvTranspose1dMachine<'dispatch, 'data>
    where
        'data: 'dispatch,
    {
        // Each decision has one positive guard and one unguarded fallback.
        // The fallback is the explicit rejected outcome; keeping its predicate
        // implicit avoids same-source guarded fanout while preserving the
        // mutually exclusive pure-guard decision.
        "view_decision"_s <= *"ready"_s + ConvTranspose1d(ConvTranspose1dRuntime<'dispatch, 'data>),
        "views_validated"_s <= "view_decision"_s
            + completion<ConvTranspose1d>(ConvTranspose1dRuntime<'dispatch, 'data>)
            [guard_conv_transpose_1d_views_valid],
        "views_rejected"_s <= "view_decision"_s
            + completion<ConvTranspose1d>(ConvTranspose1dRuntime<'dispatch, 'data>)
            ,
        "parameter_decision"_s <= "views_validated"_s
            + completion<ConvTranspose1d>(ConvTranspose1dRuntime<'dispatch, 'data>),
        "ready"_s <= "views_rejected"_s
            + completion<ConvTranspose1d>(ConvTranspose1dRuntime<'dispatch, 'data>)
            / effect_conv_transpose_1d_view_reject,
        "parameters_validated"_s <= "parameter_decision"_s
            + completion<ConvTranspose1d>(ConvTranspose1dRuntime<'dispatch, 'data>)
            [guard_conv_transpose_1d_parameters_valid],
        "parameters_rejected"_s <= "parameter_decision"_s
            + completion<ConvTranspose1d>(ConvTranspose1dRuntime<'dispatch, 'data>)
            ,
        "shape_decision"_s <= "parameters_validated"_s
            + completion<ConvTranspose1d>(ConvTranspose1dRuntime<'dispatch, 'data>),
        "ready"_s <= "parameters_rejected"_s
            + completion<ConvTranspose1d>(ConvTranspose1dRuntime<'dispatch, 'data>)
            / effect_conv_transpose_1d_parameters_reject,
        "shape_validated"_s <= "shape_decision"_s
            + completion<ConvTranspose1d>(ConvTranspose1dRuntime<'dispatch, 'data>)
            [guard_conv_transpose_1d_shape_valid],
        "shape_rejected"_s <= "shape_decision"_s
            + completion<ConvTranspose1d>(ConvTranspose1dRuntime<'dispatch, 'data>)
            ,
        "execution"_s <= "shape_validated"_s
            + completion<ConvTranspose1d>(ConvTranspose1dRuntime<'dispatch, 'data>),
        "ready"_s <= "shape_rejected"_s
            + completion<ConvTranspose1d>(ConvTranspose1dRuntime<'dispatch, 'data>)
            / effect_conv_transpose_1d_shape_reject,
        "ready"_s <= "execution"_s
            + completion<ConvTranspose1d>(ConvTranspose1dRuntime<'dispatch, 'data>)
            / effect_conv_transpose_1d_execute,

        "unexpected_dispatch"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected,
        "ready"_s <= "unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "unexpected_dispatch"_s + unexpected_event<_> / effect_generic_unexpected,
        "generic_unexpected_dispatch"_s <= "ready"_s + unexpected_event<_>
            / effect_generic_unexpected,
        "ready"_s <= "generic_unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "generic_unexpected_dispatch"_s + unexpected_event<_>
            / effect_generic_unexpected,
        "ready"_s <= "view_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "views_validated"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "views_rejected"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "parameter_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "parameters_validated"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "parameters_rejected"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "shape_decision"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "shape_validated"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "shape_rejected"_s + unexpected_event<_> / effect_generic_unexpected,
        "ready"_s <= "execution"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

/// Single-writer, run-to-completion actor for F32 transposed convolution.
pub struct ConvTranspose1dKernel {
    machine: ConvTranspose1dMachineStateMachine<Context>,
}

impl fmt::Debug for ConvTranspose1dKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConvTranspose1dKernel")
            .finish_non_exhaustive()
    }
}

impl Default for ConvTranspose1dKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvTranspose1dKernel {
    /// Constructs an independent transposed-convolution actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: ConvTranspose1dMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    ///
    /// # Errors
    ///
    /// Returns [`ConvTranspose1dError`] when a view, shape, or parameter
    /// invariant fails, or when generated dispatch cannot complete.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to its ready state.
    pub fn process_event<E: ConvTranspose1dEvent>(&mut self, event: E) -> E::Output {
        let result = event.dispatch(self);
        assert!(self.is_ready());
        result
    }

    /// Reports whether the generated machine is ready for another dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&ConvTranspose1dMachineStates::Ready)
    }

    fn conv_transpose_1d(&mut self, event: OpConvTranspose1d<'_>) -> ConvTranspose1dResult {
        let output = RefCell::new(event.output);
        let result = Cell::new(Err(ConvTranspose1dError::UnexpectedEvent));
        self.machine
            .process_event(ConvTranspose1dMachineEvents::ConvTranspose1d(
                ConvTranspose1dRuntime {
                    weights: event.weights,
                    input: event.input,
                    output: &output,
                    params: event.params,
                    result: &result,
                },
            ))
            .map_err(|_| ConvTranspose1dError::Internal)?;
        result.get()
    }
}

/// Typed event accepted by [`ConvTranspose1dKernel`].
pub trait ConvTranspose1dEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut ConvTranspose1dKernel) -> Self::Output;
}

impl ConvTranspose1dEvent for OpConvTranspose1d<'_> {
    type Output = ConvTranspose1dResult;

    fn dispatch(self, actor: &mut ConvTranspose1dKernel) -> Self::Output {
        actor.conv_transpose_1d(self)
    }
}

impl ConvTranspose1dEvent for UnexpectedConvTranspose1d {
    type Output = ConvTranspose1dResult;

    fn dispatch(self, actor: &mut ConvTranspose1dKernel) -> Self::Output {
        let result = Cell::new(Err(ConvTranspose1dError::UnexpectedEvent));
        actor
            .machine
            .process_event(ConvTranspose1dMachineEvents::Unexpected(
                UnexpectedRuntime { result: &result },
            ))
            .map_err(|_| ConvTranspose1dError::Internal)?;
        result.get()
    }
}

fn views_valid(
    weights: &TensorView<'_>,
    input: &TensorView<'_>,
    output: &RefCell<TensorViewMut<'_>>,
) -> bool {
    let output = output.borrow();
    weights.validate().is_ok()
        && input.validate().is_ok()
        && output.validate().is_ok()
        && output.layout().is_dense_contiguous()
}

fn parameters_valid(params: ConvTranspose1dParams) -> bool {
    params.stride > 0
        && u64::try_from(params.stride).is_ok_and(|stride| stride <= MAX_CONV_GUARD_EXTENT)
        && params.padding == 0
        && params.dilation == 1
}

const MAX_CONV_GUARD_EXTENT: u64 = 1 << 31;

fn extent_caps_valid(
    weights: &TensorView<'_>,
    input: &TensorView<'_>,
    params: ConvTranspose1dParams,
) -> bool {
    let weights_shape = weights.layout().ne();
    let input_shape = input.layout().ne();
    let Some(stride) = u64::try_from(params.stride).ok() else {
        return false;
    };
    weights_shape[0] <= MAX_CONV_GUARD_EXTENT
        && weights_shape[1] <= MAX_CONV_GUARD_EXTENT
        && weights_shape[2] <= MAX_CONV_GUARD_EXTENT
        && input_shape[0] <= MAX_CONV_GUARD_EXTENT
        && stride <= MAX_CONV_GUARD_EXTENT
}

fn output_length(
    input: &TensorView<'_>,
    weights: &TensorView<'_>,
    params: ConvTranspose1dParams,
) -> Option<u64> {
    let length = input.layout().ne()[0];
    let kernel = weights.layout().ne()[0];
    let stride = u64::try_from(params.stride).ok()?;
    length
        .checked_sub(1)
        .and_then(|value| value.checked_mul(stride))
        .and_then(|value| value.checked_add(kernel))
}

fn shape_valid(
    weights: &TensorView<'_>,
    input: &TensorView<'_>,
    output: &RefCell<TensorViewMut<'_>>,
    params: ConvTranspose1dParams,
) -> bool {
    let output = output.borrow();
    let weights_shape = weights.layout().ne();
    let input_shape = input.layout().ne();
    let output_shape = output.layout().ne();
    if !extent_caps_valid(weights, input, params) {
        return false;
    }
    let Some(expected_length) = output_length(input, weights, params) else {
        return false;
    };
    if expected_length > MAX_CONV_GUARD_EXTENT
        || expected_length
            .checked_mul(weights_shape[1])
            .is_none_or(|count| count > MAX_CONV_GUARD_EXTENT)
    {
        return false;
    }
    weights_shape[3] == 1
        && input_shape[1] == weights_shape[2]
        && input_shape[2] == 1
        && input_shape[3] == 1
        && output_shape[0] == expected_length
        && output_shape[1] == weights_shape[1]
        && output_shape[2] == 1
        && output_shape[3] == 1
}

impl ConvTranspose1dMachineStateMachineContext for Context {
    fn guard_conv_transpose_1d_views_valid<'dispatch, 'data>(
        &self,
        event: &ConvTranspose1dRuntime<'dispatch, 'data>,
    ) -> Result<bool, ()>
    where
        'data: 'dispatch,
    {
        Ok(views_valid(&event.weights, &event.input, event.output))
    }

    fn guard_conv_transpose_1d_parameters_valid<'dispatch, 'data>(
        &self,
        event: &ConvTranspose1dRuntime<'dispatch, 'data>,
    ) -> Result<bool, ()>
    where
        'data: 'dispatch,
    {
        Ok(parameters_valid(event.params))
    }

    fn guard_conv_transpose_1d_shape_valid<'dispatch, 'data>(
        &self,
        event: &ConvTranspose1dRuntime<'dispatch, 'data>,
    ) -> Result<bool, ()>
    where
        'data: 'dispatch,
    {
        Ok(shape_valid(
            &event.weights,
            &event.input,
            event.output,
            event.params,
        ))
    }

    fn effect_conv_transpose_1d_execute<'dispatch, 'data>(
        &mut self,
        event: ConvTranspose1dRuntime<'dispatch, 'data>,
    ) -> Result<(), ()>
    where
        'data: 'dispatch,
    {
        let weights_layout = event.weights.layout().ne();
        let input_layout = event.input.layout().ne();
        let mut output = event.output.borrow_mut();
        let output_length =
            usize::try_from(output.layout().ne()[0]).expect("validated output extent fits usize");
        let kernel =
            usize::try_from(weights_layout[0]).expect("validated kernel extent fits usize");
        let out_channels =
            usize::try_from(weights_layout[1]).expect("validated output-channel extent fits usize");
        let in_channels =
            usize::try_from(weights_layout[2]).expect("validated input-channel extent fits usize");
        let input_length =
            usize::try_from(input_layout[0]).expect("validated input extent fits usize");
        let stride =
            usize::try_from(event.params.stride).expect("validated positive stride fits usize");

        let mut ordinal = 0;
        while ordinal < output.count() {
            output.write(ordinal, 0.0);
            ordinal += 1;
        }

        let mut input_channel = 0;
        while input_channel < in_channels {
            let mut input_index = 0;
            while input_index < input_length {
                let input_ordinal = input_channel * input_length + input_index;
                let input_value = event.input.read(input_ordinal);
                let mut output_channel = 0;
                while output_channel < out_channels {
                    let weight_row = kernel * (output_channel + out_channels * input_channel);
                    let mut tap = 0;
                    while tap < kernel {
                        let weight = event.weights.read(weight_row + tap);
                        let output_index = input_index * stride + tap;
                        let output_ordinal = output_channel * output_length + output_index;
                        let value = output.read(output_ordinal) + input_value * weight;
                        output.write(output_ordinal, value);
                        tap += 1;
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

    fn effect_conv_transpose_1d_shape_reject<'dispatch, 'data>(
        &mut self,
        event: ConvTranspose1dRuntime<'dispatch, 'data>,
    ) -> Result<(), ()>
    where
        'data: 'dispatch,
    {
        event.result.set(Err(ConvTranspose1dError::ShapeMismatch));
        Ok(())
    }

    fn effect_conv_transpose_1d_parameters_reject<'dispatch, 'data>(
        &mut self,
        event: ConvTranspose1dRuntime<'dispatch, 'data>,
    ) -> Result<(), ()>
    where
        'data: 'dispatch,
    {
        event
            .result
            .set(Err(ConvTranspose1dError::InvalidParameters));
        Ok(())
    }

    fn effect_conv_transpose_1d_view_reject<'dispatch, 'data>(
        &mut self,
        event: ConvTranspose1dRuntime<'dispatch, 'data>,
    ) -> Result<(), ()>
    where
        'data: 'dispatch,
    {
        event.result.set(Err(ConvTranspose1dError::InvalidView));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ConvTranspose1dError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}
