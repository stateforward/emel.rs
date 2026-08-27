//! Allocation-free F32 1-D `im2col` with explicit SML dispatch.
//!
//! The pinned `emel.cpp` source is commit
//! `843a117386ef17dc5a50549bbfc821074c2141d6`. Its
//! `src/emel/kernel/detail.hpp:4858-4955` contract reads integer parameters
//! `{s0, p0, d0, is_2d}`, rejects 2-D mode for this route, computes
//! `out_length = (length + 2 * padding - dilation * (kernel - 1) - 1) /
//! stride + 1`, and writes rows of `[channel, tap]` columns. The F32
//! destination transition is `src/emel/kernel/x86_64/sm.hpp:697-699`.
//!
//! This module ports the safe 1-D route for both compile-time destination
//! variants. The pinned `src0` kernel is metadata-only for this operation, so
//! the event carries its `Layout` without requiring a backing allocation;
//! `src1` remains an F32 caller-owned view and the destination is either F32
//! or F16 caller-owned dense storage. 2-D variants remain residuals.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

use super::tensor_view::{DType, Layout, TensorView, TensorViewMut};

const MAX_CONV_GUARD_EXTENT: u64 = 1 << 31;

/// Parameters matching the pinned `op_params` integer slots `{s0, p0, d0,
/// is_2d}`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Im2ColParams {
    /// Positive input stride.
    pub stride: i32,
    /// Non-negative zero-padding width.
    pub padding: i32,
    /// Positive tap dilation.
    pub dilation: i32,
    /// The pinned 1-D route requires this flag to be zero.
    pub is_2d: i32,
}

/// Errors returned by F32 1-D `im2col` dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Im2ColError {
    /// One or more views failed dtype, shape, stride, or bounds validation.
    InvalidView,
    /// Valid views do not satisfy the pinned `im2col` dimensions.
    ShapeMismatch,
    /// Parameters are outside the maintained 1-D route.
    InvalidParameters,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for Im2ColError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid im2col tensor view"),
            Self::ShapeMismatch => formatter.write_str("im2col tensor shapes differ"),
            Self::InvalidParameters => formatter.write_str("invalid im2col parameters"),
            Self::UnexpectedEvent => formatter.write_str("unexpected im2col event"),
            Self::Internal => formatter.write_str("internal im2col dispatch error"),
        }
    }
}

impl std::error::Error for Im2ColError {}

/// Produces dense F32 columns from a one-dimensional input window.
#[derive(Debug)]
pub struct OpIm2Col<'a> {
    kernel: Layout,
    input: TensorView<'a>,
    output: TensorViewMut<'a>,
    params: Im2ColParams,
}

/// Dense F16 destination storage for the pinned `im2col_as<true>` route.
#[derive(Debug)]
pub struct F16OutputViewMut<'a> {
    data: &'a mut [u16],
    layout: Layout,
}

impl<'a> F16OutputViewMut<'a> {
    /// Creates an F16 output view. Validation occurs in machine guards.
    #[must_use]
    pub const fn new(data: &'a mut [u16], layout: Layout) -> Self {
        Self { data, layout }
    }

    /// Returns the output layout.
    #[must_use]
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    fn valid(&self) -> bool {
        let extents = self.layout.ne();
        if self.layout.dtype() != DType::F16
            || extents[0] == 0
            || extents[1] == 0
            || extents[3] != 1
        {
            return false;
        }
        let strides = self.layout.nb();
        let mut dimension = 0;
        while dimension < 4 {
            let zero_stride = strides[dimension] == 0 && extents[dimension] <= 1;
            if !zero_stride && (strides[dimension] < 2 || !strides[dimension].is_multiple_of(2)) {
                return false;
            }
            dimension += 1;
        }
        let mut max_offset = 0_u128;
        dimension = 0;
        while dimension < 4 {
            let Some(term) =
                u128::from(extents[dimension] - 1).checked_mul(u128::from(strides[dimension]))
            else {
                return false;
            };
            let Some(next) = max_offset.checked_add(term) else {
                return false;
            };
            max_offset = next;
            dimension += 1;
        }
        max_offset
            .checked_add(2)
            .is_some_and(|end| end <= (self.data.len() as u128).saturating_mul(2))
    }

    fn dense(&self) -> bool {
        if !self.valid() {
            return false;
        }
        let extents = self.layout.ne();
        let strides = self.layout.nb();
        let mut expected = 2_u64;
        let mut dimension = 0;
        while dimension < 4 {
            if strides[dimension] != expected {
                return false;
            }
            let Some(next) = expected.checked_mul(extents[dimension]) else {
                return false;
            };
            expected = next;
            dimension += 1;
        }
        true
    }

    #[inline]
    fn write(&mut self, ordinal: usize, value: f32) {
        let mut remaining = ordinal as u64;
        let mut byte_offset = 0_u64;
        let extents = self.layout.ne();
        let strides = self.layout.nb();
        let mut dimension = 0;
        while dimension < 4 {
            byte_offset += (remaining % extents[dimension]) * strides[dimension];
            remaining /= extents[dimension];
            dimension += 1;
        }
        self.data[(byte_offset / 2) as usize] = crate::any::quant::fp32_to_fp16(value);
    }
}

/// Produces dense F16 columns from a one-dimensional input window.
#[derive(Debug)]
pub struct OpIm2ColF16<'a> {
    kernel: Layout,
    input: TensorView<'a>,
    output: F16OutputViewMut<'a>,
    params: Im2ColParams,
}

impl<'a> OpIm2ColF16<'a> {
    /// Creates an F16 destination event. Validation occurs in machine guards.
    #[must_use]
    pub const fn new(
        kernel: Layout,
        input: TensorView<'a>,
        output: F16OutputViewMut<'a>,
        params: Im2ColParams,
    ) -> Self {
        Self {
            kernel,
            input,
            output,
            params,
        }
    }
}

impl<'a> OpIm2Col<'a> {
    /// Creates an event. View, shape, and parameter validation occurs in
    /// explicit machine guards.
    #[must_use]
    pub const fn new(
        kernel: Layout,
        input: TensorView<'a>,
        output: TensorViewMut<'a>,
        params: Im2ColParams,
    ) -> Self {
        Self {
            kernel,
            input,
            output,
            params,
        }
    }
}

/// Result returned after a synchronous `im2col` dispatch.
pub type Im2ColResult = Result<(), Im2ColError>;

/// Explicitly exercises the child actor's typed unexpected-event route.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnexpectedIm2Col;

/// Runtime payload used only within one synchronous machine dispatch.
#[derive(Debug)]
struct Im2ColRuntime<'a> {
    event: OpIm2Col<'a>,
    result: &'a Cell<Im2ColResult>,
}

#[derive(Debug)]
struct Im2ColF16Runtime<'a> {
    event: OpIm2ColF16<'a>,
    result: &'a Cell<Im2ColResult>,
}

#[derive(Debug)]
struct Im2ColUnexpectedRuntime<'a> {
    result: &'a Cell<Im2ColResult>,
}

#[derive(Default)]
struct Context;

sml! {
    Im2ColMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Im2Col(Im2ColRuntime<'dispatch>)
            [guard_im2col_dense_unit_valid] / effect_im2col_dense_unit_execute,
        "ready"_s <= "ready"_s + Im2Col(Im2ColRuntime<'dispatch>)
            [guard_im2col_dense_valid] / effect_im2col_dense_execute,
        "ready"_s <= "ready"_s + Im2Col(Im2ColRuntime<'dispatch>)
            [guard_im2col_valid] / effect_im2col_execute,
        "ready"_s <= "ready"_s + Im2Col(Im2ColRuntime<'dispatch>)
            [guard_im2col_shape_mismatch] / effect_im2col_shape_reject,
        "ready"_s <= "ready"_s + Im2Col(Im2ColRuntime<'dispatch>)
            [guard_im2col_parameters_invalid] / effect_im2col_parameters_reject,
        "ready"_s <= "ready"_s + Im2Col(Im2ColRuntime<'dispatch>)
            [guard_im2col_invalid_view] / effect_im2col_view_reject,

        "ready"_s <= "ready"_s + Im2ColF16(Im2ColF16Runtime<'dispatch>)
            [guard_im2col_f16_valid] / effect_im2col_f16_execute,
        "ready"_s <= "ready"_s + Im2ColF16(Im2ColF16Runtime<'dispatch>)
            [guard_im2col_f16_shape_mismatch] / effect_im2col_f16_shape_reject,
        "ready"_s <= "ready"_s + Im2ColF16(Im2ColF16Runtime<'dispatch>)
            [guard_im2col_f16_parameters_invalid] / effect_im2col_f16_parameters_reject,
        "ready"_s <= "ready"_s + Im2ColF16(Im2ColF16Runtime<'dispatch>)
            [guard_im2col_f16_invalid_view] / effect_im2col_f16_view_reject,

        "unexpected_dispatch"_s <= "ready"_s
            + Unexpected(Im2ColUnexpectedRuntime<'dispatch>) / effect_unexpected,
        "ready"_s <= "unexpected_dispatch"_s + completion<_>,
        "ready"_s <= "unexpected_dispatch"_s + unexpected_event<_> / effect_generic_unexpected,
        "generic_unexpected_dispatch"_s <= "ready"_s + unexpected_event<_>
            / effect_generic_unexpected,
        "ready"_s <= "generic_unexpected_dispatch"_s + completion<_>,
    }
}

/// Single-writer, run-to-completion actor for F32 1-D `im2col`.
pub struct Im2ColKernel {
    machine: Im2ColMachineStateMachine<Context>,
}

impl fmt::Debug for Im2ColKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Im2ColKernel")
            .finish_non_exhaustive()
    }
}

impl Default for Im2ColKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Im2ColKernel {
    /// Constructs an independent `im2col` actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: Im2ColMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed event synchronously to completion.
    ///
    /// # Errors
    ///
    /// Returns [`Im2ColError`] when a view, shape, or parameter invariant
    /// fails, or when generated dispatch cannot complete.
    ///
    /// # Panics
    ///
    /// Panics if the generated `im2col` machine does not return to `Ready`
    /// after the run-to-completion dispatch.
    pub fn process_event<E: Im2ColEvent>(&mut self, event: E) -> E::Output {
        let result = event.dispatch(self);
        assert!(
            self.machine.is(&Im2ColMachineStates::Ready),
            "im2col machine must return to ready after dispatch"
        );
        result
    }

    /// Reports whether the generated `im2col` machine is ready for another
    /// dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&Im2ColMachineStates::Ready)
    }

    fn im2col(&mut self, event: OpIm2Col<'_>) -> Im2ColResult {
        let result = Cell::new(Err(Im2ColError::UnexpectedEvent));
        self.machine
            .process_event(Im2ColMachineEvents::Im2Col(Im2ColRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| Im2ColError::Internal)?;
        result.get()
    }

    fn im2col_f16(&mut self, event: OpIm2ColF16<'_>) -> Im2ColResult {
        let result = Cell::new(Err(Im2ColError::UnexpectedEvent));
        self.machine
            .process_event(Im2ColMachineEvents::Im2ColF16(Im2ColF16Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| Im2ColError::Internal)?;
        result.get()
    }

    fn unexpected(&mut self) -> Im2ColResult {
        let result = Cell::new(Err(Im2ColError::UnexpectedEvent));
        self.machine
            .process_event(Im2ColMachineEvents::Unexpected(Im2ColUnexpectedRuntime {
                result: &result,
            }))
            .map_err(|_| Im2ColError::Internal)?;
        result.get()
    }
}

/// Typed event accepted by [`Im2ColKernel`].
pub trait Im2ColEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut Im2ColKernel) -> Self::Output;
}

impl Im2ColEvent for OpIm2Col<'_> {
    type Output = Im2ColResult;

    fn dispatch(self, actor: &mut Im2ColKernel) -> Self::Output {
        actor.im2col(self)
    }
}

impl Im2ColEvent for OpIm2ColF16<'_> {
    type Output = Im2ColResult;

    fn dispatch(self, actor: &mut Im2ColKernel) -> Self::Output {
        actor.im2col_f16(self)
    }
}

impl Im2ColEvent for UnexpectedIm2Col {
    type Output = Im2ColResult;

    fn dispatch(self, actor: &mut Im2ColKernel) -> Self::Output {
        actor.unexpected()
    }
}

fn views_valid(event: &OpIm2Col<'_>) -> bool {
    kernel_metadata_valid(event.kernel)
        && input_view_valid(&event.input)
        && output_view_valid(&event.output)
}

fn input_view_valid(view: &TensorView<'_>) -> bool {
    f32_layout_span_valid(view.layout(), view.storage_len())
}

fn output_view_valid(view: &TensorViewMut<'_>) -> bool {
    let layout = view.layout();
    f32_layout_span_valid(layout, view.storage_len())
        && layout
            .effective_f32_strides()
            .is_some_and(|strides| strides == dense_strides(layout.ne()))
}

fn f16_output_view_valid(view: &F16OutputViewMut<'_>) -> bool {
    view.dense()
}

fn f32_layout_span_valid(layout: Layout, data_len: usize) -> bool {
    if layout.dtype() != DType::F32 {
        return false;
    }
    let extents = layout.ne();
    if extents[0] == 0 || extents[1] == 0 || extents[3] != 1 {
        return false;
    }
    if extents[2] == 0 {
        return false;
    }
    let Some(strides) = layout.effective_f32_strides() else {
        return false;
    };
    if layout.nb()[0] != 0 {
        let raw_strides = layout.nb();
        let mut dimension = 0;
        while dimension < 4 {
            if raw_strides[dimension] < 4 || !raw_strides[dimension].is_multiple_of(4) {
                return false;
            }
            if extents[dimension] == 0 && raw_strides[dimension] != 0 {
                return false;
            }
            dimension += 1;
        }
    }
    let mut max_offset = 0_u128;
    let mut dimension = 0;
    while dimension < 4 {
        let Some(term) =
            u128::from(extents[dimension] - 1).checked_mul(u128::from(strides[dimension]))
        else {
            return false;
        };
        let Some(next) = max_offset.checked_add(term) else {
            return false;
        };
        max_offset = next;
        dimension += 1;
    }
    let Some(end) = max_offset.checked_add(4) else {
        return false;
    };
    end <= (data_len as u128).saturating_mul(4)
}

const fn dense_strides(extents: [u64; 4]) -> [u64; 4] {
    let mut strides = [0_u64; 4];
    strides[0] = 4;
    let mut dimension = 1;
    while dimension < 4 {
        strides[dimension] = strides[dimension - 1].saturating_mul(extents[dimension - 1]);
        dimension += 1;
    }
    strides
}

const fn kernel_metadata_valid(layout: Layout) -> bool {
    matches!(layout.dtype(), DType::F32 | DType::F16) && layout.ne()[0] > 0 && layout.ne()[1] > 0
}

const fn parameters_valid(params: Im2ColParams) -> bool {
    params.stride > 0 && params.padding >= 0 && params.dilation > 0 && params.is_2d == 0
}

fn output_length(event: &OpIm2Col<'_>) -> Option<u64> {
    let length = i128::from(event.input.layout().ne()[0]);
    let kernel = i128::from(event.kernel.ne()[0]);
    let stride = i128::from(event.params.stride);
    let padding = i128::from(event.params.padding);
    let dilation = i128::from(event.params.dilation);
    let numerator = length
        .checked_add(padding.checked_mul(2)?)?
        .checked_sub(dilation.checked_mul(kernel.checked_sub(1)?)?)?
        .checked_sub(1)?;
    u64::try_from(numerator / stride + 1).ok()
}

fn shape_valid(event: &OpIm2Col<'_>) -> bool {
    let kernel = event.kernel.ne();
    let input = event.input.layout().ne();
    let output = event.output.layout().ne();
    if kernel[0] > MAX_CONV_GUARD_EXTENT
        || kernel[1] > MAX_CONV_GUARD_EXTENT
        || input[0] > MAX_CONV_GUARD_EXTENT
        || input[2] > MAX_CONV_GUARD_EXTENT
    {
        return false;
    }
    let Some(out_length) = output_length(event) else {
        return false;
    };
    let Some(row_width) = kernel[1].checked_mul(kernel[0]) else {
        return false;
    };
    let Some(rows) = out_length.checked_mul(input[2]) else {
        return false;
    };
    let Some(output_elements) = rows.checked_mul(row_width) else {
        return false;
    };
    row_width <= MAX_CONV_GUARD_EXTENT
        && out_length <= MAX_CONV_GUARD_EXTENT
        && output_elements <= MAX_CONV_GUARD_EXTENT
        && out_length > 0
        && input[1] == kernel[1]
        && input[3] == 1
        && output[0] == row_width
        && output[1] == out_length
        && output[2] == input[2]
        && output[3] == 1
}

fn shape_valid_parts(kernel: Layout, input: Layout, output: Layout, params: Im2ColParams) -> bool {
    if kernel.ne()[0] > MAX_CONV_GUARD_EXTENT
        || kernel.ne()[1] > MAX_CONV_GUARD_EXTENT
        || input.ne()[0] > MAX_CONV_GUARD_EXTENT
        || input.ne()[2] > MAX_CONV_GUARD_EXTENT
    {
        return false;
    }
    let length = i128::from(input.ne()[0]);
    let kernel_length = i128::from(kernel.ne()[0]);
    let stride = i128::from(params.stride);
    let padding = i128::from(params.padding);
    let dilation = i128::from(params.dilation);
    let Some(numerator) = length
        .checked_add(padding.checked_mul(2).unwrap_or(i128::MIN))
        .and_then(|value| value.checked_sub(dilation.checked_mul(kernel_length.checked_sub(1)?)?))
        .and_then(|value| value.checked_sub(1))
    else {
        return false;
    };
    let Some(out_length) = u64::try_from(numerator / stride + 1).ok() else {
        return false;
    };
    let Some(row_width) = kernel.ne()[1].checked_mul(kernel.ne()[0]) else {
        return false;
    };
    let Some(rows) = out_length.checked_mul(input.ne()[2]) else {
        return false;
    };
    let Some(output_elements) = rows.checked_mul(row_width) else {
        return false;
    };
    row_width <= MAX_CONV_GUARD_EXTENT
        && out_length <= MAX_CONV_GUARD_EXTENT
        && output_elements <= MAX_CONV_GUARD_EXTENT
        && out_length > 0
        && input.ne()[1] == kernel.ne()[1]
        && input.ne()[3] == 1
        && output.ne()[0] == row_width
        && output.ne()[1] == out_length
        && output.ne()[2] == input.ne()[2]
        && output.ne()[3] == 1
}

impl Im2ColMachineStateMachineContext for Context {
    fn guard_im2col_dense_unit_valid(&self, event: &Im2ColRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event)
            && parameters_valid(event.event.params)
            && shape_valid(&event.event)
            && event.event.params.stride == 1
            && event.event.params.padding == 1
            && event.event.params.dilation == 1
            && event.event.input.is_dense_f32()
            && event.event.output.is_dense_f32())
    }

    fn guard_im2col_dense_valid(&self, event: &Im2ColRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event)
            && parameters_valid(event.event.params)
            && shape_valid(&event.event)
            && event.event.input.is_dense_f32()
            && event.event.output.is_dense_f32())
    }

    fn guard_im2col_valid(&self, event: &Im2ColRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event)
            && parameters_valid(event.event.params)
            && shape_valid(&event.event))
    }

    fn guard_im2col_shape_mismatch(&self, event: &Im2ColRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event)
            && parameters_valid(event.event.params)
            && !shape_valid(&event.event))
    }

    fn guard_im2col_parameters_invalid(&self, event: &Im2ColRuntime<'_>) -> Result<bool, ()> {
        Ok(views_valid(&event.event) && !parameters_valid(event.event.params))
    }

    fn guard_im2col_invalid_view(&self, event: &Im2ColRuntime<'_>) -> Result<bool, ()> {
        Ok(!views_valid(&event.event))
    }

    fn guard_im2col_f16_valid(&self, event: &Im2ColF16Runtime<'_>) -> Result<bool, ()> {
        Ok(kernel_metadata_valid(event.event.kernel)
            && input_view_valid(&event.event.input)
            && f16_output_view_valid(&event.event.output)
            && parameters_valid(event.event.params)
            && shape_valid_parts(
                event.event.kernel,
                event.event.input.layout(),
                event.event.output.layout(),
                event.event.params,
            ))
    }

    fn guard_im2col_f16_shape_mismatch(&self, event: &Im2ColF16Runtime<'_>) -> Result<bool, ()> {
        Ok(kernel_metadata_valid(event.event.kernel)
            && input_view_valid(&event.event.input)
            && f16_output_view_valid(&event.event.output)
            && parameters_valid(event.event.params)
            && !shape_valid_parts(
                event.event.kernel,
                event.event.input.layout(),
                event.event.output.layout(),
                event.event.params,
            ))
    }

    fn guard_im2col_f16_parameters_invalid(
        &self,
        event: &Im2ColF16Runtime<'_>,
    ) -> Result<bool, ()> {
        Ok(kernel_metadata_valid(event.event.kernel)
            && input_view_valid(&event.event.input)
            && f16_output_view_valid(&event.event.output)
            && !parameters_valid(event.event.params))
    }

    fn guard_im2col_f16_invalid_view(&self, event: &Im2ColF16Runtime<'_>) -> Result<bool, ()> {
        Ok(!(kernel_metadata_valid(event.event.kernel)
            && input_view_valid(&event.event.input)
            && f16_output_view_valid(&event.event.output)))
    }

    fn effect_im2col_execute(&mut self, mut event: Im2ColRuntime<'_>) -> Result<(), ()> {
        let kernel_layout = event.event.kernel.ne();
        let input_layout = event.event.input.layout().ne();
        let output_layout = event.event.output.layout().ne();
        let kernel = usize::try_from(kernel_layout[0]).expect("validated kernel extent fits usize");
        let channels =
            usize::try_from(kernel_layout[1]).expect("validated channel extent fits usize");
        let input_length =
            usize::try_from(input_layout[0]).expect("validated input extent fits usize");
        let batches = usize::try_from(input_layout[2]).expect("validated batch extent fits usize");
        let out_length =
            usize::try_from(output_layout[1]).expect("validated output extent fits usize");
        let stride = usize::try_from(event.event.params.stride)
            .expect("validated positive stride fits usize");
        let padding = usize::try_from(event.event.params.padding)
            .expect("validated non-negative padding fits usize");
        let dilation = usize::try_from(event.event.params.dilation)
            .expect("validated positive dilation fits usize");
        let row_width = channels * kernel;

        let mut batch = 0;
        while batch < batches {
            let mut output_row = 0;
            while output_row < out_length {
                let mut channel = 0;
                while channel < channels {
                    let mut tap = 0;
                    while tap < kernel {
                        let sampled = output_row * stride + tap * dilation;
                        let value = match sampled.checked_sub(padding) {
                            Some(input_index) if input_index < input_length => {
                                let input_ordinal =
                                    input_index + input_length * (channel + channels * batch);
                                event.event.input.read(input_ordinal)
                            }
                            _ => 0.0,
                        };
                        let output_ordinal =
                            tap + kernel * channel + row_width * (output_row + out_length * batch);
                        event.event.output.write(output_ordinal, value);
                        tap += 1;
                    }
                    channel += 1;
                }
                output_row += 1;
            }
            batch += 1;
        }

        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_im2col_dense_execute(&mut self, mut event: Im2ColRuntime<'_>) -> Result<(), ()> {
        let kernel_layout = event.event.kernel.ne();
        let input_layout = event.event.input.layout().ne();
        let output_layout = event.event.output.layout().ne();
        let kernel = usize::try_from(kernel_layout[0]).expect("validated kernel extent fits usize");
        let channels =
            usize::try_from(kernel_layout[1]).expect("validated channel extent fits usize");
        let input_length =
            usize::try_from(input_layout[0]).expect("validated input extent fits usize");
        let batches = usize::try_from(input_layout[2]).expect("validated batch extent fits usize");
        let out_length =
            usize::try_from(output_layout[1]).expect("validated output extent fits usize");
        let stride = usize::try_from(event.event.params.stride)
            .expect("validated positive stride fits usize");
        let padding = usize::try_from(event.event.params.padding)
            .expect("validated non-negative padding fits usize");
        let dilation = usize::try_from(event.event.params.dilation)
            .expect("validated positive dilation fits usize");
        let row_width = channels * kernel;
        let input = event.event.input.as_slice();
        let output = event.event.output.as_mut_slice();

        let mut batch = 0;
        while batch < batches {
            let mut output_row = 0;
            while output_row < out_length {
                let mut channel = 0;
                while channel < channels {
                    let mut tap = 0;
                    while tap < kernel {
                        let sampled = output_row * stride + tap * dilation;
                        let value = match sampled.checked_sub(padding) {
                            Some(input_index) if input_index < input_length => {
                                input[input_index + input_length * (channel + channels * batch)]
                            }
                            _ => 0.0,
                        };
                        let output_ordinal =
                            tap + kernel * channel + row_width * (output_row + out_length * batch);
                        output[output_ordinal] = value;
                        tap += 1;
                    }
                    channel += 1;
                }
                output_row += 1;
            }
            batch += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_im2col_dense_unit_execute(&mut self, mut event: Im2ColRuntime<'_>) -> Result<(), ()> {
        let kernel_layout = event.event.kernel.ne();
        let input_layout = event.event.input.layout().ne();
        let output_layout = event.event.output.layout().ne();
        let kernel = usize::try_from(kernel_layout[0]).expect("validated kernel extent fits usize");
        let channels =
            usize::try_from(kernel_layout[1]).expect("validated channel extent fits usize");
        let input_length =
            usize::try_from(input_layout[0]).expect("validated input extent fits usize");
        let batches = usize::try_from(input_layout[2]).expect("validated batch extent fits usize");
        let out_length =
            usize::try_from(output_layout[1]).expect("validated output extent fits usize");
        let row_width = channels * kernel;
        let input = event.event.input.as_slice();
        let output = event.event.output.as_mut_slice();

        let mut batch = 0;
        while batch < batches {
            let mut output_row = 0;
            while output_row < out_length {
                let input_start = i128::try_from(output_row).expect("validated index fits") - 1;
                let mut channel = 0;
                while channel < channels {
                    let input_base = input_length * (channel + channels * batch);
                    let output_base =
                        row_width * (output_row + out_length * batch) + kernel * channel;
                    if input_start < 0 {
                        output[output_base] = 0.0;
                        let copied = (kernel - 1).min(input_length);
                        output[output_base + 1..output_base + 1 + copied]
                            .copy_from_slice(&input[input_base..input_base + copied]);
                        output[output_base + 1 + copied..output_base + kernel].fill(0.0);
                    } else {
                        let start = usize::try_from(input_start).expect("validated index fits");
                        let copied = kernel.min(input_length.saturating_sub(start));
                        output[output_base..output_base + copied].copy_from_slice(
                            &input[input_base + start..input_base + start + copied],
                        );
                        output[output_base + copied..output_base + kernel].fill(0.0);
                    }
                    channel += 1;
                }
                output_row += 1;
            }
            batch += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_im2col_f16_execute(&mut self, mut event: Im2ColF16Runtime<'_>) -> Result<(), ()> {
        let kernel_layout = event.event.kernel.ne();
        let input_layout = event.event.input.layout().ne();
        let output_layout = event.event.output.layout().ne();
        let kernel = usize::try_from(kernel_layout[0]).expect("validated kernel extent fits usize");
        let channels =
            usize::try_from(kernel_layout[1]).expect("validated channel extent fits usize");
        let input_length =
            usize::try_from(input_layout[0]).expect("validated input extent fits usize");
        let batches = usize::try_from(input_layout[2]).expect("validated batch extent fits usize");
        let out_length =
            usize::try_from(output_layout[1]).expect("validated output extent fits usize");
        let stride = usize::try_from(event.event.params.stride)
            .expect("validated positive stride fits usize");
        let padding = usize::try_from(event.event.params.padding)
            .expect("validated non-negative padding fits usize");
        let dilation = usize::try_from(event.event.params.dilation)
            .expect("validated positive dilation fits usize");
        let row_width = channels * kernel;

        let mut batch = 0;
        while batch < batches {
            let mut output_row = 0;
            while output_row < out_length {
                let mut channel = 0;
                while channel < channels {
                    let mut tap = 0;
                    while tap < kernel {
                        let sampled = output_row * stride + tap * dilation;
                        let value = match sampled.checked_sub(padding) {
                            Some(input_index) if input_index < input_length => {
                                let input_ordinal =
                                    input_index + input_length * (channel + channels * batch);
                                event.event.input.read(input_ordinal)
                            }
                            _ => 0.0,
                        };
                        let output_ordinal =
                            tap + kernel * channel + row_width * (output_row + out_length * batch);
                        event.event.output.write(output_ordinal, value);
                        tap += 1;
                    }
                    channel += 1;
                }
                output_row += 1;
            }
            batch += 1;
        }

        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_im2col_shape_reject(&mut self, event: Im2ColRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Im2ColError::ShapeMismatch));
        Ok(())
    }

    fn effect_im2col_f16_shape_reject(&mut self, event: Im2ColF16Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Im2ColError::ShapeMismatch));
        Ok(())
    }

    fn effect_im2col_parameters_reject(&mut self, event: Im2ColRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Im2ColError::InvalidParameters));
        Ok(())
    }

    fn effect_im2col_f16_parameters_reject(
        &mut self,
        event: Im2ColF16Runtime<'_>,
    ) -> Result<(), ()> {
        event.result.set(Err(Im2ColError::InvalidParameters));
        Ok(())
    }

    fn effect_im2col_view_reject(&mut self, event: Im2ColRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Im2ColError::InvalidView));
        Ok(())
    }

    fn effect_im2col_f16_view_reject(&mut self, event: Im2ColF16Runtime<'_>) -> Result<(), ()> {
        event.result.set(Err(Im2ColError::InvalidView));
        Ok(())
    }

    fn effect_unexpected(&mut self, event: Im2ColUnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Im2ColError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}
