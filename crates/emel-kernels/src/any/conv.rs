//! Dense F32 convolution kernels for the remaining C++ conv routes.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_conv_2d`, `op_conv_2d_dw`, `op_conv_3d`, and `op_conv_transpose_2d`
//! and names `exec_op_conv*` routes on the arch machines. Those action/guard
//! types are not defined in the pinned headers, so this actor is
//! source-contract. The maintained F32 1-D transpose lives in
//! `conv_transpose_1d`.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by dense convolution dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConvError {
    /// Batch, channel, or spatial geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for ConvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid conv shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected conv event"),
            Self::Internal => formatter.write_str("internal conv dispatch error"),
        }
    }
}

impl std::error::Error for ConvError {}

/// Result returned by dense convolution dispatch.
pub type ConvResult = Result<(), ConvError>;

const fn conv_out(len: usize, kernel: usize, stride: usize) -> Option<usize> {
    if kernel == 0 || stride == 0 || len < kernel {
        None
    } else {
        Some((len - kernel) / stride + 1)
    }
}

const fn deconv_out(len: usize, kernel: usize, stride: usize) -> Option<usize> {
    if kernel == 0 || stride == 0 || len == 0 {
        None
    } else {
        Some((len - 1).saturating_mul(stride).saturating_add(kernel))
    }
}

/// Geometry for dense NCHW 2-D convolution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv2dParams {
    /// Batch size.
    pub n: usize,
    /// Input channels.
    pub c_in: usize,
    /// Output channels.
    pub c_out: usize,
    /// Input height.
    pub h: usize,
    /// Input width.
    pub w: usize,
    /// Kernel height.
    pub kh: usize,
    /// Kernel width.
    pub kw: usize,
    /// Vertical stride.
    pub stride_h: usize,
    /// Horizontal stride.
    pub stride_w: usize,
}

impl Conv2dParams {
    const fn out_hw(self) -> Option<(usize, usize)> {
        match (
            conv_out(self.h, self.kh, self.stride_h),
            conv_out(self.w, self.kw, self.stride_w),
        ) {
            (Some(oh), Some(ow)) => Some((oh, ow)),
            _ => None,
        }
    }

    const fn deconv_hw(self) -> Option<(usize, usize)> {
        match (
            deconv_out(self.h, self.kh, self.stride_h),
            deconv_out(self.w, self.kw, self.stride_w),
        ) {
            (Some(oh), Some(ow)) => Some((oh, ow)),
            _ => None,
        }
    }

    const fn input_len(self) -> usize {
        self.n
            .saturating_mul(self.c_in)
            .saturating_mul(self.h)
            .saturating_mul(self.w)
    }

    const fn weight_len(self) -> usize {
        self.c_out
            .saturating_mul(self.c_in)
            .saturating_mul(self.kh)
            .saturating_mul(self.kw)
    }

    const fn dw_weight_len(self) -> usize {
        self.c_in.saturating_mul(self.kh).saturating_mul(self.kw)
    }

    const fn output_len(self, oh: usize, ow: usize) -> usize {
        self.n
            .saturating_mul(self.c_out)
            .saturating_mul(oh)
            .saturating_mul(ow)
    }
}

/// Geometry for dense NCDHW 3-D convolution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv3dParams {
    /// 2-D plane geometry, reused for H/W/C.
    pub plane: Conv2dParams,
    /// Input depth.
    pub d: usize,
    /// Kernel depth.
    pub kd: usize,
    /// Depth stride.
    pub stride_d: usize,
}

/// 2-D convolution over dense NCHW F32 buffers.
#[derive(Debug)]
pub struct OpConv2d<'a> {
    input: &'a [f32],
    weights: &'a [f32],
    output: &'a mut [f32],
    params: Conv2dParams,
}

impl<'a> OpConv2d<'a> {
    /// Creates a 2-D conv request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        input: &'a [f32],
        weights: &'a [f32],
        output: &'a mut [f32],
        params: Conv2dParams,
    ) -> Self {
        Self {
            input,
            weights,
            output,
            params,
        }
    }

    const fn valid(&self) -> bool {
        match self.params.out_hw() {
            Some((oh, ow)) => {
                let input_len = self.params.input_len();
                let weight_len = self.params.weight_len();
                let output_len = self.params.output_len(oh, ow);
                self.params.n > 0
                    && self.params.c_in > 0
                    && self.params.c_out > 0
                    && self.input.len() == input_len
                    && self.weights.len() == weight_len
                    && self.output.len() == output_len
            }
            None => false,
        }
    }
}

/// Depthwise 2-D convolution over dense NCHW F32 buffers.
#[derive(Debug)]
pub struct OpConv2dDw<'a> {
    input: &'a [f32],
    weights: &'a [f32],
    output: &'a mut [f32],
    params: Conv2dParams,
}

impl<'a> OpConv2dDw<'a> {
    /// Creates a depthwise 2-D conv request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        input: &'a [f32],
        weights: &'a [f32],
        output: &'a mut [f32],
        params: Conv2dParams,
    ) -> Self {
        Self {
            input,
            weights,
            output,
            params,
        }
    }

    const fn valid(&self) -> bool {
        match self.params.out_hw() {
            Some((oh, ow)) => {
                let input_len = self.params.input_len();
                let weight_len = self.params.dw_weight_len();
                let output_len = self.params.output_len(oh, ow);
                self.params.n > 0
                    && self.params.c_in > 0
                    && self.params.c_out == self.params.c_in
                    && self.input.len() == input_len
                    && self.weights.len() == weight_len
                    && self.output.len() == output_len
            }
            None => false,
        }
    }
}

/// 3-D convolution over dense NCDHW F32 buffers.
#[derive(Debug)]
pub struct OpConv3d<'a> {
    input: &'a [f32],
    weights: &'a [f32],
    output: &'a mut [f32],
    params: Conv3dParams,
}

impl<'a> OpConv3d<'a> {
    /// Creates a 3-D conv request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        input: &'a [f32],
        weights: &'a [f32],
        output: &'a mut [f32],
        params: Conv3dParams,
    ) -> Self {
        Self {
            input,
            weights,
            output,
            params,
        }
    }

    const fn valid(&self) -> bool {
        let plane = self.params.plane;
        let Some((oh, ow)) = plane.out_hw() else {
            return false;
        };
        let Some(od) = conv_out(self.params.d, self.params.kd, self.params.stride_d) else {
            return false;
        };
        let weight_len = plane.weight_len().saturating_mul(self.params.kd);
        let output_len = plane.output_len(oh, ow).saturating_mul(od);
        plane.n > 0
            && plane.c_in > 0
            && plane.c_out > 0
            && self.params.d > 0
            && self.input.len()
                == plane
                    .n
                    .saturating_mul(plane.c_in)
                    .saturating_mul(self.params.d)
                    .saturating_mul(plane.h)
                    .saturating_mul(plane.w)
            && self.weights.len() == weight_len
            && self.output.len() == output_len
    }
}

/// 2-D transposed convolution over dense NCHW F32 buffers.
#[derive(Debug)]
pub struct OpConvTranspose2d<'a> {
    input: &'a [f32],
    weights: &'a [f32],
    output: &'a mut [f32],
    params: Conv2dParams,
}

impl<'a> OpConvTranspose2d<'a> {
    /// Creates a 2-D transpose-conv request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        input: &'a [f32],
        weights: &'a [f32],
        output: &'a mut [f32],
        params: Conv2dParams,
    ) -> Self {
        Self {
            input,
            weights,
            output,
            params,
        }
    }

    const fn valid(&self) -> bool {
        match self.params.deconv_hw() {
            Some((oh, ow)) => {
                let input_len = self.params.input_len();
                let weight_len = self.params.weight_len();
                let output_len = self.params.output_len(oh, ow);
                self.params.n > 0
                    && self.params.c_in > 0
                    && self.params.c_out > 0
                    && self.input.len() == input_len
                    && self.weights.len() == weight_len
                    && self.output.len() == output_len
            }
            None => false,
        }
    }
}

struct Conv2dRuntime<'a> {
    event: OpConv2d<'a>,
    result: &'a Cell<ConvResult>,
}

struct Conv2dDwRuntime<'a> {
    event: OpConv2dDw<'a>,
    result: &'a Cell<ConvResult>,
}

struct Conv3dRuntime<'a> {
    event: OpConv3d<'a>,
    result: &'a Cell<ConvResult>,
}

struct ConvTranspose2dRuntime<'a> {
    event: OpConvTranspose2d<'a>,
    result: &'a Cell<ConvResult>,
}

#[derive(Default)]
struct Context;

sml! {
    ConvMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Conv2d(Conv2dRuntime<'dispatch>) [guard_conv2d_valid] / effect_conv2d,
        "ready"_s <= "ready"_s + Conv2d(Conv2dRuntime<'dispatch>) [guard_conv2d_invalid] / effect_conv2d_reject,
        "ready"_s <= "ready"_s + Conv2dDw(Conv2dDwRuntime<'dispatch>) [guard_dw_valid] / effect_dw,
        "ready"_s <= "ready"_s + Conv2dDw(Conv2dDwRuntime<'dispatch>) [guard_dw_invalid] / effect_dw_reject,
        "ready"_s <= "ready"_s + Conv3d(Conv3dRuntime<'dispatch>) [guard_conv3d_valid] / effect_conv3d,
        "ready"_s <= "ready"_s + Conv3d(Conv3dRuntime<'dispatch>) [guard_conv3d_invalid] / effect_conv3d_reject,
        "ready"_s <= "ready"_s + ConvTranspose2d(ConvTranspose2dRuntime<'dispatch>) [guard_transpose_valid] / effect_transpose,
        "ready"_s <= "ready"_s + ConvTranspose2d(ConvTranspose2dRuntime<'dispatch>) [guard_transpose_invalid] / effect_transpose_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for dense convolution events.
pub trait ConvEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut ConvKernel) -> Self::Output;
}

/// Single-writer dense convolution actor.
pub struct ConvKernel {
    machine: ConvMachineStateMachine<Context>,
}

impl fmt::Debug for ConvKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("ConvKernel").finish_non_exhaustive()
    }
}

impl Default for ConvKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvKernel {
    /// Constructs an independent convolution actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: ConvMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed convolution event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`ConvError::InvalidShape`] when the dense buffers cannot form
    /// the requested convolution geometry.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: ConvEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn conv2d(&mut self, event: OpConv2d<'_>) -> ConvResult {
        let result = Cell::new(Err(ConvError::UnexpectedEvent));
        self.machine
            .process_event(ConvMachineEvents::Conv2d(Conv2dRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ConvError::Internal)?;
        assert!(
            self.machine.is(&ConvMachineStates::Ready),
            "conv machine must return to ready after dispatch"
        );
        result.get()
    }

    fn conv2d_dw(&mut self, event: OpConv2dDw<'_>) -> ConvResult {
        let result = Cell::new(Err(ConvError::UnexpectedEvent));
        self.machine
            .process_event(ConvMachineEvents::Conv2dDw(Conv2dDwRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ConvError::Internal)?;
        assert!(
            self.machine.is(&ConvMachineStates::Ready),
            "conv machine must return to ready after dispatch"
        );
        result.get()
    }

    fn conv3d(&mut self, event: OpConv3d<'_>) -> ConvResult {
        let result = Cell::new(Err(ConvError::UnexpectedEvent));
        self.machine
            .process_event(ConvMachineEvents::Conv3d(Conv3dRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ConvError::Internal)?;
        assert!(
            self.machine.is(&ConvMachineStates::Ready),
            "conv machine must return to ready after dispatch"
        );
        result.get()
    }

    fn conv_transpose_2d(&mut self, event: OpConvTranspose2d<'_>) -> ConvResult {
        let result = Cell::new(Err(ConvError::UnexpectedEvent));
        self.machine
            .process_event(ConvMachineEvents::ConvTranspose2d(ConvTranspose2dRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| ConvError::Internal)?;
        assert!(
            self.machine.is(&ConvMachineStates::Ready),
            "conv machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&ConvMachineStates::Ready)
    }
}

impl ConvEvent for OpConv2d<'_> {
    type Output = ConvResult;
    fn dispatch(self, kernel: &mut ConvKernel) -> Self::Output {
        kernel.conv2d(self)
    }
}

impl ConvEvent for OpConv2dDw<'_> {
    type Output = ConvResult;
    fn dispatch(self, kernel: &mut ConvKernel) -> Self::Output {
        kernel.conv2d_dw(self)
    }
}

impl ConvEvent for OpConv3d<'_> {
    type Output = ConvResult;
    fn dispatch(self, kernel: &mut ConvKernel) -> Self::Output {
        kernel.conv3d(self)
    }
}

impl ConvEvent for OpConvTranspose2d<'_> {
    type Output = ConvResult;
    fn dispatch(self, kernel: &mut ConvKernel) -> Self::Output {
        kernel.conv_transpose_2d(self)
    }
}

const fn nchw(
    batch: usize,
    channel: usize,
    row: usize,
    col: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> usize {
    ((batch * channels + channel) * height + row) * width + col
}

fn conv2d_values(event: &mut OpConv2d<'_>) {
    let p = event.params;
    let (oh, ow) = p.out_hw().unwrap_or((0, 0));
    for n in 0..p.n {
        for oc in 0..p.c_out {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0.0_f32;
                    for ic in 0..p.c_in {
                        for ky in 0..p.kh {
                            for kx in 0..p.kw {
                                let iy = oy * p.stride_h + ky;
                                let ix = ox * p.stride_w + kx;
                                let input = event.input[nchw(n, ic, iy, ix, p.c_in, p.h, p.w)];
                                let weight =
                                    event.weights[nchw(oc, ic, ky, kx, p.c_in, p.kh, p.kw)];
                                acc = input.mul_add(weight, acc);
                            }
                        }
                    }
                    event.output[nchw(n, oc, oy, ox, p.c_out, oh, ow)] = acc;
                }
            }
        }
    }
}

fn conv2d_dw_values(event: &mut OpConv2dDw<'_>) {
    let p = event.params;
    let (oh, ow) = p.out_hw().unwrap_or((0, 0));
    for n in 0..p.n {
        for c in 0..p.c_in {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0.0_f32;
                    for ky in 0..p.kh {
                        for kx in 0..p.kw {
                            let iy = oy * p.stride_h + ky;
                            let ix = ox * p.stride_w + kx;
                            let input = event.input[nchw(n, c, iy, ix, p.c_in, p.h, p.w)];
                            let weight = event.weights[(c * p.kh + ky) * p.kw + kx];
                            acc = input.mul_add(weight, acc);
                        }
                    }
                    event.output[nchw(n, c, oy, ox, p.c_out, oh, ow)] = acc;
                }
            }
        }
    }
}

fn conv3d_values(event: &mut OpConv3d<'_>) {
    let p = event.params.plane;
    let d = event.params.d;
    let kd = event.params.kd;
    let sd = event.params.stride_d;
    let (oh, ow) = p.out_hw().unwrap_or((0, 0));
    let od = conv_out(d, kd, sd).unwrap_or(0);
    for n in 0..p.n {
        for oc in 0..p.c_out {
            for oz in 0..od {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = 0.0_f32;
                        for ic in 0..p.c_in {
                            for kz in 0..kd {
                                for ky in 0..p.kh {
                                    for kx in 0..p.kw {
                                        let iz = oz * sd + kz;
                                        let iy = oy * p.stride_h + ky;
                                        let ix = ox * p.stride_w + kx;
                                        let input = event.input
                                            [(((n * p.c_in + ic) * d + iz) * p.h + iy) * p.w + ix];
                                        let weight =
                                            event.weights[(((oc * p.c_in + ic) * kd + kz) * p.kh
                                                + ky)
                                                * p.kw
                                                + kx];
                                        acc = input.mul_add(weight, acc);
                                    }
                                }
                            }
                        }
                        event.output[(((n * p.c_out + oc) * od + oz) * oh + oy) * ow + ox] = acc;
                    }
                }
            }
        }
    }
}

fn conv_transpose_2d_values(event: &mut OpConvTranspose2d<'_>) {
    let p = event.params;
    let (oh, ow) = p.deconv_hw().unwrap_or((0, 0));
    event.output.fill(0.0_f32);
    for n in 0..p.n {
        for ic in 0..p.c_in {
            for iy in 0..p.h {
                for ix in 0..p.w {
                    let sample = event.input[nchw(n, ic, iy, ix, p.c_in, p.h, p.w)];
                    for oc in 0..p.c_out {
                        for ky in 0..p.kh {
                            for kx in 0..p.kw {
                                let oy = iy * p.stride_h + ky;
                                let ox = ix * p.stride_w + kx;
                                let weight =
                                    event.weights[nchw(ic, oc, ky, kx, p.c_out, p.kh, p.kw)];
                                let index = nchw(n, oc, oy, ox, p.c_out, oh, ow);
                                event.output[index] = sample.mul_add(weight, event.output[index]);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl ConvMachineStateMachineContext for Context {
    fn guard_conv2d_valid(&self, event: &Conv2dRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_conv2d_invalid(&self, event: &Conv2dRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_dw_valid(&self, event: &Conv2dDwRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_dw_invalid(&self, event: &Conv2dDwRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_conv3d_valid(&self, event: &Conv3dRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_conv3d_invalid(&self, event: &Conv3dRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_transpose_valid(&self, event: &ConvTranspose2dRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_transpose_invalid(&self, event: &ConvTranspose2dRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn effect_conv2d(&mut self, mut event: Conv2dRuntime<'_>) -> Result<(), ()> {
        conv2d_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_conv2d_reject(&mut self, event: Conv2dRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ConvError::InvalidShape));
        Ok(())
    }

    fn effect_dw(&mut self, mut event: Conv2dDwRuntime<'_>) -> Result<(), ()> {
        conv2d_dw_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_dw_reject(&mut self, event: Conv2dDwRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ConvError::InvalidShape));
        Ok(())
    }

    fn effect_conv3d(&mut self, mut event: Conv3dRuntime<'_>) -> Result<(), ()> {
        conv3d_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_conv3d_reject(&mut self, event: Conv3dRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ConvError::InvalidShape));
        Ok(())
    }

    fn effect_transpose(&mut self, mut event: ConvTranspose2dRuntime<'_>) -> Result<(), ()> {
        conv_transpose_2d_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_transpose_reject(&mut self, event: ConvTranspose2dRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(ConvError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Conv2dParams, Conv3dParams, ConvError, ConvKernel, OpConv2d, OpConv2dDw, OpConv3d,
        OpConvTranspose2d,
    };
    use crate::Kernel;

    fn unit_plane() -> Conv2dParams {
        Conv2dParams {
            n: 1,
            c_in: 1,
            c_out: 1,
            h: 2,
            w: 2,
            kh: 1,
            kw: 1,
            stride_h: 1,
            stride_w: 1,
        }
    }

    #[test]
    fn conv2d_1x1_scales_every_sample() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weights = [2.0_f32];
        let mut output = [0.0_f32; 4];
        let mut kernel = ConvKernel::new();
        kernel
            .process_event(OpConv2d::new(&input, &weights, &mut output, unit_plane()))
            .unwrap();
        assert_eq!(output[0].to_bits(), 2.0_f32.to_bits());
        assert_eq!(output[3].to_bits(), 8.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn depthwise_matches_pointwise_when_kernel_is_one() {
        let input = [1.0_f32, -2.0, 0.5, 4.0];
        let weights = [3.0_f32];
        let mut output = [0.0_f32; 4];
        let mut kernel = ConvKernel::new();
        kernel
            .process_event(OpConv2dDw::new(&input, &weights, &mut output, unit_plane()))
            .unwrap();
        assert_eq!(output[1].to_bits(), (-6.0_f32).to_bits());
    }

    #[test]
    fn conv3d_1x1x1_scales_volume() {
        let input = [1.0_f32, 2.0];
        let weights = [4.0_f32];
        let mut output = [0.0_f32; 2];
        let params = Conv3dParams {
            plane: Conv2dParams {
                n: 1,
                c_in: 1,
                c_out: 1,
                h: 1,
                w: 1,
                kh: 1,
                kw: 1,
                stride_h: 1,
                stride_w: 1,
            },
            d: 2,
            kd: 1,
            stride_d: 1,
        };
        let mut kernel = ConvKernel::new();
        kernel
            .process_event(OpConv3d::new(&input, &weights, &mut output, params))
            .unwrap();
        assert_eq!(output[0].to_bits(), 4.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 8.0_f32.to_bits());
    }

    #[test]
    fn transpose_2d_scatters_with_stride() {
        let input = [1.0_f32];
        let weights = [5.0_f32];
        let mut output = [0.0_f32; 1];
        let params = Conv2dParams {
            n: 1,
            c_in: 1,
            c_out: 1,
            h: 1,
            w: 1,
            kh: 1,
            kw: 1,
            stride_h: 2,
            stride_w: 2,
        };
        let mut kernel = ConvKernel::new();
        kernel
            .process_event(OpConvTranspose2d::new(
                &input,
                &weights,
                &mut output,
                params,
            ))
            .unwrap();
        assert_eq!(output[0].to_bits(), 5.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_conv2d() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weights = [2.0_f32];
        let mut output = [0.0_f32; 4];
        let mut kernel = Kernel::new();
        kernel
            .process_event(OpConv2d::new(&input, &weights, &mut output, unit_plane()))
            .unwrap();
        assert_eq!(output[0].to_bits(), 2.0_f32.to_bits());
        assert_eq!(output[3].to_bits(), 8.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn conv2d_rejects_without_mutation() {
        let input = [1.0_f32];
        let weights = [1.0_f32];
        let mut output = [7.0_f32; 1];
        let mut kernel = ConvKernel::new();
        assert_eq!(
            kernel.process_event(OpConv2d::new(&input, &weights, &mut output, unit_plane())),
            Err(ConvError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 7.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_remaining_conv_family() {
        let input = [1.0_f32, -2.0, 0.5, 4.0];
        let weights = [3.0_f32];
        let mut output = [0.0_f32; 4];
        let mut kernel = Kernel::new();
        kernel
            .process_event(OpConv2dDw::new(&input, &weights, &mut output, unit_plane()))
            .unwrap();
        assert_eq!(output[1].to_bits(), (-6.0_f32).to_bits());
        let volume = [1.0_f32, 2.0];
        let mut scaled = [0.0_f32; 2];
        kernel
            .process_event(OpConv3d::new(
                &volume,
                &[4.0_f32],
                &mut scaled,
                Conv3dParams {
                    plane: Conv2dParams {
                        n: 1,
                        c_in: 1,
                        c_out: 1,
                        h: 1,
                        w: 1,
                        kh: 1,
                        kw: 1,
                        stride_h: 1,
                        stride_w: 1,
                    },
                    d: 2,
                    kd: 1,
                    stride_d: 1,
                },
            ))
            .unwrap();
        assert_eq!(scaled[1].to_bits(), 8.0_f32.to_bits());
        let mut scattered = [0.0_f32; 1];
        kernel
            .process_event(OpConvTranspose2d::new(
                &[1.0_f32],
                &[5.0_f32],
                &mut scattered,
                Conv2dParams {
                    n: 1,
                    c_in: 1,
                    c_out: 1,
                    h: 1,
                    w: 1,
                    kh: 1,
                    kw: 1,
                    stride_h: 2,
                    stride_w: 2,
                },
            ))
            .unwrap();
        assert_eq!(scattered[0].to_bits(), 5.0_f32.to_bits());
    }
}
