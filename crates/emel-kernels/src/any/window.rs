//! Dense windowing and unfold kernels for remaining C++ routes.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_win_part`, `op_win_unpart`, `op_im2col_back`, and `op_im2col_3d`
//! and names `exec_op_*` routes on the arch machines. Those action/guard
//! types are not defined in the pinned headers, so this actor is
//! source-contract. The maintained 1-D `im2col` lives in `im2col`.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by window/unfold dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowError {
    /// Spatial, channel, or window geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for WindowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid window shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected window event"),
            Self::Internal => formatter.write_str("internal window dispatch error"),
        }
    }
}

impl std::error::Error for WindowError {}

/// Result returned by window/unfold dispatch.
pub type WindowResult = Result<(), WindowError>;

const fn conv_out(len: usize, kernel: usize, stride: usize) -> Option<usize> {
    if kernel == 0 || stride == 0 || len < kernel {
        None
    } else {
        Some((len - kernel) / stride + 1)
    }
}

const fn window_grid(len: usize, window: usize) -> Option<(usize, usize)> {
    if window == 0 || len == 0 {
        None
    } else {
        let remainder = len % window;
        let pad = if remainder == 0 {
            0
        } else {
            window - remainder
        };
        Some((pad, (len + pad) / window))
    }
}

/// Geometry for a dense CHW window partition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WinPartParams {
    /// Channel count.
    pub channels: usize,
    /// Source height.
    pub height: usize,
    /// Source width.
    pub width: usize,
    /// Square window edge.
    pub window: usize,
}

impl WinPartParams {
    const fn grid(self) -> Option<(usize, usize, usize, usize)> {
        match (
            window_grid(self.height, self.window),
            window_grid(self.width, self.window),
        ) {
            (Some((pad_h, grid_h)), Some((pad_w, grid_w))) => Some((pad_h, pad_w, grid_h, grid_w)),
            _ => None,
        }
    }

    const fn source_len(self) -> usize {
        self.channels
            .saturating_mul(self.height)
            .saturating_mul(self.width)
    }

    const fn windows_len(self, grid_h: usize, grid_w: usize) -> usize {
        self.channels
            .saturating_mul(self.window)
            .saturating_mul(self.window)
            .saturating_mul(grid_h)
            .saturating_mul(grid_w)
    }
}

/// Geometry for dense 2-D `im2col` scatter-add.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Im2ColBackParams {
    /// Channel count.
    pub channels: usize,
    /// Image height.
    pub height: usize,
    /// Image width.
    pub width: usize,
    /// Kernel height.
    pub kh: usize,
    /// Kernel width.
    pub kw: usize,
    /// Vertical stride.
    pub stride_h: usize,
    /// Horizontal stride.
    pub stride_w: usize,
}

impl Im2ColBackParams {
    const fn out_hw(self) -> Option<(usize, usize)> {
        match (
            conv_out(self.height, self.kh, self.stride_h),
            conv_out(self.width, self.kw, self.stride_w),
        ) {
            (Some(out_h), Some(out_w)) => Some((out_h, out_w)),
            _ => None,
        }
    }
}

/// Geometry for dense 3-D `im2col`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Im2Col3dParams {
    /// 2-D plane geometry.
    pub plane: Im2ColBackParams,
    /// Image depth.
    pub depth: usize,
    /// Kernel depth.
    pub kd: usize,
    /// Depth stride.
    pub stride_d: usize,
}

/// Pads and tiles a dense CHW plane into square windows.
#[derive(Debug)]
pub struct OpWinPart<'a> {
    source: &'a [f32],
    windows: &'a mut [f32],
    params: WinPartParams,
}

impl<'a> OpWinPart<'a> {
    /// Creates a win-part request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(source: &'a [f32], windows: &'a mut [f32], params: WinPartParams) -> Self {
        Self {
            source,
            windows,
            params,
        }
    }

    const fn valid(&self) -> bool {
        match self.params.grid() {
            Some((_, _, grid_h, grid_w)) => {
                let source_len = self.params.source_len();
                let windows_len = self.params.windows_len(grid_h, grid_w);
                self.params.channels > 0
                    && self.source.len() == source_len
                    && self.windows.len() == windows_len
            }
            None => false,
        }
    }
}

/// Reconstructs a dense CHW plane from square windows.
#[derive(Debug)]
pub struct OpWinUnpart<'a> {
    windows: &'a [f32],
    destination: &'a mut [f32],
    params: WinPartParams,
}

impl<'a> OpWinUnpart<'a> {
    /// Creates a win-unpart request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        windows: &'a [f32],
        destination: &'a mut [f32],
        params: WinPartParams,
    ) -> Self {
        Self {
            windows,
            destination,
            params,
        }
    }

    const fn valid(&self) -> bool {
        match self.params.grid() {
            Some((_, _, grid_h, grid_w)) => {
                let source_len = self.params.source_len();
                let windows_len = self.params.windows_len(grid_h, grid_w);
                self.params.channels > 0
                    && self.windows.len() == windows_len
                    && self.destination.len() == source_len
            }
            None => false,
        }
    }
}

/// Scatter-adds 2-D `im2col` patches back into a dense CHW image.
#[derive(Debug)]
pub struct OpIm2ColBack<'a> {
    patches: &'a [f32],
    image: &'a mut [f32],
    params: Im2ColBackParams,
}

impl<'a> OpIm2ColBack<'a> {
    /// Creates an im2col-backward request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(patches: &'a [f32], image: &'a mut [f32], params: Im2ColBackParams) -> Self {
        Self {
            patches,
            image,
            params,
        }
    }

    const fn valid(&self) -> bool {
        match self.params.out_hw() {
            Some((out_h, out_w)) => {
                let image_len = self
                    .params
                    .channels
                    .saturating_mul(self.params.height)
                    .saturating_mul(self.params.width);
                let patch_len = out_h
                    .saturating_mul(out_w)
                    .saturating_mul(self.params.channels)
                    .saturating_mul(self.params.kh)
                    .saturating_mul(self.params.kw);
                self.params.channels > 0
                    && self.patches.len() == patch_len
                    && self.image.len() == image_len
            }
            None => false,
        }
    }
}

/// Unfolds a dense CDHW volume into 3-D `im2col` patches.
#[derive(Debug)]
pub struct OpIm2Col3d<'a> {
    volume: &'a [f32],
    patches: &'a mut [f32],
    params: Im2Col3dParams,
}

impl<'a> OpIm2Col3d<'a> {
    /// Creates a 3-D im2col request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(volume: &'a [f32], patches: &'a mut [f32], params: Im2Col3dParams) -> Self {
        Self {
            volume,
            patches,
            params,
        }
    }

    const fn valid(&self) -> bool {
        let plane = self.params.plane;
        let Some((out_h, out_w)) = plane.out_hw() else {
            return false;
        };
        let Some(out_d) = conv_out(self.params.depth, self.params.kd, self.params.stride_d) else {
            return false;
        };
        let volume_len = plane
            .channels
            .saturating_mul(self.params.depth)
            .saturating_mul(plane.height)
            .saturating_mul(plane.width);
        let patch_len = out_d
            .saturating_mul(out_h)
            .saturating_mul(out_w)
            .saturating_mul(plane.channels)
            .saturating_mul(self.params.kd)
            .saturating_mul(plane.kh)
            .saturating_mul(plane.kw);
        plane.channels > 0
            && self.params.depth > 0
            && self.volume.len() == volume_len
            && self.patches.len() == patch_len
    }
}

struct WinPartRuntime<'a> {
    event: OpWinPart<'a>,
    result: &'a Cell<WindowResult>,
}

struct WinUnpartRuntime<'a> {
    event: OpWinUnpart<'a>,
    result: &'a Cell<WindowResult>,
}

struct Im2ColBackRuntime<'a> {
    event: OpIm2ColBack<'a>,
    result: &'a Cell<WindowResult>,
}

struct Im2Col3dRuntime<'a> {
    event: OpIm2Col3d<'a>,
    result: &'a Cell<WindowResult>,
}

#[derive(Default)]
struct Context;

sml! {
    WindowMachine<'dispatch> {
        "ready"_s <= *"ready"_s + WinPart(WinPartRuntime<'dispatch>) [guard_part_valid] / effect_part,
        "ready"_s <= "ready"_s + WinPart(WinPartRuntime<'dispatch>) [guard_part_invalid] / effect_part_reject,
        "ready"_s <= "ready"_s + WinUnpart(WinUnpartRuntime<'dispatch>) [guard_unpart_valid] / effect_unpart,
        "ready"_s <= "ready"_s + WinUnpart(WinUnpartRuntime<'dispatch>) [guard_unpart_invalid] / effect_unpart_reject,
        "ready"_s <= "ready"_s + Im2ColBack(Im2ColBackRuntime<'dispatch>) [guard_back_valid] / effect_back,
        "ready"_s <= "ready"_s + Im2ColBack(Im2ColBackRuntime<'dispatch>) [guard_back_invalid] / effect_back_reject,
        "ready"_s <= "ready"_s + Im2Col3d(Im2Col3dRuntime<'dispatch>) [guard_col3d_valid] / effect_col3d,
        "ready"_s <= "ready"_s + Im2Col3d(Im2Col3dRuntime<'dispatch>) [guard_col3d_invalid] / effect_col3d_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for window/unfold events.
pub trait WindowEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut WindowKernel) -> Self::Output;
}

/// Single-writer window/unfold actor.
pub struct WindowKernel {
    machine: WindowMachineStateMachine<Context>,
}

impl fmt::Debug for WindowKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WindowKernel")
            .finish_non_exhaustive()
    }
}

impl Default for WindowKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowKernel {
    /// Constructs an independent window actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: WindowMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed window/unfold event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`WindowError::InvalidShape`] when the dense buffers cannot
    /// form the requested window or unfold geometry.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: WindowEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn part(&mut self, event: OpWinPart<'_>) -> WindowResult {
        let result = Cell::new(Err(WindowError::UnexpectedEvent));
        self.machine
            .process_event(WindowMachineEvents::WinPart(WinPartRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| WindowError::Internal)?;
        assert!(
            self.machine.is(&WindowMachineStates::Ready),
            "window machine must return to ready after dispatch"
        );
        result.get()
    }

    fn unpart(&mut self, event: OpWinUnpart<'_>) -> WindowResult {
        let result = Cell::new(Err(WindowError::UnexpectedEvent));
        self.machine
            .process_event(WindowMachineEvents::WinUnpart(WinUnpartRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| WindowError::Internal)?;
        assert!(
            self.machine.is(&WindowMachineStates::Ready),
            "window machine must return to ready after dispatch"
        );
        result.get()
    }

    fn back(&mut self, event: OpIm2ColBack<'_>) -> WindowResult {
        let result = Cell::new(Err(WindowError::UnexpectedEvent));
        self.machine
            .process_event(WindowMachineEvents::Im2ColBack(Im2ColBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| WindowError::Internal)?;
        assert!(
            self.machine.is(&WindowMachineStates::Ready),
            "window machine must return to ready after dispatch"
        );
        result.get()
    }

    fn col3d(&mut self, event: OpIm2Col3d<'_>) -> WindowResult {
        let result = Cell::new(Err(WindowError::UnexpectedEvent));
        self.machine
            .process_event(WindowMachineEvents::Im2Col3d(Im2Col3dRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| WindowError::Internal)?;
        assert!(
            self.machine.is(&WindowMachineStates::Ready),
            "window machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&WindowMachineStates::Ready)
    }
}

impl WindowEvent for OpWinPart<'_> {
    type Output = WindowResult;
    fn dispatch(self, kernel: &mut WindowKernel) -> Self::Output {
        kernel.part(self)
    }
}

impl WindowEvent for OpWinUnpart<'_> {
    type Output = WindowResult;
    fn dispatch(self, kernel: &mut WindowKernel) -> Self::Output {
        kernel.unpart(self)
    }
}

impl WindowEvent for OpIm2ColBack<'_> {
    type Output = WindowResult;
    fn dispatch(self, kernel: &mut WindowKernel) -> Self::Output {
        kernel.back(self)
    }
}

impl WindowEvent for OpIm2Col3d<'_> {
    type Output = WindowResult;
    fn dispatch(self, kernel: &mut WindowKernel) -> Self::Output {
        kernel.col3d(self)
    }
}

const fn chw(channel: usize, row: usize, col: usize, height: usize, width: usize) -> usize {
    (channel * height + row) * width + col
}

fn win_part_values(event: &mut OpWinPart<'_>) {
    let params = event.params;
    let Some((_, _, grid_h, grid_w)) = params.grid() else {
        return;
    };
    event.windows.fill(0.0_f32);
    let window = params.window;
    for channel in 0..params.channels {
        for grid_row in 0..grid_h {
            for grid_col in 0..grid_w {
                let window_index =
                    ((channel * grid_h + grid_row) * grid_w + grid_col) * window * window;
                for local_row in 0..window {
                    let source_row = grid_row * window + local_row;
                    for local_col in 0..window {
                        let source_col = grid_col * window + local_col;
                        if source_row < params.height && source_col < params.width {
                            event.windows[window_index + local_row * window + local_col] = event
                                .source
                                [chw(channel, source_row, source_col, params.height, params.width)];
                        }
                    }
                }
            }
        }
    }
}

fn win_unpart_values(event: &mut OpWinUnpart<'_>) {
    let params = event.params;
    let Some((_, _, grid_h, grid_w)) = params.grid() else {
        return;
    };
    let window = params.window;
    for channel in 0..params.channels {
        for grid_row in 0..grid_h {
            for grid_col in 0..grid_w {
                let window_index =
                    ((channel * grid_h + grid_row) * grid_w + grid_col) * window * window;
                for local_row in 0..window {
                    let dest_row = grid_row * window + local_row;
                    for local_col in 0..window {
                        let dest_col = grid_col * window + local_col;
                        if dest_row < params.height && dest_col < params.width {
                            event.destination
                                [chw(channel, dest_row, dest_col, params.height, params.width)] =
                                event.windows[window_index + local_row * window + local_col];
                        }
                    }
                }
            }
        }
    }
}

fn im2col_back_values(event: &mut OpIm2ColBack<'_>) {
    let params = event.params;
    let Some((out_h, out_w)) = params.out_hw() else {
        return;
    };
    event.image.fill(0.0_f32);
    let kernel = params.kh.saturating_mul(params.kw);
    for out_row in 0..out_h {
        for out_col in 0..out_w {
            let patch = (out_row * out_w + out_col) * params.channels * kernel;
            for channel in 0..params.channels {
                for kernel_row in 0..params.kh {
                    for kernel_col in 0..params.kw {
                        let image_row = out_row * params.stride_h + kernel_row;
                        let image_col = out_col * params.stride_w + kernel_col;
                        let patch_index =
                            patch + (channel * params.kh + kernel_row) * params.kw + kernel_col;
                        let image_index =
                            chw(channel, image_row, image_col, params.height, params.width);
                        event.image[image_index] += event.patches[patch_index];
                    }
                }
            }
        }
    }
}

fn im2col_3d_values(event: &mut OpIm2Col3d<'_>) {
    let plane = event.params.plane;
    let Some((out_h, out_w)) = plane.out_hw() else {
        return;
    };
    let Some(out_d) = conv_out(event.params.depth, event.params.kd, event.params.stride_d) else {
        return;
    };
    let kernel = event
        .params
        .kd
        .saturating_mul(plane.kh)
        .saturating_mul(plane.kw);
    for out_depth in 0..out_d {
        for out_row in 0..out_h {
            for out_col in 0..out_w {
                let patch =
                    ((out_depth * out_h + out_row) * out_w + out_col) * plane.channels * kernel;
                for channel in 0..plane.channels {
                    for kernel_depth in 0..event.params.kd {
                        for kernel_row in 0..plane.kh {
                            for kernel_col in 0..plane.kw {
                                let volume_depth = out_depth * event.params.stride_d + kernel_depth;
                                let volume_row = out_row * plane.stride_h + kernel_row;
                                let volume_col = out_col * plane.stride_w + kernel_col;
                                let volume_index = ((channel * event.params.depth + volume_depth)
                                    * plane.height
                                    + volume_row)
                                    * plane.width
                                    + volume_col;
                                let patch_index = patch
                                    + ((channel * event.params.kd + kernel_depth) * plane.kh
                                        + kernel_row)
                                        * plane.kw
                                    + kernel_col;
                                event.patches[patch_index] = event.volume[volume_index];
                            }
                        }
                    }
                }
            }
        }
    }
}

impl WindowMachineStateMachineContext for Context {
    fn guard_part_valid(&self, event: &WinPartRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_part_invalid(&self, event: &WinPartRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_unpart_valid(&self, event: &WinUnpartRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_unpart_invalid(&self, event: &WinUnpartRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_back_valid(&self, event: &Im2ColBackRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_back_invalid(&self, event: &Im2ColBackRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_col3d_valid(&self, event: &Im2Col3dRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }

    fn guard_col3d_invalid(&self, event: &Im2Col3dRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn effect_part(&mut self, mut event: WinPartRuntime<'_>) -> Result<(), ()> {
        win_part_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_part_reject(&mut self, event: WinPartRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(WindowError::InvalidShape));
        Ok(())
    }

    fn effect_unpart(&mut self, mut event: WinUnpartRuntime<'_>) -> Result<(), ()> {
        win_unpart_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_unpart_reject(&mut self, event: WinUnpartRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(WindowError::InvalidShape));
        Ok(())
    }

    fn effect_back(&mut self, mut event: Im2ColBackRuntime<'_>) -> Result<(), ()> {
        im2col_back_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_back_reject(&mut self, event: Im2ColBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(WindowError::InvalidShape));
        Ok(())
    }

    fn effect_col3d(&mut self, mut event: Im2Col3dRuntime<'_>) -> Result<(), ()> {
        im2col_3d_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_col3d_reject(&mut self, event: Im2Col3dRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(WindowError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Im2Col3dParams, Im2ColBackParams, OpIm2Col3d, OpIm2ColBack, OpWinPart, OpWinUnpart,
        WinPartParams, WindowError, WindowKernel,
    };
    use crate::Kernel;

    fn plane() -> WinPartParams {
        WinPartParams {
            channels: 1,
            height: 2,
            width: 2,
            window: 2,
        }
    }

    #[test]
    fn win_part_and_unpart_round_trip() {
        let source = [1.0_f32, 2.0, 3.0, 4.0];
        let mut windows = [0.0_f32; 4];
        let mut kernel = WindowKernel::new();
        kernel
            .process_event(OpWinPart::new(&source, &mut windows, plane()))
            .unwrap();
        assert_eq!(windows[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(windows[3].to_bits(), 4.0_f32.to_bits());
        let mut restored = [0.0_f32; 4];
        kernel
            .process_event(OpWinUnpart::new(&windows, &mut restored, plane()))
            .unwrap();
        assert_eq!(restored[1].to_bits(), 2.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn im2col_back_scatter_adds_ones() {
        let patches = [1.0_f32];
        let mut image = [9.0_f32; 1];
        let params = Im2ColBackParams {
            channels: 1,
            height: 1,
            width: 1,
            kh: 1,
            kw: 1,
            stride_h: 1,
            stride_w: 1,
        };
        WindowKernel::new()
            .process_event(OpIm2ColBack::new(&patches, &mut image, params))
            .unwrap();
        assert_eq!(image[0].to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn im2col_3d_copies_unit_kernel() {
        let volume = [5.0_f32, 6.0];
        let mut patches = [0.0_f32; 2];
        let params = Im2Col3dParams {
            plane: Im2ColBackParams {
                channels: 1,
                height: 1,
                width: 1,
                kh: 1,
                kw: 1,
                stride_h: 1,
                stride_w: 1,
            },
            depth: 2,
            kd: 1,
            stride_d: 1,
        };
        WindowKernel::new()
            .process_event(OpIm2Col3d::new(&volume, &mut patches, params))
            .unwrap();
        assert_eq!(patches[0].to_bits(), 5.0_f32.to_bits());
        assert_eq!(patches[1].to_bits(), 6.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_win_part() {
        let source = [1.0_f32, 2.0, 3.0, 4.0];
        let mut windows = [0.0_f32; 4];
        Kernel::new()
            .process_event(OpWinPart::new(&source, &mut windows, plane()))
            .unwrap();
        assert_eq!(windows[2].to_bits(), 3.0_f32.to_bits());
    }

    #[test]
    fn win_part_rejects_without_mutation() {
        let source = [1.0_f32];
        let mut windows = [7.0_f32; 4];
        let mut kernel = WindowKernel::new();
        assert_eq!(
            kernel.process_event(OpWinPart::new(&source, &mut windows, plane())),
            Err(WindowError::InvalidShape)
        );
        assert_eq!(windows[0].to_bits(), 7.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_window_inverses() {
        let source = [1.0_f32, 2.0, 3.0, 4.0];
        let mut windows = [0.0_f32; 4];
        let mut kernel = Kernel::new();
        kernel
            .process_event(OpWinPart::new(&source, &mut windows, plane()))
            .unwrap();
        let mut restored = [0.0_f32; 4];
        kernel
            .process_event(OpWinUnpart::new(&windows, &mut restored, plane()))
            .unwrap();
        assert_eq!(restored[1].to_bits(), 2.0_f32.to_bits());
        let mut image = [0.0_f32; 1];
        kernel
            .process_event(OpIm2ColBack::new(
                &[1.0_f32],
                &mut image,
                Im2ColBackParams {
                    channels: 1,
                    height: 1,
                    width: 1,
                    kh: 1,
                    kw: 1,
                    stride_h: 1,
                    stride_w: 1,
                },
            ))
            .unwrap();
        assert_eq!(image[0].to_bits(), 1.0_f32.to_bits());
        let mut patches = [0.0_f32; 2];
        kernel
            .process_event(OpIm2Col3d::new(
                &[5.0_f32, 6.0],
                &mut patches,
                Im2Col3dParams {
                    plane: Im2ColBackParams {
                        channels: 1,
                        height: 1,
                        width: 1,
                        kh: 1,
                        kw: 1,
                        stride_h: 1,
                        stride_w: 1,
                    },
                    depth: 2,
                    kd: 1,
                    stride_d: 1,
                },
            ))
            .unwrap();
        assert_eq!(patches[1].to_bits(), 6.0_f32.to_bits());
    }
}
