//! Avg/max pooling kernels over dense F32 buffers.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_pool_1d`, `op_pool_2d`, and `op_pool_2d_back` and names
//! `exec_op_pool*` routes on the arch machines. Those action/guard types are
//! not defined in the pinned headers, so this actor is source-contract.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Pooling variant matching pinned `pool_subop`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PoolSubOp {
    /// Window maximum. C++ `pool_subop::max`.
    Max = 0,
    /// Window mean. C++ `pool_subop::avg`.
    Avg = 1,
}

/// Errors returned by pooling dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PoolError {
    /// Kernel, stride, or buffer geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for PoolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid pool shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected pool event"),
            Self::Internal => formatter.write_str("internal pool dispatch error"),
        }
    }
}

impl std::error::Error for PoolError {}

/// Result returned by pooling dispatch.
pub type PoolResult = Result<(), PoolError>;

const fn pool_out(len: usize, kernel: usize, stride: usize) -> Option<usize> {
    if kernel == 0 || stride == 0 || len < kernel {
        None
    } else {
        Some((len - kernel) / stride + 1)
    }
}

/// 1-D pooling over a dense F32 vector.
#[derive(Debug)]
pub struct OpPool1d<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    kernel: usize,
    stride: usize,
    subop: PoolSubOp,
}

impl<'a> OpPool1d<'a> {
    /// Creates a 1-D pool request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        input: &'a [f32],
        output: &'a mut [f32],
        kernel: usize,
        stride: usize,
        subop: PoolSubOp,
    ) -> Self {
        Self {
            input,
            output,
            kernel,
            stride,
            subop,
        }
    }

    const fn valid(&self) -> bool {
        match pool_out(self.input.len(), self.kernel, self.stride) {
            Some(out) => out == self.output.len(),
            None => false,
        }
    }
}

/// Geometry for a dense 2-D pooling window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Pool2dParams {
    /// Input rows.
    pub rows: usize,
    /// Input columns.
    pub cols: usize,
    /// Window height.
    pub kernel_h: usize,
    /// Window width.
    pub kernel_w: usize,
    /// Vertical stride.
    pub stride_h: usize,
    /// Horizontal stride.
    pub stride_w: usize,
}

/// 2-D pooling over a dense row-major F32 plane.
#[derive(Debug)]
pub struct OpPool2d<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
    rows: usize,
    cols: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    subop: PoolSubOp,
}

impl<'a> OpPool2d<'a> {
    /// Creates a 2-D pool request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        input: &'a [f32],
        output: &'a mut [f32],
        params: Pool2dParams,
        subop: PoolSubOp,
    ) -> Self {
        Self {
            input,
            output,
            rows: params.rows,
            cols: params.cols,
            kernel_h: params.kernel_h,
            kernel_w: params.kernel_w,
            stride_h: params.stride_h,
            stride_w: params.stride_w,
            subop,
        }
    }

    const fn valid(&self) -> bool {
        if self.input.len() != self.rows.saturating_mul(self.cols) {
            return false;
        }
        let Some(out_h) = pool_out(self.rows, self.kernel_h, self.stride_h) else {
            return false;
        };
        let Some(out_w) = pool_out(self.cols, self.kernel_w, self.stride_w) else {
            return false;
        };
        self.output.len() == out_h.saturating_mul(out_w)
    }
}

/// 2-D pooling backward over a dense row-major F32 plane.
#[derive(Debug)]
pub struct OpPool2dBack<'a> {
    input: &'a [f32],
    grad_out: &'a [f32],
    grad_in: &'a mut [f32],
    rows: usize,
    cols: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    subop: PoolSubOp,
}

impl<'a> OpPool2dBack<'a> {
    /// Creates a 2-D pool backward request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        input: &'a [f32],
        grad_out: &'a [f32],
        grad_in: &'a mut [f32],
        params: Pool2dParams,
        subop: PoolSubOp,
    ) -> Self {
        Self {
            input,
            grad_out,
            grad_in,
            rows: params.rows,
            cols: params.cols,
            kernel_h: params.kernel_h,
            kernel_w: params.kernel_w,
            stride_h: params.stride_h,
            stride_w: params.stride_w,
            subop,
        }
    }

    const fn valid(&self) -> bool {
        if self.input.len() != self.rows.saturating_mul(self.cols)
            || self.grad_in.len() != self.input.len()
        {
            return false;
        }
        let Some(out_h) = pool_out(self.rows, self.kernel_h, self.stride_h) else {
            return false;
        };
        let Some(out_w) = pool_out(self.cols, self.kernel_w, self.stride_w) else {
            return false;
        };
        self.grad_out.len() == out_h.saturating_mul(out_w)
    }
}

struct Pool1dRuntime<'a> {
    event: OpPool1d<'a>,
    result: &'a Cell<PoolResult>,
}

struct Pool2dRuntime<'a> {
    event: OpPool2d<'a>,
    result: &'a Cell<PoolResult>,
}

struct Pool2dBackRuntime<'a> {
    event: OpPool2dBack<'a>,
    result: &'a Cell<PoolResult>,
}

#[derive(Default)]
struct Context;

sml! {
    PoolMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Pool1d(Pool1dRuntime<'dispatch>) [guard_pool1d_avg] / effect_pool1d_avg,
        "ready"_s <= "ready"_s + Pool1d(Pool1dRuntime<'dispatch>) [guard_pool1d_max] / effect_pool1d_max,
        "ready"_s <= "ready"_s + Pool1d(Pool1dRuntime<'dispatch>) [guard_pool1d_invalid] / effect_pool1d_reject,
        "ready"_s <= "ready"_s + Pool2d(Pool2dRuntime<'dispatch>) [guard_pool2d_avg] / effect_pool2d_avg,
        "ready"_s <= "ready"_s + Pool2d(Pool2dRuntime<'dispatch>) [guard_pool2d_max] / effect_pool2d_max,
        "ready"_s <= "ready"_s + Pool2d(Pool2dRuntime<'dispatch>) [guard_pool2d_invalid] / effect_pool2d_reject,
        "ready"_s <= "ready"_s + Pool2dBack(Pool2dBackRuntime<'dispatch>) [guard_pool2d_back_avg] / effect_pool2d_back_avg,
        "ready"_s <= "ready"_s + Pool2dBack(Pool2dBackRuntime<'dispatch>) [guard_pool2d_back_max] / effect_pool2d_back_max,
        "ready"_s <= "ready"_s + Pool2dBack(Pool2dBackRuntime<'dispatch>) [guard_pool2d_back_invalid] / effect_pool2d_back_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for pool-family events.
pub trait PoolEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut PoolKernel) -> Self::Output;
}

/// Single-writer pooling actor.
pub struct PoolKernel {
    machine: PoolMachineStateMachine<Context>,
}

impl fmt::Debug for PoolKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("PoolKernel").finish_non_exhaustive()
    }
}

impl Default for PoolKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl PoolKernel {
    /// Constructs an independent pooling actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: PoolMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed pool-family event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`PoolError::InvalidShape`] when the kernel, stride, or buffer
    /// geometry cannot form a dense pooling window.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: PoolEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn pool1d(&mut self, event: OpPool1d<'_>) -> PoolResult {
        let result = Cell::new(Err(PoolError::UnexpectedEvent));
        self.machine
            .process_event(PoolMachineEvents::Pool1d(Pool1dRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PoolError::Internal)?;
        assert!(
            self.machine.is(&PoolMachineStates::Ready),
            "pool machine must return to ready after dispatch"
        );
        result.get()
    }

    fn pool2d(&mut self, event: OpPool2d<'_>) -> PoolResult {
        let result = Cell::new(Err(PoolError::UnexpectedEvent));
        self.machine
            .process_event(PoolMachineEvents::Pool2d(Pool2dRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PoolError::Internal)?;
        assert!(
            self.machine.is(&PoolMachineStates::Ready),
            "pool machine must return to ready after dispatch"
        );
        result.get()
    }

    fn pool2d_back(&mut self, event: OpPool2dBack<'_>) -> PoolResult {
        let result = Cell::new(Err(PoolError::UnexpectedEvent));
        self.machine
            .process_event(PoolMachineEvents::Pool2dBack(Pool2dBackRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| PoolError::Internal)?;
        assert!(
            self.machine.is(&PoolMachineStates::Ready),
            "pool machine must return to ready after dispatch"
        );
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&PoolMachineStates::Ready)
    }
}

impl PoolEvent for OpPool1d<'_> {
    type Output = PoolResult;
    fn dispatch(self, kernel: &mut PoolKernel) -> Self::Output {
        kernel.pool1d(self)
    }
}

impl PoolEvent for OpPool2d<'_> {
    type Output = PoolResult;
    fn dispatch(self, kernel: &mut PoolKernel) -> Self::Output {
        kernel.pool2d(self)
    }
}

impl PoolEvent for OpPool2dBack<'_> {
    type Output = PoolResult;
    fn dispatch(self, kernel: &mut PoolKernel) -> Self::Output {
        kernel.pool2d_back(self)
    }
}

#[allow(clippy::cast_precision_loss)]
const fn recip_count(count: usize) -> f32 {
    if count == 0 {
        0.0
    } else {
        (count as f32).recip()
    }
}

fn pool1d_avg(event: &mut OpPool1d<'_>) {
    let scale = recip_count(event.kernel);
    for (out_index, slot) in event.output.iter_mut().enumerate() {
        let start = out_index * event.stride;
        let mut acc = 0.0_f32;
        for offset in 0..event.kernel {
            acc += event.input[start + offset];
        }
        *slot = acc * scale;
    }
}

fn pool1d_max(event: &mut OpPool1d<'_>) {
    for (out_index, slot) in event.output.iter_mut().enumerate() {
        let start = out_index * event.stride;
        let mut peak = event.input[start];
        for offset in 1..event.kernel {
            let sample = event.input[start + offset];
            if sample > peak {
                peak = sample;
            }
        }
        *slot = peak;
    }
}

fn pool2d_avg(event: &mut OpPool2d<'_>) {
    let out_w = pool_out(event.cols, event.kernel_w, event.stride_w).unwrap_or(0);
    let scale = recip_count(event.kernel_h.saturating_mul(event.kernel_w));
    for (out_index, slot) in event.output.iter_mut().enumerate() {
        let out_row = out_index / out_w;
        let out_col = out_index % out_w;
        let row0 = out_row * event.stride_h;
        let col0 = out_col * event.stride_w;
        let mut acc = 0.0_f32;
        for dh in 0..event.kernel_h {
            let row = row0 + dh;
            for dw in 0..event.kernel_w {
                acc += event.input[row * event.cols + col0 + dw];
            }
        }
        *slot = acc * scale;
    }
}

fn pool2d_max(event: &mut OpPool2d<'_>) {
    let out_w = pool_out(event.cols, event.kernel_w, event.stride_w).unwrap_or(0);
    for (out_index, slot) in event.output.iter_mut().enumerate() {
        let out_row = out_index / out_w;
        let out_col = out_index % out_w;
        let row0 = out_row * event.stride_h;
        let col0 = out_col * event.stride_w;
        let mut peak = event.input[row0 * event.cols + col0];
        for dh in 0..event.kernel_h {
            let row = row0 + dh;
            for dw in 0..event.kernel_w {
                let sample = event.input[row * event.cols + col0 + dw];
                if sample > peak {
                    peak = sample;
                }
            }
        }
        *slot = peak;
    }
}

fn pool2d_back_avg(event: &mut OpPool2dBack<'_>) {
    event.grad_in.fill(0.0_f32);
    let out_w = pool_out(event.cols, event.kernel_w, event.stride_w).unwrap_or(0);
    let scale = recip_count(event.kernel_h.saturating_mul(event.kernel_w));
    for (out_index, grad) in event.grad_out.iter().enumerate() {
        let out_row = out_index / out_w;
        let out_col = out_index % out_w;
        let row0 = out_row * event.stride_h;
        let col0 = out_col * event.stride_w;
        let share = *grad * scale;
        for dh in 0..event.kernel_h {
            let row = row0 + dh;
            for dw in 0..event.kernel_w {
                event.grad_in[row * event.cols + col0 + dw] += share;
            }
        }
    }
}

fn pool2d_back_max(event: &mut OpPool2dBack<'_>) {
    event.grad_in.fill(0.0_f32);
    let out_w = pool_out(event.cols, event.kernel_w, event.stride_w).unwrap_or(0);
    for (out_index, grad) in event.grad_out.iter().enumerate() {
        let out_row = out_index / out_w;
        let out_col = out_index % out_w;
        let row0 = out_row * event.stride_h;
        let col0 = out_col * event.stride_w;
        let mut peak = event.input[row0 * event.cols + col0];
        let mut peak_index = row0 * event.cols + col0;
        for dh in 0..event.kernel_h {
            let row = row0 + dh;
            for dw in 0..event.kernel_w {
                let index = row * event.cols + col0 + dw;
                let sample = event.input[index];
                if sample > peak {
                    peak = sample;
                    peak_index = index;
                }
            }
        }
        event.grad_in[peak_index] += *grad;
    }
}

impl PoolMachineStateMachineContext for Context {
    fn guard_pool1d_avg(&self, event: &Pool1dRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid() && event.event.subop == PoolSubOp::Avg)
    }

    fn guard_pool1d_max(&self, event: &Pool1dRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid() && event.event.subop == PoolSubOp::Max)
    }

    fn guard_pool1d_invalid(&self, event: &Pool1dRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_pool2d_avg(&self, event: &Pool2dRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid() && event.event.subop == PoolSubOp::Avg)
    }

    fn guard_pool2d_max(&self, event: &Pool2dRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid() && event.event.subop == PoolSubOp::Max)
    }

    fn guard_pool2d_invalid(&self, event: &Pool2dRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn guard_pool2d_back_avg(&self, event: &Pool2dBackRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid() && event.event.subop == PoolSubOp::Avg)
    }

    fn guard_pool2d_back_max(&self, event: &Pool2dBackRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid() && event.event.subop == PoolSubOp::Max)
    }

    fn guard_pool2d_back_invalid(&self, event: &Pool2dBackRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }

    fn effect_pool1d_avg(&mut self, mut event: Pool1dRuntime<'_>) -> Result<(), ()> {
        pool1d_avg(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_pool1d_max(&mut self, mut event: Pool1dRuntime<'_>) -> Result<(), ()> {
        pool1d_max(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_pool1d_reject(&mut self, event: Pool1dRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(PoolError::InvalidShape));
        Ok(())
    }

    fn effect_pool2d_avg(&mut self, mut event: Pool2dRuntime<'_>) -> Result<(), ()> {
        pool2d_avg(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_pool2d_max(&mut self, mut event: Pool2dRuntime<'_>) -> Result<(), ()> {
        pool2d_max(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_pool2d_reject(&mut self, event: Pool2dRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(PoolError::InvalidShape));
        Ok(())
    }

    fn effect_pool2d_back_avg(&mut self, mut event: Pool2dBackRuntime<'_>) -> Result<(), ()> {
        pool2d_back_avg(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_pool2d_back_max(&mut self, mut event: Pool2dBackRuntime<'_>) -> Result<(), ()> {
        pool2d_back_max(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_pool2d_back_reject(&mut self, event: Pool2dBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(PoolError::InvalidShape));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{OpPool1d, OpPool2d, OpPool2dBack, Pool2dParams, PoolError, PoolKernel, PoolSubOp};

    #[test]
    fn pool1d_avg_uses_kernel_mean() {
        let input = [1.0_f32, 3.0, 5.0, 7.0];
        let mut output = [0.0_f32; 2];
        let mut kernel = PoolKernel::new();
        kernel
            .process_event(OpPool1d::new(&input, &mut output, 2, 2, PoolSubOp::Avg))
            .unwrap();
        assert_eq!(output[0].to_bits(), 2.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 6.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn pool2d_max_selects_window_peak() {
        let input = [1.0_f32, 2.0, 0.5, 8.0];
        let mut output = [0.0_f32; 1];
        let mut kernel = PoolKernel::new();
        let params = Pool2dParams {
            rows: 2,
            cols: 2,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
        };
        kernel
            .process_event(OpPool2d::new(&input, &mut output, params, PoolSubOp::Max))
            .unwrap();
        assert_eq!(output[0].to_bits(), 8.0_f32.to_bits());
    }

    #[test]
    fn pool2d_back_avg_spreads_gradient() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let grad_out = [4.0_f32];
        let mut grad_in = [9.0_f32; 4];
        let mut kernel = PoolKernel::new();
        let params = Pool2dParams {
            rows: 2,
            cols: 2,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
        };
        kernel
            .process_event(OpPool2dBack::new(
                &input,
                &grad_out,
                &mut grad_in,
                params,
                PoolSubOp::Avg,
            ))
            .unwrap();
        assert_eq!(grad_in[0].to_bits(), 1.0_f32.to_bits());
        assert_eq!(grad_in[1].to_bits(), 1.0_f32.to_bits());
        assert_eq!(grad_in[2].to_bits(), 1.0_f32.to_bits());
        assert_eq!(grad_in[3].to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn pool1d_rejects_kernel_without_mutation() {
        let input = [1.0_f32, 2.0];
        let mut output = [5.0_f32; 2];
        let mut kernel = PoolKernel::new();
        assert_eq!(
            kernel.process_event(OpPool1d::new(&input, &mut output, 3, 1, PoolSubOp::Avg)),
            Err(PoolError::InvalidShape)
        );
        assert_eq!(output[0].to_bits(), 5.0_f32.to_bits());
        assert!(kernel.is_ready());
    }

    #[test]
    fn pool1d_max_and_pool2d_avg_and_back_max() {
        let input = [1.0_f32, 3.0, 2.0, 0.0];
        let mut output = [0.0_f32; 2];
        let mut kernel = PoolKernel::new();
        kernel
            .process_event(OpPool1d::new(&input, &mut output, 2, 2, PoolSubOp::Max))
            .unwrap();
        assert_eq!(output[0].to_bits(), 3.0_f32.to_bits());
        assert_eq!(output[1].to_bits(), 2.0_f32.to_bits());

        let input = [1.0_f32, 2.0, 0.5, 8.0];
        let mut avg = [0.0_f32; 1];
        let params = Pool2dParams {
            rows: 2,
            cols: 2,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
        };
        kernel
            .process_event(OpPool2d::new(&input, &mut avg, params, PoolSubOp::Avg))
            .unwrap();
        assert_eq!(avg[0].to_bits(), 2.875_f32.to_bits());

        let mut avg_reject = [5.0_f32; 1];
        assert_eq!(
            kernel.process_event(OpPool2d::new(
                &input,
                &mut avg_reject,
                Pool2dParams {
                    rows: 2,
                    cols: 2,
                    kernel_h: 3,
                    kernel_w: 2,
                    stride_h: 1,
                    stride_w: 1,
                },
                PoolSubOp::Avg,
            )),
            Err(PoolError::InvalidShape)
        );
        assert_eq!(avg_reject[0].to_bits(), 5.0_f32.to_bits());

        let grad_out = [4.0_f32];
        let mut grad_in = [9.0_f32; 4];
        kernel
            .process_event(OpPool2dBack::new(
                &input,
                &grad_out,
                &mut grad_in,
                params,
                PoolSubOp::Max,
            ))
            .unwrap();
        assert_eq!(grad_in[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(grad_in[3].to_bits(), 4.0_f32.to_bits());
        assert_eq!(PoolError::InvalidShape.to_string(), "invalid pool shape");
        assert_eq!(
            PoolError::UnexpectedEvent.to_string(),
            "unexpected pool event"
        );
        assert_eq!(
            PoolError::Internal.to_string(),
            "internal pool dispatch error"
        );
        let _ = format!("{:?}", PoolKernel::default());
        assert!(kernel.is_ready());
    }

    #[test]
    fn public_kernel_dispatches_pool_family() {
        let input = [1.0_f32, 3.0, 5.0, 7.0];
        let mut output = [0.0_f32; 2];
        crate::Kernel::new()
            .process_event(OpPool1d::new(&input, &mut output, 2, 2, PoolSubOp::Avg))
            .unwrap();
        assert_eq!(output[0].to_bits(), 2.0_f32.to_bits());
        let mut pooled = [0.0_f32; 1];
        crate::Kernel::new()
            .process_event(OpPool2d::new(
                &input,
                &mut pooled,
                Pool2dParams {
                    rows: 2,
                    cols: 2,
                    kernel_h: 2,
                    kernel_w: 2,
                    stride_h: 2,
                    stride_w: 2,
                },
                PoolSubOp::Max,
            ))
            .unwrap();
        assert_eq!(pooled[0].to_bits(), 7.0_f32.to_bits());
        let grad_out = [4.0_f32];
        let mut grad_in = [0.0_f32; 4];
        crate::Kernel::new()
            .process_event(OpPool2dBack::new(
                &input,
                &grad_out,
                &mut grad_in,
                Pool2dParams {
                    rows: 2,
                    cols: 2,
                    kernel_h: 2,
                    kernel_w: 2,
                    stride_h: 2,
                    stride_w: 2,
                },
                PoolSubOp::Avg,
            ))
            .unwrap();
        assert_eq!(grad_in[0].to_bits(), 1.0_f32.to_bits());
    }
}
