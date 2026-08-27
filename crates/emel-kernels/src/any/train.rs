//! Dense cross-entropy and parameter-update kernels.
//!
//! Pinned `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6` declares
//! `op_cross_entropy_loss`, `op_cross_entropy_loss_back`,
//! `op_opt_step_adamw`, and `op_opt_step_sgd` and names `exec_op_*` routes
//! on the arch machines. Those action/guard types are not defined in the
//! pinned headers, so this actor is source-contract.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::cast_precision_loss)]
#![allow(private_interfaces)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Errors returned by training-kernel dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainError {
    /// Batch, class, or optimizer geometry is invalid.
    InvalidShape,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for TrainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid train shape"),
            Self::UnexpectedEvent => formatter.write_str("unexpected train event"),
            Self::Internal => formatter.write_str("internal train dispatch error"),
        }
    }
}

impl std::error::Error for TrainError {}

/// Result returned by training-kernel dispatch.
pub type TrainResult = Result<(), TrainError>;

const fn labels_in_range(labels: &[i32], classes: usize) -> bool {
    let mut offset = 0;
    while offset < labels.len() {
        let label = labels[offset];
        if label < 0 || (label.cast_unsigned() as usize) >= classes {
            return false;
        }
        offset += 1;
    }
    true
}

/// Per-row negative log-softmax.
#[derive(Debug)]
pub struct OpCrossEntropyLoss<'a> {
    logits: &'a [f32],
    labels: &'a [i32],
    loss: &'a mut [f32],
    classes: usize,
}

impl<'a> OpCrossEntropyLoss<'a> {
    /// Creates a cross-entropy request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        logits: &'a [f32],
        labels: &'a [i32],
        loss: &'a mut [f32],
        classes: usize,
    ) -> Self {
        Self {
            logits,
            labels,
            loss,
            classes,
        }
    }

    const fn valid(&self) -> bool {
        self.classes > 0
            && !self.labels.is_empty()
            && self.loss.len() == self.labels.len()
            && self.logits.len() == self.labels.len().saturating_mul(self.classes)
            && labels_in_range(self.labels, self.classes)
    }
}

/// Softmax-minus-one-hot Jacobian of [`OpCrossEntropyLoss`].
#[derive(Debug)]
pub struct OpCrossEntropyLossBack<'a> {
    logits: &'a [f32],
    labels: &'a [i32],
    grad_loss: &'a [f32],
    grad_logits: &'a mut [f32],
    classes: usize,
}

impl<'a> OpCrossEntropyLossBack<'a> {
    /// Creates a cross-entropy backward request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        logits: &'a [f32],
        labels: &'a [i32],
        grad_loss: &'a [f32],
        grad_logits: &'a mut [f32],
        classes: usize,
    ) -> Self {
        Self {
            logits,
            labels,
            grad_loss,
            grad_logits,
            classes,
        }
    }

    const fn valid(&self) -> bool {
        self.classes > 0
            && !self.labels.is_empty()
            && self.grad_loss.len() == self.labels.len()
            && self.logits.len() == self.labels.len().saturating_mul(self.classes)
            && self.grad_logits.len() == self.logits.len()
            && labels_in_range(self.labels, self.classes)
    }
}

/// `AdamW` hyperparameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWParams {
    /// Step size.
    pub lr: f32,
    /// First-moment decay.
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// Denominator floor.
    pub eps: f32,
    /// Decoupled weight decay.
    pub wd: f32,
    /// 1-based update count.
    pub step: u32,
}

impl AdamWParams {
    const fn valid(self) -> bool {
        self.lr.is_finite()
            && self.beta1 >= 0.0
            && self.beta1 < 1.0
            && self.beta2 >= 0.0
            && self.beta2 < 1.0
            && self.eps > 0.0
            && self.eps.is_finite()
            && self.wd.is_finite()
            && self.wd >= 0.0
            && self.step > 0
    }
}

/// In-place `AdamW` update of dense F32 parameters.
#[derive(Debug)]
pub struct OpOptStepAdamW<'a> {
    params: &'a mut [f32],
    grads: &'a [f32],
    moment1: &'a mut [f32],
    moment2: &'a mut [f32],
    hyper: AdamWParams,
}

impl<'a> OpOptStepAdamW<'a> {
    /// Creates an `AdamW` request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(
        params: &'a mut [f32],
        grads: &'a [f32],
        moment1: &'a mut [f32],
        moment2: &'a mut [f32],
        hyper: AdamWParams,
    ) -> Self {
        Self {
            params,
            grads,
            moment1,
            moment2,
            hyper,
        }
    }

    const fn valid(&self) -> bool {
        !self.params.is_empty()
            && self.grads.len() == self.params.len()
            && self.moment1.len() == self.params.len()
            && self.moment2.len() == self.params.len()
            && self.hyper.valid()
    }
}

/// In-place SGD update of dense F32 parameters.
#[derive(Debug)]
pub struct OpOptStepSgd<'a> {
    params: &'a mut [f32],
    grads: &'a [f32],
    lr: f32,
    wd: f32,
}

impl<'a> OpOptStepSgd<'a> {
    /// Creates an SGD request. Shape validation occurs in a guard.
    #[must_use]
    pub const fn new(params: &'a mut [f32], grads: &'a [f32], lr: f32, wd: f32) -> Self {
        Self {
            params,
            grads,
            lr,
            wd,
        }
    }

    const fn valid(&self) -> bool {
        !self.params.is_empty()
            && self.grads.len() == self.params.len()
            && self.lr.is_finite()
            && self.wd.is_finite()
            && self.wd >= 0.0
    }
}

struct CrossEntropyRuntime<'a> {
    event: OpCrossEntropyLoss<'a>,
    result: &'a Cell<TrainResult>,
}

struct CrossEntropyBackRuntime<'a> {
    event: OpCrossEntropyLossBack<'a>,
    result: &'a Cell<TrainResult>,
}

struct AdamWRuntime<'a> {
    event: OpOptStepAdamW<'a>,
    result: &'a Cell<TrainResult>,
}

struct SgdRuntime<'a> {
    event: OpOptStepSgd<'a>,
    result: &'a Cell<TrainResult>,
}

#[derive(Default)]
struct Context;

sml! {
    TrainMachine<'dispatch> {
        "ready"_s <= *"ready"_s + CrossEntropy(CrossEntropyRuntime<'dispatch>) [guard_ce_valid] / effect_ce,
        "ready"_s <= "ready"_s + CrossEntropy(CrossEntropyRuntime<'dispatch>) [guard_ce_invalid] / effect_ce_reject,
        "ready"_s <= "ready"_s + CrossEntropyBack(CrossEntropyBackRuntime<'dispatch>) [guard_ce_back_valid] / effect_ce_back,
        "ready"_s <= "ready"_s + CrossEntropyBack(CrossEntropyBackRuntime<'dispatch>) [guard_ce_back_invalid] / effect_ce_back_reject,
        "ready"_s <= "ready"_s + AdamW(AdamWRuntime<'dispatch>) [guard_adamw_valid] / effect_adamw,
        "ready"_s <= "ready"_s + AdamW(AdamWRuntime<'dispatch>) [guard_adamw_invalid] / effect_adamw_reject,
        "ready"_s <= "ready"_s + Sgd(SgdRuntime<'dispatch>) [guard_sgd_valid] / effect_sgd,
        "ready"_s <= "ready"_s + Sgd(SgdRuntime<'dispatch>) [guard_sgd_invalid] / effect_sgd_reject,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Dispatch contract for training events.
pub trait TrainEvent {
    /// Result of one run-to-completion dispatch.
    type Output;
    /// Routes the event through the owning actor.
    fn dispatch(self, kernel: &mut TrainKernel) -> Self::Output;
}

/// Single-writer training actor.
pub struct TrainKernel {
    machine: TrainMachineStateMachine<Context>,
}

impl fmt::Debug for TrainKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TrainKernel")
            .finish_non_exhaustive()
    }
}

impl Default for TrainKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainKernel {
    /// Constructs an independent training actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: TrainMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one typed training event run-to-completion.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::InvalidShape`] when the dense buffers cannot form
    /// the requested loss or optimizer geometry.
    ///
    /// # Panics
    ///
    /// Panics if the generated machine does not return to `Ready`.
    pub fn process_event<E: TrainEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn ce(&mut self, event: OpCrossEntropyLoss<'_>) -> TrainResult {
        let result = Cell::new(Err(TrainError::UnexpectedEvent));
        self.machine
            .process_event(TrainMachineEvents::CrossEntropy(CrossEntropyRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| TrainError::Internal)?;
        ready(self);
        result.get()
    }

    fn ce_back(&mut self, event: OpCrossEntropyLossBack<'_>) -> TrainResult {
        let result = Cell::new(Err(TrainError::UnexpectedEvent));
        self.machine
            .process_event(TrainMachineEvents::CrossEntropyBack(
                CrossEntropyBackRuntime {
                    event,
                    result: &result,
                },
            ))
            .map_err(|_| TrainError::Internal)?;
        ready(self);
        result.get()
    }

    fn adamw(&mut self, event: OpOptStepAdamW<'_>) -> TrainResult {
        let result = Cell::new(Err(TrainError::UnexpectedEvent));
        self.machine
            .process_event(TrainMachineEvents::AdamW(AdamWRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| TrainError::Internal)?;
        ready(self);
        result.get()
    }

    fn sgd(&mut self, event: OpOptStepSgd<'_>) -> TrainResult {
        let result = Cell::new(Err(TrainError::UnexpectedEvent));
        self.machine
            .process_event(TrainMachineEvents::Sgd(SgdRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| TrainError::Internal)?;
        ready(self);
        result.get()
    }

    /// Reports whether the actor is ready for dispatch.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&TrainMachineStates::Ready)
    }
}

fn ready(kernel: &TrainKernel) {
    assert!(
        kernel.machine.is(&TrainMachineStates::Ready),
        "train machine must return to ready after dispatch"
    );
}

impl TrainEvent for OpCrossEntropyLoss<'_> {
    type Output = TrainResult;
    fn dispatch(self, kernel: &mut TrainKernel) -> Self::Output {
        kernel.ce(self)
    }
}

impl TrainEvent for OpCrossEntropyLossBack<'_> {
    type Output = TrainResult;
    fn dispatch(self, kernel: &mut TrainKernel) -> Self::Output {
        kernel.ce_back(self)
    }
}

impl TrainEvent for OpOptStepAdamW<'_> {
    type Output = TrainResult;
    fn dispatch(self, kernel: &mut TrainKernel) -> Self::Output {
        kernel.adamw(self)
    }
}

impl TrainEvent for OpOptStepSgd<'_> {
    type Output = TrainResult;
    fn dispatch(self, kernel: &mut TrainKernel) -> Self::Output {
        kernel.sgd(self)
    }
}

fn row_max(row: &[f32]) -> f32 {
    let mut peak = row[0];
    for sample in row.iter().copied().skip(1) {
        if sample > peak {
            peak = sample;
        }
    }
    peak
}

fn softmax_sum(row: &[f32], peak: f32) -> f32 {
    let mut sum = 0.0_f32;
    for sample in row.iter().copied() {
        sum += (sample - peak).exp();
    }
    sum
}

fn cross_entropy_values(event: &mut OpCrossEntropyLoss<'_>) {
    let classes = event.classes;
    for (row_index, label) in event.labels.iter().copied().enumerate() {
        let start = row_index * classes;
        let row = &event.logits[start..start + classes];
        let peak = row_max(row);
        let sum = softmax_sum(row, peak);
        let class = label.cast_unsigned() as usize;
        event.loss[row_index] = peak + sum.ln() - row[class];
    }
}

fn cross_entropy_back_values(event: &mut OpCrossEntropyLossBack<'_>) {
    let classes = event.classes;
    for (row_index, label) in event.labels.iter().copied().enumerate() {
        let start = row_index * classes;
        let row = &event.logits[start..start + classes];
        let peak = row_max(row);
        let sum = softmax_sum(row, peak);
        let inv = sum.recip();
        let class = label.cast_unsigned() as usize;
        let scale = event.grad_loss[row_index];
        for (offset, sample) in row.iter().copied().enumerate() {
            let prob = (sample - peak).exp() * inv;
            let one_hot = if offset == class { 1.0_f32 } else { 0.0 };
            event.grad_logits[start + offset] = scale * (prob - one_hot);
        }
    }
}

fn adamw_values(event: &mut OpOptStepAdamW<'_>) {
    let one_b1 = 1.0 - event.hyper.beta1;
    let one_b2 = 1.0 - event.hyper.beta2;
    let corr1 = 1.0 - event.hyper.beta1.powi(event.hyper.step.cast_signed());
    let corr2 = 1.0 - event.hyper.beta2.powi(event.hyper.step.cast_signed());
    for index in 0..event.params.len() {
        let grad = event.grads[index];
        event.moment1[index] = event
            .hyper
            .beta1
            .mul_add(event.moment1[index], one_b1 * grad);
        event.moment2[index] = event
            .hyper
            .beta2
            .mul_add(event.moment2[index], one_b2 * grad * grad);
        let mhat = event.moment1[index] / corr1;
        let vhat = event.moment2[index] / corr2;
        let update = mhat / (vhat.sqrt() + event.hyper.eps);
        let update_and_decay = event.hyper.wd.mul_add(event.params[index], update);
        event.params[index] = event
            .hyper
            .lr
            .mul_add(-update_and_decay, event.params[index]);
    }
}

fn sgd_values(event: &mut OpOptStepSgd<'_>) {
    for index in 0..event.params.len() {
        let decayed = event.params[index].mul_add(event.wd, event.grads[index]);
        event.params[index] = event.lr.mul_add(-decayed, event.params[index]);
    }
}

impl TrainMachineStateMachineContext for Context {
    fn guard_ce_valid(&self, event: &CrossEntropyRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_ce_invalid(&self, event: &CrossEntropyRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn guard_ce_back_valid(&self, event: &CrossEntropyBackRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_ce_back_invalid(&self, event: &CrossEntropyBackRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn guard_adamw_valid(&self, event: &AdamWRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_adamw_invalid(&self, event: &AdamWRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn guard_sgd_valid(&self, event: &SgdRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.valid())
    }
    fn guard_sgd_invalid(&self, event: &SgdRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.valid())
    }
    fn effect_ce(&mut self, mut event: CrossEntropyRuntime<'_>) -> Result<(), ()> {
        cross_entropy_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_ce_reject(&mut self, event: CrossEntropyRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(TrainError::InvalidShape));
        Ok(())
    }
    fn effect_ce_back(&mut self, mut event: CrossEntropyBackRuntime<'_>) -> Result<(), ()> {
        cross_entropy_back_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_ce_back_reject(&mut self, event: CrossEntropyBackRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(TrainError::InvalidShape));
        Ok(())
    }
    fn effect_adamw(&mut self, mut event: AdamWRuntime<'_>) -> Result<(), ()> {
        adamw_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_adamw_reject(&mut self, event: AdamWRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(TrainError::InvalidShape));
        Ok(())
    }
    fn effect_sgd(&mut self, mut event: SgdRuntime<'_>) -> Result<(), ()> {
        sgd_values(&mut event.event);
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_sgd_reject(&mut self, event: SgdRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(TrainError::InvalidShape));
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AdamWParams, OpCrossEntropyLoss, OpCrossEntropyLossBack, OpOptStepAdamW, OpOptStepSgd,
        TrainError, TrainKernel,
    };
    use crate::Kernel;

    #[test]
    fn cross_entropy_zero_logits_is_ln_classes() {
        let logits = [0.0_f32, 0.0];
        let labels = [0_i32];
        let mut loss = [0.0_f32; 1];
        TrainKernel::new()
            .process_event(OpCrossEntropyLoss::new(&logits, &labels, &mut loss, 2))
            .unwrap();
        assert_eq!(loss[0].to_bits(), core::f32::consts::LN_2.to_bits());
    }

    #[test]
    fn cross_entropy_back_is_softmax_minus_one_hot() {
        let logits = [0.0_f32, 0.0];
        let labels = [0_i32];
        let grad_loss = [1.0_f32];
        let mut grad_logits = [0.0_f32; 2];
        TrainKernel::new()
            .process_event(OpCrossEntropyLossBack::new(
                &logits,
                &labels,
                &grad_loss,
                &mut grad_logits,
                2,
            ))
            .unwrap();
        assert_eq!(grad_logits[0].to_bits(), (-0.5_f32).to_bits());
        assert_eq!(grad_logits[1].to_bits(), 0.5_f32.to_bits());
    }

    #[test]
    fn sgd_and_public_kernel_update_parameters() {
        let mut params = [1.0_f32];
        let grads = [1.0_f32];
        Kernel::new()
            .process_event(OpOptStepSgd::new(&mut params, &grads, 0.5, 0.0))
            .unwrap();
        assert_eq!(params[0].to_bits(), 0.5_f32.to_bits());
    }

    #[test]
    fn adamw_first_step_with_zero_betas() {
        let mut params = [1.0_f32];
        let grads = [1.0_f32];
        let mut m = [0.0_f32];
        let mut v = [0.0_f32];
        TrainKernel::new()
            .process_event(OpOptStepAdamW::new(
                &mut params,
                &grads,
                &mut m,
                &mut v,
                AdamWParams {
                    lr: 1.0,
                    beta1: 0.0,
                    beta2: 0.0,
                    eps: 1.0,
                    wd: 0.0,
                    step: 1,
                },
            ))
            .unwrap();
        assert_eq!(params[0].to_bits(), 0.5_f32.to_bits());
    }

    #[test]
    fn cross_entropy_rejects_without_mutation() {
        let logits = [0.0_f32];
        let labels = [0_i32];
        let mut loss = [9.0_f32];
        let mut kernel = TrainKernel::new();
        assert_eq!(
            kernel.process_event(OpCrossEntropyLoss::new(&logits, &labels, &mut loss, 2)),
            Err(TrainError::InvalidShape)
        );
        assert_eq!(loss[0].to_bits(), 9.0_f32.to_bits());
    }

    #[test]
    fn public_kernel_dispatches_cross_entropy_and_adamw() {
        let logits = [0.0_f32, 0.0];
        let labels = [0_i32];
        let mut loss = [0.0_f32; 1];
        let mut kernel = Kernel::new();
        kernel
            .process_event(OpCrossEntropyLoss::new(&logits, &labels, &mut loss, 2))
            .unwrap();
        assert_eq!(loss[0].to_bits(), core::f32::consts::LN_2.to_bits());
        let mut grad_logits = [0.0_f32; 2];
        kernel
            .process_event(OpCrossEntropyLossBack::new(
                &logits,
                &labels,
                &[1.0_f32],
                &mut grad_logits,
                2,
            ))
            .unwrap();
        assert_eq!(grad_logits[1].to_bits(), 0.5_f32.to_bits());
        let mut params = [1.0_f32];
        let mut m = [0.0_f32];
        let mut v = [0.0_f32];
        kernel
            .process_event(OpOptStepAdamW::new(
                &mut params,
                &[1.0_f32],
                &mut m,
                &mut v,
                AdamWParams {
                    lr: 1.0,
                    beta1: 0.0,
                    beta2: 0.0,
                    eps: 1.0,
                    wd: 0.0,
                    step: 1,
                },
            ))
            .unwrap();
        assert_eq!(params[0].to_bits(), 0.5_f32.to_bits());
    }
}
