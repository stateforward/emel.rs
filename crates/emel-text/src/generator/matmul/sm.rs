//! Source-aligned bounded text-generator matrix multiplication actor.
//!
//! The actor owns only kernel selection, bounded lane bookkeeping, and numeric
//! dispatch statistics. Operands and destination storage remain caller-owned;
//! numeric work is handed to synchronous function pointers supplied by the
//! bound runtime.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    clippy::struct_excessive_bools,
    dead_code,
    missing_docs
)]

use core::cell::RefCell;
use core::fmt;

use sml::sml;

/// Maximum number of execution lanes retained by one actor.
pub const MAX_PARALLEL_LANES: usize = 32;

/// Kernel implementation family selected by the caller.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum KernelKind {
    /// Host scalar/fallback implementation.
    #[default]
    X86_64 = 0,
    /// AArch64 implementation family.
    Aarch64 = 1,
}

/// Matrix element family used by parallel row-group selection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum KernelDType {
    /// Generic route.
    #[default]
    F32 = 0,
    /// Q4-K x8 row-group route.
    Q4KX8Bl4 = 1,
    /// Q4-K x8 row-group route.
    Q4KX8Bl8 = 2,
    /// Q6-K x8 row-group route.
    Q6KX8 = 3,
    /// Q6-K x8 prepared route.
    Q6KX8Q8Prepared = 4,
    /// Q6-K x8 prepared argmax route.
    Q6KX8Q8ArgmaxPrepared = 5,
    /// Q8-0 x4 row-group route.
    Q8_0X4Bl4 = 6,
    /// Q8-0 x4 row-group route.
    Q8_0X4Bl8 = 7,
    /// Any route without a specialized row group.
    Other = 8,
}

impl KernelDType {
    const fn uses_x8_rows(self) -> bool {
        matches!(
            self,
            Self::Q4KX8Bl4
                | Self::Q4KX8Bl8
                | Self::Q6KX8
                | Self::Q6KX8Q8Prepared
                | Self::Q6KX8Q8ArgmaxPrepared
        )
    }

    const fn uses_x4_rows(self) -> bool {
        matches!(self, Self::Q8_0X4Bl4 | Self::Q8_0X4Bl8)
    }
}

/// Bounded row range assigned to one numeric callback invocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RowSlice {
    /// First matrix row, inclusive.
    pub row_begin: usize,
    /// Number of rows in this range.
    pub row_count: usize,
}

/// Caller-owned matrix operands and destination.
#[derive(Debug)]
pub struct MatmulRequest<'a> {
    /// Element family used to select a parallel row group.
    pub dtype: KernelDType,
    /// Number of matrix rows.
    pub rows: usize,
    /// Number of right-hand-side columns.
    pub columns: usize,
    /// Caller-owned left-hand-side values.
    pub lhs: &'a [f32],
    /// Caller-owned right-hand-side values.
    pub rhs: &'a [f32],
    /// Caller-owned destination storage.
    pub output: &'a RefCell<&'a mut [f32]>,
    /// Synchronous serial numeric handoff.
    pub serial: Option<SerialKernel>,
    /// Synchronous parallel numeric handoff.
    pub parallel: Option<ParallelKernel>,
}

impl<'a> MatmulRequest<'a> {
    /// Creates a request over caller-owned operands and destination.
    #[must_use]
    pub const fn new(
        dtype: KernelDType,
        rows: usize,
        columns: usize,
        lhs: &'a [f32],
        rhs: &'a [f32],
        output: &'a RefCell<&'a mut [f32]>,
    ) -> Self {
        Self {
            dtype,
            rows,
            columns,
            lhs,
            rhs,
            output,
            serial: None,
            parallel: None,
        }
    }

    /// Installs both synchronous numeric handoffs.
    #[must_use]
    pub const fn with_kernels(
        mut self,
        serial: SerialKernel,
        parallel: ParallelKernel,
    ) -> Self {
        self.serial = Some(serial);
        self.parallel = Some(parallel);
        self
    }
}

/// Synchronous serial matrix callback.
pub type SerialKernel = for<'a> fn(&MatmulRequest<'a>, RowSlice) -> bool;
/// Synchronous parallel matrix callback; the final argument identifies a lane.
pub type ParallelKernel = for<'a> fn(&MatmulRequest<'a>, RowSlice, usize) -> bool;

/// Result fields filled during one serial or parallel dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DispatchResult {
    /// Number of lanes used by the dispatch.
    pub lane_count: usize,
    /// Whether all parallel submissions were accepted.
    pub all_submitted: bool,
    /// Whether synchronous lane joining completed.
    pub joined: bool,
    /// Whether every invoked numeric callback accepted its range.
    pub all_lanes_accepted: bool,
}

/// Runtime counters exposed by state inspection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KernelCounters {
    /// Number of accepted callback invocations.
    pub operations: u64,
    /// Number of rows handed to callbacks.
    pub rows: u64,
}

/// Stable inspection view of the serial kernel.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KernelState {
    /// Configured implementation family.
    pub kind: KernelKind,
    /// Callback invocation counters.
    pub counters: KernelCounters,
}

/// Policy used to configure the bounded actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionPolicy {
    /// Kernel implementation family.
    pub kernel_kind: KernelKind,
    /// Whether parallel lane storage and submission are available.
    pub parallel_available: bool,
    /// Number of active lanes, including the caller lane.
    pub active_lanes: usize,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            kernel_kind: KernelKind::default(),
            parallel_available: false,
            active_lanes: 1,
        }
    }
}

/// Kernel-kind configuration event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventConfigureKernelKind {
    /// Selected implementation family.
    pub kind: KernelKind,
}

impl EventConfigureKernelKind {
    /// Constructs a kernel-kind configuration event.
    #[must_use]
    pub const fn new(kind: KernelKind) -> Self { Self { kind } }
}

impl Default for EventConfigureKernelKind {
    fn default() -> Self { Self { kind: KernelKind::default() } }
}

/// Serial execution event with caller-owned result channels.
#[derive(Debug)]
pub struct EventExecuteSerial<'a> {
    /// Matrix request and numeric callback.
    pub request: MatmulRequest<'a>,
    /// Result written synchronously by the state machine.
    pub result: &'a RefCell<DispatchResult>,
    /// Acceptance written synchronously by the state machine.
    pub accepted: &'a RefCell<bool>,
}

/// Parallel execution event with caller-owned result channels.
#[derive(Debug)]
pub struct EventExecuteParallel<'a> {
    /// Matrix request and numeric callback.
    pub request: MatmulRequest<'a>,
    /// Result written synchronously by the state machine.
    pub result: &'a RefCell<DispatchResult>,
    /// Acceptance written synchronously by the state machine.
    pub accepted: &'a RefCell<bool>,
}

sml! {
    TextGeneratorMatmul<'dispatch> {
        "state_ready"_s <= *"state_ready"_s + event<EventConfigureKernelKind> / effect_configure_kernel_kind,
        "state_serial_result_decision"_s <= "state_ready"_s + event<EventExecuteSerial<'dispatch>> / effect_execute_serial,
        "state_ready"_s <= "state_serial_result_decision"_s + completion<EventExecuteSerial>(EventExecuteSerial<'dispatch>) [guard_serial_accepted] / effect_accept_serial_execution,
        "state_ready"_s <= "state_serial_result_decision"_s + completion<EventExecuteSerial>(EventExecuteSerial<'dispatch>) [guard_serial_rejected] / effect_reject_serial_execution,
        "state_parallel_result_decision"_s <= "state_ready"_s + event<EventExecuteParallel<'dispatch>> [guard_parallel_x8_ready] / effect_execute_parallel_x8,
        "state_parallel_result_decision"_s <= "state_ready"_s + event<EventExecuteParallel<'dispatch>> [guard_parallel_x4_ready] / effect_execute_parallel_x4,
        "state_parallel_result_decision"_s <= "state_ready"_s + event<EventExecuteParallel<'dispatch>> [guard_parallel_unit_ready] / effect_execute_parallel_unit,
        "state_ready"_s <= "state_ready"_s + event<EventExecuteParallel<'dispatch>> [guard_parallel_unavailable] / effect_reject_parallel_execution_from_state_ready,
        "state_ready"_s <= "state_parallel_result_decision"_s + completion<EventExecuteParallel>(EventExecuteParallel<'dispatch>) [guard_parallel_submission_failed] / effect_reject_parallel_execution_from_state_parallel_result_decision,
        "state_ready"_s <= "state_parallel_result_decision"_s + completion<EventExecuteParallel>(EventExecuteParallel<'dispatch>) [guard_parallel_join_failed] / effect_reject_parallel_execution_from_state_parallel_result_decision,
        "state_ready"_s <= "state_parallel_result_decision"_s + completion<EventExecuteParallel>(EventExecuteParallel<'dispatch>) [guard_parallel_lane_rejected] / effect_reject_parallel_execution_from_state_parallel_result_decision,
        "state_ready"_s <= "state_parallel_result_decision"_s + completion<EventExecuteParallel>(EventExecuteParallel<'dispatch>) [guard_parallel_all_lanes_accepted] / effect_accept_parallel_execution,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_serial_result_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_serial_result_decision,
        "state_ready"_s <= "state_parallel_result_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_parallel_result_decision,
    }
}

/// Mutable control-plane state for [`TextGeneratorMatmulActor`].
#[derive(Debug)]
pub struct TextGeneratorMatmulContext {
    kernel_kind: KernelKind,
    parallel_available: bool,
    active_lanes: usize,
    serial_kernel: KernelState,
    lane_kernels: [KernelState; MAX_PARALLEL_LANES],
}

impl Default for TextGeneratorMatmulContext {
    fn default() -> Self { Self::new(ExecutionPolicy::default()) }
}

impl TextGeneratorMatmulContext {
    /// Constructs context from an explicit execution policy.
    #[must_use]
    pub fn new(policy: ExecutionPolicy) -> Self {
        let mut context = Self {
            kernel_kind: policy.kernel_kind,
            parallel_available: policy.parallel_available,
            active_lanes: policy.active_lanes.min(MAX_PARALLEL_LANES).max(1),
            serial_kernel: KernelState { kind: policy.kernel_kind, counters: KernelCounters::default() },
            lane_kernels: [KernelState::default(); MAX_PARALLEL_LANES],
        };
        for lane in &mut context.lane_kernels {
            lane.kind = policy.kernel_kind;
        }
        context
    }

    /// Returns the configured kernel family.
    #[must_use]
    pub const fn kernel_kind(&self) -> KernelKind { self.kernel_kind }
    /// Returns whether parallel execution can be accepted.
    #[must_use]
    pub const fn parallel_available(&self) -> bool { self.parallel_available && self.active_lanes > 1 }
    /// Returns the configured active lane count.
    #[must_use]
    pub const fn active_lane_count(&self) -> usize { self.active_lanes }
    /// Returns the serial kernel inspection state.
    #[must_use]
    pub const fn serial_kernel(&self) -> &KernelState { &self.serial_kernel }
    /// Returns bounded lane kernel inspection states.
    #[must_use]
    pub fn parallel_lane_kernels(&self) -> &[KernelState] { &self.lane_kernels[..self.active_lanes] }

    fn configure(&mut self, kind: KernelKind) {
        self.kernel_kind = kind;
        self.serial_kernel.kind = kind;
        for lane in &mut self.lane_kernels {
            lane.kind = kind;
        }
    }

    fn parallel_ready(&self, event: &EventExecuteParallel<'_>) -> bool {
        self.parallel_available()
            && event.request.rows > 0
            && event.request.parallel.is_some()
    }

    fn dispatch_parallel(&mut self, event: &EventExecuteParallel<'_>, group_rows: usize) {
        let group_rows = group_rows.max(1);
        let groups = event.request.rows.div_ceil(group_rows);
        let lane_count = self.active_lanes.min(groups.max(1)).min(MAX_PARALLEL_LANES);
        let callback = event.request.parallel;
        let mut result = DispatchResult {
            lane_count,
            all_submitted: callback.is_some(),
            joined: true,
            all_lanes_accepted: callback.is_some(),
        };
        if let Some(callback) = callback {
            let groups = event.request.rows.div_ceil(group_rows.max(1));
            let groups_per_lane = groups / lane_count.max(1);
            let extra_groups = groups % lane_count.max(1);
            let mut begin_group = 0usize;
            for lane in 0..lane_count {
                let lane_groups = groups_per_lane + usize::from(lane < extra_groups);
                let row_begin = begin_group * group_rows.max(1);
                let row_end = (begin_group + lane_groups) * group_rows.max(1);
                let slice = RowSlice { row_begin, row_count: row_end.min(event.request.rows) - row_begin };
                let accepted = callback(&event.request, slice, lane);
                result.all_lanes_accepted &= accepted;
                if accepted {
                    self.lane_kernels[lane].counters.operations = self.lane_kernels[lane].counters.operations.saturating_add(1);
                    self.lane_kernels[lane].counters.rows = self.lane_kernels[lane].counters.rows.saturating_add(slice.row_count as u64);
                }
                begin_group += lane_groups;
            }
        }
        *event.result.borrow_mut() = result;
    }

    fn unexpected(&mut self) -> Result<(), ()> { Ok(()) }
}

impl TextGeneratorMatmulStateMachineContext for TextGeneratorMatmulContext {
    fn effect_accept_parallel_execution(&mut self, event: &EventExecuteParallel<'_>) -> Result<(), ()> {
        *event.accepted.borrow_mut() = true;
        Ok(())
    }
    fn effect_accept_serial_execution(&mut self, event: &EventExecuteSerial<'_>) -> Result<(), ()> {
        *event.accepted.borrow_mut() = true;
        Ok(())
    }
    fn effect_configure_kernel_kind(&mut self, event: &EventConfigureKernelKind) -> Result<(), ()> {
        self.configure(event.kind);
        Ok(())
    }
    fn effect_execute_parallel_unit(&mut self, event: &EventExecuteParallel<'_>) -> Result<(), ()> {
        self.dispatch_parallel(event, 1);
        Ok(())
    }
    fn effect_execute_parallel_x4(&mut self, event: &EventExecuteParallel<'_>) -> Result<(), ()> {
        self.dispatch_parallel(event, 4);
        Ok(())
    }
    fn effect_execute_parallel_x8(&mut self, event: &EventExecuteParallel<'_>) -> Result<(), ()> {
        self.dispatch_parallel(event, 8);
        Ok(())
    }
    fn effect_execute_serial(&mut self, event: &EventExecuteSerial<'_>) -> Result<(), ()> {
        let accepted = event.request.serial.is_some()
            && (event.request.serial.unwrap())(&event.request, RowSlice { row_begin: 0, row_count: event.request.rows });
        *event.result.borrow_mut() = DispatchResult {
            lane_count: 1,
            all_submitted: true,
            joined: true,
            all_lanes_accepted: accepted,
        };
        if accepted {
            self.serial_kernel.counters.operations = self.serial_kernel.counters.operations.saturating_add(1);
            self.serial_kernel.counters.rows = self.serial_kernel.counters.rows.saturating_add(event.request.rows as u64);
        }
        Ok(())
    }
    fn effect_on_unexpected_from_state_parallel_result_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_serial_result_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_reject_parallel_execution_from_state_parallel_result_decision(&mut self, event: &EventExecuteParallel<'_>) -> Result<(), ()> {
        *event.accepted.borrow_mut() = false;
        Ok(())
    }
    fn effect_reject_parallel_execution_from_state_ready(&mut self, event: &EventExecuteParallel<'_>) -> Result<(), ()> {
        *event.accepted.borrow_mut() = false;
        *event.result.borrow_mut() = DispatchResult::default();
        Ok(())
    }
    fn effect_reject_serial_execution(&mut self, event: &EventExecuteSerial<'_>) -> Result<(), ()> {
        *event.accepted.borrow_mut() = false;
        Ok(())
    }
    fn guard_parallel_all_lanes_accepted(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> { let result = *event.result.borrow(); Ok(result.all_submitted && result.joined && result.all_lanes_accepted) }
    fn guard_parallel_join_failed(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> { let result = *event.result.borrow(); Ok(result.all_submitted && !result.joined) }
    fn guard_parallel_lane_rejected(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> { let result = *event.result.borrow(); Ok(result.all_submitted && result.joined && !result.all_lanes_accepted) }
    fn guard_parallel_submission_failed(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> { Ok(!event.result.borrow().all_submitted) }
    fn guard_parallel_unavailable(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> { Ok(!self.parallel_ready(event)) }
    fn guard_parallel_unit_ready(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> { Ok(self.parallel_ready(event) && !event.request.dtype.uses_x8_rows() && !event.request.dtype.uses_x4_rows()) }
    fn guard_parallel_x4_ready(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> { Ok(self.parallel_ready(event) && event.request.dtype.uses_x4_rows()) }
    fn guard_parallel_x8_ready(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> { Ok(self.parallel_ready(event) && event.request.dtype.uses_x8_rows()) }
    fn guard_serial_accepted(&self, event: &EventExecuteSerial<'_>) -> Result<bool, ()> { Ok(event.result.borrow().all_lanes_accepted) }
    fn guard_serial_rejected(&self, event: &EventExecuteSerial<'_>) -> Result<bool, ()> { Ok(!event.result.borrow().all_lanes_accepted) }
}

/// Single-writer, run-to-completion matrix multiplication actor.
pub struct TextGeneratorMatmulActor {
    machine: TextGeneratorMatmulStateMachine<TextGeneratorMatmulContext>,
}

impl fmt::Debug for TextGeneratorMatmulActor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("TextGeneratorMatmulActor").finish_non_exhaustive()
    }
}

impl Default for TextGeneratorMatmulActor {
    fn default() -> Self { Self::new(ExecutionPolicy::default()) }
}

impl TextGeneratorMatmulActor {
    /// Constructs an actor with explicit kernel and lane configuration.
    #[must_use]
    pub fn new(policy: ExecutionPolicy) -> Self {
        let mut actor = Self { machine: TextGeneratorMatmulStateMachine::new(TextGeneratorMatmulContext::new(policy)) };
        let _ = actor.process_configure_kernel_kind(EventConfigureKernelKind { kind: policy.kernel_kind });
        actor
    }

    /// Processes kernel configuration synchronously.
    pub fn process_configure_kernel_kind(&mut self, event: EventConfigureKernelKind) -> Result<(), ()> {
        self.machine.process_event(TextGeneratorMatmulEvents::EventConfigureKernelKind(event)).map(|_| ())
    }
    /// Processes a serial matrix request synchronously.
    pub fn process_serial<'a>(&mut self, event: EventExecuteSerial<'a>) -> Result<(), ()> {
        self.machine.process_event(TextGeneratorMatmulEvents::EventExecuteSerial(event)).map(|_| ())
    }
    /// Processes a parallel matrix request synchronously.
    pub fn process_parallel<'a>(&mut self, event: EventExecuteParallel<'a>) -> Result<(), ()> {
        self.machine.process_event(TextGeneratorMatmulEvents::EventExecuteParallel(event)).map(|_| ())
    }
    /// Processes an unexpected event and returns to the ready state.
    pub fn process_unexpected_event(&mut self) -> Result<(), ()> {
        self.machine.process_event(TextGeneratorMatmulEvents::UnexpectedEvent).map(|_| ())
    }
    /// Returns the generated machine state.
    #[must_use]
    pub fn state(&self) -> &TextGeneratorMatmulStates { self.machine.state() }
    /// Tests generated machine state identity.
    #[must_use]
    pub fn is(&self, state: &TextGeneratorMatmulStates) -> bool { self.machine.is(state) }
    /// Returns immutable actor context.
    #[must_use]
    pub fn context(&self) -> &TextGeneratorMatmulContext { self.machine.context() }
    /// Returns mutable actor context.
    pub fn context_mut(&mut self) -> &mut TextGeneratorMatmulContext { self.machine.context_mut() }
    /// Computes a counter total across serial and active lane kernels.
    #[must_use]
    pub fn kernel_counter_total<F>(&self, mut counter: F) -> u64
    where
        F: FnMut(KernelCounters) -> u64,
    {
        let mut total = counter(self.context().serial_kernel.counters);
        for kernel in self.context().parallel_lane_kernels() {
            total = total.saturating_add(counter(kernel.counters));
        }
        total
    }
    /// Returns whether parallel lane execution is currently available.
    #[must_use]
    pub fn parallel_lanes_available(&self) -> bool { self.context().parallel_available() }
    /// Returns the configured number of active lanes.
    #[must_use]
    pub fn active_lane_count(&self) -> usize { self.context().active_lane_count() }
    /// Returns serial kernel inspection state.
    #[must_use]
    pub fn serial_kernel(&self) -> &KernelState { self.context().serial_kernel() }
    /// Returns bounded parallel kernel inspection states.
    #[must_use]
    pub fn parallel_lane_kernels(&self) -> &[KernelState] { self.context().parallel_lane_kernels() }
    /// Source-shaped configuration wrapper.
    pub fn process_event_configure_kernel_kind(&mut self, event: EventConfigureKernelKind) -> Result<(), ()> {
        self.process_configure_kernel_kind(event)
    }
    /// Source-shaped serial execution wrapper.
    pub fn process_event_execute_serial<'a>(&mut self, event: EventExecuteSerial<'a>) -> Result<(), ()> {
        self.process_serial(event)
    }
    /// Source-shaped parallel execution wrapper.
    pub fn process_event_execute_parallel<'a>(&mut self, event: EventExecuteParallel<'a>) -> Result<(), ()> {
        self.process_parallel(event)
    }
}
