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
    clippy::elidable_lifetime_names,
    clippy::manual_clamp,
    clippy::manual_is_multiple_of,
    clippy::unused_self,
    clippy::field_reassign_with_default,
    clippy::derivable_impls,
    clippy::enum_variant_names,
    clippy::doc_markdown,
    clippy::needless_lifetimes,
    dead_code,
    missing_docs
)]

use core::cell::RefCell;
use core::fmt;

use sml::sml;

/// Maximum number of execution lanes retained by one actor.
///
/// The pinned C++ matmul actor has eight statically materialized lane kernels;
/// parallel policy selection is bounded by the active lane count and row groups.
pub const MAX_PARALLEL_LANES: usize = 8;

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

const fn supported_lane_count(active_lanes: usize) -> bool {
    matches!(active_lanes, 2 | 4 | 8)
}
/// Typed failure outcome for one synchronous dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum DispatchError {
    /// The dispatch completed successfully.
    #[default]
    None = 0,
    /// Operands, dimensions, or required callback were invalid.
    InvalidRequest = 1,
    /// The configured lane pool cannot execute parallel work.
    ParallelUnavailable = 2,
    /// One or more worker-lane submissions failed.
    SubmissionFailed = 3,
    /// The submitted synchronous lane group did not join.
    JoinFailed = 4,
    /// A numeric callback rejected its assigned row range.
    LaneRejected = 5,
    /// An event was not valid in the current state.
    Unexpected = 6,
}

impl DispatchError {
    /// Returns the stable numeric error code.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
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

impl RowSlice {
    /// Returns whether this range is wholly contained in a matrix row extent.
    #[must_use]
    pub const fn valid_for(self, rows: usize) -> bool {
        self.row_begin <= rows && self.row_count <= rows.saturating_sub(self.row_begin)
    }
}

/// Caller-owned matrix operands and destination.
#[derive(Clone, Copy, Debug)]
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
    pub const fn with_kernels(mut self, serial: SerialKernel, parallel: ParallelKernel) -> Self {
        self.serial = Some(serial);
        self.parallel = Some(parallel);
        self
    }

    fn valid(&self) -> bool {
        let Some(lhs_elements) = self.rows.checked_mul(self.columns) else {
            return false;
        };
        self.rows > 0
            && self.columns > 0
            && self.lhs.len() >= lhs_elements
            && self.rhs.len() >= self.columns
            && self.output.borrow().len() >= self.rows
    }
}

/// Synchronous serial matrix callback.
pub type SerialKernel = for<'a> fn(&MatmulRequest<'a>, RowSlice) -> bool;
/// Synchronous parallel matrix callback; the final argument identifies a lane.
pub type ParallelKernel = for<'a> fn(&MatmulRequest<'a>, RowSlice, usize) -> bool;

/// Synchronous submission callback supplied by a caller-owned lane pool.
///
/// The pool must not retain any borrowed input after returning. A successful
/// submission means the task was accepted by the pool; the callback's result
/// is written to `accepted` before the enclosing dispatch joins.
pub type LaneSubmit =
    for<'a> fn(*mut (), usize, &MatmulRequest<'a>, RowSlice, ParallelKernel, &mut bool) -> bool;

/// Synchronous join callback supplied by a caller-owned lane pool.
pub type LaneJoin = fn(*mut ()) -> bool;

/// Caller-owned synchronous lane-pool capability.
#[derive(Clone, Copy, Debug, Default)]
pub struct LanePool {
    /// Opaque caller-owned pool state.
    pub context: *mut (),
    /// Submits one worker-lane task without retaining borrowed inputs.
    pub submit: Option<LaneSubmit>,
    /// Joins all tasks submitted for the current dispatch.
    pub join: Option<LaneJoin>,
}

impl LanePool {
    /// Creates an inline synchronous pool.
    #[must_use]
    pub const fn inline() -> Self {
        Self {
            context: core::ptr::null_mut(),
            submit: Some(inline_lane_submit),
            join: Some(inline_lane_join),
        }
    }

    /// Creates a pool backed by caller-owned synchronous callbacks.
    #[must_use]
    pub const fn with_callbacks(context: *mut (), submit: LaneSubmit, join: LaneJoin) -> Self {
        Self {
            context,
            submit: Some(submit),
            join: Some(join),
        }
    }

    const fn valid(self) -> bool {
        self.submit.is_some() && self.join.is_some()
    }

    fn submit_lane(
        self,
        lane: usize,
        request: &MatmulRequest<'_>,
        slice: RowSlice,
        callback: ParallelKernel,
        accepted: &mut bool,
    ) -> bool {
        self.submit
            .is_some_and(|submit| submit(self.context, lane, request, slice, callback, accepted))
    }

    fn join(self) -> bool {
        self.join.is_some_and(|join| join(self.context))
    }
}

fn inline_lane_submit(
    _context: *mut (),
    lane: usize,
    request: &MatmulRequest<'_>,
    slice: RowSlice,
    callback: ParallelKernel,
    accepted: &mut bool,
) -> bool {
    *accepted = callback(request, slice, lane);
    true
}

fn inline_lane_join(_context: *mut ()) -> bool {
    true
}

/// Result fields filled during one serial or parallel dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DispatchResult {
    /// Typed dispatch outcome.
    pub error: DispatchError,
    /// Number of lanes used by the dispatch.
    pub lane_count: usize,
    /// Whether all parallel submissions were accepted.
    pub all_submitted: bool,
    /// Whether synchronous lane joining completed.
    pub joined: bool,
    /// Whether every invoked numeric callback accepted its row range.
    pub all_lanes_accepted: bool,
}

/// Bounded diagnostics projection corresponding to the pinned
/// `capture_diagnostics` event. Counters are retained in the actor and copied
/// to caller-owned storage without allocating or dispatching another event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MatmulDiagnostics {
    pub serial: KernelCounters,
    pub parallel: KernelCounters,
}
/// Event-only diagnostics capture with caller-owned output and acceptance.
#[derive(Clone, Copy, Debug)]
pub struct EventCaptureDiagnostics<'a> {
    pub out: &'a RefCell<MatmulDiagnostics>,
    pub accepted: &'a RefCell<bool>,
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
#[derive(Clone, Copy, Debug)]
pub struct ExecutionPolicy {
    /// Kernel implementation family.
    pub kernel_kind: KernelKind,
    /// Whether parallel lane storage and submission are available.
    pub parallel_available: bool,
    /// Number of active lanes, including the caller lane.
    pub active_lanes: usize,
    /// Caller-owned synchronous lane-pool capability.
    pub lane_pool: Option<LanePool>,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            kernel_kind: KernelKind::default(),
            parallel_available: false,
            active_lanes: 1,
            lane_pool: None,
        }
    }
}

impl ExecutionPolicy {
    /// Returns this policy with a caller-owned synchronous lane pool.
    #[must_use]
    pub const fn with_lane_pool(mut self, lane_pool: LanePool) -> Self {
        self.parallel_available = true;
        self.lane_pool = Some(lane_pool);
        self
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
    pub const fn new(kind: KernelKind) -> Self {
        Self { kind }
    }
}

impl Default for EventConfigureKernelKind {
    fn default() -> Self {
        Self {
            kind: KernelKind::default(),
        }
    }
}

/// Completion callback invoked synchronously after an accepted dispatch.
pub type DoneCallback = fn(&DispatchResult);
/// Error callback invoked synchronously after a rejected dispatch.
pub type ErrorCallback = fn(&DispatchResult);

/// Serial execution event with caller-owned result channels.
#[derive(Clone, Copy, Debug)]
pub struct EventExecuteSerial<'a> {
    pub request: MatmulRequest<'a>,
    pub result: &'a RefCell<DispatchResult>,
    pub accepted: &'a RefCell<bool>,
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
}
impl<'a> EventExecuteSerial<'a> {
    #[must_use]
    pub const fn with_callbacks(
        mut self,
        on_done: Option<DoneCallback>,
        on_error: Option<ErrorCallback>,
    ) -> Self {
        self.on_done = on_done;
        self.on_error = on_error;
        self
    }
}

/// Parallel execution event with caller-owned result channels.
#[derive(Clone, Copy, Debug)]
pub struct EventExecuteParallel<'a> {
    pub request: MatmulRequest<'a>,
    pub result: &'a RefCell<DispatchResult>,
    pub accepted: &'a RefCell<bool>,
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
}

impl<'a> EventExecuteParallel<'a> {
    #[must_use]
    pub const fn with_callbacks(
        mut self,
        on_done: Option<DoneCallback>,
        on_error: Option<ErrorCallback>,
    ) -> Self {
        self.on_done = on_done;
        self.on_error = on_error;
        self
    }
}
sml! {
    TextGeneratorMatmul<'dispatch> {
        "state_ready"_s <= *"state_ready"_s + event<EventConfigureKernelKind> / effect_configure_kernel_kind,
        "state_serial_result_decision"_s <= "state_ready"_s + event<EventExecuteSerial<'dispatch>> / effect_execute_serial,
        "state_done_callback_decision"_s <= "state_serial_result_decision"_s + completion<EventExecuteSerial>(EventExecuteSerial<'dispatch>) [guard_serial_accepted] / effect_accept_serial_execution,
        "state_error_callback_decision"_s <= "state_serial_result_decision"_s + completion<EventExecuteSerial>(EventExecuteSerial<'dispatch>) [guard_serial_rejected] / effect_reject_serial_execution,
        "state_ready"_s <= "state_done_callback_decision"_s + completion<EventExecuteSerial>(EventExecuteSerial<'dispatch>) [guard_serial_has_done_callback] / effect_emit_serial_done,
        "state_ready"_s <= "state_done_callback_decision"_s + completion<EventExecuteSerial>(EventExecuteSerial<'dispatch>) [guard_serial_no_done_callback],
        "state_ready"_s <= "state_error_callback_decision"_s + completion<EventExecuteSerial>(EventExecuteSerial<'dispatch>) [guard_serial_has_error_callback] / effect_emit_serial_error,
        "state_ready"_s <= "state_error_callback_decision"_s + completion<EventExecuteSerial>(EventExecuteSerial<'dispatch>) [guard_serial_no_error_callback],
        "state_parallel_result_decision"_s <= "state_ready"_s + event<EventExecuteParallel<'dispatch>> [guard_parallel_x8_ready] / effect_execute_parallel_x8,
        "state_parallel_result_decision"_s <= "state_ready"_s + event<EventExecuteParallel<'dispatch>> [guard_parallel_x4_ready] / effect_execute_parallel_x4,
        "state_parallel_result_decision"_s <= "state_ready"_s + event<EventExecuteParallel<'dispatch>> [guard_parallel_unit_ready] / effect_execute_parallel_unit,
        "state_error_callback_decision"_s <= "state_ready"_s + event<EventExecuteParallel<'dispatch>> [guard_parallel_unavailable] / effect_reject_parallel_execution_from_state_ready,
        "state_error_callback_decision"_s <= "state_parallel_result_decision"_s + completion<EventExecuteParallel>(EventExecuteParallel<'dispatch>) [guard_parallel_submission_failed] / effect_reject_parallel_execution_from_state_parallel_result_decision,
        "state_error_callback_decision"_s <= "state_parallel_result_decision"_s + completion<EventExecuteParallel>(EventExecuteParallel<'dispatch>) [guard_parallel_join_failed] / effect_reject_parallel_execution_from_state_parallel_result_decision,
        "state_error_callback_decision"_s <= "state_parallel_result_decision"_s + completion<EventExecuteParallel>(EventExecuteParallel<'dispatch>) [guard_parallel_lane_rejected] / effect_reject_parallel_execution_from_state_parallel_result_decision,
        "state_done_callback_decision"_s <= "state_parallel_result_decision"_s + completion<EventExecuteParallel>(EventExecuteParallel<'dispatch>) [guard_parallel_all_lanes_accepted] / effect_accept_parallel_execution,
        "state_ready"_s <= "state_done_callback_decision"_s + completion<EventExecuteParallel>(EventExecuteParallel<'dispatch>) [guard_parallel_has_done_callback] / effect_emit_parallel_done,
        "state_ready"_s <= "state_done_callback_decision"_s + completion<EventExecuteParallel>(EventExecuteParallel<'dispatch>) [guard_parallel_no_done_callback],
        "state_ready"_s <= "state_error_callback_decision"_s + completion<EventExecuteParallel>(EventExecuteParallel<'dispatch>) [guard_parallel_has_error_callback] / effect_emit_parallel_error,
        "state_ready"_s <= "state_error_callback_decision"_s + completion<EventExecuteParallel>(EventExecuteParallel<'dispatch>) [guard_parallel_no_error_callback],
        "state_ready"_s <= "state_ready"_s + event<EventCaptureDiagnostics<'dispatch>> / effect_capture_diagnostics,
        "state_ready"_s <= "state_serial_result_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_serial_result_decision,
        "state_ready"_s <= "state_parallel_result_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_parallel_result_decision,
        "state_ready"_s <= "state_done_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_done_callback_decision,
        "state_ready"_s <= "state_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_callback_decision,
    }
}

/// Mutable control-plane state for [`TextGeneratorMatmulActor`].
#[derive(Debug)]
pub struct TextGeneratorMatmulContext {
    kernel_kind: KernelKind,
    parallel_available: bool,
    active_lanes: usize,
    lane_pool: Option<LanePool>,
    serial_kernel: KernelState,
    lane_kernels: [KernelState; MAX_PARALLEL_LANES],
    unexpected: bool,
}

impl Default for TextGeneratorMatmulContext {
    fn default() -> Self {
        Self::new(ExecutionPolicy::default())
    }
}

impl TextGeneratorMatmulContext {
    #[must_use]
    pub fn new(policy: ExecutionPolicy) -> Self {
        let mut context = Self {
            kernel_kind: policy.kernel_kind,
            parallel_available: policy.parallel_available,
            active_lanes: policy.active_lanes.min(MAX_PARALLEL_LANES).max(1),
            lane_pool: policy.lane_pool,
            serial_kernel: KernelState {
                kind: policy.kernel_kind,
                counters: KernelCounters::default(),
            },
            lane_kernels: [KernelState::default(); MAX_PARALLEL_LANES],
            unexpected: false,
        };
        for lane in &mut context.lane_kernels {
            lane.kind = policy.kernel_kind;
        }
        context
    }
    /// Applies a selected kernel family to the control-plane and bounded lane state.
    pub fn configure(&mut self, kind: KernelKind) {
        self.kernel_kind = kind;
        self.serial_kernel.kind = kind;
        for lane in &mut self.lane_kernels {
            lane.kind = kind;
        }
    }
    #[must_use]
    pub const fn kernel_kind(&self) -> KernelKind {
        self.kernel_kind
    }
    #[must_use]
    pub const fn parallel_available(&self) -> bool {
        self.parallel_available
            && supported_lane_count(self.active_lanes)
            && match self.lane_pool {
                Some(pool) => pool.valid(),
                None => false,
            }
    }
    #[must_use]
    pub const fn active_lane_count(&self) -> usize {
        self.active_lanes
    }
    #[must_use]
    pub const fn serial_kernel(&self) -> &KernelState {
        &self.serial_kernel
    }
    #[must_use]
    pub fn parallel_lane_kernels(&self) -> &[KernelState] {
        &self.lane_kernels[..self.active_lanes]
    }
    #[must_use]
    fn selected_lane_count(&self, groups: usize) -> usize {
        self.active_lanes
            .max(1)
            .min(MAX_PARALLEL_LANES)
            .min(groups.max(1))
    }
    fn parallel_ready(&self, event: &EventExecuteParallel<'_>) -> bool {
        self.parallel_available() && event.request.valid() && event.request.parallel.is_some()
    }
    fn parallel_group_count(&self, event: &EventExecuteParallel<'_>) -> usize {
        let group_rows = if event.request.dtype.uses_x8_rows() {
            8
        } else if event.request.dtype.uses_x4_rows() {
            4
        } else {
            1
        };
        event.request.rows / group_rows + usize::from(event.request.rows % group_rows != 0)
    }
    pub const fn lane_pool(&self) -> Option<LanePool> {
        self.lane_pool
    }
    /// Returns whether an unexpected event has been observed.
    #[must_use]
    pub const fn unexpected(&self) -> bool {
        self.unexpected
    }
    fn dispatch_parallel(&mut self, event: &EventExecuteParallel<'_>, group_rows: usize) {
        let mut result = DispatchResult {
            error: DispatchError::None,
            ..DispatchResult::default()
        };
        if !event.request.valid() {
            result.error = DispatchError::InvalidRequest;
            *event.result.borrow_mut() = result;
            *event.accepted.borrow_mut() = false;
            return;
        }
        let Some(callback) = event.request.parallel else {
            result.error = DispatchError::InvalidRequest;
            *event.result.borrow_mut() = result;
            *event.accepted.borrow_mut() = false;
            return;
        };
        let Some(pool) = self.lane_pool.filter(|pool| pool.valid()) else {
            result.error = DispatchError::ParallelUnavailable;
            *event.result.borrow_mut() = result;
            *event.accepted.borrow_mut() = false;
            return;
        };
        let groups =
            event.request.rows / group_rows + usize::from(event.request.rows % group_rows != 0);
        let lane_count = self
            .selected_lane_count(groups)
            .min(groups)
            .min(MAX_PARALLEL_LANES);
        result.lane_count = lane_count;
        let groups_per_lane = groups / lane_count;
        let extra_groups = groups % lane_count;
        let mut accepted = [false; MAX_PARALLEL_LANES];
        let mut all_submitted = true;
        for lane in 0..lane_count {
            let lane_groups = groups_per_lane + usize::from(lane < extra_groups);
            let begin_group = lane * groups_per_lane + lane.min(extra_groups);
            let row_begin = begin_group
                .saturating_mul(group_rows)
                .min(event.request.rows);
            let row_end = begin_group
                .saturating_add(lane_groups)
                .saturating_mul(group_rows)
                .min(event.request.rows);
            let slice = RowSlice {
                row_begin,
                row_count: row_end.saturating_sub(row_begin),
            };
            if !slice.valid_for(event.request.rows) || slice.row_count == 0 {
                all_submitted = false;
                continue;
            }
            if lane == 0 {
                accepted[0] = callback(&event.request, slice, 0);
            } else {
                all_submitted &=
                    pool.submit_lane(lane, &event.request, slice, callback, &mut accepted[lane]);
            }
            if accepted[lane] {
                self.lane_kernels[lane].counters.operations = self.lane_kernels[lane]
                    .counters
                    .operations
                    .saturating_add(1);
                self.lane_kernels[lane].counters.rows = self.lane_kernels[lane]
                    .counters
                    .rows
                    .saturating_add(slice.row_count as u64);
            }
        }
        result.all_submitted = all_submitted;
        result.joined = pool.join();
        result.all_lanes_accepted = accepted[..lane_count].iter().all(|accepted| *accepted);
        result.error = if !result.all_submitted {
            DispatchError::SubmissionFailed
        } else if !result.joined {
            DispatchError::JoinFailed
        } else if !result.all_lanes_accepted {
            DispatchError::LaneRejected
        } else {
            DispatchError::None
        };
        *event.result.borrow_mut() = result;
        *event.accepted.borrow_mut() = result.error == DispatchError::None;
    }
    fn unexpected_event(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
}
impl TextGeneratorMatmulStateMachineContext for TextGeneratorMatmulContext {
    fn effect_accept_parallel_execution(
        &mut self,
        event: &EventExecuteParallel<'_>,
    ) -> Result<(), ()> {
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
        let mut result = DispatchResult {
            lane_count: 1,
            ..DispatchResult::default()
        };
        let valid = event.request.valid();
        let accepted = if !valid {
            result.error = DispatchError::InvalidRequest;
            false
        } else if let Some(serial) = event.request.serial {
            result.all_submitted = true;
            result.joined = true;
            let accepted = serial(
                &event.request,
                RowSlice {
                    row_begin: 0,
                    row_count: event.request.rows,
                },
            );
            result.all_lanes_accepted = accepted;
            if accepted {
                self.serial_kernel.counters.operations =
                    self.serial_kernel.counters.operations.saturating_add(1);
                self.serial_kernel.counters.rows = self
                    .serial_kernel
                    .counters
                    .rows
                    .saturating_add(event.request.rows as u64);
            } else {
                result.error = DispatchError::LaneRejected;
            }
            accepted
        } else {
            result.error = DispatchError::InvalidRequest;
            false
        };
        *event.accepted.borrow_mut() = accepted;
        *event.result.borrow_mut() = result;
        Ok(())
    }
    fn effect_emit_serial_done(&mut self, event: &EventExecuteSerial<'_>) -> Result<(), ()> {
        let result = *event.result.borrow();
        if let Some(callback) = event.on_done {
            callback(&result);
        }
        Ok(())
    }
    fn effect_emit_serial_error(&mut self, event: &EventExecuteSerial<'_>) -> Result<(), ()> {
        let result = *event.result.borrow();
        if let Some(callback) = event.on_error {
            callback(&result);
        }
        Ok(())
    }
    fn effect_emit_parallel_done(&mut self, event: &EventExecuteParallel<'_>) -> Result<(), ()> {
        let result = *event.result.borrow();
        if let Some(callback) = event.on_done {
            callback(&result);
        }
        Ok(())
    }
    fn effect_emit_parallel_error(&mut self, event: &EventExecuteParallel<'_>) -> Result<(), ()> {
        let result = *event.result.borrow();
        if let Some(callback) = event.on_error {
            callback(&result);
        }
        Ok(())
    }
    fn effect_on_unexpected_from_state_done_callback_decision(&mut self) -> Result<(), ()> {
        self.unexpected_event()
    }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> {
        self.unexpected_event()
    }
    fn guard_serial_has_done_callback(&self, event: &EventExecuteSerial<'_>) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }
    fn guard_serial_no_done_callback(&self, event: &EventExecuteSerial<'_>) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }
    fn guard_serial_has_error_callback(&self, event: &EventExecuteSerial<'_>) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }
    fn guard_serial_no_error_callback(&self, event: &EventExecuteSerial<'_>) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }
    fn guard_parallel_has_done_callback(
        &self,
        event: &EventExecuteParallel<'_>,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }
    fn guard_parallel_no_done_callback(
        &self,
        event: &EventExecuteParallel<'_>,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }
    fn guard_parallel_has_error_callback(
        &self,
        event: &EventExecuteParallel<'_>,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }
    fn guard_parallel_no_error_callback(
        &self,
        event: &EventExecuteParallel<'_>,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }

    fn effect_on_unexpected_from_state_parallel_result_decision(&mut self) -> Result<(), ()> {
        self.unexpected_event()
    }
    fn effect_on_unexpected_from_state_serial_result_decision(&mut self) -> Result<(), ()> {
        self.unexpected_event()
    }
    fn effect_reject_parallel_execution_from_state_parallel_result_decision(
        &mut self,
        event: &EventExecuteParallel<'_>,
    ) -> Result<(), ()> {
        *event.accepted.borrow_mut() = false;
        Ok(())
    }
    fn effect_reject_parallel_execution_from_state_ready(
        &mut self,
        event: &EventExecuteParallel<'_>,
    ) -> Result<(), ()> {
        *event.accepted.borrow_mut() = false;
        let mut result = *event.result.borrow();
        result.error = if !event.request.valid() || event.request.parallel.is_none() {
            DispatchError::InvalidRequest
        } else {
            DispatchError::ParallelUnavailable
        };
        result.all_submitted = false;
        result.joined = false;
        result.all_lanes_accepted = false;
        *event.result.borrow_mut() = result;
        Ok(())
    }
    fn effect_reject_serial_execution(&mut self, event: &EventExecuteSerial<'_>) -> Result<(), ()> {
        *event.accepted.borrow_mut() = false;
        Ok(())
    }
    fn guard_parallel_all_lanes_accepted(
        &self,
        event: &EventExecuteParallel<'_>,
    ) -> Result<bool, ()> {
        let result = *event.result.borrow();
        Ok(result.error == DispatchError::None
            && result.all_submitted
            && result.joined
            && result.all_lanes_accepted)
    }
    fn guard_parallel_join_failed(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> {
        let result = *event.result.borrow();
        Ok(result.error == DispatchError::JoinFailed || (result.all_submitted && !result.joined))
    }
    fn guard_parallel_lane_rejected(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> {
        let result = *event.result.borrow();
        Ok(result.error == DispatchError::LaneRejected
            || (result.all_submitted && result.joined && !result.all_lanes_accepted))
    }
    fn guard_parallel_submission_failed(
        &self,
        event: &EventExecuteParallel<'_>,
    ) -> Result<bool, ()> {
        let result = *event.result.borrow();
        Ok(result.error == DispatchError::SubmissionFailed || !result.all_submitted)
    }
    fn guard_parallel_unavailable(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> {
        Ok(!self.parallel_ready(event))
    }
    fn guard_parallel_unit_ready(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> {
        Ok(self.parallel_ready(event)
            && !event.request.dtype.uses_x8_rows()
            && !event.request.dtype.uses_x4_rows())
    }
    fn guard_parallel_x4_ready(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> {
        Ok(self.parallel_ready(event) && event.request.dtype.uses_x4_rows())
    }
    fn guard_parallel_x8_ready(&self, event: &EventExecuteParallel<'_>) -> Result<bool, ()> {
        Ok(self.parallel_ready(event) && event.request.dtype.uses_x8_rows())
    }
    fn guard_serial_accepted(&self, event: &EventExecuteSerial<'_>) -> Result<bool, ()> {
        let result = *event.result.borrow();
        Ok(result.error == DispatchError::None && result.all_lanes_accepted)
    }
    fn guard_serial_rejected(&self, event: &EventExecuteSerial<'_>) -> Result<bool, ()> {
        let result = *event.result.borrow();
        Ok(result.error != DispatchError::None || !result.all_lanes_accepted)
    }
    fn effect_capture_diagnostics(
        &mut self,
        event: &EventCaptureDiagnostics<'_>,
    ) -> Result<(), ()> {
        let mut diagnostics = MatmulDiagnostics::default();
        diagnostics.serial = self.serial_kernel.counters;
        for kernel in &self.lane_kernels[..self.active_lanes] {
            diagnostics.parallel.operations = diagnostics
                .parallel
                .operations
                .saturating_add(kernel.counters.operations);
            diagnostics.parallel.rows = diagnostics
                .parallel
                .rows
                .saturating_add(kernel.counters.rows);
        }
        *event.out.borrow_mut() = diagnostics;
        *event.accepted.borrow_mut() = true;
        Ok(())
    }
}

/// Single-writer, run-to-completion matrix multiplication actor.
pub struct TextGeneratorMatmulActor {
    machine: TextGeneratorMatmulStateMachine<TextGeneratorMatmulContext>,
}

impl fmt::Debug for TextGeneratorMatmulActor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TextGeneratorMatmulActor")
            .finish_non_exhaustive()
    }
}

impl Default for TextGeneratorMatmulActor {
    fn default() -> Self {
        Self::new(ExecutionPolicy::default())
    }
}

impl TextGeneratorMatmulActor {
    /// Constructs an actor with explicit kernel and lane configuration.
    #[must_use]
    pub fn new(policy: ExecutionPolicy) -> Self {
        let mut actor = Self {
            machine: TextGeneratorMatmulStateMachine::new(TextGeneratorMatmulContext::new(policy)),
        };
        let _ = actor.process_configure_kernel_kind(EventConfigureKernelKind {
            kind: policy.kernel_kind,
        });
        actor
    }

    /// Processes kernel configuration synchronously.
    pub fn process_configure_kernel_kind(
        &mut self,
        event: EventConfigureKernelKind,
    ) -> Result<(), ()> {
        self.machine
            .process_event(TextGeneratorMatmulEvents::EventConfigureKernelKind(event))
            .map(|_| ())
            .map_err(|_| ())
    }
    /// Processes a serial matrix request synchronously.
    pub fn process_serial<'a>(&mut self, event: EventExecuteSerial<'a>) -> Result<(), ()> {
        self.machine
            .process_event(TextGeneratorMatmulEvents::EventExecuteSerial(event))
            .map(|_| ())
            .map_err(|_| ())
    }
    /// Processes a parallel matrix request synchronously.
    pub fn process_parallel<'a>(&mut self, event: EventExecuteParallel<'a>) -> Result<(), ()> {
        self.machine
            .process_event(TextGeneratorMatmulEvents::EventExecuteParallel(event))
            .map(|_| ())
            .map_err(|_| ())
    }
    /// Processes an unexpected event and returns to the ready state.
    pub fn process_unexpected_event(&mut self) -> Result<(), ()> {
        self.machine.context_mut().unexpected_event()?;
        self.machine
            .set_state(TextGeneratorMatmulStates::StateReady);
        Ok(())
    }
    /// Returns the generated machine state.
    #[must_use]
    pub fn state(&self) -> &TextGeneratorMatmulStates {
        self.machine.state()
    }
    /// Tests generated machine state identity.
    #[must_use]
    pub fn is(&self, state: &TextGeneratorMatmulStates) -> bool {
        self.machine.is(state)
    }
    /// Returns immutable actor context.
    #[must_use]
    pub fn context(&self) -> &TextGeneratorMatmulContext {
        self.machine.context()
    }
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
    pub fn parallel_lanes_available(&self) -> bool {
        self.context().parallel_available()
    }
    /// Returns the configured number of active lanes.
    #[must_use]
    pub fn active_lane_count(&self) -> usize {
        self.context().active_lane_count()
    }
    /// Returns the typed outcome from a caller-owned dispatch result directly.
    #[must_use]
    pub fn dispatch_error(result: &DispatchResult) -> DispatchError {
        result.error
    }
    /// Returns serial kernel inspection state.
    #[must_use]
    pub fn serial_kernel(&self) -> &KernelState {
        self.context().serial_kernel()
    }
    /// Returns whether an unexpected event has been observed.
    #[must_use]
    pub fn unexpected_event_seen(&self) -> bool {
        self.context().unexpected()
    }
    /// Returns bounded parallel kernel inspection states.
    #[must_use]
    pub fn parallel_lane_kernels(&self) -> &[KernelState] {
        self.context().parallel_lane_kernels()
    }
    /// Source-shaped configuration wrapper.
    pub fn process_event_configure_kernel_kind(
        &mut self,
        event: EventConfigureKernelKind,
    ) -> Result<(), ()> {
        self.process_configure_kernel_kind(event)
    }
    /// Source-shaped serial execution wrapper.
    pub fn process_event_execute_serial<'a>(
        &mut self,
        event: EventExecuteSerial<'a>,
    ) -> Result<(), ()> {
        self.process_serial(event)
    }
    /// Source-shaped parallel execution wrapper.
    /// Captures bounded counter diagnostics synchronously.
    pub fn capture_diagnostics(&mut self, event: EventCaptureDiagnostics<'_>) -> Result<(), ()> {
        self.machine
            .process_event(TextGeneratorMatmulEvents::EventCaptureDiagnostics(event))
            .map(|_| ())
            .map_err(|_| ())
    }
    pub fn process_event_execute_parallel<'a>(
        &mut self,
        event: EventExecuteParallel<'a>,
    ) -> Result<(), ()> {
        self.process_parallel(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::RefCell;
    use core::sync::atomic::{AtomicUsize, Ordering};

    static CALLBACKS: [AtomicUsize; MAX_PARALLEL_LANES] =
        [const { AtomicUsize::new(0) }; MAX_PARALLEL_LANES];
    static ROW_BEGINS: [AtomicUsize; MAX_PARALLEL_LANES] =
        [const { AtomicUsize::new(0) }; MAX_PARALLEL_LANES];
    static ROW_COUNTS: [AtomicUsize; MAX_PARALLEL_LANES] =
        [const { AtomicUsize::new(0) }; MAX_PARALLEL_LANES];
    static SUBMISSIONS: AtomicUsize = AtomicUsize::new(0);
    static JOINS: AtomicUsize = AtomicUsize::new(0);
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    static DONE_CALLBACKS: AtomicUsize = AtomicUsize::new(0);
    static ERROR_CALLBACKS: AtomicUsize = AtomicUsize::new(0);

    fn reset_observations() {
        for lane in 0..MAX_PARALLEL_LANES {
            CALLBACKS[lane].store(0, Ordering::SeqCst);
            ROW_BEGINS[lane].store(0, Ordering::SeqCst);
            ROW_COUNTS[lane].store(0, Ordering::SeqCst);
        }
        SUBMISSIONS.store(0, Ordering::SeqCst);
        JOINS.store(0, Ordering::SeqCst);
        DONE_CALLBACKS.store(0, Ordering::SeqCst);
        ERROR_CALLBACKS.store(0, Ordering::SeqCst);
    }

    fn observe_parallel<'a>(_request: &MatmulRequest<'a>, slice: RowSlice, lane: usize) -> bool {
        CALLBACKS[lane].fetch_add(1, Ordering::SeqCst);
        ROW_BEGINS[lane].store(slice.row_begin, Ordering::SeqCst);
        ROW_COUNTS[lane].store(slice.row_count, Ordering::SeqCst);
        true
    }

    fn observe_submit(
        _context: *mut (),
        lane: usize,
        request: &MatmulRequest<'_>,
        slice: RowSlice,
        callback: ParallelKernel,
        accepted: &mut bool,
    ) -> bool {
        SUBMISSIONS.fetch_add(1, Ordering::SeqCst);
        *accepted = callback(request, slice, lane);
        true
    }

    fn observe_join(_context: *mut ()) -> bool {
        JOINS.fetch_add(1, Ordering::SeqCst);
        true
    }

    fn observe_done(_result: &DispatchResult) {
        DONE_CALLBACKS.fetch_add(1, Ordering::SeqCst);
    }

    fn observe_error(_result: &DispatchResult) {
        ERROR_CALLBACKS.fetch_add(1, Ordering::SeqCst);
    }

    fn observe_serial<'a>(_request: &MatmulRequest<'a>, _slice: RowSlice) -> bool {
        false
    }

    fn run_parallel(dtype: KernelDType, rows: usize, active_lanes: usize) -> DispatchResult {
        let lhs = [0.0_f32; 64];
        let rhs = [0.0_f32; 2];
        let mut output = [0.0_f32; 32];
        let output = RefCell::new(&mut output[..]);
        let request = MatmulRequest::new(dtype, rows, 2, &lhs, &rhs, &output)
            .with_kernels(observe_serial, observe_parallel);
        let result = RefCell::new(DispatchResult::default());
        let accepted = RefCell::new(false);
        let event = EventExecuteParallel {
            request,
            result: &result,
            accepted: &accepted,
            on_done: Some(observe_done),
            on_error: Some(observe_error),
        };
        let policy = ExecutionPolicy {
            active_lanes,
            ..ExecutionPolicy::default()
        }
        .with_lane_pool(LanePool::with_callbacks(
            core::ptr::null_mut(),
            observe_submit,
            observe_join,
        ));
        let mut actor = TextGeneratorMatmulActor::new(policy);
        assert_eq!(actor.process_parallel(event), Ok(()));
        assert!(*accepted.borrow());
        *result.borrow()
    }

    #[test]
    fn partitions_one_and_multiple_groups_without_empty_lanes() {
        let _lock = TEST_LOCK.lock().expect("matmul test lock");
        for (dtype, rows, group_rows) in [
            (KernelDType::Q4KX8Bl4, 8, 8),
            (KernelDType::Q8_0X4Bl4, 4, 4),
            (KernelDType::Other, 1, 1),
        ] {
            for active_lanes in [2, 4, 8] {
                reset_observations();
                let result = run_parallel(dtype, rows, active_lanes);
                assert_eq!(
                    result,
                    DispatchResult {
                        error: DispatchError::None,
                        lane_count: 1,
                        all_submitted: true,
                        joined: true,
                        all_lanes_accepted: true,
                    }
                );
                assert_eq!(SUBMISSIONS.load(Ordering::SeqCst), 0);
                assert_eq!(JOINS.load(Ordering::SeqCst), 1);
                assert_eq!(CALLBACKS[0].load(Ordering::SeqCst), 1);
                let slice = RowSlice {
                    row_begin: ROW_BEGINS[0].load(Ordering::SeqCst),
                    row_count: ROW_COUNTS[0].load(Ordering::SeqCst),
                };
                assert_eq!(
                    slice,
                    RowSlice {
                        row_begin: 0,
                        row_count: group_rows
                    }
                );
                assert!(slice.valid_for(rows) && slice.row_count > 0);
                assert_eq!(DONE_CALLBACKS.load(Ordering::SeqCst), 1);
                assert_eq!(ERROR_CALLBACKS.load(Ordering::SeqCst), 0);
            }
        }
        for active_lanes in [2, 4, 8] {
            reset_observations();
            let result = run_parallel(KernelDType::Q4KX8Bl4, 17, active_lanes);
            let expected_lanes = active_lanes.min(3);
            assert_eq!(result.error, DispatchError::None);
            assert_eq!(result.lane_count, expected_lanes);
            assert!(result.all_submitted && result.joined && result.all_lanes_accepted);
            assert_eq!(SUBMISSIONS.load(Ordering::SeqCst), expected_lanes - 1);
            for lane in 0..expected_lanes {
                let slice = RowSlice {
                    row_begin: ROW_BEGINS[lane].load(Ordering::SeqCst),
                    row_count: ROW_COUNTS[lane].load(Ordering::SeqCst),
                };
                assert!(slice.valid_for(17) && slice.row_count > 0);
            }
            assert_eq!(
                CALLBACKS
                    .iter()
                    .map(|count| count.load(Ordering::SeqCst))
                    .sum::<usize>(),
                expected_lanes
            );
        }
    }

    #[test]
    fn unavailable_parallel_request_uses_error_callback_route() {
        let _lock = TEST_LOCK.lock().expect("matmul test lock");
        reset_observations();
        let lhs = [0.0_f32; 2];
        let rhs = [0.0_f32; 1];
        let mut output = [0.0_f32; 1];
        let output = RefCell::new(&mut output[..]);
        let request = MatmulRequest::new(KernelDType::Other, 1, 1, &lhs, &rhs, &output)
            .with_kernels(observe_serial, observe_parallel);
        let result = RefCell::new(DispatchResult::default());
        let accepted = RefCell::new(true);
        let event = EventExecuteParallel {
            request,
            result: &result,
            accepted: &accepted,
            on_done: Some(observe_done),
            on_error: Some(observe_error),
        };
        let mut actor = TextGeneratorMatmulActor::new(ExecutionPolicy::default());
        assert_eq!(actor.process_parallel(event), Ok(()));
        assert_eq!(result.borrow().error, DispatchError::ParallelUnavailable);
        assert!(!*accepted.borrow());
        assert_eq!(ERROR_CALLBACKS.load(Ordering::SeqCst), 1);
        assert_eq!(CALLBACKS[0].load(Ordering::SeqCst), 0);
    }
}
