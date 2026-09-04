//! Source-aligned assemble validation-pass actor.
//!
//! The C++ pass receives an assemble request and completion context by reference.
//! Rust carries only copied, bounded scalar data so the actor remains synchronous,
//! single-writer, allocation-free, and independent of caller-owned resources.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs
)]

use sml::sml;

/// Bounded errors retained by the validation completion context.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AssemblerError {
    /// No error was reported by the preceding phase.
    #[default]
    None = 0,
    /// The copied assemble request is incomplete or otherwise invalid.
    InvalidRequest = 1,
    /// A bounded output/workspace capacity was exceeded.
    Capacity = 2,
    /// An internal phase or callback failure occurred.
    InternalError = 4,
    /// An error value was not recognized by the actor.
    Untracked = 8,
    /// A callback reported failure.
    Callback = 16,
    /// The caller supplied a failed result.
    Result = 32,
}

/// Completion outcome matching `assemble_validate_pass::events::phase_outcome`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PassOutcome {
    /// The pass has not completed.
    #[default]
    Unknown = 0,
    /// Validation completed successfully.
    Done = 1,
    /// Validation rejected the request or encountered an internal failure.
    Failed = 2,
}

/// Copied assemble input and completion diagnostics.
///
/// `step_plan` and `output_out` are presence bits/opaque scalar identities, not
/// pointers. They preserve the source guard's null/non-null decisions without
/// allowing resources to escape into this actor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AssemblerEventAssembleGraph {
    /// Non-zero when the source step plan was present.
    pub step_plan: usize,
    /// Non-zero when the source output destination was present.
    pub output_out: usize,
    /// Bytes required for each tensor.
    pub bytes_per_tensor: u64,
    /// Available workspace capacity in bytes.
    pub workspace_capacity_bytes: u64,
    /// Optional bounded node-count hint copied from the request.
    pub node_count_hint: u32,
    /// Optional bounded tensor-count hint copied from the request.
    pub tensor_count_hint: u32,
    /// Error already carried by the completion context.
    pub err: AssemblerError,
    /// Callback error retained for diagnostics and propagation.
    pub callback_error: Option<AssemblerError>,
    /// Result error retained for diagnostics and propagation.
    pub result_error: Option<AssemblerError>,
}

impl AssemblerEventAssembleGraph {
    /// Creates a copied event from the source guard inputs.
    #[must_use]
    pub const fn new(
        step_plan: usize,
        output_out: usize,
        bytes_per_tensor: u64,
        workspace_capacity_bytes: u64,
    ) -> Self {
        Self {
            step_plan,
            output_out,
            bytes_per_tensor,
            workspace_capacity_bytes,
            node_count_hint: 0,
            tensor_count_hint: 0,
            err: AssemblerError::None,
            callback_error: None,
            result_error: None,
        }
    }

    /// Returns whether the copied request satisfies the source null/zero checks.
    #[must_use]
    pub const fn request_is_valid(&self) -> bool {
        self.step_plan != 0
            && self.output_out != 0
            && self.bytes_per_tensor != 0
            && self.workspace_capacity_bytes != 0
    }
}

sml! {
    GraphAssemblerAssembleValidatePass {
        "assembled"_s <= *"deciding"_s + completion<AssemblerEventAssembleGraph> [phase_done] / mark_done,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_invalid_request] / mark_failed_invalid_request,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "assembled"_s + unexpected_event<_> / on_unexpected_from_assembled,
        "unexpected_event"_s <= "assemble_failed"_s + unexpected_event<_> / on_unexpected_from_assemble_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "assembled"_s = X,
        "assemble_failed"_s = X,
    }
}

/// Copied context for `GraphAssemblerAssembleValidatePass`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GraphAssemblerAssembleValidatePassContext {
    /// Copied step-plan identity/presence bit.
    pub step_plan: usize,
    /// Copied output destination identity/presence bit.
    pub output_out: usize,
    /// Copied bytes-per-tensor input.
    pub bytes_per_tensor: u64,
    /// Copied workspace capacity input.
    pub workspace_capacity_bytes: u64,
    /// Copied node-count hint.
    pub node_count_hint: u32,
    /// Copied tensor-count hint.
    pub tensor_count_hint: u32,
    /// Validation outcome.
    pub outcome: PassOutcome,
    /// Effective phase error.
    pub err: AssemblerError,
    /// Callback error retained independently of the effective error.
    pub callback_error: Option<AssemblerError>,
    /// Result error retained independently of the effective error.
    pub result_error: Option<AssemblerError>,
}

impl GraphAssemblerAssembleValidatePassContext {
    fn load(&mut self, event: AssemblerEventAssembleGraph) {
        self.step_plan = event.step_plan;
        self.output_out = event.output_out;
        self.bytes_per_tensor = event.bytes_per_tensor;
        self.workspace_capacity_bytes = event.workspace_capacity_bytes;
        self.node_count_hint = event.node_count_hint;
        self.tensor_count_hint = event.tensor_count_hint;
        self.callback_error = event.callback_error;
        self.result_error = event.result_error;
        self.err = event.err;
        if self.result_error.is_some() {
            self.err = AssemblerError::Result;
        } else if self.callback_error.is_some() {
            self.err = AssemblerError::Callback;
        }
        self.outcome = PassOutcome::Unknown;
    }
}

impl GraphAssemblerAssembleValidatePassStateMachineContext
    for GraphAssemblerAssembleValidatePassContext
{
    fn mark_done(&mut self) -> Result<(), ()> {
        self.outcome = PassOutcome::Done;
        self.err = AssemblerError::None;
        Ok(())
    }

    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        self.outcome = PassOutcome::Failed;
        if self.err == AssemblerError::None {
            self.err = AssemblerError::InvalidRequest;
        }
        Ok(())
    }

    fn on_unexpected_from_assemble_failed(&mut self) -> Result<(), ()> {
        self.outcome = PassOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_assembled(&mut self) -> Result<(), ()> {
        self.outcome = PassOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.outcome = PassOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.outcome = PassOutcome::Failed;
        self.err = AssemblerError::InternalError;
        Ok(())
    }

    fn phase_done(&self) -> Result<bool, ()> {
        Ok(self.err == AssemblerError::None
            && self.step_plan != 0
            && self.output_out != 0
            && self.bytes_per_tensor != 0
            && self.workspace_capacity_bytes != 0)
    }

    fn phase_invalid_request(&self) -> Result<bool, ()> {
        Ok(self.err != AssemblerError::None
            || (self.step_plan == 0
                || self.output_out == 0
                || self.bytes_per_tensor == 0
                || self.workspace_capacity_bytes == 0))
    }
}

/// Synchronous, single-writer actor around the generated validation machine.
pub struct GraphAssemblerAssembleValidatePassActor {
    machine:
        GraphAssemblerAssembleValidatePassStateMachine<GraphAssemblerAssembleValidatePassContext>,
}

impl Default for GraphAssemblerAssembleValidatePassActor {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphAssemblerAssembleValidatePassActor {
    /// Creates an actor in the generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GraphAssemblerAssembleValidatePassStateMachine::new(
                GraphAssemblerAssembleValidatePassContext::default(),
            ),
        }
    }

    /// Processes one copied completion event synchronously.
    pub fn process_event(&mut self, event: AssemblerEventAssembleGraph) -> PassOutcome {
        if !self
            .machine
            .is(&GraphAssemblerAssembleValidatePassStates::Deciding)
        {
            return self.process_unexpected();
        }
        self.machine.context_mut().load(event);
        let _ = self
            .machine
            .process_event(GraphAssemblerAssembleValidatePassEvents::AssemblerEventAssembleGraph);
        self.machine.context().outcome
    }

    /// Records an unexpected event and enters the generated unexpected state.
    pub fn process_unexpected(&mut self) -> PassOutcome {
        self.machine.context_mut().outcome = PassOutcome::Failed;
        self.machine.context_mut().err = AssemblerError::InternalError;
        self.machine
            .set_state(GraphAssemblerAssembleValidatePassStates::UnexpectedEvent);
        self.machine.context().outcome
    }

    /// Returns the generated state.
    #[must_use]
    pub fn state(&self) -> &GraphAssemblerAssembleValidatePassStates {
        self.machine.state()
    }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &GraphAssemblerAssembleValidatePassStates) -> bool {
        self.machine.is(state)
    }

    /// Returns the copied validation context.
    #[must_use]
    pub fn context(&self) -> &GraphAssemblerAssembleValidatePassContext {
        self.machine.context()
    }
}
