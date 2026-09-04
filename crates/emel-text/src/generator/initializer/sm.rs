//! Bounded, synchronous initializer for the text generator pipeline.
//!
//! The actor mirrors the initializer state machine in `emel.cpp`: every
//! collaborator outcome is supplied by the copied runtime event, every guard
//! is explicit, and no request data is retained beyond bounded scalar state.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    unpredictable_function_pointer_comparisons,
    clippy::manual_is_multiple_of,
    dead_code,
    missing_docs
)]
use core::cell::RefCell;
use sml::sml;

/// Selection strategy chosen by the generation contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum SelectionMode {
    #[default]
    SampleLogits = 0,
    PreselectedArgmax = 1,
}

/// Error published by an initializer run.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum InitializerError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    Backend = 2,
    Unexpected = 3,
}

/// Result handed to a bounded completion callback.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InitializerResult {
    pub accepted: bool,
    pub error: InitializerError,
    pub phase_code: i32,
    pub buffers_ready: bool,
}

/// Synchronous completion callback. The actor never stores the callback.
pub type InitializeCallback = fn(&InitializerResult);

/// A collaborator phase that the initializer may execute synchronously.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InitializerOperation {
    #[default]
    BackendPrepare,
    ConditionerBind,
    RendererInitialize,
    MemoryReserve,
    GraphReserve,
    ConfigureSampler,
}

/// Outcome returned by one caller-owned synchronous collaborator dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhaseOutcome {
    pub accepted: bool,
    pub code: i32,
}

impl PhaseOutcome {
    pub const fn ok() -> Self {
        Self {
            accepted: true,
            code: 0,
        }
    }
    pub const fn invalid_request() -> Self {
        Self {
            accepted: false,
            code: 1,
        }
    }
    pub const fn backend() -> Self {
        Self {
            accepted: false,
            code: 2,
        }
    }
}

/// Caller-owned synchronous collaborator hook. The actor never stores it.
pub type InitializerDispatch = fn(InitializerOperation, &EventRun<'_>) -> PhaseOutcome;

/// Runtime input and collaborator outcomes for one initialization transaction.
//
// The dispatch hook is the source of truth for collaborator phases. The
// per-phase fields remain part of the compatibility event surface, but are
// metadata only and are not consulted to manufacture phase success.
///
/// A non-zero phase code is classified according to the collaborator which
/// produced it; unrecognised or code-less rejection is a backend failure.
fn legacy_dispatch(operation: InitializerOperation, event: &EventRun<'_>) -> PhaseOutcome {
    match operation {
        InitializerOperation::BackendPrepare => PhaseOutcome {
            accepted: event.backend_prepare_accepted,
            code: event.backend_prepare_code,
        },
        InitializerOperation::ConditionerBind => PhaseOutcome {
            accepted: event.conditioner_bind_accepted,
            code: event.conditioner_bind_code,
        },
        InitializerOperation::RendererInitialize => PhaseOutcome {
            accepted: event.renderer_initialize_accepted,
            code: event.renderer_initialize_code,
        },
        InitializerOperation::MemoryReserve => PhaseOutcome {
            accepted: event.memory_reserve_accepted,
            code: event.memory_reserve_code,
        },
        InitializerOperation::GraphReserve => PhaseOutcome {
            accepted: event.graph_reserve_accepted,
            code: event.graph_reserve_code,
        },
        InitializerOperation::ConfigureSampler => PhaseOutcome {
            accepted: event.sampler_configured,
            code: event.sampler_config_code,
        },
    }
}

fn conditioner_error_outcome(error: crate::ConditionerError) -> PhaseOutcome {
    let code = error.code();
    if code == 0 {
        PhaseOutcome::backend()
    } else {
        PhaseOutcome {
            accepted: false,
            code,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EventRun<'event> {
    pub valid_request: bool,
    pub generation_contract_valid: bool,
    pub max_prompt_tokens: u32,
    pub max_generated_tokens: u32,
    pub max_blocks: u32,
    pub block_tokens: u32,
    pub model_vocab_size: u32,
    pub model_embedding: u32,
    pub model_heads: u32,
    pub model_context: u32,
    pub backend_ready: bool,
    pub backend_kv_block_tokens: u32,
    pub backend_kv_positions_capacity: u32,
    pub backend_n_ctx: u32,
    pub backend_prepare_accepted: bool,
    pub backend_prepare_code: i32,
    pub conditioner_bind_accepted: bool,
    pub conditioner_bind_code: i32,
    pub conditioner_tokenizer_available: bool,
    pub conditioner_formatter_available: bool,
    pub conditioner_model_valid: bool,
    pub renderer_initialize_accepted: bool,
    pub renderer_initialize_code: i32,
    pub memory_reserve_accepted: bool,
    pub memory_reserve_code: i32,
    pub graph_reserve_accepted: bool,
    pub graph_reserve_code: i32,
    pub graph_reservation_present: bool,
    pub sampler_configured: bool,
    pub sampler_config_code: i32,
    pub sampler_buffers_ready: bool,
    pub logits_ready: bool,
    /// Caller validation of the mode-specific sampler callback contract.
    pub sampler_contract_valid: bool,
    pub selection_mode: SelectionMode,
    /// Optional caller-owned maintained conditioner actor. It is borrowed only
    /// for this synchronous event and is never retained by the initializer.
    pub conditioner: Option<&'event RefCell<crate::Conditioner>>,
    pub dispatch: Option<InitializerDispatch>,
    pub on_done: Option<InitializeCallback>,
    pub on_error: Option<InitializeCallback>,
}

impl EventRun<'_> {
    /// Creates a valid request with conservative bounded geometry.
    #[must_use]
    pub const fn valid() -> Self {
        Self {
            valid_request: true,
            generation_contract_valid: true,
            max_prompt_tokens: 4096,
            max_generated_tokens: 4096,
            max_blocks: 1,
            block_tokens: 16,
            model_vocab_size: 1,
            model_embedding: 1,
            model_heads: 1,
            model_context: 16,
            backend_ready: false,
            backend_kv_block_tokens: 16,
            backend_kv_positions_capacity: 16,
            backend_n_ctx: 16,
            backend_prepare_accepted: true,
            backend_prepare_code: 0,
            conditioner_bind_accepted: true,
            conditioner_bind_code: 0,
            conditioner_tokenizer_available: true,
            conditioner_formatter_available: true,
            conditioner_model_valid: true,
            renderer_initialize_accepted: true,
            renderer_initialize_code: 0,
            memory_reserve_accepted: true,
            memory_reserve_code: 0,
            graph_reserve_accepted: true,
            graph_reserve_code: 0,
            graph_reservation_present: true,
            sampler_configured: true,
            sampler_config_code: 0,
            sampler_buffers_ready: true,
            logits_ready: true,
            sampler_contract_valid: true,
            selection_mode: SelectionMode::SampleLogits,
            conditioner: None,
            dispatch: Some(legacy_dispatch),
            on_done: None,
            on_error: None,
        }
    }
}

sml! {
    TextGeneratorInitializer<'event> {
        "preparing_backend"_s <= *"idle"_s + event<EventRun<'event>> / begin_initialize,
        "binding_conditioner"_s <= "preparing_backend"_s + completion<EventRun>(EventRun<'event>) [guard_backend_reuse_allowed],
        "idle"_s <= "preparing_backend"_s + completion<EventRun>(EventRun<'event>) [guard_generation_contract_invalid] / mark_invalid_request_from_preparing_backend,
        "preparing_backend_decision"_s <= "preparing_backend"_s + completion<EventRun>(EventRun<'event>) [guard_backend_prepare_allowed] / request_backend_prepare,
        "binding_conditioner"_s <= "preparing_backend_decision"_s + completion<EventRun>(EventRun<'event>) [backend_already_ready],
        "binding_conditioner"_s <= "preparing_backend_decision"_s + completion<EventRun>(EventRun<'event>) [backend_prepare_ok] / accept_prepared_backend,
        "idle"_s <= "preparing_backend_decision"_s + completion<EventRun>(EventRun<'event>) [backend_prepare_invalid_request] / mark_invalid_request_from_preparing_backend_decision,
        "idle"_s <= "preparing_backend_decision"_s + completion<EventRun>(EventRun<'event>) [backend_prepare_backend_error] / mark_backend_error_from_preparing_backend_decision,
        "binding_conditioner_actor"_s <= "binding_conditioner"_s + completion<EventRun>(EventRun<'event>) [conditioner_actor_route] / request_conditioner_bind_actor,
        "binding_conditioner_callback"_s <= "binding_conditioner"_s + completion<EventRun>(EventRun<'event>) [conditioner_callback_route] / request_conditioner_bind_callback,
        "idle"_s <= "binding_conditioner"_s + completion<EventRun>(EventRun<'event>) [conditioner_route_missing] / mark_backend_error_from_binding_conditioner,
        "binding_conditioner_decision"_s <= "binding_conditioner_actor"_s + completion<EventRun>(EventRun<'event>),
        "binding_conditioner_decision"_s <= "binding_conditioner_callback"_s + completion<EventRun>(EventRun<'event>),
        "initializing_renderer"_s <= "binding_conditioner_decision"_s + completion<EventRun>(EventRun<'event>) [conditioner_bind_ok],
        "idle"_s <= "binding_conditioner_decision"_s + completion<EventRun>(EventRun<'event>) [conditioner_bind_invalid_request] / mark_invalid_request_from_binding_conditioner_decision,
        "idle"_s <= "binding_conditioner_decision"_s + completion<EventRun>(EventRun<'event>) [conditioner_bind_backend_error] / mark_backend_error_from_binding_conditioner_decision,
        "initializing_renderer_decision"_s <= "initializing_renderer"_s + completion<EventRun>(EventRun<'event>) / request_renderer_initialize,
        "reserving_memory"_s <= "initializing_renderer_decision"_s + completion<EventRun>(EventRun<'event>) [renderer_initialize_ok],
        "idle"_s <= "initializing_renderer_decision"_s + completion<EventRun>(EventRun<'event>) [renderer_initialize_invalid_request] / mark_invalid_request_from_initializing_renderer_decision,
        "idle"_s <= "initializing_renderer_decision"_s + completion<EventRun>(EventRun<'event>) [renderer_initialize_backend_error] / mark_backend_error_from_initializing_renderer_decision,
        "reserving_memory_decision"_s <= "reserving_memory"_s + completion<EventRun>(EventRun<'event>) [guard_memory_geometry_fits_backend] / request_memory_reserve,
        "idle"_s <= "reserving_memory"_s + completion<EventRun>(EventRun<'event>) [guard_memory_geometry_gap] / mark_invalid_request_from_reserving_memory,
        "configuring_sampling_mode_decision"_s <= "reserving_memory_decision"_s + completion<EventRun>(EventRun<'event>) [memory_reserve_with_existing_graph],
        "reserving_graph"_s <= "reserving_memory_decision"_s + completion<EventRun>(EventRun<'event>) [memory_reserve_with_missing_graph],
        "idle"_s <= "reserving_memory_decision"_s + completion<EventRun>(EventRun<'event>) [memory_reserve_invalid_request] / mark_invalid_request_from_reserving_memory_decision,
        "idle"_s <= "reserving_memory_decision"_s + completion<EventRun>(EventRun<'event>) [memory_reserve_backend_error] / mark_backend_error_from_reserving_memory_decision,
        "reserving_graph_decision"_s <= "reserving_graph"_s + completion<EventRun>(EventRun<'event>) / request_graph_reserve,
        "configuring_sampling_mode_decision"_s <= "reserving_graph_decision"_s + completion<EventRun>(EventRun<'event>) [graph_reserve_ok],
        "idle"_s <= "reserving_graph_decision"_s + completion<EventRun>(EventRun<'event>) [graph_reserve_invalid_request] / mark_invalid_request_from_reserving_graph_decision,
        "idle"_s <= "reserving_graph_decision"_s + completion<EventRun>(EventRun<'event>) [graph_reserve_backend_error] / mark_backend_error_from_reserving_graph_decision,
        "configuring_sampler"_s <= "configuring_sampling_mode_decision"_s + completion<EventRun>(EventRun<'event>) [uses_materialized_logits],
        "configure_preselected_argmax"_s <= "configuring_sampling_mode_decision"_s + completion<EventRun>(EventRun<'event>) [uses_preselected_argmax],
        "configuring_sampler_decision"_s <= "configuring_sampler"_s + completion<EventRun>(EventRun<'event>) / configure_sampler,
        "idle"_s <= "configuring_sampler_decision"_s + completion<EventRun>(EventRun<'event>) [sampler_configured],
        "idle"_s <= "configuring_sampler_decision"_s + completion<EventRun>(EventRun<'event>) [sampler_config_failed] / mark_backend_error_from_configuring_sampler_decision,
        "configure_preselected_argmax_decision"_s <= "configure_preselected_argmax"_s + completion<EventRun>(EventRun<'event>) / configure_preselected_argmax,
        "idle"_s <= "configure_preselected_argmax_decision"_s + completion<EventRun>(EventRun<'event>) [sampler_configured],
        "idle"_s <= "configure_preselected_argmax_decision"_s + completion<EventRun>(EventRun<'event>) [sampler_config_failed] / mark_backend_error_from_configure_preselected_argmax_decision,
        "idle"_s <= "idle"_s + unexpected_event<_> / on_unexpected_from_idle,
        "idle"_s <= "preparing_backend"_s + unexpected_event<_> / on_unexpected_from_preparing_backend,
        "idle"_s <= "preparing_backend_decision"_s + unexpected_event<_> / on_unexpected_from_preparing_backend_decision,
        "idle"_s <= "binding_conditioner"_s + unexpected_event<_> / on_unexpected_from_binding_conditioner,
        "idle"_s <= "binding_conditioner_decision"_s + unexpected_event<_> / on_unexpected_from_binding_conditioner_decision,
        "idle"_s <= "initializing_renderer"_s + unexpected_event<_> / on_unexpected_from_initializing_renderer,
        "idle"_s <= "initializing_renderer_decision"_s + unexpected_event<_> / on_unexpected_from_initializing_renderer_decision,
        "idle"_s <= "reserving_memory"_s + unexpected_event<_> / on_unexpected_from_reserving_memory,
        "idle"_s <= "reserving_memory_decision"_s + unexpected_event<_> / on_unexpected_from_reserving_memory_decision,
        "idle"_s <= "reserving_graph"_s + unexpected_event<_> / on_unexpected_from_reserving_graph,
        "idle"_s <= "reserving_graph_decision"_s + unexpected_event<_> / on_unexpected_from_reserving_graph_decision,
        "idle"_s <= "configuring_sampling_mode_decision"_s + unexpected_event<_> / on_unexpected_from_configuring_sampling_mode_decision,
        "idle"_s <= "configuring_sampler"_s + unexpected_event<_> / on_unexpected_from_configuring_sampler,
        "idle"_s <= "configuring_sampler_decision"_s + unexpected_event<_> / on_unexpected_from_configuring_sampler_decision,
        "idle"_s <= "configure_preselected_argmax"_s + unexpected_event<_> / on_unexpected_from_configure_preselected_argmax,
        "idle"_s <= "configure_preselected_argmax_decision"_s + unexpected_event<_> / on_unexpected_from_configure_preselected_argmax_decision,
    }
}

/// Bounded state retained by the initializer actor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TextGeneratorInitializerContext {
    error: InitializerError,
    phase_code: i32,
    phase_accepted: bool,
    buffers_ready: bool,
    backend_ready: bool,
    graph_reservation_present: bool,
    generation_contract_valid: bool,
    max_blocks: u32,
    block_tokens: u32,
    backend_kv_block_tokens: u32,
    backend_kv_positions_capacity: u32,
    backend_n_ctx: u32,
    selection_mode: SelectionMode,
}

impl TextGeneratorInitializerContext {
    #[must_use]
    pub const fn error(&self) -> InitializerError {
        self.error
    }
    #[must_use]
    pub const fn phase_code(&self) -> i32 {
        self.phase_code
    }
    #[must_use]
    pub const fn phase_accepted(&self) -> bool {
        self.phase_accepted
    }
    #[must_use]
    pub const fn buffers_ready(&self) -> bool {
        self.buffers_ready
    }
    #[must_use]
    pub const fn graph_reservation_present(&self) -> bool {
        self.graph_reservation_present
    }
    #[must_use]
    pub const fn generation_contract_valid(&self) -> bool {
        self.generation_contract_valid
    }
    #[must_use]
    pub const fn backend_ready(&self) -> bool {
        self.backend_ready
    }
    #[must_use]
    fn phase_success(&self) -> bool {
        self.phase_accepted && self.phase_code == 0
    }
    fn phase_invalid_for(&self, operation: InitializerOperation) -> bool {
        if self.phase_success() {
            return false;
        }
        match operation {
            InitializerOperation::BackendPrepare => matches!(self.phase_code, 1 | 8),
            InitializerOperation::ConditionerBind => matches!(self.phase_code, 1 | 2 | 4),
            InitializerOperation::RendererInitialize => matches!(self.phase_code, 1 | 4),
            InitializerOperation::MemoryReserve | InitializerOperation::GraphReserve => {
                self.phase_code == 1
            }
            InitializerOperation::ConfigureSampler => self.phase_code == 1,
        }
    }
    fn phase_backend_for(&self, operation: InitializerOperation) -> bool {
        !self.phase_success() && !self.phase_invalid_for(operation)
    }
    #[must_use]
    pub const fn selection_mode(&self) -> SelectionMode {
        self.selection_mode
    }

    fn mark(&mut self, error: InitializerError) -> Result<(), ()> {
        self.error = error;
        self.phase_accepted = false;
        Ok(())
    }
    fn phase_outcome(event: &EventRun<'_>, operation: InitializerOperation) -> PhaseOutcome {
        event.dispatch.map_or(PhaseOutcome::backend(), |dispatch| {
            dispatch(operation, event)
        })
    }
    fn apply_phase(&mut self, event: &EventRun<'_>, operation: InitializerOperation) {
        let outcome = Self::phase_outcome(event, operation);
        self.phase_accepted = outcome.accepted;
        self.phase_code = outcome.code;
    }
    fn request_phase(
        &mut self,
        event: &EventRun<'_>,
        operation: InitializerOperation,
    ) -> Result<(), ()> {
        self.apply_phase(event, operation);
        Ok(())
    }
    fn unexpected(&mut self) -> Result<(), ()> {
        self.mark(InitializerError::Unexpected)
    }
}

impl TextGeneratorInitializerStateMachineContext for TextGeneratorInitializerContext {
    fn begin_initialize(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        self.error = InitializerError::None;
        self.phase_code = 0;
        self.phase_accepted = false;
        self.buffers_ready = false;
        self.backend_ready = event.backend_ready;
        self.graph_reservation_present = event.graph_reservation_present;
        self.generation_contract_valid = event.generation_contract_valid;
        self.max_blocks = event.max_blocks;
        self.block_tokens = event.block_tokens;
        self.backend_kv_block_tokens = event.backend_kv_block_tokens;
        self.backend_kv_positions_capacity = event.backend_kv_positions_capacity;
        self.backend_n_ctx = event.backend_n_ctx;
        self.selection_mode = event.selection_mode;
        Ok(())
    }

    fn request_backend_prepare(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        self.backend_ready = false;
        self.request_phase(event, InitializerOperation::BackendPrepare)
    }
    fn accept_prepared_backend(&mut self, _event: &EventRun<'_>) -> Result<(), ()> {
        self.backend_ready = true;
        Ok(())
    }

    fn conditioner_actor_route(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.conditioner.is_some())
    }
    fn conditioner_callback_route(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.conditioner.is_none() && event.dispatch.is_some())
    }
    fn conditioner_route_missing(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.conditioner.is_none() && event.dispatch.is_none())
    }
    fn request_conditioner_bind_actor(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        let outcome = event
            .conditioner
            .map_or_else(PhaseOutcome::backend, |conditioner| {
                conditioner
                    .borrow_mut()
                    .process_event(crate::conditioner_event::Bind {
                        tokenizer_available: event.conditioner_tokenizer_available,
                        formatter_available: event.conditioner_formatter_available,
                        model_valid: event.conditioner_model_valid,
                    })
                    .map_or_else(conditioner_error_outcome, |()| PhaseOutcome::ok())
            });
        self.phase_accepted = outcome.accepted;
        self.phase_code = outcome.code;
        Ok(())
    }
    fn request_conditioner_bind_callback(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        self.request_phase(event, InitializerOperation::ConditionerBind)
    }
    fn mark_backend_error_from_binding_conditioner(&mut self, _: &EventRun<'_>) -> Result<(), ()> {
        self.phase_accepted = false;
        self.phase_code = 2;
        self.mark(InitializerError::Backend)
    }
    fn request_renderer_initialize(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        self.request_phase(event, InitializerOperation::RendererInitialize)
    }
    fn request_memory_reserve(&mut self, event: &EventRun) -> Result<(), ()> {
        self.request_phase(event, InitializerOperation::MemoryReserve)
    }
    fn request_graph_reserve(&mut self, event: &EventRun) -> Result<(), ()> {
        self.request_phase(event, InitializerOperation::GraphReserve)?;
        if self.phase_success() {
            self.graph_reservation_present = event.graph_reservation_present;
        }
        Ok(())
    }
    fn configure_sampler(&mut self, event: &EventRun) -> Result<(), ()> {
        self.apply_phase(event, InitializerOperation::ConfigureSampler);
        self.buffers_ready = self.phase_success() && event.sampler_buffers_ready;
        self.phase_accepted = self.buffers_ready;
        Ok(())
    }
    fn configure_preselected_argmax(&mut self, event: &EventRun) -> Result<(), ()> {
        self.buffers_ready = event.logits_ready;
        self.phase_accepted = self.buffers_ready;
        self.phase_code = 0;
        Ok(())
    }

    fn mark_invalid_request_from_preparing_backend(&mut self, _: &EventRun) -> Result<(), ()> {
        self.mark(InitializerError::InvalidRequest)
    }
    fn mark_invalid_request_from_preparing_backend_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark(InitializerError::InvalidRequest)
    }
    fn mark_invalid_request_from_binding_conditioner_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark(InitializerError::InvalidRequest)
    }
    fn mark_invalid_request_from_initializing_renderer_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark(InitializerError::InvalidRequest)
    }
    fn mark_invalid_request_from_reserving_memory(&mut self, _: &EventRun) -> Result<(), ()> {
        self.mark(InitializerError::InvalidRequest)
    }
    fn mark_invalid_request_from_reserving_memory_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark(InitializerError::InvalidRequest)
    }
    fn mark_invalid_request_from_reserving_graph_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark(InitializerError::InvalidRequest)
    }

    fn mark_backend_error_from_preparing_backend_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark(InitializerError::Backend)
    }
    fn mark_backend_error_from_binding_conditioner_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark(InitializerError::Backend)
    }
    fn mark_backend_error_from_initializing_renderer_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark(InitializerError::Backend)
    }
    fn mark_backend_error_from_reserving_memory_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark(InitializerError::Backend)
    }
    fn mark_backend_error_from_reserving_graph_decision(&mut self, _: &EventRun) -> Result<(), ()> {
        self.mark(InitializerError::Backend)
    }
    fn mark_backend_error_from_configuring_sampler_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark(InitializerError::Backend)
    }
    fn mark_backend_error_from_configure_preselected_argmax_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark(InitializerError::Backend)
    }
    fn guard_generation_contract_invalid(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(!self.generation_contract_valid
            || !event.valid_request
            || !event.sampler_contract_valid
            || event.max_prompt_tokens == 0
            || event.max_prompt_tokens > 4096
            || event.max_generated_tokens == 0
            || event.max_generated_tokens > 4096
            || event.max_blocks == 0
            || event.block_tokens == 0
            || event.model_vocab_size == 0
            || event.model_embedding == 0
            || event.model_heads == 0
            || event.model_context == 0
            || event.block_tokens > event.model_context
            || event.model_embedding % event.model_heads != 0)
    }
    fn guard_backend_reuse_allowed(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(self.backend_already_ready(event)? && !self.guard_generation_contract_invalid(event)?)
    }
    fn guard_backend_prepare_allowed(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(
            !self.backend_already_ready(event)?
                && !self.guard_generation_contract_invalid(event)?,
        )
    }
    fn backend_already_ready(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.backend_ready
            && self.block_tokens > 0
            && self.backend_kv_block_tokens == self.block_tokens
            && self.backend_kv_positions_capacity >= self.backend_n_ctx)
    }
    fn backend_prepare_ok(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_success())
    }
    fn backend_prepare_invalid_request(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_invalid_for(InitializerOperation::BackendPrepare))
    }
    fn backend_prepare_backend_error(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_backend_for(InitializerOperation::BackendPrepare))
    }

    fn conditioner_bind_ok(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_success())
    }
    fn conditioner_bind_invalid_request(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_invalid_for(InitializerOperation::ConditionerBind))
    }
    fn conditioner_bind_backend_error(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_backend_for(InitializerOperation::ConditionerBind))
    }
    fn renderer_initialize_ok(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_success())
    }
    fn renderer_initialize_invalid_request(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_invalid_for(InitializerOperation::RendererInitialize))
    }
    fn renderer_initialize_backend_error(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_backend_for(InitializerOperation::RendererInitialize))
    }

    fn guard_memory_geometry_fits_backend(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.block_tokens > 0
            && self.backend_kv_block_tokens == self.block_tokens
            && self.backend_kv_positions_capacity >= self.backend_n_ctx
            && self
                .max_blocks
                .checked_mul(self.block_tokens)
                .is_some_and(|tokens| tokens <= self.backend_kv_positions_capacity))
    }
    fn guard_memory_geometry_gap(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(!self.guard_memory_geometry_fits_backend(event)?)
    }
    fn memory_reserve_with_existing_graph(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_success() && self.graph_reservation_present)
    }
    fn memory_reserve_with_missing_graph(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_success() && !self.graph_reservation_present)
    }
    fn memory_reserve_invalid_request(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_invalid_for(InitializerOperation::MemoryReserve))
    }
    fn memory_reserve_backend_error(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_backend_for(InitializerOperation::MemoryReserve))
    }
    fn graph_reserve_ok(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_success())
    }
    fn graph_reserve_invalid_request(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_invalid_for(InitializerOperation::GraphReserve))
    }
    fn graph_reserve_backend_error(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.phase_backend_for(InitializerOperation::GraphReserve))
    }

    fn uses_materialized_logits(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.selection_mode == SelectionMode::SampleLogits)
    }
    fn uses_preselected_argmax(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.selection_mode == SelectionMode::PreselectedArgmax)
    }
    fn sampler_configured(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(self.buffers_ready)
    }
    fn sampler_config_failed(&self, _: &EventRun) -> Result<bool, ()> {
        Ok(!self.buffers_ready)
    }

    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_preparing_backend(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_preparing_backend_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_binding_conditioner(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_binding_conditioner_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_initializing_renderer(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_initializing_renderer_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_reserving_memory(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_reserving_memory_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_reserving_graph(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_reserving_graph_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_configuring_sampling_mode_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_configuring_sampler(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_configuring_sampler_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_configure_preselected_argmax(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_configure_preselected_argmax_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
}

/// Synchronous actor around the generated initializer machine.
pub struct TextGeneratorInitializerActor {
    machine: TextGeneratorInitializerStateMachine<TextGeneratorInitializerContext>,
}

impl Default for TextGeneratorInitializerActor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextGeneratorInitializerActor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextGeneratorInitializerStateMachine::new(
                TextGeneratorInitializerContext::default(),
            ),
        }
    }

    pub fn process_event(&mut self, event: EventRun<'_>) -> InitializerResult {
        let accepted = self
            .machine
            .process_event(TextGeneratorInitializerEvents::EventRun(event))
            .is_ok();
        if !accepted {
            self.machine.context_mut().error = InitializerError::Unexpected;
        }
        let context = self.machine.context();
        let result = InitializerResult {
            accepted: accepted && context.error == InitializerError::None && context.phase_accepted,
            error: context.error,
            phase_code: context.phase_code,
            buffers_ready: context.buffers_ready,
        };
        if result.accepted {
            if let Some(callback) = event.on_done {
                callback(&result);
            }
        } else if let Some(callback) = event.on_error {
            callback(&result);
        }
        result
    }
    /// Processes one bounded initialization event to completion.
    /// Source-compatible initializer dispatch spelling.
    pub fn run(&mut self, event: EventRun) -> InitializerResult {
        self.process_event(event)
    }

    /// Drives the explicit unexpected-event path and returns its result.
    pub fn process_unexpected(&mut self) -> InitializerResult {
        self.machine.context_mut().error = InitializerError::Unexpected;
        self.machine.context_mut().phase_accepted = false;
        self.machine.set_state(TextGeneratorInitializerStates::Idle);
        InitializerResult {
            accepted: false,
            error: InitializerError::Unexpected,
            phase_code: self.machine.context().phase_code,
            buffers_ready: false,
        }
    }

    #[must_use]
    pub fn state(&self) -> &TextGeneratorInitializerStates {
        self.machine.state()
    }
    #[must_use]
    pub fn is(&self, state: &TextGeneratorInitializerStates) -> bool {
        self.machine.is(state)
    }
    #[must_use]
    pub fn context(&self) -> &TextGeneratorInitializerContext {
        self.machine.context()
    }
    #[must_use]
    pub fn result(&self) -> InitializerResult {
        let c = self.machine.context();
        InitializerResult {
            accepted: c.error == InitializerError::None && c.phase_accepted,
            error: c.error,
            phase_code: c.phase_code,
            buffers_ready: c.buffers_ready,
        }
    }
    #[must_use]
    pub fn last_error(&self) -> InitializerError {
        self.machine.context().error
    }
}

/// Compatibility alias for callers that use the shorter actor name.
pub type InitializerActor = TextGeneratorInitializerActor;

#[cfg(test)]
#[allow(clippy::trivially_copy_pass_by_ref)]
mod tests {
    use super::{EventRun, InitializerError, SelectionMode, TextGeneratorInitializerActor};

    #[test]
    fn run_completes_full_sample_logits_pipeline() {
        let mut actor = TextGeneratorInitializerActor::new();
        let result = actor.run(EventRun::valid());

        assert!(result.accepted);
        assert_eq!(result.error, InitializerError::None);
        assert!(result.buffers_ready);
        assert_eq!(actor.context().phase_code(), 0);
        assert!(actor.context().backend_ready());
    }

    #[test]
    fn preselected_argmax_uses_logits_only_configuration() {
        let mut event = EventRun::valid();
        event.selection_mode = SelectionMode::PreselectedArgmax;
        event.sampler_configured = false;
        event.sampler_buffers_ready = false;
        event.logits_ready = true;

        let mut actor = TextGeneratorInitializerActor::new();
        let result = actor.run(event);

        assert!(result.accepted);
        assert_eq!(result.error, InitializerError::None);
        assert!(result.buffers_ready);
    }

    #[test]
    fn conditioner_invalid_code_is_reported_as_invalid_request() {
        let mut event = EventRun::valid();
        event.conditioner_bind_accepted = false;
        event.conditioner_bind_code = 4;

        let mut actor = TextGeneratorInitializerActor::new();
        let result = actor.run(event);

        assert!(!result.accepted);
        assert_eq!(result.error, InitializerError::InvalidRequest);
        assert_eq!(result.phase_code, 4);
    }

    #[test]
    fn memory_geometry_gap_rejects_before_graph_reservation() {
        let mut event = EventRun::valid();
        event.max_blocks = 2;

        let mut actor = TextGeneratorInitializerActor::new();
        let result = actor.run(event);

        assert!(!result.accepted);
        assert_eq!(result.error, InitializerError::InvalidRequest);
        assert_eq!(result.phase_code, 0);
        assert!(!result.buffers_ready);
    }

    static DONE_CALLS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
    static ERROR_CALLS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

    fn on_done(_: &super::InitializerResult) {
        DONE_CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }

    fn on_error(_: &super::InitializerResult) {
        ERROR_CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn run_invokes_only_matching_completion_channel() {
        DONE_CALLS.store(0, core::sync::atomic::Ordering::Relaxed);
        ERROR_CALLS.store(0, core::sync::atomic::Ordering::Relaxed);

        let mut accepted = EventRun::valid();
        accepted.on_done = Some(on_done);
        accepted.on_error = Some(on_error);
        let mut actor = TextGeneratorInitializerActor::new();
        assert!(actor.run(accepted).accepted);
        assert_eq!(DONE_CALLS.load(core::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(ERROR_CALLS.load(core::sync::atomic::Ordering::Relaxed), 0);

        let mut rejected = EventRun::valid();
        rejected.valid_request = false;
        rejected.on_done = Some(on_done);
        rejected.on_error = Some(on_error);
        assert!(!actor.run(rejected).accepted);
        assert_eq!(DONE_CALLS.load(core::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(ERROR_CALLS.load(core::sync::atomic::Ordering::Relaxed), 1);
    }
    #[test]
    fn already_compatible_backend_skips_prepare_dispatch() {
        let mut event = EventRun::valid();
        event.backend_ready = true;
        event.backend_kv_positions_capacity = event.backend_n_ctx;
        event.backend_prepare_accepted = false;
        event.backend_prepare_code = 2;

        let mut actor = TextGeneratorInitializerActor::new();
        let result = actor.run(event);

        assert!(result.accepted);
        assert_eq!(result.error, InitializerError::None);
        assert!(result.buffers_ready);
    }

    #[test]
    fn unknown_conditioner_code_is_backend_error() {
        let mut event = EventRun::valid();
        event.conditioner_bind_accepted = false;
        event.conditioner_bind_code = 64;

        let mut actor = TextGeneratorInitializerActor::new();
        let result = actor.run(event);

        assert!(!result.accepted);
        assert_eq!(result.error, InitializerError::Backend);
        assert_eq!(result.phase_code, 64);
    }
}
