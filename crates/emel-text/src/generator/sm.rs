//! Source-aligned, synchronous text-generation orchestration state machine.
//!
//! The actor owns only bounded request/result bookkeeping.  Model, tokenizer,
//! graph, sampler, renderer, and streaming work are represented by explicit
//! phase outcomes supplied by events; no phase is deferred or assumed to
//! succeed.  Output and token-id storage remain caller-owned.

#![allow(
    clippy::enum_variant_names,
    clippy::missing_errors_doc,
    clippy::missing_const_for_fn,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    dead_code,
    missing_docs
)]

use sml::sml;

/// Errors published by the generator contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum GeneratorError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    Backend = 2,
    UnexpectedEvent = 3,
    OutputCapacity = 4,
}

/// Generation stages used by the public compatibility event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GenerateStep {
    /// Prompt conditioning/prefill entry.
    Prefill,
    /// One decode step.
    Decode,
    /// Result publication/finalization.
    Result,
    /// No stage supplied.
    #[default]
    None,
}

/// Explicit result of an injected subphase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SubphaseOutcome {
    /// Whether the subphase accepted and completed the request.
    pub accepted: bool,
    /// Typed outcome error when `accepted` is false.
    pub error: GeneratorError,
}

impl SubphaseOutcome {
    #[must_use]
    pub const fn ok() -> Self {
        Self { accepted: true, error: GeneratorError::None }
    }

    #[must_use]
    pub const fn invalid_request() -> Self {
        Self { accepted: false, error: GeneratorError::InvalidRequest }
    }

    #[must_use]
    pub const fn backend() -> Self {
        Self { accepted: false, error: GeneratorError::Backend }
    }
}

/// Runtime initialization request.  The scalar fields are copied into actor
/// state; no borrowed request data is retained.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventInitializeRun {
    pub valid_request: bool,
    pub backend_available: bool,
}

/// Runtime reset request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventResetRun {
    pub valid_request: bool,
    pub backend_available: bool,
}

/// Compatibility stage request.  The injected phase outcome is explicit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventGenerateRun {
    pub step: GenerateStep,
    pub valid_request: bool,
    pub backend_available: bool,
}

/// Runtime flush request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventFlushRun {
    pub valid_request: bool,
    pub backend_available: bool,
}

/// Runtime stream request.  A stream phase is explicit and synchronous; it
/// never queues hidden work for a later event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventStreamRun {
    pub valid_request: bool,
    pub backend_available: bool,
    pub final_chunk: bool,
}

/// Event-only diagnostics capture.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventCaptureDiagnostics;

/// Event-only graph lifecycle capture.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventCaptureGraphLifecycle;

/// Select the benchmark lane without sharing actor state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventConfigureBenchmarkLane {
    pub multithreaded: bool,
}
/// Caller-owned output request for a complete generation/result phase.
/// `rendered` is read synchronously and is never retained by the actor.
pub struct GenerateRequest<'a> {
    pub valid_request: bool,
    pub backend_available: bool,
    pub rendered: &'a [u8],
    pub output: &'a mut [u8],
    pub generated_token_ids: &'a mut [i32],
}

impl<'a> GenerateRequest<'a> {
    #[must_use]
    pub const fn new(rendered: &'a [u8], output: &'a mut [u8], generated_token_ids: &'a mut [i32]) -> Self {
        Self { valid_request: true, backend_available: true, rendered, output, generated_token_ids }
    }
}

/// Caller-owned output request for stream publication.
pub struct StreamRequest<'a> {
    pub valid_request: bool,
    pub backend_available: bool,
    pub final_chunk: bool,
    pub chunk: &'a [u8],
    pub output: &'a mut [u8],
}

/// Result copied out of the actor after a synchronous operation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GenerationResult {
    pub output_length: usize,
    pub tokens_generated: u32,
    pub error: GeneratorError,
}

sml! {
    TextGenerator {
        "initializing"_s <= *"uninitialized"_s + event<EventInitializeRun> [valid_initialize] / begin_initialize,
        "initializing"_s <= "ready"_s + event<EventInitializeRun> [valid_initialize] / begin_initialize,
        "initializing"_s <= "errored"_s + event<EventInitializeRun> [valid_initialize] / begin_initialize,
        "initialization_error"_s <= "uninitialized"_s + event<EventInitializeRun> [invalid_initialize] / reject_initialize,
        "initialization_error"_s <= "ready"_s + event<EventInitializeRun> [invalid_initialize] / reject_initialize,
        "initialization_error"_s <= "errored"_s + event<EventInitializeRun> [invalid_initialize] / reject_initialize,
        "ready"_s <= "initializing"_s + completion<EventInitializeRun> [initialize_ok] / complete_initialize,
        "initialization_error"_s <= "initializing"_s + completion<EventInitializeRun> [initialize_invalid] / reject_initialize_backend,
        "errored"_s <= "initializing"_s + completion<EventInitializeRun> [initialize_backend_error] / reject_initialize_backend,

        "resetting"_s <= "ready"_s + event<EventResetRun> [valid_reset] / begin_reset,
        "ready"_s <= "resetting"_s + completion<EventResetRun> [reset_ok] / complete_reset,
        "errored"_s <= "resetting"_s + completion<EventResetRun> [reset_invalid] / reject_reset,
        "errored"_s <= "resetting"_s + completion<EventResetRun> [reset_backend_error] / reject_reset_backend,

        "prefilling"_s <= "ready"_s + event<EventGenerateRun> [valid_prefill] / begin_prefill,
        "decoding"_s <= "prefilling"_s + event<EventGenerateRun> [valid_decode] / complete_prefill,
        "ready"_s <= "decoding"_s + event<EventGenerateRun> [valid_result] / complete_result,
        "generation_error"_s <= "ready"_s + event<EventGenerateRun> [invalid_generate] / reject_generate,
        "generation_error"_s <= "prefilling"_s + event<EventGenerateRun> [invalid_generate] / reject_generate,
        "generation_error"_s <= "decoding"_s + event<EventGenerateRun> [invalid_generate] / reject_generate,
        "generation_error"_s <= "ready"_s + event<EventGenerateRun> [backend_failure] / reject_generate_backend,
        "generation_error"_s <= "prefilling"_s + event<EventGenerateRun> [backend_failure] / reject_generate_backend,
        "generation_error"_s <= "decoding"_s + event<EventGenerateRun> [backend_failure] / reject_generate_backend,

        "flushing"_s <= "decoding"_s + event<EventFlushRun> [valid_flush] / begin_flush,
        "flushing"_s <= "ready"_s + event<EventFlushRun> [valid_flush] / begin_flush,
        "ready"_s <= "flushing"_s + completion<EventFlushRun> [flush_ok] / complete_flush,
        "generation_error"_s <= "flushing"_s + completion<EventFlushRun> [flush_invalid] / reject_flush,
        "errored"_s <= "flushing"_s + completion<EventFlushRun> [flush_backend_error] / reject_flush_backend,

        "streaming"_s <= "ready"_s + event<EventStreamRun> [valid_stream] / begin_stream,
        "streaming"_s <= "decoding"_s + event<EventStreamRun> [valid_stream] / begin_stream,
        "ready"_s <= "streaming"_s + completion<EventStreamRun> [stream_final] / complete_stream,
        "decoding"_s <= "streaming"_s + completion<EventStreamRun> [stream_continue] / continue_stream,
        "generation_error"_s <= "streaming"_s + completion<EventStreamRun> [stream_invalid] / reject_stream,
        "errored"_s <= "streaming"_s + completion<EventStreamRun> [stream_backend_error] / reject_stream_backend,

        "unexpected"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "unexpected"_s <= "initializing"_s + unexpected_event<_> / on_unexpected_from_initializing,
        "unexpected"_s <= "initialization_error"_s + unexpected_event<_> / on_unexpected_from_initialization_error,
        "unexpected"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "unexpected"_s <= "resetting"_s + unexpected_event<_> / on_unexpected_from_resetting,
        "unexpected"_s <= "prefilling"_s + unexpected_event<_> / on_unexpected_from_prefilling,
        "unexpected"_s <= "decoding"_s + unexpected_event<_> / on_unexpected_from_decoding,
        "uninitialized"_s <= "uninitialized"_s + event<EventCaptureDiagnostics> / capture_diagnostics,
        "ready"_s <= "ready"_s + event<EventCaptureDiagnostics> / capture_diagnostics,
        "errored"_s <= "errored"_s + event<EventCaptureDiagnostics> / capture_diagnostics,
        "uninitialized"_s <= "uninitialized"_s + event<EventCaptureGraphLifecycle> / capture_graph_lifecycle,
        "ready"_s <= "ready"_s + event<EventCaptureGraphLifecycle> / capture_graph_lifecycle,
        "errored"_s <= "errored"_s + event<EventCaptureGraphLifecycle> / capture_graph_lifecycle,
        "uninitialized"_s <= "uninitialized"_s + event<EventConfigureBenchmarkLane> / configure_benchmark_lane,
        "ready"_s <= "ready"_s + event<EventConfigureBenchmarkLane> / configure_benchmark_lane,
        "errored"_s <= "errored"_s + event<EventConfigureBenchmarkLane> / configure_benchmark_lane,
        "unexpected"_s <= "flushing"_s + unexpected_event<_> / on_unexpected_from_flushing,
        "unexpected"_s <= "streaming"_s + unexpected_event<_> / on_unexpected_from_streaming,
        "unexpected"_s <= "generation_error"_s + unexpected_event<_> / on_unexpected_from_generation_error,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "uninitialized"_s <= "uninitialized"_s + event<EventCaptureDiagnostics> / capture_diagnostics,
        "ready"_s <= "ready"_s + event<EventCaptureDiagnostics> / capture_diagnostics,
        "errored"_s <= "errored"_s + event<EventCaptureDiagnostics> / capture_diagnostics,
        "uninitialized"_s <= "uninitialized"_s + event<EventCaptureGraphLifecycle> / capture_graph_lifecycle,
        "ready"_s <= "ready"_s + event<EventCaptureGraphLifecycle> / capture_graph_lifecycle,
        "errored"_s <= "errored"_s + event<EventCaptureGraphLifecycle> / capture_graph_lifecycle,
        "uninitialized"_s <= "uninitialized"_s + event<EventConfigureBenchmarkLane> / configure_benchmark_lane,
        "ready"_s <= "ready"_s + event<EventConfigureBenchmarkLane> / configure_benchmark_lane,
        "errored"_s <= "errored"_s + event<EventConfigureBenchmarkLane> / configure_benchmark_lane,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

/// Bounded persistent and copied request/result state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TextGeneratorContext {
    phase: GeneratorPhase,
    error: GeneratorError,
    max_prompt_tokens: u32,
    max_generated_tokens: u32,
    tokens_generated: u32,
    output_length: usize,
    output_capacity: usize,
    prefill_count: u32,
    decode_count: u32,
    result_count: u32,
    flush_count: u32,
    stream_count: u32,
    sequence_live: bool,
    pending_backend: bool,
    pending_final_chunk: bool,
    diagnostics_count: u32,
    graph_lifecycle_count: u32,
    multithreaded_benchmark: bool,
}

/// Stable logical phase inspection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GeneratorPhase {
    #[default]
    Uninitialized,
    Initializing,
    InitializationError,
    Ready,
    Resetting,
    Prefilling,
    Decoding,
    Flushing,
    Streaming,
    GenerationError,
    Errored,
    Unexpected,
}

impl TextGeneratorContext {
    pub const fn phase(&self) -> GeneratorPhase { self.phase }
    pub const fn error(&self) -> GeneratorError { self.error }
    pub const fn max_prompt_tokens(&self) -> u32 { self.max_prompt_tokens }
    pub const fn max_generated_tokens(&self) -> u32 { self.max_generated_tokens }
    pub const fn tokens_generated(&self) -> u32 { self.tokens_generated }
    pub const fn output_length(&self) -> usize { self.output_length }
    pub const fn output_capacity(&self) -> usize { self.output_capacity }
    pub const fn prefill_count(&self) -> u32 { self.prefill_count }
    pub const fn decode_count(&self) -> u32 { self.decode_count }
    pub const fn result_count(&self) -> u32 { self.result_count }
    pub const fn flush_count(&self) -> u32 { self.flush_count }
    pub const fn stream_count(&self) -> u32 { self.stream_count }
    pub const fn sequence_live(&self) -> bool { self.sequence_live }

    fn set_error(&mut self, error: GeneratorError, phase: GeneratorPhase) -> Result<(), ()> {
        self.error = error;
        self.phase = phase;
        Ok(())
    }

    fn unexpected(&mut self) -> Result<(), ()> {
        self.set_error(GeneratorError::UnexpectedEvent, GeneratorPhase::Unexpected)
    }
}


impl TextGeneratorStateMachineContext for TextGeneratorContext {
    fn begin_initialize(&mut self, event: &EventInitializeRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Initializing;
        self.error = GeneratorError::None;
        self.max_prompt_tokens = 4096;
        self.max_generated_tokens = 4096;
        self.tokens_generated = 0;
        self.output_length = 0;
        self.output_capacity = 0;
        self.pending_backend = event.backend_available;
        Ok(())
    }
    fn complete_initialize(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Ready;
        self.error = GeneratorError::None;
        self.sequence_live = false;
        Ok(())
    }
    fn reject_initialize(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        self.set_error(GeneratorError::InvalidRequest, GeneratorPhase::InitializationError)
    }
    fn reject_initialize_backend(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        self.set_error(GeneratorError::Backend, GeneratorPhase::Errored)
    }
    fn initialize_ok(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(event.backend_available && self.pending_backend)
    }
    fn initialize_invalid(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(!event.valid_request || event.max_prompt_tokens == 0 || event.max_generated_tokens == 0)
    }
    fn initialize_backend_error(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(event.valid_request && !event.backend_available)
    }
    fn valid_initialize(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(event.valid_request)
    }
    fn invalid_initialize(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(!event.valid_request)
    }

    fn begin_reset(&mut self, event: &EventResetRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Resetting;
        self.error = GeneratorError::None;
        self.pending_backend = event.backend_available;
        Ok(())
    }
    fn complete_reset(&mut self, _event: &EventResetRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Ready;
        self.error = GeneratorError::None;
        self.sequence_live = false;
        self.tokens_generated = 0;
        self.output_length = 0;
        Ok(())
    }
    fn reject_reset(&mut self, _event: &EventResetRun) -> Result<(), ()> {
        self.set_error(GeneratorError::InvalidRequest, GeneratorPhase::Errored)
    }
    fn reject_reset_backend(&mut self, _event: &EventResetRun) -> Result<(), ()> {
        self.set_error(GeneratorError::Backend, GeneratorPhase::Errored)
    }
    fn valid_reset(&self, event: &EventResetRun) -> Result<bool, ()> { Ok(event.valid_request) }
    fn reset_ok(&self, event: &EventResetRun) -> Result<bool, ()> {
        Ok(event.backend_available && self.pending_backend)
    }
    fn reset_invalid(&self, event: &EventResetRun) -> Result<bool, ()> {
        Ok(!event.valid_request)
    }
    fn reset_backend_error(&self, event: &EventResetRun) -> Result<bool, ()> {
        Ok(event.valid_request && !event.backend_available)
    }

    fn begin_prefill(&mut self, event: &EventGenerateRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Prefilling;
        self.error = GeneratorError::None;
        self.pending_backend = event.backend_available;
        self.sequence_live = true;
        self.output_length = 0;
        Ok(())
    }
    fn complete_prefill(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Decoding;
        self.prefill_count = self.prefill_count.saturating_add(1);
        Ok(())
    }
    fn complete_result(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Ready;
        self.decode_count = self.decode_count.saturating_add(1);
        self.result_count = self.result_count.saturating_add(1);
        self.tokens_generated = self.tokens_generated.saturating_add(1);
        Ok(())
    }
    fn reject_generate(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        self.set_error(GeneratorError::InvalidRequest, GeneratorPhase::GenerationError)
    }
    fn reject_generate_backend(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        self.set_error(GeneratorError::Backend, GeneratorPhase::GenerationError)
    }
    fn valid_prefill(&self, event: &EventGenerateRun) -> Result<bool, ()> {
        Ok(event.valid_request && event.backend_available && event.step == GenerateStep::Prefill)
    }
    fn valid_decode(&self, event: &EventGenerateRun) -> Result<bool, ()> {
        Ok(event.valid_request && event.backend_available && event.step == GenerateStep::Decode)
    }
    fn valid_result(&self, event: &EventGenerateRun) -> Result<bool, ()> {
        Ok(event.valid_request && event.backend_available && event.step == GenerateStep::Result)
    }
    fn invalid_generate(&self, event: &EventGenerateRun) -> Result<bool, ()> {
        Ok(!event.valid_request || event.step == GenerateStep::None)
    }
    fn backend_failure(&self, event: &EventGenerateRun) -> Result<bool, ()> {
        Ok(event.valid_request && !event.backend_available)
    }

    fn begin_flush(&mut self, event: &EventFlushRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Flushing;
        self.error = GeneratorError::None;
        self.pending_backend = event.backend_available;
        Ok(())
    }
    fn complete_flush(&mut self, _event: &EventFlushRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Ready;
        self.flush_count = self.flush_count.saturating_add(1);
        self.result_count = self.result_count.saturating_add(1);
        self.sequence_live = false;
        Ok(())
    }
    fn reject_flush(&mut self, _event: &EventFlushRun) -> Result<(), ()> {
        self.set_error(GeneratorError::InvalidRequest, GeneratorPhase::GenerationError)
    }
    fn reject_flush_backend(&mut self, _event: &EventFlushRun) -> Result<(), ()> {
        self.set_error(GeneratorError::Backend, GeneratorPhase::Errored)
    }
    fn valid_flush(&self, event: &EventFlushRun) -> Result<bool, ()> { Ok(event.valid_request) }
    fn flush_ok(&self, event: &EventFlushRun) -> Result<bool, ()> {
        Ok(event.backend_available && self.pending_backend)
    }
    fn flush_invalid(&self, event: &EventFlushRun) -> Result<bool, ()> { Ok(!event.valid_request) }
    fn flush_backend_error(&self, event: &EventFlushRun) -> Result<bool, ()> {
        Ok(event.valid_request && !event.backend_available)
    }

    fn begin_stream(&mut self, event: &EventStreamRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Streaming;
        self.error = GeneratorError::None;
        self.pending_backend = event.backend_available;
        self.pending_final_chunk = event.final_chunk;
        Ok(())
    }
    fn complete_stream(&mut self, _event: &EventStreamRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Ready;
        self.stream_count = self.stream_count.saturating_add(1);
        self.result_count = self.result_count.saturating_add(1);
        self.sequence_live = false;
        Ok(())
    }
    fn continue_stream(&mut self, _event: &EventStreamRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Decoding;
        self.stream_count = self.stream_count.saturating_add(1);
        Ok(())
    }
    fn reject_stream(&mut self, _event: &EventStreamRun) -> Result<(), ()> {
        self.set_error(GeneratorError::InvalidRequest, GeneratorPhase::GenerationError)
    }
    fn reject_stream_backend(&mut self, _event: &EventStreamRun) -> Result<(), ()> {
        self.set_error(GeneratorError::Backend, GeneratorPhase::Errored)
    }
    fn valid_stream(&self, event: &EventStreamRun) -> Result<bool, ()> {
        Ok(event.valid_request)
    }
    fn stream_final(&self, event: &EventStreamRun) -> Result<bool, ()> {
        Ok(event.final_chunk && event.backend_available && self.pending_backend)
    }
    fn stream_continue(&self, event: &EventStreamRun) -> Result<bool, ()> {
        Ok(!event.final_chunk && event.backend_available && self.pending_backend)
    }
    fn stream_invalid(&self, event: &EventStreamRun) -> Result<bool, ()> {
        Ok(!event.valid_request)
    }
    fn stream_backend_error(&self, event: &EventStreamRun) -> Result<bool, ()> {
        Ok(event.valid_request && !event.backend_available)
    }

    fn on_unexpected_from_uninitialized(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_initializing(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_initialization_error(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_resetting(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_prefilling(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_decoding(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_flushing(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_streaming(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_generation_error(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> { self.unexpected() }
}

/// Public single-writer synchronous wrapper.
pub struct TextGeneratorActor {
    machine: TextGeneratorStateMachine<TextGeneratorContext>,
}

impl Default for TextGeneratorActor {
    fn default() -> Self { Self::new() }
}

impl TextGeneratorActor {
    #[must_use]
    pub fn new() -> Self {
        Self { machine: TextGeneratorStateMachine::new(TextGeneratorContext::default()) }
    }

    pub fn state(&self) -> &TextGeneratorStates { self.machine.state() }
    pub fn is(&self, state: &TextGeneratorStates) -> bool { self.machine.is(state) }
    pub fn context(&self) -> &TextGeneratorContext { self.machine.context() }

    pub fn initialize(&mut self, event: EventInitializeRun) -> Result<(), GeneratorError> {
        self.machine.process_event(TextGeneratorEvents::EventInitializeRun(event)).map_err(|_| self.context().error())?;
        self.result()
    }

    pub fn reset(&mut self, event: EventResetRun) -> Result<(), GeneratorError> {
        self.machine.process_event(TextGeneratorEvents::EventResetRun(event)).map_err(|_| self.context().error())?;
        self.result()
    }

    pub fn generate(&mut self, event: EventGenerateRun) -> Result<(), GeneratorError> {
        self.machine.process_event(TextGeneratorEvents::EventGenerateRun(event)).map_err(|_| self.context().error())?;
        self.result()
    }

    pub fn flush(&mut self, event: EventFlushRun) -> Result<(), GeneratorError> {
        self.machine.process_event(TextGeneratorEvents::EventFlushRun(event)).map_err(|_| self.context().error())?;
        self.result()
    }

    pub fn stream(&mut self, event: EventStreamRun) -> Result<(), GeneratorError> {
        self.machine.process_event(TextGeneratorEvents::EventStreamRun(event)).map_err(|_| self.context().error())?;
        self.result()
    }
    pub fn capture_diagnostics(&mut self) -> Result<(), GeneratorError> {
        self.machine
            .process_event(TextGeneratorEvents::EventCaptureDiagnostics(EventCaptureDiagnostics))
            .map_err(|_| self.context().error())?;
        self.result()
    }

    pub fn capture_graph_lifecycle(&mut self) -> Result<(), GeneratorError> {
        self.machine
            .process_event(TextGeneratorEvents::EventCaptureGraphLifecycle(EventCaptureGraphLifecycle))
            .map_err(|_| self.context().error())?;
        self.result()
    }

    pub fn configure_benchmark_lane(&mut self, multithreaded: bool) -> Result<(), GeneratorError> {
        self.machine
            .process_event(TextGeneratorEvents::EventConfigureBenchmarkLane(EventConfigureBenchmarkLane { multithreaded }))
            .map_err(|_| self.context().error())?;
        self.result()
    }

    /// Runs one complete caller-owned result copy synchronously.  The machine
    /// is advanced first; output is copied only after the corresponding phase
    /// succeeds, and overflow is reported without partial copying.
    pub fn generate_into(&mut self, request: GenerateRequest<'_>) -> Result<GenerationResult, GeneratorError> {
        if request.rendered.len() > request.output.len() {
            let _ = self.machine.context_mut().set_error(GeneratorError::OutputCapacity, GeneratorPhase::GenerationError);
            return Err(GeneratorError::OutputCapacity);
        }
        self.generate(EventGenerateRun { step: GenerateStep::Prefill, valid_request: request.valid_request, backend_available: request.backend_available })?;
        self.generate(EventGenerateRun { step: GenerateStep::Decode, valid_request: request.valid_request, backend_available: request.backend_available })?;
        self.generate(EventGenerateRun { step: GenerateStep::Result, valid_request: request.valid_request, backend_available: request.backend_available })?;
        request.output[..request.rendered.len()].copy_from_slice(request.rendered);
        self.machine.context_mut().output_length = request.rendered.len();
        self.machine.context_mut().output_capacity = request.output.len();
        self.machine.context_mut().tokens_generated = request.generated_token_ids.len().min(u32::MAX as usize) as u32;
        Ok(self.generation_result())
    }

    /// Publishes one caller-owned stream chunk synchronously.
    pub fn stream_into(&mut self, request: StreamRequest<'_>) -> Result<GenerationResult, GeneratorError> {
        if request.chunk.len() > request.output.len() {
            let _ = self.machine.context_mut().set_error(GeneratorError::OutputCapacity, GeneratorPhase::GenerationError);
            return Err(GeneratorError::OutputCapacity);
        }
        self.stream(EventStreamRun { valid_request: request.valid_request, backend_available: request.backend_available, final_chunk: request.final_chunk })?;
        request.output[..request.chunk.len()].copy_from_slice(request.chunk);
        self.machine.context_mut().output_length = request.chunk.len();
        self.machine.context_mut().output_capacity = request.output.len();
        Ok(self.generation_result())
    }

    pub fn process_unexpected(&mut self) -> bool {
        let _ = self.machine.context_mut().unexpected();
        self.machine.set_state(TextGeneratorStates::Unexpected);
        false
    }

    fn result(&self) -> Result<(), GeneratorError> {
        match self.context().error() {
            GeneratorError::None => Ok(()),
            error => Err(error),
        }
    }

    fn generation_result(&self) -> GenerationResult {
        GenerationResult { output_length: self.context().output_length, tokens_generated: self.context().tokens_generated, error: self.context().error() }
    }
}

/// Compatibility alias for callers using the shorter actor name.
pub type TextGenerator = TextGeneratorActor;
