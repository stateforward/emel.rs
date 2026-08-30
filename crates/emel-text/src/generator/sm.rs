//! Allocation-free text-generation orchestration state machine.
//!
//! The generator owns only request-local bookkeeping. Model, tokenizer, graph,
//! and sampler work is supplied by typed events; no actor reads another actor's
//! state. The state topology mirrors the maintained C++ generator contract:
//! initialization, prefill, decode, result publication, and explicit failure
//! and unexpected-event paths.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::enum_variant_names,
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

/// Typed generator failure. Values are stable and suitable for an error out.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum GeneratorError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    Backend = 2,
    UnexpectedEvent = 3,
}

/// Generation work represented by a request event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GenerateStep {
    /// Evaluate the prompt and establish the initial decode contract.
    Prefill,
    /// Evaluate one or more decode slots.
    Decode,
    /// Publish the rendered output and return to the ready state.
    Result,
    /// No work was supplied.
    #[default]
    None,
}

/// Runtime initialization request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventInitializeRun {
    /// Whether request arguments and required model bindings are valid.
    pub valid_request: bool,
    /// Whether the injected initialization backend accepted the request.
    pub backend_available: bool,
}

/// Runtime generation request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventGenerateRun {
    /// Stage result represented by this event.
    pub step: GenerateStep,
    /// Whether request arguments and buffers are valid.
    pub valid_request: bool,
    /// Whether the injected stage backend completed successfully.
    pub backend_available: bool,
}

/// Request to capture diagnostics. Capture is deliberately event-only.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventCaptureDiagnostics;

/// Request to capture graph lifecycle information.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventCaptureGraphLifecycle;

/// Select a benchmark lane without sharing actor state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventConfigureBenchmarkLane {
    /// `true` selects the multithreaded benchmark lane.
    pub multithreaded: bool,
}

sml! {
    TextGenerator {
        "ready"_s <= *"uninitialized"_s + event<EventInitializeRun> [valid_initialize] / begin_initialize,
        "uninitialized"_s <= "uninitialized"_s + event<EventInitializeRun> [invalid_initialize] / reject_initialize,
        "errored"_s <= "uninitialized"_s + event<EventInitializeRun> [initialize_backend_failure] / reject_initialize_backend,
        "ready"_s <= "ready"_s + event<EventInitializeRun> [valid_initialize] / begin_initialize,
        "uninitialized"_s <= "ready"_s + event<EventInitializeRun> [invalid_initialize] / reject_initialize,
        "errored"_s <= "ready"_s + event<EventInitializeRun> [initialize_backend_failure] / reject_initialize_backend,

        "prefilling"_s <= "ready"_s + event<EventGenerateRun> [valid_prefill] / begin_prefill,
        "ready"_s <= "ready"_s + event<EventGenerateRun> [invalid_generate] / reject_generate,
        "errored"_s <= "ready"_s + event<EventGenerateRun> [backend_failure] / reject_generate_backend,
        "decoding"_s <= "prefilling"_s + event<EventGenerateRun> [valid_decode] / complete_prefill,
        "errored"_s <= "prefilling"_s + event<EventGenerateRun> [invalid_generate] / reject_generate,
        "errored"_s <= "prefilling"_s + event<EventGenerateRun> [backend_failure] / reject_generate_backend,
        "ready"_s <= "decoding"_s + event<EventGenerateRun> [valid_result] / complete_decode_and_publish,
        "errored"_s <= "decoding"_s + event<EventGenerateRun> [invalid_generate] / reject_generate,
        "errored"_s <= "decoding"_s + event<EventGenerateRun> [backend_failure] / reject_generate_backend,
        "ready"_s <= "result"_s + event<EventGenerateRun> [valid_result] / publish_result,
        "errored"_s <= "result"_s + event<EventGenerateRun> [invalid_generate] / reject_generate,
        "errored"_s <= "result"_s + event<EventGenerateRun> [backend_failure] / reject_generate_backend,

        "uninitialized"_s <= "uninitialized"_s + event<EventCaptureDiagnostics> / capture_diagnostics,
        "ready"_s <= "ready"_s + event<EventCaptureDiagnostics> / capture_diagnostics,
        "errored"_s <= "errored"_s + event<EventCaptureDiagnostics> / capture_diagnostics,
        "uninitialized"_s <= "uninitialized"_s + event<EventConfigureBenchmarkLane> / configure_benchmark_lane,
        "ready"_s <= "ready"_s + event<EventConfigureBenchmarkLane> / configure_benchmark_lane,
        "errored"_s <= "errored"_s + event<EventConfigureBenchmarkLane> / configure_benchmark_lane,
        "uninitialized"_s <= "uninitialized"_s + event<EventCaptureGraphLifecycle> / capture_graph_lifecycle,
        "ready"_s <= "ready"_s + event<EventCaptureGraphLifecycle> / capture_graph_lifecycle,
        "errored"_s <= "errored"_s + event<EventCaptureGraphLifecycle> / capture_graph_lifecycle,

        "unexpected"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "unexpected"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "unexpected"_s <= "prefilling"_s + unexpected_event<_> / on_unexpected_from_prefilling,
        "unexpected"_s <= "decoding"_s + unexpected_event<_> / on_unexpected_from_decoding,
        "unexpected"_s <= "result"_s + unexpected_event<_> / on_unexpected_from_result,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

/// Mutable state owned by the generator actor.
#[derive(Debug, Default)]
pub struct TextGeneratorContext {
    phase: GeneratorPhase,
    error: GeneratorError,
    prefill_count: u32,
    decode_count: u32,
    result_count: u32,
    diagnostics_count: u32,
    graph_lifecycle_count: u32,
    multithreaded_benchmark: bool,
}

/// Stable inspection of generator progress.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GeneratorPhase {
    #[default]
    Uninitialized,
    Ready,
    Prefilling,
    Decoding,
    Result,
    Errored,
    Unexpected,
}

impl TextGeneratorContext {
    /// Returns the last published failure.
    pub const fn error(&self) -> GeneratorError {
        self.error
    }
    /// Returns the logical lifecycle phase.
    pub const fn phase(&self) -> GeneratorPhase {
        self.phase
    }
    /// Returns the number of accepted prompt evaluations.
    pub const fn prefill_count(&self) -> u32 {
        self.prefill_count
    }
    /// Returns the number of accepted decode evaluations.
    pub const fn decode_count(&self) -> u32 {
        self.decode_count
    }
    /// Returns the number of published results.
    pub const fn result_count(&self) -> u32 {
        self.result_count
    }
}

impl TextGeneratorStateMachineContext for TextGeneratorContext {
    fn begin_initialize(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Ready;
        self.error = GeneratorError::None;
        Ok(())
    }
    fn reject_initialize(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Uninitialized;
        self.error = GeneratorError::InvalidRequest;
        Ok(())
    }
    fn reject_initialize_backend(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Errored;
        self.error = GeneratorError::Backend;
        Ok(())
    }
    fn begin_prefill(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Prefilling;
        self.error = GeneratorError::None;
        Ok(())
    }
    fn complete_prefill(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Decoding;
        self.prefill_count = self.prefill_count.saturating_add(1);
        Ok(())
    }
    fn complete_decode_and_publish(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Ready;
        self.decode_count = self.decode_count.saturating_add(1);
        self.result_count = self.result_count.saturating_add(1);
        Ok(())
    }
    fn publish_result(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Ready;
        self.result_count = self.result_count.saturating_add(1);
        Ok(())
    }
    fn reject_generate(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Errored;
        self.error = GeneratorError::InvalidRequest;
        Ok(())
    }
    fn reject_generate_backend(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Errored;
        self.error = GeneratorError::Backend;
        Ok(())
    }
    fn capture_diagnostics(&mut self, _event: &EventCaptureDiagnostics) -> Result<(), ()> {
        self.diagnostics_count = self.diagnostics_count.saturating_add(1);
        Ok(())
    }
    fn capture_graph_lifecycle(&mut self, _event: &EventCaptureGraphLifecycle) -> Result<(), ()> {
        self.graph_lifecycle_count = self.graph_lifecycle_count.saturating_add(1);
        Ok(())
    }
    fn configure_benchmark_lane(&mut self, event: &EventConfigureBenchmarkLane) -> Result<(), ()> {
        self.multithreaded_benchmark = event.multithreaded;
        Ok(())
    }

    fn on_unexpected_from_uninitialized(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_prefilling(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_decoding(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_result(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        self.unexpected()
    }

    fn valid_initialize(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(event.valid_request && event.backend_available)
    }
    fn invalid_initialize(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(!event.valid_request)
    }
    fn initialize_backend_failure(&self, event: &EventInitializeRun) -> Result<bool, ()> {
        Ok(event.valid_request && !event.backend_available)
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
}

impl TextGeneratorContext {
    fn unexpected(&mut self) -> Result<(), ()> {
        self.phase = GeneratorPhase::Unexpected;
        self.error = GeneratorError::UnexpectedEvent;
        Ok(())
    }
}

/// Public single-writer generator wrapper.
pub struct TextGeneratorActor {
    machine: TextGeneratorStateMachine<TextGeneratorContext>,
}

impl Default for TextGeneratorActor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextGeneratorActor {
    /// Creates an uninitialized actor with bounded, allocation-free state.
    pub fn new() -> Self {
        Self {
            machine: TextGeneratorStateMachine::new(TextGeneratorContext::default()),
        }
    }
    /// Returns the actor context for generated-state inspection.
    pub fn context(&self) -> &TextGeneratorContext {
        self.machine.context()
    }
    /// Initializes the actor and returns a typed failure when rejected.
    pub fn initialize(&mut self, event: EventInitializeRun) -> Result<(), GeneratorError> {
        self.machine
            .process_event(TextGeneratorEvents::EventInitializeRun(event))
            .map(|_| ())
            .map_err(|_| self.context().error())
            .and_then(|()| self.result())
    }

    /// Advances one explicit prefill, decode, or result stage.
    pub fn generate(&mut self, event: EventGenerateRun) -> Result<(), GeneratorError> {
        self.machine
            .process_event(TextGeneratorEvents::EventGenerateRun(event))
            .map(|_| ())
            .map_err(|_| self.context().error())
            .and_then(|()| self.result())
    }

    /// Captures diagnostics through an event-only transition.
    pub fn capture_diagnostics(&mut self) -> Result<(), GeneratorError> {
        self.machine
            .process_event(TextGeneratorEvents::EventCaptureDiagnostics(
                EventCaptureDiagnostics,
            ))
            .map(|_| ())
            .map_err(|_| self.context().error())
            .and_then(|()| self.result())
    }

    /// Captures graph lifecycle information through an event-only transition.
    pub fn capture_graph_lifecycle(&mut self) -> Result<(), GeneratorError> {
        self.machine
            .process_event(TextGeneratorEvents::EventCaptureGraphLifecycle(
                EventCaptureGraphLifecycle,
            ))
            .map(|_| ())
            .map_err(|_| self.context().error())
            .and_then(|()| self.result())
    }

    /// Selects a benchmark lane without exposing another actor's state.
    pub fn configure_benchmark_lane(&mut self, multithreaded: bool) -> Result<(), GeneratorError> {
        self.machine
            .process_event(TextGeneratorEvents::EventConfigureBenchmarkLane(
                EventConfigureBenchmarkLane { multithreaded },
            ))
            .map(|_| ())
            .map_err(|_| self.context().error())
            .and_then(|()| self.result())
    }

    fn result(&self) -> Result<(), GeneratorError> {
        match self.context().error() {
            GeneratorError::None => Ok(()),
            error => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialization_and_generation_follow_explicit_lifecycle() {
        let mut actor = TextGeneratorActor::new();
        assert_eq!(actor.context().phase(), GeneratorPhase::Uninitialized);
        actor
            .initialize(EventInitializeRun {
                valid_request: true,
                backend_available: true,
            })
            .unwrap();
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);

        actor
            .generate(EventGenerateRun {
                step: GenerateStep::Prefill,
                valid_request: true,
                backend_available: true,
            })
            .unwrap();
        actor
            .generate(EventGenerateRun {
                step: GenerateStep::Decode,
                valid_request: true,
                backend_available: true,
            })
            .unwrap();
        actor
            .generate(EventGenerateRun {
                step: GenerateStep::Result,
                valid_request: true,
                backend_available: true,
            })
            .unwrap();

        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);
        assert_eq!(actor.context().prefill_count(), 1);
        assert_eq!(actor.context().decode_count(), 1);
        assert_eq!(actor.context().result_count(), 1);
    }

    #[test]
    fn initialization_and_generation_failures_are_typed() {
        let mut actor = TextGeneratorActor::new();
        assert_eq!(
            actor.initialize(EventInitializeRun {
                valid_request: false,
                backend_available: true,
            }),
            Err(GeneratorError::InvalidRequest)
        );

        let mut actor = TextGeneratorActor::new();
        assert_eq!(
            actor.initialize(EventInitializeRun {
                valid_request: true,
                backend_available: false,
            }),
            Err(GeneratorError::Backend)
        );

        let mut actor = TextGeneratorActor::new();
        assert_eq!(
            actor.generate(EventGenerateRun {
                step: GenerateStep::Prefill,
                valid_request: true,
                backend_available: true,
            }),
            Err(GeneratorError::UnexpectedEvent)
        );
    }

    #[test]
    fn event_only_observations_do_not_change_generation_phase() {
        let mut actor = TextGeneratorActor::new();
        actor.capture_diagnostics().unwrap();
        actor.capture_graph_lifecycle().unwrap();
        actor.configure_benchmark_lane(true).unwrap();
        assert_eq!(actor.context().phase(), GeneratorPhase::Uninitialized);
    }
}
