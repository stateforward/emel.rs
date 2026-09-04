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
    clippy::derive_partial_eq_without_eq,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::derivable_impls,
    dead_code,
    missing_docs
)]

use core::cell::RefCell;
use sml::sml;

use crate::detokenizer::VocabularyView;
use crate::renderer::sm::{
    FlushContext as RendererFlushContext, FlushRequest as RendererFlushRequest,
    InitializeContext as RendererInitializeContext, InitializeRequest as RendererInitializeRequest,
    RenderContext as RendererRenderContext, RenderRequest as RendererRenderRequest, RendererError,
    SequenceStatus, TextRenderer,
};
use crate::{ChatMessage, Conditioner, ConditionerError, conditioner_event};

use super::decode_wavefront::sm::{
    DecodeWavefrontError, EventRun as DecodeEventRun, TextGeneratorDecodeWavefrontActor,
};
use super::initializer::sm::{
    EventRun as InitializerEventRun, InitializerError, InitializerResult,
    TextGeneratorInitializerActor,
};
use super::layer::sm::{
    EventChunk4Run, EventChunk8Run, EventScalarRun, LayerError, TextGeneratorLayerChunk4Actor,
    TextGeneratorLayerChunk8Actor, TextGeneratorLayerScalarActor,
};
use super::prefill::sm::{self, PrefillError, TextGeneratorPrefillActor};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RendererPhase {
    pub output_length: usize,
    pub status: SequenceStatus,
}

/// Caller-owned, statically dispatched maintained renderer actor and channels.
pub struct RendererCollaborator<'event, 'v, V: VocabularyView + ?Sized> {
    pub actor: &'event RefCell<TextRenderer<'v, V>>,
    pub initialize_request: RendererInitializeRequest<'event>,
    pub initialize_context: &'event RefCell<RendererInitializeContext>,
}

impl<V: VocabularyView + ?Sized> RendererCollaborator<'_, '_, V> {
    fn initialize(&self) -> Result<(), RendererError> {
        self.actor
            .borrow_mut()
            .process_initialize(self.initialize_request, self.initialize_context)
    }
    fn render(
        &self,
        token_id: i32,
        sequence_id: i32,
        emit_special: bool,
        output: &mut [u8],
    ) -> Result<RendererPhase, RendererError> {
        let output = RefCell::new(output);
        let output_length = RefCell::new(0_usize);
        let status = RefCell::new(SequenceStatus::Running);
        let error = RefCell::new(RendererError::None);
        let context = RefCell::new(RendererRenderContext::default());
        self.actor.borrow_mut().process_render(
            RendererRenderRequest {
                token_id,
                sequence_id,
                emit_special,
                output: &output,
                output_length: &output_length,
                status: &status,
                error: &error,
                dispatch_done: None,
                dispatch_error: None,
            },
            &context,
        )?;
        let context_output_length = context.borrow().output_length;
        if context_output_length > output.borrow().len() {
            return Err(RendererError::InvalidRequest);
        }
        Ok(RendererPhase {
            output_length: *output_length.borrow(),
            status: *status.borrow(),
        })
    }
    fn flush(&self, sequence_id: i32, output: &mut [u8]) -> Result<RendererPhase, RendererError> {
        let output = RefCell::new(output);
        let output_length = RefCell::new(0_usize);
        let status = RefCell::new(SequenceStatus::Running);
        let error = RefCell::new(RendererError::None);
        let context = RefCell::new(RendererFlushContext::default());
        self.actor.borrow_mut().process_flush(
            RendererFlushRequest {
                sequence_id,
                output: &output,
                output_length: &output_length,
                status: &status,
                error: &error,
                dispatch_done: None,
                dispatch_error: None,
            },
            &context,
        )?;
        let context_output_length = context.borrow().output_length;
        if context_output_length > output.borrow().len() {
            return Err(RendererError::InvalidRequest);
        }
        Ok(RendererPhase {
            output_length: *output_length.borrow(),
            status: *status.borrow(),
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoRenderer;

pub trait RendererDispatch {
    fn present(&self) -> bool;
    fn initialize(&mut self) -> Result<(), RendererError>;
    fn render(
        &mut self,
        token_id: i32,
        sequence_id: i32,
        emit_special: bool,
        output: &mut [u8],
    ) -> Result<RendererPhase, RendererError>;
    fn flush(
        &mut self,
        sequence_id: i32,
        output: &mut [u8],
    ) -> Result<RendererPhase, RendererError>;
}

impl RendererDispatch for NoRenderer {
    fn present(&self) -> bool {
        false
    }
    fn initialize(&mut self) -> Result<(), RendererError> {
        Ok(())
    }
    fn render(
        &mut self,
        _: i32,
        _: i32,
        _: bool,
        _: &mut [u8],
    ) -> Result<RendererPhase, RendererError> {
        Ok(RendererPhase::default())
    }
    fn flush(&mut self, _: i32, _: &mut [u8]) -> Result<RendererPhase, RendererError> {
        Ok(RendererPhase::default())
    }
}

impl<V: VocabularyView + ?Sized> RendererDispatch for RendererCollaborator<'_, '_, V> {
    fn present(&self) -> bool {
        true
    }
    fn initialize(&mut self) -> Result<(), RendererError> {
        Self::initialize(self)
    }
    fn render(
        &mut self,
        token_id: i32,
        sequence_id: i32,
        emit_special: bool,
        output: &mut [u8],
    ) -> Result<RendererPhase, RendererError> {
        Self::render(self, token_id, sequence_id, emit_special, output)
    }
    fn flush(
        &mut self,
        sequence_id: i32,
        output: &mut [u8],
    ) -> Result<RendererPhase, RendererError> {
        Self::flush(self, sequence_id, output)
    }
}

fn map_conditioner_error(error: ConditionerError) -> GeneratorError {
    match error {
        ConditionerError::None => GeneratorError::None,
        ConditionerError::InvalidArgument => GeneratorError::InvalidRequest,
        ConditionerError::ModelInvalid | ConditionerError::Backend => GeneratorError::Backend,
        ConditionerError::Capacity => GeneratorError::OutputCapacity,
        ConditionerError::Untracked => GeneratorError::UnexpectedEvent,
    }
}

/// One caller-owned layer actor and its synchronous event.
pub enum LayerCollaborator<'event> {
    Scalar(
        &'event RefCell<TextGeneratorLayerScalarActor>,
        EventScalarRun<'event>,
    ),
    Chunk4(
        &'event RefCell<TextGeneratorLayerChunk4Actor>,
        EventChunk4Run<'event>,
    ),
    Chunk8(
        &'event RefCell<TextGeneratorLayerChunk8Actor>,
        EventChunk8Run<'event>,
    ),
}

/// One caller-owned decode-wavefront actor and its synchronous event.
pub struct DecodeCollaborator<'event> {
    pub actor: &'event RefCell<TextGeneratorDecodeWavefrontActor>,
    pub event: DecodeEventRun<'event>,
}

fn map_layer_error(error: LayerError) -> GeneratorError {
    match error {
        LayerError::None => GeneratorError::None,
        LayerError::InvalidRequest => GeneratorError::InvalidRequest,
        LayerError::Unexpected => GeneratorError::UnexpectedEvent,
        LayerError::Kernel | LayerError::UnsupportedRoute => GeneratorError::Backend,
    }
}

fn map_decode_error(error: DecodeWavefrontError) -> GeneratorError {
    match error {
        DecodeWavefrontError::None => GeneratorError::None,
        DecodeWavefrontError::InvalidRequest => GeneratorError::InvalidRequest,
        DecodeWavefrontError::Unexpected => GeneratorError::UnexpectedEvent,
        DecodeWavefrontError::IncompatibleLanes
        | DecodeWavefrontError::Backend
        | DecodeWavefrontError::LaneRejected
        | DecodeWavefrontError::MissingSelectedToken
        | DecodeWavefrontError::InvalidSelectedToken => GeneratorError::Backend,
    }
}
fn map_renderer_error(error: RendererError) -> GeneratorError {
    match error {
        RendererError::None => GeneratorError::None,
        RendererError::InvalidRequest => GeneratorError::InvalidRequest,
        RendererError::Backend | RendererError::ModelInvalid => GeneratorError::Backend,
        RendererError::Internal => GeneratorError::InvalidProducerResult,
        RendererError::Untracked => GeneratorError::UnexpectedEvent,
    }
}

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
    MissingProducer = 5,
    InvalidProducerResult = 6,
    InconsistentTokenCount = 7,
    MissingCollaborator = 8,
}

/// Scalar values reported by the synchronous result producer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GenerationProduced {
    pub valid: bool,
    pub output_length: usize,
    pub tokens_generated: u32,
}

/// Caller-owned handoff supplied to the synchronous result producer.
pub struct GenerateResultRequest<'a> {
    pub output: &'a mut [u8],
    pub generated_token_ids: &'a mut [i32],
    pub result: &'a mut GenerationProduced,
}

pub type ResultProducer = for<'a> fn(GenerateResultRequest<'a>) -> Result<(), GeneratorError>;

/// Generation stages used by the public compatibility event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GenerateStep {
    Prefill,
    Decode,
    Result,
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
        Self {
            accepted: true,
            error: GeneratorError::None,
        }
    }

    #[must_use]
    pub const fn invalid_request() -> Self {
        Self {
            accepted: false,
            error: GeneratorError::InvalidRequest,
        }
    }

    #[must_use]
    pub const fn backend() -> Self {
        Self {
            accepted: false,
            error: GeneratorError::Backend,
        }
    }
}

/// Runtime initialization request. Scalar fields are copied into actor state;
/// the initializer dependency and event remain caller-owned and are dispatched
/// synchronously before completion guards run.
#[derive(Clone, Copy, Default)]
pub struct EventInitializeRun<'event> {
    pub valid_request: bool,
    pub backend_available: bool,
    pub initializer: Option<&'event RefCell<TextGeneratorInitializerActor>>,
    pub initializer_event: InitializerEventRun<'event>,
}

impl core::fmt::Debug for EventInitializeRun<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EventInitializeRun")
            .field("valid_request", &self.valid_request)
            .field("backend_available", &self.backend_available)
            .field("initializer", &self.initializer.is_some())
            .field("initializer_event", &self.initializer_event)
            .finish()
    }
}
impl<'event> EventInitializeRun<'event> {
    /// Builds a generator initialization event around a caller-owned actor.
    #[must_use]
    pub const fn new(
        valid_request: bool,
        backend_available: bool,
        initializer: &'event RefCell<TextGeneratorInitializerActor>,
        initializer_event: InitializerEventRun<'event>,
    ) -> Self {
        Self {
            valid_request,
            backend_available,
            initializer: Some(initializer),
            initializer_event,
        }
    }
}
/// Runtime reset request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventResetRun {
    pub valid_request: bool,
    pub backend_available: bool,
}

/// Compatibility stage request. The injected phase outcome is explicit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventGenerateRun {
    pub step: GenerateStep,
    pub valid_request: bool,
    pub backend_available: bool,
    /// Synchronous collaborator result consumed by transition guards.
    pub phase: SubphaseOutcome,
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

/// Bounded generator state copied to caller-owned storage by diagnostics capture.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GeneratorDiagnostics {
    pub phase: GeneratorPhase,
    pub error: GeneratorError,
    pub diagnostics_count: u32,
    pub graph_lifecycle_count: u32,
    pub prefill_count: u32,
    pub decode_count: u32,
    pub result_count: u32,
    pub flush_count: u32,
    pub stream_count: u32,
    pub tokens_generated: u32,
    pub output_length: usize,
    pub sequence_live: bool,
}

/// Bounded graph-lifecycle state copied to caller-owned storage.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GraphLifecycleDiagnostics {
    pub phase: GeneratorPhase,
    pub error: GeneratorError,
    pub graph_lifecycle_count: u32,
    pub diagnostics_count: u32,
    pub prefill_count: u32,
    pub decode_count: u32,
    pub result_count: u32,
    pub flush_count: u32,
    pub stream_count: u32,
    pub sequence_live: bool,
}

/// Diagnostics capture with caller-owned output and explicit acceptance.
#[derive(Clone, Copy, Debug)]
pub struct EventCaptureDiagnostics<'event> {
    pub out: &'event RefCell<GeneratorDiagnostics>,
    pub accepted: &'event RefCell<bool>,
}

/// Graph-lifecycle capture with caller-owned output and explicit acceptance.
#[derive(Clone, Copy, Debug)]
pub struct EventCaptureGraphLifecycle<'event> {
    pub out: &'event RefCell<GraphLifecycleDiagnostics>,
    pub accepted: &'event RefCell<bool>,
}

/// Select the benchmark lane without sharing actor state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventConfigureBenchmarkLane {
    pub multithreaded: bool,
}
/// Successful generation completion payload.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GenerationDone {
    pub tokens_generated: u32,
    pub output_length: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GenerationResult {
    pub output_length: usize,
    pub tokens_generated: u32,
    pub error: GeneratorError,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GenerationErrorEvent {
    pub error: GeneratorError,
    pub tokens_generated: u32,
    pub output_length: usize,
}

pub type DoneCallback = fn(GenerationDone) -> bool;
pub type ErrorCallback = fn(GenerationErrorEvent) -> bool;

/// Caller-owned output request for a complete generation/result phase.
pub struct GenerateRequest<'a, R = NoRenderer> {
    pub valid_request: bool,
    pub backend_available: bool,
    pub result_producer: Option<ResultProducer>,
    pub renderer: R,
    pub sequence_id: i32,
    pub emit_special: bool,
    pub selected_token_position: usize,
    pub messages: &'a [ChatMessage<'a>],
    pub formatter_available: bool,
    pub tokenizer_available: bool,
    pub model_valid: bool,
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub add_special: bool,
    pub parse_special: bool,
    pub token_capacity: usize,
    pub token_count: Option<&'a mut usize>,
    pub conditioner: Option<&'a RefCell<Conditioner>>,
    pub output: &'a mut [u8],
    pub generated_token_ids: &'a mut [i32],
    pub prefill: Option<&'a RefCell<TextGeneratorPrefillActor>>,
    pub prefill_event: Option<sm::EventRun>,
    pub layer: Option<LayerCollaborator<'a>>,
    pub decode: Option<DecodeCollaborator<'a>>,
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
    pub error_out: Option<&'a mut GeneratorError>,
}

impl<'a> GenerateRequest<'a, NoRenderer> {
    #[must_use]
    pub const fn new(output: &'a mut [u8], generated_token_ids: &'a mut [i32]) -> Self {
        Self {
            valid_request: true,
            backend_available: true,
            result_producer: None,
            renderer: NoRenderer,
            sequence_id: 0,
            emit_special: false,
            selected_token_position: 0,
            messages: &[],
            formatter_available: true,
            tokenizer_available: true,
            model_valid: true,
            add_generation_prompt: false,
            enable_thinking: false,
            add_special: true,
            parse_special: false,
            token_capacity: 0,
            token_count: None,
            conditioner: None,
            output,
            generated_token_ids,
            prefill: None,
            prefill_event: None,
            layer: None,
            decode: None,
            on_done: None,
            on_error: None,
            error_out: None,
        }
    }
    pub fn with_renderer<'v, V: VocabularyView + ?Sized>(
        self,
        renderer: RendererCollaborator<'a, 'v, V>,
    ) -> GenerateRequest<'a, RendererCollaborator<'a, 'v, V>> {
        let GenerateRequest {
            valid_request,
            backend_available,
            result_producer,
            renderer: _,
            sequence_id,
            emit_special,
            selected_token_position,
            messages,
            formatter_available,
            tokenizer_available,
            model_valid,
            add_generation_prompt,
            enable_thinking,
            add_special,
            parse_special,
            token_capacity,
            token_count,
            conditioner,
            output,
            generated_token_ids,
            prefill,
            prefill_event,
            layer,
            decode,
            on_done,
            on_error,
            error_out,
        } = self;
        GenerateRequest {
            valid_request,
            backend_available,
            result_producer,
            renderer,
            sequence_id,
            emit_special,
            selected_token_position,
            messages,
            formatter_available,
            tokenizer_available,
            model_valid,
            add_generation_prompt,
            enable_thinking,
            add_special,
            parse_special,
            token_capacity,
            token_count,
            conditioner,
            output,
            generated_token_ids,
            prefill,
            prefill_event,
            layer,
            decode,
            on_done,
            on_error,
            error_out,
        }
    }
}

impl<'a, R> GenerateRequest<'a, R> {
    #[must_use]
    pub const fn with_result_producer(mut self, producer: ResultProducer) -> Self {
        self.result_producer = Some(producer);
        self
    }

    /// Enables the prompt-conditioning mode using only caller-owned state.
    #[must_use]
    pub fn with_conditioner(
        mut self,
        conditioner: &'a RefCell<Conditioner>,
        messages: &'a [ChatMessage<'a>],
        token_count: &'a mut usize,
    ) -> Self {
        self.conditioner = Some(conditioner);
        self.messages = messages;
        self.token_capacity = self.generated_token_ids.len();
        self.token_count = Some(token_count);
        self
    }

    #[must_use]
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    pub const fn with_conditioning_flags(
        mut self,
        formatter_available: bool,
        tokenizer_available: bool,
        model_valid: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        add_special: bool,
        parse_special: bool,
    ) -> Self {
        self.formatter_available = formatter_available;
        self.tokenizer_available = tokenizer_available;
        self.model_valid = model_valid;
        self.add_generation_prompt = add_generation_prompt;
        self.enable_thinking = enable_thinking;
        self.add_special = add_special;
        self.parse_special = parse_special;
        self
    }

    #[must_use]
    pub fn with_prefill(
        mut self,
        prefill: &'a RefCell<TextGeneratorPrefillActor>,
        prefill_event: sm::EventRun,
    ) -> Self {
        self.prefill = Some(prefill);
        self.prefill_event = Some(prefill_event);
        self
    }

    #[must_use]
    pub fn with_layer(mut self, layer: LayerCollaborator<'a>) -> Self {
        self.layer = Some(layer);
        self
    }

    #[must_use]
    pub fn with_decode(mut self, decode: DecodeCollaborator<'a>) -> Self {
        self.decode = Some(decode);
        self
    }

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

    #[must_use]
    pub fn with_error_out(mut self, error_out: &'a mut GeneratorError) -> Self {
        self.error_out = Some(error_out);
        self
    }
}
/// Scalar values reported by the synchronous stream producer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StreamProduced {
    /// Producer completed the selected stream publication path and initialized scalars.
    pub valid: bool,
    pub output_length: usize,
    pub tokens_generated: u32,
}

/// Caller-owned handoff supplied to the synchronous stream producer.
pub struct StreamResultRequest<'a> {
    pub output: &'a mut [u8],
    pub generated_token_ids: &'a mut [i32],
    pub result: &'a mut StreamProduced,
}

/// Synchronous producer for one selected stream result path.
pub type StreamProducer = for<'a> fn(StreamResultRequest<'a>) -> Result<(), GeneratorError>;

/// Caller-owned request for one synchronous stream result publication.
pub struct StreamRequest<'a> {
    pub valid_request: bool,
    pub backend_available: bool,
    pub final_chunk: bool,
    pub result_producer: Option<StreamProducer>,
    pub output: &'a mut [u8],
    pub generated_token_ids: &'a mut [i32],
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
    pub error_out: Option<&'a mut GeneratorError>,
}

impl<'a> StreamRequest<'a> {
    #[must_use]
    pub const fn new(output: &'a mut [u8], generated_token_ids: &'a mut [i32]) -> Self {
        Self {
            valid_request: true,
            backend_available: true,
            final_chunk: true,
            result_producer: None,
            output,
            generated_token_ids,
            on_done: None,
            on_error: None,
            error_out: None,
        }
    }

    #[must_use]
    pub const fn with_result_producer(mut self, producer: StreamProducer) -> Self {
        self.result_producer = Some(producer);
        self
    }

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

    #[must_use]
    pub fn with_error_out(mut self, error_out: &'a mut GeneratorError) -> Self {
        self.error_out = Some(error_out);
        self
    }
}

sml! {
    TextGenerator<'event> {
        "initializing"_s <= *"uninitialized"_s + event<EventInitializeRun<'event>> [valid_initialize] / begin_initialize,
        "initializing"_s <= "ready"_s + event<EventInitializeRun<'event>> [valid_initialize] / begin_initialize,
        "initializing"_s <= "errored"_s + event<EventInitializeRun<'event>> [valid_initialize] / begin_initialize,
        "initializing"_s <= "initialization_error"_s + event<EventInitializeRun<'event>> [valid_initialize] / begin_initialize,
        "initialization_error"_s <= "uninitialized"_s + event<EventInitializeRun<'event>> [invalid_initialize] / reject_initialize,
        "initialization_error"_s <= "ready"_s + event<EventInitializeRun<'event>> [invalid_initialize] / reject_initialize,
        "initialization_error"_s <= "errored"_s + event<EventInitializeRun<'event>> [invalid_initialize] / reject_initialize,
        "ready"_s <= "initializing"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [initialize_ok] / complete_initialize,
        "initialization_error"_s <= "initializing"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [initialize_invalid] / reject_initialize,
        "errored"_s <= "initializing"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [initialize_backend_error] / reject_initialize_backend,
        "errored"_s <= "initializing"_s + completion<EventInitializeRun>(EventInitializeRun<'event>) [initialize_unexpected] / reject_initialize_unexpected,

        "uninitialized_error"_s <= "uninitialized"_s + event<EventResetRun> / reject_reset_uninitialized,
        "errored"_s <= "ready"_s + event<EventResetRun> [invalid_reset] / reject_reset,
        "errored"_s <= "generation_error"_s + event<EventResetRun> [invalid_reset] / reject_reset,
        "resetting"_s <= "ready"_s + event<EventResetRun> [valid_reset] / begin_reset,
        "resetting"_s <= "generation_error"_s + event<EventResetRun> [valid_reset] / begin_reset,
        "resetting"_s <= "errored"_s + event<EventResetRun> [valid_reset] / begin_reset,
        "ready"_s <= "resetting"_s + completion<EventResetRun> [reset_ok] / complete_reset,
        "errored"_s <= "resetting"_s + completion<EventResetRun> [reset_invalid] / reject_reset,
        "errored"_s <= "resetting"_s + completion<EventResetRun> [reset_backend_error] / reject_reset_backend,
        "ready"_s <= "errored"_s + completion<EventResetRun> [reset_error_complete] / complete_reset_error,
        "uninitialized"_s <= "uninitialized_error"_s + completion<EventResetRun> / complete_reset_uninitialized_error,

        "uninitialized_error"_s <= "uninitialized"_s + event<EventGenerateRun> / reject_uninitialized_generate,
        "prefilling"_s <= "ready"_s + event<EventGenerateRun> [valid_prefill] / begin_prefill,
        "prefilling"_s <= "generation_error"_s + event<EventGenerateRun> [valid_prefill] / begin_prefill,
        "decoding"_s <= "prefilling"_s + event<EventGenerateRun> [valid_decode] / complete_prefill,
        "ready"_s <= "decoding"_s + event<EventGenerateRun> [valid_result] / complete_result,
        "generation_error"_s <= "ready"_s + event<EventGenerateRun> [invalid_generate] / reject_generate,
        "generation_error"_s <= "generation_error"_s + event<EventGenerateRun> [invalid_generate] / reject_generate,
        "generation_error"_s <= "prefilling"_s + event<EventGenerateRun> [invalid_generate] / reject_generate,
        "generation_error"_s <= "decoding"_s + event<EventGenerateRun> [invalid_generate] / reject_generate,
        "generation_error"_s <= "ready"_s + event<EventGenerateRun> [backend_failure] / reject_generate_backend,
        "generation_error"_s <= "generation_error"_s + event<EventGenerateRun> [backend_failure] / reject_generate_backend,
        "generation_error"_s <= "prefilling"_s + event<EventGenerateRun> [backend_failure] / reject_generate_backend,
        "generation_error"_s <= "decoding"_s + event<EventGenerateRun> [backend_failure] / reject_generate_backend,
        "ready"_s <= "generation_error"_s + completion<EventGenerateRun> [generate_error_complete] / complete_generate_error,
        "uninitialized"_s <= "uninitialized_error"_s + completion<EventGenerateRun> / complete_generate_uninitialized_error,

        "uninitialized_error"_s <= "uninitialized"_s + event<EventFlushRun> / reject_flush_uninitialized,
        "generation_error"_s <= "ready"_s + event<EventFlushRun> [invalid_flush] / reject_flush,
        "generation_error"_s <= "generation_error"_s + event<EventFlushRun> [invalid_flush] / reject_flush,
        "flushing"_s <= "decoding"_s + event<EventFlushRun> [valid_flush] / begin_flush,
        "flushing"_s <= "ready"_s + event<EventFlushRun> [valid_flush] / begin_flush,
        "flushing"_s <= "generation_error"_s + event<EventFlushRun> [valid_flush] / begin_flush,
        "ready"_s <= "flushing"_s + completion<EventFlushRun> [flush_ok] / complete_flush,
        "generation_error"_s <= "flushing"_s + completion<EventFlushRun> [flush_invalid] / reject_flush,
        "errored"_s <= "flushing"_s + completion<EventFlushRun> [flush_backend_error] / reject_flush_backend,
        "ready"_s <= "generation_error"_s + completion<EventFlushRun> [flush_error_complete] / complete_flush_error,
        "ready"_s <= "errored"_s + completion<EventFlushRun> [flush_error_complete] / complete_flush_error,
        "uninitialized"_s <= "uninitialized_error"_s + completion<EventFlushRun> / complete_flush_uninitialized_error,

        "uninitialized_error"_s <= "uninitialized"_s + event<EventStreamRun> / reject_stream_uninitialized,
        "generation_error"_s <= "ready"_s + event<EventStreamRun> [stream_invalid] / reject_stream,
        "generation_error"_s <= "generation_error"_s + event<EventStreamRun> [stream_invalid] / reject_stream,
        "streaming"_s <= "ready"_s + event<EventStreamRun> [valid_stream] / begin_stream,
        "streaming"_s <= "decoding"_s + event<EventStreamRun> [valid_stream] / begin_stream,
        "streaming"_s <= "generation_error"_s + event<EventStreamRun> [valid_stream] / begin_stream,
        "ready"_s <= "streaming"_s + completion<EventStreamRun> [stream_final] / complete_stream,
        "decoding"_s <= "streaming"_s + completion<EventStreamRun> [stream_continue] / continue_stream,
        "generation_error"_s <= "streaming"_s + completion<EventStreamRun> [stream_invalid] / reject_stream,
        "errored"_s <= "streaming"_s + completion<EventStreamRun> [stream_backend_error] / reject_stream_backend,
        "ready"_s <= "generation_error"_s + completion<EventStreamRun> [stream_error_complete] / complete_stream_error,
        "ready"_s <= "errored"_s + completion<EventStreamRun> [stream_error_complete] / complete_stream_error,
        "uninitialized"_s <= "uninitialized_error"_s + completion<EventStreamRun> / complete_stream_uninitialized_error,

        "unexpected"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "unexpected"_s <= "uninitialized_error"_s + unexpected_event<_> / on_unexpected_from_uninitialized_error,
        "unexpected"_s <= "initializing"_s + unexpected_event<_> / on_unexpected_from_initializing,
        "unexpected"_s <= "initialization_error"_s + unexpected_event<_> / on_unexpected_from_initialization_error,
        "unexpected"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "unexpected"_s <= "resetting"_s + unexpected_event<_> / on_unexpected_from_resetting,
        "unexpected"_s <= "prefilling"_s + unexpected_event<_> / on_unexpected_from_prefilling,
        "unexpected"_s <= "decoding"_s + unexpected_event<_> / on_unexpected_from_decoding,
        "uninitialized"_s <= "uninitialized"_s + event<EventCaptureDiagnostics<'event>> / capture_diagnostics,
        "ready"_s <= "ready"_s + event<EventCaptureDiagnostics<'event>> / capture_diagnostics,
        "errored"_s <= "errored"_s + event<EventCaptureDiagnostics<'event>> / capture_diagnostics,
        "uninitialized"_s <= "uninitialized"_s + event<EventCaptureGraphLifecycle<'event>> / capture_graph_lifecycle,
        "ready"_s <= "ready"_s + event<EventCaptureGraphLifecycle<'event>> / capture_graph_lifecycle,
        "errored"_s <= "errored"_s + event<EventCaptureGraphLifecycle<'event>> / capture_graph_lifecycle,
        "uninitialized"_s <= "uninitialized"_s + event<EventConfigureBenchmarkLane> / configure_benchmark_lane,
        "ready"_s <= "ready"_s + event<EventConfigureBenchmarkLane> / configure_benchmark_lane,
        "errored"_s <= "errored"_s + event<EventConfigureBenchmarkLane> / configure_benchmark_lane,
        "unexpected"_s <= "flushing"_s + unexpected_event<_> / on_unexpected_from_flushing,
        "unexpected"_s <= "streaming"_s + unexpected_event<_> / on_unexpected_from_streaming,
        "unexpected"_s <= "generation_error"_s + unexpected_event<_> / on_unexpected_from_generation_error,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
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
    initializer_result: InitializerResult,
    diagnostics_count: u32,
    graph_lifecycle_count: u32,
    multithreaded_benchmark: bool,
}

/// Stable logical phase inspection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GeneratorPhase {
    #[default]
    Uninitialized,
    UninitializedError,
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
    pub const fn phase(&self) -> GeneratorPhase {
        self.phase
    }
    pub const fn error(&self) -> GeneratorError {
        self.error
    }
    pub const fn max_prompt_tokens(&self) -> u32 {
        self.max_prompt_tokens
    }
    pub const fn max_generated_tokens(&self) -> u32 {
        self.max_generated_tokens
    }
    pub const fn tokens_generated(&self) -> u32 {
        self.tokens_generated
    }
    pub const fn output_length(&self) -> usize {
        self.output_length
    }
    pub const fn output_capacity(&self) -> usize {
        self.output_capacity
    }
    pub const fn prefill_count(&self) -> u32 {
        self.prefill_count
    }
    pub const fn decode_count(&self) -> u32 {
        self.decode_count
    }
    pub const fn result_count(&self) -> u32 {
        self.result_count
    }
    pub const fn flush_count(&self) -> u32 {
        self.flush_count
    }
    pub const fn stream_count(&self) -> u32 {
        self.stream_count
    }
    pub const fn sequence_live(&self) -> bool {
        self.sequence_live
    }

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
    fn begin_initialize(&mut self, event: &EventInitializeRun<'_>) -> Result<(), ()> {
        self.phase = GeneratorPhase::Initializing;
        self.error = GeneratorError::None;
        self.max_prompt_tokens = 4096;
        self.max_generated_tokens = 4096;
        self.tokens_generated = 0;
        self.output_length = 0;
        self.initializer_result = event.initializer.map_or(
            InitializerResult {
                accepted: false,
                error: InitializerError::Backend,
                phase_code: 0,
                buffers_ready: false,
            },
            |initializer| {
                initializer
                    .borrow_mut()
                    .process_event(event.initializer_event)
            },
        );
        self.pending_backend = event.backend_available;
        Ok(())
    }
    fn complete_initialize(&mut self, _event: &EventInitializeRun<'_>) -> Result<(), ()> {
        self.phase = GeneratorPhase::Ready;
        self.error = GeneratorError::None;
        self.sequence_live = false;
        Ok(())
    }
    fn reject_initialize(&mut self, _event: &EventInitializeRun<'_>) -> Result<(), ()> {
        self.set_error(
            GeneratorError::InvalidRequest,
            GeneratorPhase::InitializationError,
        )
    }
    fn reject_initialize_backend(&mut self, _event: &EventInitializeRun<'_>) -> Result<(), ()> {
        self.set_error(GeneratorError::Backend, GeneratorPhase::Errored)
    }
    fn reject_initialize_unexpected(&mut self, _event: &EventInitializeRun<'_>) -> Result<(), ()> {
        self.set_error(GeneratorError::UnexpectedEvent, GeneratorPhase::Errored)
    }
    fn initialize_ok(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> {
        Ok(event.backend_available
            && self.pending_backend
            && self.initializer_result.accepted
            && self.initializer_result.error == InitializerError::None
            && self.initializer_result.buffers_ready)
    }
    fn initialize_invalid(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> {
        Ok(event.valid_request
            && !self.initializer_result.accepted
            && self.initializer_result.error == InitializerError::InvalidRequest)
    }
    fn initialize_backend_error(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> {
        Ok(event.valid_request
            && (!event.backend_available
                || self.initializer_result.error == InitializerError::Backend))
    }
    fn initialize_unexpected(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> {
        Ok(event.valid_request && self.initializer_result.error == InitializerError::Unexpected)
    }
    fn valid_initialize(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> {
        Ok(event.valid_request)
    }
    fn invalid_initialize(&self, event: &EventInitializeRun<'_>) -> Result<bool, ()> {
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
    fn valid_reset(&self, event: &EventResetRun) -> Result<bool, ()> {
        Ok(event.valid_request)
    }
    fn reset_ok(&self, event: &EventResetRun) -> Result<bool, ()> {
        Ok(event.backend_available && self.pending_backend)
    }
    fn reset_invalid(&self, event: &EventResetRun) -> Result<bool, ()> {
        Ok(!event.valid_request)
    }
    fn reset_backend_error(&self, event: &EventResetRun) -> Result<bool, ()> {
        Ok(event.valid_request && !event.backend_available)
    }
    fn reject_reset_uninitialized(&mut self, _event: &EventResetRun) -> Result<(), ()> {
        self.set_error(
            GeneratorError::InvalidRequest,
            GeneratorPhase::UninitializedError,
        )
    }
    fn complete_reset_uninitialized_error(&mut self, _event: &EventResetRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Uninitialized;
        Ok(())
    }
    fn complete_reset_error(&mut self, _event: &EventResetRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Ready;
        self.sequence_live = false;
        self.pending_final_chunk = false;
        Ok(())
    }
    fn invalid_reset(&self, event: &EventResetRun) -> Result<bool, ()> {
        Ok(!event.valid_request)
    }
    fn reset_error_complete(&self, _event: &EventResetRun) -> Result<bool, ()> {
        Ok(true)
    }

    fn complete_generate_error(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Ready;
        self.sequence_live = false;
        self.pending_final_chunk = false;
        Ok(())
    }
    fn complete_generate_uninitialized_error(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        self.phase = GeneratorPhase::Uninitialized;
        self.sequence_live = false;
        Ok(())
    }
    fn generate_error_complete(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        Ok(true)
    }
    fn reject_uninitialized_generate(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        self.set_error(
            GeneratorError::InvalidRequest,
            GeneratorPhase::UninitializedError,
        )
    }

    fn complete_flush_error(&mut self, _event: &EventFlushRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Ready;
        self.sequence_live = false;
        self.pending_final_chunk = false;
        Ok(())
    }
    fn complete_flush_uninitialized_error(&mut self, _event: &EventFlushRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Uninitialized;
        Ok(())
    }
    fn flush_error_complete(&self, _event: &EventFlushRun) -> Result<bool, ()> {
        Ok(true)
    }
    fn invalid_flush(&self, event: &EventFlushRun) -> Result<bool, ()> {
        Ok(!event.valid_request)
    }
    fn reject_flush_uninitialized(&mut self, _event: &EventFlushRun) -> Result<(), ()> {
        self.set_error(
            GeneratorError::InvalidRequest,
            GeneratorPhase::UninitializedError,
        )
    }

    fn complete_stream_error(&mut self, _event: &EventStreamRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Ready;
        self.sequence_live = false;
        self.pending_final_chunk = false;
        Ok(())
    }
    fn complete_stream_uninitialized_error(&mut self, _event: &EventStreamRun) -> Result<(), ()> {
        self.phase = GeneratorPhase::Uninitialized;
        Ok(())
    }
    fn stream_error_complete(&self, _event: &EventStreamRun) -> Result<bool, ()> {
        Ok(true)
    }
    fn reject_stream_uninitialized(&mut self, _event: &EventStreamRun) -> Result<(), ()> {
        self.set_error(
            GeneratorError::InvalidRequest,
            GeneratorPhase::UninitializedError,
        )
    }
    fn on_unexpected_from_uninitialized_error(&mut self) -> Result<(), ()> {
        self.unexpected()
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
        self.set_error(
            GeneratorError::InvalidRequest,
            GeneratorPhase::GenerationError,
        )
    }
    fn reject_generate_backend(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        self.set_error(GeneratorError::Backend, GeneratorPhase::GenerationError)
    }
    fn valid_prefill(&self, event: &EventGenerateRun) -> Result<bool, ()> {
        Ok(event.valid_request
            && event.backend_available
            && event.step == GenerateStep::Prefill
            && event.phase.accepted
            && event.phase.error == GeneratorError::None)
    }
    fn valid_decode(&self, event: &EventGenerateRun) -> Result<bool, ()> {
        Ok(event.valid_request
            && event.backend_available
            && event.step == GenerateStep::Decode
            && event.phase.accepted
            && event.phase.error == GeneratorError::None)
    }
    fn valid_result(&self, event: &EventGenerateRun) -> Result<bool, ()> {
        Ok(event.valid_request
            && event.backend_available
            && event.step == GenerateStep::Result
            && event.phase.accepted
            && event.phase.error == GeneratorError::None)
    }
    fn invalid_generate(&self, event: &EventGenerateRun) -> Result<bool, ()> {
        Ok(!event.valid_request
            || event.step == GenerateStep::None
            || (!event.phase.accepted && event.phase.error == GeneratorError::InvalidRequest))
    }
    fn backend_failure(&self, event: &EventGenerateRun) -> Result<bool, ()> {
        Ok(event.valid_request
            && (!event.backend_available
                || (!event.phase.accepted && event.phase.error == GeneratorError::Backend)))
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
        self.set_error(
            GeneratorError::InvalidRequest,
            GeneratorPhase::GenerationError,
        )
    }
    fn reject_flush_backend(&mut self, _event: &EventFlushRun) -> Result<(), ()> {
        self.set_error(GeneratorError::Backend, GeneratorPhase::Errored)
    }
    fn valid_flush(&self, event: &EventFlushRun) -> Result<bool, ()> {
        Ok(event.valid_request)
    }
    fn flush_ok(&self, event: &EventFlushRun) -> Result<bool, ()> {
        Ok(event.backend_available && self.pending_backend)
    }
    fn flush_invalid(&self, event: &EventFlushRun) -> Result<bool, ()> {
        Ok(!event.valid_request)
    }
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
        self.set_error(
            GeneratorError::InvalidRequest,
            GeneratorPhase::GenerationError,
        )
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
    fn capture_diagnostics(&mut self, event: &EventCaptureDiagnostics<'_>) -> Result<(), ()> {
        self.diagnostics_count = self.diagnostics_count.saturating_add(1);
        *event.out.borrow_mut() = GeneratorDiagnostics {
            phase: self.phase,
            error: self.error,
            diagnostics_count: self.diagnostics_count,
            graph_lifecycle_count: self.graph_lifecycle_count,
            prefill_count: self.prefill_count,
            decode_count: self.decode_count,
            result_count: self.result_count,
            flush_count: self.flush_count,
            stream_count: self.stream_count,
            tokens_generated: self.tokens_generated,
            output_length: self.output_length,
            sequence_live: self.sequence_live,
        };
        *event.accepted.borrow_mut() = true;
        Ok(())
    }

    fn capture_graph_lifecycle(
        &mut self,
        event: &EventCaptureGraphLifecycle<'_>,
    ) -> Result<(), ()> {
        self.graph_lifecycle_count = self.graph_lifecycle_count.saturating_add(1);
        *event.out.borrow_mut() = GraphLifecycleDiagnostics {
            phase: self.phase,
            error: self.error,
            graph_lifecycle_count: self.graph_lifecycle_count,
            diagnostics_count: self.diagnostics_count,
            prefill_count: self.prefill_count,
            decode_count: self.decode_count,
            result_count: self.result_count,
            flush_count: self.flush_count,
            stream_count: self.stream_count,
            sequence_live: self.sequence_live,
        };
        *event.accepted.borrow_mut() = true;
        Ok(())
    }

    fn configure_benchmark_lane(&mut self, event: &EventConfigureBenchmarkLane) -> Result<(), ()> {
        self.multithreaded_benchmark = event.multithreaded;
        Ok(())
    }

    fn on_unexpected_from_uninitialized(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_initializing(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_initialization_error(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_resetting(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_prefilling(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_decoding(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_flushing(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_streaming(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_generation_error(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
}

/// Public single-writer synchronous wrapper.
pub struct TextGeneratorActor {
    machine: TextGeneratorStateMachine<TextGeneratorContext>,
}

impl Default for TextGeneratorActor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextGeneratorActor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextGeneratorStateMachine::new(TextGeneratorContext::default()),
        }
    }

    pub fn state(&self) -> &TextGeneratorStates {
        self.machine.state()
    }
    pub fn is(&self, state: &TextGeneratorStates) -> bool {
        self.machine.is(state)
    }
    pub fn context(&self) -> &TextGeneratorContext {
        self.machine.context()
    }

    pub fn initialize(&mut self, event: EventInitializeRun<'_>) -> Result<(), GeneratorError> {
        let accepted = self
            .machine
            .process_event(TextGeneratorEvents::EventInitializeRun(event))
            .is_ok();
        let error = self.context().error();
        if !accepted {
            return Err(error);
        }
        self.result()
    }

    pub fn reset(&mut self, event: EventResetRun) -> Result<(), GeneratorError> {
        let accepted = self
            .machine
            .process_event(TextGeneratorEvents::EventResetRun(event))
            .is_ok();
        let error = self.context().error();
        if !accepted {
            return Err(error);
        }
        self.result()
    }

    pub fn generate(&mut self, event: EventGenerateRun) -> Result<(), GeneratorError> {
        let accepted = self
            .machine
            .process_event(TextGeneratorEvents::EventGenerateRun(event))
            .is_ok();
        let error = self.context().error();
        if !accepted {
            return Err(error);
        }
        self.result()
    }

    pub fn flush(&mut self, event: EventFlushRun) -> Result<(), GeneratorError> {
        let accepted = self
            .machine
            .process_event(TextGeneratorEvents::EventFlushRun(event))
            .is_ok();
        let error = self.context().error();
        if !accepted {
            return Err(error);
        }
        self.result()
    }

    pub fn stream(&mut self, event: EventStreamRun) -> Result<(), GeneratorError> {
        let accepted = self
            .machine
            .process_event(TextGeneratorEvents::EventStreamRun(event))
            .is_ok();
        let error = self.context().error();
        if !accepted {
            return Err(error);
        }
        self.result()
    }

    /// Captures bounded diagnostics synchronously into caller-owned storage.
    pub fn capture_diagnostics(
        &mut self,
        event: EventCaptureDiagnostics<'_>,
    ) -> Result<(), GeneratorError> {
        let accepted = self
            .machine
            .process_event(TextGeneratorEvents::EventCaptureDiagnostics(event))
            .is_ok();
        let error = self.context().error();
        if !accepted {
            return Err(error);
        }
        self.result()
    }

    /// Captures bounded graph-lifecycle state synchronously into caller-owned storage.
    pub fn capture_graph_lifecycle(
        &mut self,
        event: EventCaptureGraphLifecycle<'_>,
    ) -> Result<(), GeneratorError> {
        let accepted = self
            .machine
            .process_event(TextGeneratorEvents::EventCaptureGraphLifecycle(event))
            .is_ok();
        let error = self.context().error();
        if !accepted {
            return Err(error);
        }
        self.result()
    }

    pub fn configure_benchmark_lane(&mut self, multithreaded: bool) -> Result<(), GeneratorError> {
        let accepted = self
            .machine
            .process_event(TextGeneratorEvents::EventConfigureBenchmarkLane(
                EventConfigureBenchmarkLane { multithreaded },
            ))
            .is_ok();
        let error = self.context().error();
        if !accepted {
            return Err(error);
        }
        self.result()
    }

    /// Runs one complete caller-owned conditioning, prefill, decode, and result
    /// sequence synchronously, dispatching a maintained renderer actor when supplied.
    #[allow(clippy::too_many_lines)]
    pub fn generate_into<R: RendererDispatch>(
        &mut self,
        mut request: GenerateRequest<'_, R>,
    ) -> Result<GenerationResult, GeneratorError> {
        if !request.valid_request {
            return self.fail_generation(&mut request, GeneratorError::InvalidRequest);
        }
        if !request.backend_available {
            return self.fail_generation(&mut request, GeneratorError::Backend);
        }
        if !request.renderer.present() && request.result_producer.is_none() {
            return self.fail_generation(&mut request, GeneratorError::MissingProducer);
        }
        if request.prefill.is_none()
            || request.prefill_event.is_none()
            || request.layer.is_none()
            || request.decode.is_none()
        {
            return self.fail_generation(&mut request, GeneratorError::MissingCollaborator);
        }

        let conditioning_requested = request.conditioner.is_some() || !request.messages.is_empty();
        let conditioned_count = if conditioning_requested {
            let Some(conditioner) = request.conditioner else {
                return self.fail_generation(&mut request, GeneratorError::InvalidRequest);
            };
            let Some(token_count) = request.token_count.as_deref_mut() else {
                return self.fail_generation(&mut request, GeneratorError::InvalidRequest);
            };
            let prepared = conditioner
                .borrow_mut()
                .prepare(conditioner_event::Prepare {
                    messages: request.messages,
                    formatter_available: request.formatter_available,
                    tokenizer_available: request.tokenizer_available,
                    model_valid: request.model_valid,
                    token_capacity: request.token_capacity,
                    token_ids: request.generated_token_ids,
                    token_count,
                    add_generation_prompt: request.add_generation_prompt,
                    enable_thinking: request.enable_thinking,
                    add_special: request.add_special,
                    parse_special: request.parse_special,
                });
            match prepared {
                Ok(done) => Some(done.token_count.min(u32::MAX as usize) as u32),
                Err(error) => {
                    return self.fail_generation(&mut request, map_conditioner_error(error));
                }
            }
        } else {
            None
        };

        let prefill_phase = match (request.prefill, request.prefill_event.take()) {
            (Some(actor), Some(mut event)) => {
                if let Some(count) = conditioned_count {
                    event.prompt_token_count = count;
                    event.snapshot_sequence_length = count;
                }
                actor
                    .borrow_mut()
                    .process_event(event)
                    .map(|()| SubphaseOutcome::ok())
                    .map_err(|error| match error {
                        PrefillError::None => GeneratorError::None,
                        PrefillError::InvalidRequest => GeneratorError::InvalidRequest,
                        PrefillError::Backend => GeneratorError::Backend,
                        PrefillError::UnexpectedEvent => GeneratorError::UnexpectedEvent,
                    })
            }
            _ => Err(GeneratorError::MissingCollaborator),
        };
        let prefill_phase = match prefill_phase {
            Ok(outcome) => outcome,
            Err(error) => return self.fail_generation(&mut request, error),
        };
        if let Err(error) = self.generate(EventGenerateRun {
            step: GenerateStep::Prefill,
            valid_request: request.valid_request,
            backend_available: request.backend_available,
            phase: prefill_phase,
        }) {
            return self.fail_generation(&mut request, error);
        }

        let layer_phase = match request.layer.take() {
            None => Err(GeneratorError::MissingCollaborator),
            Some(LayerCollaborator::Scalar(actor, event)) => actor
                .borrow_mut()
                .process_event(event)
                .map(|()| SubphaseOutcome::ok())
                .map_err(map_layer_error),
            Some(LayerCollaborator::Chunk4(actor, event)) => actor
                .borrow_mut()
                .process_event(event)
                .map(|()| SubphaseOutcome::ok())
                .map_err(map_layer_error),
            Some(LayerCollaborator::Chunk8(actor, event)) => actor
                .borrow_mut()
                .process_event(event)
                .map(|()| SubphaseOutcome::ok())
                .map_err(map_layer_error),
        };
        let layer_phase = match layer_phase {
            Ok(outcome) => outcome,
            Err(error) => return self.fail_generation(&mut request, error),
        };
        let decode_collaborator = request
            .decode
            .take()
            .ok_or(GeneratorError::MissingCollaborator);
        let decode_collaborator = match decode_collaborator {
            Ok(collaborator) => collaborator,
            Err(error) => return self.fail_generation(&mut request, error),
        };
        let decode_phase = decode_collaborator
            .actor
            .borrow_mut()
            .process_event(decode_collaborator.event)
            .map(|()| SubphaseOutcome::ok())
            .map_err(map_decode_error);
        let decode_phase = match decode_phase {
            Ok(outcome) => outcome,
            Err(error) => return self.fail_generation(&mut request, error),
        };
        let selected_token = *decode_collaborator.event.selected_token.borrow();
        if !selected_token.valid || selected_token.token < 0 {
            return self.fail_generation(&mut request, GeneratorError::Backend);
        }
        if request.selected_token_position >= request.generated_token_ids.len() {
            return self.fail_generation(&mut request, GeneratorError::OutputCapacity);
        }
        request.generated_token_ids[request.selected_token_position] = selected_token.token;
        if let Err(error) = self.generate(EventGenerateRun {
            step: GenerateStep::Decode,
            valid_request: request.valid_request,
            backend_available: request.backend_available,
            phase: layer_phase,
        }) {
            return self.fail_generation(&mut request, error);
        }
        if let Err(error) = request.renderer.initialize() {
            return self.fail_generation(&mut request, map_renderer_error(error));
        }
        let renderer_present = request.renderer.present();
        let mut renderer_length = 0_usize;
        if renderer_present {
            if request.output.len() < renderer_length {
                return self.fail_generation(&mut request, GeneratorError::OutputCapacity);
            }
            let phase = match request.renderer.render(
                selected_token.token,
                request.sequence_id,
                request.emit_special,
                &mut request.output[renderer_length..],
            ) {
                Ok(phase) => phase,
                Err(error) => return self.fail_generation(&mut request, map_renderer_error(error)),
            };
            if phase.output_length > request.output.len() - renderer_length {
                return self.fail_generation(&mut request, GeneratorError::OutputCapacity);
            }
            renderer_length += phase.output_length;
            self.machine.context_mut().output_length = renderer_length;
            if phase.status == SequenceStatus::StopSequenceMatched {
                self.machine.context_mut().tokens_generated = 1;
            }
        }
        if let Err(error) = self.generate(EventGenerateRun {
            step: GenerateStep::Result,
            valid_request: request.valid_request,
            backend_available: request.backend_available,
            phase: decode_phase,
        }) {
            return self.fail_generation(&mut request, error);
        }
        request.generated_token_ids[request.selected_token_position] = selected_token.token;
        if renderer_present {
            if request.output.len() < renderer_length {
                return self.fail_generation(&mut request, GeneratorError::OutputCapacity);
            }
            let phase = match request
                .renderer
                .flush(request.sequence_id, &mut request.output[renderer_length..])
            {
                Ok(phase) => phase,
                Err(error) => return self.fail_generation(&mut request, map_renderer_error(error)),
            };
            if phase.output_length > request.output.len() - renderer_length {
                return self.fail_generation(&mut request, GeneratorError::OutputCapacity);
            }
            renderer_length += phase.output_length;
            self.machine.context_mut().output_length = renderer_length;
            self.machine.context_mut().output_capacity = request.output.len();
            self.machine.context_mut().tokens_generated = 1;
        } else {
            let mut generation_output = GenerationProduced::default();
            let result_producer = request
                .result_producer
                .expect("producer presence validated above");
            if let Err(error) = result_producer(GenerateResultRequest {
                output: &mut *request.output,
                generated_token_ids: &mut *request.generated_token_ids,
                result: &mut generation_output,
            }) {
                return self.fail_generation(&mut request, error);
            }
            if !generation_output.valid {
                return self.fail_generation(&mut request, GeneratorError::InvalidProducerResult);
            }
            if generation_output.output_length > request.output.len() {
                return self.fail_generation(&mut request, GeneratorError::OutputCapacity);
            }
            if generation_output.tokens_generated as usize > request.generated_token_ids.len() {
                return self.fail_generation(&mut request, GeneratorError::InconsistentTokenCount);
            }
            self.machine.context_mut().output_length = generation_output.output_length;
            self.machine.context_mut().output_capacity = request.output.len();
            self.machine.context_mut().tokens_generated = generation_output.tokens_generated;
        }
        let result = self.generation_result();
        publish_generation_done(&mut request.error_out, request.on_done, result);
        Ok(result)
    }

    fn fail_generation<R: RendererDispatch>(
        &mut self,
        request: &mut GenerateRequest<'_, R>,
        error: GeneratorError,
    ) -> Result<GenerationResult, GeneratorError> {
        let uninitialized = self.machine.is(&TextGeneratorStates::Uninitialized);
        let error_state = if uninitialized {
            GeneratorPhase::UninitializedError
        } else {
            GeneratorPhase::GenerationError
        };
        let _ = self.machine.context_mut().set_error(error, error_state);
        self.machine.set_state(if uninitialized {
            TextGeneratorStates::UninitializedError
        } else {
            TextGeneratorStates::GenerationError
        });
        let result = self.generation_result();
        publish_generation_error(&mut request.error_out, request.on_error, result);
        self.recover_generation_error(uninitialized);
        Err(error)
    }

    fn recover_generation_error(&mut self, uninitialized: bool) {
        if uninitialized {
            self.machine.context_mut().phase = GeneratorPhase::Uninitialized;
            self.machine.context_mut().sequence_live = false;
            self.machine.set_state(TextGeneratorStates::Uninitialized);
        } else {
            self.machine.context_mut().phase = GeneratorPhase::Ready;
            self.machine.context_mut().sequence_live = false;
            self.machine.context_mut().pending_final_chunk = false;
            self.machine.set_state(TextGeneratorStates::Ready);
        }
    }

    /// Publishes one caller-owned stream result synchronously.
    pub fn stream_into(
        &mut self,
        mut request: StreamRequest<'_>,
    ) -> Result<GenerationResult, GeneratorError> {
        let uninitialized = self.machine.is(&TextGeneratorStates::Uninitialized);
        if request.result_producer.is_none() {
            let error = GeneratorError::MissingProducer;
            let _ = self.machine.context_mut().set_error(
                error,
                if uninitialized {
                    GeneratorPhase::UninitializedError
                } else {
                    GeneratorPhase::GenerationError
                },
            );
            self.machine.set_state(if uninitialized {
                TextGeneratorStates::UninitializedError
            } else {
                TextGeneratorStates::GenerationError
            });
            let result = self.generation_result();
            publish_generation_error(&mut request.error_out, request.on_error, result);
            self.recover_generation_error(uninitialized);
            return Err(error);
        }
        if let Err(error) = self.stream(EventStreamRun {
            valid_request: request.valid_request,
            backend_available: request.backend_available,
            final_chunk: request.final_chunk,
        }) {
            let _ = self.machine.context_mut().set_error(
                error,
                if uninitialized {
                    GeneratorPhase::UninitializedError
                } else {
                    GeneratorPhase::GenerationError
                },
            );
            self.machine.set_state(if uninitialized {
                TextGeneratorStates::UninitializedError
            } else {
                TextGeneratorStates::GenerationError
            });
            let result = self.generation_result();
            publish_generation_error(&mut request.error_out, request.on_error, result);
            self.recover_generation_error(uninitialized);
            return Err(error);
        }
        let result_producer = request
            .result_producer
            .expect("producer presence validated above");
        let mut stream_output = StreamProduced::default();
        if let Err(error) = result_producer(StreamResultRequest {
            output: &mut *request.output,
            generated_token_ids: &mut *request.generated_token_ids,
            result: &mut stream_output,
        }) {
            return self.fail_stream(&mut request, error);
        }
        if !stream_output.valid {
            return self.fail_stream(&mut request, GeneratorError::InvalidProducerResult);
        }
        if stream_output.output_length > request.output.len() {
            return self.fail_stream(&mut request, GeneratorError::OutputCapacity);
        }
        if stream_output.tokens_generated as usize > request.generated_token_ids.len() {
            return self.fail_stream(&mut request, GeneratorError::InconsistentTokenCount);
        }
        self.machine.context_mut().output_length = stream_output.output_length;
        self.machine.context_mut().output_capacity = request.output.len();
        self.machine.context_mut().tokens_generated = stream_output.tokens_generated;
        let result = self.generation_result();
        publish_generation_done(&mut request.error_out, request.on_done, result);
        Ok(result)
    }

    fn fail_stream(
        &mut self,
        request: &mut StreamRequest<'_>,
        error: GeneratorError,
    ) -> Result<GenerationResult, GeneratorError> {
        let _ = self
            .machine
            .context_mut()
            .set_error(error, GeneratorPhase::GenerationError);
        self.machine.set_state(TextGeneratorStates::GenerationError);
        let result = self.generation_result();
        publish_generation_error(&mut request.error_out, request.on_error, result);
        self.recover_generation_error(false);
        Err(error)
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
        GenerationResult {
            output_length: self.context().output_length,
            tokens_generated: self.context().tokens_generated,
            error: self.context().error(),
        }
    }
}

fn publish_generation_done(
    error_out: &mut Option<&mut GeneratorError>,
    callback: Option<DoneCallback>,
    result: GenerationResult,
) {
    if let Some(error_out) = error_out.as_deref_mut() {
        *error_out = GeneratorError::None;
    }
    if let Some(callback) = callback {
        let _ = callback(GenerationDone {
            tokens_generated: result.tokens_generated,
            output_length: result.output_length,
        });
    }
}

fn publish_generation_error(
    error_out: &mut Option<&mut GeneratorError>,
    callback: Option<ErrorCallback>,
    result: GenerationResult,
) {
    if let Some(error_out) = error_out.as_deref_mut() {
        *error_out = result.error;
    }
    if let Some(callback) = callback {
        let _ = callback(GenerationErrorEvent {
            error: result.error,
            tokens_generated: result.tokens_generated,
            output_length: result.output_length,
        });
    }
}

/// Compatibility alias for callers using the shorter actor name.
pub type TextGenerator = TextGeneratorActor;

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};

    fn produce_ok(request: GenerateResultRequest<'_>) -> Result<(), GeneratorError> {
        request.output[..2].copy_from_slice(b"ok");
        request.generated_token_ids[0] = 42;
        request.result.valid = true;
        request.result.output_length = 2;
        request.result.tokens_generated = 1;
        Ok(())
    }
    fn conditioner_test_formatter(
        request: crate::FormatRequest<'_>,
    ) -> Result<(), crate::ConditionerError> {
        *request.output_length = 2;
        request.output[..2].copy_from_slice(b"ok");
        Ok(())
    }

    fn conditioner_test_tokenizer(
        text: &[u8],
        _add_special: bool,
        _parse_special: bool,
        output: &mut [i32],
    ) -> Result<usize, crate::ConditionerError> {
        for (slot, byte) in output.iter_mut().zip(text.iter().copied()) {
            *slot = i32::from(byte);
        }
        Ok(text.len())
    }

    fn prefill_dispatch_ok(
        _operation: super::super::prefill::sm::PrefillOperation,
        _event: &super::super::prefill::sm::EventRun,
    ) -> super::super::prefill::sm::PhaseOutcome {
        super::super::prefill::sm::PhaseOutcome::ok()
    }

    fn layer_kernel_ok(
        _operation: super::super::layer::sm::LayerOperation,
        _request: &super::super::layer::sm::LayerRequest<'_>,
    ) -> bool {
        true
    }

    fn decode_lane_ok(
        _key: super::super::decode_wavefront::sm::CompatibilityKey,
        selected: &mut super::super::decode_wavefront::sm::SelectedToken,
    ) -> bool {
        *selected = super::super::decode_wavefront::sm::SelectedToken::new(42);
        true
    }

    fn stream_produce_ok(request: StreamResultRequest<'_>) -> Result<(), GeneratorError> {
        request.output[..3].copy_from_slice(b"hey");
        request.generated_token_ids[0] = 7;
        request.result.valid = true;
        request.result.output_length = 3;
        request.result.tokens_generated = 1;
        Ok(())
    }

    fn stream_produce_error(_request: StreamResultRequest<'_>) -> Result<(), GeneratorError> {
        Err(GeneratorError::Backend)
    }

    #[test]
    fn stream_into_requires_producer_and_publishes_error() {
        let _lock = CALLBACK_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ERROR_CALLS.store(0, Ordering::SeqCst);
        LAST_OUTPUT_LENGTH.store(0, Ordering::SeqCst);
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        actor.initialize(initialize_event(&initializer)).unwrap();
        let mut output = [0_u8; 4];
        let mut token_ids = [0_i32; 1];
        let mut error = GeneratorError::None;
        let request = StreamRequest::new(&mut output, &mut token_ids)
            .with_callbacks(None, Some(on_error))
            .with_error_out(&mut error);
        assert_eq!(
            actor.stream_into(request),
            Err(GeneratorError::MissingProducer)
        );
        assert_eq!(error, GeneratorError::MissingProducer);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stream_into_publishes_synchronous_producer_scalars() {
        let _lock = CALLBACK_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        DONE_CALLS.store(0, Ordering::SeqCst);
        ERROR_CALLS.store(0, Ordering::SeqCst);
        LAST_OUTPUT_LENGTH.store(0, Ordering::SeqCst);
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        actor.initialize(initialize_event(&initializer)).unwrap();
        let mut output = [0_u8; 4];
        let mut token_ids = [0_i32; 1];
        let mut error = GeneratorError::Backend;
        let request = StreamRequest::new(&mut output, &mut token_ids)
            .with_result_producer(stream_produce_ok)
            .with_callbacks(Some(on_done), Some(on_error))
            .with_error_out(&mut error);
        let result = actor.stream_into(request).unwrap();
        assert_eq!(result.output_length, 3);
        assert_eq!(result.tokens_generated, 1);
        assert_eq!(&output[..3], b"hey");
        assert_eq!(token_ids[0], 7);
        assert_eq!(error, GeneratorError::None);
        assert_eq!(DONE_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(LAST_OUTPUT_LENGTH.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn stream_into_publishes_producer_error_callback() {
        let _lock = CALLBACK_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        DONE_CALLS.store(0, Ordering::SeqCst);
        ERROR_CALLS.store(0, Ordering::SeqCst);
        LAST_OUTPUT_LENGTH.store(0, Ordering::SeqCst);
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        actor.initialize(initialize_event(&initializer)).unwrap();
        let mut output = [0_u8; 4];
        let mut token_ids = [0_i32; 1];
        let mut error = GeneratorError::None;
        let request = StreamRequest::new(&mut output, &mut token_ids)
            .with_result_producer(stream_produce_error)
            .with_callbacks(Some(on_done), Some(on_error))
            .with_error_out(&mut error);
        assert_eq!(actor.stream_into(request), Err(GeneratorError::Backend));
        assert_eq!(error, GeneratorError::Backend);
        assert_eq!(DONE_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(LAST_OUTPUT_LENGTH.load(Ordering::SeqCst), 0);
    }
    use std::sync::Mutex;

    use super::super::decode_wavefront::sm::{
        CompatibilityKey, DispatchSummary, EventRun as DecodeEventRun, Lane,
        TextGeneratorDecodeWavefrontActor,
    };
    use super::super::initializer::sm::{
        EventRun as InitializerEventRun, InitializerOperation, PhaseOutcome,
        TextGeneratorInitializerActor,
    };
    use super::super::layer::sm::{
        EventScalarRun, LayerBuffers, LayerDtype, LayerRequest, TextGeneratorLayerScalarActor,
    };
    use super::super::prefill::sm::TextGeneratorPrefillActor;

    static DONE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static ERROR_CALLS: AtomicUsize = AtomicUsize::new(0);
    static LAST_OUTPUT_LENGTH: AtomicUsize = AtomicUsize::new(0);
    static INITIALIZER_DISPATCH_CALLS: AtomicUsize = AtomicUsize::new(0);
    static CALLBACK_TEST_LOCK: Mutex<()> = Mutex::new(());

    struct TestVocabulary;
    impl crate::detokenizer::VocabularyView for TestVocabulary {
        fn token_count(&self) -> u32 {
            43
        }
        fn token(&self, index: u32) -> Option<crate::detokenizer::TokenEntry<'_>> {
            (index == 42).then_some(crate::detokenizer::TokenEntry {
                piece: b"ok",
                token_type: crate::detokenizer::TokenType::Normal,
            })
        }
    }

    fn on_done(event: GenerationDone) -> bool {
        DONE_CALLS.fetch_add(1, Ordering::SeqCst);
        LAST_OUTPUT_LENGTH.store(event.output_length, Ordering::SeqCst);
        false
    }

    fn on_error(event: GenerationErrorEvent) -> bool {
        ERROR_CALLS.fetch_add(1, Ordering::SeqCst);
        LAST_OUTPUT_LENGTH.store(event.output_length, Ordering::SeqCst);
        false
    }

    fn initializer_dispatch(
        _operation: InitializerOperation,
        _event: &InitializerEventRun,
    ) -> PhaseOutcome {
        INITIALIZER_DISPATCH_CALLS.fetch_add(1, Ordering::SeqCst);
        PhaseOutcome::ok()
    }

    fn initialize_event(
        initializer: &RefCell<TextGeneratorInitializerActor>,
    ) -> EventInitializeRun<'_> {
        let mut initializer_event = InitializerEventRun::valid();
        initializer_event.dispatch = Some(initializer_dispatch);
        EventInitializeRun::new(true, true, initializer, initializer_event)
    }
    #[test]
    fn generate_into_publishes_done_callback_and_clears_error_out() {
        let _lock = CALLBACK_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        DONE_CALLS.store(0, Ordering::SeqCst);
        ERROR_CALLS.store(0, Ordering::SeqCst);
        LAST_OUTPUT_LENGTH.store(0, Ordering::SeqCst);
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        actor
            .initialize(initialize_event(&initializer))
            .expect("initialization succeeds");
        let mut output = [0_u8; 2];
        let mut token_ids = [0_i32; 1];
        let prefill = RefCell::new(TextGeneratorPrefillActor::new());
        let prefill_event = super::super::prefill::sm::EventRun::new(1);
        prefill
            .borrow_mut()
            .set_dispatcher(Some(prefill_dispatch_ok));
        let mut hidden_storage = [0.0_f32; 1];
        let mut norm_storage = [0.0_f32; 1];
        let mut scratch_storage = [0.0_f32; 1];
        let hidden = RefCell::new(&mut hidden_storage[..]);
        let norm = RefCell::new(&mut norm_storage[..]);
        let scratch = RefCell::new(&mut scratch_storage[..]);
        let layer = RefCell::new(TextGeneratorLayerScalarActor::new());
        let layer_event = EventScalarRun::new(
            LayerRequest::new(LayerDtype::F32, 1, 1, 1, 1, 1, 0, 0)
                .with_buffers(LayerBuffers {
                    hidden: &hidden,
                    norm: &norm,
                    scratch: &scratch,
                })
                .with_callback(layer_kernel_ok),
        );
        let mut lanes_storage = [Lane::new(
            1,
            CompatibilityKey::default(),
            Some(decode_lane_ok),
        )];
        let lanes = RefCell::new(&mut lanes_storage[..]);
        let decode_out = RefCell::new(DispatchSummary::default());
        let decode = RefCell::new(TextGeneratorDecodeWavefrontActor::new());
        let selected_token =
            RefCell::new(super::super::decode_wavefront::sm::SelectedToken::default());
        let decode_event = DecodeEventRun::new(&lanes, &decode_out, &selected_token, 0);
        let renderer = RefCell::new(crate::renderer::sm::TextRenderer::new(&TestVocabulary));
        let renderer_init_context = RefCell::new(crate::renderer::sm::InitializeContext::default());
        let renderer_collaborator = RendererCollaborator {
            actor: &renderer,
            initialize_request: crate::renderer::sm::InitializeRequest {
                strip_leading_space: false,
                stop_sequences: &[],
                dispatch_done: None,
                dispatch_error: None,
            },
            initialize_context: &renderer_init_context,
        };
        let mut error = GeneratorError::Backend;
        let request = GenerateRequest::new(&mut output, &mut token_ids)
            .with_renderer(renderer_collaborator)
            .with_prefill(&prefill, prefill_event)
            .with_layer(LayerCollaborator::Scalar(&layer, layer_event))
            .with_decode(DecodeCollaborator {
                actor: &decode,
                event: decode_event,
            })
            .with_callbacks(Some(on_done), Some(on_error))
            .with_error_out(&mut error);
        let result = actor.generate_into(request).expect("generation succeeds");
        assert_eq!(result.output_length, 2);
        assert_eq!(result.tokens_generated, 1);
        assert_eq!(&output[..2], b"ok");
        assert_eq!(token_ids[0], 42);
    }

    #[test]
    fn generate_into_publishes_error_callback_and_error_out() {
        let _lock = CALLBACK_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        DONE_CALLS.store(0, Ordering::SeqCst);
        ERROR_CALLS.store(0, Ordering::SeqCst);
        LAST_OUTPUT_LENGTH.store(0, Ordering::SeqCst);
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        actor
            .initialize(initialize_event(&initializer))
            .expect("initialization succeeds");
        let mut output = [0_u8; 2];
        let mut token_ids = [0_i32; 1];
        let mut error = GeneratorError::None;
        let mut request = GenerateRequest::new(&mut output, &mut token_ids)
            .with_result_producer(produce_ok)
            .with_callbacks(Some(on_done), Some(on_error))
            .with_error_out(&mut error);
        request.valid_request = false;
        assert_eq!(
            actor.generate_into(request),
            Err(GeneratorError::InvalidRequest)
        );
        assert_eq!(error, GeneratorError::InvalidRequest);
        assert_eq!(DONE_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(LAST_OUTPUT_LENGTH.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn invalid_generate_into_rejects_before_conditioner_side_effects() {
        let _lock = CALLBACK_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ERROR_CALLS.store(0, Ordering::SeqCst);
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        actor.initialize(initialize_event(&initializer)).unwrap();

        let mut conditioner = RefCell::new(Conditioner::default());
        conditioner
            .get_mut()
            .set_dependencies(conditioner_test_formatter, conditioner_test_tokenizer);
        conditioner
            .borrow_mut()
            .process_event(conditioner_event::Bind {
                tokenizer_available: true,
                formatter_available: true,
                model_valid: true,
            })
            .unwrap();
        let messages = [ChatMessage {
            role: b"user",
            content: b"prompt",
        }];
        let mut output = [0xA5_u8; 2];
        let mut token_ids = [7_i32, 8_i32];
        let mut token_count = 99_usize;
        let mut error = GeneratorError::None;
        let mut request = GenerateRequest::new(&mut output, &mut token_ids)
            .with_result_producer(produce_ok)
            .with_conditioner(&conditioner, &messages, &mut token_count)
            .with_callbacks(None, Some(on_error))
            .with_error_out(&mut error);
        request.valid_request = false;

        assert_eq!(
            actor.generate_into(request),
            Err(GeneratorError::InvalidRequest)
        );
        assert_eq!(token_ids, [7, 8]);
        assert_eq!(token_count, 99);
        assert_eq!(output, [0xA5; 2]);
        assert_eq!(error, GeneratorError::InvalidRequest);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);
    }

    #[test]
    fn unavailable_backend_rejects_before_conditioner_side_effects() {
        let _lock = CALLBACK_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ERROR_CALLS.store(0, Ordering::SeqCst);
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        actor.initialize(initialize_event(&initializer)).unwrap();

        let mut conditioner = RefCell::new(Conditioner::default());
        conditioner
            .get_mut()
            .set_dependencies(conditioner_test_formatter, conditioner_test_tokenizer);
        conditioner
            .borrow_mut()
            .process_event(conditioner_event::Bind {
                tokenizer_available: true,
                formatter_available: true,
                model_valid: true,
            })
            .unwrap();
        let messages = [ChatMessage {
            role: b"user",
            content: b"prompt",
        }];
        let mut output = [0xA5_u8; 2];
        let mut token_ids = [7_i32, 8_i32];
        let mut token_count = 99_usize;
        let mut error = GeneratorError::None;
        let mut request = GenerateRequest::new(&mut output, &mut token_ids)
            .with_result_producer(produce_ok)
            .with_conditioner(&conditioner, &messages, &mut token_count)
            .with_callbacks(None, Some(on_error))
            .with_error_out(&mut error);
        request.backend_available = false;

        assert_eq!(actor.generate_into(request), Err(GeneratorError::Backend));
        assert_eq!(token_ids, [7, 8]);
        assert_eq!(token_count, 99);
        assert_eq!(output, [0xA5; 2]);
        assert_eq!(error, GeneratorError::Backend);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);
    }

    #[test]
    fn generate_into_requires_result_producer_and_publishes_error() {
        let _lock = CALLBACK_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ERROR_CALLS.store(0, Ordering::SeqCst);
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        actor.initialize(initialize_event(&initializer)).unwrap();
        let mut output = [0_u8; 2];
        let mut token_ids = [0_i32; 1];
        let mut error = GeneratorError::None;
        let request = GenerateRequest::new(&mut output, &mut token_ids)
            .with_callbacks(None, Some(on_error))
            .with_error_out(&mut error);
        assert_eq!(
            actor.generate_into(request),
            Err(GeneratorError::MissingProducer)
        );
        assert_eq!(error, GeneratorError::MissingProducer);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn generate_into_missing_collaborator_reports_failure_and_recovers_ready() {
        let _lock = CALLBACK_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        DONE_CALLS.store(0, Ordering::SeqCst);
        ERROR_CALLS.store(0, Ordering::SeqCst);
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        actor.initialize(initialize_event(&initializer)).unwrap();

        let mut output = [0xA5_u8; 2];
        let mut token_ids = [7_i32, 8_i32];
        let mut error = GeneratorError::None;
        let request = GenerateRequest::new(&mut output, &mut token_ids)
            .with_result_producer(produce_ok)
            .with_callbacks(Some(on_done), Some(on_error))
            .with_error_out(&mut error);

        assert_eq!(
            actor.generate_into(request),
            Err(GeneratorError::MissingCollaborator)
        );
        assert_eq!(output, [0xA5; 2]);
        assert_eq!(token_ids, [7, 8]);
        assert_eq!(error, GeneratorError::MissingCollaborator);
        assert_eq!(DONE_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);
        assert_eq!(actor.context().result_count(), 0);

        let retry_request = GenerateRequest::new(&mut output, &mut token_ids)
            .with_result_producer(produce_ok)
            .with_callbacks(Some(on_done), Some(on_error));
        assert_eq!(
            actor.generate_into(retry_request),
            Err(GeneratorError::MissingCollaborator)
        );
        assert_eq!(DONE_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 2);
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);
        assert_eq!(actor.context().result_count(), 0);
    }

    #[test]
    fn stream_into_missing_producer_recovers_to_uninitialized_before_initialize() {
        let _lock = CALLBACK_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ERROR_CALLS.store(0, Ordering::SeqCst);
        let mut actor = TextGeneratorActor::new();
        let mut output = [0_u8; 2];
        let mut token_ids = [0_i32; 1];
        let mut error = GeneratorError::None;
        let request = StreamRequest::new(&mut output, &mut token_ids)
            .with_callbacks(None, Some(on_error))
            .with_error_out(&mut error);
        assert_eq!(
            actor.stream_into(request),
            Err(GeneratorError::MissingProducer)
        );
        assert_eq!(error, GeneratorError::MissingProducer);
        assert_eq!(actor.context().phase(), GeneratorPhase::Uninitialized);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn initialize_dispatches_initializer_actor_and_reaches_ready() {
        INITIALIZER_DISPATCH_CALLS.store(0, Ordering::SeqCst);
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        actor
            .initialize(initialize_event(&initializer))
            .expect("initializer bridge succeeds");
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);
        assert!(INITIALIZER_DISPATCH_CALLS.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn initialize_maps_initializer_invalid_request_without_ready() {
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut initializer_event = InitializerEventRun::valid();
        initializer_event.valid_request = false;
        let mut actor = TextGeneratorActor::new();
        assert_eq!(
            actor.initialize(EventInitializeRun::new(
                true,
                true,
                &initializer,
                initializer_event,
            )),
            Err(GeneratorError::InvalidRequest)
        );
        assert_ne!(actor.context().phase(), GeneratorPhase::Ready);
    }

    #[test]
    fn generate_into_dispatches_caller_owned_prefill_synchronously_and_surfaces_rejection() {
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        actor
            .initialize(initialize_event(&initializer))
            .expect("initialization succeeds");

        let prefill = RefCell::new(TextGeneratorPrefillActor::new());
        let prefill_event = super::super::prefill::sm::EventRun::new(1);
        let mut hidden_storage = [0.0_f32; 1];
        let mut norm_storage = [0.0_f32; 1];
        let mut scratch_storage = [0.0_f32; 1];
        let hidden = RefCell::new(&mut hidden_storage[..]);
        let norm = RefCell::new(&mut norm_storage[..]);
        let scratch = RefCell::new(&mut scratch_storage[..]);
        let layer = RefCell::new(TextGeneratorLayerScalarActor::new());
        let layer_event = EventScalarRun::new(
            LayerRequest::new(LayerDtype::F32, 1, 1, 1, 1, 1, 0, 0)
                .with_buffers(LayerBuffers {
                    hidden: &hidden,
                    norm: &norm,
                    scratch: &scratch,
                })
                .with_callback(layer_kernel_ok),
        );
        let mut lanes_storage = [Lane::new(
            1,
            CompatibilityKey::default(),
            Some(decode_lane_ok),
        )];
        let lanes = RefCell::new(&mut lanes_storage[..]);
        let decode_out = RefCell::new(DispatchSummary::default());
        let decode = RefCell::new(TextGeneratorDecodeWavefrontActor::new());
        let selected_token =
            RefCell::new(super::super::decode_wavefront::sm::SelectedToken::default());
        let decode_event = DecodeEventRun::new(&lanes, &decode_out, &selected_token, 0);
        let mut output = [0_u8; 2];
        let mut token_ids = [0_i32; 1];
        let request = GenerateRequest::new(&mut output, &mut token_ids)
            .with_result_producer(produce_ok)
            .with_prefill(&prefill, prefill_event)
            .with_layer(LayerCollaborator::Scalar(&layer, layer_event))
            .with_decode(DecodeCollaborator {
                actor: &decode,
                event: decode_event,
            });

        assert_eq!(actor.generate_into(request), Err(GeneratorError::Backend));
        assert_eq!(
            prefill.borrow().context().last_operation(),
            super::super::prefill::sm::PrefillOperation::Slots
        );
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);
        assert_eq!(actor.context().result_count(), 0);
        assert_eq!(actor.context().prefill_count(), 0);
    }

    #[test]
    fn initialize_maps_initializer_backend_rejection_without_ready() {
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut initializer_event = InitializerEventRun::valid();
        initializer_event.conditioner_bind_accepted = false;
        initializer_event.conditioner_bind_code = 64;
        let mut actor = TextGeneratorActor::new();
        assert_eq!(
            actor.initialize(EventInitializeRun::new(
                true,
                true,
                &initializer,
                initializer_event,
            )),
            Err(GeneratorError::Backend)
        );
        assert_ne!(actor.context().phase(), GeneratorPhase::Ready);
    }

    #[test]
    fn initialize_retries_from_initialization_error_explicitly() {
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        let mut invalid = InitializerEventRun::valid();
        invalid.valid_request = false;
        assert_eq!(
            actor.initialize(EventInitializeRun::new(true, true, &initializer, invalid)),
            Err(GeneratorError::InvalidRequest)
        );
        actor
            .initialize(initialize_event(&initializer))
            .expect("explicit retry succeeds");
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);
    }

    #[test]
    fn lifecycle_requests_reject_uninitialized_and_return_to_uninitialized() {
        let mut actor = TextGeneratorActor::new();
        assert_eq!(actor.context().phase(), GeneratorPhase::Uninitialized);
        assert_eq!(
            actor.generate(EventGenerateRun::default()),
            Err(GeneratorError::InvalidRequest)
        );
        assert_eq!(actor.context().phase(), GeneratorPhase::Uninitialized);
        assert_eq!(actor.context().error(), GeneratorError::InvalidRequest);
        assert_eq!(
            actor.stream(EventStreamRun::default()),
            Err(GeneratorError::InvalidRequest)
        );
        assert_eq!(actor.context().phase(), GeneratorPhase::Uninitialized);
        assert_eq!(
            actor.flush(EventFlushRun::default()),
            Err(GeneratorError::InvalidRequest)
        );
        assert_eq!(actor.context().phase(), GeneratorPhase::Uninitialized);
        assert_eq!(
            actor.reset(EventResetRun::default()),
            Err(GeneratorError::InvalidRequest)
        );
        assert_eq!(actor.context().phase(), GeneratorPhase::Uninitialized);
    }

    #[test]
    fn invalid_generate_publishes_error_then_valid_generate_recovers() {
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        actor.initialize(initialize_event(&initializer)).unwrap();
        assert_eq!(
            actor.generate(EventGenerateRun {
                step: GenerateStep::None,
                valid_request: true,
                backend_available: true,
                phase: SubphaseOutcome::ok(),
            }),
            Err(GeneratorError::InvalidRequest)
        );
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);
        assert_eq!(actor.context().error(), GeneratorError::InvalidRequest);
        actor
            .generate(EventGenerateRun {
                step: GenerateStep::Prefill,
                valid_request: true,
                backend_available: true,
                phase: SubphaseOutcome::ok(),
            })
            .unwrap();
        actor
            .generate(EventGenerateRun {
                step: GenerateStep::Decode,
                valid_request: true,
                backend_available: true,
                phase: SubphaseOutcome::ok(),
            })
            .unwrap();
        actor
            .generate(EventGenerateRun {
                step: GenerateStep::Result,
                valid_request: true,
                backend_available: true,
                phase: SubphaseOutcome::ok(),
            })
            .unwrap();
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);
        assert_eq!(actor.context().error(), GeneratorError::None);
        assert_eq!(actor.context().result_count(), 1);
    }

    #[test]
    fn diagnostics_capture_populates_caller_owned_snapshot_and_acceptance() {
        let mut actor = TextGeneratorActor::new();
        let diagnostics = RefCell::new(GeneratorDiagnostics::default());
        let accepted = RefCell::new(false);

        actor
            .capture_diagnostics(EventCaptureDiagnostics {
                out: &diagnostics,
                accepted: &accepted,
            })
            .expect("diagnostics capture is accepted while uninitialized");

        assert!(*accepted.borrow());
        let snapshot = *diagnostics.borrow();
        assert_eq!(snapshot.phase, GeneratorPhase::Uninitialized);
        assert_eq!(snapshot.error, GeneratorError::None);
        assert_eq!(snapshot.diagnostics_count, 1);
        assert_eq!(snapshot.graph_lifecycle_count, 0);
        assert!(!snapshot.sequence_live);
        assert_eq!(snapshot.prefill_count, 0);
        assert_eq!(snapshot.decode_count, 0);
        assert_eq!(snapshot.result_count, 0);
    }

    #[test]
    fn graph_lifecycle_capture_populates_snapshot_and_rejects_unsupported_state() {
        let mut actor = TextGeneratorActor::new();
        let lifecycle = RefCell::new(GraphLifecycleDiagnostics::default());
        let accepted = RefCell::new(false);

        actor
            .capture_graph_lifecycle(EventCaptureGraphLifecycle {
                out: &lifecycle,
                accepted: &accepted,
            })
            .expect("lifecycle capture is accepted while uninitialized");

        assert!(*accepted.borrow());
        let snapshot = *lifecycle.borrow();
        assert_eq!(snapshot.phase, GeneratorPhase::Uninitialized);
        assert_eq!(snapshot.error, GeneratorError::None);
        assert_eq!(snapshot.graph_lifecycle_count, 1);
        assert_eq!(snapshot.diagnostics_count, 0);

        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        actor.initialize(initialize_event(&initializer)).unwrap();
        actor
            .generate(EventGenerateRun {
                step: GenerateStep::Prefill,
                valid_request: true,
                backend_available: true,
                phase: SubphaseOutcome::ok(),
            })
            .unwrap();
        let rejected_lifecycle = RefCell::new(GraphLifecycleDiagnostics::default());
        let rejected = RefCell::new(false);
        assert_eq!(
            actor.capture_graph_lifecycle(EventCaptureGraphLifecycle {
                out: &rejected_lifecycle,
                accepted: &rejected,
            }),
            Err(GeneratorError::UnexpectedEvent)
        );
        assert!(!*rejected.borrow());
        assert_eq!(rejected_lifecycle.borrow().graph_lifecycle_count, 0);
    }

    #[test]
    fn stream_flush_and_reset_rejections_recover_to_ready() {
        let initializer = RefCell::new(TextGeneratorInitializerActor::new());
        let mut actor = TextGeneratorActor::new();
        actor.initialize(initialize_event(&initializer)).unwrap();

        assert_eq!(
            actor.stream(EventStreamRun {
                valid_request: false,
                backend_available: true,
                final_chunk: true,
            }),
            Err(GeneratorError::InvalidRequest)
        );
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);
        assert_eq!(actor.context().error(), GeneratorError::InvalidRequest);
        actor
            .stream(EventStreamRun {
                valid_request: true,
                backend_available: true,
                final_chunk: true,
            })
            .unwrap();
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);

        assert_eq!(
            actor.flush(EventFlushRun {
                valid_request: false,
                backend_available: true,
            }),
            Err(GeneratorError::InvalidRequest)
        );
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);
        actor
            .flush(EventFlushRun {
                valid_request: true,
                backend_available: true,
            })
            .unwrap();
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);

        assert_eq!(
            actor.reset(EventResetRun {
                valid_request: false,
                backend_available: true,
            }),
            Err(GeneratorError::InvalidRequest)
        );
        assert_eq!(actor.context().phase(), GeneratorPhase::Ready);
        actor
            .reset(EventResetRun {
                valid_request: true,
                backend_available: true,
            })
            .unwrap();
        assert_eq!(actor.context().error(), GeneratorError::None);
    }
}
