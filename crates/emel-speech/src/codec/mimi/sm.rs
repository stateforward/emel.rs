//! Source-aligned bounded Mimi speech codec facade.
//!
//! The facade owns lifecycle and sequencing only. Binding metadata and arena
//! capacities are validated before the generated machine enters its bind
//! states. Frame data, streaming state, and numeric stage contracts remain
//! caller-owned and are dispatched synchronously through the child actors.

#![allow(
    clippy::enum_variant_names,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::module_name_repetitions,
    clippy::missing_const_for_fn,
    clippy::struct_excessive_bools,
    clippy::derive_partial_eq_without_eq,
    missing_debug_implementations,
    dead_code,
    missing_docs,
    private_interfaces,
    reason = "SML-generated state and event types mirror the composed machine contract"
)]

use super::{binding, decoder as decoder_mod, encoder as encoder_mod, quantizer as quantizer_mod};
use binding::PreparedMimiBinding;
use core::cell::{Cell, RefCell};
use core::fmt;
use decoder_mod::sm as decoder;
use encoder_mod::sm as encoder;
use quantizer_mod::sm as quantizer;
use sml::sml;
/// Caller-owned quantizer runtime used by the public Mimi encode/decode façade.
pub type MimiQuantizerRuntime<'a> = quantizer::QuantizerRuntime<'a>;
/// Caller-owned decoder streaming state used by the public Mimi decode façade.
pub type MimiDecoderStreamingState<'a> = decoder::CodecStreamingState<'a>;
/// Synchronous decoder-transformer stage contract used by [`DecodeRun`].
pub type MimiDecoderTransformerStage = decoder::DecoderTransformerStage;
/// Synchronous decoder backend stage contract used by [`DecodeRun`].
pub type MimiDecoderBackendStage = decoder::DecoderBackendStage;

const MAX_LATENT_FLOATS: usize = 512;

/// Top-level Mimi errors corresponding to `mimi::error`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum MimiError {
    /// No error was selected.
    #[default]
    None = 0,
    /// A frame operation was requested before initialization.
    NotInitialized = 1,
    /// The prepared model binding does not satisfy the Mimi contract.
    BindFailed = 2,
    /// A caller-owned arena is smaller than the binding contract.
    ArenaCapacity = 3,
    /// A frame request has invalid shape or capacity.
    RequestShape = 4,
    /// A decode code is outside the bound codebook range.
    CodeRange = 5,
    /// The event was not modeled in the current state.
    UnexpectedEvent = 6,
}

impl fmt::Display for MimiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "no error",
            Self::NotInitialized => "Mimi codec is not initialized",
            Self::BindFailed => "Mimi codec binding failed",
            Self::ArenaCapacity => "Mimi codec arena capacity is invalid",
            Self::RequestShape => "Mimi codec request shape is invalid",
            Self::CodeRange => "Mimi codec code is out of range",
            Self::UnexpectedEvent => "unexpected Mimi codec event",
        })
    }
}
impl std::error::Error for MimiError {}
impl MimiError {
    fn from_decoder_error(error: decoder::DecoderError) -> Self {
        match error {
            decoder::DecoderError::None => Self::None,
            decoder::DecoderError::RuntimeUnbound => Self::NotInitialized,
            decoder::DecoderError::RequestShape => Self::RequestShape,
            decoder::DecoderError::BufferCapacity => Self::ArenaCapacity,
            decoder::DecoderError::StageUnavailable | decoder::DecoderError::StageFailed => {
                Self::UnexpectedEvent
            }
        }
    }
}
/// Read-only facade diagnostics available without invoking numeric backends.
///
/// The pinned C++ facade exposes projection-kernel counters through its
/// diagnostics event. Rust stage callbacks intentionally remain opaque and
/// caller-owned, so this boundary publishes the exact scalar binding and
/// lifecycle facts available here rather than manufacturing kernel counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MimiDiagnostics {
    /// Whether a valid binding has completed initialization.
    pub initialized: bool,
    /// Samples in one bound PCM frame.
    pub frame_samples: u32,
    /// Total bound quantizer levels.
    pub n_q: u32,
    /// Latent and transformer channel width.
    pub dim: usize,
    /// Codebook cardinality.
    pub card: usize,
    /// Number of semantic quantizer levels.
    pub semantic_n_q: usize,
    /// Quantizer projection width.
    pub codebook_dim: usize,
    /// Prepared-weight arena capacity.
    pub prepared_floats: usize,
    /// Persistent streaming-state arena capacity.
    pub state_floats: usize,
    /// Per-dispatch workspace arena capacity.
    pub workspace_floats: usize,
    /// One-frame staging arena capacity.
    pub frame_floats: usize,
}

/// Read-only diagnostics capture with caller-owned output.
#[derive(Clone, Copy, Debug)]
pub struct EventCaptureDiagnostics<'event> {
    /// Destination populated synchronously by the facade.
    pub out: &'event Cell<MimiDiagnostics>,
}

impl<'event> EventCaptureDiagnostics<'event> {
    /// Creates a diagnostics request over caller-owned output storage.
    #[must_use]
    pub const fn new(out: &'event Cell<MimiDiagnostics>) -> Self {
        Self { out }
    }
}

/// Bounded initialize completion payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitializeDone {
    pub frame_samples: u32,
    pub n_q: u32,
}
/// Bounded initialize failure payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitializeError {
    pub error: MimiError,
}
/// Bounded encode completion payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodeDone {
    pub n_q: u32,
}
/// Bounded encode failure payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodeError {
    pub error: MimiError,
}
/// Bounded decode completion payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeDone {
    pub frame_samples: u32,
}
/// Bounded decode failure payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeError {
    pub error: MimiError,
}

/// Synchronous initialize completion callback.
pub type InitializeDoneCallback = fn(InitializeDone) -> bool;
/// Synchronous initialize failure callback.
pub type InitializeErrorCallback = fn(InitializeError) -> bool;
/// Synchronous encode completion callback.
pub type EncodeDoneCallback = fn(EncodeDone) -> bool;
/// Synchronous encode failure callback.
pub type EncodeErrorCallback = fn(EncodeError) -> bool;
/// Synchronous decode completion callback.
pub type DecodeDoneCallback = fn(DecodeDone) -> bool;
/// Synchronous decode failure callback.
pub type DecodeErrorCallback = fn(DecodeError) -> bool;

/// One-time binding request. All four arenas remain caller-owned.
#[derive(Debug)]
pub struct InitRun<'a> {
    /// Prepared immutable binding produced by the public binding factory.
    pub binding: PreparedMimiBinding<'a>,
    /// Prepared-weight arena supplied by the caller.
    pub prepared: &'a mut [f32],
    /// Persistent streaming-state arena supplied by the caller.
    pub state_arena: &'a mut [f32],
    /// Per-dispatch workspace arena supplied by the caller.
    pub workspace: &'a mut [f32],
    /// One-frame staging arena supplied by the caller.
    pub frame: &'a mut [f32],
    /// Optional synchronous error destination.
    pub error_out: Option<&'a mut MimiError>,
    /// Optional synchronous success callback.
    pub on_done: Option<InitializeDoneCallback>,
    /// Optional synchronous failure callback.
    pub on_error: Option<InitializeErrorCallback>,
}
impl<'a> InitRun<'a> {
    /// Creates a binding request over caller-owned arenas.
    #[must_use]
    pub const fn new(
        binding: PreparedMimiBinding<'a>,
        prepared: &'a mut [f32],
        state_arena: &'a mut [f32],
        workspace: &'a mut [f32],
        frame: &'a mut [f32],
    ) -> Self {
        Self {
            binding,
            prepared,
            state_arena,
            workspace,
            frame,
            error_out: None,
            on_done: None,
            on_error: None,
        }
    }
    /// Installs an optional error destination.
    #[must_use]
    pub const fn with_error_out(mut self, error_out: &'a mut MimiError) -> Self {
        self.error_out = Some(error_out);
        self
    }
    /// Installs optional synchronous callbacks.
    #[must_use]
    pub const fn with_callbacks(
        mut self,
        on_done: Option<InitializeDoneCallback>,
        on_error: Option<InitializeErrorCallback>,
    ) -> Self {
        self.on_done = on_done;
        self.on_error = on_error;
        self
    }
}

/// One caller-owned PCM-to-code frame request.
///
/// A request is valid only for one synchronous, run-to-completion dispatch.
/// The [`SpeechCodecMimi`] owner must remain the sole mutable dispatcher for
/// that duration: do not share a request's mutable storage with another
/// thread, actor, or callback, and do not re-enter the owner from a callback.
/// The `RefCell` fields provide temporary, checked-at-runtime mutable borrows
/// while each child action runs; those borrows are released before the next
/// phase or callback, and are never a deferred-work mechanism. Buffers remain
/// caller-owned for the entire dispatch and must outlive the request.
#[derive(Debug)]
pub struct EncodeRun<'a> {
    /// Bound encoder runtime metadata. This is shared read-only for the
    /// synchronous dispatch and must not be mutated or replaced by callbacks.
    pub encoder_runtime: &'a encoder::EncoderRuntime,
    /// Bound quantizer runtime metadata and caller-owned quantizer state. It
    /// is read-only during this dispatch and must not be shared for mutation.
    pub quantizer_runtime: &'a quantizer::QuantizerRuntime<'a>,
    /// Caller-owned persistent streaming state. The facade takes a temporary
    /// mutable borrow while the encoder child runs and releases it before
    /// advancing or invoking a callback; callers must not alias or re-enter it.
    pub streaming: RefCell<&'a mut encoder::EncoderStreamingState<'a>>,
    /// Caller-owned PCM input. It is borrowed read-only for this synchronous
    /// request and must remain live and unmodified until dispatch returns.
    pub pcm: &'a [f32],
    /// Caller-owned frame staging buffer. The facade temporarily mutably
    /// borrows and releases it around the encoder child action; it does not
    /// take ownership, retain it, or replace the caller's allocation.
    pub frame: RefCell<&'a mut [f32]>,
    /// Caller-owned per-dispatch workspace. The facade temporarily mutably
    /// borrows this slot for synchronous child work and releases it before any
    /// callback; it must not be shared or accessed through re-entry.
    pub workspace: RefCell<&'a mut [f32]>,
    /// Caller-owned destination for quantized codes. The facade temporarily
    /// mutably borrows it during the quantizer phase, then releases the borrow
    /// before callbacks; callers retain ownership throughout the dispatch.
    pub codes_out: RefCell<&'a mut [i32]>,
    /// Optional caller-owned error destination. Its mutable reference is
    /// temporarily borrowed and written synchronously on an error route, then
    /// released; it is never retained, shared, or replaced by the facade.
    pub error_out: RefCell<Option<&'a mut MimiError>>,
    /// Optional synchronous success callback. It runs before dispatch returns,
    /// after all temporary `RefCell` borrows are released; it must not retain
    /// borrowed request data, share mutable buffers, or re-enter the owner.
    pub on_done: Option<EncodeDoneCallback>,
    /// Optional synchronous failure callback. It runs before dispatch returns,
    /// after all temporary `RefCell` borrows are released; it must not retain
    /// borrowed request data, share mutable buffers, or re-enter the owner.
    pub on_error: Option<EncodeErrorCallback>,
}
impl<'a> EncodeRun<'a> {
    #[must_use]
    pub const fn new(
        encoder_runtime: &'a encoder::EncoderRuntime,
        quantizer_runtime: &'a quantizer::QuantizerRuntime<'a>,
        streaming: &'a mut encoder::EncoderStreamingState<'a>,
        pcm: &'a [f32],
        frame: &'a mut [f32],
        workspace: &'a mut [f32],
        codes_out: &'a mut [i32],
    ) -> Self {
        Self {
            encoder_runtime,
            quantizer_runtime,
            streaming: RefCell::new(streaming),
            pcm,
            frame: RefCell::new(frame),
            workspace: RefCell::new(workspace),
            codes_out: RefCell::new(codes_out),
            error_out: RefCell::new(None),
            on_done: None,
            on_error: None,
        }
    }
    /// Installs an optional caller-owned error destination for this dispatch.
    #[must_use]
    pub const fn with_error_out(mut self, error_out: &'a mut MimiError) -> Self {
        self.error_out = RefCell::new(Some(error_out));
        self
    }
    /// Installs callbacks that run synchronously after temporary borrows end.
    /// Neither callback may retain request data, share mutable buffers, or
    /// re-enter the Mimi owner.
    #[must_use]
    pub const fn with_callbacks(
        mut self,
        on_done: Option<EncodeDoneCallback>,
        on_error: Option<EncodeErrorCallback>,
    ) -> Self {
        self.on_done = on_done;
        self.on_error = on_error;
        self
    }
}

/// One caller-owned code-to-PCM frame request.
///
/// A request is valid only for one synchronous, run-to-completion dispatch.
/// The [`SpeechCodecMimi`] owner remains the sole mutable writer for the
/// request and its buffers. Do not share these mutable fields, invoke another
/// dispatch from a callback, or otherwise re-enter the owner. Each `RefCell`
/// is borrowed mutably only for the child action that needs it; the borrow is
/// released before the next phase and before callbacks. This temporary
/// take/replace-style access does not transfer ownership or schedule work.
/// Every buffer and streaming state remains caller-owned and must outlive the
/// request and its synchronous dispatch.
#[derive(Debug)]
pub struct DecodeRun<'a> {
    /// Bound decoder runtime metadata, borrowed read-only for this dispatch.
    pub runtime: binding::CodecRuntime<'a>,
    /// Caller-bound quantizer runtime, borrowed read-only and not replaceable
    /// or mutable through callbacks during this dispatch.
    pub quantizer_runtime: &'a MimiQuantizerRuntime<'a>,
    /// Caller-owned persistent decoder streaming state. The facade temporarily
    /// mutably borrows it for backend work, then releases it before advancing
    /// or invoking a callback; callers must not alias or re-enter it.
    pub streaming: RefCell<&'a mut MimiDecoderStreamingState<'a>>,
    /// Caller-owned code input, borrowed read-only until synchronous return.
    pub codes: &'a [i32],
    /// Caller-owned decoder frame staging buffer. It is temporarily mutably
    /// borrowed by child work and released between phases; no ownership is
    /// retained or allocation replaced.
    pub frame: RefCell<&'a mut [f32]>,
    /// Caller-owned per-dispatch workspace. It is temporarily mutably borrowed
    /// by child work and must not be shared or accessed through re-entry.
    pub workspace: RefCell<&'a mut [f32]>,
    /// Caller-owned PCM output destination. It is temporarily mutably borrowed
    /// during backend work, then released; callers retain ownership throughout.
    pub pcm_out: RefCell<&'a mut [f32]>,
    /// Caller-supplied synchronous decoder-transformer stage. It must finish
    /// before returning, retain no request references, and never re-enter Mimi.
    pub transformer: Option<MimiDecoderTransformerStage>,
    /// Caller-supplied synchronous decoder backend stage. It must finish before
    /// returning, retain no request references, and never re-enter Mimi.
    pub backend: Option<MimiDecoderBackendStage>,
    /// Optional caller-owned error destination. It is temporarily mutably
    /// borrowed and written on an error route, then released before callbacks;
    /// the facade never retains, shares, or replaces it.
    pub error_out: RefCell<Option<&'a mut MimiError>>,
    /// Optional synchronous success callback. It runs before dispatch returns,
    /// after temporary borrows are released; it must not retain request data,
    /// share mutable buffers, or re-enter the owner.
    pub on_done: Option<DecodeDoneCallback>,
    /// Optional synchronous failure callback. It runs before dispatch returns,
    /// after temporary borrows are released; it must not retain request data,
    /// share mutable buffers, or re-enter the owner.
    pub on_error: Option<DecodeErrorCallback>,
}
impl<'a> DecodeRun<'a> {
    #[must_use]
    pub const fn new(
        runtime: binding::CodecRuntime<'a>,
        quantizer_runtime: &'a MimiQuantizerRuntime<'a>,
        streaming: &'a mut MimiDecoderStreamingState<'a>,
        codes: &'a [i32],
        frame: &'a mut [f32],
        workspace: &'a mut [f32],
        pcm_out: &'a mut [f32],
    ) -> Self {
        Self {
            runtime,
            quantizer_runtime,
            streaming: RefCell::new(streaming),
            codes,
            frame: RefCell::new(frame),
            workspace: RefCell::new(workspace),
            pcm_out: RefCell::new(pcm_out),
            transformer: None,
            backend: None,
            error_out: RefCell::new(None),
            on_done: None,
            on_error: None,
        }
    }
    /// Installs the caller-owned synchronous transformer and backend stages.
    #[must_use]
    pub const fn with_stages(
        mut self,
        transformer: MimiDecoderTransformerStage,
        backend: MimiDecoderBackendStage,
    ) -> Self {
        self.transformer = Some(transformer);
        self.backend = Some(backend);
        self
    }
    /// Installs an optional caller-owned error destination for this dispatch.
    #[must_use]
    pub const fn with_error_out(mut self, error_out: &'a mut MimiError) -> Self {
        self.error_out = RefCell::new(Some(error_out));
        self
    }
    /// Installs callbacks that run synchronously after temporary borrows end.
    /// Neither callback may retain request data, share mutable buffers, or
    /// re-enter the Mimi owner.
    #[must_use]
    pub const fn with_callbacks(
        mut self,
        on_done: Option<DecodeDoneCallback>,
        on_error: Option<DecodeErrorCallback>,
    ) -> Self {
        self.on_done = on_done;
        self.on_error = on_error;
        self
    }
}

/// Caller-owned streaming state reset request.
#[derive(Debug)]
pub struct EventResetStreamRun<'a> {
    /// Encoder state to rewind.
    pub encoder: &'a mut encoder::EncoderStreamingState<'a>,
    /// Decoder state to rewind.
    pub decoder: &'a mut MimiDecoderStreamingState<'a>,
}
impl<'a> EventResetStreamRun<'a> {
    /// Creates a reset request over both direction states.
    #[must_use]
    pub const fn new(
        encoder: &'a mut encoder::EncoderStreamingState<'a>,
        decoder: &'a mut MimiDecoderStreamingState<'a>,
    ) -> Self {
        Self { encoder, decoder }
    }
}
/// Public top-level event accepted by the synchronous facade.
pub enum MimiEvent<'a> {
    /// Bind the prepared model and caller-owned arenas.
    Init(InitRun<'a>),
    /// Encode one PCM frame.
    Encode(EncodeRun<'a>),
    /// Decode one code prefix into one PCM frame.
    Decode(DecodeRun<'a>),
    /// Reset both streaming directions.
    Reset(EventResetStreamRun<'a>),
    /// Capture scalar binding and lifecycle diagnostics without numeric work.
    CaptureDiagnostics(EventCaptureDiagnostics<'a>),
}
/// Owned runtime payload for the synchronous initialize dispatch.
///
/// The generated machine keeps this payload borrowed immutably while its
/// phases advance; the request itself remains caller-borrowed through its
/// arena references, and `RefCell` provides the one controlled mutable access
/// needed for the optional error destination.
#[derive(Debug)]
pub struct EventInitRun<'event> {
    /// Caller-owned initialize request carried through all phases.
    pub request: RefCell<InitRun<'event>>,
}

impl<'event> EventInitRun<'event> {
    /// Wraps one initialize request for a synchronous dispatch.
    #[must_use]
    pub fn new(request: InitRun<'event>) -> Self {
        Self {
            request: RefCell::new(request),
        }
    }
}

pub struct EventEncodeRun<'event> {
    pub request: EncodeRun<'event>,
}
impl<'event> EventEncodeRun<'event> {
    #[must_use]
    pub fn new(request: EncodeRun<'event>) -> Self {
        Self { request }
    }
}

/// Owned runtime payload for one decode dispatch.
#[derive(Debug)]
pub struct EventDecodeRun<'event> {
    pub request: DecodeRun<'event>,
}
impl<'event> EventDecodeRun<'event> {
    #[must_use]
    pub fn new(request: DecodeRun<'event>) -> Self {
        Self { request }
    }
}

sml! {
    SpeechCodecMimi<'dispatch, 'event>
    where
        'event: 'dispatch,
    {
        "state_bind_contract_decision"_s <= *"state_uninitialized"_s + EventInitRun(&'dispatch EventInitRun<'event>),
        "state_bind_f32_capacity_decision"_s <= "state_bind_contract_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_bind_f32_contract_valid],
        "state_bind_native_capacity_decision"_s <= "state_bind_contract_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_bind_native_contract_valid],
        "state_bind_q8_capacity_decision"_s <= "state_bind_contract_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_bind_q8_contract_valid],
        "state_init_failed_error_out_decision"_s <= "state_bind_contract_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_bind_contract_invalid] / effect_mark_bind_failed,
        "state_init_failed_callback_decision"_s <= "state_init_failed_error_out_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_has_error_out_init_run] / effect_store_error_out_init_run,
        "state_init_failed_callback_decision"_s <= "state_init_failed_error_out_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_no_error_out_init_run],
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_has_error_callback_init_run] / effect_emit_initialize_error,
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_no_error_callback_init_run],
        "state_binding"_s <= "state_bind_f32_capacity_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_arena_capacity_valid] / effect_bind,
        "state_init_failed_error_out_decision"_s <= "state_bind_f32_capacity_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_arena_capacity_invalid] / effect_mark_arena_capacity_invalid,
        "state_binding"_s <= "state_bind_native_capacity_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_arena_capacity_valid] / effect_bind,
        "state_init_failed_error_out_decision"_s <= "state_bind_native_capacity_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_arena_capacity_invalid] / effect_mark_arena_capacity_invalid,
        "state_binding"_s <= "state_bind_q8_capacity_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_arena_capacity_valid] / effect_bind,
        "state_init_error_out_decision"_s <= "state_binding"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>),
        "state_init_callback_decision"_s <= "state_init_error_out_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_has_error_out_init_run] / effect_store_error_out_init_run,
        "state_init_callback_decision"_s <= "state_init_error_out_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_no_error_out_init_run],
        "state_session_ready"_s <= "state_init_callback_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_has_done_callback_init_run] / effect_emit_initialize_done,
        "state_session_ready"_s <= "state_init_callback_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_no_done_callback_init_run],
        "state_init_failed_error_out_decision"_s <= "state_bind_q8_capacity_decision"_s + completion<EventInitRun>(&'dispatch EventInitRun<'event>) [guard_arena_capacity_invalid] / effect_mark_arena_capacity_invalid,
        "state_encode_request_decision"_s <= "state_session_ready"_s + EventEncodeRun(&'dispatch EventEncodeRun<'event>),
        "state_encoding"_s <= "state_encode_request_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_encode_request_valid] / effect_run_frontend_child,
        "state_encode_failed_error_out_decision"_s <= "state_encode_request_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_encode_request_invalid] / effect_mark_request_shape_invalid_encode_run,
        "state_quantizing"_s <= "state_encoding"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) / effect_run_quantize_child,
        "state_encode_error_out_decision"_s <= "state_quantizing"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>),
        "state_encode_callback_decision"_s <= "state_encode_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_has_error_out_encode_run] / effect_store_error_out_encode_run,
        "state_encode_callback_decision"_s <= "state_encode_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_no_error_out_encode_run],
        "state_encode_failed_callback_decision"_s <= "state_encode_failed_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_has_error_out_encode_run] / effect_store_error_out_encode_run,
        "state_encode_failed_callback_decision"_s <= "state_encode_failed_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_no_error_out_encode_run],
        "state_session_ready"_s <= "state_encode_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_has_done_callback_encode_run] / effect_emit_encode_done,
        "state_session_ready"_s <= "state_encode_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_no_done_callback_encode_run],
        "state_session_ready"_s <= "state_encode_failed_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_has_error_callback_encode_run] / effect_emit_encode_error,
        "state_session_ready"_s <= "state_encode_failed_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_no_error_callback_encode_run],
        "state_decode_request_decision"_s <= "state_session_ready"_s + EventDecodeRun(&'dispatch EventDecodeRun<'event>),
        "state_decode_codes_decision"_s <= "state_decode_request_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_decode_request_valid],
        "state_decode_failed_error_out_decision"_s <= "state_decode_request_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_decode_request_invalid] / effect_mark_request_shape_invalid_decode_run,
        "state_dequantizing"_s <= "state_decode_codes_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_decode_codes_valid] / effect_run_dequantize_child,
        "state_decode_failed_error_out_decision"_s <= "state_decode_codes_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_decode_codes_invalid] / effect_mark_code_range_invalid,
        "state_decoding"_s <= "state_dequantizing"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) / effect_run_backend_child,
        "state_decode_error_out_decision"_s <= "state_decoding"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>),
        "state_decode_callback_decision"_s <= "state_decode_error_out_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_has_error_out_decode_run] / effect_store_error_out_decode_run,
        "state_decode_callback_decision"_s <= "state_decode_error_out_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_no_error_out_decode_run],
        "state_decode_failed_callback_decision"_s <= "state_decode_failed_error_out_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_has_error_out_decode_run] / effect_store_error_out_decode_run,
        "state_decode_failed_callback_decision"_s <= "state_decode_failed_error_out_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_no_error_out_decode_run],
        "state_session_ready"_s <= "state_decode_callback_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_has_done_callback_decode_run] / effect_emit_decode_done,
        "state_session_ready"_s <= "state_decode_callback_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_no_done_callback_decode_run],
        "state_session_ready"_s <= "state_decode_failed_callback_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_has_error_callback_decode_run] / effect_emit_decode_error,
        "state_session_ready"_s <= "state_decode_failed_callback_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_no_error_callback_decode_run],
        "state_session_ready"_s <= "state_session_ready"_s + ResetStream(&'dispatch mut EventResetStreamRun<'event>) / effect_reset_stream,
        "state_uninitialized"_s <= "state_uninitialized"_s + EventCaptureDiagnostics(&'dispatch EventCaptureDiagnostics<'event>) / effect_capture_diagnostics,
        "state_session_ready"_s <= "state_session_ready"_s + EventCaptureDiagnostics(&'dispatch EventCaptureDiagnostics<'event>) / effect_capture_diagnostics,
        "state_uninit_encode_error_out_decision"_s <= "state_uninitialized"_s + EventEncodeRun(&'dispatch EventEncodeRun<'event>) / effect_mark_not_initialized_encode_run,
        "state_uninit_encode_callback_decision"_s <= "state_uninit_encode_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_has_error_out_encode_run] / effect_store_error_out_encode_run,
        "state_uninit_encode_callback_decision"_s <= "state_uninit_encode_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_no_error_out_encode_run],
        "state_uninitialized"_s <= "state_uninit_encode_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_has_error_callback_encode_run] / effect_emit_encode_error,
        "state_uninitialized"_s <= "state_uninit_encode_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_no_error_callback_encode_run],
        "state_uninit_decode_error_out_decision"_s <= "state_uninitialized"_s + EventDecodeRun(&'dispatch EventDecodeRun<'event>) / effect_mark_not_initialized_decode_run,
        "state_uninit_decode_callback_decision"_s <= "state_uninit_decode_error_out_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_has_error_out_decode_run] / effect_store_error_out_decode_run,
        "state_uninit_decode_callback_decision"_s <= "state_uninit_decode_error_out_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_no_error_out_decode_run],
        "state_uninitialized"_s <= "state_uninit_decode_callback_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_has_error_callback_decode_run] / effect_emit_decode_error,
        "state_uninitialized"_s <= "state_uninit_decode_callback_decision"_s + completion<EventDecodeRun>(&'dispatch EventDecodeRun<'event>) [guard_no_error_callback_decode_run],
        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> / effect_mark_unexpected,
        "state_session_ready"_s <= "state_session_ready"_s + unexpected_event<_> / effect_mark_unexpected,
    }
}

/// Persistent top-level control-plane state.
pub struct SpeechCodecMimiContext {
    initialized: bool,
    frame_samples: u32,
    n_q: u32,
    dim: usize,
    card: usize,
    semantic_n_q: usize,
    codebook_dim: usize,
    prepared_floats: usize,
    state_floats: usize,
    workspace_floats: usize,
    frame_floats: usize,
    error: MimiError,
    latent: [f32; MAX_LATENT_FLOATS],
    frontend: encoder::SpeechCodecMimiEncoder,
    quantizer: quantizer::SpeechCodecMimiQuantizer,
    backend: decoder::SpeechCodecMimiDecoder,
}
impl Default for SpeechCodecMimiContext {
    fn default() -> Self {
        Self {
            initialized: false,
            frame_samples: 0,
            n_q: 0,
            dim: 0,
            card: 0,
            semantic_n_q: 0,
            codebook_dim: 0,
            prepared_floats: 0,
            state_floats: 0,
            workspace_floats: 0,
            frame_floats: 0,
            error: MimiError::None,
            latent: [0.0; MAX_LATENT_FLOATS],
            frontend: encoder::SpeechCodecMimiEncoder::new(),
            quantizer: quantizer::SpeechCodecMimiQuantizer::new(),
            backend: decoder::SpeechCodecMimiDecoder::new(),
        }
    }
}
impl fmt::Debug for SpeechCodecMimiContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpeechCodecMimiContext")
            .field("initialized", &self.initialized)
            .field("frame_samples", &self.frame_samples)
            .field("n_q", &self.n_q)
            .field("dim", &self.dim)
            .field("card", &self.card)
            .field("semantic_n_q", &self.semantic_n_q)
            .field("codebook_dim", &self.codebook_dim)
            .field("prepared_floats", &self.prepared_floats)
            .field("state_floats", &self.state_floats)
            .field("workspace_floats", &self.workspace_floats)
            .field("frame_floats", &self.frame_floats)
            .field("error", &self.error)
            .field("latent", &self.latent)
            // Child actors are intentionally opaque at this public boundary.
            .finish_non_exhaustive()
    }
}
impl SpeechCodecMimiContext {
    #[allow(
        clippy::unnecessary_wraps,
        reason = "SML effect callbacks must retain the generated Result signature"
    )]
    fn mark(&mut self, error: MimiError) -> Result<(), ()> {
        self.error = error;
        Ok(())
    }
    fn clear(&mut self) {
        self.error = MimiError::None;
    }
}
impl SpeechCodecMimiContext {
    #[allow(
        clippy::unused_self,
        reason = "SML guard callbacks require the composed context receiver"
    )]
    fn guard_bind_contract_valid(&self, event: &EventInitRun<'_>) -> Result<bool, ()> {
        let request = event.request.borrow();
        let runtime = request.binding.codec_runtime();
        let h = runtime.model().hparams();
        Ok(
            runtime.model().component() == emel_model::bridge::MoshiComponent::Mimi
                && runtime.model().architecture_name() == b"moshi"
                && runtime.model().tensor_count() > 0
                && h.validate().is_ok()
                && runtime.frame_samples() == h.frame_samples().ok_or(())?
                && runtime.n_q() == u32::try_from(h.n_q()).map_err(|_| ())?
                && h.dim() > 0
                && h.n_q() > 0
                && h.card() > 0
                && h.semantic_n_q() > 0
                && h.codebook_dim() > 0
                && h.semantic_n_q() < h.n_q(),
        )
    }
}

impl SpeechCodecMimiStateMachineContext for SpeechCodecMimiContext {
    fn effect_capture_diagnostics<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventCaptureDiagnostics<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event.out.set(MimiDiagnostics {
            initialized: self.initialized,
            frame_samples: self.frame_samples,
            n_q: self.n_q,
            dim: self.dim,
            card: self.card,
            semantic_n_q: self.semantic_n_q,
            codebook_dim: self.codebook_dim,
            prepared_floats: self.prepared_floats,
            state_floats: self.state_floats,
            workspace_floats: self.workspace_floats,
            frame_floats: self.frame_floats,
        });
        Ok(())
    }

    fn effect_bind<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let request = event.request.borrow();
        let runtime = request.binding.codec_runtime();
        let h = runtime.model().hparams();
        let arenas = runtime.arenas();
        self.initialized = true;
        self.frame_samples = runtime.frame_samples();
        self.n_q = runtime.n_q();
        self.dim = usize::try_from(h.dim()).map_err(|_| ())?;
        self.card = usize::try_from(h.card()).map_err(|_| ())?;
        self.semantic_n_q = usize::try_from(h.semantic_n_q()).map_err(|_| ())?;
        self.codebook_dim = usize::try_from(h.codebook_dim()).map_err(|_| ())?;
        self.prepared_floats = arenas.prepared_floats();
        self.state_floats = arenas.state_floats();
        self.workspace_floats = arenas.workspace_floats();
        self.frame_floats = arenas.frame_floats();
        self.error = MimiError::None;
        Ok(())
    }
    fn effect_emit_initialize_done<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.request.borrow().on_done {
            callback(InitializeDone {
                frame_samples: self.frame_samples,
                n_q: self.n_q,
            });
        }
        Ok(())
    }
    fn effect_emit_initialize_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.request.borrow().on_error {
            callback(InitializeError { error: self.error });
        }
        Ok(())
    }
    fn effect_emit_encode_done<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.request.on_done {
            callback(EncodeDone { n_q: self.n_q });
        }
        Ok(())
    }
    fn effect_emit_encode_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.request.on_error {
            callback(EncodeError { error: self.error });
        }
        Ok(())
    }
    fn effect_emit_decode_done<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.request.on_done {
            callback(DecodeDone {
                frame_samples: self.frame_samples,
            });
        }
        Ok(())
    }
    fn effect_emit_decode_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.request.on_error {
            callback(DecodeError { error: self.error });
        }
        Ok(())
    }
    fn effect_mark_bind_failed<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventInitRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(MimiError::BindFailed)
    }
    fn effect_mark_arena_capacity_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventInitRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(MimiError::ArenaCapacity)
    }
    fn effect_mark_request_shape_invalid_encode_run<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(MimiError::RequestShape)
    }
    fn effect_mark_request_shape_invalid_decode_run<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventDecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(MimiError::RequestShape)
    }
    fn effect_mark_code_range_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventDecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(MimiError::CodeRange)
    }
    fn effect_mark_not_initialized_encode_run<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(MimiError::NotInitialized)
    }
    fn effect_mark_not_initialized_decode_run<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventDecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.mark(MimiError::NotInitialized)
    }
    fn effect_run_frontend_child<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let result = {
            let request = &event.request;
            let mut streaming = request.streaming.borrow_mut();
            let mut frame = request.frame.borrow_mut();
            let mut workspace = request.workspace.borrow_mut();
            let child = encoder::Encode::new(
                request.encoder_runtime,
                &mut streaming,
                request.pcm,
                &mut frame,
                &mut workspace,
                &mut self.latent[..self.dim],
            );
            self.frontend.process_event(child)
        };
        if result.is_err() {
            self.error = MimiError::RequestShape;
        }
        Ok(())
    }
    fn effect_run_quantize_child<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let result = {
            let request = &event.request;
            let mut codes_out = request.codes_out.borrow_mut();
            let mut workspace = request.workspace.borrow_mut();
            let child = quantizer::Encode::new(
                request.quantizer_runtime,
                &self.latent[..self.dim],
                &mut codes_out,
                &mut workspace,
            );
            self.quantizer.process_encode(child)
        };
        if result.is_err() {
            self.error = MimiError::RequestShape;
        }
        Ok(())
    }
    fn effect_run_dequantize_child<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let request = &event.request;
        let result = {
            let mut workspace = request.workspace.borrow_mut();
            let child = quantizer::Decode::new(
                request.quantizer_runtime,
                request.codes,
                &mut self.latent[..self.dim],
                &mut workspace,
            );
            self.quantizer.process_decode(child)
        };
        if let Err(error) = result {
            self.error = match error {
                quantizer::QuantizerError::None => MimiError::None,
                quantizer::QuantizerError::RuntimeUnbound => MimiError::NotInitialized,
                quantizer::QuantizerError::RequestShape
                | quantizer::QuantizerError::BufferCapacity => MimiError::RequestShape,
                quantizer::QuantizerError::CodeRange => MimiError::CodeRange,
                quantizer::QuantizerError::StageUnavailable
                | quantizer::QuantizerError::UnexpectedEvent => MimiError::UnexpectedEvent,
            };
        }
        Ok(())
    }
    fn effect_run_backend_child<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let request = &event.request;
        let result = {
            let mut streaming = request.streaming.borrow_mut();
            let mut frame = request.frame.borrow_mut();
            let mut workspace = request.workspace.borrow_mut();
            let mut pcm_out = request.pcm_out.borrow_mut();
            let child = decoder::DecodeRequest::new(
                &request.runtime,
                &mut streaming,
                &self.latent[..self.dim],
                &mut frame,
                &mut workspace,
                &mut pcm_out,
            );
            let child = match (request.transformer, request.backend) {
                (Some(transformer), Some(backend)) => {
                    child.with_stages(decoder::native_upsample_stage, transformer, backend)
                }
                _ => child,
            };
            let child_event = decoder::EventDecodeRun::new(child);
            self.backend.process_event(&child_event)
        };
        if let Err(error) = result {
            // Keep the pinned six-value facade taxonomy while preserving the
            // child route's semantics; stage failures are not request shape.
            self.error = MimiError::from_decoder_error(error);
        }
        Ok(())
    }
    fn effect_reset_stream<'dispatch, 'event>(
        &mut self,
        event: &'dispatch mut EventResetStreamRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event.encoder.encoder_positions = 0;
        event.decoder.decoder_positions = 0;
        event.encoder.arena.fill(0.0);
        event.decoder.arena.fill(0.0);
        Ok(())
    }
    fn effect_store_error_out_init_run<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(out) = event.request.borrow_mut().error_out.as_deref_mut() {
            *out = self.error;
        }
        Ok(())
    }
    fn effect_store_error_out_encode_run<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(out) = event.request.error_out.borrow_mut().as_deref_mut() {
            *out = self.error;
        }
        Ok(())
    }
    fn effect_store_error_out_decode_run<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(out) = event.request.error_out.borrow_mut().as_deref_mut() {
            *out = self.error;
        }
        Ok(())
    }

    fn guard_bind_f32_contract_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(
            event.request.borrow().binding.codec_runtime().variant()
                == binding::RuntimeVariant::F32
                && self.guard_bind_contract_valid(event)?,
        )
    }
    fn guard_bind_native_contract_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(
            event.request.borrow().binding.codec_runtime().variant()
                == binding::RuntimeVariant::F16
                && self.guard_bind_contract_valid(event)?,
        )
    }
    fn guard_bind_q8_contract_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(
            event.request.borrow().binding.codec_runtime().variant() == binding::RuntimeVariant::Q8
                && self.guard_bind_contract_valid(event)?,
        )
    }
    fn guard_bind_contract_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_bind_contract_valid(event)?)
    }
    fn guard_arena_capacity_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let request = event.request.borrow();
        let needed = request.binding.codec_runtime().arenas();
        let h = request.binding.codec_runtime().model().hparams();
        Ok(request.prepared.len() >= needed.prepared_floats()
            && request.state_arena.len() >= needed.state_floats()
            && request.workspace.len() >= needed.workspace_floats()
            && request.frame.len() >= needed.frame_floats()
            && h.dim() > 0
            && usize::try_from(h.dim()).is_ok_and(|dim| dim <= MAX_LATENT_FLOATS))
    }
    fn guard_arena_capacity_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_arena_capacity_valid(event)?)
    }
    fn guard_encode_request_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let request = &event.request;
        let codes_out = request.codes_out.borrow();
        Ok(self.initialized
            && request.pcm.len() == self.frame_samples as usize
            && !request.pcm.is_empty()
            && codes_out.len() >= self.n_q as usize
            && request.encoder_runtime.model_bound
            && request.encoder_runtime.frame_samples == self.frame_samples as usize
            && request.encoder_runtime.dim == self.dim
            && request.quantizer_runtime.model_bound
            && request.quantizer_runtime.n_q == self.n_q as usize
            && request.quantizer_runtime.dim == self.dim)
    }
    fn guard_encode_request_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_encode_request_valid(event)?)
    }
    fn guard_decode_request_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let request = &event.request;
        let pcm_out = request.pcm_out.borrow();
        Ok(self.initialized
            && !request.codes.is_empty()
            && request.codes.len() >= self.semantic_n_q
            && request.codes.len() <= self.n_q as usize
            && !pcm_out.is_empty()
            && pcm_out.len() >= self.frame_samples as usize
            && request.quantizer_runtime.model_bound
            && request.quantizer_runtime.n_q == self.n_q as usize
            && request.quantizer_runtime.dim == self.dim
            && request.quantizer_runtime.semantic_n_q == self.semantic_n_q
            && request.quantizer_runtime.codebook_dim == self.codebook_dim
            && request.quantizer_runtime.codebook_entries == self.card)
    }
    fn guard_decode_request_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_decode_request_valid(event)?)
    }
    fn guard_decode_codes_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event
            .request
            .codes
            .iter()
            .all(|code| *code >= 0 && usize::try_from(*code).is_ok_and(|code| code < self.card)))
    }
    fn guard_decode_codes_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_decode_codes_valid(event)?)
    }
    fn guard_has_error_out_init_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.borrow().error_out.is_some())
    }
    fn guard_no_error_out_init_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.borrow().error_out.is_none())
    }
    fn guard_has_error_out_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.error_out.borrow().is_some())
    }
    fn guard_no_error_out_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.error_out.borrow().is_none())
    }
    fn guard_has_error_out_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.error_out.borrow().is_some())
    }
    fn guard_no_error_out_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.error_out.borrow().is_none())
    }
    fn guard_has_done_callback_init_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.borrow().on_done.is_some())
    }
    fn guard_no_done_callback_init_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.borrow().on_done.is_none())
    }
    fn guard_has_error_callback_init_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.borrow().on_error.is_some())
    }
    fn guard_no_error_callback_init_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventInitRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.borrow().on_error.is_none())
    }
    fn guard_has_done_callback_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_done.is_some())
    }
    fn guard_no_done_callback_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_done.is_none())
    }
    fn guard_has_error_callback_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_error.is_some())
    }
    fn guard_no_error_callback_encode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_error.is_none())
    }
    fn guard_has_done_callback_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_done.is_some())
    }
    fn guard_no_done_callback_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_done.is_none())
    }
    fn guard_has_error_callback_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_error.is_some())
    }
    fn guard_no_error_callback_decode_run<'dispatch, 'event>(
        &self,
        event: &'dispatch EventDecodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.on_error.is_none())
    }
    fn effect_mark_unexpected(&mut self) -> Result<(), ()> {
        self.mark(MimiError::UnexpectedEvent)
    }
}

/// Single-writer synchronous top-level Mimi actor.
pub struct SpeechCodecMimi {
    machine: SpeechCodecMimiStateMachine<SpeechCodecMimiContext>,
}
impl Default for SpeechCodecMimi {
    fn default() -> Self {
        Self::new()
    }
}
impl SpeechCodecMimi {
    /// Constructs an actor in generated `state_uninitialized`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: SpeechCodecMimiStateMachine::new(SpeechCodecMimiContext::default()),
        }
    }
    /// Dispatches one top-level event synchronously to completion.
    pub fn process_event(&mut self, event: MimiEvent<'_>) -> Result<(), MimiError> {
        match event {
            MimiEvent::Init(event) => self.process_init(event),
            MimiEvent::Encode(event) => self.process_encode(event),
            MimiEvent::Decode(event) => self.process_decode(event),
            MimiEvent::Reset(event) => self.process_reset(event),
            MimiEvent::CaptureDiagnostics(event) => self.capture_diagnostics(event),
        }
    }
    pub fn capture_diagnostics(
        &mut self,
        event: EventCaptureDiagnostics<'_>,
    ) -> Result<(), MimiError> {
        // Diagnostics are a read-only observer and must not inherit a prior
        // operation's transient facade error.
        self.machine.context_mut().clear();
        if self
            .machine
            .process_event(SpeechCodecMimiEvents::EventCaptureDiagnostics(&event))
            .is_err()
        {
            self.machine.context_mut().error = MimiError::UnexpectedEvent;
        }
        self.result()
    }
    pub fn process_init(&mut self, event: InitRun<'_>) -> Result<(), MimiError> {
        self.machine.context_mut().clear();
        let event = EventInitRun::new(event);
        if self
            .machine
            .process_event(SpeechCodecMimiEvents::EventInitRun(&event))
            .is_err()
        {
            self.machine.context_mut().error = MimiError::UnexpectedEvent;
        }
        self.result()
    }
    /// Dispatches encode synchronously.
    pub fn process_encode(&mut self, event: EncodeRun<'_>) -> Result<(), MimiError> {
        self.machine.context_mut().clear();
        let event = EventEncodeRun::new(event);
        if self
            .machine
            .process_event(SpeechCodecMimiEvents::EventEncodeRun(&event))
            .is_err()
        {
            self.machine.context_mut().error = MimiError::UnexpectedEvent;
        }
        self.result()
    }
    /// Dispatches decode synchronously.
    pub fn process_decode(&mut self, event: DecodeRun<'_>) -> Result<(), MimiError> {
        self.machine.context_mut().clear();
        let event = EventDecodeRun::new(event);
        if self
            .machine
            .process_event(SpeechCodecMimiEvents::EventDecodeRun(&event))
            .is_err()
        {
            self.machine.context_mut().error = MimiError::UnexpectedEvent;
        }
        self.result()
    }
    /// Dispatches reset synchronously.
    pub fn process_reset(&mut self, mut event: EventResetStreamRun<'_>) -> Result<(), MimiError> {
        self.machine.context_mut().clear();
        if self
            .machine
            .process_event(SpeechCodecMimiEvents::ResetStream(&mut event))
            .is_err()
        {
            self.machine.context_mut().error = MimiError::UnexpectedEvent;
        }
        self.result()
    }
    fn result(&self) -> Result<(), MimiError> {
        let error = self.machine.context().error;
        if error == MimiError::None {
            Ok(())
        } else {
            Err(error)
        }
    }
    /// Returns generated state inspection data.
    /// Reports whether the generated facade remains in its initial state.
    #[must_use]
    pub fn is_uninitialized(&self) -> bool {
        matches!(
            self.machine.state(),
            SpeechCodecMimiStates::StateUninitialized
        )
    }
    #[must_use]
    pub fn state(&self) -> &SpeechCodecMimiStates {
        self.machine.state()
    }
    #[must_use]
    pub fn context(&self) -> &SpeechCodecMimiContext {
        self.machine.context()
    }
}
/// Short alias matching the pinned C++ `sm` surface.
pub type Mimi = SpeechCodecMimi;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_zeroed_diagnostics_before_initialization() {
        let mut actor = SpeechCodecMimi::new();
        let diagnostics = Cell::new(MimiDiagnostics::default());
        assert_eq!(
            actor.capture_diagnostics(EventCaptureDiagnostics::new(&diagnostics)),
            Ok(())
        );
        assert_eq!(diagnostics.get(), MimiDiagnostics::default());
        assert!(matches!(
            actor.state(),
            SpeechCodecMimiStates::StateUninitialized
        ));
    }

    #[test]
    fn decoder_child_error_mapping_preserves_facade_semantics() {
        assert_eq!(
            MimiError::from_decoder_error(decoder::DecoderError::RuntimeUnbound),
            MimiError::NotInitialized
        );
        assert_eq!(
            MimiError::from_decoder_error(decoder::DecoderError::RequestShape),
            MimiError::RequestShape
        );
        assert_eq!(
            MimiError::from_decoder_error(decoder::DecoderError::BufferCapacity),
            MimiError::ArenaCapacity
        );
        assert_eq!(
            MimiError::from_decoder_error(decoder::DecoderError::StageUnavailable),
            MimiError::UnexpectedEvent
        );
        assert_eq!(
            MimiError::from_decoder_error(decoder::DecoderError::StageFailed),
            MimiError::UnexpectedEvent
        );
    }
}
