//! Bounded synchronous text renderer actor aligned with the pinned C++ renderer contract.

#![allow(
    clippy::enum_variant_names,
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

// --- machine TextRenderer from emel.cpp/src/emel/text/renderer/sm.hpp ---
#[derive(Debug)]
pub struct EventFlushRuntime<'a> { pub request: &'a mut FlushRequest<'a>, pub context: &'a mut FlushContext }

#[derive(Debug)]
pub struct EventInitializeRuntime<'a> { pub request: &'a mut InitializeRequest<'a>, pub context: &'a mut InitializeContext }

#[derive(Debug)]
pub struct EventRenderRuntime<'a> { pub request: &'a mut RenderRequest<'a>, pub context: &'a mut RenderContext }

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)] #[repr(u8)]
pub enum SequenceStatus { #[default] Running = 0, StopSequenceMatched = 1 }
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)] #[repr(i32)]
pub enum RendererError { #[default] None = 0, InvalidRequest = 1, Backend = 2, ModelInvalid = 4, Internal = 8, Untracked = 16 }
impl RendererError { pub const fn code(self) -> i32 { self as i32 } }
#[derive(Clone, Copy, Debug, Eq, PartialEq)] pub struct InitializeDone;
#[derive(Clone, Copy, Debug, Eq, PartialEq)] pub struct InitializeError { pub error: RendererError }
#[derive(Clone, Copy, Debug, Eq, PartialEq)] pub struct RenderDone { pub output_length: usize, pub status: SequenceStatus }
#[derive(Clone, Copy, Debug, Eq, PartialEq)] pub struct RenderError { pub error: RendererError }
#[derive(Clone, Copy, Debug, Eq, PartialEq)] pub struct FlushDone { pub output_length: usize, pub status: SequenceStatus }
#[derive(Clone, Copy, Debug, Eq, PartialEq)] pub struct FlushError { pub error: RendererError }
pub type InitializeDoneCallback = fn(InitializeDone) -> bool;
pub type InitializeErrorCallback = fn(InitializeError) -> bool;
pub type RenderDoneCallback = fn(RenderDone) -> bool;
pub type RenderErrorCallback = fn(RenderError) -> bool;
pub type FlushDoneCallback = fn(FlushDone) -> bool;
pub type FlushErrorCallback = fn(FlushError) -> bool;
pub struct InitializeRequest<'a> { pub strip_leading_space: bool, pub stop_sequences: &'a [&'a [u8]], pub dispatch_done: Option<InitializeDoneCallback>, pub dispatch_error: Option<InitializeErrorCallback> }
pub struct RenderRequest<'a> { pub token_id: i32, pub sequence_id: i32, pub emit_special: bool, pub output: &'a mut [u8], pub output_length: &'a mut usize, pub status: &'a mut SequenceStatus, pub error: &'a mut RendererError, pub dispatch_done: Option<RenderDoneCallback>, pub dispatch_error: Option<RenderErrorCallback> }
pub struct FlushRequest<'a> { pub sequence_id: i32, pub output: &'a mut [u8], pub output_length: &'a mut usize, pub status: &'a mut SequenceStatus, pub error: &'a mut RendererError, pub dispatch_done: Option<FlushDoneCallback>, pub dispatch_error: Option<FlushErrorCallback> }
#[derive(Clone, Copy, Debug, Default)] pub struct InitializeContext { pub error: RendererError, pub detokenizer_error: Option<crate::detokenizer::DetokenizeError> }
#[derive(Clone, Copy, Debug, Default)] pub struct RenderContext { pub error: RendererError, pub output_length: usize, pub status: SequenceStatus, pub sequence_index: usize, pub detokenizer_error: Option<crate::detokenizer::DetokenizeError>, pub detokenizer_output_length: usize, pub detokenizer_pending_length: usize, pub produced_length: usize, pub leading_space_prefix_length: usize }
#[derive(Clone, Copy, Debug, Default)] pub struct FlushContext { pub error: RendererError, pub output_length: usize, pub status: SequenceStatus, pub sequence_index: usize }

sml! {
    TextRenderer {
        "initializing"_s <= *"uninitialized"_s + event<EventInitializeRuntime<'event>> [valid_initialize] / begin_initialize_from_uninitialized,
        "initialize_publish_error"_s <= "uninitialized"_s + event<EventInitializeRuntime<'event>> [invalid_initialize] / reject_initialize_from_uninitialized,
        "render_publish_error"_s <= "uninitialized"_s + event<EventRenderRuntime<'event>> / reject_render_from_uninitialized,
        "flush_publish_error"_s <= "uninitialized"_s + event<EventFlushRuntime<'event>> / reject_flush_from_uninitialized,
        "initializing"_s <= "initialized"_s + event<EventInitializeRuntime<'event>> [valid_initialize] / begin_initialize_from_initialized,
        "initialize_publish_error"_s <= "initialized"_s + event<EventInitializeRuntime<'event>> [invalid_initialize] / reject_initialize_from_initialized,
        "rendering"_s <= "initialized"_s + event<EventRenderRuntime<'event>> [valid_render] / begin_render_from_initialized,
        "render_publish_error"_s <= "initialized"_s + event<EventRenderRuntime<'event>> [invalid_render] / reject_render_from_initialized,
        "flushing"_s <= "initialized"_s + event<EventFlushRuntime<'event>> [valid_flush] / begin_flush_from_initialized,
        "flush_publish_error"_s <= "initialized"_s + event<EventFlushRuntime<'event>> [invalid_flush] / reject_flush_from_initialized,
        "initializing"_s <= "done"_s + event<EventInitializeRuntime<'event>> [valid_initialize] / begin_initialize_from_done,
        "initialize_publish_error"_s <= "done"_s + event<EventInitializeRuntime<'event>> [invalid_initialize] / reject_initialize_from_done,
        "rendering"_s <= "done"_s + event<EventRenderRuntime<'event>> [valid_render] / begin_render_from_done,
        "render_publish_error"_s <= "done"_s + event<EventRenderRuntime<'event>> [invalid_render] / reject_render_from_done,
        "flushing"_s <= "done"_s + event<EventFlushRuntime<'event>> [valid_flush] / begin_flush_from_done,
        "flush_publish_error"_s <= "done"_s + event<EventFlushRuntime<'event>> [invalid_flush] / reject_flush_from_done,
        "initializing"_s <= "errored"_s + event<EventInitializeRuntime<'event>> [valid_initialize] / begin_initialize_from_errored,
        "initialize_publish_error"_s <= "errored"_s + event<EventInitializeRuntime<'event>> [invalid_initialize] / reject_initialize_from_errored,
        "rendering"_s <= "errored"_s + event<EventRenderRuntime<'event>> [valid_render] / begin_render_from_errored,
        "render_publish_error"_s <= "errored"_s + event<EventRenderRuntime<'event>> [invalid_render] / reject_render_from_errored,
        "flushing"_s <= "errored"_s + event<EventFlushRuntime<'event>> [valid_flush] / begin_flush_from_errored,
        "flush_publish_error"_s <= "errored"_s + event<EventFlushRuntime<'event>> [invalid_flush] / reject_flush_from_errored,
        "initializing"_s <= "unexpected"_s + event<EventInitializeRuntime<'event>> [valid_initialize] / begin_initialize_from_unexpected,
        "unexpected"_s <= "unexpected"_s + event<EventInitializeRuntime<'event>> [invalid_initialize] / reject_initialize_from_unexpected,
        "rendering"_s <= "unexpected"_s + event<EventRenderRuntime<'event>> [valid_render] / begin_render_from_unexpected,
        "unexpected"_s <= "unexpected"_s + event<EventRenderRuntime<'event>> [invalid_render] / reject_render_from_unexpected,
        "flushing"_s <= "unexpected"_s + event<EventFlushRuntime<'event>> [valid_flush] / begin_flush_from_unexpected,
        "unexpected"_s <= "unexpected"_s + event<EventFlushRuntime<'event>> [invalid_flush] / reject_flush_from_unexpected,
        "initialize_publish_success"_s <= "initialization_decision"_s + completion<EventInitializeRuntime<'event>> [initialize_dispatch_ok] / commit_initialize_success,
        "initialize_publish_error"_s <= "initialization_decision"_s + completion<EventInitializeRuntime<'event>> [initialize_dispatch_backend_failure] / set_backend_error_event_initialize_runtime,
        "initialize_publish_error"_s <= "initialization_decision"_s + completion<EventInitializeRuntime<'event>> [initialize_dispatch_reported_error] / set_error_from_detokenizer_event_initialize_runtime,
        "initialize_publish_error"_s <= "initialization_decision"_s + completion<EventInitializeRuntime<'event>> / set_error_from_detokenizer_event_initialize_runtime,
        "initialized"_s <= "initialize_publish_success"_s + completion<EventInitializeRuntime<'event>> / publish_initialize_done,
        "errored"_s <= "initialize_publish_error"_s + completion<EventInitializeRuntime<'event>> / publish_initialize_error,
        "initialization_decision"_s <= "initializing"_s + completion<EventInitializeRuntime<'event>> / dispatch_initialize_detokenizer,
        "render_publish_success"_s <= "rendering"_s + completion<EventRenderRuntime<'event>> [sequence_stop_matched] / render_sequence_already_stopped,
        "render_dispatch_decision"_s <= "rendering"_s + completion<EventRenderRuntime<'event>> [sequence_running] / dispatch_render_detokenizer,
        "render_result_decision"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime<'event>> [render_dispatch_ok],
        "render_publish_error"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime<'event>> [render_dispatch_backend_failure] / set_backend_error_event_render_runtime,
        "render_publish_error"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime<'event>> [render_dispatch_reported_error] / set_error_from_detokenizer_event_render_runtime,
        "render_publish_error"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime<'event>> [render_dispatch_lengths_invalid] / set_invalid_request_event_render_runtime,
        "render_publish_error"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime<'event>> / ensure_last_error_from_render_dispatch_decision,
        "render_commit_output_exec"_s <= "render_result_decision"_s + completion<EventRenderRuntime<'event>>,
        "render_strip_decision"_s <= "render_commit_output_exec"_s + completion<EventRenderRuntime<'event>> / commit_render_detokenizer_output,
        "render_strip_prefix_scan_exec"_s <= "render_strip_decision"_s + completion<EventRenderRuntime<'event>> [strip_needed],
        "render_strip_state_exec"_s <= "render_strip_decision"_s + completion<EventRenderRuntime<'event>> [strip_not_needed],
        "render_publish_error"_s <= "render_strip_decision"_s + completion<EventRenderRuntime<'event>> / ensure_last_error_from_render_strip_decision,
        "render_strip_prefix_decision"_s <= "render_strip_prefix_scan_exec"_s + completion<EventRenderRuntime<'event>> / compute_render_leading_space_prefix,
        "render_strip_apply_exec"_s <= "render_strip_prefix_decision"_s + completion<EventRenderRuntime<'event>> [strip_prefix_nonzero] / apply_render_leading_space_strip,
        "render_strip_state_exec"_s <= "render_strip_prefix_decision"_s + completion<EventRenderRuntime<'event>> [strip_prefix_zero],
        "render_publish_error"_s <= "render_strip_prefix_decision"_s + completion<EventRenderRuntime<'event>> / ensure_last_error_from_render_strip_prefix_decision,
        "render_strip_state_exec"_s <= "render_strip_apply_exec"_s + completion<EventRenderRuntime<'event>>,
        "render_stop_match_exec"_s <= "render_strip_state_exec"_s + completion<EventRenderRuntime<'event>> / update_render_strip_state,
        "render_finalize_decision"_s <= "render_stop_match_exec"_s + completion<EventRenderRuntime<'event>> / apply_render_stop_matching,
        "render_publish_success"_s <= "render_finalize_decision"_s + completion<EventRenderRuntime<'event>> [request_ok] / mark_done,
        "render_publish_error"_s <= "render_finalize_decision"_s + completion<EventRenderRuntime<'event>> [request_failed] / ensure_last_error_from_render_finalize_decision,
        "render_publish_error"_s <= "render_finalize_decision"_s + completion<EventRenderRuntime<'event>> / ensure_last_error_from_render_finalize_decision,
        "done"_s <= "render_publish_success"_s + completion<EventRenderRuntime<'event>> / publish_render_done,
        "errored"_s <= "render_publish_error"_s + completion<EventRenderRuntime<'event>> / publish_render_error,
        "flush_publish_success"_s <= "flushing"_s + completion<EventFlushRuntime<'event>> [flush_output_fits] / flush_copy_sequence_buffers,
        "flush_publish_error"_s <= "flushing"_s + completion<EventFlushRuntime<'event>> [flush_output_too_large] / set_invalid_request_event_flush_runtime,
        "done"_s <= "flush_publish_success"_s + completion<EventFlushRuntime<'event>> / publish_flush_done,
        "errored"_s <= "flush_publish_error"_s + completion<EventFlushRuntime<'event>> / publish_flush_error,
        "unexpected"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "unexpected"_s <= "initializing"_s + unexpected_event<_> / on_unexpected_from_initializing,
        "unexpected"_s <= "initialization_decision"_s + unexpected_event<_> / on_unexpected_from_initialization_decision,
        "unexpected"_s <= "initialize_publish_success"_s + unexpected_event<_> / on_unexpected_from_initialize_publish_success,
        "unexpected"_s <= "initialize_publish_error"_s + unexpected_event<_> / on_unexpected_from_initialize_publish_error,
        "unexpected"_s <= "initialized"_s + unexpected_event<_> / on_unexpected_from_initialized,
        "unexpected"_s <= "rendering"_s + unexpected_event<_> / on_unexpected_from_rendering,
        "unexpected"_s <= "render_dispatch_decision"_s + unexpected_event<_> / on_unexpected_from_render_dispatch_decision,
        "unexpected"_s <= "render_result_decision"_s + unexpected_event<_> / on_unexpected_from_render_result_decision,
        "unexpected"_s <= "render_commit_output_exec"_s + unexpected_event<_> / on_unexpected_from_render_commit_output_exec,
        "unexpected"_s <= "render_strip_decision"_s + unexpected_event<_> / on_unexpected_from_render_strip_decision,
        "unexpected"_s <= "render_strip_prefix_scan_exec"_s + unexpected_event<_> / on_unexpected_from_render_strip_prefix_scan_exec,
        "unexpected"_s <= "render_strip_prefix_decision"_s + unexpected_event<_> / on_unexpected_from_render_strip_prefix_decision,
        "unexpected"_s <= "render_strip_apply_exec"_s + unexpected_event<_> / on_unexpected_from_render_strip_apply_exec,
        "unexpected"_s <= "render_strip_state_exec"_s + unexpected_event<_> / on_unexpected_from_render_strip_state_exec,
        "unexpected"_s <= "render_stop_match_exec"_s + unexpected_event<_> / on_unexpected_from_render_stop_match_exec,
        "unexpected"_s <= "render_finalize_decision"_s + unexpected_event<_> / on_unexpected_from_render_finalize_decision,
        "unexpected"_s <= "render_publish_success"_s + unexpected_event<_> / on_unexpected_from_render_publish_success,
        "unexpected"_s <= "render_publish_error"_s + unexpected_event<_> / on_unexpected_from_render_publish_error,
        "unexpected"_s <= "flushing"_s + unexpected_event<_> / on_unexpected_from_flushing,
        "unexpected"_s <= "flush_publish_success"_s + unexpected_event<_> / on_unexpected_from_flush_publish_success,
        "unexpected"_s <= "flush_publish_error"_s + unexpected_event<_> / on_unexpected_from_flush_publish_error,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

#[derive(Clone, Copy, Debug, Default)] struct StopEntry { offset: usize, length: usize }
#[derive(Clone, Copy, Debug)] struct SequenceState { pending: [u8; 4], pending_length: usize, holdback: [u8; 31], holdback_length: usize, strip_leading_space: bool, stop_matched: bool }
impl Default for SequenceState { fn default() -> Self { Self { pending: [0;4], pending_length: 0, holdback: [0;31], holdback_length: 0, strip_leading_space: false, stop_matched: false } } }
#[derive(Debug)] pub struct TextRendererContext<'v,V: crate::detokenizer::VocabularyView+?Sized> { pub detokenizer: crate::detokenizer::TextDetokenizer<'v,V>, pub strip_default: bool, pub stop_sequences: [StopEntry;8], pub stop_count: usize, pub stop_storage: [u8;256], pub stop_storage_used: usize, pub stop_max_length: usize, sequences: [SequenceState;64] }
impl<'v,V: crate::detokenizer::VocabularyView+?Sized> TextRendererContext<'v,V> { fn new(v:&'v V)->Self { Self { detokenizer: crate::detokenizer::TextDetokenizer::new(v), strip_default:false, stop_sequences:[StopEntry::default();8], stop_count:0, stop_storage:[0;256], stop_storage_used:0, stop_max_length:0, sequences:[SequenceState::default();64] } } }
impl<'v,V: crate::detokenizer::VocabularyView+?Sized> TextRendererStateMachineContext for TextRendererContext<'v,V> {
 fn apply_render_leading_space_strip(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>Ok(())
 fn apply_render_stop_matching(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>Ok(())
 fn begin_flush_from_done(&mut self, e:&EventFlushRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; e.context.output_length=0; e.context.status=SequenceStatus::Running; e.context.sequence_index=e.request.sequence_id as usize; *e.request.output_length=0; *e.request.status=SequenceStatus::Running; *e.request.error=RendererError::None; Ok(()) }
 fn begin_flush_from_errored(&mut self, e:&EventFlushRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; e.context.output_length=0; e.context.status=SequenceStatus::Running; e.context.sequence_index=e.request.sequence_id as usize; *e.request.output_length=0; *e.request.status=SequenceStatus::Running; *e.request.error=RendererError::None; Ok(()) }
 fn begin_flush_from_initialized(&mut self, e:&EventFlushRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; e.context.output_length=0; e.context.status=SequenceStatus::Running; e.context.sequence_index=e.request.sequence_id as usize; *e.request.output_length=0; *e.request.status=SequenceStatus::Running; *e.request.error=RendererError::None; Ok(()) }
 fn begin_flush_from_unexpected(&mut self, e:&EventFlushRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; e.context.output_length=0; e.context.status=SequenceStatus::Running; e.context.sequence_index=e.request.sequence_id as usize; *e.request.output_length=0; *e.request.status=SequenceStatus::Running; *e.request.error=RendererError::None; Ok(()) }
 fn begin_initialize_from_done(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; Ok(()) }
 fn begin_initialize_from_errored(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; Ok(()) }
 fn begin_initialize_from_initialized(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; Ok(()) }
 fn begin_initialize_from_unexpected(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; Ok(()) }
 fn begin_initialize_from_uninitialized(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; Ok(()) }
 fn begin_render_from_done(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; e.context.output_length=0; e.context.status=SequenceStatus::Running; e.context.sequence_index=e.request.sequence_id as usize; *e.request.output_length=0; *e.request.status=SequenceStatus::Running; *e.request.error=RendererError::None; Ok(()) }
 fn begin_render_from_errored(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; e.context.output_length=0; e.context.status=SequenceStatus::Running; e.context.sequence_index=e.request.sequence_id as usize; *e.request.output_length=0; *e.request.status=SequenceStatus::Running; *e.request.error=RendererError::None; Ok(()) }
 fn begin_render_from_initialized(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; e.context.output_length=0; e.context.status=SequenceStatus::Running; e.context.sequence_index=e.request.sequence_id as usize; *e.request.output_length=0; *e.request.status=SequenceStatus::Running; *e.request.error=RendererError::None; Ok(()) }
 fn begin_render_from_unexpected(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; e.context.output_length=0; e.context.status=SequenceStatus::Running; e.context.sequence_index=e.request.sequence_id as usize; *e.request.output_length=0; *e.request.status=SequenceStatus::Running; *e.request.error=RendererError::None; Ok(()) }
 fn commit_initialize_success(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>Ok(())
 fn commit_render_detokenizer_output(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>Ok(())
 fn compute_render_leading_space_prefix(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>Ok(())
 fn dispatch_initialize_detokenizer(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>Ok(())
 fn dispatch_render_detokenizer(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>Ok(())
 fn ensure_last_error_from_render_dispatch_decision(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ if e.context.error==RendererError::None { e.context.error=RendererError::Backend; } Ok(()) }
 fn ensure_last_error_from_render_finalize_decision(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ if e.context.error==RendererError::None { e.context.error=RendererError::Backend; } Ok(()) }
 fn ensure_last_error_from_render_strip_decision(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ if e.context.error==RendererError::None { e.context.error=RendererError::Backend; } Ok(()) }
 fn ensure_last_error_from_render_strip_prefix_decision(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ if e.context.error==RendererError::None { e.context.error=RendererError::Backend; } Ok(()) }
 fn flush_copy_sequence_buffers(&mut self, e:&EventFlushRuntime<'_>)->Result<(),()>Ok(())
 fn flush_output_fits(&self, e:&EventFlushRuntime<'_>)->Result<bool,()>{ let s=&self.sequences[e.context.sequence_index]; Ok(s.pending_length+s.holdback_length<=e.request.output.len()) }
 fn flush_output_too_large(&self, e:&EventFlushRuntime<'_>)->Result<bool,()>{ Ok(!self.flush_output_fits(e)?) }
 fn initialize_dispatch_backend_failure(&self, e:&EventInitializeRuntime<'_>)->Result<bool,()>{ Ok(false) }
 fn initialize_dispatch_ok(&self, e:&EventInitializeRuntime<'_>)->Result<bool,()>{ Ok(e.context.detokenizer_error.is_none()) }
 fn initialize_dispatch_reported_error(&self, e:&EventInitializeRuntime<'_>)->Result<bool,()>{ Ok(e.context.detokenizer_error.is_some()) }
 fn invalid_flush(&self, e:&EventFlushRuntime<'_>)->Result<bool,()>{ Ok(!self.valid_flush(e)?) }
 fn invalid_initialize(&self, e:&EventInitializeRuntime<'_>)->Result<bool,()>{ Ok(!self.valid_initialize(e)?) }
 fn invalid_render(&self, e:&EventRenderRuntime<'_>)->Result<bool,()>{ Ok(!self.valid_render(e)?) }
 fn mark_done(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::None; Ok(()) }
 fn on_unexpected_from_done(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_errored(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_flush_publish_error(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_flush_publish_success(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_flushing(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_initialization_decision(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_initialize_publish_error(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_initialize_publish_success(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_initialized(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_initializing(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_render_commit_output_exec(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_render_dispatch_decision(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_render_finalize_decision(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_render_publish_error(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_render_publish_success(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_render_result_decision(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_render_stop_match_exec(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_render_strip_apply_exec(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_render_strip_decision(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_render_strip_prefix_decision(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_render_strip_prefix_scan_exec(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_render_strip_state_exec(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_rendering(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_unexpected(&mut self)->Result<(),()>{Ok(())}
 fn on_unexpected_from_uninitialized(&mut self)->Result<(),()>{Ok(())}
 fn publish_flush_done(&mut self, e:&EventFlushRuntime<'_>)->Result<(),()>Ok(())
 fn publish_flush_error(&mut self, e:&EventFlushRuntime<'_>)->Result<(),()>Ok(())
 fn publish_initialize_done(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>Ok(())
 fn publish_initialize_error(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>Ok(())
 fn publish_render_done(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>Ok(())
 fn publish_render_error(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>Ok(())
 fn reject_flush_from_done(&mut self, e:&EventFlushRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_flush_from_errored(&mut self, e:&EventFlushRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_flush_from_initialized(&mut self, e:&EventFlushRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_flush_from_unexpected(&mut self, e:&EventFlushRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_flush_from_uninitialized(&mut self, e:&EventFlushRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_initialize_from_done(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_initialize_from_errored(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_initialize_from_initialized(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_initialize_from_unexpected(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_initialize_from_uninitialized(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_render_from_done(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_render_from_errored(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_render_from_initialized(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_render_from_unexpected(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn reject_render_from_uninitialized(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn render_dispatch_backend_failure(&self, e:&EventRenderRuntime<'_>)->Result<bool,()>{ Ok(false) }
 fn render_dispatch_lengths_invalid(&self, e:&EventRenderRuntime<'_>)->Result<bool,()>{ Ok(e.context.detokenizer_output_length>e.request.output.len()||e.context.detokenizer_pending_length>4) }
 fn render_dispatch_ok(&self, e:&EventRenderRuntime<'_>)->Result<bool,()>{ Ok(e.context.detokenizer_error.is_none()&&e.context.detokenizer_output_length<=e.request.output.len()&&e.context.detokenizer_pending_length<=4) }
 fn render_dispatch_reported_error(&self, e:&EventRenderRuntime<'_>)->Result<bool,()>{ Ok(e.context.detokenizer_error.is_some()) }
 fn render_sequence_already_stopped(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>Ok(())
 fn request_failed(&self, e:&EventInitializeRuntime<'_>)->Result<bool,()>{ Ok(e.context.error!=RendererError::None) }
 fn request_ok(&self, e:&EventInitializeRuntime<'_>)->Result<bool,()>{ Ok(e.context.error==RendererError::None) }
 fn sequence_running(&self, e:&EventInitializeRuntime<'_>)->Result<bool,()>{ Ok(!self.sequences[e.context.sequence_index].stop_matched) }
 fn sequence_stop_matched(&self, e:&EventInitializeRuntime<'_>)->Result<bool,()>{ Ok(self.sequences[e.context.sequence_index].stop_matched) }
 fn set_backend_error_event_initialize_runtime(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::Backend; Ok(()) }
 fn set_backend_error_event_render_runtime(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::Backend; Ok(()) }
 fn set_error_from_detokenizer_event_initialize_runtime(&mut self, e:&EventInitializeRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::Internal; Ok(()) }
 fn set_error_from_detokenizer_event_render_runtime(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::Internal; Ok(()) }
 fn set_invalid_request_event_flush_runtime(&mut self, e:&EventFlushRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn set_invalid_request_event_render_runtime(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>{ e.context.error=RendererError::InvalidRequest; Ok(()) }
 fn strip_needed(&self, e:&EventInitializeRuntime<'_>)->Result<bool,()>{ Ok(e.context.detokenizer_output_length>0&&self.sequences[e.context.sequence_index].strip_leading_space&&matches!(e.request.output.first(),Some(b' '|b'\t'|b'\n'|b'\r'))) }
 fn strip_not_needed(&self, e:&EventInitializeRuntime<'_>)->Result<bool,()>{ Ok(!self.strip_needed(e)?) }
 fn strip_prefix_nonzero(&self, e:&EventInitializeRuntime<'_>)->Result<bool,()>{ Ok(e.context.leading_space_prefix_length!=0) }
 fn strip_prefix_zero(&self, e:&EventInitializeRuntime<'_>)->Result<bool,()>{ Ok(e.context.leading_space_prefix_length==0) }
 fn update_render_strip_state(&mut self, e:&EventRenderRuntime<'_>)->Result<(),()>Ok(())
 fn valid_flush(&self, e:&EventFlushRuntime<'_>)->Result<bool,()>{ Ok(e.request.sequence_id>=0 && (e.request.sequence_id as usize)<64) }
 fn valid_initialize(&self, e:&EventInitializeRuntime<'_>)->Result<bool,()>{ Ok(e.request.stop_sequences.len()<=8 && e.request.stop_sequences.iter().all(|s|!s.is_empty()&&s.len()<=32) && e.request.stop_sequences.iter().map(|s|s.len()).sum::<usize>()<=256) }
 fn valid_render(&self, e:&EventRenderRuntime<'_>)->Result<bool,()>{ Ok(e.request.token_id>=0 && e.request.sequence_id>=0 && (e.request.sequence_id as usize)<64) }
}
pub struct TextRenderer<'v,V:crate::detokenizer::VocabularyView+?Sized>{ machine: TextRendererStateMachine<'v,TextRendererContext<'v,V>> }
impl<'v,V:crate::detokenizer::VocabularyView+?Sized> TextRenderer<'v,V>{ pub fn new(v:&'v V)->Self{Self{machine:TextRendererStateMachine::new(TextRendererContext::new(v))}} pub fn state(&self)->&TextRendererStates{self.machine.state()} pub fn context(&self)->&TextRendererContext<'v,V>{self.machine.context()} }
pub type Renderer<'v,V> = TextRenderer<'v,V>;
