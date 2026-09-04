//! Bounded synchronous text renderer actor aligned with the pinned C++ renderer contract.

#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::empty_structs_with_brackets,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::needless_pass_by_ref_mut,
    clippy::derivable_impls,
    clippy::elidable_lifetime_names,
    clippy::unused_self,
    clippy::needless_lifetimes,
    private_interfaces,
    dead_code,
    unused_imports,
    missing_docs
)]

use core::cell::RefCell;
use sml::sml;

// --- machine TextRenderer from emel.cpp/src/emel/text/renderer/sm.hpp ---
#[derive(Clone, Copy, Debug)]
pub struct EventFlushRuntime<'a> {
    pub request: FlushRequest<'a>,
    pub context: &'a RefCell<FlushContext>,
}

#[derive(Clone, Copy, Debug)]
pub struct EventInitializeRuntime<'a> {
    pub request: InitializeRequest<'a>,
    pub context: &'a RefCell<InitializeContext>,
}

#[derive(Clone, Copy, Debug)]
pub struct EventRenderRuntime<'a> {
    pub request: RenderRequest<'a>,
    pub context: &'a RefCell<RenderContext>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum SequenceStatus {
    #[default]
    Running = 0,
    StopSequenceMatched = 1,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i32)]
pub enum RendererError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    Backend = 2,
    ModelInvalid = 4,
    Internal = 8,
    Untracked = 16,
}
impl RendererError {
    pub const fn code(self) -> i32 {
        self as i32
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitializeDone;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitializeError {
    pub error: RendererError,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderDone {
    pub output_length: usize,
    pub status: SequenceStatus,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderError {
    pub error: RendererError,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlushDone {
    pub output_length: usize,
    pub status: SequenceStatus,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlushError {
    pub error: RendererError,
}
pub type InitializeDoneCallback = fn(InitializeDone) -> bool;
pub type InitializeErrorCallback = fn(InitializeError) -> bool;
pub type RenderDoneCallback = fn(RenderDone) -> bool;
pub type RenderErrorCallback = fn(RenderError) -> bool;
pub type FlushDoneCallback = fn(FlushDone) -> bool;
pub type FlushErrorCallback = fn(FlushError) -> bool;
#[derive(Clone, Copy, Debug)]
pub struct InitializeRequest<'a> {
    pub strip_leading_space: bool,
    pub stop_sequences: &'a [&'a [u8]],
    pub dispatch_done: Option<InitializeDoneCallback>,
    pub dispatch_error: Option<InitializeErrorCallback>,
}
#[derive(Clone, Copy, Debug)]
pub struct RenderRequest<'a> {
    pub token_id: i32,
    pub sequence_id: i32,
    pub emit_special: bool,
    pub output: &'a RefCell<&'a mut [u8]>,
    pub output_length: &'a RefCell<usize>,
    pub status: &'a RefCell<SequenceStatus>,
    pub error: &'a RefCell<RendererError>,
    pub dispatch_done: Option<RenderDoneCallback>,
    pub dispatch_error: Option<RenderErrorCallback>,
}
#[derive(Clone, Copy, Debug)]
pub struct FlushRequest<'a> {
    pub sequence_id: i32,
    pub output: &'a RefCell<&'a mut [u8]>,
    pub output_length: &'a RefCell<usize>,
    pub status: &'a RefCell<SequenceStatus>,
    pub error: &'a RefCell<RendererError>,
    pub dispatch_done: Option<FlushDoneCallback>,
    pub dispatch_error: Option<FlushErrorCallback>,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct InitializeContext {
    pub error: RendererError,
    pub detokenizer_error: Option<crate::detokenizer::DetokenizeError>,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderContext {
    pub error: RendererError,
    pub output_length: usize,
    pub status: SequenceStatus,
    pub sequence_index: usize,
    pub detokenizer_error: Option<crate::detokenizer::DetokenizeError>,
    pub detokenizer_output_length: usize,
    pub detokenizer_pending_length: usize,
    pub produced_length: usize,
    pub leading_space_prefix_length: usize,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct FlushContext {
    pub error: RendererError,
    pub output_length: usize,
    pub status: SequenceStatus,
    pub sequence_index: usize,
}

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
        "initialize_publish_success"_s <= "initialization_decision"_s + completion<EventInitializeRuntime>(EventInitializeRuntime<'event>) [initialize_dispatch_ok] / commit_initialize_success,
        "initialize_publish_error"_s <= "initialization_decision"_s + completion<EventInitializeRuntime>(EventInitializeRuntime<'event>) [initialize_dispatch_backend_failure] / set_backend_error_event_initialize_runtime,
        "initialize_publish_error"_s <= "initialization_decision"_s + completion<EventInitializeRuntime>(EventInitializeRuntime<'event>) [initialize_dispatch_reported_error] / set_error_from_detokenizer_event_initialize_runtime,
        "initialize_publish_error"_s <= "initialization_decision"_s + completion<EventInitializeRuntime>(EventInitializeRuntime<'event>) / set_error_from_detokenizer_event_initialize_runtime,
        "initialized"_s <= "initialize_publish_success"_s + completion<EventInitializeRuntime>(EventInitializeRuntime<'event>) / publish_initialize_done,
        "errored"_s <= "initialize_publish_error"_s + completion<EventInitializeRuntime>(EventInitializeRuntime<'event>) / publish_initialize_error,
        "initialization_decision"_s <= "initializing"_s + completion<EventInitializeRuntime>(EventInitializeRuntime<'event>) / dispatch_initialize_detokenizer,
        "render_publish_success"_s <= "rendering"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [sequence_stop_matched] / render_sequence_already_stopped,
        "render_dispatch_decision"_s <= "rendering"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [sequence_running] / dispatch_render_detokenizer,
        "render_result_decision"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [render_dispatch_ok],
        "render_publish_error"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [render_dispatch_backend_failure] / set_backend_error_event_render_runtime,
        "render_publish_error"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [render_dispatch_reported_error] / set_error_from_detokenizer_event_render_runtime,
        "render_publish_error"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [render_dispatch_lengths_invalid] / set_invalid_request_event_render_runtime,
        "render_publish_error"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) / ensure_last_error_from_render_dispatch_decision,
        "render_commit_output_exec"_s <= "render_result_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>),
        "render_strip_decision"_s <= "render_commit_output_exec"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) / commit_render_detokenizer_output,
        "render_strip_prefix_scan_exec"_s <= "render_strip_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [strip_needed],
        "render_strip_state_exec"_s <= "render_strip_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [strip_not_needed],
        "render_publish_error"_s <= "render_strip_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) / ensure_last_error_from_render_strip_decision,
        "render_strip_prefix_decision"_s <= "render_strip_prefix_scan_exec"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) / compute_render_leading_space_prefix,
        "render_strip_apply_exec"_s <= "render_strip_prefix_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [strip_prefix_nonzero] / apply_render_leading_space_strip,
        "render_strip_state_exec"_s <= "render_strip_prefix_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [strip_prefix_zero],
        "render_publish_error"_s <= "render_strip_prefix_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) / ensure_last_error_from_render_strip_prefix_decision,
        "render_strip_state_exec"_s <= "render_strip_apply_exec"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>),
        "render_stop_match_exec"_s <= "render_strip_state_exec"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) / update_render_strip_state,
        "render_finalize_decision"_s <= "render_stop_match_exec"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) / apply_render_stop_matching,
        "render_publish_success"_s <= "render_finalize_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [request_ok] / mark_done,
        "render_publish_error"_s <= "render_finalize_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [request_failed] / ensure_last_error_from_render_finalize_decision,
        "render_publish_error"_s <= "render_finalize_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) / ensure_last_error_from_render_finalize_decision,
        "done"_s <= "render_publish_success"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) / publish_render_done,
        "errored"_s <= "render_publish_error"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) / publish_render_error,
        "flush_publish_success"_s <= "flushing"_s + completion<EventFlushRuntime>(EventFlushRuntime<'event>) [flush_output_fits] / flush_copy_sequence_buffers,
        "flush_publish_error"_s <= "flushing"_s + completion<EventFlushRuntime>(EventFlushRuntime<'event>) [flush_output_too_large] / set_invalid_request_event_flush_runtime,
        "done"_s <= "flush_publish_success"_s + completion<EventFlushRuntime>(EventFlushRuntime<'event>) / publish_flush_done,
        "errored"_s <= "flush_publish_error"_s + completion<EventFlushRuntime>(EventFlushRuntime<'event>) / publish_flush_error,
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

#[derive(Clone, Copy, Debug, Default)]
struct StopEntry {
    offset: usize,
    length: usize,
}
fn reset_flush_request(event: &EventFlushRuntime<'_>) {
    let mut context = event.context.borrow_mut();
    context.error = RendererError::None;
    context.output_length = 0;
    context.status = SequenceStatus::Running;
    context.sequence_index =
        usize::try_from(event.request.sequence_id).expect("validated sequence id is nonnegative");
    *event.request.output_length.borrow_mut() = 0;
    *event.request.status.borrow_mut() = SequenceStatus::Running;
    *event.request.error.borrow_mut() = RendererError::None;
}

fn reset_render_request(event: &EventRenderRuntime<'_>) {
    let mut context = event.context.borrow_mut();
    context.error = RendererError::None;
    context.output_length = 0;
    context.status = SequenceStatus::Running;
    context.sequence_index =
        usize::try_from(event.request.sequence_id).expect("validated sequence id is nonnegative");
    context.detokenizer_error = None;
    context.detokenizer_output_length = 0;
    context.detokenizer_pending_length = 0;
    context.produced_length = 0;
    context.leading_space_prefix_length = 0;
    *event.request.output_length.borrow_mut() = 0;
    *event.request.status.borrow_mut() = SequenceStatus::Running;
    *event.request.error.borrow_mut() = RendererError::None;
}

#[derive(Clone, Copy, Debug)]
struct SequenceState {
    pending: [u8; 4],
    pending_length: usize,
    holdback: [u8; 31],
    holdback_length: usize,
    strip_leading_space: bool,
    stop_matched: bool,
}
impl Default for SequenceState {
    fn default() -> Self {
        Self {
            pending: [0; 4],
            pending_length: 0,
            holdback: [0; 31],
            holdback_length: 0,
            strip_leading_space: false,
            stop_matched: false,
        }
    }
}
pub struct TextRendererContext<'v, V: crate::detokenizer::VocabularyView + ?Sized> {
    pub detokenizer: crate::detokenizer::TextDetokenizer<'v, V>,
    pub strip_default: bool,
    pub stop_sequences: [StopEntry; 8],
    pub stop_count: usize,
    pub stop_storage: [u8; 256],
    pub stop_storage_used: usize,
    pub stop_max_length: usize,
    sequences: [SequenceState; 64],
    pub bound: bool,
    pub unexpected: bool,
}
impl<'v, V: crate::detokenizer::VocabularyView + ?Sized> core::fmt::Debug
    for TextRendererContext<'v, V>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TextRendererContext")
            .field("strip_default", &self.strip_default)
            .field("stop_count", &self.stop_count)
            .field("stop_storage_used", &self.stop_storage_used)
            .field("stop_max_length", &self.stop_max_length)
            .finish_non_exhaustive()
    }
}
impl<'v, V: crate::detokenizer::VocabularyView + ?Sized> TextRendererContext<'v, V> {
    fn new(v: &'v V) -> Self {
        Self {
            detokenizer: crate::detokenizer::TextDetokenizer::new(v),
            strip_default: false,
            stop_sequences: [StopEntry::default(); 8],
            stop_count: 0,
            stop_storage: [0; 256],
            stop_storage_used: 0,
            stop_max_length: 0,
            sequences: [SequenceState::default(); 64],
            bound: false,
            unexpected: false,
        }
    }
}
impl<'v, V: crate::detokenizer::VocabularyView + ?Sized> TextRendererStateMachineContext
    for TextRendererContext<'v, V>
{
    fn apply_render_leading_space_strip(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        let mut c = e.context.borrow_mut();
        let n = c.leading_space_prefix_length;
        let produced = c.produced_length;
        if n > produced {
            c.error = RendererError::Internal;
            return Ok(());
        }
        if n != 0 {
            let mut out = e.request.output.borrow_mut();
            out.copy_within(n..produced, 0);
            c.produced_length = produced - n;
        }
        c.leading_space_prefix_length = 0;
        Ok(())
    }
    fn apply_render_stop_matching(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        let mut c = e.context.borrow_mut();
        let idx = c.sequence_index;
        let seq = &mut self.sequences[idx];
        let new_len = c.produced_length;
        let total = seq.holdback_length + new_len;
        let mut matched_start = total;
        let mut matched_len = 0usize;
        let out = e.request.output.borrow();
        for si in 0..self.stop_count {
            let stop = self.stop_sequences[si];
            if stop.length == 0 || stop.length > total {
                continue;
            }
            for cursor in 0..=(total - stop.length) {
                let mut matched = true;
                for off in 0..stop.length {
                    let pos = cursor + off;
                    let lhs = if pos < seq.holdback_length {
                        seq.holdback[pos]
                    } else {
                        out[pos - seq.holdback_length]
                    };
                    let rhs = self.stop_storage[stop.offset + off];
                    if lhs != rhs {
                        matched = false;
                        break;
                    }
                }
                if matched && cursor < matched_start {
                    matched_start = cursor;
                    matched_len = stop.length;
                }
            }
        }
        let emit_total = if matched_len > 0 {
            matched_start
        } else if self.stop_max_length > 1 {
            total.saturating_sub(self.stop_max_length - 1)
        } else {
            total
        };
        let emit_hold = core::cmp::min(emit_total, seq.holdback_length);
        let emit_new = emit_total - emit_hold;
        let hold_target = if matched_len > 0 {
            0
        } else {
            total - emit_total
        };
        let mut next = [0u8; 31];
        for n in 0..hold_target {
            let p = total - hold_target + n;
            next[n] = if p < seq.holdback_length {
                seq.holdback[p]
            } else {
                out[p - seq.holdback_length]
            };
        }
        if emit_hold + emit_new > out.len() {
            c.error = RendererError::InvalidRequest;
            return Ok(());
        }
        drop(out);
        let mut output = e.request.output.borrow_mut();
        if emit_new > 0 {
            output.copy_within(0..emit_new, emit_hold);
        }
        if emit_hold > 0 {
            output[..emit_hold].copy_from_slice(&seq.holdback[..emit_hold]);
        }
        c.output_length = emit_hold + emit_new;
        c.status = if matched_len > 0 {
            SequenceStatus::StopSequenceMatched
        } else {
            SequenceStatus::Running
        };
        seq.holdback[..hold_target].copy_from_slice(&next[..hold_target]);
        seq.holdback_length = hold_target;
        if matched_len > 0 {
            seq.stop_matched = true;
        }
        Ok(())
    }
    fn begin_flush_from_done(&mut self, e: &EventFlushRuntime<'_>) -> Result<(), ()> {
        reset_flush_request(e);
        e.context.borrow_mut().sequence_index =
            usize::try_from(e.request.sequence_id).expect("validated sequence id is nonnegative");
        Ok(())
    }
    fn begin_flush_from_errored(&mut self, e: &EventFlushRuntime<'_>) -> Result<(), ()> {
        reset_flush_request(e);
        e.context.borrow_mut().sequence_index =
            usize::try_from(e.request.sequence_id).expect("validated sequence id is nonnegative");
        Ok(())
    }
    fn begin_flush_from_initialized(&mut self, e: &EventFlushRuntime<'_>) -> Result<(), ()> {
        reset_flush_request(e);
        e.context.borrow_mut().sequence_index =
            usize::try_from(e.request.sequence_id).expect("validated sequence id is nonnegative");
        Ok(())
    }
    fn begin_flush_from_unexpected(&mut self, e: &EventFlushRuntime<'_>) -> Result<(), ()> {
        reset_flush_request(e);
        self.unexpected = false;
        e.context.borrow_mut().sequence_index =
            usize::try_from(e.request.sequence_id).expect("validated sequence id is nonnegative");
        Ok(())
    }
    fn begin_initialize_from_done(&mut self, e: &EventInitializeRuntime<'_>) -> Result<(), ()> {
        self.begin_initialize(e)
    }
    fn begin_initialize_from_errored(&mut self, e: &EventInitializeRuntime<'_>) -> Result<(), ()> {
        self.begin_initialize(e)
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_flush_publish_error(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_flush_publish_success(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_flushing(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_initialization_decision(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_initialize_publish_error(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_initialize_publish_success(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_initialized(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_initializing(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_render_commit_output_exec(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_render_dispatch_decision(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_render_finalize_decision(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_render_publish_error(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_render_publish_success(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_render_result_decision(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_render_stop_match_exec(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_render_strip_apply_exec(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_render_strip_decision(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_render_strip_prefix_decision(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_render_strip_prefix_scan_exec(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_render_strip_state_exec(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_rendering(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn on_unexpected_from_uninitialized(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
    fn mark_done(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        e.context.borrow_mut().error = RendererError::None;
        Ok(())
    }
    fn publish_flush_done(&mut self, e: &EventFlushRuntime<'_>) -> Result<(), ()> {
        let c = *e.context.borrow();
        *e.request.output_length.borrow_mut() = c.output_length;
        *e.request.status.borrow_mut() = c.status;
        *e.request.error.borrow_mut() = c.error;
        if let Some(cb) = e.request.dispatch_done {
            let _ = cb(FlushDone {
                output_length: c.output_length,
                status: c.status,
            });
        }
        Ok(())
    }
    fn publish_flush_error(&mut self, e: &EventFlushRuntime<'_>) -> Result<(), ()> {
        let c = *e.context.borrow();
        *e.request.output_length.borrow_mut() = c.output_length;
        *e.request.status.borrow_mut() = c.status;
        *e.request.error.borrow_mut() = c.error;
        if let Some(cb) = e.request.dispatch_error {
            let _ = cb(FlushError { error: c.error });
        }
        Ok(())
    }
    fn publish_initialize_done(&mut self, e: &EventInitializeRuntime<'_>) -> Result<(), ()> {
        if let Some(cb) = e.request.dispatch_done {
            let _ = cb(InitializeDone);
        }
        Ok(())
    }
    fn publish_initialize_error(&mut self, e: &EventInitializeRuntime<'_>) -> Result<(), ()> {
        let err = e.context.borrow().error;
        if let Some(cb) = e.request.dispatch_error {
            let _ = cb(InitializeError { error: err });
        }
        Ok(())
    }
    fn publish_render_done(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        let c = *e.context.borrow();
        *e.request.output_length.borrow_mut() = c.output_length;
        *e.request.status.borrow_mut() = c.status;
        *e.request.error.borrow_mut() = c.error;
        if let Some(cb) = e.request.dispatch_done {
            let _ = cb(RenderDone {
                output_length: c.output_length,
                status: c.status,
            });
        }
        Ok(())
    }
    fn publish_render_error(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        let c = *e.context.borrow();
        *e.request.output_length.borrow_mut() = c.output_length;
        *e.request.status.borrow_mut() = c.status;
        *e.request.error.borrow_mut() = c.error;
        if let Some(cb) = e.request.dispatch_error {
            let _ = cb(RenderError { error: c.error });
        }
        Ok(())
    }
    fn reject_flush_from_done(&mut self, e: &EventFlushRuntime<'_>) -> Result<(), ()> {
        reset_flush_request(e);
        e.context.borrow_mut().error = RendererError::InvalidRequest;
        Ok(())
    }
    fn reject_flush_from_errored(&mut self, e: &EventFlushRuntime<'_>) -> Result<(), ()> {
        self.reject_flush_from_done(e)
    }
    fn reject_flush_from_initialized(&mut self, e: &EventFlushRuntime<'_>) -> Result<(), ()> {
        self.reject_flush_from_done(e)
    }
    fn reject_flush_from_unexpected(&mut self, e: &EventFlushRuntime<'_>) -> Result<(), ()> {
        self.reject_flush_from_done(e)
    }
    fn reject_flush_from_uninitialized(&mut self, e: &EventFlushRuntime<'_>) -> Result<(), ()> {
        self.reject_flush_from_done(e)
    }
    fn reject_initialize_from_done(&mut self, e: &EventInitializeRuntime<'_>) -> Result<(), ()> {
        e.context.borrow_mut().error = RendererError::InvalidRequest;
        Ok(())
    }
    fn reject_initialize_from_errored(&mut self, e: &EventInitializeRuntime<'_>) -> Result<(), ()> {
        self.reject_initialize_from_done(e)
    }
    fn reject_initialize_from_initialized(
        &mut self,
        e: &EventInitializeRuntime<'_>,
    ) -> Result<(), ()> {
        self.reject_initialize_from_done(e)
    }
    fn reject_initialize_from_unexpected(
        &mut self,
        e: &EventInitializeRuntime<'_>,
    ) -> Result<(), ()> {
        self.reject_initialize_from_done(e)
    }
    fn reject_initialize_from_uninitialized(
        &mut self,
        e: &EventInitializeRuntime<'_>,
    ) -> Result<(), ()> {
        self.reject_initialize_from_done(e)
    }
    fn reject_render_from_done(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        reset_render_request(e);
        e.context.borrow_mut().error = RendererError::InvalidRequest;
        Ok(())
    }
    fn reject_render_from_errored(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        self.reject_render_from_done(e)
    }
    fn reject_render_from_initialized(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        self.reject_render_from_done(e)
    }
    fn reject_render_from_unexpected(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        self.reject_render_from_done(e)
    }
    fn reject_render_from_uninitialized(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        self.reject_render_from_done(e)
    }
    fn render_dispatch_backend_failure(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(
            e.context.borrow().detokenizer_error,
            Some(crate::detokenizer::DetokenizeError::Backend)
        ))
    }
    fn render_dispatch_lengths_invalid(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(
            c.detokenizer_output_length > e.request.output.borrow().len()
                || c.detokenizer_pending_length > 4,
        )
    }
    fn render_dispatch_ok(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(c.detokenizer_error.is_none()
            && c.detokenizer_output_length <= e.request.output.borrow().len()
            && c.detokenizer_pending_length <= 4)
    }
    fn render_dispatch_reported_error(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(
            e.context.borrow().detokenizer_error,
            Some(
                crate::detokenizer::DetokenizeError::InvalidRequest
                    | crate::detokenizer::DetokenizeError::ModelInvalid
                    | crate::detokenizer::DetokenizeError::Internal
                    | crate::detokenizer::DetokenizeError::Unexpected
            )
        ))
    }
    fn render_sequence_already_stopped(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        let mut c = e.context.borrow_mut();
        c.output_length = 0;
        c.produced_length = 0;
        c.status = SequenceStatus::StopSequenceMatched;
        c.error = RendererError::None;
        Ok(())
    }
    fn request_failed(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(e.context.borrow().error != RendererError::None)
    }
    fn request_ok(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(e.context.borrow().error == RendererError::None)
    }
    fn sequence_running(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(!self.sequences[c.sequence_index].stop_matched)
    }
    fn sequence_stop_matched(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        let c = e.context.borrow();
        Ok(self.sequences[c.sequence_index].stop_matched)
    }
    fn set_backend_error_event_initialize_runtime(
        &mut self,
        e: &EventInitializeRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().error = RendererError::Backend;
        Ok(())
    }
    fn set_backend_error_event_render_runtime(
        &mut self,
        e: &EventRenderRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().error = RendererError::Backend;
        Ok(())
    }
    fn set_error_from_detokenizer_event_initialize_runtime(
        &mut self,
        e: &EventInitializeRuntime<'_>,
    ) -> Result<(), ()> {
        let error = map_detokenizer_error(e.context.borrow().detokenizer_error);
        e.context.borrow_mut().error = error;
        Ok(())
    }
    fn set_error_from_detokenizer_event_render_runtime(
        &mut self,
        e: &EventRenderRuntime<'_>,
    ) -> Result<(), ()> {
        let error = map_detokenizer_error(e.context.borrow().detokenizer_error);
        e.context.borrow_mut().error = error;
        Ok(())
    }
    fn set_invalid_request_event_flush_runtime(
        &mut self,
        e: &EventFlushRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().error = RendererError::InvalidRequest;
        Ok(())
    }
    fn set_invalid_request_event_render_runtime(
        &mut self,
        e: &EventRenderRuntime<'_>,
    ) -> Result<(), ()> {
        e.context.borrow_mut().error = RendererError::InvalidRequest;
        Ok(())
    }
    fn strip_needed(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        let c = e.context.borrow();
        let s = &self.sequences[c.sequence_index];
        Ok(c.detokenizer_error.is_none()
            && c.detokenizer_output_length <= e.request.output.borrow().len()
            && c.detokenizer_output_length > 0
            && s.strip_leading_space
            && e.request
                .output
                .borrow()
                .first()
                .is_some_and(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r')))
    }
    fn strip_not_needed(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(self.render_dispatch_ok(e)? && !self.strip_needed(e)?)
    }
    fn strip_prefix_nonzero(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(e.context.borrow().leading_space_prefix_length != 0)
    }
    fn strip_prefix_zero(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(e.context.borrow().leading_space_prefix_length == 0)
    }
    fn update_render_strip_state(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        let c = e.context.borrow();
        let s = &mut self.sequences[c.sequence_index];
        s.strip_leading_space &= c.produced_length == 0;
        Ok(())
    }
    fn valid_flush(&self, e: &EventFlushRuntime<'_>) -> Result<bool, ()> {
        Ok(self.bound && e.request.sequence_id >= 0 && (e.request.sequence_id as usize) < 64)
    }
    fn valid_initialize(&self, e: &EventInitializeRuntime<'_>) -> Result<bool, ()> {
        Ok(e.request.stop_sequences.len() <= 8
            && e.request
                .stop_sequences
                .iter()
                .all(|s| !s.is_empty() && s.len() <= 32)
            && e.request
                .stop_sequences
                .iter()
                .map(|s| s.len())
                .sum::<usize>()
                <= 256)
    }
    fn valid_render(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(self.bound
            && e.request.token_id >= 0
            && e.request.sequence_id >= 0
            && (e.request.sequence_id as usize) < 64)
    }
    fn invalid_flush(&self, e: &EventFlushRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.valid_flush(e)?)
    }
    fn invalid_render(&self, e: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.valid_render(e)?)
    }
    fn invalid_initialize(&self, e: &EventInitializeRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.valid_initialize(e)?)
    }
    fn initialize_dispatch_ok(&self, e: &EventInitializeRuntime<'_>) -> Result<bool, ()> {
        Ok(e.context.borrow().detokenizer_error.is_none())
    }
    fn initialize_dispatch_backend_failure(
        &self,
        e: &EventInitializeRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(matches!(
            e.context.borrow().detokenizer_error,
            Some(crate::detokenizer::DetokenizeError::Backend)
        ))
    }
    fn initialize_dispatch_reported_error(
        &self,
        e: &EventInitializeRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(matches!(
            e.context.borrow().detokenizer_error,
            Some(
                crate::detokenizer::DetokenizeError::InvalidRequest
                    | crate::detokenizer::DetokenizeError::ModelInvalid
                    | crate::detokenizer::DetokenizeError::Internal
                    | crate::detokenizer::DetokenizeError::Unexpected
            )
        ))
    }
    fn flush_output_fits(&self, e: &EventFlushRuntime<'_>) -> Result<bool, ()> {
        let c = e.context.borrow();
        let s = &self.sequences[c.sequence_index];
        Ok(s.pending_length + s.holdback_length <= e.request.output.borrow().len())
    }
    fn flush_output_too_large(&self, e: &EventFlushRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.flush_output_fits(e)?)
    }
    fn ensure_last_error_from_render_dispatch_decision(
        &mut self,
        e: &EventRenderRuntime<'_>,
    ) -> Result<(), ()> {
        self.ensure_error(e)
    }
    fn ensure_last_error_from_render_strip_decision(
        &mut self,
        e: &EventRenderRuntime<'_>,
    ) -> Result<(), ()> {
        self.ensure_error(e)
    }
    fn ensure_last_error_from_render_strip_prefix_decision(
        &mut self,
        e: &EventRenderRuntime<'_>,
    ) -> Result<(), ()> {
        self.ensure_error(e)
    }
    fn ensure_last_error_from_render_finalize_decision(
        &mut self,
        e: &EventRenderRuntime<'_>,
    ) -> Result<(), ()> {
        self.ensure_error(e)
    }
    fn begin_render_from_done(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        reset_render_request(e);
        e.context.borrow_mut().sequence_index = e.request.sequence_id as usize;
        Ok(())
    }
    fn begin_render_from_errored(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        reset_render_request(e);
        e.context.borrow_mut().sequence_index = e.request.sequence_id as usize;
        Ok(())
    }
    fn begin_render_from_initialized(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        reset_render_request(e);
        e.context.borrow_mut().sequence_index = e.request.sequence_id as usize;
        Ok(())
    }
    fn begin_render_from_unexpected(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        reset_render_request(e);
        self.unexpected = false;
        e.context.borrow_mut().sequence_index = e.request.sequence_id as usize;
        Ok(())
    }
    fn commit_initialize_success(&mut self, e: &EventInitializeRuntime<'_>) -> Result<(), ()> {
        self.bound = true;
        self.strip_default = e.request.strip_leading_space;
        self.stop_count = e.request.stop_sequences.len();
        self.stop_storage_used = 0;
        self.stop_max_length = 0;
        for (i, s) in e.request.stop_sequences.iter().enumerate() {
            let off = self.stop_storage_used;
            self.stop_storage[off..off + s.len()].copy_from_slice(s);
            self.stop_sequences[i] = StopEntry {
                offset: off,
                length: s.len(),
            };
            self.stop_storage_used += s.len();
            self.stop_max_length = self.stop_max_length.max(s.len());
        }
        for seq in &mut self.sequences {
            *seq = SequenceState {
                strip_leading_space: self.strip_default,
                ..SequenceState::default()
            };
        }
        Ok(())
    }
    fn begin_initialize_from_initialized(
        &mut self,
        e: &EventInitializeRuntime<'_>,
    ) -> Result<(), ()> {
        self.begin_initialize(e)
    }
    fn begin_initialize_from_uninitialized(
        &mut self,
        e: &EventInitializeRuntime<'_>,
    ) -> Result<(), ()> {
        self.begin_initialize(e)
    }
    fn begin_initialize_from_unexpected(
        &mut self,
        e: &EventInitializeRuntime<'_>,
    ) -> Result<(), ()> {
        self.begin_initialize(e)?;
        self.unexpected = false;
        Ok(())
    }
    fn dispatch_initialize_detokenizer(
        &mut self,
        e: &EventInitializeRuntime<'_>,
    ) -> Result<(), ()> {
        let mut result = Err(crate::detokenizer::BindError::Internal);
        let bind_result = self.detokenizer.bind(crate::detokenizer::Bind {
            result: &mut result,
        });
        let error = match bind_result {
            Ok(_) => None,
            Err(crate::detokenizer::BindError::InvalidRequest) => {
                Some(crate::detokenizer::DetokenizeError::InvalidRequest)
            }
            Err(crate::detokenizer::BindError::ModelInvalid) => {
                Some(crate::detokenizer::DetokenizeError::ModelInvalid)
            }
            Err(crate::detokenizer::BindError::Backend) => {
                Some(crate::detokenizer::DetokenizeError::Backend)
            }
            Err(crate::detokenizer::BindError::Internal) => {
                Some(crate::detokenizer::DetokenizeError::Internal)
            }
        };
        e.context.borrow_mut().detokenizer_error = error;
        Ok(())
    }
    fn dispatch_render_detokenizer(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        let idx = e.context.borrow().sequence_index;
        let pending_length = self.sequences[idx].pending_length;
        let mut out = e.request.output.borrow_mut();
        let mut result = crate::detokenizer::DetokenizeResult::done(0, pending_length);
        let req = crate::detokenizer::Detokenize {
            token_id: e.request.token_id,
            emit_special: e.request.emit_special,
            pending: &mut self.sequences[idx].pending,
            pending_length,
            output: &mut out,
            result: &mut result,
        };
        let decoded = self.detokenizer.detokenize(req);
        drop(out);
        let mut c = e.context.borrow_mut();
        c.detokenizer_error = decoded.error_kind();
        c.detokenizer_output_length = decoded.output_length();
        c.detokenizer_pending_length = decoded.pending_length();
        Ok(())
    }
    fn commit_render_detokenizer_output(&mut self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        let mut c = e.context.borrow_mut();
        let idx = c.sequence_index;
        let pending = c.detokenizer_pending_length;
        let produced = c.detokenizer_output_length;
        if pending > 4 {
            c.error = RendererError::InvalidRequest;
            return Ok(());
        }
        self.sequences[idx].pending_length = pending;
        c.produced_length = produced;
        c.leading_space_prefix_length = 0;
        Ok(())
    }
    fn compute_render_leading_space_prefix(
        &mut self,
        e: &EventRenderRuntime<'_>,
    ) -> Result<(), ()> {
        let mut c = e.context.borrow_mut();
        let produced = c.produced_length;
        if produced > e.request.output.borrow().len() {
            c.error = RendererError::InvalidRequest;
            c.leading_space_prefix_length = 0;
            return Ok(());
        }
        let out = e.request.output.borrow();
        c.leading_space_prefix_length = out[..produced]
            .iter()
            .take_while(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
            .count();
        Ok(())
    }
    fn flush_copy_sequence_buffers(&mut self, e: &EventFlushRuntime<'_>) -> Result<(), ()> {
        let mut c = e.context.borrow_mut();
        let s = &mut self.sequences[c.sequence_index];
        let n = s.pending_length + s.holdback_length;
        if n > e.request.output.borrow().len() {
            c.error = RendererError::InvalidRequest;
            return Ok(());
        }
        let mut out = e.request.output.borrow_mut();
        out[..s.pending_length].copy_from_slice(&s.pending[..s.pending_length]);
        out[s.pending_length..n].copy_from_slice(&s.holdback[..s.holdback_length]);
        c.output_length = n;
        c.status = if s.stop_matched {
            SequenceStatus::StopSequenceMatched
        } else {
            SequenceStatus::Running
        };
        s.pending_length = 0;
        s.holdback_length = 0;
        s.strip_leading_space &= n == 0;
        Ok(())
    }
}
impl<'v, V: crate::detokenizer::VocabularyView + ?Sized> TextRendererContext<'v, V> {
    fn begin_initialize(&self, e: &EventInitializeRuntime<'_>) -> Result<(), ()> {
        let mut c = e.context.borrow_mut();
        c.error = RendererError::None;
        c.detokenizer_error = None;
        Ok(())
    }
    fn ensure_error(&self, e: &EventRenderRuntime<'_>) -> Result<(), ()> {
        let mut c = e.context.borrow_mut();
        if c.error == RendererError::None {
            c.error = RendererError::Backend;
        }
        Ok(())
    }
}
const fn map_detokenizer_error(e: Option<crate::detokenizer::DetokenizeError>) -> RendererError {
    match e {
        None | Some(crate::detokenizer::DetokenizeError::Backend) => RendererError::Backend,
        Some(crate::detokenizer::DetokenizeError::InvalidRequest) => RendererError::InvalidRequest,
        Some(crate::detokenizer::DetokenizeError::ModelInvalid) => RendererError::ModelInvalid,
        Some(crate::detokenizer::DetokenizeError::Internal) => RendererError::Internal,
        Some(crate::detokenizer::DetokenizeError::Unexpected) => RendererError::Untracked,
    }
}
pub struct TextRenderer<'v, V: crate::detokenizer::VocabularyView + ?Sized> {
    machine: TextRendererStateMachine<TextRendererContext<'v, V>>,
}
impl<'v, V: crate::detokenizer::VocabularyView + ?Sized> TextRenderer<'v, V> {
    pub fn new(v: &'v V) -> Self {
        Self {
            machine: TextRendererStateMachine::new(TextRendererContext::new(v)),
        }
    }
    pub fn process_initialize(
        &mut self,
        request: InitializeRequest<'_>,
        context: &RefCell<InitializeContext>,
    ) -> Result<(), RendererError> {
        let event = EventInitializeRuntime { request, context };
        let accepted = self
            .machine
            .process_event(TextRendererEvents::EventInitializeRuntime(event))
            .is_ok();
        let err = if accepted {
            context.borrow().error
        } else {
            RendererError::Backend
        };
        if err == RendererError::None {
            Ok(())
        } else {
            Err(err)
        }
    }
    pub fn process_render<'e>(
        &mut self,
        request: RenderRequest<'e>,
        context: &'e RefCell<RenderContext>,
    ) -> Result<(), RendererError> {
        let event = EventRenderRuntime { request, context };
        let accepted = self
            .machine
            .process_event(TextRendererEvents::EventRenderRuntime(event))
            .is_ok();
        let err = if accepted {
            context.borrow().error
        } else {
            RendererError::Backend
        };
        if err == RendererError::None {
            Ok(())
        } else {
            Err(err)
        }
    }
    pub fn process_flush<'e>(
        &mut self,
        request: FlushRequest<'e>,
        context: &'e RefCell<FlushContext>,
    ) -> Result<(), RendererError> {
        let event = EventFlushRuntime { request, context };
        let accepted = self
            .machine
            .process_event(TextRendererEvents::EventFlushRuntime(event))
            .is_ok();
        let err = if accepted {
            context.borrow().error
        } else {
            RendererError::Backend
        };
        if err == RendererError::None {
            Ok(())
        } else {
            Err(err)
        }
    }
    pub fn process_unexpected(&mut self) {
        self.machine.context_mut().unexpected = true;
        self.machine.set_state(TextRendererStates::Unexpected);
    }
    pub fn state(&self) -> &TextRendererStates {
        self.machine.state()
    }
    pub fn context(&self) -> &TextRendererContext<'v, V> {
        self.machine.context()
    }
}
pub type Renderer<'v, V> = TextRenderer<'v, V>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_backend_and_unexpected_without_collapsing_classes() {
        assert_eq!(
            map_detokenizer_error(Some(crate::detokenizer::DetokenizeError::Backend)),
            RendererError::Backend
        );
        assert_eq!(
            map_detokenizer_error(Some(crate::detokenizer::DetokenizeError::Unexpected)),
            RendererError::Untracked
        );
    }
}
