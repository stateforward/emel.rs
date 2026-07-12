//! State machine scaffold port — not a stable public API.
//! Bodies are stubs (`todo!`) until contexts/guards/actions are ported from C++.

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

// --- machine TextDetokenizer from emel.cpp/src/emel/text/detokenizer/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventBind;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventDetokenize;

sml! {
    TextDetokenizer {
        "binding"_s <= *"uninitialized"_s + event<EventBind> [valid_bind] / begin_bind_from_uninitialized,
        "binding_error_decision"_s <= "uninitialized"_s + event<EventBind> [invalid_bind] / reject_bind_event_bind,
        "detokenize_error_decision"_s <= "uninitialized"_s + event<EventDetokenize> / reject_detokenize_event_detokenize,
        "binding"_s <= "idle"_s + event<EventBind> [valid_bind] / begin_bind_from_idle,
        "binding_error_decision"_s <= "idle"_s + event<EventBind> [invalid_bind] / reject_bind_event_bind,
        "decoding"_s <= "idle"_s + event<EventDetokenize> [valid_detokenize] / begin_detokenize_from_idle,
        "detokenize_error_decision"_s <= "idle"_s + event<EventDetokenize> [invalid_detokenize] / reject_detokenize_event_detokenize,
        "binding"_s <= "done"_s + event<EventBind> [valid_bind] / begin_bind_from_done,
        "binding_error_decision"_s <= "done"_s + event<EventBind> [invalid_bind] / reject_bind_event_bind,
        "decoding"_s <= "done"_s + event<EventDetokenize> [valid_detokenize] / begin_detokenize_from_done,
        "detokenize_error_decision"_s <= "done"_s + event<EventDetokenize> [invalid_detokenize] / reject_detokenize_event_detokenize,
        "binding"_s <= "errored"_s + event<EventBind> [valid_bind] / begin_bind_from_errored,
        "binding_error_decision"_s <= "errored"_s + event<EventBind> [invalid_bind] / reject_bind_event_bind,
        "decoding"_s <= "errored"_s + event<EventDetokenize> [valid_detokenize] / begin_detokenize_from_errored,
        "detokenize_error_decision"_s <= "errored"_s + event<EventDetokenize> [invalid_detokenize] / reject_detokenize_event_detokenize,
        "binding"_s <= "unexpected"_s + event<EventBind> [valid_bind] / begin_bind_from_unexpected,
        "binding_error_decision"_s <= "unexpected"_s + event<EventBind> [invalid_bind] / reject_bind_event_bind,
        "decoding"_s <= "unexpected"_s + event<EventDetokenize> [valid_detokenize] / begin_detokenize_from_unexpected,
        "detokenize_error_decision"_s <= "unexpected"_s + event<EventDetokenize> [invalid_detokenize] / reject_detokenize_event_detokenize,
        "binding_error_decision"_s <= "binding"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "binding"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "binding_decision"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "binding_decision"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "binding_done_decision"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "binding_done_decision"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "binding_done_callback"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "binding_done_callback"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "binding_error_decision"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "binding_error_decision"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "binding_error_callback"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "binding_error_callback"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "decoding"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "decoding"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "decode_token_validation"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "decode_token_validation"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "decode_piece_decision"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "decode_piece_decision"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "decode_byte_capacity_decision"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "decode_byte_capacity_decision"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "decode_byte_pending_decision"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "decode_byte_pending_decision"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "decode_byte_pending_write"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "decode_byte_pending_write"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "decode_text_pending_decision"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "decode_text_pending_decision"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "decode_text_pending_write"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "decode_text_pending_write"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "decode_text_write"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "decode_text_write"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "decode_decision"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "decode_decision"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "detokenize_done_decision"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "detokenize_done_decision"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "detokenize_done_callback"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "detokenize_done_callback"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "detokenize_error_decision"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "detokenize_error_decision"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_error_decision"_s <= "detokenize_error_callback"_s + unexpected_event<EventBind> / reject_bind_unexp_event_bind,
        "detokenize_error_decision"_s <= "detokenize_error_callback"_s + unexpected_event<EventDetokenize> / reject_detokenize_unexp_event_detokenize,
        "binding_decision"_s <= "binding"_s + completion<EventBind> / commit_bind,
        "binding_done_decision"_s <= "binding_decision"_s + completion<EventBind> [bind_error_none],
        "binding_error_decision"_s <= "binding_decision"_s + completion<EventBind> [bind_error_invalid_request],
        "binding_error_decision"_s <= "binding_decision"_s + completion<EventBind> [bind_error_model_invalid],
        "binding_error_decision"_s <= "binding_decision"_s + completion<EventBind> [bind_error_backend_error],
        "binding_error_decision"_s <= "binding_decision"_s + completion<EventBind> [bind_error_internal_error],
        "binding_error_decision"_s <= "binding_decision"_s + completion<EventBind> [bind_error_untracked],
        "binding_error_decision"_s <= "binding_decision"_s + completion<EventBind> [bind_error_unknown],
        "binding_done_callback"_s <= "binding_done_decision"_s + completion<EventBind> [has_bind_done_callback] / notify_bind_done,
        "idle"_s <= "binding_done_decision"_s + completion<EventBind> [no_bind_done_callback],
        "idle"_s <= "binding_done_callback"_s + completion<EventBind>,
        "binding_error_callback"_s <= "binding_error_decision"_s + completion<EventBind> [has_bind_error_callback] / notify_bind_error,
        "errored"_s <= "binding_error_decision"_s + completion<EventBind> [no_bind_error_callback],
        "errored"_s <= "binding_error_callback"_s + completion<EventBind>,
        "decode_token_validation"_s <= "decoding"_s + completion<EventDetokenize>,
        "decode_piece_decision"_s <= "decode_token_validation"_s + completion<EventDetokenize> [detokenize_token_in_vocab],
        "detokenize_error_decision"_s <= "decode_token_validation"_s + completion<EventDetokenize> [detokenize_token_out_of_vocab] / mark_model_invalid,
        "detokenize_done_decision"_s <= "decode_piece_decision"_s + completion<EventDetokenize> [detokenize_skip_special_piece] / mark_done_from_decode_piece_decision,
        "decode_byte_capacity_decision"_s <= "decode_piece_decision"_s + completion<EventDetokenize> [detokenize_byte_piece],
        "decode_text_pending_decision"_s <= "decode_piece_decision"_s + completion<EventDetokenize> [detokenize_text_piece],
        "detokenize_error_decision"_s <= "decode_piece_decision"_s + completion<EventDetokenize> / mark_internal_error,
        "decode_byte_pending_decision"_s <= "decode_byte_capacity_decision"_s + completion<EventDetokenize> [detokenize_pending_has_capacity_for_byte] / append_byte_piece,
        "detokenize_error_decision"_s <= "decode_byte_capacity_decision"_s + completion<EventDetokenize> [detokenize_pending_no_capacity_for_byte] / mark_invalid_pending_full,
        "detokenize_error_decision"_s <= "decode_byte_pending_decision"_s + completion<EventDetokenize> [detokenize_error_invalid_request],
        "detokenize_error_decision"_s <= "decode_byte_pending_decision"_s + completion<EventDetokenize> [detokenize_error_model_invalid],
        "detokenize_error_decision"_s <= "decode_byte_pending_decision"_s + completion<EventDetokenize> [detokenize_error_backend_error],
        "detokenize_error_decision"_s <= "decode_byte_pending_decision"_s + completion<EventDetokenize> [detokenize_error_internal_error],
        "detokenize_error_decision"_s <= "decode_byte_pending_decision"_s + completion<EventDetokenize> [detokenize_error_untracked],
        "detokenize_error_decision"_s <= "decode_byte_pending_decision"_s + completion<EventDetokenize> [detokenize_error_unknown],
        "decode_byte_pending_write"_s <= "decode_byte_pending_decision"_s + completion<EventDetokenize> [detokenize_pending_head_complete] / write_pending_head_sequence_from_decode_byte_pending_decision,
        "decode_decision"_s <= "decode_byte_pending_decision"_s + completion<EventDetokenize> [detokenize_pending_empty],
        "decode_decision"_s <= "decode_byte_pending_decision"_s + completion<EventDetokenize> [detokenize_pending_head_incomplete],
        "detokenize_error_decision"_s <= "decode_byte_pending_decision"_s + completion<EventDetokenize> [detokenize_pending_head_invalid] / mark_invalid_pending_sequence_from_decode_byte_pending_decision,
        "decode_byte_pending_decision"_s <= "decode_byte_pending_write"_s + completion<EventDetokenize>,
        "detokenize_error_decision"_s <= "decode_text_pending_decision"_s + completion<EventDetokenize> [detokenize_error_invalid_request],
        "detokenize_error_decision"_s <= "decode_text_pending_decision"_s + completion<EventDetokenize> [detokenize_error_model_invalid],
        "detokenize_error_decision"_s <= "decode_text_pending_decision"_s + completion<EventDetokenize> [detokenize_error_backend_error],
        "detokenize_error_decision"_s <= "decode_text_pending_decision"_s + completion<EventDetokenize> [detokenize_error_internal_error],
        "detokenize_error_decision"_s <= "decode_text_pending_decision"_s + completion<EventDetokenize> [detokenize_error_untracked],
        "detokenize_error_decision"_s <= "decode_text_pending_decision"_s + completion<EventDetokenize> [detokenize_error_unknown],
        "decode_text_pending_write"_s <= "decode_text_pending_decision"_s + completion<EventDetokenize> [detokenize_pending_head_complete] / write_pending_head_sequence_from_decode_text_pending_decision,
        "decode_text_write"_s <= "decode_text_pending_decision"_s + completion<EventDetokenize> [detokenize_pending_empty] / write_text_piece,
        "detokenize_error_decision"_s <= "decode_text_pending_decision"_s + completion<EventDetokenize> [detokenize_pending_head_incomplete] / mark_invalid_pending_not_empty,
        "detokenize_error_decision"_s <= "decode_text_pending_decision"_s + completion<EventDetokenize> [detokenize_pending_head_invalid] / mark_invalid_pending_sequence_from_decode_text_pending_decision,
        "decode_text_pending_decision"_s <= "decode_text_pending_write"_s + completion<EventDetokenize>,
        "decode_decision"_s <= "decode_text_write"_s + completion<EventDetokenize>,
        "detokenize_done_decision"_s <= "decode_decision"_s + completion<EventDetokenize> [detokenize_error_none] / mark_done_from_decode_decision,
        "detokenize_error_decision"_s <= "decode_decision"_s + completion<EventDetokenize> [detokenize_error_invalid_request],
        "detokenize_error_decision"_s <= "decode_decision"_s + completion<EventDetokenize> [detokenize_error_model_invalid],
        "detokenize_error_decision"_s <= "decode_decision"_s + completion<EventDetokenize> [detokenize_error_backend_error],
        "detokenize_error_decision"_s <= "decode_decision"_s + completion<EventDetokenize> [detokenize_error_internal_error],
        "detokenize_error_decision"_s <= "decode_decision"_s + completion<EventDetokenize> [detokenize_error_untracked],
        "detokenize_error_decision"_s <= "decode_decision"_s + completion<EventDetokenize> [detokenize_error_unknown],
        "detokenize_done_callback"_s <= "detokenize_done_decision"_s + completion<EventDetokenize> [has_detokenize_done_callback],
        "done"_s <= "detokenize_done_decision"_s + completion<EventDetokenize> [no_detokenize_done_callback],
        "done"_s <= "detokenize_done_callback"_s + completion<EventDetokenize> / notify_detokenize_done,
        "detokenize_error_callback"_s <= "detokenize_error_decision"_s + completion<EventDetokenize> [has_detokenize_error_callback] / notify_detokenize_error,
        "errored"_s <= "detokenize_error_decision"_s + completion<EventDetokenize> [no_detokenize_error_callback],
        "errored"_s <= "detokenize_error_callback"_s + completion<EventDetokenize>,
        "unexpected"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "unexpected"_s <= "binding"_s + unexpected_event<_> / on_unexpected_from_binding,
        "unexpected"_s <= "binding_decision"_s + unexpected_event<_> / on_unexpected_from_binding_decision,
        "unexpected"_s <= "binding_done_decision"_s + unexpected_event<_> / on_unexpected_from_binding_done_decision,
        "unexpected"_s <= "binding_done_callback"_s + unexpected_event<_> / on_unexpected_from_binding_done_callback,
        "unexpected"_s <= "binding_error_decision"_s + unexpected_event<_> / on_unexpected_from_binding_error_decision,
        "unexpected"_s <= "binding_error_callback"_s + unexpected_event<_> / on_unexpected_from_binding_error_callback,
        "unexpected"_s <= "idle"_s + unexpected_event<_> / on_unexpected_from_idle,
        "unexpected"_s <= "decoding"_s + unexpected_event<_> / on_unexpected_from_decoding,
        "unexpected"_s <= "decode_token_validation"_s + unexpected_event<_> / on_unexpected_from_decode_token_validation,
        "unexpected"_s <= "decode_piece_decision"_s + unexpected_event<_> / on_unexpected_from_decode_piece_decision,
        "unexpected"_s <= "decode_byte_capacity_decision"_s + unexpected_event<_> / on_unexpected_from_decode_byte_capacity_decision,
        "unexpected"_s <= "decode_byte_pending_decision"_s + unexpected_event<_> / on_unexpected_from_decode_byte_pending_decision,
        "unexpected"_s <= "decode_byte_pending_write"_s + unexpected_event<_> / on_unexpected_from_decode_byte_pending_write,
        "unexpected"_s <= "decode_text_pending_decision"_s + unexpected_event<_> / on_unexpected_from_decode_text_pending_decision,
        "unexpected"_s <= "decode_text_pending_write"_s + unexpected_event<_> / on_unexpected_from_decode_text_pending_write,
        "unexpected"_s <= "decode_text_write"_s + unexpected_event<_> / on_unexpected_from_decode_text_write,
        "unexpected"_s <= "decode_decision"_s + unexpected_event<_> / on_unexpected_from_decode_decision,
        "unexpected"_s <= "detokenize_done_decision"_s + unexpected_event<_> / on_unexpected_from_detokenize_done_decision,
        "unexpected"_s <= "detokenize_done_callback"_s + unexpected_event<_> / on_unexpected_from_detokenize_done_callback,
        "unexpected"_s <= "detokenize_error_decision"_s + unexpected_event<_> / on_unexpected_from_detokenize_error_decision,
        "unexpected"_s <= "detokenize_error_callback"_s + unexpected_event<_> / on_unexpected_from_detokenize_error_callback,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

/// Context for `TextDetokenizer` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextDetokenizerContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextDetokenizerStateMachineContext for TextDetokenizerContext {
    fn append_byte_piece(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::append_byte_piece
        todo!(
            "TODO: port action `append_byte_piece` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn begin_bind_from_done(&mut self, _event: &EventBind) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/detokenizer/actions.hpp")
    }
    fn begin_bind_from_errored(&mut self, _event: &EventBind) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/detokenizer/actions.hpp")
    }
    fn begin_bind_from_idle(&mut self, _event: &EventBind) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/detokenizer/actions.hpp")
    }
    fn begin_bind_from_unexpected(&mut self, _event: &EventBind) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/detokenizer/actions.hpp")
    }
    fn begin_bind_from_uninitialized(&mut self, _event: &EventBind) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/detokenizer/actions.hpp")
    }
    fn begin_detokenize_from_done(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::begin_detokenize
        todo!(
            "TODO: port action `begin_detokenize` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn begin_detokenize_from_errored(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::begin_detokenize
        todo!(
            "TODO: port action `begin_detokenize` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn begin_detokenize_from_idle(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::begin_detokenize
        todo!(
            "TODO: port action `begin_detokenize` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn begin_detokenize_from_unexpected(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::begin_detokenize
        todo!(
            "TODO: port action `begin_detokenize` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn bind_error_backend_error(&self, _event: &EventBind) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::bind_error_backend_error
        todo!(
            "TODO: port guard `bind_error_backend_error` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn bind_error_internal_error(&self, _event: &EventBind) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::bind_error_internal_error
        todo!(
            "TODO: port guard `bind_error_internal_error` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn bind_error_invalid_request(&self, _event: &EventBind) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::bind_error_invalid_request
        todo!(
            "TODO: port guard `bind_error_invalid_request` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn bind_error_model_invalid(&self, _event: &EventBind) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::bind_error_model_invalid
        todo!(
            "TODO: port guard `bind_error_model_invalid` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn bind_error_none(&self, _event: &EventBind) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::bind_error_none
        todo!(
            "TODO: port guard `bind_error_none` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn bind_error_unknown(&self, _event: &EventBind) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::bind_error_unknown
        todo!(
            "TODO: port guard `bind_error_unknown` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn bind_error_untracked(&self, _event: &EventBind) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::bind_error_untracked
        todo!(
            "TODO: port guard `bind_error_untracked` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn commit_bind(&mut self, _event: &EventBind) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::commit_bind
        todo!("TODO: port action `commit_bind` from emel.cpp/src/emel/text/detokenizer/actions.hpp")
    }
    fn detokenize_byte_piece(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_byte_piece
        todo!(
            "TODO: port guard `detokenize_byte_piece` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_error_backend_error(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_error_backend_error
        todo!(
            "TODO: port guard `detokenize_error_backend_error` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_error_internal_error(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_error_internal_error
        todo!(
            "TODO: port guard `detokenize_error_internal_error` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_error_invalid_request(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_error_invalid_request
        todo!(
            "TODO: port guard `detokenize_error_invalid_request` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_error_model_invalid(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_error_model_invalid
        todo!(
            "TODO: port guard `detokenize_error_model_invalid` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_error_none(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_error_none
        todo!(
            "TODO: port guard `detokenize_error_none` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_error_unknown(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_error_unknown
        todo!(
            "TODO: port guard `detokenize_error_unknown` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_error_untracked(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_error_untracked
        todo!(
            "TODO: port guard `detokenize_error_untracked` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_pending_empty(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_pending_empty
        todo!(
            "TODO: port guard `detokenize_pending_empty` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_pending_has_capacity_for_byte(
        &self,
        _event: &EventDetokenize,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_pending_has_capacity_for_byte
        todo!(
            "TODO: port guard `detokenize_pending_has_capacity_for_byte` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_pending_head_complete(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_pending_head_complete
        todo!(
            "TODO: port guard `detokenize_pending_head_complete` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_pending_head_incomplete(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_pending_head_incomplete
        todo!(
            "TODO: port guard `detokenize_pending_head_incomplete` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_pending_head_invalid(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_pending_head_invalid
        todo!(
            "TODO: port guard `detokenize_pending_head_invalid` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_pending_no_capacity_for_byte(
        &self,
        _event: &EventDetokenize,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_pending_no_capacity_for_byte
        todo!(
            "TODO: port guard `detokenize_pending_no_capacity_for_byte` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_skip_special_piece(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_skip_special_piece
        todo!(
            "TODO: port guard `detokenize_skip_special_piece` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_text_piece(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_text_piece
        todo!(
            "TODO: port guard `detokenize_text_piece` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_token_in_vocab(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_token_in_vocab
        todo!(
            "TODO: port guard `detokenize_token_in_vocab` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn detokenize_token_out_of_vocab(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::detokenize_token_out_of_vocab
        todo!(
            "TODO: port guard `detokenize_token_out_of_vocab` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn has_bind_done_callback(&self, _event: &EventBind) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::has_bind_done_callback
        todo!(
            "TODO: port guard `has_bind_done_callback` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn has_bind_error_callback(&self, _event: &EventBind) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::has_bind_error_callback
        todo!(
            "TODO: port guard `has_bind_error_callback` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn has_detokenize_done_callback(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::has_detokenize_done_callback
        todo!(
            "TODO: port guard `has_detokenize_done_callback` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn has_detokenize_error_callback(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::has_detokenize_error_callback
        todo!(
            "TODO: port guard `has_detokenize_error_callback` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn invalid_bind(&self, _event: &EventBind) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::invalid_bind
        todo!("TODO: port guard `invalid_bind` from emel.cpp/src/emel/text/detokenizer/guards.hpp")
    }
    fn invalid_detokenize(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::invalid_detokenize
        todo!(
            "TODO: port guard `invalid_detokenize` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn mark_done_from_decode_decision(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::mark_done
        todo!("TODO: port action `mark_done` from emel.cpp/src/emel/text/detokenizer/actions.hpp")
    }
    fn mark_done_from_decode_piece_decision(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::mark_done
        todo!("TODO: port action `mark_done` from emel.cpp/src/emel/text/detokenizer/actions.hpp")
    }
    fn mark_internal_error(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn mark_invalid_pending_full(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::mark_invalid_pending_full
        todo!(
            "TODO: port action `mark_invalid_pending_full` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn mark_invalid_pending_not_empty(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::mark_invalid_pending_not_empty
        todo!(
            "TODO: port action `mark_invalid_pending_not_empty` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn mark_invalid_pending_sequence_from_decode_byte_pending_decision(
        &mut self,
        _event: &EventDetokenize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::mark_invalid_pending_sequence
        todo!(
            "TODO: port action `mark_invalid_pending_sequence` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn mark_invalid_pending_sequence_from_decode_text_pending_decision(
        &mut self,
        _event: &EventDetokenize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::mark_invalid_pending_sequence
        todo!(
            "TODO: port action `mark_invalid_pending_sequence` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn mark_model_invalid(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::mark_model_invalid
        todo!(
            "TODO: port action `mark_model_invalid` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn no_bind_done_callback(&self, _event: &EventBind) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::no_bind_done_callback
        todo!(
            "TODO: port guard `no_bind_done_callback` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn no_bind_error_callback(&self, _event: &EventBind) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::no_bind_error_callback
        todo!(
            "TODO: port guard `no_bind_error_callback` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn no_detokenize_done_callback(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::no_detokenize_done_callback
        todo!(
            "TODO: port guard `no_detokenize_done_callback` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn no_detokenize_error_callback(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::no_detokenize_error_callback
        todo!(
            "TODO: port guard `no_detokenize_error_callback` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn notify_bind_done(&mut self, _event: &EventBind) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::notify_bind_done
        todo!(
            "TODO: port action `notify_bind_done` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn notify_bind_error(&mut self, _event: &EventBind) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::notify_bind_error
        todo!(
            "TODO: port action `notify_bind_error` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn notify_detokenize_done(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::notify_detokenize_done
        todo!(
            "TODO: port action `notify_detokenize_done` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn notify_detokenize_error(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::notify_detokenize_error
        todo!(
            "TODO: port action `notify_detokenize_error` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_binding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_binding_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_binding_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_binding_done_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_binding_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_binding_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_decode_byte_capacity_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_decode_byte_pending_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_decode_byte_pending_write(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_decode_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_decode_piece_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_decode_text_pending_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_decode_text_pending_write(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_decode_text_write(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_decode_token_validation(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_decoding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_detokenize_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_detokenize_done_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_detokenize_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_detokenize_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn on_unexpected_from_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn reject_bind_event_bind(&mut self, _event: &EventBind) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::reject_bind
        todo!("TODO: port action `reject_bind` from emel.cpp/src/emel/text/detokenizer/actions.hpp")
    }
    fn reject_bind_unexp_event_bind(&mut self, _event: &EventBind) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::reject_bind
        todo!("TODO: port action `reject_bind` from emel.cpp/src/emel/text/detokenizer/actions.hpp")
    }
    fn reject_detokenize_event_detokenize(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::reject_detokenize
        todo!(
            "TODO: port action `reject_detokenize` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn reject_detokenize_unexp_event_detokenize(
        &mut self,
        _event: &EventDetokenize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::reject_detokenize
        todo!(
            "TODO: port action `reject_detokenize` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn valid_bind(&self, _event: &EventBind) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::valid_bind
        todo!("TODO: port guard `valid_bind` from emel.cpp/src/emel/text/detokenizer/guards.hpp")
    }
    fn valid_detokenize(&self, _event: &EventDetokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/guards.hpp::valid_detokenize
        todo!(
            "TODO: port guard `valid_detokenize` from emel.cpp/src/emel/text/detokenizer/guards.hpp"
        )
    }
    fn write_pending_head_sequence_from_decode_byte_pending_decision(
        &mut self,
        _event: &EventDetokenize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::write_pending_head_sequence
        todo!(
            "TODO: port action `write_pending_head_sequence` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn write_pending_head_sequence_from_decode_text_pending_decision(
        &mut self,
        _event: &EventDetokenize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::write_pending_head_sequence
        todo!(
            "TODO: port action `write_pending_head_sequence` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
    fn write_text_piece(&mut self, _event: &EventDetokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/detokenizer/actions.hpp::write_text_piece
        todo!(
            "TODO: port action `write_text_piece` from emel.cpp/src/emel/text/detokenizer/actions.hpp"
        )
    }
}
