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

// --- machine TextTokenizer from emel.cpp/src/emel/text/tokenizer/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventBindRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventTokenizeRuntime;

sml! {
    TextTokenizer {
        "binding_preprocessor"_s <= *"uninitialized"_s + event<EventBindRuntime> [can_bind] / begin_bind_from_uninitialized,
        "errored"_s <= "uninitialized"_s + event<EventBindRuntime> / reject_bind_from_uninitialized,
        "errored"_s <= "uninitialized"_s + event<EventTokenizeRuntime> / reject_invalid_from_uninitialized,
        "binding_preprocessor"_s <= "idle"_s + event<EventBindRuntime> [can_bind] / begin_bind_from_idle,
        "errored"_s <= "idle"_s + event<EventBindRuntime> / reject_bind_from_idle,
        "preprocessing"_s <= "idle"_s + event<EventTokenizeRuntime> [can_tokenize] / begin_tokenize_from_idle,
        "errored"_s <= "idle"_s + event<EventTokenizeRuntime> / reject_invalid_from_idle,
        "binding_preprocessor"_s <= "done"_s + event<EventBindRuntime> [can_bind] / begin_bind_from_done,
        "errored"_s <= "done"_s + event<EventBindRuntime> / reject_bind_from_done,
        "preprocessing"_s <= "done"_s + event<EventTokenizeRuntime> [can_tokenize] / begin_tokenize_from_done,
        "errored"_s <= "done"_s + event<EventTokenizeRuntime> / reject_invalid_from_done,
        "binding_preprocessor"_s <= "errored"_s + event<EventBindRuntime> [can_bind] / begin_bind_from_errored,
        "errored"_s <= "errored"_s + event<EventBindRuntime> / reject_bind_from_errored,
        "preprocessing"_s <= "errored"_s + event<EventTokenizeRuntime> [can_tokenize] / begin_tokenize_from_errored,
        "errored"_s <= "errored"_s + event<EventTokenizeRuntime> / reject_invalid_from_errored,
        "binding_preprocessor"_s <= "unexpected"_s + event<EventBindRuntime> [can_bind] / begin_bind_from_unexpected,
        "unexpected"_s <= "unexpected"_s + event<EventBindRuntime> / reject_bind_from_unexpected,
        "preprocessing"_s <= "unexpected"_s + event<EventTokenizeRuntime> [can_tokenize] / begin_tokenize_from_unexpected,
        "unexpected"_s <= "unexpected"_s + event<EventTokenizeRuntime> / reject_invalid_from_unexpected,
        "binding_preprocessor_decision"_s <= "binding_preprocessor"_s + completion<EventBindRuntime> / bind_preprocessor,
        "binding_encoder"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime> [bind_preprocessor_error_none],
        "errored"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime> [bind_preprocessor_error_invalid_request],
        "errored"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime> [bind_preprocessor_error_model_invalid],
        "errored"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime> [bind_preprocessor_error_backend_error],
        "errored"_s <= "binding_preprocessor_decision"_s + completion<EventBindRuntime> [bind_preprocessor_error_unknown],
        "binding_encoder_decision"_s <= "binding_encoder"_s + completion<EventBindRuntime> / bind_encoder,
        "idle"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime> [bind_encoder_error_none] / mark_bind_success,
        "errored"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime> [bind_encoder_error_invalid_request],
        "errored"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime> [bind_encoder_error_model_invalid],
        "errored"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime> [bind_encoder_error_backend_error],
        "errored"_s <= "binding_encoder_decision"_s + completion<EventBindRuntime> [bind_encoder_error_unknown],
        "preprocess_decision"_s <= "preprocessing"_s + completion<EventTokenizeRuntime> / dispatch_preprocess,
        "errored"_s <= "preprocess_decision"_s + completion<EventTokenizeRuntime> [preprocess_rejected_no_error] / set_backend_error,
        "errored"_s <= "preprocess_decision"_s + completion<EventTokenizeRuntime> [preprocess_reported_error] / set_error_from_preprocess,
        "errored"_s <= "preprocess_decision"_s + completion<EventTokenizeRuntime> [preprocess_fragment_count_invalid] / set_invalid_request_error_from_preprocess_decision,
        "prefix_decision"_s <= "preprocess_decision"_s + completion<EventTokenizeRuntime> [preprocess_success],
        "encoding_ready"_s <= "prefix_decision"_s + completion<EventTokenizeRuntime> [bos_ready] / append_bos,
        "errored"_s <= "prefix_decision"_s + completion<EventTokenizeRuntime> [bos_no_capacity] / set_invalid_request_error_from_prefix_decision,
        "errored"_s <= "prefix_decision"_s + completion<EventTokenizeRuntime> [bos_invalid_id] / set_invalid_id_error_from_prefix_decision,
        "encoding_ready"_s <= "prefix_decision"_s + completion<EventTokenizeRuntime> [no_prefix],
        "suffix_decision"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime> [no_more_fragments],
        "errored"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime> [more_fragments_no_capacity] / set_invalid_request_error_from_encoding_ready,
        "errored"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime> [more_fragments_token_invalid] / set_invalid_request_error_from_encoding_ready,
        "encoding_token_fragment"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime> [more_fragments_token_valid],
        "encoding_raw_fragment"_s <= "encoding_ready"_s + completion<EventTokenizeRuntime> [more_fragments_raw],
        "encoding_ready"_s <= "encoding_token_fragment"_s + completion<EventTokenizeRuntime> / append_fragment_token,
        "encoding_raw_decision"_s <= "encoding_raw_fragment"_s + completion<EventTokenizeRuntime> / dispatch_encode_raw_fragment,
        "errored"_s <= "encoding_raw_decision"_s + completion<EventTokenizeRuntime> [encode_rejected_no_error] / set_invalid_id_error_from_encoding_raw_decision,
        "errored"_s <= "encoding_raw_decision"_s + completion<EventTokenizeRuntime> [encode_reported_error] / set_error_from_encode,
        "errored"_s <= "encoding_raw_decision"_s + completion<EventTokenizeRuntime> [encode_count_invalid] / set_invalid_request_error_from_encoding_raw_decision,
        "encoding_ready"_s <= "encoding_raw_decision"_s + completion<EventTokenizeRuntime> [encode_success] / commit_encoded_fragment,
        "finalizing"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime> [sep_ready] / append_sep,
        "errored"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime> [sep_no_capacity] / set_invalid_request_error_from_suffix_decision,
        "errored"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime> [sep_invalid_id] / set_invalid_id_error_from_suffix_decision,
        "finalizing"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime> [eos_ready] / append_eos,
        "errored"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime> [eos_no_capacity] / set_invalid_request_error_from_suffix_decision,
        "errored"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime> [eos_invalid_id] / set_invalid_id_error_from_suffix_decision,
        "finalizing"_s <= "suffix_decision"_s + completion<EventTokenizeRuntime> [no_suffix],
        "done"_s <= "finalizing"_s + completion<EventTokenizeRuntime> / finalize,
        "unexpected"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "unexpected"_s <= "binding_preprocessor"_s + unexpected_event<_> / on_unexpected_from_binding_preprocessor,
        "unexpected"_s <= "binding_preprocessor_decision"_s + unexpected_event<_> / on_unexpected_from_binding_preprocessor_decision,
        "unexpected"_s <= "binding_encoder"_s + unexpected_event<_> / on_unexpected_from_binding_encoder,
        "unexpected"_s <= "binding_encoder_decision"_s + unexpected_event<_> / on_unexpected_from_binding_encoder_decision,
        "unexpected"_s <= "idle"_s + unexpected_event<_> / on_unexpected_from_idle,
        "unexpected"_s <= "preprocessing"_s + unexpected_event<_> / on_unexpected_from_preprocessing,
        "unexpected"_s <= "preprocess_decision"_s + unexpected_event<_> / on_unexpected_from_preprocess_decision,
        "unexpected"_s <= "prefix_decision"_s + unexpected_event<_> / on_unexpected_from_prefix_decision,
        "unexpected"_s <= "encoding_ready"_s + unexpected_event<_> / on_unexpected_from_encoding_ready,
        "unexpected"_s <= "encoding_token_fragment"_s + unexpected_event<_> / on_unexpected_from_encoding_token_fragment,
        "unexpected"_s <= "encoding_raw_fragment"_s + unexpected_event<_> / on_unexpected_from_encoding_raw_fragment,
        "unexpected"_s <= "encoding_raw_decision"_s + unexpected_event<_> / on_unexpected_from_encoding_raw_decision,
        "unexpected"_s <= "suffix_decision"_s + unexpected_event<_> / on_unexpected_from_suffix_decision,
        "unexpected"_s <= "finalizing"_s + unexpected_event<_> / on_unexpected_from_finalizing,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

/// Context for `TextTokenizer` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextTokenizerContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextTokenizerStateMachineContext for TextTokenizerContext {
    fn append_bos(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::append_bos
        todo!("TODO: port action `append_bos` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn append_eos(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::append_eos
        todo!("TODO: port action `append_eos` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn append_fragment_token(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::append_fragment_token
        todo!(
            "TODO: port action `append_fragment_token` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn append_sep(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::append_sep
        todo!("TODO: port action `append_sep` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn begin_bind_from_done(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn begin_bind_from_errored(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn begin_bind_from_idle(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn begin_bind_from_unexpected(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn begin_bind_from_uninitialized(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn begin_tokenize_from_done(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::begin_tokenize
        todo!(
            "TODO: port action `begin_tokenize` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn begin_tokenize_from_errored(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::begin_tokenize
        todo!(
            "TODO: port action `begin_tokenize` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn begin_tokenize_from_idle(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::begin_tokenize
        todo!(
            "TODO: port action `begin_tokenize` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn begin_tokenize_from_unexpected(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::begin_tokenize
        todo!(
            "TODO: port action `begin_tokenize` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn bind_encoder(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::bind_encoder
        todo!("TODO: port action `bind_encoder` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn bind_encoder_error_backend_error(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::bind_encoder_error_backend_error
        todo!(
            "TODO: port guard `bind_encoder_error_backend_error` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn bind_encoder_error_invalid_request(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::bind_encoder_error_invalid_request
        todo!(
            "TODO: port guard `bind_encoder_error_invalid_request` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn bind_encoder_error_model_invalid(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::bind_encoder_error_model_invalid
        todo!(
            "TODO: port guard `bind_encoder_error_model_invalid` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn bind_encoder_error_none(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::bind_encoder_error_none
        todo!(
            "TODO: port guard `bind_encoder_error_none` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn bind_encoder_error_unknown(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::bind_encoder_error_unknown
        todo!(
            "TODO: port guard `bind_encoder_error_unknown` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn bind_preprocessor(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::bind_preprocessor
        todo!(
            "TODO: port action `bind_preprocessor` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn bind_preprocessor_error_backend_error(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::bind_preprocessor_error_backend_error
        todo!(
            "TODO: port guard `bind_preprocessor_error_backend_error` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn bind_preprocessor_error_invalid_request(
        &self,
        _event: &EventBindRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::bind_preprocessor_error_invalid_request
        todo!(
            "TODO: port guard `bind_preprocessor_error_invalid_request` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn bind_preprocessor_error_model_invalid(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::bind_preprocessor_error_model_invalid
        todo!(
            "TODO: port guard `bind_preprocessor_error_model_invalid` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn bind_preprocessor_error_none(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::bind_preprocessor_error_none
        todo!(
            "TODO: port guard `bind_preprocessor_error_none` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn bind_preprocessor_error_unknown(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::bind_preprocessor_error_unknown
        todo!(
            "TODO: port guard `bind_preprocessor_error_unknown` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn bos_invalid_id(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::bos_invalid_id
        todo!("TODO: port guard `bos_invalid_id` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn bos_no_capacity(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::bos_no_capacity
        todo!("TODO: port guard `bos_no_capacity` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn bos_ready(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::bos_ready
        todo!("TODO: port guard `bos_ready` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn can_bind(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::can_bind
        todo!("TODO: port guard `can_bind` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn can_tokenize(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::can_tokenize
        todo!("TODO: port guard `can_tokenize` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn commit_encoded_fragment(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::commit_encoded_fragment
        todo!(
            "TODO: port action `commit_encoded_fragment` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn dispatch_encode_raw_fragment(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::dispatch_encode_raw_fragment
        todo!(
            "TODO: port action `dispatch_encode_raw_fragment` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn dispatch_preprocess(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::dispatch_preprocess
        todo!(
            "TODO: port action `dispatch_preprocess` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn encode_count_invalid(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::encode_count_invalid
        todo!(
            "TODO: port guard `encode_count_invalid` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn encode_rejected_no_error(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::encode_rejected_no_error
        todo!(
            "TODO: port guard `encode_rejected_no_error` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn encode_reported_error(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::encode_reported_error
        todo!(
            "TODO: port guard `encode_reported_error` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn encode_success(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::encode_success
        todo!("TODO: port guard `encode_success` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn eos_invalid_id(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::eos_invalid_id
        todo!("TODO: port guard `eos_invalid_id` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn eos_no_capacity(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::eos_no_capacity
        todo!("TODO: port guard `eos_no_capacity` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn eos_ready(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::eos_ready
        todo!("TODO: port guard `eos_ready` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn finalize(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::finalize
        todo!("TODO: port action `finalize` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn mark_bind_success(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::mark_bind_success
        todo!(
            "TODO: port action `mark_bind_success` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn more_fragments_no_capacity(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::more_fragments_no_capacity
        todo!(
            "TODO: port guard `more_fragments_no_capacity` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn more_fragments_raw(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::more_fragments_raw
        todo!(
            "TODO: port guard `more_fragments_raw` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn more_fragments_token_invalid(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::more_fragments_token_invalid
        todo!(
            "TODO: port guard `more_fragments_token_invalid` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn more_fragments_token_valid(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::more_fragments_token_valid
        todo!(
            "TODO: port guard `more_fragments_token_valid` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn no_more_fragments(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::no_more_fragments
        todo!(
            "TODO: port guard `no_more_fragments` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn no_prefix(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::no_prefix
        todo!("TODO: port guard `no_prefix` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn no_suffix(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::no_suffix
        todo!("TODO: port guard `no_suffix` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn on_unexpected_from_binding_encoder(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_binding_encoder_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_binding_preprocessor(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_binding_preprocessor_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_encoding_raw_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_encoding_raw_fragment(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_encoding_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_encoding_token_fragment(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_finalizing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_prefix_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_preprocess_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_preprocessing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_suffix_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn on_unexpected_from_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn preprocess_fragment_count_invalid(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::preprocess_fragment_count_invalid
        todo!(
            "TODO: port guard `preprocess_fragment_count_invalid` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn preprocess_rejected_no_error(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::preprocess_rejected_no_error
        todo!(
            "TODO: port guard `preprocess_rejected_no_error` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn preprocess_reported_error(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::preprocess_reported_error
        todo!(
            "TODO: port guard `preprocess_reported_error` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn preprocess_success(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::preprocess_success
        todo!(
            "TODO: port guard `preprocess_success` from emel.cpp/src/emel/text/tokenizer/guards.hpp"
        )
    }
    fn reject_bind_from_done(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::reject_bind
        todo!("TODO: port action `reject_bind` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn reject_bind_from_errored(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::reject_bind
        todo!("TODO: port action `reject_bind` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn reject_bind_from_idle(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::reject_bind
        todo!("TODO: port action `reject_bind` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn reject_bind_from_unexpected(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::reject_bind
        todo!("TODO: port action `reject_bind` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn reject_bind_from_uninitialized(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::reject_bind
        todo!("TODO: port action `reject_bind` from emel.cpp/src/emel/text/tokenizer/actions.hpp")
    }
    fn reject_invalid_from_done(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::reject_invalid
        todo!(
            "TODO: port action `reject_invalid` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn reject_invalid_from_errored(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::reject_invalid
        todo!(
            "TODO: port action `reject_invalid` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn reject_invalid_from_idle(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::reject_invalid
        todo!(
            "TODO: port action `reject_invalid` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn reject_invalid_from_unexpected(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::reject_invalid
        todo!(
            "TODO: port action `reject_invalid` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn reject_invalid_from_uninitialized(
        &mut self,
        _event: &EventTokenizeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::reject_invalid
        todo!(
            "TODO: port action `reject_invalid` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn sep_invalid_id(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::sep_invalid_id
        todo!("TODO: port guard `sep_invalid_id` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn sep_no_capacity(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::sep_no_capacity
        todo!("TODO: port guard `sep_no_capacity` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn sep_ready(&self, _event: &EventTokenizeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/guards.hpp::sep_ready
        todo!("TODO: port guard `sep_ready` from emel.cpp/src/emel/text/tokenizer/guards.hpp")
    }
    fn set_backend_error(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::set_backend_error
        todo!(
            "TODO: port action `set_backend_error` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn set_error_from_encode(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::set_error_from_encode
        todo!(
            "TODO: port action `set_error_from_encode` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn set_error_from_preprocess(&mut self, _event: &EventTokenizeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::set_error_from_preprocess
        todo!(
            "TODO: port action `set_error_from_preprocess` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn set_invalid_id_error_from_encoding_raw_decision(
        &mut self,
        _event: &EventTokenizeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::set_invalid_id_error
        todo!(
            "TODO: port action `set_invalid_id_error` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn set_invalid_id_error_from_prefix_decision(
        &mut self,
        _event: &EventTokenizeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::set_invalid_id_error
        todo!(
            "TODO: port action `set_invalid_id_error` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn set_invalid_id_error_from_suffix_decision(
        &mut self,
        _event: &EventTokenizeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::set_invalid_id_error
        todo!(
            "TODO: port action `set_invalid_id_error` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn set_invalid_request_error_from_encoding_raw_decision(
        &mut self,
        _event: &EventTokenizeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::set_invalid_request_error
        todo!(
            "TODO: port action `set_invalid_request_error` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn set_invalid_request_error_from_encoding_ready(
        &mut self,
        _event: &EventTokenizeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::set_invalid_request_error
        todo!(
            "TODO: port action `set_invalid_request_error` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn set_invalid_request_error_from_prefix_decision(
        &mut self,
        _event: &EventTokenizeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::set_invalid_request_error
        todo!(
            "TODO: port action `set_invalid_request_error` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn set_invalid_request_error_from_preprocess_decision(
        &mut self,
        _event: &EventTokenizeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::set_invalid_request_error
        todo!(
            "TODO: port action `set_invalid_request_error` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
    fn set_invalid_request_error_from_suffix_decision(
        &mut self,
        _event: &EventTokenizeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/actions.hpp::set_invalid_request_error
        todo!(
            "TODO: port action `set_invalid_request_error` from emel.cpp/src/emel/text/tokenizer/actions.hpp"
        )
    }
}
