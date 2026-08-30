//! Source-aligned bounded TextConditioner state machine.
#![allow(clippy::derive_partial_eq_without_eq, clippy::module_name_repetitions, clippy::missing_errors_doc, clippy::must_use_candidate, clippy::return_self_not_must_use, clippy::empty_structs_with_brackets, clippy::missing_const_for_fn, dead_code, unused_imports, missing_docs)]
use core::cell::{Cell, RefCell};
use sml::sml;

/// Runtime bind request and its run-to-completion fields.
pub struct EventBindRuntime {
    pub tokenizer_available: bool, pub formatter_available: bool, pub model_valid: bool,
    pub formatter: Option<fn(crate::FormatRequest<'_>) -> Result<(), crate::ConditionerError>>,
    pub tokenizer: Option<fn(&[u8], bool, bool, &mut [i32]) -> Result<usize, crate::ConditionerError>>,
    pub tokenizer_bind: Option<fn(bool, &mut i32) -> bool>, pub add_special: bool, pub parse_special: bool,
    pub bind_accepted: Cell<bool>, pub bind_err_code: Cell<i32>, pub err: Cell<crate::ConditionerError>, pub result: Cell<bool>,
    pub error_out: Option<&'static mut i32>, pub done_callback: Option<fn(crate::BindingDone) -> bool>, pub error_callback: Option<fn(crate::ConditionerError) -> bool>,
}
impl Default for EventBindRuntime { fn default()->Self { Self { tokenizer_available:false, formatter_available:false, model_valid:false, formatter:None, tokenizer:None, tokenizer_bind:None, add_special:true, parse_special:false, bind_accepted:Cell::new(false), bind_err_code:Cell::new(0), err:Cell::new(crate::ConditionerError::None), result:Cell::new(false), error_out:None, done_callback:None, error_callback:None } } }
impl core::fmt::Debug for EventBindRuntime { fn fmt(&self,f:&mut core::fmt::Formatter<'_>)->core::fmt::Result { f.debug_struct("EventBindRuntime").field("tokenizer_available",&self.tokenizer_available).field("formatter_available",&self.formatter_available).field("model_valid",&self.model_valid).finish() } }
impl EventBindRuntime { fn reset(&self) { self.bind_accepted.set(false); self.bind_err_code.set(0); self.err.set(crate::ConditionerError::None); self.result.set(false); } }

/// Runtime prepare request and its run-to-completion fields.
pub struct EventPrepareRuntime<'a> {
    pub messages: &'a [crate::ChatMessage<'a>], pub formatter_available: bool, pub tokenizer_available: bool, pub model_valid: bool, pub token_capacity: usize, pub token_ids: RefCell<&'a mut [i32]>, pub token_ids_present: bool, pub token_count_out: &'a mut usize, pub error_out: Option<&'a mut i32>,
    pub add_generation_prompt: bool, pub enable_thinking: bool, pub add_special_request: bool, pub parse_special_request: bool, pub use_bind_defaults: bool, pub add_special: Cell<bool>, pub parse_special: Cell<bool>, pub formatted_length: Cell<usize>, pub formatted_capacity: usize, pub format_accepted: Cell<bool>, pub format_err_code: Cell<i32>, pub tokenize_accepted: Cell<bool>, pub tokenize_err_code: Cell<i32>, pub token_count: Cell<usize>, pub err: Cell<crate::ConditionerError>, pub result: Cell<bool>, pub done_callback: Option<fn(crate::ConditioningDone) -> bool>, pub error_callback: Option<fn(crate::ConditionerError) -> bool>,
}
impl<'a> core::fmt::Debug for EventPrepareRuntime<'a> { fn fmt(&self,f:&mut core::fmt::Formatter<'_>)->core::fmt::Result { f.debug_struct("EventPrepareRuntime").field("token_capacity",&self.token_capacity).field("messages",&self.messages.len()).finish() } }
impl<'a> EventPrepareRuntime<'a> { fn reset(&self) { self.formatted_length.set(0); self.format_accepted.set(false); self.format_err_code.set(0); self.tokenize_accepted.set(false); self.tokenize_err_code.set(0); self.token_count.set(0); self.err.set(crate::ConditionerError::None); self.result.set(false); } }

sml! {
    TextConditioner<'event> {
        "binding"_s <= *"uninitialized"_s + event<EventBindRuntime> [valid_bind] / begin_bind_from_uninitialized,
        "bind_error"_s <= "uninitialized"_s + event<EventBindRuntime> [invalid_bind] / reject_bind_from_uninitialized,
        "prepare_error"_s <= "uninitialized"_s + event<EventPrepareRuntime<'event>> / reject_prepare_from_uninitialized,
        "binding"_s <= "idle"_s + event<EventBindRuntime> [valid_bind] / begin_bind_from_idle,
        "bind_error"_s <= "idle"_s + event<EventBindRuntime> [invalid_bind] / reject_bind_from_idle,
        "preparing"_s <= "idle"_s + event<EventPrepareRuntime<'event>> [valid_prepare_with_bind_defaults] / begin_prepare_bind_defaults_from_idle,
        "preparing"_s <= "idle"_s + event<EventPrepareRuntime<'event>> [valid_prepare_with_request_overrides] / begin_prepare_from_request_from_idle,
        "prepare_error"_s <= "idle"_s + event<EventPrepareRuntime<'event>> [invalid_prepare] / reject_prepare_from_idle,
        "binding"_s <= "done"_s + event<EventBindRuntime> [valid_bind] / begin_bind_from_done,
        "bind_error"_s <= "done"_s + event<EventBindRuntime> [invalid_bind] / reject_bind_from_done,
        "preparing"_s <= "done"_s + event<EventPrepareRuntime<'event>> [valid_prepare_with_bind_defaults] / begin_prepare_bind_defaults_from_done,
        "preparing"_s <= "done"_s + event<EventPrepareRuntime<'event>> [valid_prepare_with_request_overrides] / begin_prepare_from_request_from_done,
        "prepare_error"_s <= "done"_s + event<EventPrepareRuntime<'event>> [invalid_prepare] / reject_prepare_from_done,
        "binding"_s <= "errored"_s + event<EventBindRuntime> [valid_bind] / begin_bind_from_errored,
        "bind_error"_s <= "errored"_s + event<EventBindRuntime> [invalid_bind] / reject_bind_from_errored,
        "preparing"_s <= "errored"_s + event<EventPrepareRuntime<'event>> [valid_prepare_with_bind_defaults] / begin_prepare_bind_defaults_from_errored,
        "preparing"_s <= "errored"_s + event<EventPrepareRuntime<'event>> [valid_prepare_with_request_overrides] / begin_prepare_from_request_from_errored,
        "prepare_error"_s <= "errored"_s + event<EventPrepareRuntime<'event>> [invalid_prepare] / reject_prepare_from_errored,
        "binding"_s <= "unexpected"_s + event<EventBindRuntime> [valid_bind] / begin_bind_from_unexpected,
        "bind_error"_s <= "unexpected"_s + event<EventBindRuntime> [invalid_bind] / reject_bind_from_unexpected,
        "preparing"_s <= "unexpected"_s + event<EventPrepareRuntime<'event>> [valid_prepare_with_bind_defaults] / begin_prepare_bind_defaults_from_unexpected,
        "preparing"_s <= "unexpected"_s + event<EventPrepareRuntime<'event>> [valid_prepare_with_request_overrides] / begin_prepare_from_request_from_unexpected,
        "prepare_error"_s <= "unexpected"_s + event<EventPrepareRuntime<'event>> [invalid_prepare] / reject_prepare_from_unexpected,
        "bind_decision"_s <= "binding"_s + completion<EventBindRuntime> / dispatch_bind_tokenizer,
        "bind_error"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_rejected_no_error] / bind_error_backend,
        "bind_error"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_error_invalid_argument_code] / set_error_invalid_argument_event_bind_runtime,
        "bind_error"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_error_model_invalid_code] / set_error_model_invalid_event_bind_runtime,
        "bind_error"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_error_capacity_code] / set_error_capacity_event_bind_runtime,
        "bind_error"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_error_backend_code] / set_error_backend_event_bind_runtime,
        "bind_error"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_error_untracked_code] / set_error_untracked_event_bind_runtime,
        "bind_success"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_successful] / bind_success,
        "bind_publish_success"_s <= "bind_success"_s + completion<EventBindRuntime> [has_bind_error_out] / write_bind_error_out_from_bind_success,
        "bind_publish_success"_s <= "bind_success"_s + completion<EventBindRuntime> [no_bind_error_out],
        "idle"_s <= "bind_publish_success"_s + completion<EventBindRuntime> [has_bind_done_callback] / emit_bind_done,
        "idle"_s <= "bind_publish_success"_s + completion<EventBindRuntime> [no_bind_done_callback],
        "bind_publish_error"_s <= "bind_error"_s + completion<EventBindRuntime> [has_bind_error_out] / write_bind_error_out_from_bind_error,
        "bind_publish_error"_s <= "bind_error"_s + completion<EventBindRuntime> [no_bind_error_out],
        "errored"_s <= "bind_publish_error"_s + completion<EventBindRuntime> [has_bind_error_callback] / emit_bind_error,
        "errored"_s <= "bind_publish_error"_s + completion<EventBindRuntime> [no_bind_error_callback],
        "format_decision"_s <= "preparing"_s + completion<EventPrepareRuntime<'event>> / dispatch_format,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime<'event>> [format_rejected_no_error] / format_error_backend,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime<'event>> [format_error_invalid_argument_code] / set_error_invalid_argument_event_prepare_runtime,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime<'event>> [format_error_model_invalid_code] / set_error_model_invalid_event_prepare_runtime,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime<'event>> [format_error_capacity_code] / set_error_capacity_event_prepare_runtime,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime<'event>> [format_error_backend_code] / set_error_backend_event_prepare_runtime,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime<'event>> [format_error_untracked_code] / set_error_untracked_event_prepare_runtime,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime<'event>> [format_length_overflow] / format_error_invalid_argument,
        "tokenizing"_s <= "format_decision"_s + completion<EventPrepareRuntime<'event>> [format_successful],
        "tokenize_decision"_s <= "tokenizing"_s + completion<EventPrepareRuntime<'event>> / dispatch_tokenize,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime<'event>> [tokenize_rejected_no_error] / tokenize_error_backend_from_tokenize_decision,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime<'event>> [tokenize_error_invalid_argument_code] / set_error_invalid_argument_event_prepare_runtime,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime<'event>> [tokenize_error_model_invalid_code] / set_error_model_invalid_event_prepare_runtime,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime<'event>> [tokenize_error_capacity_code] / set_error_capacity_event_prepare_runtime,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime<'event>> [tokenize_error_backend_code] / set_error_backend_event_prepare_runtime,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime<'event>> [tokenize_error_untracked_code] / set_error_untracked_event_prepare_runtime,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime<'event>> [tokenize_count_invalid] / tokenize_error_backend_from_tokenize_decision,
        "prepare_success"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime<'event>> [tokenize_successful] / prepare_success,
        "prepare_publish_success_count"_s <= "prepare_success"_s + completion<EventPrepareRuntime<'event>> / write_prepare_token_count_from_prepare_success,
        "prepare_publish_success_error"_s <= "prepare_publish_success_count"_s + completion<EventPrepareRuntime<'event>> / write_prepare_error_out_from_prepare_publish_success_count,
        "done"_s <= "prepare_publish_success_error"_s + completion<EventPrepareRuntime<'event>> [has_prepare_done_callback] / emit_prepare_done,
        "done"_s <= "prepare_publish_success_error"_s + completion<EventPrepareRuntime<'event>> [no_prepare_done_callback],
        "prepare_publish_error_count"_s <= "prepare_error"_s + completion<EventPrepareRuntime<'event>> / write_prepare_token_count_from_prepare_error,
        "prepare_publish_error"_s <= "prepare_publish_error_count"_s + completion<EventPrepareRuntime<'event>> / write_prepare_error_out_from_prepare_publish_error_count,
        "errored"_s <= "prepare_publish_error"_s + completion<EventPrepareRuntime<'event>> [has_prepare_error_callback] / emit_prepare_error,
        "errored"_s <= "prepare_publish_error"_s + completion<EventPrepareRuntime<'event>> [no_prepare_error_callback],
        "unexpected"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "unexpected"_s <= "binding"_s + unexpected_event<_> / on_unexpected_from_binding,
        "unexpected"_s <= "bind_decision"_s + unexpected_event<_> / on_unexpected_from_bind_decision,
        "unexpected"_s <= "bind_success"_s + unexpected_event<_> / on_unexpected_from_bind_success,
        "unexpected"_s <= "bind_error"_s + unexpected_event<_> / on_unexpected_from_bind_error,
        "unexpected"_s <= "bind_publish_success"_s + unexpected_event<_> / on_unexpected_from_bind_publish_success,
        "unexpected"_s <= "bind_publish_error"_s + unexpected_event<_> / on_unexpected_from_bind_publish_error,
        "unexpected"_s <= "preparing"_s + unexpected_event<_> / on_unexpected_from_preparing,
        "unexpected"_s <= "format_decision"_s + unexpected_event<_> / on_unexpected_from_format_decision,
        "unexpected"_s <= "tokenizing"_s + unexpected_event<_> / on_unexpected_from_tokenizing,
        "unexpected"_s <= "tokenize_decision"_s + unexpected_event<_> / on_unexpected_from_tokenize_decision,
        "unexpected"_s <= "prepare_success"_s + unexpected_event<_> / on_unexpected_from_prepare_success,
        "unexpected"_s <= "prepare_error"_s + unexpected_event<_> / on_unexpected_from_prepare_error,
        "unexpected"_s <= "prepare_publish_success_count"_s + unexpected_event<_> / on_unexpected_from_prepare_publish_success_count,
        "unexpected"_s <= "prepare_publish_success_error"_s + unexpected_event<_> / on_unexpected_from_prepare_publish_success_error,
        "unexpected"_s <= "prepare_publish_error_count"_s + unexpected_event<_> / on_unexpected_from_prepare_publish_error_count,
        "unexpected"_s <= "prepare_publish_error"_s + unexpected_event<_> / on_unexpected_from_prepare_publish_error,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "idle"_s + unexpected_event<_> / on_unexpected_from_idle,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

/// Bounded context retained by the generated machine.
#[derive(Debug)]
pub struct TextConditionerContext { pub bound: bool, pub formatted: [u8; 32768], pub formatter: Option<fn(crate::FormatRequest<'_>) -> Result<(), crate::ConditionerError>>, pub tokenizer: Option<fn(&[u8], bool, bool, &mut [i32]) -> Result<usize, crate::ConditionerError>>, pub tokenizer_bind: Option<fn(bool, &mut i32) -> bool>, pub add_special_default: bool, pub parse_special_default: bool, pub error: crate::ConditionerError, pub result: bool, pub token_count: usize }
impl Default for TextConditionerContext { fn default()->Self { Self { bound:false, formatted:[0;32768], formatter:None, tokenizer:None, tokenizer_bind:None, add_special_default:true, parse_special_default:false, error:crate::ConditionerError::None, result:false, token_count:0 } } }

impl<'event> TextConditionerStateMachineContext for TextConditionerContext {
    fn begin_bind_from_done(&mut self, event: &EventBindRuntime) -> Result<(), ()> { self.begin_bind(event) }
    fn begin_bind_from_errored(&mut self, event: &EventBindRuntime) -> Result<(), ()> { self.begin_bind(event) }
    fn begin_bind_from_idle(&mut self, event: &EventBindRuntime) -> Result<(), ()> { self.begin_bind(event) }
    fn begin_bind_from_unexpected(&mut self, event: &EventBindRuntime) -> Result<(), ()> { self.begin_bind(event) }
    fn begin_bind_from_uninitialized(&mut self, event: &EventBindRuntime) -> Result<(), ()> { self.begin_bind(event) }
    fn begin_prepare_bind_defaults_from_done(&mut self, event: &EventPrepareRuntime<'event>) -> Result<(), ()> { self.begin_prepare_defaults(event) }
    fn begin_prepare_bind_defaults_from_errored(&mut self, event: &EventPrepareRuntime<'event>) -> Result<(), ()> { self.begin_prepare_defaults(event) }
    fn begin_prepare_bind_defaults_from_idle(&mut self, event: &EventPrepareRuntime<'event>) -> Result<(), ()> { self.begin_prepare_defaults(event) }
    fn begin_prepare_bind_defaults_from_unexpected(&mut self, event: &EventPrepareRuntime<'event>) -> Result<(), ()> { self.begin_prepare_defaults(event) }
    fn begin_prepare_from_request_from_done(&mut self, event: &EventPrepareRuntime<'event>) -> Result<(), ()> { self.begin_prepare_request(event) }
    fn begin_prepare_from_request_from_errored(&mut self, event: &EventPrepareRuntime<'event>) -> Result<(), ()> { self.begin_prepare_request(event) }
    fn begin_prepare_from_request_from_idle(&mut self, event: &EventPrepareRuntime<'event>) -> Result<(), ()> { self.begin_prepare_request(event) }
    fn begin_prepare_from_request_from_unexpected(&mut self, event: &EventPrepareRuntime<'event>) -> Result<(), ()> { self.begin_prepare_request(event) }
    fn bind_error_backend(&mut self, event: &EventBindRuntime) -> Result<(), ()> { self.set_bind_error(event, crate::ConditionerError::Backend); Ok(()) }
    fn bind_error_backend_code(&self, event: &EventBindRuntime) -> Result<bool, ()> { Ok(event.bind_err_code.get() == 8) }
    fn bind_error_capacity_code(&self, event: &EventBindRuntime) -> Result<bool, ()> { Ok(event.bind_err_code.get() == 4) }
    fn bind_error_invalid_argument_code(&self, event: &EventBindRuntime) -> Result<bool, ()> { Ok(event.bind_err_code.get() == 1) }
    fn bind_error_model_invalid_code(&self, event: &EventBindRuntime) -> Result<bool, ()> { Ok(event.bind_err_code.get() == 2) }
    fn bind_error_untracked_code(&self, event: &EventBindRuntime) -> Result<bool, ()> { Ok(event.bind_err_code.get() != 0 && !matches!(event.bind_err_code.get(), 1|2|4|8)) }
    fn bind_rejected_no_error(&self, event: &EventBindRuntime) -> Result<bool, ()> { Ok(!event.bind_accepted.get() && event.bind_err_code.get() == 0) }
    fn bind_success(&mut self, event: &EventBindRuntime) -> Result<(), ()> { self.bound=true; self.error=crate::ConditionerError::None; self.result=true; event.err.set(crate::ConditionerError::None); event.result.set(true); Ok(()) }
    fn bind_successful(&self, event: &EventBindRuntime) -> Result<bool, ()> { Ok(event.bind_accepted.get() && event.bind_err_code.get() == 0) }
    fn dispatch_bind_tokenizer(&mut self, event: &EventBindRuntime) -> Result<(), ()> { let mut code=0; let accepted=self.tokenizer_bind.map(|f| f(event.tokenizer_available,&mut code)).unwrap_or(event.tokenizer_available); event.bind_accepted.set(accepted); event.bind_err_code.set(code); Ok(()) }
    fn dispatch_format(&mut self, event: &EventPrepareRuntime<'event>) -> Result<(), ()> { let mut length=0; let result=self.formatter.map(|f| f(crate::FormatRequest{messages:event.messages,add_generation_prompt:event.add_generation_prompt,enable_thinking:event.enable_thinking,output:&mut self.formatted,output_length:&mut length})).unwrap_or(Err(crate::ConditionerError::Backend)); event.formatted_length.set(length); match result { Ok(())=>{event.format_accepted.set(true);event.format_err_code.set(0)},Err(e)=>{event.format_err_code.set(e.code())} }; Ok(()) }
    fn dispatch_tokenize(&mut self, event: &EventPrepareRuntime<'event>) -> Result<(), ()> { let length=event.formatted_length.get().min(self.formatted.len()); let mut ids=event.token_ids.borrow_mut(); let result=self.tokenizer.map(|f| f(&self.formatted[..length],event.add_special.get(),event.parse_special.get(),&mut ids[..event.token_capacity.min(ids.len())])).unwrap_or(Err(crate::ConditionerError::Backend)); match result { Ok(n)=>{event.tokenize_accepted.set(true);event.token_count.set(n);self.token_count=n},Err(e)=>{event.tokenize_err_code.set(e.code());event.token_count.set(0);self.token_count=0} }; Ok(()) }
    fn emit_bind_done(&mut self,event:&EventBindRuntime)->Result<(), ()>{if let Some(f)=event.done_callback{let _=f(crate::BindingDone);} Ok(())}
    fn emit_bind_error(&mut self,event:&EventBindRuntime)->Result<(), ()>{if let Some(f)=event.error_callback{let _=f(event.err.get());} Ok(())}
    fn emit_prepare_done(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{if let Some(f)=event.done_callback{let _=f(crate::ConditioningDone{token_count:event.token_count.get()});} Ok(())}
    fn emit_prepare_error(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{if let Some(f)=event.error_callback{let _=f(event.err.get());} Ok(())}
    fn format_error_backend(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.set_prepare_error(event,crate::ConditionerError::Backend);Ok(())}
    fn format_error_backend_code(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.format_err_code.get()==8)}
    fn format_error_capacity_code(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.format_err_code.get()==4)}
    fn format_error_invalid_argument(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.set_prepare_error(event,crate::ConditionerError::InvalidArgument);Ok(())}
    fn format_error_invalid_argument_code(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.format_err_code.get()==1)}
    fn format_error_model_invalid_code(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.format_err_code.get()==2)}
    fn format_error_untracked_code(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.format_err_code.get()!=0&&!matches!(event.format_err_code.get(),1|2|4|8))}
    fn format_length_overflow(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.formatted_length.get()>event.formatted_capacity)}
    fn format_rejected_no_error(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(!event.format_accepted.get()&&event.format_err_code.get()==0)}
    fn format_successful(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.format_accepted.get()&&event.format_err_code.get()==0&&event.formatted_length.get()<=event.formatted_capacity)}
    fn has_bind_done_callback(&self,event:&EventBindRuntime)->Result<bool, ()>{Ok(event.done_callback.is_some())}
    fn has_bind_error_callback(&self,event:&EventBindRuntime)->Result<bool, ()>{Ok(event.error_callback.is_some())}
    fn has_bind_error_out(&self,event:&EventBindRuntime)->Result<bool, ()>{Ok(event.error_out.is_some())}
    fn has_prepare_done_callback(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.done_callback.is_some())}
    fn has_prepare_error_callback(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.error_callback.is_some())}
    fn invalid_bind(&self,event:&EventBindRuntime)->Result<bool, ()>{Ok(!self.valid_bind(event)?)}
    fn invalid_prepare(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(!self.valid_prepare(event)?)}
    fn no_bind_done_callback(&self,event:&EventBindRuntime)->Result<bool, ()>{Ok(!self.has_bind_done_callback(event)?)}
    fn no_bind_error_callback(&self,event:&EventBindRuntime)->Result<bool, ()>{Ok(!self.has_bind_error_callback(event)?)}
    fn no_bind_error_out(&self,event:&EventBindRuntime)->Result<bool, ()>{Ok(!self.has_bind_error_out(event)?)}
    fn no_prepare_done_callback(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(!self.has_prepare_done_callback(event)?)}
    fn no_prepare_error_callback(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(!self.has_prepare_error_callback(event)?)}
    fn on_unexpected_from_bind_decision(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_bind_error(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_bind_publish_error(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_bind_publish_success(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_bind_success(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_binding(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_done(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_errored(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_format_decision(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_idle(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_prepare_error(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_prepare_publish_error(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_prepare_publish_error_count(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_prepare_publish_success_count(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_prepare_publish_success_error(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_prepare_success(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_preparing(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_tokenize_decision(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_tokenizing(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_unexpected(&mut self)->Result<(), ()>{self.unexpected()}
    fn on_unexpected_from_uninitialized(&mut self)->Result<(), ()>{self.unexpected()}
    fn prepare_success(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.error=crate::ConditionerError::None;self.result=true;event.result.set(true);Ok(())}
    fn reject_bind_from_done(&mut self,event:&EventBindRuntime)->Result<(), ()>{self.reject_bind(event)}
    fn reject_bind_from_errored(&mut self,event:&EventBindRuntime)->Result<(), ()>{self.reject_bind(event)}
    fn reject_bind_from_idle(&mut self,event:&EventBindRuntime)->Result<(), ()>{self.reject_bind(event)}
    fn reject_bind_from_unexpected(&mut self,event:&EventBindRuntime)->Result<(), ()>{self.reject_bind(event)}
    fn reject_bind_from_uninitialized(&mut self,event:&EventBindRuntime)->Result<(), ()>{self.reject_bind(event)}
    fn reject_prepare_from_done(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.reject_prepare(event)}
    fn reject_prepare_from_errored(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.reject_prepare(event)}
    fn reject_prepare_from_idle(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.reject_prepare(event)}
    fn reject_prepare_from_unexpected(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.reject_prepare(event)}
    fn reject_prepare_from_uninitialized(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.reject_prepare(event)}
    fn set_error_backend_event_bind_runtime(&mut self,event:&EventBindRuntime)->Result<(), ()>{self.set_bind_error(event,crate::ConditionerError::Backend);Ok(())}
    fn set_error_backend_event_prepare_runtime(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.set_prepare_error(event,crate::ConditionerError::Backend);Ok(())}
    fn set_error_capacity_event_bind_runtime(&mut self,event:&EventBindRuntime)->Result<(), ()>{self.set_bind_error(event,crate::ConditionerError::Capacity);Ok(())}
    fn set_error_capacity_event_prepare_runtime(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.set_prepare_error(event,crate::ConditionerError::Capacity);Ok(())}
    fn set_error_invalid_argument_event_bind_runtime(&mut self,event:&EventBindRuntime)->Result<(), ()>{self.set_bind_error(event,crate::ConditionerError::InvalidArgument);Ok(())}
    fn set_error_invalid_argument_event_prepare_runtime(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.set_prepare_error(event,crate::ConditionerError::InvalidArgument);Ok(())}
    fn set_error_model_invalid_event_bind_runtime(&mut self,event:&EventBindRuntime)->Result<(), ()>{self.set_bind_error(event,crate::ConditionerError::ModelInvalid);Ok(())}
    fn set_error_model_invalid_event_prepare_runtime(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.set_prepare_error(event,crate::ConditionerError::ModelInvalid);Ok(())}
    fn set_error_untracked_event_bind_runtime(&mut self,event:&EventBindRuntime)->Result<(), ()>{self.set_bind_error(event,crate::ConditionerError::Untracked);Ok(())}
    fn set_error_untracked_event_prepare_runtime(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.set_prepare_error(event,crate::ConditionerError::Untracked);Ok(())}
    fn tokenize_count_invalid(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.tokenize_accepted.get()&&event.token_count.get()>event.token_capacity)}
    fn tokenize_error_backend_from_tokenize_decision(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{self.set_prepare_error(event,crate::ConditionerError::Backend);Ok(())}
    fn tokenize_error_capacity_code(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.tokenize_err_code.get()==4)}
    fn tokenize_error_backend_code(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.tokenize_err_code.get()==8)}
    fn tokenize_error_invalid_argument_code(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.tokenize_err_code.get()==1)}
    fn tokenize_error_model_invalid_code(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.tokenize_err_code.get()==2)}
    fn tokenize_error_untracked_code(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.tokenize_err_code.get()!=0&&!matches!(event.tokenize_err_code.get(),1|2|4|8))}
    fn tokenize_rejected_no_error(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(!event.tokenize_accepted.get()&&event.tokenize_err_code.get()==0)}
    fn tokenize_successful(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(event.tokenize_accepted.get()&&event.tokenize_err_code.get()==0&&event.token_count.get()<=event.token_capacity)}
    fn valid_bind(&self,event:&EventBindRuntime)->Result<bool, ()>{Ok(event.tokenizer_available&&event.formatter_available&&event.model_valid&&event.formatter.is_some()&&event.tokenizer.is_some())}
    fn valid_prepare_with_bind_defaults(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(self.valid_prepare(event)?&&event.use_bind_defaults)}
    fn valid_prepare_with_request_overrides(&self,event:&EventPrepareRuntime<'event>)->Result<bool, ()>{Ok(self.valid_prepare(event)?&&!event.use_bind_defaults)}
    fn write_bind_error_out_from_bind_error(&mut self,event:&EventBindRuntime)->Result<(), ()>{if let Some(out)=event.error_out.as_deref_mut(){*out=event.err.get().code()}Ok(())}
    fn write_bind_error_out_from_bind_success(&mut self,event:&EventBindRuntime)->Result<(), ()>{if let Some(out)=event.error_out.as_deref_mut(){*out=0}Ok(())}
    fn write_prepare_error_out_from_prepare_publish_error_count(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{if let Some(out)=event.error_out.as_deref_mut(){*out=event.err.get().code()}Ok(())}
    fn write_prepare_error_out_from_prepare_publish_success_count(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{if let Some(out)=event.error_out.as_deref_mut(){*out=0}Ok(())}
    fn write_prepare_token_count_from_prepare_error(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{*event.token_count_out=0;Ok(())}
    fn write_prepare_token_count_from_prepare_success(&mut self,event:&EventPrepareRuntime<'event>)->Result<(), ()>{*event.token_count_out=event.token_count.get();Ok(())}
}
impl TextConditionerContext {
    fn begin_bind(&mut self,event:&EventBindRuntime)->Result<(), ()>{self.bound=false;self.error=crate::ConditionerError::None;self.result=false;event.reset();self.formatter=event.formatter;self.tokenizer=event.tokenizer;self.tokenizer_bind=event.tokenizer_bind;self.add_special_default=event.add_special;self.parse_special_default=event.parse_special;Ok(())}
    fn begin_prepare_defaults(&mut self,event:&EventPrepareRuntime<'_>)->Result<(), ()>{event.reset();event.add_special.set(self.add_special_default);event.parse_special.set(self.parse_special_default);Ok(())}
    fn begin_prepare_request(&mut self,event:&EventPrepareRuntime<'_>)->Result<(), ()>{event.reset();event.add_special.set(event.add_special_request);event.parse_special.set(event.parse_special_request);Ok(())}
    fn valid_prepare(&self,event:&EventPrepareRuntime<'_>)->Result<bool, ()>{Ok(self.bound&&event.formatter_available&&event.tokenizer_available&&event.model_valid&&event.token_ids_present&&event.token_capacity>0&&event.token_capacity<=event.token_ids.borrow().len()&&event.formatted_capacity>0)}
    fn set_bind_error(&mut self,event:&EventBindRuntime,error:crate::ConditionerError){self.bound=false;self.error=error;self.result=false;event.err.set(error);event.result.set(false)}
    fn set_prepare_error(&mut self,event:&EventPrepareRuntime<'_>,error:crate::ConditionerError){self.error=error;self.result=false;self.token_count=0;event.err.set(error);event.result.set(false);event.token_count.set(0)}
    fn reject_bind(&mut self,event:&EventBindRuntime)->Result<(), ()>{self.set_bind_error(event,crate::ConditionerError::InvalidArgument);Ok(())}
    fn reject_prepare(&mut self,event:&EventPrepareRuntime<'_>)->Result<(), ()>{self.set_prepare_error(event,crate::ConditionerError::InvalidArgument);Ok(())}
    fn unexpected(&mut self)->Result<(), ()>{self.error=crate::ConditionerError::InvalidArgument;self.result=false;self.token_count=0;Ok(())}
}

/// Synchronous actor wrapper with generated state inspection.
pub struct TextConditioner { machine: TextConditionerStateMachine<TextConditionerContext> }
impl Default for TextConditioner { fn default()->Self { Self::new() } }
impl TextConditioner { pub fn new()->Self { Self { machine:TextConditionerStateMachine::new(TextConditionerContext::default()) } } pub fn state(&self)->&TextConditionerStates{self.machine.state()} pub fn is(&self,state:&TextConditionerStates)->bool{self.machine.is(state)} pub fn context(&self)->&TextConditionerContext{self.machine.context()} }
