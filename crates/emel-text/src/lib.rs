//! Text generation orchestration, expressed with `stateforward-sml` as the port grows.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::type_complexity)]
#![allow(clippy::large_stack_arrays)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::cast_lossless)]
extern crate alloc;

/// The state-machine DSL used to express generation lifecycles.
pub use sml;

/// The lifecycle phase of a text-generation request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GenerationPhase {
    /// Request validation and model binding.
    Initializing,
    /// Prompt tokens are being evaluated.
    Prefilling,
    /// New tokens are being decoded.
    Decoding,
    /// The request has completed.
    Complete,
}

#[cfg(test)]
mod conditioner_contract_tests {
    use super::{
        BindingDone, ChatMessage, Conditioner, ConditionerError, ConditionerObserver,
        ConditioningDone, FormatRequest, conditioner_event, format_raw,
    };

    #[test]
    fn raw_formatter_matches_reference_concatenation_and_capacity() {
        let messages = [
            ChatMessage {
                role: b"user",
                content: b"hello",
            },
            ChatMessage {
                role: b"assistant",
                content: b" world",
            },
        ];
        let mut output = [0u8; 11];
        assert_eq!(format_raw(&messages, &mut output), Ok(11));
        assert_eq!(&output, b"hello world");

        let mut too_small = [0u8; 10];
        assert_eq!(
            format_raw(&messages, &mut too_small),
            Err(ConditionerError::InvalidArgument)
        );
        let mut empty = [9u8; 1];
        assert_eq!(format_raw(&[], &mut empty), Ok(0));
        assert_eq!(empty, [9]);
    }

    #[test]
    fn bind_classifies_reference_error_paths() {
        let mut conditioner = Conditioner::default();
        assert_eq!(
            conditioner.process_event(conditioner_event::Bind {
                tokenizer_available: false,
                formatter_available: true,
                model_valid: true,
            }),
            Err(ConditionerError::Backend)
        );
        assert_eq!(
            conditioner.process_event(conditioner_event::Bind {
                tokenizer_available: true,
                formatter_available: true,
                model_valid: false,
            }),
            Err(ConditionerError::ModelInvalid)
        );
        assert!(
            conditioner
                .process_event(conditioner_event::Bind {
                    tokenizer_available: true,
                    formatter_available: true,
                    model_valid: true,
                })
                .is_ok()
        );
    }

    #[test]
    fn prepare_requires_bound_capacity_and_caller_storage() {
        let mut conditioner = Conditioner::default();
        let mut token_ids = [7, 8];
        let mut token_count = 99;
        assert_eq!(
            conditioner.prepare(conditioner_event::Prepare {
                messages: &[],
                formatter_available: true,
                tokenizer_available: true,
                model_valid: true,
                token_capacity: 2,
                token_ids: &mut token_ids,
                token_count: &mut token_count,
                add_generation_prompt: false,
                enable_thinking: false,
                add_special: true,
                parse_special: false,
            }),
            Err(ConditionerError::InvalidArgument)
        );
        conditioner
            .process_event(conditioner_event::Bind {
                tokenizer_available: true,
                formatter_available: true,
                model_valid: true,
            })
            .unwrap();
        assert_eq!(
            conditioner.prepare(conditioner_event::Prepare {
                messages: &[],
                formatter_available: true,
                tokenizer_available: true,
                model_valid: true,
                token_capacity: 0,
                token_ids: &mut token_ids,
                token_count: &mut token_count,
                add_generation_prompt: false,
                enable_thinking: false,
                add_special: true,
                parse_special: false,
            }),
            Err(ConditionerError::Capacity)
        );
        conditioner
            .prepare(conditioner_event::Prepare {
                messages: &[],
                formatter_available: true,
                tokenizer_available: true,
                model_valid: true,
                token_capacity: 2,
                token_ids: &mut token_ids,
                token_count: &mut token_count,
                add_generation_prompt: false,
                enable_thinking: false,
                add_special: true,
                parse_special: false,
            })
            .unwrap();
        assert_eq!(token_count, 0);
        assert_eq!(token_ids, [0, 0]);
    }

    fn injected_formatter(request: FormatRequest<'_>) -> Result<(), ConditionerError> {
        request.output[..2].copy_from_slice(b"ok");
        *request.output_length = 2;
        Ok(())
    }

    fn injected_tokenizer(
        text: &[u8],
        _add_special: bool,
        _parse_special: bool,
        output: &mut [i32],
    ) -> Result<usize, ConditionerError> {
        for (slot, byte) in output.iter_mut().zip(text.iter().copied()) {
            *slot = i32::from(byte);
        }
        Ok(text.len())
    }

    #[test]
    fn injected_dependencies_drive_prepare_without_allocations() {
        let mut conditioner = Conditioner::default();
        conditioner.set_dependencies(injected_formatter, injected_tokenizer);
        conditioner
            .process_event(conditioner_event::Bind {
                tokenizer_available: true,
                formatter_available: true,
                model_valid: true,
            })
            .unwrap();
        let mut tokens = [0; 2];
        let mut count = 0;
        conditioner
            .prepare(conditioner_event::Prepare {
                messages: &[],
                formatter_available: true,
                tokenizer_available: true,
                model_valid: true,
                token_capacity: 2,
                token_ids: &mut tokens,
                token_count: &mut count,
                add_generation_prompt: false,
                enable_thinking: false,
                add_special: true,
                parse_special: false,
            })
            .unwrap();
        assert_eq!(count, 2);
        assert_eq!(tokens, [b'o' as i32, b'k' as i32]);
    }

    fn rejecting_formatter(_: FormatRequest<'_>) -> Result<(), ConditionerError> {
        Err(ConditionerError::ModelInvalid)
    }

    #[test]
    fn injected_formatter_error_is_preserved() {
        let mut conditioner = Conditioner::default();
        conditioner.set_dependencies(rejecting_formatter, injected_tokenizer);
        conditioner
            .process_event(conditioner_event::Bind {
                tokenizer_available: true,
                formatter_available: true,
                model_valid: true,
            })
            .unwrap();
        let mut tokens = [0; 2];
        let mut count = 0;
        assert_eq!(
            conditioner.prepare(conditioner_event::Prepare {
                messages: &[],
                formatter_available: true,
                tokenizer_available: true,
                model_valid: true,
                token_capacity: 2,
                token_ids: &mut tokens,
                token_count: &mut count,
                add_generation_prompt: false,
                enable_thinking: false,
                add_special: true,
                parse_special: false,
            }),
            Err(ConditionerError::ModelInvalid)
        );
    }

    fn over_capacity_tokenizer(
        _: &[u8],
        _: bool,
        _: bool,
        _: &mut [i32],
    ) -> Result<usize, ConditionerError> {
        Ok(3)
    }

    #[test]
    fn injected_tokenizer_capacity_is_classified() {
        let mut conditioner = Conditioner::default();
        conditioner.set_dependencies(injected_formatter, over_capacity_tokenizer);
        conditioner
            .process_event(conditioner_event::Bind {
                tokenizer_available: true,
                formatter_available: true,
                model_valid: true,
            })
            .unwrap();
        let mut ids = [0; 2];
        let mut count = 0;
        assert_eq!(
            conditioner.prepare(conditioner_event::Prepare {
                messages: &[],
                formatter_available: true,
                tokenizer_available: true,
                model_valid: true,
                token_capacity: 2,
                token_ids: &mut ids,
                token_count: &mut count,
                add_generation_prompt: false,
                enable_thinking: false,
                add_special: true,
                parse_special: false,
            }),
            Err(ConditionerError::Capacity)
        );
    }

    #[derive(Default)]
    struct Observer {
        binding_done: usize,
        binding_errors: usize,
        conditioning_done: usize,
        conditioning_errors: usize,
    }

    impl ConditionerObserver for Observer {
        fn binding_done(&mut self, _: BindingDone) {
            self.binding_done += 1;
        }
        fn binding_error(&mut self, _: ConditionerError) {
            self.binding_errors += 1;
        }
        fn conditioning_done(&mut self, _: ConditioningDone) {
            self.conditioning_done += 1;
        }
        fn conditioning_error(&mut self, _: ConditionerError) {
            self.conditioning_errors += 1;
        }
    }

    #[test]
    fn observer_receives_rtc_success_and_error_outcomes() {
        let mut conditioner = Conditioner::default();
        let mut observer = Observer::default();
        let bind = conditioner.bind_with_observer(
            conditioner_event::Bind {
                tokenizer_available: true,
                formatter_available: true,
                model_valid: true,
            },
            &mut observer,
        );
        assert!(bind.is_ok());
        let mut ids = [0; 1];
        let mut count = 0;
        let prepared = conditioner.prepare_with_observer(
            conditioner_event::Prepare {
                messages: &[],
                formatter_available: true,
                tokenizer_available: true,
                model_valid: true,
                token_capacity: 1,
                token_ids: &mut ids,
                token_count: &mut count,
                add_generation_prompt: false,
                enable_thinking: false,
                add_special: true,
                parse_special: false,
            },
            &mut observer,
        );
        assert!(prepared.is_ok());
        assert_eq!((observer.binding_done, observer.conditioning_done), (1, 1));
        assert_eq!(
            (observer.binding_errors, observer.conditioning_errors),
            (0, 0)
        );
    }
}

/// Typed result of binding a tokenizer and formatter to a conditioner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConditionerError {
    None,
    InvalidArgument,
    ModelInvalid,
    Capacity,
    Backend,
    Untracked,
}

/// Formatting request supplied to an injected formatter.
#[derive(Debug)]
pub struct FormatRequest<'a> {
    pub messages: &'a [ChatMessage<'a>],
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub output: &'a mut [u8],
    pub output_length: &'a mut usize,
}

/// Synchronous formatter dependency. It must not retain the request borrows.
pub trait Formatter {
    fn format(&mut self, request: FormatRequest<'_>) -> Result<(), ConditionerError>;
}

/// Synchronous tokenizer dependency. It must write only to caller-owned output.
pub trait Tokenizer {
    fn bind(&mut self) -> Result<(), ConditionerError>;
    fn tokenize(
        &mut self,
        text: &[u8],
        add_special: bool,
        parse_special: bool,
        token_ids: &mut [i32],
    ) -> Result<usize, ConditionerError>;
}

impl ConditionerError {
    /// Returns the stable local error bit used by the reference contract.
    pub const fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::InvalidArgument => 1,
            Self::ModelInvalid => 2,
            Self::Capacity => 4,
            Self::Backend => 8,
            Self::Untracked => 16,
        }
    }
}

/// Public, allocation-free conditioner actor contract.
#[derive(Debug)]
pub struct Conditioner {
    bound: bool,
    formatted: [u8; 32 * 1024],
    formatter: Option<fn(FormatRequest<'_>) -> Result<(), ConditionerError>>,
    tokenizer: Option<fn(&[u8], bool, bool, &mut [i32]) -> Result<usize, ConditionerError>>,
}

impl Default for Conditioner {
    fn default() -> Self {
        Self {
            bound: false,
            formatted: [0; 32 * 1024],
            formatter: None,
            tokenizer: None,
        }
    }
}

/// Events accepted by [`Conditioner::process_event`].
pub mod conditioner_event {
    use super::ChatMessage;
    /// Bind dependencies required for formatting and tokenization.
    #[derive(Clone, Copy, Debug)]
    pub struct Bind {
        pub tokenizer_available: bool,
        pub formatter_available: bool,
        pub model_valid: bool,
    }

    /// Prepare a prompt into caller-owned token storage.
    #[derive(Debug)]
    pub struct Prepare<'a> {
        pub messages: &'a [ChatMessage<'a>],
        pub formatter_available: bool,
        pub tokenizer_available: bool,
        pub model_valid: bool,
        pub token_capacity: usize,
        pub token_ids: &'a mut [i32],
        pub token_count: &'a mut usize,
        pub add_generation_prompt: bool,
        pub enable_thinking: bool,
        pub add_special: bool,
        pub parse_special: bool,
    }
}

/// Successful completion of a conditioning request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConditioningDone {
    pub token_count: usize,
}

/// Successful completion of a binding request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindingDone;

/// Immediate synchronous outcome sink for conditioner dispatch.
pub trait ConditionerObserver {
    fn binding_done(&mut self, outcome: BindingDone);
    fn binding_error(&mut self, error: ConditionerError);
    fn conditioning_done(&mut self, outcome: ConditioningDone);
    fn conditioning_error(&mut self, error: ConditionerError);
}

/// A role/content pair accepted by the raw formatter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChatMessage<'a> {
    pub role: &'a [u8],
    pub content: &'a [u8],
}

/// Formats messages into caller-owned bytes using the reference raw contract.
///
/// # Errors
///
/// Returns [`ConditionerError::InvalidArgument`] when the output is too small
/// or the total message length overflows.
pub fn format_raw(
    messages: &[ChatMessage<'_>],
    output: &mut [u8],
) -> Result<usize, ConditionerError> {
    let required = messages
        .iter()
        .try_fold(0usize, |total, message| {
            total.checked_add(message.content.len())
        })
        .ok_or(ConditionerError::InvalidArgument)?;
    if required > output.len() {
        return Err(ConditionerError::InvalidArgument);
    }
    let mut offset = 0;
    for message in messages {
        let end = offset + message.content.len();
        output[offset..end].copy_from_slice(message.content);
        offset = end;
    }
    Ok(required)
}

impl Conditioner {
    /// Installs allocation-free function dependencies for the actor.
    pub fn set_dependencies(
        &mut self,
        formatter: fn(FormatRequest<'_>) -> Result<(), ConditionerError>,
        tokenizer: fn(&[u8], bool, bool, &mut [i32]) -> Result<usize, ConditionerError>,
    ) {
        self.formatter = Some(formatter);
        self.tokenizer = Some(tokenizer);
    }

    /// Binds dependencies and publishes the typed outcome synchronously.
    pub fn bind_with_observer<O: ConditionerObserver>(
        &mut self,
        event: conditioner_event::Bind,
        observer: &mut O,
    ) -> Result<BindingDone, ConditionerError> {
        match self.process_event(event) {
            Ok(()) => {
                let outcome = BindingDone;
                observer.binding_done(outcome);
                Ok(outcome)
            }
            Err(error) => {
                observer.binding_error(error);
                Err(error)
            }
        }
    }

    /// Prepares a prompt and publishes the typed outcome synchronously.
    pub fn prepare_with_observer<O: ConditionerObserver>(
        &mut self,
        event: conditioner_event::Prepare<'_>,
        observer: &mut O,
    ) -> Result<ConditioningDone, ConditionerError> {
        match self.prepare(event) {
            Ok(outcome) => {
                observer.conditioning_done(outcome);
                Ok(outcome)
            }
            Err(error) => {
                observer.conditioning_error(error);
                Err(error)
            }
        }
    }

    /// Dispatch one bind or prepare event without allocating.
    ///
    /// # Errors
    ///
    /// Returns a typed dependency or model-validation error when binding is
    /// not possible.
    pub const fn process_event(
        &mut self,
        event: conditioner_event::Bind,
    ) -> Result<(), ConditionerError> {
        if !event.tokenizer_available || !event.formatter_available {
            return Err(ConditionerError::Backend);
        }
        if !event.model_valid {
            return Err(ConditionerError::ModelInvalid);
        }
        self.bound = true;
        Ok(())
    }

    /// Prepare caller-owned token storage after a successful bind.
    ///
    /// # Errors
    ///
    /// Returns a typed lifecycle, dependency, model, formatting, or capacity
    /// error when preparation cannot complete.
    #[allow(clippy::needless_pass_by_value)]
    pub fn prepare(
        &mut self,
        event: conditioner_event::Prepare<'_>,
    ) -> Result<ConditioningDone, ConditionerError> {
        if !self.bound {
            return Err(ConditionerError::InvalidArgument);
        }
        if !event.formatter_available || !event.tokenizer_available {
            return Err(ConditionerError::Backend);
        }
        if !event.model_valid {
            return Err(ConditionerError::ModelInvalid);
        }
        if event.token_capacity == 0 {
            return Err(ConditionerError::Capacity);
        }
        if event.token_capacity > event.token_ids.len() {
            return Err(ConditionerError::InvalidArgument);
        }
        let mut formatted_len = 0;
        let formatter = self.formatter.unwrap_or(default_formatter);
        formatter(FormatRequest {
            messages: event.messages,
            add_generation_prompt: event.add_generation_prompt,
            enable_thinking: event.enable_thinking,
            output: &mut self.formatted,
            output_length: &mut formatted_len,
        })?;
        if formatted_len > self.formatted.len() {
            return Err(ConditionerError::InvalidArgument);
        }
        event.token_ids.fill(0);
        if formatted_len == 0 {
            *event.token_count = 0;
            return Ok(ConditioningDone { token_count: 0 });
        }
        let count = if let Some(tokenizer) = self.tokenizer {
            tokenizer(
                &self.formatted[..formatted_len],
                true,
                false,
                &mut event.token_ids[..event.token_capacity],
            )?
        } else {
            let count = formatted_len.min(event.token_capacity);
            for (slot, byte) in event.token_ids[..count]
                .iter_mut()
                .zip(self.formatted[..count].iter().copied())
            {
                *slot = i32::from(byte);
            }
            count
        };
        if count > event.token_capacity {
            return Err(ConditionerError::Capacity);
        }
        *event.token_count = count;
        Ok(ConditioningDone { token_count: count })
    }
}

fn default_formatter(request: FormatRequest<'_>) -> Result<(), ConditionerError> {
    *request.output_length = format_raw(request.messages, request.output)?;
    Ok(())
}
pub(crate) mod conditioner;
pub(crate) mod detokenizer;
pub(crate) mod encoders;
pub(crate) mod formatter;
pub(crate) mod generator;
pub(crate) mod jinja;
pub(crate) mod renderer;
pub(crate) mod tokenizer;

#[cfg(test)]
mod detokenizer_port_tests;

#[cfg(test)]
#[path = "generator/layer_port_tests.rs"]
mod layer_port_tests;
