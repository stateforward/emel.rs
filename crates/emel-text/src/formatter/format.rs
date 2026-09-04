//! Pure, allocation-free formatter dependency and reference raw implementation.

use super::super::ConditionerError;

/// A role/content pair accepted by the raw formatter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChatMessage<'a> {
    /// Message role, accepted for API compatibility but ignored by raw formatting.
    pub role: &'a [u8],
    /// Message content copied into the output in message order.
    pub content: &'a [u8],
}

/// Formatting request supplied by an injected, stateless formatter.
#[derive(Debug)]
pub struct FormatRequest<'a> {
    /// Messages to format.
    pub messages: &'a [ChatMessage<'a>],
    /// Accepted for API compatibility; raw formatting ignores this flag.
    pub add_generation_prompt: bool,
    /// Accepted for API compatibility; raw formatting ignores this flag.
    pub enable_thinking: bool,
    /// Caller-owned output storage.
    pub output: &'a mut [u8],
    /// Set to zero before formatting and to the required length only on success.
    pub output_length: &'a mut usize,
}

/// C++-shaped raw formatting request with explicit null-output semantics.
#[derive(Debug)]
pub struct RawFormatRequest<'a> {
    /// Messages to format in order.
    pub messages: &'a [ChatMessage<'a>],
    /// Accepted for API compatibility; raw formatting ignores this flag.
    pub add_generation_prompt: bool,
    /// Accepted for API compatibility; raw formatting ignores this flag.
    pub enable_thinking: bool,
    /// Caller-owned output storage, or `None` for a null output pointer.
    pub output: Option<&'a mut [u8]>,
    /// Declared output capacity, independent of whether output is present.
    pub output_capacity: usize,
    /// Optional caller-owned output-length publication sink.
    pub output_length_out: Option<&'a mut usize>,
}

/// Synchronous formatter dependency. Implementations must not retain request borrows.
pub trait Formatter {
    /// Formats one request synchronously.
    fn format(&mut self, request: FormatRequest<'_>) -> Result<(), ConditionerError>;
}

/// Function-shaped formatter dependency used by the conditioner boundary.
pub type FormatterFn = fn(FormatRequest<'_>) -> Result<(), ConditionerError>;

/// Formats one raw request using the pinned C++ contract.
///
/// Roles and formatting flags are ignored. A null output is valid only with zero
/// capacity, and output length is reset before validation and published only on success.
/// Length accounting and capacity checks happen before any destination write. The declared
/// capacity is independent of the Rust slice length, while the slice length is checked
/// separately before writing to preserve the safe API boundary.
pub fn format_raw_request(mut request: RawFormatRequest<'_>) -> Result<(), ConditionerError> {
    if let Some(output_length_out) = request.output_length_out.as_deref_mut() {
        *output_length_out = 0;
    }
    if request.output.is_none() && request.output_capacity > 0 {
        return Err(ConditionerError::InvalidArgument);
    }
    let required = checked_total(request.messages.iter().map(|message| message.content.len()))
        .ok_or(ConditionerError::InvalidArgument)?;
    if required > request.output_capacity {
        return Err(ConditionerError::InvalidArgument);
    }
    if let Some(output) = request.output.as_ref()
        && required > output.len()
    {
        return Err(ConditionerError::InvalidArgument);
    }

    if let Some(output) = request.output {
        let mut offset = 0;
        for message in request.messages {
            let end = offset + message.content.len();
            output[offset..end].copy_from_slice(message.content);
            offset = end;
        }
    }
    if let Some(output_length_out) = request.output_length_out {
        *output_length_out = required;
    }
    Ok(())
}

fn checked_total<I>(lengths: I) -> Option<usize>
where
    I: IntoIterator<Item = usize>,
{
    lengths.into_iter().try_fold(0usize, usize::checked_add)
}

/// Formats message content into caller-owned bytes using the compatibility contract.
///
/// This wrapper retains the original safe slice API; use [`format_raw_request`] when
/// explicit null-output and declared-capacity semantics are required.
pub fn format_raw(
    messages: &[ChatMessage<'_>],
    output: &mut [u8],
) -> Result<usize, ConditionerError> {
    let mut output_length = 0;
    let output_capacity = output.len();
    format_raw_request(RawFormatRequest {
        messages,
        add_generation_prompt: false,
        enable_thinking: false,
        output: Some(output),
        output_capacity,
        output_length_out: Some(&mut output_length),
    })?;
    Ok(output_length)
}

/// Adapts [`format_raw_request`] to the injected formatter request contract.
pub fn raw_formatter(request: FormatRequest<'_>) -> Result<(), ConditionerError> {
    let output_capacity = request.output.len();
    format_raw_request(RawFormatRequest {
        messages: request.messages,
        add_generation_prompt: request.add_generation_prompt,
        enable_thinking: request.enable_thinking,
        output: Some(request.output),
        output_capacity,
        output_length_out: Some(request.output_length),
    })
}

#[cfg(test)]
mod tests {
    use super::{ChatMessage, FormatRequest, format_raw, raw_formatter};
    use crate::ConditionerError;

    #[test]
    fn concatenates_content_only_and_ignores_roles() {
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
        let mut output = [0; 11];
        assert_eq!(format_raw(&messages, &mut output), Ok(11));
        assert_eq!(&output, b"hello world");
    }

    #[test]
    fn zero_capacity_empty_request_succeeds_without_writing_destination() {
        let mut output: [u8; 0] = [];
        assert_eq!(format_raw(&[], &mut output), Ok(0));
    }

    #[test]
    fn zero_capacity_nonempty_request_rejects_before_writing() {
        let messages = [ChatMessage {
            role: b"ignored",
            content: b"hello",
        }];
        let mut output: [u8; 0] = [];
        assert_eq!(
            format_raw(&messages, &mut output),
            Err(ConditionerError::InvalidArgument)
        );
    }

    #[test]
    fn too_small_output_fails_without_partial_writes() {
        let messages = [ChatMessage {
            role: b"ignored",
            content: b"hello",
        }];
        let mut output = [0xA5; 4];
        assert_eq!(
            format_raw(&messages, &mut output),
            Err(ConditionerError::InvalidArgument)
        );
        assert_eq!(output, [0xA5; 4]);
    }

    #[test]
    fn checked_accounting_handles_representable_lengths() {
        let messages = [
            ChatMessage {
                role: &[],
                content: &[0; 1],
            },
            ChatMessage {
                role: &[],
                content: &[0; 1],
            },
        ];
        let mut output = [0; 2];
        assert_eq!(format_raw(&messages, &mut output), Ok(2));
    }

    #[test]
    fn aggregate_length_overflow_is_rejected_before_writing() {
        assert_eq!(super::checked_total([usize::MAX, 1]), None);
    }

    #[test]
    fn raw_request_accepts_null_output_only_at_zero_capacity() {
        let mut output_length = 99;
        assert_eq!(
            super::format_raw_request(super::RawFormatRequest {
                messages: &[],
                add_generation_prompt: false,
                enable_thinking: false,
                output: None,
                output_capacity: 0,
                output_length_out: Some(&mut output_length),
            }),
            Ok(())
        );
        assert_eq!(output_length, 0);

        output_length = 99;
        assert_eq!(
            super::format_raw_request(super::RawFormatRequest {
                messages: &[],
                add_generation_prompt: false,
                enable_thinking: false,
                output: None,
                output_capacity: 1,
                output_length_out: Some(&mut output_length),
            }),
            Err(ConditionerError::InvalidArgument)
        );
        assert_eq!(output_length, 0);
    }

    #[test]
    fn raw_request_checks_declared_capacity_and_slice_capacity_before_writing() {
        let messages = [ChatMessage {
            role: &[],
            content: b"hello",
        }];
        let mut output = [0xA5; 5];
        let mut output_length = 99;
        assert_eq!(
            super::format_raw_request(super::RawFormatRequest {
                messages: &messages,
                add_generation_prompt: false,
                enable_thinking: false,
                output: Some(&mut output),
                output_capacity: 4,
                output_length_out: Some(&mut output_length),
            }),
            Err(ConditionerError::InvalidArgument)
        );
        assert_eq!(output_length, 0);
        assert_eq!(output, [0xA5; 5]);

        output_length = 99;
        assert_eq!(
            super::format_raw_request(super::RawFormatRequest {
                messages: &messages,
                add_generation_prompt: false,
                enable_thinking: false,
                output: Some(&mut output[..4]),
                output_capacity: 5,
                output_length_out: Some(&mut output_length),
            }),
            Err(ConditionerError::InvalidArgument)
        );
        assert_eq!(output_length, 0);
        assert_eq!(output, [0xA5; 5]);
    }

    #[test]
    fn raw_formatter_publishes_zero_for_empty_request() {
        let mut output: [u8; 0] = [];
        let mut output_length = 99;
        assert_eq!(
            raw_formatter(FormatRequest {
                messages: &[],
                add_generation_prompt: true,
                enable_thinking: true,
                output: &mut output,
                output_length: &mut output_length,
            }),
            Ok(())
        );
        assert_eq!(output_length, 0);
    }

    #[test]
    fn raw_formatter_resets_length_on_error_and_publishes_on_success() {
        let messages = [ChatMessage {
            role: &[],
            content: b"hello",
        }];
        let mut output = [0xA5; 4];
        let mut output_length = 99;
        assert_eq!(
            raw_formatter(FormatRequest {
                messages: &messages,
                add_generation_prompt: true,
                enable_thinking: true,
                output: &mut output,
                output_length: &mut output_length,
            }),
            Err(ConditionerError::InvalidArgument)
        );
        assert_eq!(output_length, 0);
        assert_eq!(output, [0xA5; 4]);

        let mut output = [0xA5; 5];
        output_length = 99;
        assert_eq!(
            raw_formatter(FormatRequest {
                messages: &messages,
                add_generation_prompt: false,
                enable_thinking: false,
                output: &mut output,
                output_length: &mut output_length,
            }),
            Ok(())
        );
        assert_eq!(output_length, 5);
        assert_eq!(output, *b"hello");
    }
}
