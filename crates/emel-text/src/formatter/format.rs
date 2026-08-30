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

/// Synchronous formatter dependency. Implementations must not retain request borrows.
pub trait Formatter {
    /// Formats one request synchronously.
    fn format(&mut self, request: FormatRequest<'_>) -> Result<(), ConditionerError>;
}

/// Formats message content into caller-owned bytes using the reference raw contract.
///
/// Roles and formatting flags are intentionally ignored. Length accounting is checked
/// before any output write, so invalid requests never partially modify the destination.
///
/// # Errors
///
/// Returns [`ConditionerError::InvalidArgument`] when the total content length overflows
/// or the output capacity is insufficient.
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

/// Adapts [`format_raw`] to the injected formatter request contract.
pub fn raw_formatter(request: FormatRequest<'_>) -> Result<(), ConditionerError> {
    *request.output_length = 0;
    *request.output_length = format_raw(request.messages, request.output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ChatMessage, format_raw};
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
    fn empty_input_succeeds_without_writing_destination() {
        let mut output = [0xA5; 2];
        assert_eq!(format_raw(&[], &mut output), Ok(0));
        assert_eq!(output, [0xA5; 2]);
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
}
