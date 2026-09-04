//! Source-aligned synchronous GBNF lexer actor.
#![allow(
    clippy::cast_possible_truncation,
    clippy::trivially_copy_pass_by_ref,
    clippy::missing_const_for_fn
)]

use super::{Cursor, Error, EventScanNext, NextDone, NextError, Token, TokenKind};

/// States exposed by the lexer actor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GbnfRuleParserLexerStates {
    #[default]
    Initialized,
    Scanning,
    ScanReady,
}

/// Run-to-completion lexer actor.
#[derive(Debug, Default)]
pub struct GbnfRuleParserLexerStateMachine {
    state: GbnfRuleParserLexerStates,
    context: GbnfRuleParserLexerContext,
}

impl GbnfRuleParserLexerStateMachine {
    #[must_use]
    pub const fn new(context: GbnfRuleParserLexerContext) -> Self {
        Self {
            state: GbnfRuleParserLexerStates::Initialized,
            context,
        }
    }
    #[must_use]
    pub const fn state(&self) -> GbnfRuleParserLexerStates {
        self.state
    }
    #[must_use]
    pub fn is(&self, state: GbnfRuleParserLexerStates) -> bool {
        self.state == state
    }

    /// Processes one bounded scan request. Invalid callback/cursor requests are
    /// rejected before any token dispatch, matching the pinned guard ordering.
    pub fn process_event(&mut self, event: EventScanNext<'_, '_>) -> bool {
        if event.on_done.is_none() || event.on_error.is_none() {
            if let Some(callback) = event.on_error {
                callback(NextError {
                    err: Error::InvalidRequest,
                });
            }
            return true;
        }
        let offset = event.cursor.offset as usize;
        // The pinned lexer validates byte offsets, not UTF-8 scalar boundaries.
        // Keep that contract while avoiding invalid borrowed `str` slices below.
        if offset > event.cursor.input.len() {
            if let Some(callback) = event.on_error {
                callback(NextError {
                    err: Error::InvalidRequest,
                });
            }
            return true;
        }
        self.state = GbnfRuleParserLexerStates::ScanReady;
        self.context.scan(event);
        self.state = GbnfRuleParserLexerStates::Scanning;
        true
    }

    /// Dispatches an unsupported event without changing the current state.
    #[allow(
        clippy::unused_self,
        reason = "The public unexpected-event API keeps the state-machine receiver shape"
    )]
    pub fn process_unexpected_event<'callback>(
        &self,
        on_error: Option<&'callback mut super::ErrorCallback<'callback>>,
    ) -> bool {
        emit_error(on_error, Error::InternalError);
        true
    }
}

#[derive(Debug, Default)]
pub struct GbnfRuleParserLexerContext {
    start: usize,
    has_input: bool,
    first_char: u8,
}

impl GbnfRuleParserLexerContext {
    fn scan(&mut self, event: EventScanNext<'_, '_>) {
        let input = event.cursor.input.as_bytes();
        self.start = skip_layout(input, event.cursor.offset as usize);
        self.has_input = self.start < input.len();
        self.first_char = input.get(self.start).copied().unwrap_or_default();
        if !self.has_input {
            if self.start == event.cursor.offset as usize {
                emit_done(
                    event.on_done,
                    NextDone {
                        token: empty_token(event.cursor.input),
                        has_token: false,
                        next_cursor: event.cursor,
                    },
                );
            } else {
                // Pinned layout-exhausted emits a default token while advancing
                // over comments/spacing in the returned cursor.
                emit_token(
                    event.on_done,
                    event.cursor,
                    empty_token(event.cursor.input),
                    self.start,
                );
            }
            return;
        }
        let (kind, end) = self.classify(input);
        let end = end.min(input.len()).max(self.start + 1);
        let token = Token {
            kind,
            // Byte offsets may begin inside UTF-8; only expose representable
            // borrowed text and preserve the pinned byte cursor progression.
            text: token_text(event.cursor.input, self.start, end),
            start: self.start as u32,
            end: end as u32,
        };
        emit_token(event.on_done, event.cursor, token, end);
    }

    fn classify(&self, input: &[u8]) -> (TokenKind, usize) {
        let start = self.start;
        if input.get(start..start + 2) == Some(b"\r\n") {
            return (TokenKind::Newline, start + 2);
        }
        if matches!(self.first_char, b'\r' | b'\n') {
            return (TokenKind::Newline, start + 1);
        }
        if input.get(start..start + 3) == Some(b"::=") {
            return (TokenKind::DefinitionOperator, start + 3);
        }
        if let Some(kind) = match self.first_char {
            b'|' => Some(TokenKind::Alternation),
            b'.' => Some(TokenKind::Dot),
            b'(' => Some(TokenKind::OpenGroup),
            b')' => Some(TokenKind::CloseGroup),
            b'+' | b'*' | b'?' => Some(TokenKind::Quantifier),
            _ => None,
        } {
            return (kind, start + 1);
        }
        if self.first_char == b'"' {
            return (TokenKind::StringLiteral, quoted_end(input, start, b'"'));
        }
        if self.first_char == b'[' {
            return (TokenKind::CharacterClass, quoted_end(input, start, b']'));
        }
        if self.first_char == b'{' {
            return (TokenKind::Quantifier, braced_end(input, start));
        }
        if self.first_char == b'!' && input.get(start + 1..start + 3) == Some(b"<[") {
            let end = rule_ref_end(input, start + 1);
            if end > start + 1 {
                return (TokenKind::RuleReference, end);
            }
            return (TokenKind::Unknown, start + 1);
        }
        if input.get(start..start + 2) == Some(b"<[") {
            let end = rule_ref_end(input, start);
            if end > start {
                return (TokenKind::RuleReference, end);
            }
        }
        if is_word_char(self.first_char) {
            let mut end = start + 1;
            while end < input.len() && is_word_char(input[end]) {
                end += 1;
            }
            return (TokenKind::Identifier, end);
        }
        (TokenKind::Unknown, start + char_len(input, start))
    }
}

fn emit_done(callback: Option<&mut super::DoneCallback<'_>>, done: NextDone<'_>) {
    if let Some(callback) = callback {
        callback(done);
    }
}
fn emit_token<'input, 'callback>(
    callback: Option<&'callback mut super::DoneCallback<'callback>>,
    cursor: Cursor<'input>,
    token: Token<'input>,
    end: usize,
) {
    emit_done(
        callback,
        NextDone {
            token,
            has_token: true,
            next_cursor: Cursor {
                input: cursor.input,
                offset: end as u32,
                token_count: cursor.token_count.saturating_add(1),
            },
        },
    );
}
fn emit_error<'callback>(
    callback: Option<&'callback mut super::ErrorCallback<'callback>>,
    err: Error,
) {
    if let Some(callback) = callback {
        callback(NextError { err });
    }
}
fn empty_token(input: &str) -> Token<'_> {
    Token {
        kind: TokenKind::Unknown,
        text: &input[0..0],
        start: 0,
        end: 0,
    }
}
fn token_text(input: &str, start: usize, end: usize) -> &str {
    input.get(start..end).unwrap_or("")
}
fn char_len(input: &[u8], pos: usize) -> usize {
    core::str::from_utf8(&input[pos..])
        .ok()
        .and_then(|s| s.chars().next())
        .map_or(1, char::len_utf8)
}
const fn is_word_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'-'
}
const fn skip_layout(input: &[u8], mut pos: usize) -> usize {
    while pos < input.len() {
        match input[pos] {
            b' ' | b'\t' => pos += 1,
            b'#' => {
                pos += 1;
                while pos < input.len() && !matches!(input[pos], b'\r' | b'\n') {
                    pos += 1;
                }
            }
            _ => break,
        }
    }
    pos
}
const fn quoted_end(input: &[u8], pos: usize, terminator: u8) -> usize {
    let mut scan = pos + 1;
    while scan < input.len() {
        let c = input[scan];
        scan += 1;
        if c == b'\\' && scan < input.len() {
            scan += 1;
        } else if c == terminator {
            break;
        }
    }
    scan
}
const fn braced_end(input: &[u8], pos: usize) -> usize {
    let mut scan = pos + 1;
    while scan < input.len() {
        if input[scan] == b'}' {
            return scan + 1;
        }
        scan += 1;
    }
    scan
}
fn rule_ref_end(input: &[u8], pos: usize) -> usize {
    if input.get(pos..pos + 2) != Some(b"<[") {
        return pos;
    }
    let mut scan = pos + 2;
    while scan < input.len() && input[scan].is_ascii_digit() {
        scan += 1;
    }
    if input.get(scan..scan + 2) == Some(b"]>") {
        scan + 2
    } else {
        pos
    }
}
