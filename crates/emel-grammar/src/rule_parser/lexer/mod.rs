//! Allocation-free synchronous GBNF rule-parser lexer.

pub mod sm;

/// A caller-owned cursor into a grammar source.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Cursor<'a> {
    pub input: &'a str,
    pub offset: u32,
    pub token_count: u32,
}

/// Token categories emitted by the lexer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenKind {
    Unknown = 0,
    Identifier = 1,
    StringLiteral = 2,
    CharacterClass = 3,
    RuleReference = 4,
    DefinitionOperator = 5,
    Alternation = 6,
    Dot = 7,
    OpenGroup = 8,
    CloseGroup = 9,
    Quantifier = 10,
    Newline = 11,
}

/// A borrowed token view into the request input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Token<'a> {
    pub kind: TokenKind,
    pub text: &'a str,
    pub start: u32,
    pub end: u32,
}

/// Successful completion of a `next` request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NextDone<'a> {
    pub token: Token<'a>,
    pub has_token: bool,
    pub next_cursor: Cursor<'a>,
}

/// Rejected request or unexpected-event completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NextError {
    pub err: Error,
}

/// Error categories from the pinned lexer contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum Error {
    None = 0,
    InvalidRequest = 1,
    ParseFailed = 2,
    InternalError = 4,
    Untracked = 8,
}
/// Synchronous completion callback.
pub type DoneCallback<'callback> = dyn for<'a> FnMut(NextDone<'a>) -> bool + 'callback;

/// Synchronous error callback.
pub type ErrorCallback<'callback> = dyn FnMut(NextError) -> bool + 'callback;

/// A scan request.
pub struct EventScanNext<'input, 'callback> {
    pub cursor: Cursor<'input>,
    pub on_done: Option<&'callback mut DoneCallback<'callback>>,
    pub on_error: Option<&'callback mut ErrorCallback<'callback>>,
}

impl<'input, 'callback> EventScanNext<'input, 'callback> {
    #[must_use]
    pub fn new(cursor: Cursor<'input>, on_done: &'callback mut DoneCallback<'callback>, on_error: &'callback mut ErrorCallback<'callback>) -> Self { Self { cursor, on_done: Some(on_done), on_error: Some(on_error) } }
    #[must_use]
    pub fn with_callbacks(cursor: Cursor<'input>, on_done: Option<&'callback mut DoneCallback<'callback>>, on_error: Option<&'callback mut ErrorCallback<'callback>>) -> Self { Self { cursor, on_done, on_error } }
}


impl core::fmt::Debug for EventScanNext<'_, '_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_struct("EventScanNext").field("cursor", &self.cursor).finish_non_exhaustive()
    }
}

/// Namespace aliases mirroring the C++ event layout.
pub mod event {
    pub use super::{Cursor, DoneCallback, Error, ErrorCallback, EventScanNext, NextDone, NextError, Token, TokenKind};
}

/// Namespace aliases mirroring the C++ completion layout.
pub mod events {
    pub use super::{NextDone, NextError};
}


#[cfg(test)]
mod tests {
    use super::sm::{GbnfRuleParserLexerStateMachine, GbnfRuleParserLexerStates};
    use super::*;

    fn scan(input: &str) -> Vec<(TokenKind, String)> {
        let mut machine = GbnfRuleParserLexerStateMachine::default();
        let mut cursor = Cursor { input, offset: 0, token_count: 0 };
        let mut tokens = Vec::new();
        while (cursor.offset as usize) < input.len() {
            let mut next_offset = cursor.offset;
            let mut next_count = cursor.token_count;
            let mut done = |event: NextDone<'_>| {
                if event.has_token { tokens.push((event.token.kind, event.token.text.to_owned())); }
                next_offset = event.next_cursor.offset;
                next_count = event.next_cursor.token_count;
                true
            };
            let mut error = |_event: NextError| false;
            machine.process_event(EventScanNext::new(cursor, &mut done, &mut error));
            cursor.offset = next_offset;
            cursor.token_count = next_count;
        }
        tokens
    }

    #[test]
    fn tokenizes_operators_layout_and_literals() {
        let tokens = scan(" # c\r\nroot ::= (foo|\"x\")+[a-z]{2,3}");
        assert_eq!(tokens, vec![
            (TokenKind::Newline, "\r\n".into()), (TokenKind::Identifier, "root".into()),
            (TokenKind::DefinitionOperator, "::=".into()), (TokenKind::OpenGroup, "(".into()),
            (TokenKind::Identifier, "foo".into()), (TokenKind::Alternation, "|".into()),
            (TokenKind::StringLiteral, "\"x\"".into()), (TokenKind::CloseGroup, ")".into()),
            (TokenKind::Quantifier, "+".into()), (TokenKind::CharacterClass, "[a-z]".into()),
            (TokenKind::Quantifier, "{2,3}".into()),
        ]);
    }

    #[test]
    fn classifies_rule_references_and_unknowns() {
        let tokens = scan("<[12]> !<[3]> @");
        assert_eq!(tokens[0], (TokenKind::RuleReference, "<[12]>".into()));
        assert_eq!(tokens[1], (TokenKind::RuleReference, "!<[3]>".into()));
        assert_eq!(tokens[2].0, TokenKind::Unknown);
    }

    #[test]
    fn starts_initialized_and_rejects_missing_callbacks() {
        let mut machine = GbnfRuleParserLexerStateMachine::default();
        assert!(machine.is(GbnfRuleParserLexerStates::Initialized));
        let cursor = Cursor { input: "a", offset: 0, token_count: 0 };
        let mut error = |event: NextError| { assert_eq!(event.err, Error::InvalidRequest); true };
        assert!(machine.process_event(EventScanNext::with_callbacks(cursor, None, Some(&mut error))));
        assert!(machine.is(GbnfRuleParserLexerStates::Initialized));
    }
}