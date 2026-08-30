//! Source-aligned synchronous GBNF lexer actor.

use super::{Cursor, Error, EventScanNext, NextDone, NextError, Token, TokenKind};

/// States exposed by the lexer actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GbnfRuleParserLexerStates { Initialized, Scanning }

/// Run-to-completion lexer actor.
#[derive(Debug, Default)]
pub struct GbnfRuleParserLexerStateMachine { state: GbnfRuleParserLexerStates, context: GbnfRuleParserLexerContext }

/// Concise actor alias.
pub type GbnfRuleParserLexer = GbnfRuleParserLexerStateMachine;

impl Default for GbnfRuleParserLexerStates { fn default() -> Self { Self::Initialized } }
impl GbnfRuleParserLexerStateMachine {
    #[must_use] pub const fn new(context: GbnfRuleParserLexerContext) -> Self { Self { state: GbnfRuleParserLexerStates::Initialized, context } }
    #[must_use] pub const fn state(&self) -> GbnfRuleParserLexerStates { self.state }
    #[must_use] pub fn is(&self, state: GbnfRuleParserLexerStates) -> bool { self.state == state }
    pub fn process_event<'input, 'callback>(&mut self, event: EventScanNext<'input, 'callback>) -> bool {
        if event.on_done.is_none() || event.on_error.is_none() { emit_error(event.on_error, Error::InvalidRequest); return true; }
        if event.cursor.offset as usize > event.cursor.input.len() { emit_error(event.on_error, Error::InvalidRequest); return true; }
        self.state = GbnfRuleParserLexerStates::Scanning;
        self.context.scan(event);
        true
    }
    pub fn process_unexpected_event<'callback>(&mut self, on_error: Option<&'callback mut super::ErrorCallback<'callback>>) -> bool {
        if let Some(callback) = on_error { callback(NextError { err: Error::InternalError }); }
        true
    }
}

#[derive(Debug, Default)]
pub struct GbnfRuleParserLexerContext { start: usize, has_input: bool, first_char: u8 }

impl GbnfRuleParserLexerContext {
    fn scan<'input, 'callback>(&mut self, event: EventScanNext<'input, 'callback>) {
        let input = event.cursor.input.as_bytes();
        self.start = skip_layout(input, event.cursor.offset as usize);
        self.has_input = self.start < input.len();
        self.first_char = input.get(self.start).copied().unwrap_or_default();
        if !self.has_input {
            if self.start == event.cursor.offset as usize { emit_done(event.on_done, NextDone { token: empty_token(event.cursor.input), has_token: false, next_cursor: event.cursor }); }
            else { emit_token(event.on_done, event.cursor, empty_token(event.cursor.input), self.start); }
            return;
        }
        let (kind, end) = self.classify(input);
        let end = valid_boundary(event.cursor.input, self.start, end.min(input.len()).max(self.start + 1));
        let token = Token { kind, text: &event.cursor.input[self.start..end], start: self.start as u32, end: end as u32 };
        emit_token(event.on_done, event.cursor, token, end);
    }
    fn classify(&self, input: &[u8]) -> (TokenKind, usize) {
        let start = self.start;
        if input.get(start..start + 2) == Some(b"\r\n") { return (TokenKind::Newline, start + 2); }
        if matches!(self.first_char, b'\r' | b'\n') { return (TokenKind::Newline, start + 1); }
        if input.get(start..start + 3) == Some(b"::=") { return (TokenKind::DefinitionOperator, start + 3); }
        if let Some(kind) = match self.first_char { b'|' => Some(TokenKind::Alternation), b'.' => Some(TokenKind::Dot), b'(' => Some(TokenKind::OpenGroup), b')' => Some(TokenKind::CloseGroup), b'+' | b'*' | b'?' => Some(TokenKind::Quantifier), _ => None } { return (kind, start + 1); }
        if self.first_char == b'"' { return (TokenKind::StringLiteral, quoted_end(input, start, b'"')); }
        if self.first_char == b'[' { return (TokenKind::CharacterClass, quoted_end(input, start, b']')); }
        if self.first_char == b'{' { return (TokenKind::Quantifier, braced_end(input, start)); }
        if self.first_char == b'!' && input.get(start + 1..start + 3) == Some(b"<[") { let end = rule_ref_end(input, start + 1); if end > start + 1 { return (TokenKind::RuleReference, end); } return (TokenKind::Unknown, start + 1); }
        if input.get(start..start + 2) == Some(b"<[") { let end = rule_ref_end(input, start); if end > start { return (TokenKind::RuleReference, end); } }
        if is_word_char(self.first_char) { let mut end = start + 1; while end < input.len() && is_word_char(input[end]) { end += 1; } return (TokenKind::Identifier, end); }
        (TokenKind::Unknown, start + char_len(input, start))
    }
}

fn emit_done<'callback>(callback: Option<&'callback mut super::DoneCallback<'callback>>, done: NextDone<'_>) { if let Some(callback) = callback { callback(done); } }
fn emit_token<'input, 'callback>(callback: Option<&'callback mut super::DoneCallback<'callback>>, cursor: Cursor<'input>, token: Token<'input>, end: usize) { emit_done(callback, NextDone { token, has_token: true, next_cursor: Cursor { input: cursor.input, offset: end as u32, token_count: cursor.token_count.saturating_add(1) } }); }
fn emit_error<'callback>(callback: Option<&'callback mut super::ErrorCallback<'callback>>, err: Error) { if let Some(callback) = callback { callback(NextError { err }); } }
fn empty_token(input: &str) -> Token<'_> { Token { kind: TokenKind::Unknown, text: &input[0..0], start: 0, end: 0 } }
fn valid_boundary(input: &str, start: usize, mut end: usize) -> usize { while end > start && !input.is_char_boundary(end) { end -= 1; } end }
fn char_len(input: &[u8], pos: usize) -> usize { core::str::from_utf8(&input[pos..]).ok().and_then(|s| s.chars().next()).map_or(1, char::len_utf8) }
fn is_word_char(c: u8) -> bool { c.is_ascii_alphanumeric() || c == b'-' }
fn skip_layout(input: &[u8], mut pos: usize) -> usize { while pos < input.len() { match input[pos] { b' ' | b'\t' => pos += 1, b'#' => { pos += 1; while pos < input.len() && !matches!(input[pos], b'\r' | b'\n') { pos += 1; } }, _ => break } } pos }
fn quoted_end(input: &[u8], pos: usize, terminator: u8) -> usize { let mut scan = pos + 1; while scan < input.len() { let c = input[scan]; scan += 1; if c == b'\\' && scan < input.len() { scan += 1; } else if c == terminator { break; } } scan }
fn braced_end(input: &[u8], pos: usize) -> usize { let mut scan = pos + 1; while scan < input.len() { if input[scan] == b'}' { return scan + 1; } scan += 1; } scan }
fn rule_ref_end(input: &[u8], pos: usize) -> usize { if input.get(pos..pos + 2) != Some(b"<[") { return pos; } let mut scan = pos + 2; while scan < input.len() && input[scan].is_ascii_digit() { scan += 1; } if input.get(scan..scan + 2) == Some(b"]>") { scan + 2 } else { pos } }
