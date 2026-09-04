//! Source-aligned synchronous [`TextJinjaParserLexer`] actor.

#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    // Token names and scanner size preserve the pinned generated API shape.
    clippy::enum_variant_names,
    clippy::too_many_lines,
    dead_code,
    missing_docs
)]

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenKind {
    #[default]
    Eof = 0,
    Text,
    NumericLiteral,
    StringLiteral,
    Identifier,
    Equals,
    OpenParen,
    CloseParen,
    OpenStatement,
    CloseStatement,
    OpenExpression,
    CloseExpression,
    OpenSquareBracket,
    CloseSquareBracket,
    OpenCurlyBracket,
    CloseCurlyBracket,
    Comma,
    Dot,
    Colon,
    Pipe,
    CallOperator,
    AdditiveBinaryOperator,
    MultiplicativeBinaryOperator,
    ComparisonBinaryOperator,
    UnaryOperator,
    Comment,
}

/// Maximum bytes retained from one lexer token value.
pub const MAX_TOKEN_VALUE: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cursor<'a> {
    pub source: &'a str,
    pub offset: usize,
    pub token_index: usize,
    pub curly_bracket_depth: usize,
    pub last_token_type: TokenKind,
    pub last_block_rstrip: bool,
    pub last_block_can_trim_newline: bool,
}
impl Default for Cursor<'_> {
    fn default() -> Self {
        Self {
            source: "",
            offset: 0,
            token_index: 0,
            curly_bracket_depth: 0,
            last_token_type: TokenKind::CloseStatement,
            last_block_rstrip: false,
            last_block_can_trim_newline: false,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub value: [u8; MAX_TOKEN_VALUE],
    pub value_len: u16,
    /// Whether the complete source value fit in the bounded storage.
    pub value_valid: bool,
    pub pos: usize,
}
impl Default for Token {
    fn default() -> Self {
        Self {
            kind: TokenKind::Eof,
            value: [0; MAX_TOKEN_VALUE],
            value_len: 0,
            value_valid: true,
            pos: 0,
        }
    }
}
impl Token {
    #[must_use]
    pub fn new(kind: TokenKind, value: &[u8], pos: usize) -> Self {
        let copied = value.len().min(MAX_TOKEN_VALUE);
        let mut token = Self {
            kind,
            value_len: u16::try_from(copied).expect("bounded token value fits u16"),
            value_valid: value.len() <= MAX_TOKEN_VALUE,
            pos,
            ..Self::default()
        };
        token.value[..copied].copy_from_slice(&value[..copied]);
        token
    }
    #[must_use]
    pub const fn kind(kind: TokenKind, pos: usize) -> Self {
        Self {
            kind,
            value: [0; MAX_TOKEN_VALUE],
            value_len: 0,
            value_valid: true,
            pos,
        }
    }
    #[must_use]
    pub fn value(&self) -> &[u8] {
        &self.value[..self.value_len as usize]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NextDone<'a> {
    pub token: Token,
    pub has_token: bool,
    pub next_cursor: Cursor<'a>,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i32)]
pub enum Error {
    #[default]
    None = 0,
    InvalidRequest = 1,
    ParseFailed = 2,
    InternalError = 4,
    Untracked = 8,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NextError {
    pub err: Error,
    pub error_pos: usize,
}
pub type DoneCallback<'a> = dyn FnMut(NextDone<'a>) -> bool + 'a;
pub type ErrorCallback<'a> = dyn FnMut(NextError) -> bool + 'a;

/// Bounded request copied from the pinned `lexer::event::next` contract.
pub struct EventNextRuntime<'a, 'cb> {
    pub cursor: Cursor<'a>,
    pub dispatch_done: Option<&'cb mut DoneCallback<'a>>,
    pub dispatch_error: Option<&'cb mut ErrorCallback<'cb>>,
}
impl<'a, 'cb> EventNextRuntime<'a, 'cb> {
    pub fn new(
        cursor: Cursor<'a>,
        done: &'cb mut DoneCallback<'a>,
        error: &'cb mut ErrorCallback<'cb>,
    ) -> Self {
        Self {
            cursor,
            dispatch_done: Some(done),
            dispatch_error: Some(error),
        }
    }
    pub fn with_callbacks(
        cursor: Cursor<'a>,
        done: Option<&'cb mut DoneCallback<'a>>,
        error: Option<&'cb mut ErrorCallback<'cb>>,
    ) -> Self {
        Self {
            cursor,
            dispatch_done: done,
            dispatch_error: error,
        }
    }
}
impl core::fmt::Debug for EventNextRuntime<'_, '_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EventNextRuntime")
            .field("cursor", &self.cursor)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TextJinjaParserLexerStates {
    #[default]
    Initialized,
    Scanning,
    TextBoundaryCandidateDecision,
    TextScanExec,
    TextOpeningBlockDecision,
    TextTrimOpeningBlockExec,
    TextTrimOpeningBlockResultDecision,
    TextMaterializeExec,
    TextFinalizeExec,
    TextFinalizeResultDecision,
    TextFinalizeTokenExec,
    TextEmitResultDecision,
    CommentCandidateDecision,
    CommentScanExec,
    CommentScanResultDecision,
    CommentFinalizeExec,
    CommentFinalizeResultDecision,
    CommentUnterminatedExec,
    CommentUnterminatedResultDecision,
    TrimPrefixScanExec,
    TrimPrefixEofExec,
    SpaceScanExec,
    SpaceEofExec,
    UnaryCandidateDecision,
    UnaryPrefixContextDecision,
    UnaryPrefixAllowedDecision,
    UnaryScanExec,
    MappingCandidateDecision,
    MappingCloseCurlyExec,
    MappingScanExec,
    StringScanExec,
    StringContentScanExec,
    StringContentPolicyDecision,
    StringScanResultDecision,
    StringMaterializeExec,
    StringStatusDecision,
    StringFinalizeExec,
    StringFinalizeResultDecision,
    StringUnterminatedExec,
    StringUnterminatedResultDecision,
    NumericScanExec,
    WordScanExec,
    InvalidCharExec,
    InvalidCharResultDecision,
}

/// Action context for the lexer state machine.
///
/// The request cursor is the sole owner of template source and persistent
/// cursor bookkeeping.  The pinned parser lexer action context is empty, so
/// this context must not duplicate or otherwise retain source storage.
#[derive(Debug, Default)]
pub struct TextJinjaParserLexerContext;
#[derive(Debug, Default)]
pub struct TextJinjaParserLexerStateMachine {
    state: TextJinjaParserLexerStates,
    context: TextJinjaParserLexerContext,
}
pub type TextJinjaParserLexer = TextJinjaParserLexerStateMachine;
pub type Lexer = TextJinjaParserLexerStateMachine;

impl TextJinjaParserLexerStateMachine {
    #[must_use]
    pub const fn new(context: TextJinjaParserLexerContext) -> Self {
        Self {
            state: TextJinjaParserLexerStates::Initialized,
            context,
        }
    }
    #[must_use]
    pub const fn state(&self) -> TextJinjaParserLexerStates {
        self.state
    }
    #[must_use]
    pub fn is(&self, state: TextJinjaParserLexerStates) -> bool {
        self.state == state
    }
    #[must_use]
    pub const fn context(&self) -> &TextJinjaParserLexerContext {
        &self.context
    }
    pub fn process_event(&mut self, mut event: EventNextRuntime<'_, '_>) -> bool {
        if event.dispatch_done.is_none() || event.dispatch_error.is_none() {
            emit_error(
                &mut event.dispatch_error,
                Error::InvalidRequest,
                event.cursor.offset,
            );
            return true;
        }
        if event.cursor.offset > event.cursor.source.len()
            || !event.cursor.source.is_char_boundary(event.cursor.offset)
        {
            emit_error(
                &mut event.dispatch_error,
                Error::InvalidRequest,
                event.cursor.offset,
            );
            return true;
        }
        self.state = TextJinjaParserLexerStates::Scanning;
        scan(event);
        true
    }
    pub fn process_unexpected_event<'cb>(
        &mut self,
        mut error: Option<&'cb mut ErrorCallback<'cb>>,
        pos: usize,
    ) -> bool {
        emit_error(&mut error, Error::InternalError, pos);
        self.state = TextJinjaParserLexerStates::Scanning;
        true
    }
}

fn scan(mut event: EventNextRuntime<'_, '_>) {
    let c = event.cursor;
    let src = c.source;
    let mut pos = c.offset;
    let b = src.as_bytes();
    if pos == src.len() {
        emit_done(event.dispatch_done, eof(c, pos));
        return;
    }
    if text_boundary(c.last_token_type) {
        let end = find_boundary(src, pos);
        if end > pos {
            let mut start = pos;
            let mut finish = end;
            if c.last_block_rstrip {
                while start < finish && trim_char(b[start]) {
                    start += 1;
                }
            }
            if end < src.len() && trim_open(src, end) {
                while finish > start && trim_char(b[finish - 1]) {
                    finish -= 1;
                }
            }
            if c.last_block_can_trim_newline && b.get(start) == Some(&b'\n') {
                start += 1;
            }
            if start < finish {
                emit_token(
                    event.dispatch_done,
                    c,
                    Token::new(TokenKind::Text, &b[start..finish], start),
                    end,
                    false,
                    false,
                );
                return;
            }
            pos = end;
        }
    }
    if pos >= src.len() {
        emit_done(event.dispatch_done, eof(c, pos));
        return;
    }
    if (c.last_token_type == TokenKind::OpenStatement
        || c.last_token_type == TokenKind::OpenExpression)
        && b[pos] == b'-'
    {
        pos += 1;
    }
    while pos < b.len() && space(b[pos]) {
        pos += 1;
    }
    if pos >= src.len() {
        emit_done(event.dispatch_done, eof(c, pos));
        return;
    }
    if b[pos] == b'{' && at(b, pos + 1) == Some(b'#') {
        let start = pos;
        let content = pos + 2;
        let close = src[content..].find("#}").map_or(src.len(), |n| content + n);
        if close + 1 >= src.len() {
            emit_error(&mut event.dispatch_error, Error::ParseFailed, close);
            return;
        }
        emit_token(
            event.dispatch_done,
            c,
            Token::new(TokenKind::Comment, &b[content..close], start),
            close + 2,
            false,
            false,
        );
        return;
    }
    if let Some((kind, width, rstrip, trim)) = mapping(src, pos, c.curly_bracket_depth) {
        emit_token_with_flags(
            event.dispatch_done,
            c,
            Token::new(kind, &b[pos..pos + width], pos),
            pos + width,
            rstrip,
            trim,
        );
        return;
    }
    let unary =
        b[pos] == b'+' || (b[pos] == b'-' && at(b, pos + 1).is_none_or(|x| x != b'%' && x != b'}'));
    if unary {
        if !unary_allowed(c.last_token_type) {
            emit_error(&mut event.dispatch_error, Error::ParseFailed, pos);
            return;
        }
        let start = pos;
        pos += 1;
        let nstart = pos;
        consume_number(b, &mut pos);
        let kind = if pos > nstart {
            TokenKind::NumericLiteral
        } else {
            TokenKind::UnaryOperator
        };
        emit_token(
            event.dispatch_done,
            c,
            Token::new(kind, &b[start..pos], start),
            pos,
            false,
            false,
        );
        return;
    }
    if b[pos] == b'\'' || b[pos] == b'"' {
        let start = pos;
        let term = b[pos];
        pos += 1;
        let mut value = [0_u8; MAX_TOKEN_VALUE];
        let mut value_len = 0_usize;
        while pos < b.len() && b[pos] != term {
            if b[pos] == b'\\' {
                pos += 1;
                if pos >= b.len() {
                    emit_error(&mut event.dispatch_error, Error::ParseFailed, pos);
                    return;
                }
                let ch = match b[pos] {
                    b'n' => b'\n',
                    b't' => b'\t',
                    b'r' => b'\r',
                    b'b' => b'\x08',
                    b'f' => b'\x0c',
                    b'v' => b'\x0b',
                    b'\\' => b'\\',
                    b'\'' => b'\'',
                    b'"' => b'"',
                    _ => {
                        emit_error(&mut event.dispatch_error, Error::ParseFailed, pos);
                        return;
                    }
                };
                if value_len == MAX_TOKEN_VALUE {
                    emit_error(&mut event.dispatch_error, Error::InvalidRequest, pos);
                    return;
                }
                value[value_len] = ch;
                value_len += 1;
                pos += 1;
            } else {
                let width = if b[pos] < 0x80 {
                    1
                } else if b[pos] < 0xE0 {
                    2
                } else if b[pos] < 0xF0 {
                    3
                } else {
                    4
                };
                if pos + width > b.len() {
                    emit_error(&mut event.dispatch_error, Error::ParseFailed, pos);
                    return;
                }
                if value_len + width > MAX_TOKEN_VALUE {
                    emit_error(&mut event.dispatch_error, Error::InvalidRequest, pos);
                    return;
                }
                let Ok(v) = core::str::from_utf8(&b[pos..pos + width]) else {
                    emit_error(&mut event.dispatch_error, Error::ParseFailed, pos);
                    return;
                };
                value[value_len..value_len + width].copy_from_slice(v.as_bytes());
                value_len += width;
                pos += width;
            }
        }
        if pos >= b.len() {
            emit_error(&mut event.dispatch_error, Error::ParseFailed, pos);
            return;
        }
        pos += 1;
        emit_token(
            event.dispatch_done,
            c,
            Token::new(TokenKind::StringLiteral, &value[..value_len], start),
            pos,
            false,
            false,
        );
        return;
    }
    if b[pos].is_ascii_digit() {
        let start = pos;
        consume_number(b, &mut pos);
        emit_token(
            event.dispatch_done,
            c,
            Token::new(TokenKind::NumericLiteral, &b[start..pos], start),
            pos,
            false,
            false,
        );
        return;
    }
    if word(b[pos]) {
        let start = pos;
        pos += 1;
        while pos < b.len() && word(b[pos]) {
            pos += 1;
        }
        emit_token(
            event.dispatch_done,
            c,
            Token::new(TokenKind::Identifier, &b[start..pos], start),
            pos,
            false,
            false,
        );
        return;
    }
    emit_error(&mut event.dispatch_error, Error::ParseFailed, pos);
}
fn emit_done<'a>(mut cb: Option<&mut DoneCallback<'a>>, done: NextDone<'a>) {
    if let Some(f) = cb.as_mut() {
        let _ = f(done);
    }
}
fn emit_error<'cb>(cb: &mut Option<&'cb mut ErrorCallback<'cb>>, err: Error, pos: usize) {
    if let Some(f) = cb.as_mut() {
        let _ = f(NextError {
            err,
            error_pos: pos,
        });
    }
}
const fn eof(mut c: Cursor<'_>, pos: usize) -> NextDone<'_> {
    c.offset = pos;
    NextDone {
        token: Token::kind(TokenKind::Eof, pos),
        has_token: false,
        next_cursor: c,
    }
}
fn emit_token<'a>(
    cb: Option<&mut DoneCallback<'a>>,
    c: Cursor<'a>,
    tok: Token,
    end: usize,
    r: bool,
    t: bool,
) {
    emit_token_with_flags(cb, c, tok, end, r, t);
}
fn emit_token_with_flags<'a>(
    cb: Option<&mut DoneCallback<'a>>,
    c: Cursor<'a>,
    tok: Token,
    end: usize,
    r: bool,
    t: bool,
) {
    let mut n = c;
    n.offset = end;
    n.token_index = c.token_index.saturating_add(1);
    n.last_token_type = tok.kind;
    n.last_block_can_trim_newline = t || tok.kind == TokenKind::Comment;
    n.last_block_rstrip = r;
    if tok.kind == TokenKind::OpenExpression {
        n.curly_bracket_depth = 0;
    } else if tok.kind == TokenKind::OpenCurlyBracket {
        n.curly_bracket_depth = c.curly_bracket_depth.saturating_add(1);
    } else if tok.kind == TokenKind::CloseCurlyBracket {
        n.curly_bracket_depth = c.curly_bracket_depth.saturating_sub(1);
    }
    emit_done(
        cb,
        NextDone {
            token: tok,
            has_token: true,
            next_cursor: n,
        },
    );
}
fn at(b: &[u8], p: usize) -> Option<u8> {
    b.get(p).copied()
}
const fn word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}
const fn trim_char(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\r' | b'\n')
}
const fn space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\r' | b'\n' | b'\x0c' | b'\x0b')
}
const fn text_boundary(k: TokenKind) -> bool {
    matches!(
        k,
        TokenKind::CloseStatement | TokenKind::CloseExpression | TokenKind::Comment
    )
}
fn trim_open(s: &str, p: usize) -> bool {
    at(s.as_bytes(), p + 2) == Some(b'-')
}
const fn find_boundary(s: &str, mut p: usize) -> usize {
    let b = s.as_bytes();
    while p + 1 < b.len() {
        if b[p] == b'{' && matches!(b[p + 1], b'%' | b'{' | b'#') {
            return p;
        }
        p += 1;
    }
    b.len()
}
fn consume_number(b: &[u8], p: &mut usize) {
    while *p < b.len() && b[*p].is_ascii_digit() {
        *p += 1;
    }
    if *p < b.len() && b[*p] == b'.' && b.get(*p + 1).is_some_and(u8::is_ascii_digit) {
        *p += 1;
        while *p < b.len() && b[*p].is_ascii_digit() {
            *p += 1;
        }
    }
}
const fn unary_allowed(k: TokenKind) -> bool {
    matches!(
        k,
        TokenKind::OpenExpression
            | TokenKind::OpenStatement
            | TokenKind::OpenParen
            | TokenKind::Comma
            | TokenKind::Colon
            | TokenKind::Pipe
            | TokenKind::AdditiveBinaryOperator
            | TokenKind::MultiplicativeBinaryOperator
            | TokenKind::ComparisonBinaryOperator
            | TokenKind::Equals
            | TokenKind::UnaryOperator
    )
}
fn mapping(s: &str, p: usize, depth: usize) -> Option<(TokenKind, usize, bool, bool)> {
    let r = &s[p..];
    let a: [(&str, TokenKind, bool, bool); 31] = [
        ("{%-", TokenKind::OpenStatement, false, false),
        ("-%}", TokenKind::CloseStatement, true, true),
        ("{{-", TokenKind::OpenExpression, false, false),
        ("-}}", TokenKind::CloseExpression, true, true),
        ("{%", TokenKind::OpenStatement, false, false),
        ("%}", TokenKind::CloseStatement, false, true),
        ("{{", TokenKind::OpenExpression, false, false),
        ("}}", TokenKind::CloseExpression, false, true),
        ("<=", TokenKind::ComparisonBinaryOperator, false, false),
        (">=", TokenKind::ComparisonBinaryOperator, false, false),
        ("==", TokenKind::ComparisonBinaryOperator, false, false),
        ("!=", TokenKind::ComparisonBinaryOperator, false, false),
        ("(", TokenKind::OpenParen, false, false),
        (")", TokenKind::CloseParen, false, false),
        ("[", TokenKind::OpenSquareBracket, false, false),
        ("]", TokenKind::CloseSquareBracket, false, false),
        ("{", TokenKind::OpenCurlyBracket, false, false),
        ("}", TokenKind::CloseCurlyBracket, false, false),
        (",", TokenKind::Comma, false, false),
        (".", TokenKind::Dot, false, false),
        (":", TokenKind::Colon, false, false),
        ("|", TokenKind::Pipe, false, false),
        ("<", TokenKind::ComparisonBinaryOperator, false, false),
        (">", TokenKind::ComparisonBinaryOperator, false, false),
        ("+", TokenKind::AdditiveBinaryOperator, false, false),
        ("-", TokenKind::AdditiveBinaryOperator, false, false),
        ("~", TokenKind::AdditiveBinaryOperator, false, false),
        ("*", TokenKind::MultiplicativeBinaryOperator, false, false),
        ("/", TokenKind::MultiplicativeBinaryOperator, false, false),
        ("%", TokenKind::MultiplicativeBinaryOperator, false, false),
        ("=", TokenKind::Equals, false, false),
    ];
    for &(x, k, rstrip, trim) in &a {
        if r.starts_with(x) {
            if k == TokenKind::CloseExpression && depth > 0 {
                continue;
            }
            return Some((k, x.len(), rstrip, trim));
        }
    }
    None
}

#[allow(unused_imports)]
pub mod event {
    pub use super::{
        Cursor, DoneCallback, Error, ErrorCallback, EventNextRuntime, NextDone, NextError, Token,
        TokenKind,
    };
}
#[allow(unused_imports)]
pub mod events {
    pub use super::{NextDone, NextError};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_cursor_offset_inside_utf8_character() {
        let mut reported = None;
        let mut done = |_event: NextDone<'_>| true;
        let mut error = |event: NextError| {
            reported = Some(event);
            true
        };
        let cursor = Cursor {
            source: "é",
            offset: 1,
            ..Cursor::default()
        };
        let request = EventNextRuntime::new(cursor, &mut done, &mut error);
        let mut lexer = TextJinjaParserLexerStateMachine::new(TextJinjaParserLexerContext);
        assert!(lexer.process_event(request));
        assert_eq!(
            reported,
            Some(NextError {
                err: Error::InvalidRequest,
                error_pos: 1
            })
        );
        assert_eq!(lexer.state(), TextJinjaParserLexerStates::Initialized);
    }

    #[test]
    fn marks_values_larger_than_capacity_invalid() {
        let bytes = [b'a'; MAX_TOKEN_VALUE + 1];
        let token = Token::new(TokenKind::Identifier, &bytes, 7);
        assert_eq!(token.value().len(), MAX_TOKEN_VALUE);
        assert!(!token.value_valid);
        assert_eq!(token.pos, 7);
    }

    #[test]
    fn preserves_decoded_string_bytes_in_bounded_token() {
        let mut done_token = None;
        let mut done = |event: NextDone<'_>| {
            done_token = Some(event.token);
            true
        };
        let mut error = |_event: NextError| true;
        let mut lexer = TextJinjaParserLexerStateMachine::new(TextJinjaParserLexerContext);
        let source = "{{ 'it\\'s' }}";
        let cursor = Cursor {
            source,
            ..Cursor::default()
        };
        assert!(lexer.process_event(EventNextRuntime::new(cursor, &mut done, &mut error)));
        let token = done_token.expect("lexer emits one token");
        assert_eq!(token.kind, TokenKind::OpenExpression);
        assert_eq!(token.value(), b"{{");
    }

    #[test]
    fn context_has_no_source_storage_and_scan_advances_cursor() {
        assert_eq!(core::mem::size_of::<TextJinjaParserLexerContext>(), 0);

        let source = "hello {{name";
        let mut lexer = TextJinjaParserLexerStateMachine::new(TextJinjaParserLexerContext);
        let mut first = None;
        let mut done = |event: NextDone<'_>| {
            let token = event.token;
            let offset = event.next_cursor.offset;
            let token_index = event.next_cursor.token_index;
            first = Some((token, offset, token_index));
            true
        };
        let mut error = |_event: NextError| true;
        let cursor = Cursor {
            source,
            ..Cursor::default()
        };
        assert!(lexer.process_event(EventNextRuntime::new(cursor, &mut done, &mut error)));

        let (first_token, first_offset, first_token_index) =
            first.expect("lexer emits text before the opening expression");
        assert_eq!(first_token.kind, TokenKind::Text);
        assert_eq!(first_token.value(), b"hello ");
        assert_eq!(first_offset, 6);
        assert_eq!(first_token_index, 1);
        let first_cursor = Cursor {
            source,
            offset: first_offset,
            token_index: first_token_index,
            ..cursor
        };

        let mut second = None;
        let mut done = |event: NextDone<'_>| {
            let token = event.token;
            let offset = event.next_cursor.offset;
            let token_index = event.next_cursor.token_index;
            second = Some((token, offset, token_index));
            true
        };
        assert!(lexer.process_event(EventNextRuntime::new(first_cursor, &mut done, &mut error,)));

        let (second_token, second_offset, second_token_index) =
            second.expect("lexer emits the opening expression");
        assert_eq!(second_token.kind, TokenKind::OpenExpression);
        assert_eq!(second_token.value(), b"{{");
        assert_eq!(second_offset, 8);
        assert_eq!(second_token_index, 2);
    }
}
