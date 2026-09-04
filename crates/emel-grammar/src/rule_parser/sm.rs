//! Bounded, synchronous GBNF rule parser.
//!
//! The root actor mirrors the pinned rule-parser phases while keeping all
//! storage inline. Lexing and token classification are delegated to sibling
//! child actors; one dispatch runs to completion without queues.

#![allow(
    dead_code,
    missing_docs,
    unused_imports,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::large_stack_arrays,
    clippy::large_stack_frames,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_ref_mut,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref
)]

use super::lexer::sm::GbnfRuleParserLexerStateMachine;
use super::lexer::{Cursor, Error as LexerError, EventScanNext, NextDone, NextError, TokenKind};
use super::nonterm_parser::sm::ParseMode;
use crate::gbnf::{
    element, element_type, grammar, k_max_gbnf_elements, k_max_gbnf_rule_elements, k_max_gbnf_rules,
};

/// Maximum source bytes retained by one dispatch.
pub const MAX_RULE_SOURCE_BYTES: usize = 65_536;
/// Maximum nested groups accepted by the pinned parser.
pub const MAX_GROUP_NESTING_DEPTH: usize = 32;
const MAX_SYMBOL_NAME_BYTES: usize = 128;
const MAX_REPETITION: usize = 2000;
const MAX_SYMBOLS: usize = 2048;

/// Root parser request over caller-owned grammar storage.
#[derive(Debug)]
pub struct EventParseRules<'source, 'output> {
    /// Grammar source text copied into the bounded parser context.
    pub source: &'source str,
    /// Caller-owned destination populated on success and reset on failure.
    pub grammar_out: &'output mut grammar,
    /// Synchronous successful completion callback.
    pub on_done: Option<DoneCallback>,
    /// Synchronous failed completion callback.
    pub on_error: Option<ErrorCallback>,
}

/// Successful root parser completion.
#[derive(Clone, Copy, Debug)]
pub struct ParseDone<'a> {
    /// Caller-owned grammar populated by this dispatch.
    pub grammar: &'a grammar,
}

/// Failed root parser completion.
#[derive(Clone, Copy, Debug)]
pub struct ParseError<'a> {
    /// Caller-owned grammar, reset before this callback is invoked.
    pub grammar: &'a grammar,
    /// Explicit typed parser outcome.
    pub error: ParseOutcome,
}

/// Synchronous successful completion callback.
pub type DoneCallback = for<'a> fn(ParseDone<'a>) -> bool;
/// Synchronous failed completion callback.
pub type ErrorCallback = for<'a> fn(ParseError<'a>) -> bool;

fn ignore_parse_done(_: ParseDone<'_>) -> bool {
    true
}

fn ignore_parse_error(_: ParseError<'_>) -> bool {
    true
}

impl<'source, 'output> EventParseRules<'source, 'output> {
    /// Creates a request with no-op callbacks, suitable for direct parsing.
    #[must_use]
    pub fn new(source: &'source str, grammar_out: &'output mut grammar) -> Self {
        Self {
            source,
            grammar_out,
            on_done: Some(ignore_parse_done),
            on_error: Some(ignore_parse_error),
        }
    }

    /// Replaces the synchronous completion callbacks.
    #[must_use]
    pub fn with_callbacks(mut self, on_done: DoneCallback, on_error: ErrorCallback) -> Self {
        self.on_done = Some(on_done);
        self.on_error = Some(on_error);
        self
    }
}

/// Bounded result of one root parser dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseOutcome {
    Parsed,
    InvalidRequest,
    ParseFailed,
    InternalError,
    Unexpected,
}

/// Generated topology state inspection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GbnfRuleParserStates {
    Ready,
    ExpectRuleName,
    ExpectRuleNameDecision,
    ExpectDefinition,
    ExpectDefinitionDecision,
    InRuleExpressionNeedTerm,
    InRuleExpressionNeedTermDecision,
    InRuleExpressionAfterTerm,
    InRuleExpressionAfterTermDecision,
    RuleReferenceDecision,
    RuleReferencePlainExec,
    RuleReferenceNegatedExec,
    QuantifierDecision,
    QuantifierStarExec,
    QuantifierPlusExec,
    QuantifierQuestionExec,
    QuantifierBracedExactExec,
    QuantifierBracedOpenExec,
    QuantifierBracedRangeExec,
    EofSymbolsDecision,
    ParseDecision,
    UnexpectedEvent,
}

#[derive(Clone, Copy, Debug, Default)]
struct GroupFrame {
    sequence_start: u16,
    generated_rule_id: u16,
}

#[derive(Clone, Copy, Debug)]
struct SymbolEntry {
    hash: u32,
    id: u32,
    len: u16,
    bytes: [u8; MAX_SYMBOL_NAME_BYTES],
    occupied: bool,
    defined: bool,
}
impl Default for SymbolEntry {
    fn default() -> Self {
        Self {
            hash: 0,
            id: 0,
            len: 0,
            bytes: [0; MAX_SYMBOL_NAME_BYTES],
            occupied: false,
            defined: false,
        }
    }
}

/// Context retained by the bounded root actor.
#[derive(Debug)]
pub struct GbnfRuleParserContext {
    source: [u8; MAX_RULE_SOURCE_BYTES],
    source_len: usize,
    cursor_offset: u32,
    cursor_token_count: u32,
    token_kind: TokenKind,
    token_start: usize,
    token_end: usize,
    has_token: bool,
    current_rule: [element; k_max_gbnf_rule_elements],
    repeat_scratch: [element; k_max_gbnf_rule_elements],
    current_rule_size: usize,
    current_rule_id: u32,
    last_sym_start: usize,
    groups: [GroupFrame; MAX_GROUP_NESTING_DEPTH],
    group_depth: usize,
    symbols: [SymbolEntry; MAX_SYMBOLS],
    symbol_count: usize,
    rule_defined: [bool; k_max_gbnf_rules],
    next_symbol_id: u32,
}

impl Default for GbnfRuleParserContext {
    fn default() -> Self {
        Self {
            source: [0; MAX_RULE_SOURCE_BYTES],
            source_len: 0,
            cursor_offset: 0,
            cursor_token_count: 0,
            token_kind: TokenKind::Unknown,
            token_start: 0,
            token_end: 0,
            has_token: false,
            current_rule: [element::default(); k_max_gbnf_rule_elements],
            repeat_scratch: [element::default(); k_max_gbnf_rule_elements],
            current_rule_size: 0,
            current_rule_id: 0,
            last_sym_start: 0,
            groups: [GroupFrame::default(); MAX_GROUP_NESTING_DEPTH],
            group_depth: 0,
            symbols: [SymbolEntry::default(); MAX_SYMBOLS],
            symbol_count: 0,
            rule_defined: [false; k_max_gbnf_rules],
            next_symbol_id: 0,
        }
    }
}

impl GbnfRuleParserContext {
    fn reset(&mut self, source: &str) -> bool {
        if source.is_empty() || source.len() > MAX_RULE_SOURCE_BYTES {
            return false;
        }
        self.source[..source.len()].copy_from_slice(source.as_bytes());
        self.source_len = source.len();
        self.cursor_offset = 0;
        self.cursor_token_count = 0;
        self.token_kind = TokenKind::Unknown;
        self.token_start = 0;
        self.token_end = 0;
        self.has_token = false;
        self.current_rule_size = 0;
        self.current_rule_id = 0;
        self.last_sym_start = 0;
        self.group_depth = 0;
        self.symbols.fill(SymbolEntry::default());
        self.symbol_count = 0;
        self.rule_defined.fill(false);
        self.next_symbol_id = 0;
        true
    }
    fn source(&self) -> &str {
        core::str::from_utf8(&self.source[..self.source_len]).unwrap_or("")
    }
    fn text(&self) -> &str {
        self.source()
            .get(self.token_start..self.token_end)
            .unwrap_or("")
    }
    fn push(&mut self, item: element) -> bool {
        if self.current_rule_size >= k_max_gbnf_rule_elements {
            false
        } else {
            self.current_rule[self.current_rule_size] = item;
            self.current_rule_size += 1;
            true
        }
    }
    fn hash(name: &[u8]) -> u32 {
        let mut h = 2_166_136_261u32;
        for b in name {
            h = (h ^ u32::from(*b)).wrapping_mul(16_777_619);
        }
        if h == 0 { 1 } else { h }
    }
    fn symbol(&mut self, name: &[u8], definition: bool) -> Option<u32> {
        if name.is_empty() || name.len() > MAX_SYMBOL_NAME_BYTES {
            return None;
        }
        let hash = Self::hash(name);
        for entry in &mut self.symbols[..self.symbol_count] {
            if entry.occupied
                && entry.hash == hash
                && entry.len as usize == name.len()
                && entry.bytes[..name.len()] == *name
            {
                if definition && entry.defined {
                    return None;
                }
                entry.defined |= definition;
                self.rule_defined[entry.id as usize] |= definition;
                return Some(entry.id);
            }
        }
        if self.symbol_count >= MAX_SYMBOLS || self.next_symbol_id as usize >= k_max_gbnf_rules {
            return None;
        }
        let id = self.next_symbol_id;
        self.next_symbol_id += 1;
        let entry = &mut self.symbols[self.symbol_count];
        entry.hash = hash;
        entry.id = id;
        entry.len = name.len() as u16;
        entry.bytes[..name.len()].copy_from_slice(name);
        entry.occupied = true;
        entry.defined = definition;
        self.rule_defined[id as usize] = definition;
        self.symbol_count += 1;
        Some(id)
    }
    fn append_rule(&mut self, out: &mut grammar, id: u32) -> bool {
        if id as usize >= k_max_gbnf_rules
            || self.current_rule_size == 0
            || out.rule_lengths[id as usize] != 0
        {
            return false;
        }
        let offset = out.element_count as usize;
        let Some(end) = offset.checked_add(self.current_rule_size) else {
            return false;
        };
        if end > k_max_gbnf_elements {
            return false;
        }
        out.elements[offset..end].copy_from_slice(&self.current_rule[..self.current_rule_size]);
        out.rule_offsets[id as usize] = offset as u32;
        out.rule_lengths[id as usize] = self.current_rule_size as u32;
        out.element_count = end as u32;
        out.rule_count = out.rule_count.max(id + 1);
        true
    }
    fn finalize(&mut self, out: &mut grammar) -> bool {
        if self.group_depth != 0
            || self.current_rule_size == 0
            || !self.push(element {
                r#type: element_type::end,
                value: 0,
            })
        {
            return false;
        }
        let ok = self.append_rule(out, self.current_rule_id);
        self.current_rule_size = 0;
        self.last_sym_start = 0;
        ok
    }
}
/// Synchronous bounded root parser actor.
#[derive(Debug)]
pub struct GbnfRuleParserActor {
    state: GbnfRuleParserStates,
    context: Box<GbnfRuleParserContext>,
    lexer: GbnfRuleParserLexerStateMachine,
}
impl Default for GbnfRuleParserActor {
    fn default() -> Self {
        Self::new()
    }
}
impl GbnfRuleParserActor {
    /// Creates an actor in the generated ready state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: GbnfRuleParserStates::Ready,
            context: Box::new(GbnfRuleParserContext::default()),
            lexer: GbnfRuleParserLexerStateMachine::default(),
        }
    }
    /// Parses one source in a single run-to-completion dispatch.
    pub fn process_event(&mut self, mut event: EventParseRules<'_, '_>) -> ParseOutcome {
        if self.state != GbnfRuleParserStates::Ready {
            return self.publish_error(&mut event, ParseOutcome::InternalError);
        }
        if event.on_done.is_none() || event.on_error.is_none() {
            event.grammar_out.reset();
            if let Some(callback) = event.on_error {
                let _ = callback(ParseError {
                    grammar: event.grammar_out,
                    error: ParseOutcome::InvalidRequest,
                });
            }
            return ParseOutcome::InvalidRequest;
        }
        if !self.context.reset(event.source) {
            return self.publish_error(&mut event, ParseOutcome::InvalidRequest);
        }
        event.grammar_out.reset();
        self.state = GbnfRuleParserStates::ExpectRuleName;
        let outcome = self.run(event.grammar_out);
        self.state = GbnfRuleParserStates::Ready;
        if outcome == ParseOutcome::Parsed {
            if let Some(callback) = event.on_done {
                let _ = callback(ParseDone {
                    grammar: event.grammar_out,
                });
            }
        } else {
            event.grammar_out.reset();
            if let Some(callback) = event.on_error {
                let _ = callback(ParseError {
                    grammar: event.grammar_out,
                    error: outcome,
                });
            }
        }
        outcome
    }
    fn publish_error(
        &mut self,
        event: &mut EventParseRules<'_, '_>,
        outcome: ParseOutcome,
    ) -> ParseOutcome {
        event.grammar_out.reset();
        if let Some(callback) = event.on_error {
            let _ = callback(ParseError {
                grammar: event.grammar_out,
                error: outcome,
            });
        }
        self.state = GbnfRuleParserStates::Ready;
        outcome
    }
    /// Convenience parser over caller-owned output.
    pub fn parse(&mut self, source: &str, grammar_out: &mut grammar) -> ParseOutcome {
        self.process_event(EventParseRules::new(source, grammar_out))
    }
    /// Returns generated state inspection.
    #[must_use]
    pub fn state(&self) -> &GbnfRuleParserStates {
        &self.state
    }
    /// Tests generated state identity.
    #[must_use]
    pub fn is(&self, state: &GbnfRuleParserStates) -> bool {
        self.state == *state
    }
    /// Reports an explicit unexpected event.
    pub fn process_unexpected_event(&mut self) -> ParseOutcome {
        self.state = GbnfRuleParserStates::Ready;
        ParseOutcome::Unexpected
    }

    fn next_token(&mut self) -> Result<bool, ParseOutcome> {
        let cursor = Cursor {
            input: self.context.source(),
            offset: self.context.cursor_offset,
            token_count: self.context.cursor_token_count,
        };
        let mut got = (
            false,
            TokenKind::Unknown,
            0usize,
            0usize,
            cursor.offset,
            cursor.token_count,
        );
        let mut lexer_error = None;
        let mut done = |event: NextDone<'_>| {
            got = (
                event.has_token,
                event.token.kind,
                event.token.start as usize,
                event.token.end as usize,
                event.next_cursor.offset,
                event.next_cursor.token_count,
            );
            true
        };
        let mut error = |event: NextError| {
            lexer_error = Some(event.err);
            true
        };
        if !self
            .lexer
            .process_event(EventScanNext::new(cursor, &mut done, &mut error))
        {
            return Err(ParseOutcome::InternalError);
        }
        if let Some(error) = lexer_error {
            return Err(match error {
                LexerError::InvalidRequest => ParseOutcome::InvalidRequest,
                LexerError::ParseFailed => ParseOutcome::ParseFailed,
                LexerError::InternalError | LexerError::Untracked | LexerError::None => {
                    ParseOutcome::InternalError
                }
            });
        }
        self.context.has_token = got.0;
        self.context.token_kind = got.1;
        self.context.token_start = got.2;
        self.context.token_end = got.3;
        self.context.cursor_offset = got.4;
        self.context.cursor_token_count = got.5;
        Ok(got.0)
    }
    fn classify_nonterm(&mut self, mode: ParseMode) -> Result<u32, ParseOutcome> {
        let text = self.context.text();
        if text.is_empty() || text.len() > MAX_SYMBOL_NAME_BYTES {
            return Err(ParseOutcome::ParseFailed);
        }
        let mut name = [0u8; MAX_SYMBOL_NAME_BYTES];
        name[..text.len()].copy_from_slice(text.as_bytes());
        self.context
            .symbol(&name[..text.len()], mode == ParseMode::Definition)
            .ok_or(ParseOutcome::ParseFailed)
    }
    fn parse_char(bytes: &[u8], pos: usize, end: usize) -> Option<(u32, usize)> {
        if pos >= end {
            return None;
        }
        if bytes[pos] != b'\\' {
            let s = core::str::from_utf8(&bytes[pos..end]).ok()?;
            let c = s.chars().next()?;
            return Some((c as u32, pos + c.len_utf8()));
        }
        if pos + 1 >= end {
            return None;
        }
        match bytes[pos + 1] {
            b'n' => Some((10, pos + 2)),
            b'r' => Some((13, pos + 2)),
            b't' => Some((9, pos + 2)),
            b'\\' | b'"' | b'[' | b']' => Some((bytes[pos + 1] as u32, pos + 2)),
            b'x' | b'u' | b'U' => {
                let count = if bytes[pos + 1] == b'x' {
                    2
                } else if bytes[pos + 1] == b'u' {
                    4
                } else {
                    8
                };
                if pos + 2 + count > end {
                    return None;
                }
                let mut value = 0u32;
                let mut i = pos + 2;
                while i < pos + 2 + count {
                    let d = match bytes[i] {
                        b'0'..=b'9' => bytes[i] - b'0',
                        b'a'..=b'f' => bytes[i] - b'a' + 10,
                        b'A'..=b'F' => bytes[i] - b'A' + 10,
                        _ => return None,
                    };
                    value = value.checked_mul(16)?.checked_add(u32::from(d))?;
                    i += 1;
                }
                Some((value, pos + 2 + count))
            }
            _ => None,
        }
    }
    fn consume_literal(&mut self) -> bool {
        let len = self.context.text().len();
        let valid = {
            let bytes = self.context.text().as_bytes();
            len >= 2 && bytes[0] == b'"' && bytes[len - 1] == b'"'
        };
        if !valid {
            return false;
        }
        self.context.last_sym_start = self.context.current_rule_size;
        let mut pos = 1;
        while pos + 1 < len {
            let Some((value, next)) = ({
                let bytes = self.context.text().as_bytes();
                Self::parse_char(bytes, pos, len - 1)
            }) else {
                return false;
            };
            if !self.context.push(element {
                r#type: element_type::character,
                value,
            }) {
                return false;
            }
            pos = next;
        }
        pos == len - 1
    }
    fn consume_class(&mut self) -> bool {
        let len = self.context.text().len();
        let valid = {
            let bytes = self.context.text().as_bytes();
            len >= 3 && bytes[0] == b'[' && bytes[len - 1] == b']'
        };
        if !valid {
            return false;
        }
        self.context.last_sym_start = self.context.current_rule_size;
        let end = len - 1;
        let mut pos = 1;
        let negated = {
            let bytes = self.context.text().as_bytes();
            bytes[pos] == b'^'
        };
        if negated {
            pos += 1;
        }
        let mut first = true;
        while pos < end {
            let Some((value, next)) = ({
                let bytes = self.context.text().as_bytes();
                Self::parse_char(bytes, pos, end)
            }) else {
                return false;
            };
            if !self.context.push(element {
                r#type: if first {
                    if negated {
                        element_type::char_not
                    } else {
                        element_type::character
                    }
                } else {
                    element_type::char_alt
                },
                value,
            }) {
                return false;
            }
            first = false;
            pos = next;
            let range = {
                let bytes = self.context.text().as_bytes();
                pos + 1 < end && bytes[pos] == b'-' && bytes[pos + 1] != b']'
            };
            if range {
                pos += 1;
                let Some((upper, next_upper)) = ({
                    let bytes = self.context.text().as_bytes();
                    Self::parse_char(bytes, pos, end)
                }) else {
                    return false;
                };
                if !self.context.push(element {
                    r#type: element_type::char_rng_upper,
                    value: upper,
                }) {
                    return false;
                }
                pos = next_upper;
            }
        }
        !first
    }
    fn parse_rule_reference_digits_text(text: &str) -> Option<u32> {
        if text.is_empty() {
            return None;
        }
        let mut value = 0u32;
        for byte in text.bytes() {
            if !byte.is_ascii_digit() {
                return None;
            }
            value = value
                .checked_mul(10)
                .and_then(|value| value.checked_add(u32::from(byte - b'0')))?;
        }
        Some(value)
    }
    fn consume_rule_reference(&mut self) -> bool {
        let text = self.context.text();
        let bytes = text.as_bytes();
        let negated = bytes.first() == Some(&b'!');
        let (prefix, minimum_len): (&[u8], usize) = if negated {
            (b"!<[" as &[u8], 5)
        } else {
            (b"<[" as &[u8], 4)
        };
        if bytes.len() < minimum_len || !bytes.starts_with(prefix) || !bytes.ends_with(b"]>") {
            return false;
        }
        let digits_start = prefix.len();
        let digits_end = bytes.len() - 2;
        let Some(token_id) = Self::parse_rule_reference_digits_text(
            text.get(digits_start..digits_end).unwrap_or(""),
        ) else {
            return false;
        };
        if self.context.current_rule_size >= k_max_gbnf_rule_elements {
            return false;
        }
        self.context.last_sym_start = self.context.current_rule_size;
        self.context.push(element {
            r#type: if negated {
                element_type::token_not
            } else {
                element_type::token
            },
            value: token_id,
        })
    }

    fn quantifier_bounds(text: &str) -> Option<(usize, usize)> {
        if text == "*" {
            return Some((0, usize::MAX));
        }
        if text == "+" {
            return Some((1, usize::MAX));
        }
        if text == "?" {
            return Some((0, 1));
        }
        if !text.starts_with('{') || !text.ends_with('}') {
            return None;
        }
        let core = &text[1..text.len() - 1];
        match core.find(',') {
            None => {
                let n = core.parse().ok()?;
                Some((n, n))
            }
            Some(i) => {
                let min = core[..i].parse().ok()?;
                let max = if i + 1 == core.len() {
                    usize::MAX
                } else {
                    core[i + 1..].parse().ok()?
                };
                Some((min, max))
            }
        }
    }
    fn can_apply_quantifier(&self, out: &grammar, min: usize, max: usize) -> bool {
        if min > MAX_REPETITION || (max != usize::MAX && (max > MAX_REPETITION || max < min)) {
            return false;
        }
        let Some(prev_len) = self
            .context
            .current_rule_size
            .checked_sub(self.context.last_sym_start)
        else {
            return false;
        };
        let repeated_len = if min == 0 {
            self.context.last_sym_start
        } else {
            let Some(repeated) = prev_len.checked_mul(min) else {
                return false;
            };
            let Some(repeated) = self.context.last_sym_start.checked_add(repeated) else {
                return false;
            };
            repeated
        };
        if repeated_len > k_max_gbnf_rule_elements {
            return false;
        }
        let n_opt = if max == usize::MAX { 1 } else { max - min };
        let Some(next_symbol_limit) = (self.context.next_symbol_id as usize).checked_add(n_opt)
        else {
            return false;
        };
        if next_symbol_limit > k_max_gbnf_rules {
            return false;
        }
        let mut added_grammar_elements = 0usize;
        for index in 0..n_opt {
            let rule_id = self.context.next_symbol_id as usize + index;
            let Some(rec_rule_len) = prev_len
                .checked_add(usize::from(index > 0 || max == usize::MAX))
                .and_then(|length| length.checked_add(2))
            else {
                return false;
            };
            if rec_rule_len > k_max_gbnf_rule_elements || out.rule_lengths[rule_id] != 0 {
                return false;
            }
            let Some(total) = added_grammar_elements.checked_add(rec_rule_len) else {
                return false;
            };
            added_grammar_elements = total;
        }
        let Some(grammar_end) = (out.element_count as usize).checked_add(added_grammar_elements)
        else {
            return false;
        };
        if grammar_end > k_max_gbnf_elements {
            return false;
        }
        repeated_len
            .checked_add(usize::from(n_opt > 0))
            .is_some_and(|length| length <= k_max_gbnf_rule_elements)
    }

    fn apply_quantifier(&mut self, out: &mut grammar) -> bool {
        let Some((min, max)) = Self::quantifier_bounds(self.context.text()) else {
            return false;
        };
        if self.context.last_sym_start == self.context.current_rule_size
            || !self.can_apply_quantifier(out, min, max)
        {
            return false;
        }
        let start = self.context.last_sym_start;
        let len = self.context.current_rule_size - start;
        self.context.repeat_scratch[..len]
            .copy_from_slice(&self.context.current_rule[start..start + len]);
        if min == 0 {
            self.context.current_rule_size = start;
        } else {
            for _ in 1..min {
                if self.context.current_rule_size + len > k_max_gbnf_rule_elements {
                    return false;
                }
                let end = self.context.current_rule_size + len;
                self.context.current_rule[end - len..end]
                    .copy_from_slice(&self.context.repeat_scratch[..len]);
                self.context.current_rule_size = end;
            }
        }
        let optional = if max == usize::MAX { 1 } else { max - min };
        let mut last = 0u32;
        for index in 0..optional {
            if self.context.next_symbol_id as usize >= k_max_gbnf_rules {
                return false;
            }
            let id = self.context.next_symbol_id;
            self.context.next_symbol_id += 1;
            self.context.rule_defined[id as usize] = true;
            let mut count = len;
            if index > 0 || max == usize::MAX {
                self.context.repeat_scratch[count] = element {
                    r#type: element_type::rule_ref,
                    value: if max == usize::MAX { id } else { last },
                };
                count += 1;
            }
            self.context.repeat_scratch[count] = element {
                r#type: element_type::alt,
                value: 0,
            };
            count += 1;
            self.context.repeat_scratch[count] = element {
                r#type: element_type::end,
                value: 0,
            };
            count += 1;
            if out.element_count as usize + count > k_max_gbnf_elements {
                return false;
            }
            let offset = out.element_count as usize;
            out.elements[offset..offset + count]
                .copy_from_slice(&self.context.repeat_scratch[..count]);
            out.rule_offsets[id as usize] = offset as u32;
            out.rule_lengths[id as usize] = count as u32;
            out.element_count += count as u32;
            out.rule_count = out.rule_count.max(id + 1);
            last = id;
        }
        optional == 0
            || self.context.push(element {
                r#type: element_type::rule_ref,
                value: last,
            })
    }

    fn run(&mut self, out: &mut grammar) -> ParseOutcome {
        loop {
            let has = match self.next_token() {
                Ok(x) => x,
                Err(e) => return e,
            };
            if !has {
                if self.state == GbnfRuleParserStates::InRuleExpressionAfterTerm
                    && !self.context.finalize(out)
                {
                    return ParseOutcome::ParseFailed;
                }
                if self.state == GbnfRuleParserStates::ExpectRuleName && out.rule_count == 0 {
                    return ParseOutcome::ParseFailed;
                }
                if self.state != GbnfRuleParserStates::ExpectRuleName
                    && self.state != GbnfRuleParserStates::InRuleExpressionAfterTerm
                {
                    return ParseOutcome::ParseFailed;
                }
                for entry in &self.context.symbols[..self.context.symbol_count] {
                    if entry.occupied && !entry.defined {
                        return ParseOutcome::ParseFailed;
                    }
                }
                return ParseOutcome::Parsed;
            }
            match self.state {
                GbnfRuleParserStates::ExpectRuleName => match self.context.token_kind {
                    TokenKind::Newline => {}
                    TokenKind::Identifier => match self.classify_nonterm(ParseMode::Definition) {
                        Ok(id) => {
                            self.context.current_rule_id = id;
                            self.state = GbnfRuleParserStates::ExpectDefinition;
                        }
                        Err(e) => return e,
                    },
                    _ => return ParseOutcome::ParseFailed,
                },
                GbnfRuleParserStates::ExpectDefinition => {
                    if self.context.token_kind != TokenKind::DefinitionOperator {
                        return ParseOutcome::ParseFailed;
                    }
                    self.context.current_rule_size = 0;
                    self.context.group_depth = 0;
                    self.state = GbnfRuleParserStates::InRuleExpressionNeedTerm;
                }
                GbnfRuleParserStates::InRuleExpressionNeedTerm => match self.context.token_kind {
                    TokenKind::Newline => {
                        if self.context.group_depth == 0 {
                            return ParseOutcome::ParseFailed;
                        }
                    }
                    TokenKind::Identifier => {
                        let id = match self.classify_nonterm(ParseMode::Reference) {
                            Ok(x) => x,
                            Err(e) => return e,
                        };
                        self.context.last_sym_start = self.context.current_rule_size;
                        if !self.context.push(element {
                            r#type: element_type::rule_ref,
                            value: id,
                        }) {
                            return ParseOutcome::ParseFailed;
                        }
                        self.state = GbnfRuleParserStates::InRuleExpressionAfterTerm;
                    }
                    TokenKind::RuleReference => {
                        if !self.consume_rule_reference() {
                            return ParseOutcome::ParseFailed;
                        }
                        self.state = GbnfRuleParserStates::InRuleExpressionAfterTerm;
                    }
                    TokenKind::StringLiteral => {
                        if !self.consume_literal() {
                            return ParseOutcome::ParseFailed;
                        }
                        self.state = GbnfRuleParserStates::InRuleExpressionAfterTerm;
                    }
                    TokenKind::CharacterClass => {
                        if !self.consume_class() {
                            return ParseOutcome::ParseFailed;
                        }
                        self.state = GbnfRuleParserStates::InRuleExpressionAfterTerm;
                    }
                    TokenKind::Dot => {
                        self.context.last_sym_start = self.context.current_rule_size;
                        if !self.context.push(element {
                            r#type: element_type::char_any,
                            value: 0,
                        }) {
                            return ParseOutcome::ParseFailed;
                        }
                        self.state = GbnfRuleParserStates::InRuleExpressionAfterTerm;
                    }
                    TokenKind::OpenGroup => {
                        if self.context.group_depth >= MAX_GROUP_NESTING_DEPTH
                            || self.context.next_symbol_id as usize >= k_max_gbnf_rules
                        {
                            return ParseOutcome::ParseFailed;
                        }
                        let id = self.context.next_symbol_id;
                        self.context.next_symbol_id += 1;
                        self.context.rule_defined[id as usize] = true;
                        self.context.groups[self.context.group_depth] = GroupFrame {
                            sequence_start: self.context.current_rule_size as u16,
                            generated_rule_id: id as u16,
                        };
                        self.context.group_depth += 1;
                    }
                    _ => return ParseOutcome::ParseFailed,
                },
                GbnfRuleParserStates::InRuleExpressionAfterTerm => match self.context.token_kind {
                    TokenKind::Newline => {
                        if self.context.group_depth == 0 {
                            if !self.context.finalize(out) {
                                return ParseOutcome::ParseFailed;
                            }
                            self.state = GbnfRuleParserStates::ExpectRuleName;
                        }
                    }
                    TokenKind::Alternation => {
                        if !self.context.push(element {
                            r#type: element_type::alt,
                            value: 0,
                        }) {
                            return ParseOutcome::ParseFailed;
                        }
                        self.state = GbnfRuleParserStates::InRuleExpressionNeedTerm;
                    }
                    TokenKind::Quantifier => {
                        if !self.apply_quantifier(out) {
                            return ParseOutcome::ParseFailed;
                        }
                    }
                    TokenKind::StringLiteral => {
                        if !self.consume_literal() {
                            return ParseOutcome::ParseFailed;
                        }
                    }
                    TokenKind::CharacterClass => {
                        if !self.consume_class() {
                            return ParseOutcome::ParseFailed;
                        }
                    }
                    TokenKind::Dot => {
                        self.context.last_sym_start = self.context.current_rule_size;
                        if !self.context.push(element {
                            r#type: element_type::char_any,
                            value: 0,
                        }) {
                            return ParseOutcome::ParseFailed;
                        }
                    }
                    TokenKind::Identifier => {
                        let id = match self.classify_nonterm(ParseMode::Reference) {
                            Ok(x) => x,
                            Err(e) => return e,
                        };
                        self.context.last_sym_start = self.context.current_rule_size;
                        if !self.context.push(element {
                            r#type: element_type::rule_ref,
                            value: id,
                        }) {
                            return ParseOutcome::ParseFailed;
                        }
                    }
                    TokenKind::RuleReference => {
                        if !self.consume_rule_reference() {
                            return ParseOutcome::ParseFailed;
                        }
                    }
                    TokenKind::CloseGroup => {
                        if self.context.group_depth == 0 {
                            return ParseOutcome::ParseFailed;
                        }
                        let frame = self.context.groups[self.context.group_depth - 1];
                        let len = self.context.current_rule_size - frame.sequence_start as usize;
                        if len + 1 > k_max_gbnf_rule_elements
                            || out.element_count as usize + len + 1 > k_max_gbnf_elements
                        {
                            return ParseOutcome::ParseFailed;
                        }
                        let offset = out.element_count as usize;
                        out.elements[offset..offset + len].copy_from_slice(
                            &self.context.current_rule
                                [frame.sequence_start as usize..self.context.current_rule_size],
                        );
                        out.elements[offset + len] = element {
                            r#type: element_type::end,
                            value: 0,
                        };
                        out.rule_offsets[frame.generated_rule_id as usize] = offset as u32;
                        out.rule_lengths[frame.generated_rule_id as usize] = (len + 1) as u32;
                        out.element_count += (len + 1) as u32;
                        out.rule_count = out.rule_count.max(u32::from(frame.generated_rule_id) + 1);
                        self.context.current_rule_size = frame.sequence_start as usize;
                        self.context.last_sym_start = self.context.current_rule_size;
                        if !self.context.push(element {
                            r#type: element_type::rule_ref,
                            value: u32::from(frame.generated_rule_id),
                        }) {
                            return ParseOutcome::ParseFailed;
                        }
                        self.context.group_depth -= 1;
                    }
                    _ => return ParseOutcome::ParseFailed,
                },
                _ => return ParseOutcome::InternalError,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_rule_elements(source: &str, expected: &[element]) {
        let mut parser = GbnfRuleParserActor::new();
        let mut grammar = grammar::default();
        assert_eq!(parser.parse(source, &mut grammar), ParseOutcome::Parsed);
        assert_eq!(grammar.rule(0).elements, Some(expected));
    }

    #[test]
    fn parses_plain_rule_reference_in_need_term_position() {
        assert_rule_elements(
            "root ::= <[12]>\n",
            &[
                element {
                    r#type: element_type::token,
                    value: 12,
                },
                element {
                    r#type: element_type::end,
                    value: 0,
                },
            ],
        );
    }

    #[test]
    fn parses_negated_rule_reference_after_term() {
        assert_rule_elements(
            "root ::= \"a\" !<[34]>\n",
            &[
                element {
                    r#type: element_type::character,
                    value: b'a' as u32,
                },
                element {
                    r#type: element_type::token_not,
                    value: 34,
                },
                element {
                    r#type: element_type::end,
                    value: 0,
                },
            ],
        );
    }

    #[test]
    fn rejects_malformed_or_overflowing_rule_references() {
        for source in [
            "root ::= <[]>\n",
            "root ::= <[12x]>\n",
            "root ::= <[4294967296]>\n",
            "root ::= !<[7>\n",
        ] {
            let mut parser = GbnfRuleParserActor::new();
            let mut output = grammar::default();
            assert_eq!(
                parser.parse(source, &mut output),
                ParseOutcome::ParseFailed,
                "{source:?}"
            );
            assert_eq!(output.rule_count, 0);
        }
    }

    #[test]
    fn emits_character_class_lead_types_like_pinned_parser() {
        assert_rule_elements(
            "root ::= [a-z]\n",
            &[
                element {
                    r#type: element_type::character,
                    value: b'a' as u32,
                },
                element {
                    r#type: element_type::char_rng_upper,
                    value: b'z' as u32,
                },
                element {
                    r#type: element_type::end,
                    value: 0,
                },
            ],
        );
        assert_rule_elements(
            "root ::= [^a-z]\n",
            &[
                element {
                    r#type: element_type::char_not,
                    value: b'a' as u32,
                },
                element {
                    r#type: element_type::char_rng_upper,
                    value: b'z' as u32,
                },
                element {
                    r#type: element_type::end,
                    value: 0,
                },
            ],
        );
    }

    #[test]
    fn callbacks_publish_caller_owned_grammar_in_order() {
        fn done(event: ParseDone<'_>) -> bool {
            let ParseDone { grammar } = event;
            assert_eq!(grammar.rule(0).elements.unwrap()[0].value, b'a' as u32);
            true
        }
        fn error(_: ParseError<'_>) -> bool {
            panic!("unexpected error callback")
        }
        let mut parser = GbnfRuleParserActor::new();
        let mut output = grammar::default();
        assert_eq!(
            parser.process_event(
                EventParseRules::new("root ::= \"a\"\n", &mut output).with_callbacks(done, error)
            ),
            ParseOutcome::Parsed
        );
        assert_eq!(output.rule_count, 1);
    }

    #[test]
    fn malformed_source_resets_output_and_invokes_error_once() {
        fn done(_: ParseDone<'_>) -> bool {
            panic!("unexpected done callback")
        }
        fn error(event: ParseError<'_>) -> bool {
            let ParseError { grammar, error } = event;
            assert_eq!(error, ParseOutcome::ParseFailed);
            assert_eq!(grammar.rule_count, 0);
            true
        }
        let mut parser = GbnfRuleParserActor::new();
        let mut output = grammar::default();
        assert_eq!(
            parser.parse("root ::= \"a\"\n", &mut output),
            ParseOutcome::Parsed
        );
        assert_eq!(
            parser.process_event(
                EventParseRules::new("root ::= \"a\n", &mut output).with_callbacks(done, error)
            ),
            ParseOutcome::ParseFailed
        );
    }

    #[test]
    fn invalid_requests_reset_output_and_report_invalid_request() {
        fn error(event: ParseError<'_>) -> bool {
            let ParseError { grammar, error } = event;
            assert_eq!(error, ParseOutcome::InvalidRequest);
            assert_eq!(grammar.rule_count, 0);
            true
        }
        let mut parser = GbnfRuleParserActor::new();
        let mut output = grammar::default();
        assert_eq!(
            parser.parse("root ::= \"a\"\n", &mut output),
            ParseOutcome::Parsed
        );
        assert_eq!(
            parser.process_event(
                EventParseRules::new("", &mut output).with_callbacks(ignore_parse_done, error)
            ),
            ParseOutcome::InvalidRequest
        );
        let oversized = "x".repeat(MAX_RULE_SOURCE_BYTES + 1);
        assert_eq!(
            parser.process_event(
                EventParseRules::new(&oversized, &mut output)
                    .with_callbacks(ignore_parse_done, error)
            ),
            ParseOutcome::InvalidRequest
        );
    }

    #[test]
    fn recovers_from_unexpected_event_for_subsequent_parse() {
        let mut parser = GbnfRuleParserActor::new();
        assert_eq!(parser.process_unexpected_event(), ParseOutcome::Unexpected);
        let mut output = grammar::default();
        assert_eq!(
            parser.parse("root ::= \"a\"\n", &mut output),
            ParseOutcome::Parsed
        );
    }

    #[test]
    fn preserves_multiline_group_until_group_close() {
        let mut parser = GbnfRuleParserActor::new();
        let mut output = grammar::default();
        assert_eq!(
            parser.parse("root ::= (\"a\"\n\"b\")\n", &mut output),
            ParseOutcome::Parsed
        );
        assert_eq!(output.rule_count, 2);
        assert_eq!(output.element_count, 5);
    }

    #[test]
    fn preserves_trailing_hyphen_as_character_class_member() {
        assert_rule_elements(
            "root ::= [a-]\n",
            &[
                element {
                    r#type: element_type::character,
                    value: b'a' as u32,
                },
                element {
                    r#type: element_type::char_alt,
                    value: b'-' as u32,
                },
                element {
                    r#type: element_type::end,
                    value: 0,
                },
            ],
        );
    }
}
