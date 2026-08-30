//! Bounded, synchronous GBNF rule parser.
//!
//! The root actor mirrors the pinned rule-parser phases while keeping all
//! storage inline. Lexing and token classification are delegated to sibling
//! child actors; one dispatch runs to completion without queues.

#![allow(dead_code, missing_docs, unused_imports)]

use super::lexer::{Cursor, EventScanNext, NextDone, NextError, TokenKind};
use super::lexer::sm::GbnfRuleParserLexerStateMachine;
use super::nonterm_parser::sm::{NontermParser, ParseMode, ParseOutcome as NontermOutcome};
use crate::gbnf::{element, element_type, grammar, k_max_gbnf_elements, k_max_gbnf_rule_elements, k_max_gbnf_rules};

/// Maximum source bytes retained by one dispatch.
pub const MAX_RULE_SOURCE_BYTES: usize = 65_536;
/// Maximum nested groups accepted by the pinned parser.
pub const MAX_GROUP_NESTING_DEPTH: usize = 32;
const MAX_SYMBOL_NAME_BYTES: usize = 128;
const MAX_REPETITION: usize = 2000;
const MAX_SYMBOLS: usize = 2048;

/// Root parser request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventParseRules<'a> { pub source: &'a str }

/// Bounded result of one root parser dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseOutcome { Parsed, InvalidRequest, ParseFailed, InternalError, Unexpected }

/// Generated topology state inspection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GbnfRuleParserStates {
    Ready, ExpectRuleName, ExpectRuleNameDecision, ExpectDefinition,
    ExpectDefinitionDecision, InRuleExpressionNeedTerm,
    InRuleExpressionNeedTermDecision, InRuleExpressionAfterTerm,
    InRuleExpressionAfterTermDecision, RuleReferenceDecision,
    RuleReferencePlainExec, RuleReferenceNegatedExec, QuantifierDecision,
    QuantifierStarExec, QuantifierPlusExec, QuantifierQuestionExec,
    QuantifierBracedExactExec, QuantifierBracedOpenExec,
    QuantifierBracedRangeExec, EofSymbolsDecision, ParseDecision,
    UnexpectedEvent,
}

#[derive(Clone, Copy, Debug, Default)]
struct GroupFrame { sequence_start: u16, generated_rule_id: u16 }

#[derive(Clone, Copy, Debug)]
struct SymbolEntry { hash: u32, id: u32, len: u16, bytes: [u8; MAX_SYMBOL_NAME_BYTES], occupied: bool, defined: bool }
impl Default for SymbolEntry {
    fn default() -> Self { Self { hash: 0, id: 0, len: 0, bytes: [0; MAX_SYMBOL_NAME_BYTES], occupied: false, defined: false } }
}

/// Context retained by the bounded root actor.
#[derive(Debug)]
pub struct GbnfRuleParserContext {
    source: [u8; MAX_RULE_SOURCE_BYTES], source_len: usize,
    cursor_offset: u32, cursor_token_count: u32,
    token_kind: TokenKind, token_start: usize, token_end: usize, has_token: bool,
    current_rule: [element; k_max_gbnf_rule_elements], repeat_scratch: [element; k_max_gbnf_rule_elements],
    current_rule_size: usize, current_rule_id: u32, last_sym_start: usize,
    groups: [GroupFrame; MAX_GROUP_NESTING_DEPTH], group_depth: usize,
    symbols: [SymbolEntry; MAX_SYMBOLS], symbol_count: usize,
    rule_defined: [bool; k_max_gbnf_rules], next_symbol_id: u32,
}

impl Default for GbnfRuleParserContext {
    fn default() -> Self { Self {
        source: [0; MAX_RULE_SOURCE_BYTES], source_len: 0, cursor_offset: 0, cursor_token_count: 0,
        token_kind: TokenKind::Unknown, token_start: 0, token_end: 0, has_token: false,
        current_rule: [element::default(); k_max_gbnf_rule_elements], repeat_scratch: [element::default(); k_max_gbnf_rule_elements],
        current_rule_size: 0, current_rule_id: 0, last_sym_start: 0,
        groups: [GroupFrame::default(); MAX_GROUP_NESTING_DEPTH], group_depth: 0,
        symbols: [SymbolEntry::default(); MAX_SYMBOLS], symbol_count: 0,
        rule_defined: [false; k_max_gbnf_rules], next_symbol_id: 0,
    } }
}

impl GbnfRuleParserContext {
    fn reset(&mut self, source: &str) -> bool {
        if source.is_empty() || source.len() > MAX_RULE_SOURCE_BYTES { return false; }
        self.source[..source.len()].copy_from_slice(source.as_bytes()); self.source_len = source.len();
        self.cursor_offset = 0; self.cursor_token_count = 0; self.token_kind = TokenKind::Unknown;
        self.token_start = 0; self.token_end = 0; self.has_token = false; self.current_rule_size = 0;
        self.current_rule_id = 0; self.last_sym_start = 0; self.group_depth = 0;
        self.symbols.fill(SymbolEntry::default()); self.symbol_count = 0;
        self.rule_defined.fill(false); self.next_symbol_id = 0; true
    }
    fn source(&self) -> &str { core::str::from_utf8(&self.source[..self.source_len]).unwrap_or("") }
    fn text(&self) -> &str { self.source().get(self.token_start..self.token_end).unwrap_or("") }
    fn push(&mut self, item: element) -> bool {
        if self.current_rule_size >= k_max_gbnf_rule_elements { false } else { self.current_rule[self.current_rule_size] = item; self.current_rule_size += 1; true }
    }
    fn hash(name: &[u8]) -> u32 { let mut h = 2_166_136_261u32; for b in name { h = (h ^ u32::from(*b)).wrapping_mul(16_777_619); } if h == 0 { 1 } else { h } }
    fn symbol(&mut self, name: &[u8], definition: bool) -> Option<u32> {
        if name.is_empty() || name.len() > MAX_SYMBOL_NAME_BYTES { return None; }
        let hash = Self::hash(name);
        for entry in &mut self.symbols[..self.symbol_count] {
            if entry.occupied && entry.hash == hash && entry.len as usize == name.len() && entry.bytes[..name.len()] == *name {
                if definition && entry.defined { return None; }
                entry.defined |= definition; self.rule_defined[entry.id as usize] |= definition; return Some(entry.id);
            }
        }
        if self.symbol_count >= MAX_SYMBOLS || self.next_symbol_id as usize >= k_max_gbnf_rules { return None; }
        let id = self.next_symbol_id; self.next_symbol_id += 1;
        let entry = &mut self.symbols[self.symbol_count]; entry.hash = hash; entry.id = id; entry.len = name.len() as u16; entry.bytes[..name.len()].copy_from_slice(name); entry.occupied = true; entry.defined = definition;
        self.rule_defined[id as usize] = definition; self.symbol_count += 1; Some(id)
    }
    fn append_rule(&mut self, out: &mut grammar, id: u32) -> bool {
        if id as usize >= k_max_gbnf_rules || self.current_rule_size == 0 || out.rule_lengths[id as usize] != 0 { return false; }
        let offset = out.element_count as usize; let Some(end) = offset.checked_add(self.current_rule_size) else { return false; }; if end > k_max_gbnf_elements { return false; }
        out.elements[offset..end].copy_from_slice(&self.current_rule[..self.current_rule_size]); out.rule_offsets[id as usize] = offset as u32; out.rule_lengths[id as usize] = self.current_rule_size as u32; out.element_count = end as u32; out.rule_count = out.rule_count.max(id + 1); true
    }
    fn finalize(&mut self, out: &mut grammar) -> bool {
        if self.group_depth != 0 || self.current_rule_size == 0 || !self.push(element { r#type: element_type::end, value: 0 }) { return false }
        let ok = self.append_rule(out, self.current_rule_id); self.current_rule_size = 0; self.last_sym_start = 0; ok
    }
}

/// Synchronous bounded root parser actor.
pub struct GbnfRuleParserActor { state: GbnfRuleParserStates, context: GbnfRuleParserContext, grammar: grammar, lexer: GbnfRuleParserLexerStateMachine }
impl Default for GbnfRuleParserActor { fn default() -> Self { Self::new() } }
impl GbnfRuleParserActor {
    /// Creates an actor in the generated ready state.
    #[must_use] pub fn new() -> Self { Self { state: GbnfRuleParserStates::Ready, context: GbnfRuleParserContext::default(), grammar: grammar::default(), lexer: GbnfRuleParserLexerStateMachine::default() } }
    /// Parses one source in a single run-to-completion dispatch.
    pub fn process_event(&mut self, event: EventParseRules<'_>) -> ParseOutcome {
        if self.state != GbnfRuleParserStates::Ready { return ParseOutcome::InternalError; }
        if !self.context.reset(event.source) { return ParseOutcome::InvalidRequest; }
        self.grammar.reset(); self.state = GbnfRuleParserStates::ExpectRuleName;
        let outcome = self.run(); self.state = GbnfRuleParserStates::Ready; outcome
    }
    /// Convenience source parser.
    pub fn parse(&mut self, source: &str) -> ParseOutcome { self.process_event(EventParseRules { source }) }
    /// Returns generated state inspection.
    #[must_use] pub fn state(&self) -> &GbnfRuleParserStates { &self.state }
    /// Tests generated state identity.
    #[must_use] pub fn is(&self, state: &GbnfRuleParserStates) -> bool { self.state == *state }
    /// Returns parsed fixed-capacity grammar storage.
    #[must_use] pub fn grammar(&self) -> &grammar { &self.grammar }
    /// Returns bounded context inspection.
    #[must_use] pub fn context(&self) -> &GbnfRuleParserContext { &self.context }
    /// Reports an explicit unexpected event.
    pub fn process_unexpected_event(&mut self) -> ParseOutcome { self.state = GbnfRuleParserStates::UnexpectedEvent; ParseOutcome::Unexpected }

    fn next_token(&mut self) -> Result<bool, ParseOutcome> {
        let cursor = Cursor { input: self.context.source(), offset: self.context.cursor_offset, token_count: self.context.cursor_token_count };
        let mut got = (false, TokenKind::Unknown, 0usize, 0usize, cursor.offset, cursor.token_count);
        let mut done = |event: NextDone<'_>| { got = (event.has_token, event.token.kind, event.token.start as usize, event.token.end as usize, event.next_cursor.offset, event.next_cursor.token_count); true };
        let mut error = |_event: NextError| false;
        if !self.lexer.process_event(EventScanNext::new(cursor, &mut done, &mut error)) { return Err(ParseOutcome::InternalError); }
        self.context.has_token = got.0; self.context.token_kind = got.1; self.context.token_start = got.2; self.context.token_end = got.3; self.context.cursor_offset = got.4; self.context.cursor_token_count = got.5; Ok(got.0)
    }
    fn classify_nonterm(&mut self, mode: ParseMode) -> Result<u32, ParseOutcome> {
        let text = self.context.text(); let mut child = NontermParser::new();
        if !matches!(child.classify(mode, text), NontermOutcome::Parsed { .. }) { return Err(ParseOutcome::ParseFailed); }
        self.context.symbol(text.as_bytes(), mode == ParseMode::Definition).ok_or(ParseOutcome::ParseFailed)
    }
    fn parse_char(bytes: &[u8], pos: usize, end: usize) -> Option<(u32, usize)> {
        if pos >= end { return None; }
        if bytes[pos] != b'\\' { let s = core::str::from_utf8(&bytes[pos..end]).ok()?; let c = s.chars().next()?; return Some((c as u32, pos + c.len_utf8())); }
        if pos + 1 >= end { return None; }
        match bytes[pos + 1] { b'n' => Some((10, pos+2)), b'r' => Some((13, pos+2)), b't' => Some((9, pos+2)), b'\\'|b'"'|b'['|b']' => Some((bytes[pos+1] as u32, pos+2)), b'x'|b'u'|b'U' => { let count = if bytes[pos+1] == b'x' { 2 } else if bytes[pos+1] == b'u' { 4 } else { 8 }; if pos + 2 + count > end { return None; } let mut value=0u32; let mut i=pos+2; while i < pos+2+count { let d=match bytes[i] { b'0'..=b'9'=>bytes[i]-b'0', b'a'..=b'f'=>bytes[i]-b'a'+10, b'A'..=b'F'=>bytes[i]-b'A'+10, _=>return None }; value=value.checked_mul(16)?.checked_add(u32::from(d))?; i+=1; } Some((value,pos+2+count)) }, _ => None }
    }
    fn consume_literal(&mut self) -> bool {
        let bytes=self.context.text().as_bytes(); if bytes.len()<2 || bytes[0]!=b'"' || bytes[bytes.len()-1]!=b'"' { return false; }
        self.context.last_sym_start=self.context.current_rule_size; let mut pos=1; while pos+1<bytes.len() { let Some((value,next))=Self::parse_char(bytes,pos,bytes.len()-1) else{return false}; if !self.context.push(element{r#type:element_type::character,value}){return false}; pos=next; } pos==bytes.len()-1
    }
    fn consume_class(&mut self) -> bool {
        let bytes=self.context.text().as_bytes(); if bytes.len()<3 || bytes[0]!=b'[' || bytes[bytes.len()-1]!=b']' { return false; }
        self.context.last_sym_start=self.context.current_rule_size; let end=bytes.len()-1; let mut pos=1; let negated=bytes[pos]==b'^'; if negated{pos+=1}; let mut first=true;
        while pos<end { let Some((value,next))=Self::parse_char(bytes,pos,end) else{return false}; if !self.context.push(element{r#type:if negated&&first{element_type::char_not}else{element_type::char_alt},value}){return false}; first=false; pos=next; if pos+1<end&&bytes[pos]==b'-' { pos+=1; let Some((upper,next_upper))=Self::parse_char(bytes,pos,end) else{return false}; if !self.context.push(element{r#type:element_type::char_rng_upper,value:upper}){return false}; pos=next_upper; } }
        !first
    }
    fn quantifier_bounds(text:&str)->Option<(usize,usize)> { if text=="*"{return Some((0,usize::MAX))} if text=="+"{return Some((1,usize::MAX))} if text=="?"{return Some((0,1))} if !text.starts_with('{')||!text.ends_with('}') {return None} let core=&text[1..text.len()-1]; match core.find(',') { None=>{let n=core.parse().ok()?;Some((n,n))}, Some(i)=>{let min=core[..i].parse().ok()?;let max=if i+1==core.len(){usize::MAX}else{core[i+1..].parse().ok()?};Some((min,max))} } }
    fn apply_quantifier(&mut self) -> bool {
        let Some((min,max))=Self::quantifier_bounds(self.context.text()) else{return false}; if min>MAX_REPETITION||(max!=usize::MAX&&(max>MAX_REPETITION||max<min))||self.context.last_sym_start==self.context.current_rule_size{return false};
        let start=self.context.last_sym_start; let len=self.context.current_rule_size-start; self.context.repeat_scratch[..len].copy_from_slice(&self.context.current_rule[start..start+len]);
        if min==0 {self.context.current_rule_size=start} else {for _ in 1..min {if self.context.current_rule_size+len>k_max_gbnf_rule_elements{return false}; let end=self.context.current_rule_size+len; self.context.current_rule[end-len..end].copy_from_slice(&self.context.repeat_scratch[..len]); self.context.current_rule_size=end;}}
        let optional=if max==usize::MAX{1}else{max-min}; let mut last=0u32;
        for index in 0..optional { if self.context.next_symbol_id as usize>=k_max_gbnf_rules{return false}; let id=self.context.next_symbol_id; self.context.next_symbol_id+=1; self.context.rule_defined[id as usize]=true; let mut count=len; if index>0||max==usize::MAX {self.context.repeat_scratch[count]=element{r#type:element_type::rule_ref,value:if max==usize::MAX{id}else{last}};count+=1}; self.context.repeat_scratch[count]=element{r#type:element_type::alt,value:0};count+=1;self.context.repeat_scratch[count]=element{r#type:element_type::end,value:0};count+=1; if self.grammar.element_count as usize+count>k_max_gbnf_elements{return false}; let offset=self.grammar.element_count as usize; self.grammar.elements[offset..offset+count].copy_from_slice(&self.context.repeat_scratch[..count]);self.grammar.rule_offsets[id as usize]=offset as u32;self.grammar.rule_lengths[id as usize]=count as u32;self.grammar.element_count+=count as u32;self.grammar.rule_count=self.grammar.rule_count.max(id+1);last=id; }
        optional==0 || self.context.push(element{r#type:element_type::rule_ref,value:last})
    }
    fn run(&mut self) -> ParseOutcome {
        loop {
            let has=match self.next_token(){Ok(x)=>x,Err(e)=>return e};
            if !has { if self.state==GbnfRuleParserStates::InRuleExpressionAfterTerm&&!self.context.finalize(&mut self.grammar){return ParseOutcome::ParseFailed}; if self.state==GbnfRuleParserStates::ExpectRuleName&&self.grammar.rule_count==0{return ParseOutcome::ParseFailed}; if self.state!=GbnfRuleParserStates::ExpectRuleName&&self.state!=GbnfRuleParserStates::InRuleExpressionAfterTerm{return ParseOutcome::ParseFailed}; for entry in &self.context.symbols[..self.context.symbol_count]{if entry.occupied&&!entry.defined{return ParseOutcome::ParseFailed}} return ParseOutcome::Parsed; }
            match self.state {
                GbnfRuleParserStates::ExpectRuleName => match self.context.token_kind { TokenKind::Newline=>{}, TokenKind::Identifier=>match self.classify_nonterm(ParseMode::Definition){Ok(id)=>{self.context.current_rule_id=id;self.state=GbnfRuleParserStates::ExpectDefinition},Err(e)=>return e}, _=>return ParseOutcome::ParseFailed },
                GbnfRuleParserStates::ExpectDefinition => {if self.context.token_kind!=TokenKind::DefinitionOperator{return ParseOutcome::ParseFailed};self.context.current_rule_size=0;self.context.group_depth=0;self.state=GbnfRuleParserStates::InRuleExpressionNeedTerm},
                GbnfRuleParserStates::InRuleExpressionNeedTerm => match self.context.token_kind { TokenKind::Newline=>{if self.context.group_depth==0{return ParseOutcome::ParseFailed}}, TokenKind::Identifier=>{let id=match self.classify_nonterm(ParseMode::Reference){Ok(x)=>x,Err(e)=>return e};self.context.last_sym_start=self.context.current_rule_size;if !self.context.push(element{r#type:element_type::rule_ref,value:id}){return ParseOutcome::ParseFailed};self.state=GbnfRuleParserStates::InRuleExpressionAfterTerm},TokenKind::StringLiteral=>{if !self.consume_literal(){return ParseOutcome::ParseFailed};self.state=GbnfRuleParserStates::InRuleExpressionAfterTerm},TokenKind::CharacterClass=>{if !self.consume_class(){return ParseOutcome::ParseFailed};self.state=GbnfRuleParserStates::InRuleExpressionAfterTerm},TokenKind::Dot=>{self.context.last_sym_start=self.context.current_rule_size;if !self.context.push(element{r#type:element_type::char_any,value:0}){return ParseOutcome::ParseFailed};self.state=GbnfRuleParserStates::InRuleExpressionAfterTerm},TokenKind::OpenGroup=>{if self.context.group_depth>=MAX_GROUP_NESTING_DEPTH||self.context.next_symbol_id as usize>=k_max_gbnf_rules{return ParseOutcome::ParseFailed};let id=self.context.next_symbol_id;self.context.next_symbol_id+=1;self.context.rule_defined[id as usize]=true;self.context.groups[self.context.group_depth]=GroupFrame{sequence_start:self.context.current_rule_size as u16,generated_rule_id:id as u16};self.context.group_depth+=1},_=>return ParseOutcome::ParseFailed},
                GbnfRuleParserStates::InRuleExpressionAfterTerm => match self.context.token_kind { TokenKind::Newline=>{if !self.context.finalize(&mut self.grammar){return ParseOutcome::ParseFailed};self.state=GbnfRuleParserStates::ExpectRuleName},TokenKind::Alternation=>{if !self.context.push(element{r#type:element_type::alt,value:0}){return ParseOutcome::ParseFailed};self.state=GbnfRuleParserStates::InRuleExpressionNeedTerm},TokenKind::Quantifier=>{if !self.apply_quantifier(){return ParseOutcome::ParseFailed}},TokenKind::StringLiteral=>{if !self.consume_literal(){return ParseOutcome::ParseFailed}},TokenKind::CharacterClass=>{if !self.consume_class(){return ParseOutcome::ParseFailed}},TokenKind::Dot=>{self.context.last_sym_start=self.context.current_rule_size;if !self.context.push(element{r#type:element_type::char_any,value:0}){return ParseOutcome::ParseFailed}},TokenKind::Identifier=>{let id=match self.classify_nonterm(ParseMode::Reference){Ok(x)=>x,Err(e)=>return e};self.context.last_sym_start=self.context.current_rule_size;if !self.context.push(element{r#type:element_type::rule_ref,value:id}){return ParseOutcome::ParseFailed}},TokenKind::CloseGroup=>{if self.context.group_depth==0{return ParseOutcome::ParseFailed};let frame=self.context.groups[self.context.group_depth-1];let len=self.context.current_rule_size-frame.sequence_start as usize;if len+1>k_max_gbnf_rule_elements||self.grammar.element_count as usize+len+1>k_max_gbnf_elements{return ParseOutcome::ParseFailed};let offset=self.grammar.element_count as usize;self.grammar.elements[offset..offset+len].copy_from_slice(&self.context.current_rule[frame.sequence_start as usize..self.context.current_rule_size]);self.grammar.elements[offset+len]=element{r#type:element_type::end,value:0};self.grammar.rule_offsets[frame.generated_rule_id as usize]=offset as u32;self.grammar.rule_lengths[frame.generated_rule_id as usize]=(len+1) as u32;self.grammar.element_count+=(len+1) as u32;self.grammar.rule_count=self.grammar.rule_count.max(u32::from(frame.generated_rule_id)+1);self.context.current_rule_size=frame.sequence_start as usize;self.context.last_sym_start=self.context.current_rule_size;if !self.context.push(element{r#type:element_type::rule_ref,value:u32::from(frame.generated_rule_id)}){return ParseOutcome::ParseFailed};self.context.group_depth-=1},_=>return ParseOutcome::ParseFailed},
                _=>return ParseOutcome::InternalError,
            }
        }
    }
}

/// Compatibility alias for root parser callers.
pub type RuleParser = GbnfRuleParserActor;
/// Compatibility alias naming the root actor.
pub type GbnfRuleParser = GbnfRuleParserActor;
