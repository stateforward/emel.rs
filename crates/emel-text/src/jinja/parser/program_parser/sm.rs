//! Source-aligned bounded Jinja program parser.
#![allow(clippy::derive_partial_eq_without_eq, clippy::module_name_repetitions, clippy::missing_errors_doc, clippy::must_use_candidate, clippy::return_self_not_must_use, clippy::empty_structs_with_brackets, clippy::missing_const_for_fn, dead_code, unused_imports, missing_docs)]

use sml::sml;
use super::{expression_parser, statement_parser};
use super::super::lexer::sm::TokenKind;

pub const MAX_PROGRAM_TOKENS: usize = 512;
pub const MAX_TOKEN_VALUE: usize = 128;
pub type TokenType = TokenKind;
pub type TokenKindAlias = TokenKind;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Token { pub kind: TokenKind, pub value: [u8; MAX_TOKEN_VALUE], pub value_len: u16, pub pos: u32 }
impl Default for Token { fn default() -> Self { Self { kind: TokenKind::Eof, value: [0; MAX_TOKEN_VALUE], value_len: 0, pos: 0 } } }
impl Token {
    #[must_use] pub fn new(kind: TokenKind, value: &[u8], pos: usize) -> Self { let mut t=Self { kind, pos: pos.min(u32::MAX as usize) as u32, ..Self::default() }; let n=value.len().min(MAX_TOKEN_VALUE); t.value[..n].copy_from_slice(&value[..n]); t.value_len=n as u16; t }
    #[must_use] pub fn value(&self) -> &[u8] { &self.value[..self.value_len as usize] }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i32)]
pub enum ProgramParserError { #[default] None=0, InvalidRequest=1, ParseFailed=2, InternalError=4, Untracked=8, Unknown=-1 }
pub type ParseError = ProgramParserError;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProgramParseResult { pub parsed: bool, pub failed: bool, pub unexpected: bool, pub error: ProgramParserError, pub error_pos: u32, pub emitted_count: u16, pub emitted: Token, pub emitted_present: bool }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramParserInput { pub tokens: [Token; MAX_PROGRAM_TOKENS], pub token_count: u16 }
impl Default for ProgramParserInput { fn default() -> Self { Self { tokens: [Token::default(); MAX_PROGRAM_TOKENS], token_count: 0 } } }
impl ProgramParserInput {
    #[must_use] pub fn new(tokens: &[Token]) -> Self { let n=tokens.len().min(MAX_PROGRAM_TOKENS); let mut x=Self::default(); x.tokens[..n].copy_from_slice(&tokens[..n]); x.token_count=n as u16; x }
    #[must_use] pub fn from_tokens(tokens: &[Token]) -> Self { Self::new(tokens) }
    #[must_use] pub const fn empty() -> Self { Self { tokens: [Token { kind: TokenKind::Eof, value: [0; MAX_TOKEN_VALUE], value_len: 0, pos: 0 }; MAX_PROGRAM_TOKENS], token_count: 0 } }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventParseRuntime { pub tokens: [Token; MAX_PROGRAM_TOKENS], pub token_count: u16, pub token_index: u16, pub result: ProgramParseResult }
impl EventParseRuntime {
    #[must_use] pub fn from_tokens(tokens: &[Token]) -> Self { let input=ProgramParserInput::new(tokens); Self { tokens: input.tokens, token_count: input.token_count, token_index: 0, result: ProgramParseResult::default() } }
    #[must_use] pub fn from_input(input: ProgramParserInput) -> Self { Self { tokens: input.tokens, token_count: input.token_count, token_index: 0, result: ProgramParseResult::default() } }
}
impl From<ProgramParserInput> for EventParseRuntime { fn from(input: ProgramParserInput) -> Self { Self::from_input(input) } }
pub type ProgramRuntime = EventParseRuntime;

sml! {
    TextJinjaParserProgramParser {
        "parse_begin"_s <= *"deciding"_s + completion<EventParseRuntime> / start_program_parse,
        "dispatch_decision"_s <= "parse_begin"_s + completion<EventParseRuntime>,
        "parsed"_s <= "dispatch_decision"_s + completion<EventParseRuntime> [at_eof] / finish_parsed,
        "text_emit"_s <= "dispatch_decision"_s + completion<EventParseRuntime> [token_text],
        "comment_emit"_s <= "dispatch_decision"_s + completion<EventParseRuntime> [token_comment],
        "statement_parser_model"_s <= "dispatch_decision"_s + completion<EventParseRuntime> [token_open_statement],
        "expression_parser_model"_s <= "dispatch_decision"_s + completion<EventParseRuntime> [token_open_expression],
        "parse_failed"_s <= "dispatch_decision"_s + completion<EventParseRuntime> [token_unexpected] / fail_current_token,
        "dispatch_decision"_s <= "text_emit"_s + completion<EventParseRuntime> / consume_text,
        "dispatch_decision"_s <= "comment_emit"_s + completion<EventParseRuntime> / consume_comment,
        "statement_parse_result_decision"_s <= "statement_parser_model"_s + completion<EventParseRuntime> / run_statement_parser,
        "dispatch_decision"_s <= "statement_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_none],
        "parse_failed"_s <= "statement_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_invalid_request],
        "parse_failed"_s <= "statement_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_parse_failed],
        "parse_failed"_s <= "statement_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_internal_error],
        "parse_failed"_s <= "statement_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_untracked],
        "parse_failed"_s <= "statement_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_unknown],
        "expression_parse_result_decision"_s <= "expression_parser_model"_s + completion<EventParseRuntime> / run_expression_parser,
        "dispatch_decision"_s <= "expression_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_none],
        "parse_failed"_s <= "expression_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_invalid_request],
        "parse_failed"_s <= "expression_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_parse_failed],
        "parse_failed"_s <= "expression_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_internal_error],
        "parse_failed"_s <= "expression_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_untracked],
        "parse_failed"_s <= "expression_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_unknown],
        "parsed"_s = X, "parse_failed"_s = X,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "parse_begin"_s + unexpected_event<_> / on_unexpected_from_parse_begin,
        "unexpected_event"_s <= "dispatch_decision"_s + unexpected_event<_> / on_unexpected_from_dispatch_decision,
        "unexpected_event"_s <= "text_emit"_s + unexpected_event<_> / on_unexpected_from_text_emit,
        "unexpected_event"_s <= "comment_emit"_s + unexpected_event<_> / on_unexpected_from_comment_emit,
        "unexpected_event"_s <= "statement_parse_result_decision"_s + unexpected_event<_> / on_unexpected_from_statement_parse_result_decision,
        "unexpected_event"_s <= "expression_parse_result_decision"_s + unexpected_event<_> / on_unexpected_from_expression_parse_result_decision,
        "unexpected_event"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
    }
}

#[derive(Debug)]
pub struct TextJinjaParserProgramParserContext { pub input: EventParseRuntime, pub result: ProgramParseResult }
impl Default for TextJinjaParserProgramParserContext { fn default() -> Self { Self { input: EventParseRuntime::default(), result: ProgramParseResult::default() } } }
impl TextJinjaParserProgramParserContext {
    fn index(&self)->usize { self.input.token_index as usize }
    fn current(&self)->Option<Token> { let i=self.index(); (i < self.input.token_count as usize).then(||self.input.tokens[i]) }
    fn set_index(&mut self,n:usize) { self.input.token_index=n.min(self.input.token_count as usize) as u16; }
    fn fail(&mut self,e:ProgramParserError,p:u32) { self.result.failed=true; self.result.error=e; self.result.error_pos=p; }
    fn unexpected(&mut self) { self.result.unexpected=true; self.result.failed=true; self.result.error=ProgramParserError::InternalError; self.result.error_pos=self.current().map_or(0,|t|t.pos); }
    fn child_end(&self)->usize { let mut i=self.index(); while i < self.input.token_count as usize { let k=self.input.tokens[i].kind; i+=1; if matches!(k,TokenKind::CloseStatement|TokenKind::CloseExpression){break;} } i }
    fn run_statement(&mut self) {
        let start=self.index(); let end=self.child_end(); let mut child=[statement_parser::Token::default();statement_parser::MAX_STATEMENT_TOKENS]; let mut n=0usize;
        for i in start..end { if n==child.len(){self.fail(ProgramParserError::InvalidRequest,self.input.tokens[start].pos);return;} let Some(v)=core::str::from_utf8(self.input.tokens[i].value()).ok() else {self.fail(ProgramParserError::InvalidRequest,self.input.tokens[i].pos);return;}; child[n]=statement_parser::Token::new(map_statement_kind(self.input.tokens[i].kind),v,self.input.tokens[i].pos as usize); n+=1; }
        let mut actor=statement_parser::StatementParser::new(); let r=actor.process_event(statement_parser::EventParseRuntime::from_tokens(&child[..n])); self.result.error=map_statement_error(r.error,r.unexpected); self.result.error_pos=r.error_pos.min(u32::MAX as usize) as u32;
        if self.result.error!=ProgramParserError::None {self.result.failed=true;return;} self.result.emitted_count=self.result.emitted_count.saturating_add(r.emitted_count); self.set_index(end);
    }
    fn run_expression(&mut self) {
        let start=self.index(); let end=self.child_end(); let mut child=[expression_parser::Token::default();expression_parser::MAX_EXPRESSION_TOKENS]; let mut n=0usize;
        for i in start..end { if n==child.len(){self.fail(ProgramParserError::InvalidRequest,self.input.tokens[start].pos);return;} let t=self.input.tokens[i]; child[n]=expression_parser::Token::new(t.kind,t.value(),t.pos as usize); n+=1; }
        let mut actor=expression_parser::ExpressionParser::new(); let r=actor.process_event(expression_parser::ExpressionInput::new(&child[..n])); self.result.error=map_expression_error(r.error); self.result.error_pos=r.error_pos;
        if self.result.error!=ProgramParserError::None {self.result.failed=true;return;} if r.emitted_present {self.result.emitted=Token::new(r.emitted.kind,r.emitted.value(),r.emitted.pos as usize);self.result.emitted_present=true;self.result.emitted_count=self.result.emitted_count.saturating_add(1);} self.set_index(start.saturating_add(r.consumed as usize));
    }
}

impl TextJinjaParserProgramParserStateMachineContext for TextJinjaParserProgramParserContext {
    fn start_program_parse(&mut self)->Result<(),()>{self.input.token_index=0;self.result=ProgramParseResult::default();Ok(())}
    fn at_eof(&self)->Result<bool,()>{Ok(self.index()>=self.input.token_count as usize)}
    fn token_text(&self)->Result<bool,()>{Ok(self.current().is_some_and(|t|t.kind==TokenKind::Text))}
    fn token_comment(&self)->Result<bool,()>{Ok(self.current().is_some_and(|t|t.kind==TokenKind::Comment))}
    fn token_open_expression(&self)->Result<bool,()>{Ok(self.current().is_some_and(|t|t.kind==TokenKind::OpenExpression))}
    fn token_open_statement(&self)->Result<bool,()>{Ok(self.current().is_some_and(|t|t.kind==TokenKind::OpenStatement))}
    fn token_unexpected(&self)->Result<bool,()>{Ok(self.current().is_some_and(|t|!matches!(t.kind,TokenKind::Eof|TokenKind::Text|TokenKind::Comment|TokenKind::OpenExpression|TokenKind::OpenStatement)))}
    fn consume_text(&mut self)->Result<(),()>{self.set_index(self.index()+1);self.result.emitted_count=self.result.emitted_count.saturating_add(1);Ok(())}
    fn consume_comment(&mut self)->Result<(),()>{self.set_index(self.index()+1);self.result.emitted_count=self.result.emitted_count.saturating_add(1);Ok(())}
    fn finish_parsed(&mut self)->Result<(),()>{self.result.parsed=true;Ok(())}
    fn fail_current_token(&mut self)->Result<(),()>{self.fail(ProgramParserError::ParseFailed,self.current().map_or(0,|t|t.pos));Ok(())}
    fn run_statement_parser(&mut self)->Result<(),()>{self.run_statement();Ok(())}
    fn run_expression_parser(&mut self)->Result<(),()>{self.run_expression();Ok(())}
    fn parse_error_none(&self)->Result<bool,()>{Ok(self.result.error==ProgramParserError::None)}
    fn parse_error_invalid_request(&self)->Result<bool,()>{Ok(self.result.error==ProgramParserError::InvalidRequest)}
    fn parse_error_parse_failed(&self)->Result<bool,()>{Ok(self.result.error==ProgramParserError::ParseFailed)}
    fn parse_error_internal_error(&self)->Result<bool,()>{Ok(self.result.error==ProgramParserError::InternalError)}
    fn parse_error_untracked(&self)->Result<bool,()>{Ok(self.result.error==ProgramParserError::Untracked)}
    fn parse_error_unknown(&self)->Result<bool,()>{Ok(self.result.error==ProgramParserError::Unknown)}
    fn on_unexpected_from_deciding(&mut self)->Result<(),()>{self.unexpected();Ok(())} fn on_unexpected_from_parse_begin(&mut self)->Result<(),()>{self.unexpected();Ok(())} fn on_unexpected_from_dispatch_decision(&mut self)->Result<(),()>{self.unexpected();Ok(())} fn on_unexpected_from_text_emit(&mut self)->Result<(),()>{self.unexpected();Ok(())} fn on_unexpected_from_comment_emit(&mut self)->Result<(),()>{self.unexpected();Ok(())} fn on_unexpected_from_statement_parse_result_decision(&mut self)->Result<(),()>{self.unexpected();Ok(())} fn on_unexpected_from_expression_parse_result_decision(&mut self)->Result<(),()>{self.unexpected();Ok(())} fn on_unexpected_from_parsed(&mut self)->Result<(),()>{self.unexpected();Ok(())} fn on_unexpected_from_parse_failed(&mut self)->Result<(),()>{self.unexpected();Ok(())} fn on_unexpected_from_unexpected_event(&mut self)->Result<(),()>{self.unexpected();Ok(())}
}

fn map_statement_kind(k:TokenKind)->statement_parser::TokenType { use statement_parser::TokenType as T; match k {TokenKind::Eof=>T::Eof,TokenKind::Text=>T::Text,TokenKind::NumericLiteral=>T::NumericLiteral,TokenKind::StringLiteral=>T::StringLiteral,TokenKind::Identifier=>T::Identifier,TokenKind::Equals=>T::Equals,TokenKind::OpenParen=>T::OpenParen,TokenKind::CloseParen=>T::CloseParen,TokenKind::OpenStatement=>T::OpenStatement,TokenKind::CloseStatement=>T::CloseStatement,TokenKind::OpenExpression=>T::OpenExpression,TokenKind::CloseExpression=>T::CloseExpression,TokenKind::OpenSquareBracket=>T::OpenSquareBracket,TokenKind::CloseSquareBracket=>T::CloseSquareBracket,TokenKind::OpenCurlyBracket=>T::OpenCurlyBracket,TokenKind::CloseCurlyBracket=>T::CloseCurlyBracket,TokenKind::Comma=>T::Comma,TokenKind::Dot=>T::Dot,TokenKind::Colon=>T::Colon,TokenKind::Pipe=>T::Pipe,TokenKind::CallOperator=>T::CallOperator,TokenKind::AdditiveBinaryOperator=>T::AdditiveBinaryOperator,TokenKind::MultiplicativeBinaryOperator=>T::MultiplicativeBinaryOperator,TokenKind::ComparisonBinaryOperator=>T::ComparisonBinaryOperator,TokenKind::UnaryOperator=>T::UnaryOperator,TokenKind::Comment=>T::Comment}}
fn map_statement_error(e:statement_parser::StatementParserError,u:bool)->ProgramParserError {if u{return ProgramParserError::InternalError}match e{statement_parser::StatementParserError::None=>ProgramParserError::None,statement_parser::StatementParserError::InvalidRequest=>ProgramParserError::InvalidRequest,statement_parser::StatementParserError::ParseFailed=>ProgramParserError::ParseFailed,statement_parser::StatementParserError::InternalError=>ProgramParserError::InternalError,statement_parser::StatementParserError::Untracked=>ProgramParserError::Untracked}}
fn map_expression_error(e:expression_parser::ExpressionParserError)->ProgramParserError {match e{expression_parser::ExpressionParserError::None=>ProgramParserError::None,expression_parser::ExpressionParserError::InvalidRequest=>ProgramParserError::InvalidRequest,expression_parser::ExpressionParserError::ParseFailed=>ProgramParserError::ParseFailed,expression_parser::ExpressionParserError::InternalError=>ProgramParserError::InternalError,expression_parser::ExpressionParserError::Untracked=>ProgramParserError::Untracked}}

pub struct TextJinjaParserProgramParserActor { machine: TextJinjaParserProgramParserStateMachine<TextJinjaParserProgramParserContext> }
impl Default for TextJinjaParserProgramParserActor {fn default()->Self{Self::new()}}
impl TextJinjaParserProgramParserActor {
    #[must_use] pub fn new()->Self{Self{machine:TextJinjaParserProgramParserStateMachine::new(Default::default())}}
    pub fn process_event(&mut self,event:EventParseRuntime)->ProgramParseResult{if !self.machine.is(&TextJinjaParserProgramParserStates::Deciding){return self.process_unexpected();}self.machine.context_mut().input=event;self.machine.context_mut().result=ProgramParseResult::default();let _=self.machine.process_event(TextJinjaParserProgramParserEvents::EventParseRuntime(event));self.machine.context().result}
    pub fn process_unexpected(&mut self)->ProgramParseResult{self.machine.context_mut().unexpected();self.machine.set_state(TextJinjaParserProgramParserStates::UnexpectedEvent);self.machine.context().result}
    #[must_use] pub fn state(&self)->&TextJinjaParserProgramParserStates{self.machine.state()}
    #[must_use] pub fn is(&self,s:&TextJinjaParserProgramParserStates)->bool{self.machine.is(s)}
    #[must_use] pub fn context(&self)->&TextJinjaParserProgramParserContext{self.machine.context()}
}
pub type ProgramParser=TextJinjaParserProgramParserActor;
pub type Input=ProgramParserInput;
pub type Result=ProgramParseResult;
