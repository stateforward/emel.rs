//! Source-aligned synchronous TextJinjaParserLexer actor.

#![allow(clippy::module_name_repetitions, clippy::missing_errors_doc, clippy::must_use_candidate, dead_code, missing_docs)]

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenKind { #[default] Eof=0, Text, NumericLiteral, StringLiteral, Identifier, Equals, OpenParen, CloseParen, OpenStatement, CloseStatement, OpenExpression, CloseExpression, OpenSquareBracket, CloseSquareBracket, OpenCurlyBracket, CloseCurlyBracket, Comma, Dot, Colon, Pipe, CallOperator, AdditiveBinaryOperator, MultiplicativeBinaryOperator, ComparisonBinaryOperator, UnaryOperator, Comment }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cursor<'a> { pub source: &'a str, pub offset: usize, pub token_index: usize, pub curly_bracket_depth: usize, pub last_token_type: TokenKind, pub last_block_rstrip: bool, pub last_block_can_trim_newline: bool }
impl<'a> Default for Cursor<'a> { fn default() -> Self { Self { source: "", offset: 0, token_index: 0, curly_bracket_depth: 0, last_token_type: TokenKind::CloseStatement, last_block_rstrip: false, last_block_can_trim_newline: false } } }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Token { pub kind: TokenKind, pub value: String, pub pos: usize }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NextDone<'a> { pub token: Token, pub has_token: bool, pub next_cursor: Cursor<'a> }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum Error { None=0, InvalidRequest=1, ParseFailed=2, InternalError=4, Untracked=8 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NextError { pub err: Error, pub error_pos: usize }
pub type DoneCallback<'a> = dyn FnMut(NextDone<'a>) -> bool + 'a;
pub type ErrorCallback<'a> = dyn FnMut(NextError) -> bool + 'a;

/// Bounded request copied from the pinned `lexer::event::next` contract.
pub struct EventNextRuntime<'a, 'cb> { pub cursor: Cursor<'a>, pub dispatch_done: Option<&'cb mut DoneCallback<'a>>, pub dispatch_error: Option<&'cb mut ErrorCallback<'cb>> }
impl<'a, 'cb> EventNextRuntime<'a, 'cb> {
    pub fn new(cursor: Cursor<'a>, done: &'cb mut DoneCallback<'a>, error: &'cb mut ErrorCallback<'cb>) -> Self { Self { cursor, dispatch_done: Some(done), dispatch_error: Some(error) } }
    pub fn with_callbacks(cursor: Cursor<'a>, done: Option<&'cb mut DoneCallback<'a>>, error: Option<&'cb mut ErrorCallback<'cb>>) -> Self { Self { cursor, dispatch_done: done, dispatch_error: error } }
}
impl core::fmt::Debug for EventNextRuntime<'_, '_> { fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.debug_struct("EventNextRuntime").field("cursor", &self.cursor).finish_non_exhaustive() } }

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TextJinjaParserLexerStates {
    #[default] Initialized, Scanning, TextBoundaryCandidateDecision, TextScanExec, TextOpeningBlockDecision, TextTrimOpeningBlockExec, TextTrimOpeningBlockResultDecision, TextMaterializeExec, TextFinalizeExec, TextFinalizeResultDecision, TextFinalizeTokenExec, TextEmitResultDecision, CommentCandidateDecision, CommentScanExec, CommentScanResultDecision, CommentFinalizeExec, CommentFinalizeResultDecision, CommentUnterminatedExec, CommentUnterminatedResultDecision, TrimPrefixScanExec, TrimPrefixEofExec, SpaceScanExec, SpaceEofExec, UnaryCandidateDecision, UnaryPrefixContextDecision, UnaryPrefixAllowedDecision, UnaryScanExec, MappingCandidateDecision, MappingCloseCurlyExec, MappingScanExec, StringScanExec, StringContentScanExec, StringContentPolicyDecision, StringScanResultDecision, StringMaterializeExec, StringStatusDecision, StringFinalizeExec, StringFinalizeResultDecision, StringUnterminatedExec, StringUnterminatedResultDecision, NumericScanExec, WordScanExec, InvalidCharExec, InvalidCharResultDecision,
}

#[derive(Debug, Default)]
pub struct TextJinjaParserLexerContext { pub source: String, pub size: usize, pub pos: usize, pub text_start: usize, pub text_end: usize, pub text_trim_probe: usize, pub string_start: usize, pub string_terminal: u8, pub handled: bool, pub error: Error, pub error_pos: usize, pub token: Option<Token> }
#[derive(Debug, Default)]
pub struct TextJinjaParserLexerStateMachine { state: TextJinjaParserLexerStates, context: TextJinjaParserLexerContext }
pub type TextJinjaParserLexer = TextJinjaParserLexerStateMachine;
pub type Lexer = TextJinjaParserLexerStateMachine;

impl TextJinjaParserLexerStateMachine {
    #[must_use] pub const fn new(context: TextJinjaParserLexerContext) -> Self { Self { state: TextJinjaParserLexerStates::Initialized, context } }
    #[must_use] pub const fn state(&self) -> TextJinjaParserLexerStates { self.state }
    #[must_use] pub fn is(&self, state: TextJinjaParserLexerStates) -> bool { self.state == state }
    #[must_use] pub const fn context(&self) -> &TextJinjaParserLexerContext { &self.context }
    pub fn context_mut(&mut self) -> &mut TextJinjaParserLexerContext { &mut self.context }
    pub fn process_event<'a, 'cb>(&mut self, event: EventNextRuntime<'a, 'cb>) -> bool {
        if event.dispatch_done.is_none() || event.dispatch_error.is_none() { emit_error(event.dispatch_error, Error::InvalidRequest, event.cursor.offset); return true; }
        if event.cursor.offset > event.cursor.source.len() { emit_error(event.dispatch_error, Error::InvalidRequest, event.cursor.offset); return true; }
        self.state = TextJinjaParserLexerStates::Scanning;
        scan(event); true
    }
    pub fn process_unexpected_event<'cb>(&mut self, error: Option<&'cb mut ErrorCallback<'cb>>, pos: usize) -> bool { emit_error(error, Error::InternalError, pos); self.state = TextJinjaParserLexerStates::Scanning; true }
}

fn scan<'a, 'cb>(event: EventNextRuntime<'a, 'cb>) {
    let c = event.cursor; let src = c.source; let mut pos = c.offset; let b = src.as_bytes();
    if pos == src.len() { emit_done(event.dispatch_done, eof(c, pos)); return; }
    if text_boundary(c.last_token_type) {
        let end = find_boundary(src, pos);
        if end > pos {
            let mut start = pos; let mut finish = end;
            if c.last_block_rstrip { while start < finish && trim_char(b[start]) { start += 1; } }
            if end < src.len() && trim_open(src, end) { while finish > start && trim_char(b[finish-1]) { finish -= 1; } }
            if c.last_block_can_trim_newline && b.get(start) == Some(&b'\n') { start += 1; }
            if start < finish { let token=Token{kind:TokenKind::Text,value:src[start..finish].to_owned(),pos:start}; emit_token(event.dispatch_done,c,token,end,false,false); return; }
            pos=end;
        }
    }
    if pos >= src.len() { emit_done(event.dispatch_done,eof(c,pos)); return; }
    // The pinned machine consumes bounded inter-token whitespace, and a '-' immediately
    // after an opening block is a trim prefix rather than an additive operator.
    if c.last_token_type == TokenKind::OpenStatement || c.last_token_type == TokenKind::OpenExpression {
        if b[pos] == b'-' { pos += 1; while pos < b.len() && space(b[pos]) { pos += 1; } }
    }
    while pos < b.len() && space(b[pos]) { pos += 1; }
    if pos >= src.len() { emit_done(event.dispatch_done,eof(c,pos)); return; }
    if b[pos] == b'{' && at(b,pos+1)==Some(b'#') {
        let start=pos; let content=pos+2; let close=src[content..].find("#}").map_or(src.len(),|n|content+n);
        if close+1 >= src.len() { emit_error(event.dispatch_error,Error::ParseFailed,close); return; }
        let tok=Token{kind:TokenKind::Comment,value:src[content..close].to_owned(),pos:start}; emit_token(event.dispatch_done,c,tok,close+2,false,false); return;
    }
    if let Some((kind,width,rstrip,trim)) = mapping(src,pos,c.curly_bracket_depth) {
        let tok=Token{kind,value:src[pos..pos+width].to_owned(),pos}; emit_token_with_flags(event.dispatch_done,c,tok,pos+width,rstrip,trim); return;
    }
    let unary = b[pos]==b'+' || (b[pos]==b'-' && at(b,pos+1).map_or(true,|x|x!=b'%'&&x!=b'}'));
    if unary {
        if !unary_allowed(c.last_token_type) { emit_error(event.dispatch_error,Error::ParseFailed,pos); return; }
        let start=pos; pos+=1; let nstart=pos; consume_number(b,&mut pos); let kind=if pos>nstart {TokenKind::NumericLiteral} else {TokenKind::UnaryOperator}; let tok=Token{kind,value:src[start..pos].to_owned(),pos:start}; emit_token(event.dispatch_done,c,tok,pos,false,false); return;
    }
    if b[pos]==b'\'' || b[pos]==b'"' {
        let start=pos; let term=b[pos]; pos+=1; let mut value=String::new();
        while pos<b.len() && b[pos]!=term { if b[pos]==b'\\' { pos+=1; if pos>=b.len(){emit_error(event.dispatch_error,Error::ParseFailed,pos);return;} let ch=match b[pos]{b'n'=>'\n',b't'=>'\t',b'r'=>'\r',b'b'=>'\x08',b'f'=>'\x0c',b'v'=>'\x0b',b'\\'=>'\\',b'\''=>'\'',b'"'=>'"',_=>{emit_error(event.dispatch_error,Error::ParseFailed,pos);return;}}; value.push(ch); pos+=1; } else { let n=b[pos]; if n.is_ascii() {value.push(n as char);} else {emit_error(event.dispatch_error,Error::ParseFailed,pos);return;} pos+=1; } }
        if pos>=b.len(){emit_error(event.dispatch_error,Error::ParseFailed,pos);return;} pos+=1; emit_token(event.dispatch_done,c,Token{kind:TokenKind::StringLiteral,value,pos:start},pos,false,false); return;
    }
    if b[pos].is_ascii_digit() { let start=pos; consume_number(b,&mut pos); emit_token(event.dispatch_done,c,Token{kind:TokenKind::NumericLiteral,value:src[start..pos].to_owned(),pos:start},pos,false,false); return; }
    if word(b[pos]) { let start=pos; pos+=1; while pos<b.len()&&word(b[pos]){pos+=1;} emit_token(event.dispatch_done,c,Token{kind:TokenKind::Identifier,value:src[start..pos].to_owned(),pos:start},pos,false,false); return; }
    emit_error(event.dispatch_error,Error::ParseFailed,pos);
}

fn emit_done<'a,'cb>(mut cb:Option<&'cb mut DoneCallback<'a>>, done:NextDone<'a>) { if let Some(f)=cb.as_mut(){let _=f(done);} }
fn emit_error<'cb>(mut cb:Option<&'cb mut ErrorCallback<'cb>>,err:Error,pos:usize){if let Some(f)=cb.as_mut(){let _=f(NextError{err,error_pos:pos});}}
fn eof<'a>(mut c:Cursor<'a>,pos:usize)->NextDone<'a>{c.offset=pos;NextDone{token:Token{kind:TokenKind::Eof,value:String::new(),pos},has_token:false,next_cursor:c}}
fn emit_token<'a,'cb>(cb:Option<&'cb mut DoneCallback<'a>>,c:Cursor<'a>,tok:Token,end:usize,r:bool,t:bool){emit_token_with_flags(cb,c,tok,end,r,t)}
fn emit_token_with_flags<'a,'cb>(cb:Option<&'cb mut DoneCallback<'a>>,c:Cursor<'a>,tok:Token,end:usize,_r:bool,_t:bool){let mut n=c;n.offset=end;n.token_index=c.token_index.saturating_add(1);n.last_token_type=tok.kind;n.last_block_can_trim_newline=matches!(tok.kind,TokenKind::CloseStatement|TokenKind::CloseExpression|TokenKind::Comment);n.last_block_rstrip=matches!(tok.kind,TokenKind::CloseStatement|TokenKind::CloseExpression)&&tok.value.len()>=3&&tok.value.as_bytes()[0]==b'-'&&tok.value.as_bytes().last()==Some(&b'}');if tok.kind==TokenKind::OpenExpression{n.curly_bracket_depth=0;}else if tok.kind==TokenKind::OpenCurlyBracket{n.curly_bracket_depth=c.curly_bracket_depth.saturating_add(1);}else if tok.kind==TokenKind::CloseCurlyBracket{n.curly_bracket_depth=c.curly_bracket_depth.saturating_sub(1);}emit_done(cb,NextDone{token:tok,has_token:true,next_cursor:n});}
fn at(b:&[u8],p:usize)->Option<u8>{b.get(p).copied()}
fn word(c:u8)->bool{c.is_ascii_alphanumeric()||c==b'_'}
fn trim_char(c:u8)->bool{matches!(c,b' '|b'\t'|b'\r'|b'\n')}
fn space(c:u8)->bool{matches!(c,b' '|b'\t'|b'\r'|b'\n'|b'\f'|b'\x0b')}
fn text_boundary(k:TokenKind)->bool{matches!(k,TokenKind::CloseStatement|TokenKind::CloseExpression|TokenKind::Comment)}
fn trim_open(s:&str,p:usize)->bool{at(s.as_bytes(),p+2)==Some(b'-')}
fn find_boundary(s:&str,mut p:usize)->usize{let b=s.as_bytes();while p+1<b.len(){if b[p]==b'{'&&matches!(b[p+1],b'%'|b'{'|b'#'){return p;}p+=1;}b.len()}
fn consume_number(b:&[u8],p:&mut usize){while *p<b.len()&&b[*p].is_ascii_digit(){*p+=1;}if *p<b.len()&&b[*p]==b'.'&&b.get(*p+1).is_some_and(u8::is_ascii_digit){*p+=1;while *p<b.len()&&b[*p].is_ascii_digit(){*p+=1;}}}
fn unary_allowed(k:TokenKind)->bool{matches!(k,TokenKind::OpenExpression|TokenKind::OpenStatement|TokenKind::OpenParen|TokenKind::Comma|TokenKind::Colon|TokenKind::Pipe|TokenKind::AdditiveBinaryOperator|TokenKind::MultiplicativeBinaryOperator|TokenKind::ComparisonBinaryOperator|TokenKind::Equals|TokenKind::UnaryOperator)}
fn mapping<'a>(s:&'a str,p:usize,depth:usize)->Option<(TokenKind,usize,bool,bool)>{let r=&s[p..];let a:[(&str,TokenKind,bool,bool);37]=[("{%-",TokenKind::OpenStatement,false,false),("-%}",TokenKind::CloseStatement,true,true),("{{-",TokenKind::OpenExpression,false,false),("-}}",TokenKind::CloseExpression,true,true),("{%",TokenKind::OpenStatement,false,false),("%}",TokenKind::CloseStatement,false,true),("{{",TokenKind::OpenExpression,false,false),("}}",TokenKind::CloseExpression,false,true),("<=",TokenKind::ComparisonBinaryOperator,false,false),(">=",TokenKind::ComparisonBinaryOperator,false,false),("==",TokenKind::ComparisonBinaryOperator,false,false),("!=",TokenKind::ComparisonBinaryOperator,false,false),("(",TokenKind::OpenParen,false,false),(")",TokenKind::CloseParen,false,false),("[",TokenKind::OpenSquareBracket,false,false),("]",TokenKind::CloseSquareBracket,false,false),("{",TokenKind::OpenCurlyBracket,false,false),("}",TokenKind::CloseCurlyBracket,false,false),(",",TokenKind::Comma,false,false),(".",TokenKind::Dot,false,false),(":",TokenKind::Colon,false,false),("|",TokenKind::Pipe,false,false),("<",TokenKind::ComparisonBinaryOperator,false,false),(">",TokenKind::ComparisonBinaryOperator,false,false),("+",TokenKind::AdditiveBinaryOperator,false,false),("-",TokenKind::AdditiveBinaryOperator,false,false),("~",TokenKind::AdditiveBinaryOperator,false,false),("*",TokenKind::MultiplicativeBinaryOperator,false,false),("/",TokenKind::MultiplicativeBinaryOperator,false,false),("%",TokenKind::MultiplicativeBinaryOperator,false,false),("=",TokenKind::Equals,false,false)];for&(x,k,rstrip,trim)in&a{if r.starts_with(x){if k==TokenKind::CloseExpression&&depth>0{continue;}return Some((k,x.len(),rstrip,trim));}}None}

pub mod event { pub use super::{Cursor,DoneCallback,Error,ErrorCallback,EventNextRuntime,NextDone,NextError,Token,TokenKind}; }
pub mod events { pub use super::{NextDone,NextError}; }
