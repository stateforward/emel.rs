//! Typed vocabulary actor events and immutable callback views.

use core::fmt;

use emel_gguf::Loader as GgufLoader;
use emel_token::profile::event::{Model, PreId};

use super::Loader;

mod sealed {
    pub trait Sealed {}
}

pub trait Event<D: emel_token::profile::Dependency>: sealed::Sealed {
    type Output;
    #[doc(hidden)]
    fn dispatch(self, actor: &mut Loader<D>) -> Self::Output;
}

#[derive(Debug)]
pub struct Load<'a> {
    pub(crate) gguf: &'a mut GgufLoader,
}

impl<'a> Load<'a> {
    #[must_use]
    pub const fn new(gguf: &'a mut GgufLoader) -> Self {
        Self { gguf }
    }
}

impl sealed::Sealed for Load<'_> {}

impl<D: emel_token::profile::Dependency> Event<D> for Load<'_> {
    type Output = Result<Loaded, Error>;

    fn dispatch(self, actor: &mut Loader<D>) -> Self::Output {
        actor.load(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Loaded {
    token_count: u32,
    merge_count: u32,
}

impl Loaded {
    pub(crate) const fn new(token_count: u32, merge_count: u32) -> Self {
        Self {
            token_count,
            merge_count,
        }
    }
    #[must_use]
    pub const fn token_count(self) -> u32 {
        self.token_count
    }
    #[must_use]
    pub const fn merge_count(self) -> u32 {
        self.merge_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Phase {
    Reset,
    Model,
    Pre,
    Profile,
    TokenTypeCount,
    Tokens,
    Scores,
    TokenTypes,
    Merges,
    Charmap,
    SpecialIds,
    Flags,
    Patch,
    Final,
    Query,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Key {
    TokenizerModel,
    TokenizerPre,
    TokenTypeCount,
    Tokens,
    Scores,
    TokenTypes,
    Merges,
    Charmap,
    SpecialId,
    Flag,
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
    NotLoaded,
    Missing,
    Malformed,
    WrongKind,
    Query,
    Profile,
    Capacity,
    Count,
    Range,
    Unsupported,
    Unexpected,
    Internal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Error {
    phase: Phase,
    key: Key,
    kind: ErrorKind,
}

impl Error {
    pub(crate) const fn new(phase: Phase, key: Key, kind: ErrorKind) -> Self {
        Self { phase, key, kind }
    }
    #[must_use]
    pub const fn phase(self) -> Phase {
        self.phase
    }
    #[must_use]
    pub const fn key(self) -> Key {
        self.key
    }
    #[must_use]
    pub const fn kind(self) -> ErrorKind {
        self.kind
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "vocabulary {:?} error at {:?}: {:?}",
            self.key, self.phase, self.kind
        )
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpecialIds {
    pub bos: i32,
    pub eos: i32,
    pub eot: i32,
    pub eom: i32,
    pub unknown: i32,
    pub separator: i32,
    pub padding: i32,
    pub classification: i32,
    pub mask: i32,
    pub prefix: i32,
    pub suffix: i32,
    pub middle: i32,
    pub fim_pre: i32,
    pub fim_suf: i32,
    pub fim_mid: i32,
    pub fim_pad: i32,
    pub fim_rep: i32,
    pub fim_sep: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct Flags {
    pub add_bos: bool,
    pub add_eos: bool,
    pub add_sep: bool,
    pub add_space_prefix: bool,
    pub remove_extra_whitespaces: bool,
    pub escape_whitespaces: bool,
    pub treat_whitespace_as_suffix: bool,
    pub ignore_merges: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Info<'a> {
    pub model: Model,
    pub pre: PreId,
    pub model_name: &'a [u8],
    pub pre_name: &'a [u8],
    pub token_count: u32,
    pub token_type_count: u32,
    pub token_bytes: u32,
    pub merge_count: u32,
    pub merge_bytes: u32,
    pub charmap_bytes: u32,
    pub special_ids: SpecialIds,
    pub flags: Flags,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Token<'a> {
    pub text: &'a [u8],
    pub score: f32,
    pub r#type: i32,
    pub lstrip: bool,
    pub rstrip: bool,
}

#[derive(Debug)]
pub struct WithInfo<F, R> {
    pub(crate) callback: F,
    pub(crate) result: Result<Option<R>, Error>,
    pub(crate) marker: core::marker::PhantomData<R>,
}
impl<F, R> WithInfo<F, R> {
    #[must_use]
    pub const fn new(callback: F) -> Self {
        Self {
            callback,
            result: Err(Error::new(Phase::Query, Key::None, ErrorKind::Internal)),
            marker: core::marker::PhantomData,
        }
    }
}
impl<F, R> sealed::Sealed for WithInfo<F, R> where F: for<'a> FnMut(Info<'a>) -> R {}
impl<D: emel_token::profile::Dependency, F, R> Event<D> for WithInfo<F, R>
where
    F: for<'a> FnMut(Info<'a>) -> R,
{
    type Output = Result<R, Error>;
    fn dispatch(mut self, actor: &mut Loader<D>) -> Self::Output {
        actor.query(&mut self);
        self.result
            .and_then(|value| value.ok_or(Error::new(Phase::Query, Key::None, ErrorKind::Internal)))
    }
}

#[derive(Debug)]
pub struct WithToken<F, R> {
    pub(crate) index: u32,
    pub(crate) callback: F,
    pub(crate) result: Result<Option<R>, Error>,
    pub(crate) marker: core::marker::PhantomData<R>,
}
impl<F, R> WithToken<F, R> {
    #[must_use]
    pub const fn new(index: u32, callback: F) -> Self {
        Self {
            index,
            callback,
            result: Err(Error::new(Phase::Query, Key::None, ErrorKind::Internal)),
            marker: core::marker::PhantomData,
        }
    }
}
impl<F, R> sealed::Sealed for WithToken<F, R> where F: for<'a> FnMut(Token<'a>) -> R {}
impl<D: emel_token::profile::Dependency, F, R> Event<D> for WithToken<F, R>
where
    F: for<'a> FnMut(Token<'a>) -> R,
{
    type Output = Result<Option<R>, Error>;
    fn dispatch(mut self, actor: &mut Loader<D>) -> Self::Output {
        actor.query(&mut self);
        self.result
    }
}

#[derive(Debug)]
pub struct WithMerge<F, R> {
    pub(crate) index: u32,
    pub(crate) callback: F,
    pub(crate) result: Result<Option<R>, Error>,
    pub(crate) marker: core::marker::PhantomData<R>,
}
impl<F, R> WithMerge<F, R> {
    #[must_use]
    pub const fn new(index: u32, callback: F) -> Self {
        Self {
            index,
            callback,
            result: Err(Error::new(Phase::Query, Key::None, ErrorKind::Internal)),
            marker: core::marker::PhantomData,
        }
    }
}
impl<F, R> sealed::Sealed for WithMerge<F, R> where F: for<'a> FnMut(&'a [u8]) -> R {}
impl<D: emel_token::profile::Dependency, F, R> Event<D> for WithMerge<F, R>
where
    F: for<'a> FnMut(&'a [u8]) -> R,
{
    type Output = Result<Option<R>, Error>;
    fn dispatch(mut self, actor: &mut Loader<D>) -> Self::Output {
        actor.query(&mut self);
        self.result
    }
}

#[derive(Debug)]
pub struct WithCharmap<F, R> {
    pub(crate) callback: F,
    pub(crate) result: Result<Option<R>, Error>,
    pub(crate) marker: core::marker::PhantomData<R>,
}
impl<F, R> WithCharmap<F, R> {
    #[must_use]
    pub const fn new(callback: F) -> Self {
        Self {
            callback,
            result: Err(Error::new(Phase::Query, Key::None, ErrorKind::Internal)),
            marker: core::marker::PhantomData,
        }
    }
}
impl<F, R> sealed::Sealed for WithCharmap<F, R> where F: for<'a> FnMut(&'a [u8]) -> R {}
impl<D: emel_token::profile::Dependency, F, R> Event<D> for WithCharmap<F, R>
where
    F: for<'a> FnMut(&'a [u8]) -> R,
{
    type Output = Result<R, Error>;
    fn dispatch(mut self, actor: &mut Loader<D>) -> Self::Output {
        actor.query(&mut self);
        self.result
            .and_then(|value| value.ok_or(Error::new(Phase::Query, Key::None, ErrorKind::Internal)))
    }
}
