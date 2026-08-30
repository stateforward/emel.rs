//! GBNF parsing and constrained-generation building blocks.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// A validated grammar source, retained as UTF-8 for parsing and diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrammarSource(String);

impl GrammarSource {
    /// Creates a grammar source from its textual representation.
    #[must_use]
    pub fn new(source: impl Into<String>) -> Self {
        Self(source.into())
    }

    /// Returns the original grammar text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
pub mod gbnf;
pub(crate) mod rule_parser;
pub(crate) mod sampler;
