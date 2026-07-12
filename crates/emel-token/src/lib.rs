//! Token representations and batching contracts.

#![forbid(unsafe_code)]

/// A vocabulary index.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TokenId(pub u32);
