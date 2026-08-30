//! Empty context retained for the formatter state-machine scaffold.

/// Formatter state-machine context has no state: formatting is injected and pure.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Context;
