//! Actions for the formatter state-machine scaffold.

use super::context::Context;

/// No-op action corresponding to the pinned formatter scaffold.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Noop;

impl Noop {
    /// Applies the no-op action without changing context.
    pub const fn apply(self, _context: &mut Context) {}
}
