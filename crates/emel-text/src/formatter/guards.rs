//! Guards for the formatter state-machine scaffold.

use super::context::Context;

/// Guard that always permits the transition.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Always;

impl Always {
    /// Returns true for every formatter context.
    pub const fn check(self, _context: &Context) -> bool {
        true
    }
}
