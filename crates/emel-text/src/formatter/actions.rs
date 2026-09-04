//! Actions for the stateless formatter lifecycle.

use super::context::Context;

/// No-op action corresponding to the formatter lifecycle transition.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Noop;

impl Noop {
    /// Applies the no-op action without changing context.
    #[allow(clippy::unused_self)]
    pub const fn apply(self, _context: &mut Context) {}
}
