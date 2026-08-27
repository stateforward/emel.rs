//! Module for `conditioner` state machines.
#![allow(clippy::redundant_pub_crate)]
// The generated SML port is retained as an internal migration artifact until
// its callbacks are fully implemented.  The public boundary is `Conditioner`
// in the crate root; consumers must not reach scaffold internals.
pub(crate) mod sm;
