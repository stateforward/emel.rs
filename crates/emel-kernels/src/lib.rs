//! Portable and target-specialized numerical kernels.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

#[cfg(test)]
use allocation_counter as _;

pub mod capability;

/// Returns the default kernel dispatch name for this build target.
#[must_use]
pub const fn dispatch_name() -> &'static str {
    "portable"
}
pub(crate) mod aarch64;
pub(crate) mod x86_64;

/// Kernel SM type-alias scaffold (`kernel/sm.hpp`).
pub(crate) mod sm;
