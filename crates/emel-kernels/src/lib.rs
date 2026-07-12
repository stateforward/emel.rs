//! Portable and target-specialized numerical kernels.

#![forbid(unsafe_code)]

/// Returns the default kernel dispatch name for this build target.
#[must_use]
pub const fn dispatch_name() -> &'static str {
    "portable"
}
