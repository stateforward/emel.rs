//! Scaffold for `emel.cpp/src/emel/kernel/sm.hpp`.
//!
//! C++ defines `using sm = any` / `using Kernel = any` — not a transition table.
//! TODO: port `emel::kernel::any` and re-export the real kernel SM alias from
//! `emel.cpp/src/emel/kernel/any.hpp` (referenced by `kernel/sm.hpp`).

#![allow(missing_docs, dead_code, clippy::missing_const_for_fn)]

/// Placeholder for C++ `emel::kernel::sm` (alias of `any`).
#[derive(Debug, Default)]
pub struct Sm;

impl Sm {
    /// Construct the kernel SM alias shell.
    ///
    /// TODO: convert from emel.cpp/src/emel/kernel/sm.hpp (alias of `any`).
    #[must_use]
    pub fn new() -> Self {
        // TODO: convert from emel.cpp/src/emel/kernel/sm.hpp / any.hpp
        Self
    }
}

/// C++ `emel::kernel::Kernel` type alias shell.
pub type Kernel = Sm;
