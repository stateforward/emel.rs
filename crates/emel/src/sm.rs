//! Scaffold for `emel.cpp/src/emel/sm.hpp`.
//!
//! That header is infrastructure (schedulers, `co_sm`, completion policies),
//! not a domain transition table. Domain machines include it as the C++ SM base.
//!
//! TODO: port the utilities in emel.cpp/src/emel/sm.hpp into idiomatic Rust
//! helpers (or rely on `stateforward-sml` equivalents where they already exist).

#![allow(missing_docs, dead_code, clippy::missing_const_for_fn)]

/// Marker that the C++ `emel/sm.hpp` infrastructure is not yet ported.
#[derive(Debug, Default, Clone, Copy)]
pub struct Infrastructure;

impl Infrastructure {
    /// Placeholder entry for emel SM infrastructure.
    ///
    /// TODO: convert from `emel.cpp/src/emel/sm.hpp` (schedulers, `co_sm`, policies).
    #[must_use]
    pub const fn placeholder() -> Self {
        Self
    }
}
