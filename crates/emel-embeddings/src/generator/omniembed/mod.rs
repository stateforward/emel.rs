//! Scaffold for `emel.cpp/src/emel/embeddings/generator/omniembed/sm.hpp`.
//!
//! C++ specializes the shared embeddings generator SM:
//! `struct model : emel::embeddings::generator::model<route> {}`
//! and `using sm = emel::embeddings::generator::basic_sm<route>`.
//!
//! TODO: port `route.hpp` and wire
//! `emel.cpp/src/emel/embeddings/generator/sm.hpp` with the omniembed route.

#![allow(missing_docs, dead_code, clippy::missing_const_for_fn)]

/// Omniembed route marker (TODO: fields/logic from route.hpp).
#[derive(Debug, Default, Clone, Copy)]
pub struct Route;

/// C++ `emel::embeddings::generator::omniembed::model` shell.
#[derive(Debug, Default)]
pub struct Model;

/// C++ `emel::embeddings::generator::omniembed::sm` alias shell.
#[derive(Debug, Default)]
pub struct Sm;

impl Sm {
    /// Construct the omniembed embeddings-generator SM.
    ///
    /// TODO: convert from `emel.cpp/src/emel/embeddings/generator/omniembed/sm.hpp`
    /// (`basic_sm<route>` over `generator/sm.hpp`).
    #[must_use]
    pub fn new() -> Self {
        // TODO: convert from emel.cpp/src/emel/embeddings/generator/omniembed/sm.hpp
        // and emel.cpp/src/emel/embeddings/generator/sm.hpp
        Self
    }
}
