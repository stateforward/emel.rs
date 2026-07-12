//! State machine scaffold port — not a stable public API.
//! Bodies are stubs (`todo!`) until contexts/guards/actions are ported from C++.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs
)]

use sml::sml;

// --- machine TextFormatter from emel.cpp/src/emel/text/formatter/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventScaffold;

sml! {
    TextFormatter {
        "idle"_s <= *"idle"_s + event<EventScaffold>,
        "idle"_s <= "idle"_s + unexpected_event<_>,
    }
}

/// Context for `TextFormatter` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextFormatterContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextFormatterStateMachineContext for TextFormatterContext {}
