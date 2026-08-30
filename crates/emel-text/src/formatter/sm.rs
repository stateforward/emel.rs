//! State-machine scaffold for the pure formatter dependency.

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

use super::events::Scaffold;

/// Runtime event shell retained by the pinned formatter scaffold.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventScaffold;

sml! {
    TextFormatter {
        "idle"_s <= *"idle"_s + event<Scaffold>,
        "idle"_s <= "idle"_s + unexpected_event<_>,
    }
}

/// Empty context: the injected formatter is pure and stateless.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TextFormatterContext;

impl TextFormatterStateMachineContext for TextFormatterContext {}
