//! Pure formatter dependency and formatter state-machine scaffold.

pub mod actions;
pub mod context;
pub mod events;
pub mod format;
pub mod guards;
pub mod sm;

pub use format::{ChatMessage, FormatRequest, Formatter, format_raw, raw_formatter};
