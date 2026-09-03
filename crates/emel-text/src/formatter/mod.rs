//! Pure formatter dependency and formatter state-machine scaffold.

pub mod actions;
pub mod context;
pub mod events;
pub mod format;
pub mod guards;
pub mod sm;

pub use format::{ChatMessage, FormatRequest, Formatter, RawFormatRequest, format_raw, format_raw_request, raw_formatter};
