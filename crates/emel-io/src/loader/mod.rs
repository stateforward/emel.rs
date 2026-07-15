//! Static composition actor for model tensor I/O strategies.

mod actor;
pub mod event;
mod sm;

pub use actor::Loader;

#[cfg(test)]
mod tests;
