//! Ownership-safe tensor residency actor.

mod actor;
pub mod event;
mod sm;
pub mod window;

pub use actor::Store;

#[cfg(test)]
mod bulk_tests;
#[cfg(test)]
mod tests;
