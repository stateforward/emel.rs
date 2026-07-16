//! Ownership-safe tensor residency actor.

mod actor;
pub mod dependency;
pub mod event;
mod sm;
pub(crate) mod window;

pub use actor::Store;

#[cfg(test)]
mod bulk_tests;
#[cfg(test)]
mod io_tests;
#[cfg(test)]
mod tests;
