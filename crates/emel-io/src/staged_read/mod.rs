//! Bounded caller-owned staged-copy actor.

mod actor;
pub mod event;
mod sm;
#[cfg(test)]
mod tests;

pub use actor::Stager;
