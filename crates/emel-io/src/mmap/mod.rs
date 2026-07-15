//! Safe owned native file-mapping actor.

mod actor;
pub mod event;
mod platform;
mod sm;
#[cfg(test)]
mod tests;

pub use actor::Mapper;
