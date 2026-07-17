//! Deterministic, fail-closed inventory extraction for the model parity program.
//!
//! The extractor reads both sides from immutable Git objects. C++ declarations come from real
//! Clang JSON ASTs, while Rust declarations come from `syn`. Text scanners only add orchestration,
//! test, assertion, fixture, and runtime-entrypoint rows; they never stand in for compiler proof.

mod ast_stream;
mod clang;
mod engine;
mod git;
mod hash;
mod scan;
mod schema;

use std::ffi::OsString;

pub use engine::{
    ExtractOptions, RustPreflightReport, extract, rust_preflight, validate_artifacts,
};
/// Validates an untrusted inventory TSV without exposing internal row representation.
///
/// # Errors
///
/// Returns a diagnostic when UTF-8, schema, vocabulary, or ID uniqueness is invalid.
pub fn validate_inventory_tsv(bytes: &[u8]) -> Result<(), String> {
    schema::parse_inventory_tsv(bytes).map(|_| ())
}

/// Closed parser selection for untrusted parity TSV data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TsvSchema {
    /// Nineteen-field primary or reconciled inventory rows.
    Inventory,
    /// Eleven-field proof-coverage rows.
    Coverage,
    /// Six-field open-gap rows.
    Gap,
}

/// Validates untrusted TSV bytes against one explicitly selected closed schema.
///
/// # Errors
///
/// Returns a diagnostic when UTF-8, headers, fields, vocabulary, or uniqueness are invalid.
pub fn validate_tsv(schema: TsvSchema, bytes: &[u8]) -> Result<(), String> {
    match schema {
        TsvSchema::Inventory => crate::schema::parse_inventory_tsv(bytes).map(|_| ()),
        TsvSchema::Coverage => crate::schema::parse_coverage_tsv(bytes).map(|_| ()),
        TsvSchema::Gap => crate::schema::parse_gap_tsv(bytes).map(|_| ()),
    }
}

/// Validates an untrusted concatenated filtered Clang AST stream with production bounds.
///
/// # Errors
///
/// Returns a diagnostic for malformed, truncated, over-depth, or otherwise invalid AST input.
pub fn validate_clang_ast_stream(bytes: &[u8]) -> Result<(), String> {
    ast_stream::validate_untrusted_filtered_ast(bytes)
}

/// Executes the command-line interface.
///
/// # Errors
///
/// Returns a diagnostic when arguments, immutable inputs, extraction, validation, or publication fail.
pub fn cli(arguments: impl IntoIterator<Item = OsString>) -> Result<(), String> {
    engine::cli(arguments)
}
use proc_macro2 as _;
