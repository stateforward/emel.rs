use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use crate::git::{self, Blob};
use crate::hash::sha256;
use crate::scan::{
    Extracted, cpp_emel_io_sm_forward_declarations, cpp_test_assertion_invocations,
    cpp_transition_rows, scanners,
};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use std::io::Cursor;

pub const CLANG: &str = "/usr/bin/clang++";
pub const CLANG_SHA256: &str = "12bed4523661307059b879b9b54e77a73176e9d27d27a0e40363271d8f0668ba";
pub const CLANG_IDENTITY: &str =
    "Apple clang version 16.0.0 (clang-1600.0.26.6) arm64-apple-darwin25.4.0";
// This is the SHA-256 of the canonical compilation manifest, not of the raw
// `compile_commands.json` bytes. CMake records absolute source and build
// paths in that file, so hashing its bytes would bind this proof to one
// workstation directory rather than to the compiler invocation semantics.
pub const COMPILE_COMMANDS_SHA256: &str =
    "6542a28009363497df7f758973dc89b1f7ef3f9a5da8c35f0a9001935e6ae378";
pub const EXTERNAL_INCLUDES_SHA256: &str =
    "0a0df295baabcd40f04baaf8164a897e81160b101e4ea63f0e86abd2d4998d11";
const EXTRACTION_WALL_LIMIT: Duration = Duration::from_secs(300);
const JSON_AST_FILTER: &str = "emel";
const TEXTUAL_AST_FILTER: &str = "emel::model";
const CROSS_DOMAIN_TEXTUAL_AST_FILTER: &str = "emel::io";
const TEST_TEXTUAL_AST_FILTER: &str = "emel_window_test";
const REQUIRED_DETAILED_TRANSLATION_UNIT: &str = "src/emel/model/generation/any.cpp";
const PINNED_MODEL_IO_FORWARD_DECLARATIONS: &[&str] = &[
    "emel::io::loader::sm",
    "emel::io::mmap::sm",
    "emel::io::read::sm",
    "emel::io::staged_read::sm",
];
const MATERIALIZED_SOURCE_ROOTS: &[&str] = &[
    "include",
    "src",
    "tests",
    "third_party/doctest",
    "tools/bench",
];

#[derive(Clone, Debug, Deserialize)]
struct CompileEntry {
    #[serde(default)]
    directory: Option<String>,
    file: String,
    command: Option<String>,
    arguments: Option<Vec<String>>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct CanonicalCompileEntry {
    directory: String,
    file: String,
    arguments: Vec<String>,
}

#[derive(Clone, Debug)]
struct RootReplacement {
    raw: String,
    placeholder: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Coverage {
    pub path: String,
    pub line: u64,
    pub column: u64,
    pub begin_line: u64,
    pub begin_column: u64,
    pub end_line: u64,
    pub end_column: u64,
    pub kind: String,
    pub name: String,
    pub signature: String,
    pub node_id: String,
}

pub struct CppOutput {
    pub items: Vec<Extracted>,
    pub coverage: Vec<Coverage>,
    pub external_includes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DeclarationKey {
    path: String,
    kind: String,
    line: u64,
    column: u64,
    begin_line: u64,
    begin_column: u64,
    end_line: u64,
    end_column: u64,
    qualified_identity: String,
    canonical_signature: String,
}

#[allow(clippy::too_many_lines)]
pub fn extract(
    repo: &Path,
    commit: &str,
    compile_commands: &Path,
    blobs: &[Blob],
) -> Result<CppOutput, String> {
    let entries =
        retained_compile_database_with(compile_commands, repo, COMPILE_COMMANDS_SHA256, || {})?;
    let source_cross_manifest = source_cross_domain_manifest(blobs)?;
    let expected_units = expected_translation_units(blobs)?;
    let selected = select_translation_units(repo, entries, &expected_units)?;
    verify_compiler()?;

    let temporary = tempfile::tempdir()
        .map_err(|error| format!("cannot create temporary directory: {error}"))?;
    let materialized = temporary.path().join("source");
    fs::create_dir(&materialized)
        .map_err(|error| format!("cannot create materialized root: {error}"))?;
    let materialization_deadline = Instant::now() + EXTRACTION_WALL_LIMIT;
    let materialized_blobs = git::blobs(
        repo,
        commit,
        MATERIALIZED_SOURCE_ROOTS,
        materialization_deadline,
    )?;
    materialize(&materialized_blobs, &materialized)?;
    let external_roots = external_include_roots(&selected)?;
    let (external_rewrites, external_includes) =
        materialize_external_includes(&external_roots, temporary.path())?;
    if sha256(&external_includes) != EXTERNAL_INCLUDES_SHA256 {
        return Err("external dependency include digest drift".to_owned());
    }
    let source_root = repo.to_string_lossy();
    let target_root = materialized.to_string_lossy();
    let mut items = Vec::new();
    let mut coverage = Vec::new();
    let mut expectations = BTreeMap::<DeclarationKey, usize>::new();
    let mut cross_named_manifest = BTreeMap::<String, usize>::new();
    let deadline = Instant::now() + EXTRACTION_WALL_LIMIT;
    for entry in &selected {
        let arguments = rewritten_arguments(entry, &source_root, &target_root, &external_rewrites)?;
        crate::ast_stream::run_clang(
            &arguments,
            temporary.path(),
            &materialized,
            repo,
            &mut items,
            &mut coverage,
            deadline,
        )?;
        collect_textual_expectations(
            &arguments,
            temporary.path(),
            &materialized,
            repo,
            deadline,
            &mut expectations,
            &mut cross_named_manifest,
        )?;
    }

    let headers = blobs
        .iter()
        .filter(|blob| extension_is(&blob.path, "hpp"))
        .map(|blob| blob.path.clone())
        .collect::<BTreeSet<_>>();
    let covered = coverage
        .iter()
        .map(|row| row.path.clone())
        .collect::<BTreeSet<_>>();
    let missing = headers.difference(&covered).cloned().collect::<Vec<_>>();
    if !missing.is_empty() {
        let audit = materialized.join("emel_model_inventory_audit.cpp");
        let mut body = String::new();
        for path in &missing {
            writeln!(body, "#include \"{path}\"").expect("writing to String cannot fail");
        }
        fs::write(&audit, body)
            .map_err(|error| format!("cannot write audit translation unit: {error}"))?;
        let base = selected.first().ok_or("no compile commands selected")?;
        let mut arguments =
            rewritten_arguments(base, &source_root, &target_root, &external_rewrites)?;
        if let Some(last) = arguments.last_mut() {
            *last = audit.to_string_lossy().into_owned();
        }
        crate::ast_stream::run_clang(
            &arguments,
            temporary.path(),
            &materialized,
            repo,
            &mut items,
            &mut coverage,
            deadline,
        )?;
        collect_textual_expectations(
            &arguments,
            temporary.path(),
            &materialized,
            repo,
            deadline,
            &mut expectations,
            &mut cross_named_manifest,
        )?;
    }

    coverage.sort();
    coverage.dedup();
    validate_required_translation_unit_ast_coverage(&expected_units, &coverage)?;
    validate_cross_domain_manifests(&source_cross_manifest, &cross_named_manifest, &coverage)?;
    validate_declaration_coverage(&expectations, &coverage)?;
    let final_covered = coverage
        .iter()
        .map(|row| row.path.clone())
        .collect::<BTreeSet<_>>();
    let absent = headers
        .difference(&final_covered)
        .cloned()
        .collect::<Vec<_>>();
    if !absent.is_empty() {
        return Err(format!(
            "pinned headers have no real Clang AST declarations: {}",
            absent.join(", ")
        ));
    }
    append_scanner_items(blobs, &mut items)?;
    Ok(CppOutput {
        items,
        coverage,
        external_includes,
    })
}

fn append_scanner_items(blobs: &[Blob], items: &mut Vec<Extracted>) -> Result<(), String> {
    for blob in blobs {
        if let Ok(source) = std::str::from_utf8(&blob.bytes) {
            cpp_transition_rows(&blob.path, source, items)?;
            cpp_test_assertion_invocations(&blob.path, source, items)?;
            scanners(&blob.path, source, "cpp", items)?;
        } else if extension_is(&blob.path, "cpp") || extension_is(&blob.path, "hpp") {
            return Err(format!("{} is not UTF-8", blob.path));
        }
    }
    Ok(())
}

fn collect_textual_expectations(
    arguments: &[String],
    temporary: &Path,
    materialized: &Path,
    original: &Path,
    deadline: Instant,
    expectations: &mut BTreeMap<DeclarationKey, usize>,
    cross_named_expectations: &mut BTreeMap<String, usize>,
) -> Result<(), String> {
    let list_arguments = ast_list_arguments(arguments)?;
    let list_capture = crate::ast_stream::run_clang_capture(&list_arguments, temporary, deadline)?;
    let list_reader = BufReader::new(
        list_capture
            .file
            .reopen()
            .map_err(|error| format!("cannot reopen Clang AST name manifest: {error}"))?,
    );
    let listed_names = parse_ast_list(list_reader)?;
    let mut textual = independent_textual_arguments(arguments, TEXTUAL_AST_FILTER)?;
    let dump = textual
        .iter_mut()
        .find(|argument| argument.as_str() == "-ast-dump=json")
        .ok_or("JSON AST argument missing")?;
    *dump = "-ast-dump".into();
    let output = crate::ast_stream::run_clang_capture(&textual, temporary, deadline)?;
    let unit = arguments.last().map_or("<missing-unit>", String::as_str);
    eprintln!(
        "oracle-bytes unit={unit:?} ast_list={} textual={}",
        list_capture.bytes, output.bytes
    );
    let reader = BufReader::new(
        output
            .file
            .reopen()
            .map_err(|error| format!("cannot reopen textual Clang AST capture: {error}"))?,
    );
    let mut per_unit = parse_textual_expectations_reader(reader, unit, materialized, original)?;
    let mut detailed_all_manifest = per_unit.all_named_manifest.clone();
    let mut cross_textual =
        independent_textual_arguments(arguments, CROSS_DOMAIN_TEXTUAL_AST_FILTER)?;
    let dump = cross_textual
        .iter_mut()
        .find(|argument| argument.as_str() == "-ast-dump=json")
        .ok_or("JSON AST argument missing")?;
    *dump = "-ast-dump".into();
    let cross_output = crate::ast_stream::run_clang_capture(&cross_textual, temporary, deadline)?;
    let cross_reader = BufReader::new(
        cross_output
            .file
            .reopen()
            .map_err(|error| format!("cannot reopen cross-domain textual AST capture: {error}"))?,
    );
    let cross = parse_textual_expectations_reader(cross_reader, unit, materialized, original)?;
    let TextualExpectations {
        declarations: cross_declarations,
        named_manifest: cross_manifest,
        all_named_manifest: cross_all_manifest,
    } = cross;
    merge_manifest_max(cross_named_expectations, cross_manifest.clone());
    merge_manifest_max(&mut per_unit.named_manifest, cross_manifest);
    merge_manifest_max(&mut detailed_all_manifest, cross_all_manifest);
    for (key, count) in cross_declarations {
        per_unit
            .declarations
            .entry(key)
            .and_modify(|existing| *existing = (*existing).max(count))
            .or_insert(count);
    }
    let mut test_textual = independent_textual_arguments(arguments, TEST_TEXTUAL_AST_FILTER)?;
    let dump = test_textual
        .iter_mut()
        .find(|argument| argument.as_str() == "-ast-dump=json")
        .ok_or("JSON AST argument missing")?;
    *dump = "-ast-dump".into();
    let test_output = crate::ast_stream::run_clang_capture(&test_textual, temporary, deadline)?;
    eprintln!(
        "oracle-supplemental-bytes unit={unit:?} cross_domain={} test={}",
        cross_output.bytes, test_output.bytes
    );
    let test_reader = BufReader::new(
        test_output
            .file
            .reopen()
            .map_err(|error| format!("cannot reopen test textual AST capture: {error}"))?,
    );
    let test = parse_textual_expectations_reader(test_reader, unit, materialized, original)?;
    merge_manifest_max(&mut per_unit.named_manifest, test.named_manifest);
    merge_manifest_max(&mut detailed_all_manifest, test.all_named_manifest);
    for (key, count) in test.declarations {
        per_unit
            .declarations
            .entry(key)
            .and_modify(|existing| *existing = (*existing).max(count))
            .or_insert(count);
    }
    validate_named_manifest(&listed_names, &detailed_all_manifest, unit)?;
    validate_owned_manifest_is_listed(&listed_names, &per_unit.named_manifest, unit)?;
    for (key, count) in per_unit {
        expectations
            .entry(key)
            .and_modify(|existing| *existing = (*existing).max(count))
            .or_insert(count);
    }
    Ok(())
}

fn source_cross_domain_manifest(blobs: &[Blob]) -> Result<BTreeMap<String, usize>, String> {
    let mut forward_manifest = BTreeMap::<String, usize>::new();
    for blob in blobs.iter().filter(|blob| {
        blob.path.starts_with("src/emel/model/") || blob.path.starts_with("tests/model/")
    }) {
        let source = std::str::from_utf8(&blob.bytes)
            .map_err(|error| format!("{} is not UTF-8: {error}", blob.path))?;
        for declaration in cpp_emel_io_sm_forward_declarations(source)? {
            *forward_manifest.entry(declaration).or_insert(0) += 1;
        }
    }
    let expected_forward = PINNED_MODEL_IO_FORWARD_DECLARATIONS
        .iter()
        .map(|name| ((*name).to_owned(), 1usize))
        .collect::<BTreeMap<_, _>>();
    if forward_manifest != expected_forward {
        return Err(manifest_mismatch(
            "pinned model emel::io forward-declaration source",
            &expected_forward,
            &forward_manifest,
        ));
    }

    let mut named_manifest = BTreeMap::new();
    for (declaration, count) in forward_manifest {
        let component = declaration
            .strip_prefix("emel::io::")
            .and_then(|suffix| suffix.strip_suffix("::sm"))
            .ok_or_else(|| format!("invalid emel::io forward declaration {declaration:?}"))?;
        for name in [
            "emel::io".to_owned(),
            format!("emel::io::{component}"),
            declaration,
        ] {
            *named_manifest.entry(name).or_insert(0) += count;
        }
    }
    Ok(named_manifest)
}

fn validate_cross_domain_manifests(
    source: &BTreeMap<String, usize>,
    filtered: &BTreeMap<String, usize>,
    coverage: &[Coverage],
) -> Result<(), String> {
    if source != filtered {
        return Err(manifest_mismatch(
            "model-owned cross-domain filtered textual AST",
            source,
            filtered,
        ));
    }
    let expected_forward = source
        .iter()
        .filter(|(name, _)| name.ends_with("::sm"))
        .map(|(name, count)| (name.clone(), *count))
        .collect::<BTreeMap<_, _>>();
    let json_forward = cross_domain_json_forward_manifest(coverage);
    if expected_forward != json_forward {
        return Err(manifest_mismatch(
            "model-owned cross-domain JSON AST",
            &expected_forward,
            &json_forward,
        ));
    }
    Ok(())
}

fn cross_domain_json_forward_manifest(coverage: &[Coverage]) -> BTreeMap<String, usize> {
    let mut manifest = BTreeMap::new();
    for row in coverage.iter().filter(|row| {
        (row.path.starts_with("src/emel/model/") || row.path.starts_with("tests/model/"))
            && matches!(row.kind.as_str(), "CXXRecordDecl" | "RecordDecl")
    }) {
        let name = row
            .name
            .split_once("::<declaration@")
            .map_or(row.name.as_str(), |(name, _)| name);
        if name.starts_with("emel::io::") && name.ends_with("::sm") {
            *manifest.entry(name.to_owned()).or_insert(0) += 1;
        }
    }
    manifest
}

fn manifest_mismatch(
    label: &str,
    expected: &BTreeMap<String, usize>,
    actual: &BTreeMap<String, usize>,
) -> String {
    let names = expected
        .keys()
        .chain(actual.keys())
        .collect::<BTreeSet<_>>();
    let differences = names
        .into_iter()
        .filter_map(|name| {
            let expected = expected.get(name).copied().unwrap_or(0);
            let actual = actual.get(name).copied().unwrap_or(0);
            (expected != actual).then(|| format!("{name:?}: expected={expected}, actual={actual}"))
        })
        .take(12)
        .collect::<Vec<_>>();
    format!("{label} manifest mismatch: {}", differences.join("; "))
}

fn merge_manifest_max(target: &mut BTreeMap<String, usize>, source: BTreeMap<String, usize>) {
    for (name, count) in source {
        target
            .entry(name)
            .and_modify(|existing| *existing = (*existing).max(count))
            .or_insert(count);
    }
}

fn parse_ast_list(reader: impl BufRead) -> Result<BTreeMap<String, usize>, String> {
    let mut names = BTreeMap::new();
    for line in reader.lines() {
        let line =
            line.map_err(|error| format!("cannot decode Clang AST name manifest: {error}"))?;
        if (line == "emel::model"
            || line.starts_with("emel::model::")
            || line == "emel::io"
            || line.starts_with("emel::io::")
            || line == TEST_TEXTUAL_AST_FILTER
            || line.starts_with("emel_window_test::"))
            && !line.contains("(anonymous")
        {
            *names.entry(line).or_insert(0) += 1;
        }
    }
    Ok(names)
}

fn validate_named_manifest(
    listed: &BTreeMap<String, usize>,
    detailed: &BTreeMap<String, usize>,
    unit: &str,
) -> Result<(), String> {
    if listed == detailed {
        return Ok(());
    }
    let names = listed
        .keys()
        .chain(detailed.keys())
        .collect::<BTreeSet<_>>();
    let differences = names
        .into_iter()
        .filter_map(|name| {
            let listed = listed.get(name).copied().unwrap_or(0);
            let detailed = detailed.get(name).copied().unwrap_or(0);
            (listed != detailed)
                .then(|| format!("{name:?}: ast-list={listed}, detailed={detailed}"))
        })
        .take(12)
        .collect::<Vec<_>>();
    Err(format!(
        "independent Clang named-declaration manifest mismatch for {unit:?}: {}",
        differences.join("; ")
    ))
}

fn validate_owned_manifest_is_listed(
    listed: &BTreeMap<String, usize>,
    owned: &BTreeMap<String, usize>,
    unit: &str,
) -> Result<(), String> {
    let missing = owned
        .iter()
        .filter_map(|(name, count)| {
            let listed_count = listed.get(name).copied().unwrap_or(0);
            (listed_count < *count)
                .then(|| format!("{name:?}: ast-list={listed_count}, model-owned={count}"))
        })
        .take(12)
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "model-owned named declarations are absent from the independent Clang manifest for \
             {unit:?}: {}",
            missing.join("; ")
        ))
    }
}

fn independent_textual_arguments(
    arguments: &[String],
    textual_filter: &str,
) -> Result<Vec<String>, String> {
    if JSON_AST_FILTER == textual_filter {
        return Err("JSON and textual AST filters must remain independent".into());
    }
    let mut textual = arguments.to_vec();
    let filter = ["-Xclang", "-ast-dump-filter", "-Xclang", JSON_AST_FILTER];
    let start = textual
        .windows(filter.len())
        .position(|window| window.iter().map(String::as_str).eq(filter.iter().copied()))
        .ok_or("filtered JSON AST arguments lack the pinned Clang filter sequence")?;
    if textual[start + 3] != JSON_AST_FILTER {
        return Err("JSON AST filter position drift".into());
    }
    textual[start + 3] = textual_filter.into();
    if textual
        .windows(filter.len())
        .filter(|window| window.get(1).is_some_and(|item| item == "-ast-dump-filter"))
        .count()
        != 1
    {
        return Err("multiple Clang AST filters are unsupported".to_owned());
    }
    Ok(textual)
}

fn ast_list_arguments(arguments: &[String]) -> Result<Vec<String>, String> {
    let mut listed = arguments.to_vec();
    let filter = ["-Xclang", "-ast-dump-filter", "-Xclang", JSON_AST_FILTER];
    let start = listed
        .windows(filter.len())
        .position(|window| window.iter().map(String::as_str).eq(filter.iter().copied()))
        .ok_or("filtered JSON AST arguments lack the pinned Clang filter sequence")?;
    listed.drain(start..start + filter.len());
    let dump = listed
        .iter_mut()
        .find(|argument| argument.as_str() == "-ast-dump=json")
        .ok_or("JSON AST argument missing")?;
    *dump = "-ast-list".into();
    if listed.iter().any(|argument| argument == "-ast-dump-filter") {
        return Err("Clang AST name manifest must remain unfiltered".into());
    }
    Ok(listed)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextScopeKind {
    Namespace,
    Record,
    Enum,
    Function,
    Lexical,
    Other,
}

#[derive(Clone, Debug)]
struct TextScope {
    kind: TextScopeKind,
    identity: Option<String>,
    manifest_identity: Option<String>,
    template_parameters: BTreeSet<String>,
    path: Option<String>,
    line: u64,
}

#[derive(Debug)]
struct TextualExpectations {
    declarations: BTreeMap<DeclarationKey, usize>,
    named_manifest: BTreeMap<String, usize>,
    all_named_manifest: BTreeMap<String, usize>,
}

impl std::ops::Deref for TextualExpectations {
    type Target = BTreeMap<DeclarationKey, usize>;

    fn deref(&self) -> &Self::Target {
        &self.declarations
    }
}

impl IntoIterator for TextualExpectations {
    type Item = (DeclarationKey, usize);
    type IntoIter = std::collections::btree_map::IntoIter<DeclarationKey, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.declarations.into_iter()
    }
}

#[derive(Clone, Debug)]
struct TextPoint {
    path: Option<String>,
    line: u64,
    column: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextHeaderRange {
    Absent,
    InvalidSLoc,
    Span { start: usize, end: usize },
}

#[allow(clippy::too_many_lines)]
#[cfg(test)]
fn parse_textual_expectations(
    text: &str,
    unit: &str,
    materialized: &Path,
    original: &Path,
) -> Result<TextualExpectations, String> {
    parse_textual_expectations_reader(Cursor::new(text.as_bytes()), unit, materialized, original)
}

#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
fn parse_textual_expectations_reader(
    reader: impl BufRead,
    unit: &str,
    materialized: &Path,
    original: &Path,
) -> Result<TextualExpectations, String> {
    let mut output = BTreeMap::new();
    let mut named_manifest = BTreeMap::<String, usize>::new();
    let mut all_named_manifest = BTreeMap::<String, usize>::new();
    let mut scopes = Vec::<TextScope>::new();
    let mut owned_path = None::<String>;
    let mut owned_line = 0u64;
    for line in reader.lines() {
        let line = line.map_err(|error| format!("cannot decode textual Clang AST: {error}"))?;
        let Some(kind_start) = line.find(|character: char| character.is_ascii_alphabetic()) else {
            continue;
        };
        let rest = &line[kind_start..];
        let kind = rest.split_whitespace().next().unwrap_or_default();
        if kind == "Dumping" || kind.is_empty() {
            continue;
        }
        let depth = kind_start / 2;
        scopes.truncate(depth);
        let external_parent = scopes.last().is_some_and(|scope| {
            scope
                .path
                .as_deref()
                .is_some_and(|path| normalize_text_owner(path, materialized, original).is_none())
        });
        let mut current_path = if external_parent {
            scopes.last().and_then(|scope| scope.path.clone())
        } else {
            owned_path.clone()
        };
        let mut current_line = if external_parent {
            scopes.last().map_or(0, |scope| scope.line)
        } else {
            owned_line
        };
        let local = scopes
            .iter()
            .any(|scope| matches!(scope.kind, TextScopeKind::Function | TextScopeKind::Lexical));
        let implicit = textual_has_flag(rest, "implicit");
        let included = kind.ends_with("Decl")
            && !implicit
            && !crate::ast_stream::excluded_declaration_kind(kind)
            && !local;
        let scope_kind = textual_scope_kind(kind);
        let (range_start, range_end) = match textual_header_range(rest)? {
            TextHeaderRange::Span { start, end } => (start, end),
            TextHeaderRange::Absent => {
                let possibly_owned = current_path.as_deref().is_none_or(|path| {
                    normalize_text_owner(path, materialized, original).is_some()
                });
                if (included || matches!(scope_kind, TextScopeKind::Lexical)) && possibly_owned {
                    return Err(format!(
                        "textual inventory node lacks anchored source range: unit={unit:?} \
                         kind={kind:?} local={local} included={included} \
                         current_path={current_path:?} scopes={scopes:?} raw={line:?}"
                    ));
                }
                scopes.push(TextScope {
                    kind: scope_kind,
                    identity: None,
                    manifest_identity: None,
                    template_parameters: BTreeSet::new(),
                    path: current_path.clone(),
                    line: current_line,
                });
                continue;
            }
            TextHeaderRange::InvalidSLoc => {
                let possibly_owned = current_path.as_deref().is_none_or(|path| {
                    normalize_text_owner(path, materialized, original).is_some()
                });
                if (included || matches!(scope_kind, TextScopeKind::Lexical)) && possibly_owned {
                    return Err(format!(
                        "textual inventory node has invalid source-location sentinel: unit={unit:?} \
                         kind={kind:?} local={local} included={included} \
                         current_path={current_path:?} scopes={scopes:?} raw={line:?}"
                    ));
                }
                scopes.push(TextScope {
                    kind: scope_kind,
                    identity: None,
                    manifest_identity: None,
                    template_parameters: BTreeSet::new(),
                    path: current_path.clone(),
                    line: current_line,
                });
                continue;
            }
        };
        let range = &rest[range_start + 1..range_end];
        let (begin_text, end_text) = if let Some((begin, end)) = range.split_once(',') {
            if begin.trim().is_empty() || end.trim().is_empty() {
                return Err(format!(
                    "textual AST source range has an empty endpoint: unit={unit:?} \
                     kind={kind:?} range={range:?} raw={line:?}"
                ));
            }
            (begin, end)
        } else {
            (range, range)
        };
        let tail = rest[range_end + 1..].trim();
        let location_token = textual_location_token(tail);
        let location_anchor = if location_token.starts_with("line:")
            || location_token.starts_with('/')
        {
            Some(
                parse_text_location_token(location_token, current_path.as_deref(), current_line)
                    .map_err(|error| {
                        format!(
                            "textual location-anchor parse failed: {error}; unit={unit:?} \
                         kind={kind:?} current_path={current_path:?} range={range:?} \
                         tail={tail:?} raw={line:?}"
                        )
                    })?,
            )
        } else {
            None
        };
        let begin_path = location_anchor
            .as_ref()
            .and_then(|point| point.path.as_deref())
            .or(current_path.as_deref());
        let begin_line_anchor = location_anchor
            .as_ref()
            .map_or(current_line, |point| point.line);
        let begin = parse_text_point(begin_text.trim(), begin_path, begin_line_anchor).map_err(
            |error| {
                format!(
                    "textual range-begin parse failed: {error}; unit={unit:?} kind={kind:?} \
                     current_path={current_path:?} range={range:?} raw={line:?}"
                )
            },
        )?;
        if let Some(path) = begin.path.as_deref() {
            current_path = Some(path.into());
        }
        if begin.line != 0 {
            current_line = begin.line;
        }
        let end = parse_text_point(end_text.trim(), current_path.as_deref(), current_line)
            .map_err(|error| {
                format!(
                    "textual range-end parse failed: {error}; unit={unit:?} kind={kind:?} \
                     current_path={current_path:?} range={range:?} raw={line:?}"
                )
            })?;
        if let Some(path) = end.path.as_deref() {
            current_path = Some(path.into());
        }
        if end.line != 0 {
            current_line = end.line;
        }
        let location =
            parse_text_location_token(location_token, current_path.as_deref(), current_line)
                .map_err(|error| {
                    format!(
                        "textual location parse failed: {error}; unit={unit:?} kind={kind:?} \
                     current_path={current_path:?} range={range:?} tail={tail:?} \
                     location_token={location_token:?} raw={line:?}"
                    )
                })?;
        if let Some(path) = location.path.as_deref() {
            current_path = Some(path.into());
        }
        if location.line != 0 {
            current_line = location.line;
        }
        let signature = crate::ast_stream::canonicalize_source_roots(
            textual_signature(tail).unwrap_or(kind),
            materialized,
            original,
        );
        let name = if matches!(
            kind,
            "TemplateTypeParmDecl" | "NonTypeTemplateParmDecl" | "TemplateTemplateParmDecl"
        ) {
            textual_template_parameter_name(tail)
        } else {
            textual_declaration_name(kind, tail)
        };
        let identity_name = crate::ast_stream::canonical_declaration_identity(
            kind,
            &name,
            Some(&signature),
            location.line,
            location.column,
        );
        let parents = scopes
            .iter()
            .filter_map(|scope| scope.identity.as_deref())
            .collect::<Vec<_>>();
        let manifest_parents = scopes
            .iter()
            .filter_map(|scope| scope.manifest_identity.as_deref())
            .collect::<Vec<_>>();
        let instantiated_template_child = scopes.last().is_some_and(|scope| {
            !scope.template_parameters.is_empty()
                && (crate::ast_stream::is_callable_declaration_kind(kind)
                    || matches!(kind, "CXXRecordDecl" | "RecordDecl"))
                && !scope
                    .template_parameters
                    .iter()
                    .any(|parameter| signature_has_identifier(&signature, parameter))
        });
        let normalized = location
            .path
            .as_deref()
            .and_then(|path| normalize_text_owner(path, materialized, original));
        let owned = normalized.is_some();
        if owned {
            owned_path.clone_from(&current_path);
            owned_line = current_line;
        }
        if included
            && !name.is_empty()
            && !instantiated_template_child
            && manifest_parents
                .iter()
                .all(|parent| !parent.starts_with("(anonymous"))
        {
            let manifest_name = name.rsplit("::").next().unwrap_or(&name);
            let qualified_manifest = if manifest_parents.is_empty() {
                manifest_name.to_owned()
            } else {
                format!("{}::{manifest_name}", manifest_parents.join("::"))
            };
            if let Some(qualified_manifest) = normalize_model_manifest_name(&qualified_manifest) {
                *all_named_manifest
                    .entry(qualified_manifest.clone())
                    .or_insert(0) += 1;
                if owned {
                    *named_manifest.entry(qualified_manifest).or_insert(0) += 1;
                }
            }
        }
        if matches!(scope_kind, TextScopeKind::Lexical)
            && owned
            && [begin.line, begin.column, end.line, end.column].contains(&0)
        {
            return Err(format!(
                "textual lexical scope lacks exact range: unit={unit:?} kind={kind:?} \
                 current_path={current_path:?} parents={parents:?} range={range:?} raw={line:?}"
            ));
        }
        let qualified = if name.is_empty() && (!included || !owned) {
            String::new()
        } else if name.is_empty() {
            if location.line == 0 || location.column == 0 {
                return Err(format!(
                    "anonymous textual node lacks exact location: unit={unit:?} kind={kind:?} \
                     implicit={} definition={} local={local} included={included} \
                     current_path={current_path:?} parents={parents:?} range={range:?} tail={tail:?} \
                     raw={line:?}",
                    implicit,
                    line.contains(" definition "),
                ));
            }
            let anonymous =
                crate::ast_stream::anonymous_scope_identity(kind, location.line, location.column);
            if parents.is_empty() {
                format!("::{anonymous}")
            } else {
                format!("{}::{anonymous}", parents.join("::"))
            }
        } else if parents.is_empty() {
            identity_name.clone()
        } else {
            format!("{}::{identity_name}", parents.join("::"))
        };
        let qualified = if qualified == "model"
            || qualified.starts_with("model::")
            || qualified == "io"
            || qualified.starts_with("io::")
        {
            format!("emel::{qualified}")
        } else {
            qualified
        };
        let inventory_identity = crate::ast_stream::canonical_redeclaration_identity(
            kind,
            &qualified,
            textual_has_flag(rest, "definition"),
            location.line,
            location.column,
        );
        if included && let Some(path) = normalized.clone() {
            let key = DeclarationKey {
                path,
                kind: kind.into(),
                line: location.line,
                column: location.column,
                begin_line: begin.line,
                begin_column: begin.column,
                end_line: end.line,
                end_column: end.column,
                qualified_identity: inventory_identity,
                canonical_signature: signature,
            };
            if [
                key.line,
                key.column,
                key.begin_line,
                key.begin_column,
                key.end_line,
                key.end_column,
            ]
            .contains(&0)
            {
                return Err(format!(
                    "textual declaration lacks exact source span: unit={unit:?} kind={kind:?} \
                     path={:?} location={}:{} begin={}:{} end={}:{} parents={parents:?} \
                     range={range:?} tail={tail:?} raw={line:?}",
                    key.path,
                    key.line,
                    key.column,
                    key.begin_line,
                    key.begin_column,
                    key.end_line,
                    key.end_column,
                ));
            }
            *output.entry(key).or_insert(0) += 1;
        }
        let scope_identity = if matches!(
            scope_kind,
            TextScopeKind::Namespace
                | TextScopeKind::Record
                | TextScopeKind::Enum
                | TextScopeKind::Function
        ) {
            if !identity_name.is_empty() {
                Some(identity_name)
            } else if kind == "NamespaceDecl" {
                if location.line == 0 || location.column == 0 {
                    return Err(format!(
                        "anonymous namespace lacks exact scope identity: unit={unit:?} \
                         current_path={current_path:?} range={range:?} tail={tail:?} raw={line:?}"
                    ));
                }
                Some(crate::ast_stream::anonymous_scope_identity(
                    kind,
                    location.line,
                    location.column,
                ))
            } else {
                None
            }
        } else {
            None
        };
        if matches!(
            kind,
            "TemplateTypeParmDecl" | "NonTypeTemplateParmDecl" | "TemplateTemplateParmDecl"
        ) && !name.is_empty()
            && let Some(parent) = scopes.last_mut()
        {
            parent.template_parameters.insert(name.clone());
        }
        scopes.push(TextScope {
            kind: scope_kind,
            identity: scope_identity,
            manifest_identity: if matches!(scope_kind, TextScopeKind::Namespace) && name.is_empty()
            {
                Some("(anonymous namespace)".into())
            } else if matches!(
                scope_kind,
                TextScopeKind::Namespace
                    | TextScopeKind::Record
                    | TextScopeKind::Enum
                    | TextScopeKind::Function
            ) && !name.is_empty()
            {
                Some(name.rsplit("::").next().unwrap_or(&name).to_owned())
            } else {
                None
            },
            template_parameters: BTreeSet::new(),
            path: current_path.clone(),
            line: current_line,
        });
    }
    Ok(TextualExpectations {
        declarations: output,
        named_manifest,
        all_named_manifest,
    })
}

fn signature_has_identifier(signature: &str, identifier: &str) -> bool {
    signature
        .split(|character: char| !(character == '_' || character.is_ascii_alphanumeric()))
        .any(|token| token == identifier)
}

fn normalize_model_manifest_name(name: &str) -> Option<String> {
    if name == "model" {
        Some("emel::model".into())
    } else if name.starts_with("model::") {
        Some(format!("emel::{name}"))
    } else if name == "io" {
        Some("emel::io".into())
    } else if name.starts_with("io::") {
        Some(format!("emel::{name}"))
    } else if name == "emel::model"
        || name.starts_with("emel::model::")
        || name == "emel::io"
        || name.starts_with("emel::io::")
        || name == TEST_TEXTUAL_AST_FILTER
        || name.starts_with("emel_window_test::")
    {
        Some(name.into())
    } else {
        None
    }
}

fn textual_header_range(rest: &str) -> Result<TextHeaderRange, String> {
    let kind_end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let mut cursor = kind_end;
    skip_text_whitespace(rest, &mut cursor);
    let Some(node_id) = take_text_token(rest, &mut cursor) else {
        return Ok(TextHeaderRange::Absent);
    };
    if !is_text_node_id(node_id) {
        return Ok(TextHeaderRange::Absent);
    }
    loop {
        skip_text_whitespace(rest, &mut cursor);
        let before_marker = cursor;
        let Some(marker) = take_text_token(rest, &mut cursor) else {
            return Ok(TextHeaderRange::Absent);
        };
        if !matches!(marker, "parent" | "prev") {
            cursor = before_marker;
            break;
        }
        skip_text_whitespace(rest, &mut cursor);
        let reference = take_text_token(rest, &mut cursor)
            .ok_or_else(|| format!("textual AST {marker} marker lacks node address: {rest:?}"))?;
        if !is_text_node_id(reference) {
            return Err(format!(
                "textual AST {marker} marker has invalid node address {reference:?}: {rest:?}"
            ));
        }
    }
    skip_text_whitespace(rest, &mut cursor);
    if !rest[cursor..].starts_with('<') {
        return Ok(TextHeaderRange::Absent);
    }
    if rest[cursor..].starts_with("<<invalid sloc>>") {
        return Ok(TextHeaderRange::InvalidSLoc);
    }
    let end = rest[cursor + 1..]
        .find('>')
        .and_then(|offset| cursor.checked_add(1)?.checked_add(offset))
        .ok_or_else(|| format!("textual AST header range is unterminated: {rest:?}"))?;
    if end == cursor + 1 {
        return Err(format!("textual AST header range is empty: {rest:?}"));
    }
    Ok(TextHeaderRange::Span { start: cursor, end })
}

fn skip_text_whitespace(text: &str, cursor: &mut usize) {
    while text
        .as_bytes()
        .get(*cursor)
        .is_some_and(u8::is_ascii_whitespace)
    {
        *cursor += 1;
    }
}

fn take_text_token<'a>(text: &'a str, cursor: &mut usize) -> Option<&'a str> {
    let start = *cursor;
    while text
        .as_bytes()
        .get(*cursor)
        .is_some_and(|byte| !byte.is_ascii_whitespace())
    {
        *cursor += 1;
    }
    (start != *cursor).then(|| &text[start..*cursor])
}

fn is_text_node_id(token: &str) -> bool {
    token.strip_prefix("0x").is_some_and(|digits| {
        !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

fn parse_text_location_token(
    token: &str,
    current_path: Option<&str>,
    current_line: u64,
) -> Result<TextPoint, String> {
    let has_numeric_path_suffix = || {
        let mut parts = token.rsplitn(3, ':');
        let column = parts.next().unwrap_or_default();
        let line = parts.next().unwrap_or_default();
        let path = parts.next().unwrap_or_default();
        !path.is_empty()
            && line.bytes().all(|byte| byte.is_ascii_digit())
            && !line.is_empty()
            && column.bytes().all(|byte| byte.is_ascii_digit())
            && !column.is_empty()
    };
    if token.starts_with("line:")
        || token.starts_with("col:")
        || token.starts_with('/')
        || has_numeric_path_suffix()
    {
        parse_text_point(token, current_path, current_line)
    } else {
        Ok(TextPoint {
            path: current_path.map(Into::into),
            line: current_line,
            column: 0,
        })
    }
}

fn textual_location_token(tail: &str) -> &str {
    if tail.starts_with('<')
        && let Some(close) = tail.find('>')
    {
        let suffix = &tail[close + 1..];
        let end = suffix
            .find(char::is_whitespace)
            .map_or(tail.len(), |offset| close + 1 + offset);
        return &tail[..end];
    }
    tail.split_whitespace().next().unwrap_or_default()
}

fn parse_text_point(
    text: &str,
    current_path: Option<&str>,
    current_line: u64,
) -> Result<TextPoint, String> {
    if let Some(value) = text.strip_prefix("line:") {
        let (line, column) = value
            .split_once(':')
            .ok_or_else(|| format!("invalid textual line point: {text}"))?;
        return Ok(TextPoint {
            path: current_path.map(Into::into),
            line: line
                .parse()
                .map_err(|_| format!("invalid line in {text}"))?,
            column: column
                .parse()
                .map_err(|_| format!("invalid column in {text}"))?,
        });
    }
    if let Some(column) = text.strip_prefix("col:") {
        return Ok(TextPoint {
            path: current_path.map(Into::into),
            line: current_line,
            column: column
                .parse()
                .map_err(|_| format!("invalid column in {text}"))?,
        });
    }
    let mut parts = text.rsplitn(3, ':');
    let column = parts.next().unwrap_or_default();
    let line = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default();
    if path.is_empty() {
        return Ok(TextPoint {
            path: current_path.map(Into::into),
            line: current_line,
            column: 0,
        });
    }
    Ok(TextPoint {
        path: Some(path.into()),
        line: line
            .parse()
            .map_err(|_| format!("invalid line in {text}"))?,
        column: column
            .parse()
            .map_err(|_| format!("invalid column in {text}"))?,
    })
}

fn textual_signature(tail: &str) -> Option<&str> {
    let start = tail.find('\'')? + 1;
    let end = tail[start..].find('\'')? + start;
    Some(&tail[start..end])
}

fn textual_declaration_name(kind: &str, tail: &str) -> String {
    let before_signature = tail.split('\'').next().unwrap_or(tail);
    let words = before_signature.split_whitespace().collect::<Vec<_>>();
    if matches!(kind, "CXXRecordDecl" | "RecordDecl")
        && let Some(index) = words
            .iter()
            .position(|word| matches!(*word, "struct" | "class" | "union"))
    {
        return words
            .get(index + 1)
            .and_then(|word| textual_name_token(word))
            .unwrap_or_default();
    }
    let eligible = words
        .iter()
        .filter_map(|word| textual_name_token(word))
        .collect::<Vec<_>>();
    if eligible.len() >= 2 && eligible[eligible.len() - 2] == "operator" {
        return format!("operator {}", eligible.last().expect("length checked"));
    }
    eligible.last().cloned().unwrap_or_default()
}

fn textual_template_parameter_name(tail: &str) -> String {
    tail.split_whitespace()
        .rev()
        .find_map(textual_name_token)
        .unwrap_or_default()
}

fn textual_name_token(word: &str) -> Option<String> {
    const MODIFIERS: &[&str] = &[
        "class",
        "consteval",
        "constexpr",
        "default",
        "definition",
        "deleted",
        "enum",
        "extern",
        "implicit",
        "inline",
        "mutable",
        "nested",
        "pure",
        "referenced",
        "register",
        "scoped",
        "static",
        "struct",
        "thread_local",
        "typename",
        "union",
        "used",
        "virtual",
    ];
    let token = word;
    if token.is_empty()
        || token.starts_with("line:")
        || token.starts_with("col:")
        || (token.starts_with("0x") && token[2..].bytes().all(|byte| byte.is_ascii_hexdigit()))
        || MODIFIERS.contains(&token)
    {
        return None;
    }
    let identifier = |candidate: &str| {
        let mut bytes = candidate.bytes();
        bytes
            .next()
            .is_some_and(|byte| byte == b'_' || byte.is_ascii_alphabetic())
            && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
    };
    let qualified_identifier = token.contains("::") && token.split("::").all(identifier);
    if identifier(token)
        || qualified_identifier
        || token.strip_prefix('~').is_some_and(identifier)
        || (token.starts_with("operator")
            && token.len() > "operator".len()
            && token["operator".len()..]
                .bytes()
                .all(|byte| !byte.is_ascii_whitespace() && !byte.is_ascii_alphanumeric()))
    {
        Some(token.into())
    } else {
        None
    }
}

fn textual_scope_kind(kind: &str) -> TextScopeKind {
    match kind {
        "NamespaceDecl" => TextScopeKind::Namespace,
        "CXXRecordDecl" | "RecordDecl" | "ClassTemplateDecl" => TextScopeKind::Record,
        "EnumDecl" => TextScopeKind::Enum,
        "FunctionTemplateDecl" => TextScopeKind::Other,
        kind if crate::ast_stream::is_callable_declaration_kind(kind) => TextScopeKind::Function,
        "CompoundStmt" | "ForStmt" | "CXXForRangeStmt" | "IfStmt" | "WhileStmt" | "SwitchStmt"
        | "DoStmt" | "LambdaExpr" | "CXXCatchStmt" => TextScopeKind::Lexical,
        _ => TextScopeKind::Other,
    }
}

fn textual_has_flag(row: &str, flag: &str) -> bool {
    row.split_whitespace().any(|token| token == flag)
}

fn normalize_text_owner(file: &str, materialized: &Path, original: &Path) -> Option<String> {
    let path = PathBuf::from(file);
    let relative = path
        .strip_prefix(materialized)
        .or_else(|_| path.strip_prefix(original))
        .ok()?;
    let normalized = crate::ast_stream::normalize_relative_owner_path(relative)?;
    (normalized.starts_with("src/emel/model/") || normalized.starts_with("tests/model/"))
        .then_some(normalized)
}

fn validate_declaration_coverage(
    expected: &BTreeMap<DeclarationKey, usize>,
    coverage: &[Coverage],
) -> Result<(), String> {
    let mut actual = BTreeMap::<DeclarationKey, usize>::new();
    for row in coverage {
        let key = DeclarationKey {
            path: row.path.clone(),
            kind: row.kind.clone(),
            line: row.line,
            column: row.column,
            begin_line: row.begin_line,
            begin_column: row.begin_column,
            end_line: row.end_line,
            end_column: row.end_column,
            qualified_identity: row.name.clone(),
            canonical_signature: row.signature.clone(),
        };
        *actual.entry(key).or_insert(0) += 1;
    }
    let actual = actual
        .into_iter()
        .filter(|(key, _)| {
            !(!expected.contains_key(key)
                && key.kind == "NamespaceDecl"
                && (key.qualified_identity == "emel"
                    || key.qualified_identity.starts_with("emel::<declaration@")))
        })
        .collect::<BTreeMap<_, _>>();
    if &actual != expected {
        let missing = expected
            .iter()
            .find(|(key, count)| actual.get(*key).copied().unwrap_or(0) != **count);
        let actual_at_span = missing.map_or_else(Vec::new, |(missing_key, _)| {
            actual
                .iter()
                .filter(|(key, _)| {
                    key.path == missing_key.path
                        && key.kind == missing_key.kind
                        && key.line == missing_key.line
                        && key.column == missing_key.column
                        && key.begin_line == missing_key.begin_line
                        && key.begin_column == missing_key.begin_column
                        && key.end_line == missing_key.end_line
                        && key.end_column == missing_key.end_column
                })
                .collect::<Vec<_>>()
        });
        let unexpected = actual
            .iter()
            .find(|(key, count)| expected.get(*key).copied().unwrap_or(0) != **count);
        return Err(format!(
            "textual/JSON declaration coverage mismatch: expected={missing:?} \
             actual_at_span={actual_at_span:?} unexpected={unexpected:?}"
        ));
    }
    Ok(())
}

fn expected_translation_units(blobs: &[Blob]) -> Result<BTreeSet<String>, String> {
    let mut expected = BTreeSet::new();
    for blob in blobs {
        git::validate_path(&blob.path)?;
        if is_model_translation_unit(&blob.path) && !expected.insert(blob.path.clone()) {
            return Err(format!(
                "duplicate pinned model translation-unit path: {}",
                blob.path
            ));
        }
    }
    if expected.is_empty() {
        return Err("pinned Git manifest has no model translation units".to_owned());
    }
    Ok(expected)
}

fn is_model_translation_unit(path: &str) -> bool {
    extension_is(path, "cpp")
        && (path.starts_with("src/emel/model/") || path.starts_with("tests/model/"))
}

fn canonical_compile_path(repo: &Path, entry: &CompileEntry) -> Result<Option<String>, String> {
    let file = Path::new(&entry.file);
    let joined = if file.is_absolute() {
        file.to_owned()
    } else {
        let directory = entry.directory.as_deref().ok_or_else(|| {
            format!(
                "relative compile-database file has no directory: {}",
                entry.file
            )
        })?;
        Path::new(directory).join(file)
    };
    let canonical = joined.canonicalize().map_err(|error| {
        format!(
            "cannot canonicalize compile-database file {}: {error}",
            joined.display()
        )
    })?;
    let Ok(relative) = canonical.strip_prefix(repo) else {
        return Ok(None);
    };
    let normalized = relative.to_string_lossy().replace('\\', "/");
    git::validate_path(&normalized)?;
    Ok(Some(normalized))
}

fn select_translation_units(
    repo: &Path,
    entries: Vec<CompileEntry>,
    expected: &BTreeSet<String>,
) -> Result<Vec<CompileEntry>, String> {
    let canonical_repo = repo
        .canonicalize()
        .map_err(|error| format!("cannot canonicalize source repository: {error}"))?;
    let mut selected = BTreeMap::<String, CompileEntry>::new();
    let mut duplicates = BTreeSet::new();
    for entry in entries {
        let Some(path) = canonical_compile_path(&canonical_repo, &entry)? else {
            continue;
        };
        if !is_model_translation_unit(&path) {
            continue;
        }
        if selected.insert(path.clone(), entry).is_some() {
            duplicates.insert(path);
        }
    }
    if !duplicates.is_empty() {
        return Err(format!(
            "duplicate model translation-unit compile-database entries: {}",
            duplicates.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    let actual = selected.keys().cloned().collect::<BTreeSet<_>>();
    let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
    let extra = actual.difference(expected).cloned().collect::<Vec<_>>();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(format!(
            "compile-database model translation-unit manifest mismatch: missing=[{}]; extra=[{}]",
            missing.join(", "),
            extra.join(", ")
        ));
    }
    Ok(selected.into_values().collect())
}

fn validate_required_translation_unit_ast_coverage(
    expected: &BTreeSet<String>,
    coverage: &[Coverage],
) -> Result<(), String> {
    if !expected.contains(REQUIRED_DETAILED_TRANSLATION_UNIT) {
        return Err(format!(
            "pinned Git manifest lacks required detailed translation unit: {REQUIRED_DETAILED_TRANSLATION_UNIT}"
        ));
    }
    if !coverage
        .iter()
        .any(|row| row.path == REQUIRED_DETAILED_TRANSLATION_UNIT)
    {
        return Err(format!(
            "required translation unit has no detailed Clang AST coverage: {REQUIRED_DETAILED_TRANSLATION_UNIT}"
        ));
    }
    Ok(())
}

fn parse_compile_database(bytes: &[u8]) -> Result<Vec<CompileEntry>, String> {
    serde_json::from_slice(bytes).map_err(|error| format!("invalid compile database: {error}"))
}

fn retained_compile_database_with(
    path: &Path,
    repo: &Path,
    expected_sha256: &str,
    after_read: impl FnOnce(),
) -> Result<Vec<CompileEntry>, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    after_read();
    let entries = parse_compile_database(&bytes)?;
    if canonical_compile_database_digest(path, repo, &entries)? != expected_sha256 {
        return Err("compile database digest drift".to_owned());
    }
    Ok(entries)
}

fn canonical_compile_database_digest(
    path: &Path,
    repo: &Path,
    entries: &[CompileEntry],
) -> Result<String, String> {
    let normalizer = compile_database_normalizer(path, repo, entries)?;
    let mut canonical = entries
        .iter()
        .map(|entry| canonical_compile_entry(entry, &normalizer))
        .collect::<Result<Vec<_>, _>>()?;
    canonical.sort();
    if canonical.windows(2).any(|entries| entries[0] == entries[1]) {
        return Err("duplicate canonical compile-database entry".to_owned());
    }
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| format!("cannot encode canonical compile database: {error}"))?;
    Ok(sha256(&bytes))
}

fn compile_database_normalizer(
    path: &Path,
    repo: &Path,
    entries: &[CompileEntry],
) -> Result<Vec<RootReplacement>, String> {
    let source_input = absolute_normalized_path(repo)?;
    let source_root = repo
        .canonicalize()
        .map_err(|error| format!("cannot canonicalize source repository: {error}"))?;
    let build_path = path
        .parent()
        .ok_or_else(|| format!("compile database has no parent: {}", path.display()))?
        .to_owned();
    let build_input = absolute_normalized_path(&build_path)?;
    let build_root = build_path
        .canonicalize()
        .map_err(|error| format!("cannot canonicalize compile database directory: {error}"))?;
    let mut roots = vec![
        RootReplacement {
            raw: source_input,
            placeholder: "<SOURCE_ROOT>".into(),
        },
        RootReplacement {
            raw: normalized_path(&source_root),
            placeholder: "<SOURCE_ROOT>".into(),
        },
        RootReplacement {
            raw: build_input,
            placeholder: "<BUILD_ROOT>".into(),
        },
        RootReplacement {
            raw: normalized_path(&build_root),
            placeholder: "<BUILD_ROOT>".into(),
        },
    ];
    for root in external_include_roots_optional(entries)? {
        let input = absolute_normalized_path(&root)?;
        let canonical = root.canonicalize().map_err(|error| {
            format!(
                "cannot canonicalize external dependency include root {}: {error}",
                root.display()
            )
        })?;
        let canonical = normalized_path(&canonical);
        let marker = canonical.find("/_deps/").ok_or_else(|| {
            format!("external dependency include root lacks _deps marker: {canonical}")
        })?;
        let placeholder = format!("<EXTERNAL_ROOT>{}", &canonical[marker..]);
        roots.push(RootReplacement {
            raw: input,
            placeholder: placeholder.clone(),
        });
        roots.push(RootReplacement {
            raw: canonical,
            placeholder,
        });
    }
    roots.sort_by_key(|left| std::cmp::Reverse(left.raw.len()));
    roots.dedup_by(|left, right| left.raw == right.raw);
    Ok(roots)
}

fn canonical_compile_entry(
    entry: &CompileEntry,
    normalizer: &[RootReplacement],
) -> Result<CanonicalCompileEntry, String> {
    let directory = entry
        .directory
        .as_deref()
        .ok_or_else(|| format!("compile-database entry has no directory: {}", entry.file))?;
    Ok(CanonicalCompileEntry {
        directory: normalize_compile_value(directory, normalizer)?,
        file: normalize_compile_value(&entry.file, normalizer)?,
        arguments: raw_arguments(entry)?
            .into_iter()
            .map(|argument| normalize_compile_value(&argument, normalizer))
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn normalize_compile_value(value: &str, roots: &[RootReplacement]) -> Result<String, String> {
    let mut normalized = value.replace('\\', "/");
    for root in roots {
        normalized = replace_root_prefixes(&normalized, &root.raw, &root.placeholder);
    }
    let has_unresolved_absolute_path = normalized.starts_with('/')
        || normalized.contains("=/")
        || ["-I/", "-isystem/", "-iquote/", "-include/", "-imacros/"]
            .iter()
            .any(|prefix| normalized.starts_with(prefix))
        || has_windows_absolute_path(&normalized);
    if has_unresolved_absolute_path && normalized != CLANG {
        return Err(format!(
            "compile database contains an unnormalized absolute or path-bearing value: {value}"
        ));
    }
    Ok(normalized)
}

fn has_windows_absolute_path(value: &str) -> bool {
    let bytes = value.as_bytes();
    value.contains("//")
        || bytes
            .windows(3)
            .any(|window| window[0].is_ascii_alphabetic() && window[1] == b':' && window[2] == b'/')
}

fn replace_root_prefixes(value: &str, root: &str, placeholder: &str) -> String {
    let mut output = String::new();
    let mut remaining = value;
    while let Some(index) = remaining.find(root) {
        let (before, candidate) = remaining.split_at(index);
        let after = &candidate[root.len()..];
        let left_boundary = before.is_empty()
            || ["-I", "-isystem", "-iquote", "-include", "-imacros"]
                .iter()
                .any(|prefix| before.ends_with(prefix))
            || before
                .chars()
                .last()
                .is_some_and(|character| !character.is_ascii_alphanumeric() && character != '_');
        let right_boundary = after.is_empty()
            || after
                .chars()
                .next()
                .is_some_and(|character| character == '/');
        if left_boundary && right_boundary {
            output.push_str(before);
            output.push_str(placeholder);
            remaining = after;
        } else {
            let (head, tail) = candidate.split_at(1);
            output.push_str(before);
            output.push_str(head);
            remaining = tail;
        }
    }
    output.push_str(remaining);
    output
}

fn normalized_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn absolute_normalized_path(path: &Path) -> Result<String, String> {
    let path = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("cannot read current directory: {error}"))?
            .join(path)
    };
    Ok(normalized_path(&path))
}

fn verify_compiler() -> Result<(), String> {
    let clang =
        fs::read(CLANG).map_err(|error| format!("cannot read Clang executable: {error}"))?;
    if sha256(&clang) != CLANG_SHA256 {
        return Err("Clang executable digest drift".to_owned());
    }
    let output = Command::new(CLANG)
        .arg("--version")
        .output()
        .map_err(|error| error.to_string())?;
    let version = String::from_utf8_lossy(&output.stdout);
    let mut lines = version.lines();
    let first = lines.next().unwrap_or_default().trim();
    let target = lines
        .find_map(|line| line.strip_prefix("Target: "))
        .unwrap_or_default();
    let identity = format!("{first} {target}");
    if identity != CLANG_IDENTITY {
        return Err(format!("Clang identity drift: {identity}"));
    }
    Ok(())
}

fn materialize(blobs: &[Blob], target: &Path) -> Result<(), String> {
    if blobs.is_empty() {
        return Err("cannot materialize an empty Git blob manifest".to_owned());
    }
    let mut paths = BTreeSet::new();
    for blob in blobs {
        git::validate_path(&blob.path)?;
        if !paths.insert(&blob.path) {
            return Err(format!("duplicate materialized Git path: {}", blob.path));
        }
        let destination = target.join(&blob.path);
        let parent = destination
            .parent()
            .ok_or_else(|| format!("materialized path has no parent: {}", blob.path))?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create materialized directory: {error}"))?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)
            .map_err(|error| format!("cannot create materialized file {}: {error}", blob.path))?;
        file.write_all(&blob.bytes)
            .map_err(|error| format!("cannot write materialized file {}: {error}", blob.path))?;
    }
    Ok(())
}

fn rewritten_arguments(
    entry: &CompileEntry,
    source_root: &str,
    target_root: &str,
    external_rewrites: &BTreeMap<String, String>,
) -> Result<Vec<String>, String> {
    let raw = raw_arguments(entry)?;
    let mut output = Vec::new();
    let mut skip_next = false;
    for (index, argument) in raw.into_iter().enumerate() {
        if index == 0 {
            continue;
        }
        if skip_next {
            skip_next = false;
            continue;
        }
        if matches!(argument.as_str(), "-o" | "-MF" | "-MT" | "-MQ") {
            skip_next = true;
            continue;
        }
        if argument == "-c" || argument.starts_with("-M") {
            continue;
        }
        if argument == entry.file {
            continue;
        }
        output.push(
            if let Some(replacement) = external_rewrites.get(&argument) {
                replacement.clone()
            } else if argument.replace('\\', "/").contains("/_deps/") {
                return Err(format!("unbound external dependency argument: {argument}"));
            } else {
                argument.replace(source_root, target_root)
            },
        );
    }
    output.extend([
        "-fsyntax-only".into(),
        "-Xclang".into(),
        "-ast-dump=json".into(),
        "-Xclang".into(),
        "-ast-dump-filter".into(),
        "-Xclang".into(),
        JSON_AST_FILTER.into(),
    ]);
    output.push(entry.file.replace(source_root, target_root));
    Ok(output)
}

fn raw_arguments(entry: &CompileEntry) -> Result<Vec<String>, String> {
    if let Some(arguments) = &entry.arguments {
        Ok(arguments.clone())
    } else {
        shell_words::split(
            entry
                .command
                .as_deref()
                .ok_or("compile entry lacks command and arguments")?,
        )
        .map_err(|error| format!("cannot split compile command: {error}"))
    }
}

fn external_include_roots(entries: &[CompileEntry]) -> Result<Vec<PathBuf>, String> {
    let roots = external_include_roots_optional(entries)?;
    if roots.is_empty() {
        return Err("compile database has no bound external dependency includes".to_owned());
    }
    Ok(roots)
}

fn external_include_roots_optional(entries: &[CompileEntry]) -> Result<Vec<PathBuf>, String> {
    let mut roots = BTreeSet::new();
    for entry in entries {
        let arguments = raw_arguments(entry)?;
        for (index, argument) in arguments.iter().enumerate() {
            let candidate = if argument == "-isystem" || argument == "-I" {
                arguments.get(index + 1).map(String::as_str)
            } else {
                argument.strip_prefix("-I")
            };
            if let Some(candidate) = candidate
                && candidate.replace('\\', "/").contains("/_deps/")
            {
                roots.insert(PathBuf::from(candidate));
            }
        }
    }
    Ok(roots.into_iter().collect())
}

fn materialize_external_includes(
    roots: &[PathBuf],
    temporary: &Path,
) -> Result<(BTreeMap<String, String>, Vec<u8>), String> {
    let mut rewrites = BTreeMap::new();
    let mut manifest_rows = Vec::new();
    for (index, root) in roots.iter().enumerate() {
        if !root.is_dir() {
            return Err(format!(
                "external dependency include root is missing: {}",
                root.display()
            ));
        }
        let target = temporary.join("external-includes").join(index.to_string());
        let mut files = Vec::new();
        collect_regular_files(root, root, &mut files)?;
        for (relative, bytes) in files {
            let normalized = relative
                .components()
                .map(|part| part.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            manifest_rows.push((
                format!("external-include-{index}/{normalized}"),
                sha256(&bytes),
            ));
            let destination = target.join(&relative);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent).map_err(|error| {
                    format!("cannot create external include directory: {error}")
                })?;
            }
            fs::write(&destination, bytes)
                .map_err(|error| format!("cannot materialize external include: {error}"))?;
        }
        rewrites.insert(
            root.to_string_lossy().into_owned(),
            target.to_string_lossy().into_owned(),
        );
    }
    manifest_rows.sort();
    let mut manifest = String::from("sha256\tpath\n");
    for (path, digest) in manifest_rows {
        writeln!(manifest, "{digest}\t{path}").expect("writing to String cannot fail");
    }
    Ok((rewrites, manifest.into_bytes()))
}

fn collect_regular_files(
    root: &Path,
    directory: &Path,
    output: &mut Vec<(PathBuf, Vec<u8>)>,
) -> Result<(), String> {
    let mut entries = fs::read_dir(directory)
        .map_err(|error| format!("cannot read external include directory: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("cannot read external include entry: {error}"))?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let file_type = entry
            .file_type()
            .map_err(|error| format!("cannot inspect external include entry: {error}"))?;
        if file_type.is_symlink() {
            return Err(format!(
                "external include symlink is unsupported: {}",
                entry.path().display()
            ));
        }
        if file_type.is_dir() {
            collect_regular_files(root, &entry.path(), output)?;
        } else if file_type.is_file() {
            let relative = entry
                .path()
                .strip_prefix(root)
                .map_err(|error| error.to_string())?
                .to_owned();
            let bytes = fs::read(entry.path())
                .map_err(|error| format!("cannot read external include: {error}"))?;
            output.push((relative, bytes));
        } else {
            return Err(format!(
                "unsupported external include entry: {}",
                entry.path().display()
            ));
        }
    }
    Ok(())
}

pub fn coverage_tsv(rows: &[Coverage]) -> Vec<u8> {
    let mut output = String::from(
        "path\tline\tcolumn\tbegin_line\tbegin_column\tend_line\tend_column\tkind\tqualified_name\tcanonical_signature\tnode_id\n",
    );
    for row in rows {
        writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            row.path,
            row.line,
            row.column,
            row.begin_line,
            row.begin_column,
            row.end_line,
            row.end_column,
            row.kind,
            row.name,
            row.signature,
            row.node_id
        )
        .expect("writing to String cannot fail");
    }
    output.into_bytes()
}

fn extension_is(path: &str, extension: &str) -> bool {
    Path::new(path)
        .extension()
        .is_some_and(|actual| actual.eq_ignore_ascii_case(extension))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_database_swap_after_read_uses_only_retained_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        let build = temp.path().join("build");
        fs::create_dir_all(source.join("src/emel/model")).unwrap();
        fs::create_dir_all(&build).unwrap();
        let file = source.join("src/emel/model/a.cpp");
        fs::write(&file, b"namespace emel {}\n").unwrap();
        let path = build.join("compile_commands.json");
        let original = format!(
            r#"[{{"directory":"{}","file":"{}","arguments":["clang++"]}}]"#,
            build.display(),
            file.display()
        );
        fs::write(&path, &original).unwrap();
        let entries = parse_compile_database(original.as_bytes()).unwrap();
        let expected = canonical_compile_database_digest(&path, &source, &entries).unwrap();

        let entries = retained_compile_database_with(&path, &source, &expected, || {
            fs::write(&path, b"not-json-after-read").unwrap();
        })
        .unwrap();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file, file.to_string_lossy());
        assert_eq!(fs::read(path).unwrap(), b"not-json-after-read");
    }

    #[test]
    fn canonical_compile_database_identity_normalizes_roots_but_binds_arguments() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let first_digest = fixture_compile_database_digest(first.path(), "-DINVOCATION=one");
        let second_digest = fixture_compile_database_digest(second.path(), "-DINVOCATION=one");
        let changed_digest = fixture_compile_database_digest(second.path(), "-DINVOCATION=two");

        assert_eq!(first_digest, second_digest);
        assert_ne!(first_digest, changed_digest);
    }

    #[test]
    fn canonical_compile_database_identity_rejects_duplicate_rows() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        let build = temp.path().join("build");
        fs::create_dir_all(source.join("src/emel/model")).unwrap();
        fs::create_dir_all(build.join("_deps/stateforward-src/include")).unwrap();
        let file = source.join("src/emel/model/a.cpp");
        fs::write(&file, b"namespace emel {}\n").unwrap();
        let entry = CompileEntry {
            directory: Some(build.to_string_lossy().into_owned()),
            file: file.to_string_lossy().into_owned(),
            command: None,
            arguments: Some(vec![
                CLANG.into(),
                format!("-I{}", source.join("include").display()),
                "-DINVOCATION=one".into(),
                file.to_string_lossy().into_owned(),
            ]),
        };
        let error = canonical_compile_database_digest(
            &build.join("compile_commands.json"),
            &source,
            &[entry.clone(), entry],
        )
        .unwrap_err();
        assert_eq!(error, "duplicate canonical compile-database entry");
    }

    fn fixture_compile_database_digest(root: &Path, define: &str) -> String {
        let source = root.join("checkout");
        let build = root.join("build");
        let include = source.join("include");
        let external = build.join("_deps/stateforward-src/include");
        fs::create_dir_all(source.join("src/emel/model")).unwrap();
        fs::create_dir_all(&include).unwrap();
        fs::create_dir_all(&external).unwrap();
        let file = source.join("src/emel/model/a.cpp");
        fs::write(&file, b"namespace emel {}\n").unwrap();
        let entry = CompileEntry {
            directory: Some(build.to_string_lossy().into_owned()),
            file: file.to_string_lossy().into_owned(),
            command: None,
            arguments: Some(vec![
                CLANG.into(),
                format!("-I{}", include.display()),
                "-isystem".into(),
                external.to_string_lossy().into_owned(),
                define.into(),
                file.to_string_lossy().into_owned(),
            ]),
        };
        canonical_compile_database_digest(&build.join("compile_commands.json"), &source, &[entry])
            .unwrap()
    }

    fn translation_unit_fixture(
        paths: &[&str],
    ) -> (tempfile::TempDir, Vec<Blob>, Vec<CompileEntry>) {
        let repo = tempfile::tempdir().unwrap();
        let mut blobs = Vec::new();
        let mut entries = Vec::new();
        for path in paths {
            let file = repo.path().join(path);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            fs::write(&file, b"namespace emel {}\n").unwrap();
            blobs.push(Blob {
                path: (*path).into(),
                bytes: b"namespace emel {}\n".to_vec(),
            });
            entries.push(CompileEntry {
                directory: Some(repo.path().to_string_lossy().into_owned()),
                file: file.to_string_lossy().into_owned(),
                command: None,
                arguments: Some(vec!["clang++".into()]),
            });
        }
        (repo, blobs, entries)
    }

    #[test]
    fn exact_manifest_rejects_one_of_many_omission_including_generation_any() {
        let paths = [
            "src/emel/model/data.cpp",
            "src/emel/model/generation/any.cpp",
            "tests/model/data_tests.cpp",
        ];
        let (repo, blobs, mut entries) = translation_unit_fixture(&paths);
        entries.remove(1);
        let expected = expected_translation_units(&blobs).unwrap();

        let error = select_translation_units(repo.path(), entries, &expected).unwrap_err();
        assert!(
            error.contains("missing=[src/emel/model/generation/any.cpp]"),
            "{error}"
        );
        assert!(error.contains("extra=[]"), "{error}");
    }

    #[test]
    fn exact_manifest_rejects_duplicate_compile_database_entry() {
        let paths = ["src/emel/model/generation/any.cpp"];
        let (repo, blobs, mut entries) = translation_unit_fixture(&paths);
        entries.push(entries[0].clone());
        let expected = expected_translation_units(&blobs).unwrap();

        let error = select_translation_units(repo.path(), entries, &expected).unwrap_err();
        assert!(
            error.contains("duplicate model translation-unit"),
            "{error}"
        );
        assert!(
            error.contains("src/emel/model/generation/any.cpp"),
            "{error}"
        );
    }

    #[test]
    fn exact_manifest_rejects_unexpected_substitution_with_both_diagnostics() {
        let paths = [
            "src/emel/model/data.cpp",
            "src/emel/model/generation/any.cpp",
            "src/emel/model/unexpected.cpp",
        ];
        let (repo, mut blobs, mut entries) = translation_unit_fixture(&paths);
        blobs.pop();
        entries.remove(1);
        let expected = expected_translation_units(&blobs).unwrap();

        let error = select_translation_units(repo.path(), entries, &expected).unwrap_err();
        assert!(
            error.contains("missing=[src/emel/model/generation/any.cpp]"),
            "{error}"
        );
        assert!(
            error.contains("extra=[src/emel/model/unexpected.cpp]"),
            "{error}"
        );
    }

    #[test]
    fn exact_manifest_selects_generation_any_once() {
        let paths = [
            "src/emel/model/generation/any.cpp",
            "tests/model/data_tests.cpp",
        ];
        let (repo, blobs, entries) = translation_unit_fixture(&paths);
        let expected = expected_translation_units(&blobs).unwrap();

        let selected = select_translation_units(repo.path(), entries, &expected).unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(
            selected
                .iter()
                .filter(|entry| entry.file.ends_with("src/emel/model/generation/any.cpp"))
                .count(),
            1
        );
    }

    #[test]
    fn compile_argument_transform_removes_outputs_and_rewrites_root() {
        let entry = CompileEntry {
            directory: Some("/old".into()),
            file: "/old/src/emel/model/a.cpp".into(),
            command: Some("clang++ -I/old/include -o a.o -c /old/src/emel/model/a.cpp".into()),
            arguments: None,
        };
        let args = rewritten_arguments(&entry, "/old", "/new", &BTreeMap::new()).unwrap();
        assert!(
            !args
                .iter()
                .any(|argument| argument == "-o" || argument == "-c")
        );
        assert!(args.iter().any(|argument| argument == "-I/new/include"));
        assert!(
            args.windows(2)
                .any(|arguments| arguments == ["-Xclang", "-ast-dump-filter"])
        );
        assert!(
            args.windows(2)
                .any(|arguments| arguments == ["-Xclang", "emel"])
        );
        assert_eq!(args.last().unwrap(), "/new/src/emel/model/a.cpp");
    }

    #[test]
    fn external_include_roots_accept_deps_without_a_build_directory_component() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("_deps/stateforward-src/include");
        fs::create_dir_all(&root).unwrap();
        let entries = [CompileEntry {
            directory: Some(temp.path().to_string_lossy().into_owned()),
            file: temp.path().join("a.cpp").to_string_lossy().into_owned(),
            command: None,
            arguments: Some(vec![
                "clang++".into(),
                "-isystem".into(),
                root.display().to_string(),
            ]),
        }];

        assert_eq!(external_include_roots(&entries).unwrap(), vec![root]);
    }

    #[test]
    fn external_include_roots_normalize_windows_separators_before_matching_deps() {
        let windows_root = r"C:\\build\\_deps\\stateforward-src\\include";
        let entries = [CompileEntry {
            directory: Some(r"C:\\build".into()),
            file: r"C:\\source\\a.cpp".into(),
            command: None,
            arguments: Some(vec![
                "clang++".into(),
                "-isystem".into(),
                windows_root.into(),
            ]),
        }];

        assert_eq!(
            external_include_roots_optional(&entries).unwrap(),
            vec![PathBuf::from(windows_root)]
        );
    }

    #[test]
    fn canonical_compile_database_rejects_unresolvable_windows_dependency_roots() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        let build = temp.path().join("build");
        fs::create_dir_all(source.join("src/emel/model")).unwrap();
        fs::create_dir_all(&build).unwrap();
        let file = source.join("src/emel/model/a.cpp");
        fs::write(&file, b"namespace emel {}\n").unwrap();
        let entry = CompileEntry {
            directory: Some(build.to_string_lossy().into_owned()),
            file: file.to_string_lossy().into_owned(),
            command: None,
            arguments: Some(vec![
                CLANG.into(),
                "-isystem".into(),
                r"C:\\build\\_deps\\stateforward-src\\include".into(),
                file.to_string_lossy().into_owned(),
            ]),
        };

        let error = canonical_compile_database_digest(
            &build.join("compile_commands.json"),
            &source,
            &[entry],
        )
        .unwrap_err();
        assert!(error.contains("cannot canonicalize external dependency include root"));
    }

    #[test]
    fn windows_absolute_paths_are_never_left_in_a_canonical_manifest() {
        assert!(has_windows_absolute_path(r"C:/build/_deps/include"));
        assert!(has_windows_absolute_path(r"-IC:/build/_deps/include"));
        assert!(has_windows_absolute_path("//server/share/include"));
        assert!(has_windows_absolute_path("-DROOT=C:/build/_deps/include"));
        assert!(has_windows_absolute_path("--sysroot=C:/sdk"));
        assert!(has_windows_absolute_path("--sysroot=//server/share"));
        assert!(!has_windows_absolute_path("<SOURCE_ROOT>/include"));
    }

    #[test]
    fn object_materialization_rejects_duplicate_and_non_canonical_paths() {
        let temp = tempfile::tempdir().unwrap();
        let duplicate = Blob {
            path: "src/a.hpp".into(),
            bytes: b"first".to_vec(),
        };
        assert!(materialize(&[duplicate.clone(), duplicate], temp.path()).is_err());
        assert!(
            materialize(
                &[Blob {
                    path: "../escape.hpp".into(),
                    bytes: Vec::new(),
                }],
                temp.path(),
            )
            .is_err()
        );
    }

    #[test]
    fn object_materialization_resolves_model_test_relative_kernel_include() {
        let temp = tempfile::tempdir().unwrap();
        materialize(
            &[
                Blob {
                    path: "tests/model/loader/lifecycle_tests.cpp".into(),
                    bytes: b"#include \"../../kernel/test_helpers.hpp\"\n".to_vec(),
                },
                Blob {
                    path: "tests/kernel/test_helpers.hpp".into(),
                    bytes: b"#pragma once\n".to_vec(),
                },
            ],
            temp.path(),
        )
        .unwrap();
        assert_eq!(
            fs::read(
                temp.path()
                    .join("tests/model/loader/../../kernel/test_helpers.hpp")
            )
            .unwrap(),
            b"#pragma once\n"
        );
        assert!(MATERIALIZED_SOURCE_ROOTS.contains(&"tests"));
        assert!(!MATERIALIZED_SOURCE_ROOTS.contains(&"tests/model"));
    }

    #[test]
    fn statement_without_location_tail_uses_exact_range_only_for_local_scope() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/a.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            "namespace emel { int cast() { int local = 0; return local; } }\n",
        )
        .unwrap();
        let text = format!(
            "NamespaceDecl 0x1 <{}:1:1, col:64> col:11 emel\n\
             |-FunctionDecl 0x2 <col:18, col:62> col:22 cast 'int ()'\n\
             | `-CompoundStmt 0x3 <col:29, col:62>\n\
             |   `-VarDecl 0x4 <col:31, col:43> col:35 local 'int'\n",
            path.display()
        );
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        assert_eq!(expected.len(), 2);
        assert!(expected.keys().any(|key| key.kind == "NamespaceDecl"));
        assert!(expected.keys().any(|key| key.kind == "FunctionDecl"));
    }

    #[test]
    fn included_anonymous_declaration_still_requires_exact_location() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/a.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "enum { value };\n").unwrap();
        let text = format!("EnumDecl 0x1 <{}:1:1, col:14> col:0\n", path.display());
        let error = parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path())
            .unwrap_err();
        assert!(error.contains("included=true"), "{error}");
        assert!(error.contains("kind=\"EnumDecl\""), "{error}");

        let namespace = format!("NamespaceDecl 0x2 <{}:1:1, col:14> col:0\n", path.display());
        let error =
            parse_textual_expectations(&namespace, "test-unit.cpp", temp.path(), temp.path())
                .unwrap_err();
        assert!(error.contains("included=true"), "{error}");
        assert!(error.contains("kind=\"NamespaceDecl\""), "{error}");
    }

    #[test]
    fn textual_anonymous_namespace_owners_are_exact_and_distinct() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/a.cpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            "namespace emel {\nnamespace { int first; }\nnamespace { int second; }\n}\n",
        )
        .unwrap();
        let text = [
            format!(
                "NamespaceDecl 0x1 <{}:1:1, line:4:1> line:1:11 emel",
                path.display()
            ),
            "|-NamespaceDecl 0x2 <line:2:1, col:24> line:2:11".into(),
            "| `-VarDecl 0x3 <col:13, col:21> col:17 first 'int'".into(),
            "`-NamespaceDecl 0x4 <line:3:1, col:25> line:3:11".into(),
            "  `-VarDecl 0x5 <col:13, col:22> col:17 second 'int'".into(),
        ]
        .join("\n");
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        assert!(expected.keys().any(|key| key.qualified_identity == "emel"));
        assert!(expected.keys().any(|key| {
            key.qualified_identity == "emel::<anonymous-NamespaceDecl@2:11>::first"
        }));
        assert!(expected.keys().any(|key| {
            key.qualified_identity == "emel::<anonymous-NamespaceDecl@3:11>::second"
        }));
    }

    #[test]
    fn quoted_qualified_type_is_payload_not_location() {
        let token = "'emel::gguf::loader::error':'emel::gguf::loader::error'";
        let point = parse_text_location_token(token, Some("/pinned/error.hpp"), 14).unwrap();
        assert_eq!(point.path.as_deref(), Some("/pinned/error.hpp"));
        assert_eq!(point.line, 14);
        assert_eq!(point.column, 0);
        assert_eq!(textual_signature(token), Some("emel::gguf::loader::error"));
    }

    #[test]
    fn textual_fixed_enum_uses_the_spelled_underlying_type_signature() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/data.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "enum class route : uint8_t { value };\n").unwrap();
        let text = format!(
            "EnumDecl 0x1 <{}:1:1, col:39> col:12 class route 'uint8_t':'unsigned char'",
            path.display()
        );
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        let declaration = expected.keys().next().unwrap();
        assert_eq!(declaration.qualified_identity, "route");
        assert_eq!(declaration.canonical_signature, "uint8_t");
    }

    #[test]
    fn textual_names_are_structural_when_name_and_type_match() {
        assert_eq!(
            textual_declaration_name(
                "FieldDecl",
                "col:12 referenced topology 'topology':'emel::model::generation::topology'"
            ),
            "topology"
        );
        assert_eq!(
            textual_declaration_name("EnumDecl", "line:4:12 scoped class route 'uint8_t'"),
            "route"
        );
        assert_eq!(
            textual_declaration_name(
                "FunctionDecl",
                "line:8:5 used constexpr inline static compute 'void ()'"
            ),
            "compute"
        );
        assert_eq!(
            textual_declaration_name(
                "UsingDecl",
                "col:32 emel::model::generation::build_quantized_path_audit"
            ),
            "emel::model::generation::build_quantized_path_audit"
        );
    }

    #[test]
    fn textual_and_json_callable_keys_match_for_pinned_constructor_row() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp
            .path()
            .join("tests/model/tensor/window/window_test_fixture.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            "namespace emel_window_test { struct window_fixture { window_fixture(); }; }\n",
        )
        .unwrap();
        let text = [
            format!(
                "NamespaceDecl 0x1 <{}:1:1, col:79> col:11 emel_window_test",
                path.display()
            ),
            "`-CXXRecordDecl 0x2 <col:30, col:77> col:37 struct window_fixture definition".into(),
            "  `-CXXConstructorDecl 0x3 <col:54, col:73> col:54 window_fixture 'void ()'".into(),
        ]
        .join("\n");
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        let textual = expected
            .keys()
            .find(|key| key.kind == "CXXConstructorDecl")
            .unwrap();
        let json_identity = crate::ast_stream::canonical_callable_identity(
            "CXXConstructorDecl",
            "window_fixture",
            Some("void ()"),
        )
        .unwrap();
        assert_eq!(
            textual.qualified_identity,
            format!("emel_window_test::window_fixture::{json_identity}")
        );
        assert_eq!(textual.canonical_signature, "void ()");
    }

    #[test]
    fn textual_callable_family_keeps_names_signatures_and_collisions_distinct() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/callable.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            "struct actor { actor(); actor(const actor &); ~actor(); actor & operator=(const actor &); }; bool run(int);\n",
        )
        .unwrap();
        let text = [
            format!(
                "CXXRecordDecl 0x1 <{}:1:1, col:91> col:8 struct actor definition",
                path.display()
            ),
            "|-CXXConstructorDecl 0x2 <col:16, col:22> col:16 actor 'void ()'".into(),
            "|-CXXConstructorDecl 0x3 <col:25, col:44> col:25 actor 'void (const actor &)'".into(),
            "|-CXXDestructorDecl 0x4 <col:47, col:54> col:47 ~actor 'void () noexcept'".into(),
            "`-CXXMethodDecl 0x5 <col:57, col:88> col:65 operator= 'actor &(const actor &)'".into(),
            "FunctionDecl 0x6 <col:94, col:106> col:99 run 'bool (int)'".into(),
        ]
        .join("\n");
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        let callable = expected
            .keys()
            .filter(|key| crate::ast_stream::is_callable_declaration_kind(&key.kind))
            .map(|key| {
                (
                    key.qualified_identity.as_str(),
                    key.canonical_signature.as_str(),
                )
            })
            .collect::<Vec<_>>();
        assert!(callable.contains(&("actor::actor(void ())", "void ()")));
        assert!(callable.contains(&("actor::actor(void (const actor &))", "void (const actor &)")));
        assert!(callable.contains(&("actor::~actor(void () noexcept)", "void () noexcept")));
        assert!(callable.contains(&(
            "actor::operator=(actor &(const actor &))",
            "actor &(const actor &)"
        )));
        assert!(callable.contains(&("run(bool (int))", "bool (int)")));
        assert_eq!(callable.len(), 5);
    }

    #[test]
    fn textual_redeclarations_keep_forward_locations_and_canonical_definition() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/events.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "struct bind_storage;\nstruct bind_storage {};\n").unwrap();
        let text = [
            format!(
                "CXXRecordDecl 0x1 <{}:1:1, col:19> col:8 struct bind_storage",
                path.display()
            ),
            "CXXRecordDecl 0x2 prev 0x1 <line:2:1, col:23> col:8 struct bind_storage definition"
                .into(),
        ]
        .join("\n");
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        let records = expected
            .keys()
            .filter(|key| key.kind == "CXXRecordDecl")
            .map(|key| key.qualified_identity.as_str())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 2);
        assert!(records.contains(&"bind_storage::<declaration@1:8>"));
        assert!(records.contains(&"bind_storage"));
    }

    #[test]
    fn textual_template_parents_do_not_hide_typed_function_children() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/templates.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            "template <class T> int load(T);\ntemplate <class T> int load(T*);\n",
        )
        .unwrap();
        let text = [
            format!(
                "FunctionTemplateDecl 0x1 <{}:1:1, col:31> col:24 load",
                path.display()
            ),
            "`-FunctionDecl 0x2 <col:20, col:30> col:24 load 'int (T)'".into(),
            "FunctionTemplateDecl 0x3 <line:2:1, col:32> col:24 load".into(),
            "`-FunctionDecl 0x4 <col:20, col:31> col:24 load 'int (T *)'".into(),
        ]
        .join("\n");
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        let names = expected
            .keys()
            .map(|key| key.qualified_identity.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"load::<template@1:24>"));
        assert!(names.contains(&"load::<template@2:24>"));
        assert!(names.contains(&"load(int (T))"));
        assert!(names.contains(&"load(int (T *))"));
        assert_eq!(names.len(), 4);
    }

    #[test]
    fn textual_qualified_names_reject_unsupported_or_empty_segments() {
        for malformed in [
            "::build",
            "emel::model::",
            "emel::::build",
            "emel::model<int>::build",
            "emel::model::build!",
        ] {
            assert_eq!(
                textual_declaration_name("UsingDecl", &format!("col:5 {malformed}")),
                "",
                "accepted malformed qualified declaration name {malformed:?}"
            );
        }
    }

    #[test]
    fn textual_flags_are_exact_tokens_in_every_position() {
        for row in [
            "implicit ConstructorUsingShadowDecl 0x1",
            "ConstructorUsingShadowDecl implicit 0x1",
            "ConstructorUsingShadowDecl 0x1 implicit",
        ] {
            assert!(textual_has_flag(row, "implicit"), "{row:?}");
        }
        for row in [
            "ConstructorUsingShadowDecl 0x1 implicitly",
            "ConstructorUsingShadowDecl 0x1 nonimplicit",
            "ConstructorUsingShadowDecl 0x1 implicit-inline",
        ] {
            assert!(!textual_has_flag(row, "implicit"), "{row:?}");
        }
    }

    #[test]
    fn textual_explicit_using_is_retained_and_constructor_shadows_are_excluded() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/tensor/sm.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "using base_type::base_type;\n").unwrap();
        let text = [
            format!(
                "UsingDecl 0x1 <{}:1:1, col:20> col:20 base_type::sm",
                path.display()
            ),
            "ConstructorUsingShadowDecl 0x2 <col:20> col:20 implicit".into(),
        ]
        .join("\n");
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        assert_eq!(expected.len(), 1);
        let declaration = expected.keys().next().unwrap();
        assert_eq!(declaration.kind, "UsingDecl");
        assert_eq!(declaration.qualified_identity, "base_type::sm");
        assert!(crate::ast_stream::excluded_declaration_kind(
            "ConstructorUsingShadowDecl"
        ));
    }

    #[test]
    fn textual_truly_anonymous_declaration_keeps_location_identity() {
        assert_eq!(
            textual_declaration_name("FieldDecl", "col:5 referenced 'unsigned int'"),
            ""
        );
        assert_eq!(
            textual_declaration_name("FieldDecl", "col:5 0x123abc 'unsigned int'"),
            ""
        );
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/data.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "unsigned : 1;\n").unwrap();
        let text = format!(
            "FieldDecl 0x1 <{}:1:1, col:12> col:5 'unsigned int'",
            path.display()
        );
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        let declaration = expected.keys().next().unwrap();
        assert_eq!(
            declaration.qualified_identity,
            "::<anonymous-FieldDecl@1:5>"
        );
    }

    #[test]
    fn textual_location_cursor_advances_through_complete_ranges() {
        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("src/emel/model/first.hpp");
        let second = temp.path().join("src/emel/model/second.hpp");
        fs::create_dir_all(first.parent().unwrap()).unwrap();
        fs::write(&first, "first\ncontinued\nsame\n").unwrap();
        fs::write(&second, "one\ntwo\nsecond\n").unwrap();
        let text = [
            format!(
                "FieldDecl 0x1 <{}:1:1, line:2:9> col:5 multiline 'int'",
                first.display()
            ),
            "FieldDecl 0x2 <line:3:1, col:9> col:5 same_line 'int'".into(),
            format!(
                "FieldDecl 0x3 <{}:3:1, col:9> col:5 switched_file 'int'",
                second.display()
            ),
        ]
        .join("\n");
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        let multiline = expected
            .keys()
            .find(|key| key.qualified_identity == "multiline")
            .unwrap();
        assert_eq!((multiline.line, multiline.column), (2, 5));
        let same_line = expected
            .keys()
            .find(|key| key.qualified_identity == "same_line")
            .unwrap();
        assert_eq!((same_line.line, same_line.column), (3, 5));
        let switched = expected
            .keys()
            .find(|key| key.qualified_identity == "switched_file")
            .unwrap();
        assert_eq!(switched.path, "src/emel/model/second.hpp");
        assert_eq!((switched.line, switched.column), (3, 5));
    }

    #[test]
    fn textual_macro_scratch_location_does_not_claim_an_owned_range_end() {
        let temp = tempfile::tempdir().unwrap();
        let owned = temp.path().join("tests/model/fixture_tests.cpp");
        let external = temp.path().join("third_party/doctest.hpp");
        fs::create_dir_all(owned.parent().unwrap()).unwrap();
        fs::create_dir_all(external.parent().unwrap()).unwrap();
        fs::write(&owned, "TEST_CASE();\nint owned();\n").unwrap();
        fs::write(&external, "macro\n").unwrap();
        let text = [
            format!(
                "FunctionDecl 0x1 <{}:1:1, {}:1:12> <scratch space>:17:1 \
                 DOCTEST_ANON_FUNC_1 'void ()' static",
                external.display(),
                owned.display(),
            ),
            format!(
                "FunctionDecl 0x2 <{}:2:1, col:11> col:5 owned 'int ()'",
                owned.display(),
            ),
        ]
        .join("\n");
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        assert_eq!(expected.len(), 1);
        assert_eq!(
            expected.keys().next().unwrap().qualified_identity,
            "owned(int ())"
        );
    }

    #[test]
    fn textual_explicit_location_anchors_a_leading_column_range() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/error.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "namespace emel::error {\n}\n").unwrap();
        let text = format!(
            "NamespaceDecl 0x1 <{}:1:1, line:2:1> line:1:11 emel\n\
             `-NamespaceDecl 0x2 <col:15, line:2:1> line:1:17 error nested",
            path.display()
        );
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        let declaration = expected
            .keys()
            .find(|key| key.qualified_identity == "emel::error")
            .unwrap();
        assert_eq!((declaration.line, declaration.column), (1, 17));
        assert_eq!((declaration.begin_line, declaration.begin_column), (1, 15));
        assert_eq!((declaration.end_line, declaration.end_column), (2, 1));
    }

    #[test]
    fn textual_ranges_reject_empty_endpoints() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/data.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "one\ntwo\nthree\n").unwrap();
        let text = format!(
            "FieldDecl 0x2 <{}:1:1, > col:5 empty_end 'int'",
            path.display()
        );
        let error = parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path())
            .unwrap_err();
        assert!(error.contains("empty endpoint"), "{error}");
        assert!(error.contains("raw="), "{error}");
    }

    #[test]
    fn location_token_grammar_accepts_closed_forms_and_rejects_malformed_forms() {
        let line = parse_text_location_token("line:7:9", Some("/pinned/a.hpp"), 3).unwrap();
        assert_eq!((line.line, line.column), (7, 9));
        let column = parse_text_location_token("col:11", Some("/pinned/a.hpp"), 7).unwrap();
        assert_eq!((column.line, column.column), (7, 11));
        let path = parse_text_location_token("/pinned/b.hpp:13:17", None, 0).unwrap();
        assert_eq!(path.path.as_deref(), Some("/pinned/b.hpp"));
        assert_eq!((path.line, path.column), (13, 17));

        for malformed in [
            "line:not-a-line:9",
            "col:not-a-column",
            "/pinned/b.hpp:no:17",
        ] {
            assert!(
                parse_text_location_token(malformed, Some("/pinned/a.hpp"), 7).is_err(),
                "accepted malformed location token {malformed:?}"
            );
        }
    }

    #[test]
    fn textual_header_range_ignores_template_angle_payloads() {
        let exact =
            "ElaboratedType 0x7d5ec8a30 'emel::callback<void (const events::probe_done &)>' sugar";
        assert_eq!(
            textual_header_range(exact).unwrap(),
            TextHeaderRange::Absent
        );
        let nested = "TemplateArgument 0x123 type 'vector<pair<int, int>>'";
        assert_eq!(
            textual_header_range(nested).unwrap(),
            TextHeaderRange::Absent
        );
    }

    #[test]
    fn textual_header_range_is_anchored_after_node_references() {
        let row = "FunctionDecl 0x123 parent 0x456 prev 0x789 <line:2:1, col:9> col:5 f 'void ()'";
        let TextHeaderRange::Span { start, end } = textual_header_range(row).unwrap() else {
            panic!("anchored range was not recognized");
        };
        assert_eq!(&row[start + 1..end], "line:2:1, col:9");
    }

    #[test]
    fn textual_header_range_rejects_malformed_header_ranges() {
        for row in [
            "FunctionDecl 0x123 <line:2:1, col:9",
            "FunctionDecl 0x123 <>",
            "FunctionDecl 0x123 prev not-an-address <line:2:1, col:9>",
        ] {
            assert!(
                textual_header_range(row).is_err(),
                "accepted malformed header {row:?}"
            );
        }
    }

    #[test]
    fn invalid_sloc_is_allowed_only_for_non_inventory_non_lexical_nodes() {
        let sentinel = "ImplicitValueInitExpr 0x7d0c23338 <<invalid sloc>> 'long long[4]'";
        assert_eq!(
            textual_header_range(sentinel).unwrap(),
            TextHeaderRange::InvalidSLoc
        );
        let parsed = parse_textual_expectations(
            sentinel,
            "src/emel/model/architecture/detail.cpp",
            Path::new("/materialized"),
            Path::new("/original"),
        )
        .unwrap();
        assert!(parsed.is_empty());

        for row in [
            "FunctionDecl 0x123 <<invalid sloc>> f 'void ()'",
            "CompoundStmt 0x456 <<invalid sloc>>",
        ] {
            let error = parse_textual_expectations(
                row,
                "src/emel/model/architecture/detail.cpp",
                Path::new("/materialized"),
                Path::new("/original"),
            )
            .unwrap_err();
            assert!(
                error.contains("invalid source-location sentinel"),
                "{error}"
            );
            assert!(error.contains("architecture/detail.cpp"), "{error}");
            assert!(error.contains("raw="), "{error}");
        }
    }

    #[test]
    fn per_declaration_coverage_rejects_one_missing_in_same_header() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/a.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "namespace emel { int first(); int second(int); }\n").unwrap();
        let text = format!(
            "NamespaceDecl 0x1 <{}:1:1, line:4:1> line:1:11 emel\n\
             |-FunctionDecl 0x2 <line:2:1, col:10> col:5 first 'int ()'\n\
             `-FunctionDecl 0x3 <line:3:1, col:20> col:5 second 'int (int)'\n",
            path.display()
        );
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        assert_eq!(expected.len(), 3);
        let mut coverage = expected.keys().map(coverage_from_key).collect::<Vec<_>>();
        assert!(validate_declaration_coverage(&expected, &coverage).is_ok());
        coverage.retain(|row| !row.name.contains("second"));
        assert!(validate_declaration_coverage(&expected, &coverage).is_err());
    }

    #[test]
    fn independent_textual_oracle_uses_a_distinct_filter_and_exposes_json_omissions() {
        let arguments = vec![
            "-Xclang".into(),
            "-ast-dump=json".into(),
            "-Xclang".into(),
            "-ast-dump-filter".into(),
            "-Xclang".into(),
            JSON_AST_FILTER.into(),
            "unit.cpp".into(),
        ];
        let textual = independent_textual_arguments(&arguments, TEXTUAL_AST_FILTER).unwrap();
        let filter_index = textual
            .iter()
            .position(|argument| argument == "-ast-dump-filter")
            .unwrap();
        assert_eq!(textual[filter_index + 2], TEXTUAL_AST_FILTER);
        assert_ne!(JSON_AST_FILTER, TEXTUAL_AST_FILTER);
        assert!(textual.iter().any(|argument| argument == "-ast-dump=json"));

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/a.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            "namespace emel::model { int kept(); int omitted(); }\n",
        )
        .unwrap();
        let dump = format!(
            "NamespaceDecl 0x1 <{}:1:1, line:4:1> line:1:17 model\n\
             |-FunctionDecl 0x2 <line:2:1, col:10> col:5 kept 'int ()'\n\
             `-FunctionDecl 0x3 <line:3:1, col:20> col:5 omitted 'int ()'\n",
            path.display()
        );
        let expected =
            parse_textual_expectations(&dump, "unit.cpp", temp.path(), temp.path()).unwrap();
        assert_eq!(expected.len(), 3);
        let mut filtered_json_coverage = expected.keys().map(coverage_from_key).collect::<Vec<_>>();
        filtered_json_coverage.pop();
        assert!(validate_declaration_coverage(&expected, &filtered_json_coverage).is_err());
    }

    #[test]
    fn ast_list_manifest_matches_template_parent_and_primary_but_not_instantiation() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/template.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            "namespace emel::model { template<class T> void load(T); }\n",
        )
        .unwrap();
        let dump = format!(
            "NamespaceDecl 0x1 <{}:1:1, col:62> col:17 model\n\
             `-FunctionTemplateDecl 0x2 <col:25, col:60> col:48 load\n\
               |-TemplateTypeParmDecl 0x3 <col:34, col:40> col:40 referenced class depth 0 index 0 T\n\
               |-FunctionDecl 0x4 <col:43, col:60> col:48 load 'void (T)'\n\
               `-FunctionDecl 0x5 <col:43, col:60> col:48 used load 'void (int)'",
            path.display()
        );
        let detailed =
            parse_textual_expectations(&dump, "unit.cpp", temp.path(), temp.path()).unwrap();
        let listed = parse_ast_list(std::io::Cursor::new(
            b"emel::model\nemel::model::load\nemel::model::load\n\
              emel::model::(anonymous namespace)::hidden\nother::load\n",
        ))
        .unwrap();
        assert_eq!(detailed.named_manifest, listed);
        validate_named_manifest(&listed, &detailed.named_manifest, "unit.cpp").unwrap();

        let mut omitted = detailed.named_manifest;
        *omitted.get_mut("emel::model::load").unwrap() -= 1;
        assert!(validate_named_manifest(&listed, &omitted, "unit.cpp").is_err());
    }

    #[test]
    fn ast_list_manifest_includes_model_owned_cross_domain_names_and_exact_counts() {
        let listed = parse_ast_list(std::io::Cursor::new(
            b"emel::model\nemel::io\nemel::io::loader\nemel::io::loader::sm\
              \nemel::io::loader::sm\nemel_window_test\nother::io\n",
        ))
        .unwrap();
        let detailed = BTreeMap::from([
            ("emel::io".to_owned(), 1),
            ("emel::io::loader".to_owned(), 1),
            ("emel::io::loader::sm".to_owned(), 2),
            ("emel::model".to_owned(), 1),
            ("emel_window_test".to_owned(), 1),
        ]);
        assert_eq!(listed, detailed);
        validate_named_manifest(&listed, &detailed, "unit.cpp").unwrap();

        let mut omitted = detailed.clone();
        omitted.remove("emel::io::loader::sm");
        assert!(validate_named_manifest(&listed, &omitted, "unit.cpp").is_err());

        let mut undercounted = detailed;
        *undercounted.get_mut("emel::io::loader::sm").unwrap() -= 1;
        assert!(validate_named_manifest(&listed, &undercounted, "unit.cpp").is_err());

        let mut owned = BTreeMap::from([("emel::model".to_owned(), 1)]);
        merge_manifest_max(
            &mut owned,
            BTreeMap::from([
                ("emel::io".to_owned(), 1),
                ("emel::io::loader".to_owned(), 1),
                ("emel::io::loader::sm".to_owned(), 2),
            ]),
        );
        assert_eq!(owned.get("emel::io::loader::sm"), Some(&2));
        validate_owned_manifest_is_listed(&listed, &owned, "unit.cpp").unwrap();
        *owned.get_mut("emel::io::loader::sm").unwrap() += 1;
        assert!(validate_owned_manifest_is_listed(&listed, &owned, "unit.cpp").is_err());
    }

    #[test]
    fn pinned_source_cross_manifest_requires_all_four_forward_declarations() {
        let blobs = vec![
            Blob {
                path: "src/emel/model/loader/events.hpp".into(),
                bytes: b"namespace emel::io::loader { struct sm; }\n".to_vec(),
            },
            Blob {
                path: "src/emel/model/tensor/context.hpp".into(),
                bytes: b"namespace emel::io::mmap { struct sm; }\n\
                         namespace emel::io::read { struct sm; }\n\
                         namespace emel::io::staged_read { struct sm; }\n"
                    .to_vec(),
            },
        ];
        let manifest = source_cross_domain_manifest(&blobs).unwrap();
        assert_eq!(manifest.get("emel::io"), Some(&4));
        for name in PINNED_MODEL_IO_FORWARD_DECLARATIONS {
            assert_eq!(manifest.get(*name), Some(&1));
        }

        let mut missing = blobs;
        missing[1].bytes =
            b"namespace emel::io::mmap { struct sm; }\nnamespace emel::io::read { struct sm; }\n"
                .to_vec();
        assert!(source_cross_domain_manifest(&missing).is_err());
    }

    #[test]
    fn cross_manifest_omission_cannot_be_masked_by_external_ast_list_names() {
        let source = BTreeMap::from([
            ("emel::io".to_owned(), 4),
            ("emel::io::loader".to_owned(), 1),
            ("emel::io::loader::sm".to_owned(), 1),
            ("emel::io::mmap".to_owned(), 1),
            ("emel::io::mmap::sm".to_owned(), 1),
            ("emel::io::read".to_owned(), 1),
            ("emel::io::read::sm".to_owned(), 1),
            ("emel::io::staged_read".to_owned(), 1),
            ("emel::io::staged_read::sm".to_owned(), 1),
        ]);
        let coverage = PINNED_MODEL_IO_FORWARD_DECLARATIONS
            .iter()
            .enumerate()
            .map(|(index, name)| Coverage {
                path: "src/emel/model/forward.hpp".into(),
                line: index as u64 + 1,
                column: 1,
                begin_line: index as u64 + 1,
                begin_column: 1,
                end_line: index as u64 + 1,
                end_column: 10,
                kind: "CXXRecordDecl".into(),
                name: format!("{name}::<declaration@{}:1>", index + 1),
                signature: "sm".into(),
                node_id: format!("0x{index:x}"),
            })
            .collect::<Vec<_>>();
        validate_cross_domain_manifests(&source, &source, &coverage).unwrap();

        let mut filtered = source.clone();
        filtered.remove("emel::io::staged_read::sm");
        let listed_with_external_mask = BTreeMap::from([
            ("emel::io::staged_read::sm".to_owned(), 100usize),
            ("emel".to_owned(), 100),
        ]);
        validate_owned_manifest_is_listed(
            &listed_with_external_mask,
            &BTreeMap::from([("emel::io::staged_read::sm".to_owned(), 1)]),
            "unit.cpp",
        )
        .unwrap();
        let error = validate_cross_domain_manifests(&source, &filtered, &coverage).unwrap_err();
        assert!(error.contains("staged_read::sm"), "{error}");

        let missing_json = &coverage[..coverage.len() - 1];
        let error = validate_cross_domain_manifests(&source, &source, missing_json).unwrap_err();
        assert!(error.contains("cross-domain JSON AST"), "{error}");
    }

    #[test]
    fn per_declaration_coverage_distinguishes_same_span_signatures() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("src/emel/model/a.hpp");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "namespace emel { int load(); int load(int); }\n").unwrap();
        let text = format!(
            "NamespaceDecl 0x1 <{}:1:1, line:2:1> line:1:11 emel\n\
             |-FunctionDecl 0x2 <line:1:18, col:28> col:22 load 'int ()'\n\
             `-FunctionDecl 0x3 <line:1:18, col:28> col:22 load 'int (int)'\n",
            path.display()
        );
        let expected =
            parse_textual_expectations(&text, "test-unit.cpp", temp.path(), temp.path()).unwrap();
        let overloads = expected
            .keys()
            .filter(|key| key.kind == "FunctionDecl")
            .collect::<Vec<_>>();
        assert_eq!(overloads.len(), 2);
        assert_ne!(
            overloads[0].canonical_signature,
            overloads[1].canonical_signature
        );
        let mut coverage = expected.keys().map(coverage_from_key).collect::<Vec<_>>();
        coverage.retain(|row| row.signature != "int (int)");
        assert!(validate_declaration_coverage(&expected, &coverage).is_err());
    }

    fn coverage_from_key(key: &DeclarationKey) -> Coverage {
        Coverage {
            path: key.path.clone(),
            line: key.line,
            column: key.column,
            begin_line: key.begin_line,
            begin_column: key.begin_column,
            end_line: key.end_line,
            end_column: key.end_column,
            kind: key.kind.clone(),
            name: key.qualified_identity.clone(),
            signature: key.canonical_signature.clone(),
            node_id: "test".into(),
        }
    }
}
