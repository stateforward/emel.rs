use std::collections::BTreeMap;
use std::fs;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};

use crate::clang::{CLANG, Coverage};
use crate::hash::semantic_hash;
use crate::scan::{Extracted, canonical_tokens};

// Exact extraction at source commit 843a117386ef17dc5a50549bbfc821074c2141d6 measured the
// largest compiler-oracle lane at 8,675,113 bytes. The 16 MiB ceiling leaves deterministic
// headroom for representation overhead while rejecting accidental unfiltered multi-gigabyte dumps.
const PINNED_TEXTUAL_AST_MAX_BYTES: u64 = 8_675_113;
const TEXTUAL_AST_CAPTURE_LIMIT_BYTES: u64 = 16 * 1024 * 1024;
const _: () = assert!(TEXTUAL_AST_CAPTURE_LIMIT_BYTES > PINNED_TEXTUAL_AST_MAX_BYTES);

pub struct TextualAstCapture {
    pub file: tempfile::NamedTempFile,
    pub bytes: u64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Location {
    file: Option<String>,
    line: Option<u64>,
    col: Option<u64>,
    offset: Option<u64>,
    tok_len: Option<u64>,
    spelling_loc: Option<Box<Self>>,
}

impl Location {
    fn spelling(&self) -> &Self {
        self.spelling_loc.as_deref().unwrap_or(self)
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AstType {
    qual_type: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct SourceRange {
    begin: Option<Location>,
    end: Option<Location>,
}

#[derive(Clone, Debug, Default)]
struct Node {
    id: String,
    kind: String,
    location: Option<Location>,
    range: Option<SourceRange>,
    implicit: bool,
    complete_definition: bool,
    is_this_declaration_a_definition: bool,
    name: String,
    ast_type: Option<AstType>,
    fixed_underlying_type: Option<AstType>,
    storage_class: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScopeKind {
    Namespace(Option<&'static str>),
    Record(&'static str),
    Enum,
    Function,
    Lexical,
}

struct State<'a> {
    materialized: &'a Path,
    original: &'a Path,
    current_file: Option<String>,
    parents: Vec<String>,
    scope_kinds: Vec<ScopeKind>,
    scope_files: Vec<Option<String>>,
    file_cache: BTreeMap<String, Vec<u8>>,
    items: &'a mut Vec<Extracted>,
    coverage: &'a mut Vec<Coverage>,
    nodes_seen: u64,
    depth: usize,
    max_depth: usize,
}

impl State<'_> {
    #[allow(clippy::too_many_lines)]
    fn process(&mut self, node: &Node) -> Result<(), String> {
        self.nodes_seen = self
            .nodes_seen
            .checked_add(1)
            .ok_or("AST node count overflow")?;
        self.max_depth = self.max_depth.max(self.parents.len());
        let explicit_file = node
            .location
            .as_ref()
            .and_then(|location| location.spelling().file.clone());
        if let Some(file) = explicit_file.as_ref() {
            self.current_file = Some(file.clone());
        }
        if node.implicit || !node.kind.ends_with("Decl") {
            return Ok(());
        }
        if excluded_declaration_kind(&node.kind) {
            return Ok(());
        }
        let Some(file) = explicit_file
            .as_deref()
            .or_else(|| self.scope_files.last().and_then(Option::as_deref))
            .or(self.current_file.as_deref())
            .and_then(|file| normalize_owner(file, self.materialized, self.original))
        else {
            return Ok(());
        };
        let local = self
            .scope_kinds
            .iter()
            .any(|scope| matches!(scope, ScopeKind::Function | ScopeKind::Lexical));
        if local {
            // Full canonical function bodies already bind parameters, local variables, structured
            // bindings, local types, and nested block items. Keep declaration granularity symmetric
            // with Rust by inventorying namespace/module/type/member declarations only.
            return Ok(());
        }
        let (line, column) = self.location(node.location.as_ref(), &file)?;
        let range = node
            .range
            .as_ref()
            .ok_or_else(|| format!("included {} in {file} lacks source range", node.kind))?;
        let (begin_line, begin_column) = self.location(range.begin.as_ref(), &file)?;
        let (end_line, end_column) = self.location(range.end.as_ref(), &file)?;
        if [line, column, begin_line, begin_column, end_line, end_column].contains(&0) {
            return Err(format!(
                "included {} in {file} lacks exact spelling span",
                node.kind
            ));
        }
        let signature = canonicalize_source_roots(
            declaration_signature(node)?,
            self.materialized,
            self.original,
        );
        let identity_name =
            canonical_declaration_identity(&node.kind, &node.name, Some(&signature), line, column);
        let qualified = if node.name.is_empty() {
            if line == 0 || column == 0 {
                return Err(format!(
                    "anonymous {} in {file} lacks exact spelling location",
                    node.kind
                ));
            }
            let anonymous = anonymous_scope_identity(&node.kind, line, column);
            if self.parents.is_empty() {
                format!("::{anonymous}")
            } else {
                format!("{}::{anonymous}", self.parents.join("::"))
            }
        } else if self.parents.is_empty() {
            identity_name
        } else {
            format!("{}::{identity_name}", self.parents.join("::"))
        };
        let owner_kind = self.scope_kinds.iter().rev().find_map(|scope| match scope {
            ScopeKind::Record(kind) | ScopeKind::Namespace(Some(kind)) => Some(*kind),
            _ => None,
        });
        let externally_visible = node.storage_class.as_deref() != Some("static");
        let item_kind = classify_decl(
            &node.kind,
            &node.name,
            local,
            externally_visible,
            owner_kind,
        )?;
        let payload = if requires_spelling_payload(&node.kind) {
            format!(
                "{}|{signature}|{}",
                node.kind,
                self.spelling_payload(node, &file)?
            )
        } else {
            format!("{}|{signature}", node.kind)
        };
        let inventory_identity = canonical_redeclaration_identity(
            &node.kind,
            &qualified,
            node.complete_definition || node.is_this_declaration_a_definition,
            line,
            column,
        );
        let node_id = semantic_hash(&[
            "clang-ast-node/v1",
            &file,
            &node.kind,
            &inventory_identity,
            &payload,
            &line.to_string(),
            &column.to_string(),
        ]);
        self.items.push(Extracted {
            path: file.clone(),
            kind: item_kind.into(),
            name: inventory_identity.clone(),
            payload,
            note: "clang-ast".into(),
        });
        self.coverage.push(Coverage {
            path: file,
            line,
            column,
            begin_line,
            begin_column,
            end_line,
            end_column,
            kind: node.kind.clone(),
            name: inventory_identity,
            signature,
            node_id,
        });
        Ok(())
    }

    fn location(&mut self, location: Option<&Location>, owner: &str) -> Result<(u64, u64), String> {
        let Some(location) = location.map(Location::spelling) else {
            return Ok((0, 0));
        };
        let line = location.line.unwrap_or(0);
        let column = location.col.unwrap_or(0);
        if line != 0 && column != 0 {
            return Ok((line, column));
        }
        let Some(offset) = location.offset else {
            return Ok((line, column));
        };
        if !self.file_cache.contains_key(owner) {
            let bytes = fs::read(self.materialized.join(owner))
                .map_err(|error| format!("cannot resolve spelling offset in {owner}: {error}"))?;
            self.file_cache.insert(owner.to_owned(), bytes);
        }
        let bytes = self.file_cache.get(owner).expect("inserted source bytes");
        let offset =
            usize::try_from(offset).map_err(|_| format!("spelling offset overflow in {owner}"))?;
        if offset > bytes.len() {
            return Err(format!("spelling offset outside {owner}"));
        }
        let prefix = &bytes[..offset];
        let newline_count = prefix
            .iter()
            .fold(0_usize, |count, byte| count + usize::from(*byte == b'\n'));
        let resolved_line =
            u64::try_from(newline_count + 1).map_err(|_| format!("line overflow in {owner}"))?;
        let resolved_column = u64::try_from(
            prefix
                .iter()
                .rev()
                .take_while(|byte| **byte != b'\n')
                .count()
                + 1,
        )
        .map_err(|_| format!("column overflow in {owner}"))?;
        Ok((resolved_line, resolved_column))
    }

    fn spelling_payload(&mut self, node: &Node, owner: &str) -> Result<String, String> {
        let range = node
            .range
            .as_ref()
            .ok_or_else(|| format!("behavior-bearing {} in {owner} lacks a range", node.kind))?;
        let begin = range
            .begin
            .as_ref()
            .map(Location::spelling)
            .ok_or_else(|| {
                format!(
                    "behavior-bearing {} in {owner} lacks range begin",
                    node.kind
                )
            })?;
        let end =
            range.end.as_ref().map(Location::spelling).ok_or_else(|| {
                format!("behavior-bearing {} in {owner} lacks range end", node.kind)
            })?;
        self.validate_range_owner(begin, owner, &node.kind)?;
        self.validate_range_owner(end, owner, &node.kind)?;
        let start = begin
            .offset
            .ok_or_else(|| format!("{} range begin in {owner} lacks spelling offset", node.kind))?;
        let end_offset = end
            .offset
            .ok_or_else(|| format!("{} range end in {owner} lacks spelling offset", node.kind))?;
        let token_length = end
            .tok_len
            .ok_or_else(|| format!("{} range end in {owner} lacks token length", node.kind))?;
        let start = usize::try_from(start)
            .map_err(|_| format!("{} range begin overflow in {owner}", node.kind))?;
        let end_offset = usize::try_from(end_offset)
            .map_err(|_| format!("{} range end overflow in {owner}", node.kind))?;
        let token_length = usize::try_from(token_length)
            .map_err(|_| format!("{} token length overflow in {owner}", node.kind))?;
        let exclusive = end_offset
            .checked_add(token_length)
            .ok_or_else(|| format!("{} range end arithmetic overflow in {owner}", node.kind))?;
        if start >= exclusive {
            return Err(format!(
                "{} has an empty or reversed range in {owner}",
                node.kind
            ));
        }
        if !self.file_cache.contains_key(owner) {
            let bytes = fs::read(self.materialized.join(owner))
                .map_err(|error| format!("cannot read spelling source {owner}: {error}"))?;
            self.file_cache.insert(owner.to_owned(), bytes);
        }
        let bytes = self.file_cache.get(owner).expect("inserted source bytes");
        let spelling = bytes
            .get(start..exclusive)
            .ok_or_else(|| format!("{} spelling range is outside {owner}", node.kind))?;
        let spelling = std::str::from_utf8(spelling).map_err(|error| {
            format!(
                "{} spelling range in {owner} is not UTF-8: {error}",
                node.kind
            )
        })?;
        let canonical = canonical_tokens(spelling)?;
        if canonical.is_empty() {
            return Err(format!("{} spelling range in {owner} is empty", node.kind));
        }
        Ok(canonical)
    }

    fn validate_range_owner(
        &self,
        location: &Location,
        owner: &str,
        kind: &str,
    ) -> Result<(), String> {
        if let Some(file) = location.file.as_deref() {
            let normalized = normalize_owner(file, self.materialized, self.original)
                .ok_or_else(|| format!("{kind} spelling range is outside pinned model sources"))?;
            if normalized != owner {
                return Err(format!(
                    "{kind} spelling range crosses from {owner} into {normalized}"
                ));
            }
        }
        Ok(())
    }

    fn node_owner_file<'a>(&'a self, node: &'a Node) -> Option<&'a str> {
        node.location
            .as_ref()
            .and_then(|location| location.spelling().file.as_deref())
            .or_else(|| self.scope_files.last().and_then(Option::as_deref))
            .or(self.current_file.as_deref())
    }

    fn push_scope(&mut self, node: &Node, identity: String, scope_kind: ScopeKind) {
        let owner_file = self.node_owner_file(node).map(str::to_owned);
        self.parents.push(identity);
        self.scope_kinds.push(scope_kind);
        self.scope_files.push(owner_file);
    }

    fn push_parent(&mut self, node: &Node) -> Result<bool, String> {
        let scope_kind = if node.kind == "NamespaceDecl" {
            Some(ScopeKind::Namespace(namespace_semantic_kind(&node.name)))
        } else if matches!(
            node.kind.as_str(),
            "CXXRecordDecl" | "RecordDecl" | "ClassTemplateDecl"
        ) {
            Some(ScopeKind::Record(classify_decl(
                &node.kind,
                &node.name,
                false,
                true,
                self.scope_kinds.iter().rev().find_map(|scope| match scope {
                    ScopeKind::Record(kind) | ScopeKind::Namespace(Some(kind)) => Some(*kind),
                    _ => None,
                }),
            )?))
        } else if node.kind == "EnumDecl" {
            Some(ScopeKind::Enum)
        } else if matches!(
            node.kind.as_str(),
            "FunctionDecl" | "CXXMethodDecl" | "CXXConstructorDecl" | "CXXDestructorDecl"
        ) {
            Some(ScopeKind::Function)
        } else {
            None
        };
        if let Some(scope_kind) = scope_kind {
            if !node.name.is_empty() {
                self.push_scope(node, declaration_name(node), scope_kind);
                return Ok(true);
            }
            if node.kind == "NamespaceDecl" {
                let owner = self
                    .node_owner_file(node)
                    .and_then(|file| normalize_owner(file, self.materialized, self.original))
                    .ok_or("anonymous namespace is outside pinned model sources")?;
                let (line, column) = self.location(node.location.as_ref(), &owner)?;
                if line == 0 || column == 0 {
                    return Err(format!(
                        "anonymous NamespaceDecl in {owner} lacks exact spelling location"
                    ));
                }
                self.push_scope(
                    node,
                    anonymous_scope_identity(&node.kind, line, column),
                    scope_kind,
                );
                return Ok(true);
            }
            Ok(false)
        } else if matches!(
            node.kind.as_str(),
            "CompoundStmt"
                | "ForStmt"
                | "CXXForRangeStmt"
                | "IfStmt"
                | "WhileStmt"
                | "SwitchStmt"
                | "DoStmt"
                | "LambdaExpr"
                | "CXXCatchStmt"
        ) {
            let owner = self
                .current_file
                .as_deref()
                .and_then(|file| normalize_owner(file, self.materialized, self.original));
            let Some(owner) = owner else { return Ok(false) };
            let (line, column) = self.location(
                node.range.as_ref().and_then(|range| range.begin.as_ref()),
                &owner,
            )?;
            if line == 0 || column == 0 {
                return Err(format!(
                    "{} in {owner} lacks exact lexical-scope location",
                    node.kind
                ));
            }
            self.push_scope(
                node,
                format!("<{}@{line}:{column}>", node.kind),
                ScopeKind::Lexical,
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

fn declaration_name(node: &Node) -> String {
    canonical_callable_identity(
        &node.kind,
        &node.name,
        node.ast_type
            .as_ref()
            .and_then(|ast_type| ast_type.qual_type.as_deref()),
    )
    .unwrap_or_else(|| node.name.clone())
}

pub fn canonical_callable_identity(
    kind: &str,
    declaration_name: &str,
    direct_signature: Option<&str>,
) -> Option<String> {
    is_callable_declaration_kind(kind).then(|| {
        let signature = direct_signature.unwrap_or(kind);
        format!("{declaration_name}({signature})")
    })
}

pub fn canonical_declaration_identity(
    kind: &str,
    declaration_name: &str,
    direct_signature: Option<&str>,
    line: u64,
    column: u64,
) -> String {
    if kind == "FunctionTemplateDecl" {
        format!("{declaration_name}::<template@{line}:{column}>")
    } else {
        canonical_callable_identity(kind, declaration_name, direct_signature)
            .unwrap_or_else(|| declaration_name.to_owned())
    }
}

pub fn is_callable_declaration_kind(kind: &str) -> bool {
    matches!(
        kind,
        "FunctionDecl"
            | "CXXMethodDecl"
            | "CXXConstructorDecl"
            | "CXXDestructorDecl"
            | "FunctionTemplateDecl"
    )
}

pub fn canonical_redeclaration_identity(
    kind: &str,
    qualified_identity: &str,
    is_definition: bool,
    line: u64,
    column: u64,
) -> String {
    if is_redeclarable_type_kind(kind) && !is_definition {
        format!("{qualified_identity}::<declaration@{line}:{column}>")
    } else {
        qualified_identity.to_owned()
    }
}

pub fn is_redeclarable_type_kind(kind: &str) -> bool {
    matches!(
        kind,
        "CXXRecordDecl" | "RecordDecl" | "ClassTemplateSpecializationDecl"
    )
}

fn declaration_signature(node: &Node) -> Result<&str, String> {
    if node.kind == "EnumDecl"
        && let Some(fixed_underlying_type) = node.fixed_underlying_type.as_ref()
    {
        return fixed_underlying_type.qual_type.as_deref().ok_or_else(|| {
            format!(
                "EnumDecl {:?} fixedUnderlyingType lacks a string qualType",
                node.name
            )
        });
    }
    Ok(node
        .ast_type
        .as_ref()
        .and_then(|ast_type| ast_type.qual_type.as_deref())
        .unwrap_or(&node.kind))
}

pub fn anonymous_scope_identity(kind: &str, line: u64, column: u64) -> String {
    format!("<anonymous-{kind}@{line}:{column}>")
}

fn requires_spelling_payload(kind: &str) -> bool {
    matches!(
        kind,
        "FunctionDecl"
            | "CXXMethodDecl"
            | "CXXConstructorDecl"
            | "CXXDestructorDecl"
            | "FunctionTemplateDecl"
            | "CXXRecordDecl"
            | "RecordDecl"
            | "ClassTemplateDecl"
            | "ClassTemplateSpecializationDecl"
            | "EnumDecl"
            | "EnumConstantDecl"
            | "VarDecl"
            | "FieldDecl"
            | "TypedefDecl"
            | "TypeAliasDecl"
            | "TypeAliasTemplateDecl"
            | "VarTemplateDecl"
            | "StaticAssertDecl"
            | "ConceptDecl"
    )
}

fn namespace_semantic_kind(name: &str) -> Option<&'static str> {
    match name {
        "guard" | "guards" => Some("guard"),
        "action" | "actions" => Some("action"),
        _ => None,
    }
}

pub fn excluded_declaration_kind(kind: &str) -> bool {
    matches!(
        kind,
        "TranslationUnitDecl"
            | "ParmVarDecl"
            | "DecompositionDecl"
            | "BindingDecl"
            | "TemplateTypeParmDecl"
            | "NonTypeTemplateParmDecl"
            | "TemplateTemplateParmDecl"
            | "AccessSpecDecl"
            | "EmptyDecl"
            | "UsingShadowDecl"
            | "ConstructorUsingShadowDecl"
            | "LinkageSpecDecl"
            | "FriendDecl"
    )
}

struct NodeSeed<'state, 'paths> {
    state: &'state mut State<'paths>,
}

impl<'de> DeserializeSeed<'de> for NodeSeed<'_, '_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(NodeVisitor { state: self.state })
    }
}

struct NodeVisitor<'state, 'paths> {
    state: &'state mut State<'paths>,
}

impl<'de> Visitor<'de> for NodeVisitor<'_, '_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a Clang AST node object")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut node = Node::default();
        let mut processed = false;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "id" => node.id = map.next_value()?,
                "kind" => node.kind = map.next_value()?,
                "loc" => node.location = Some(map.next_value()?),
                "range" => node.range = Some(map.next_value()?),
                "isImplicit" => node.implicit = map.next_value()?,
                "completeDefinition" => node.complete_definition = map.next_value()?,
                "isThisDeclarationADefinition" => {
                    node.is_this_declaration_a_definition = map.next_value()?;
                }
                "name" => node.name = map.next_value()?,
                "type" => node.ast_type = Some(map.next_value()?),
                "fixedUnderlyingType" => node.fixed_underlying_type = Some(map.next_value()?),
                "storageClass" => node.storage_class = Some(map.next_value()?),
                "inner" => {
                    if node.kind.is_empty() {
                        return Err(de::Error::custom("AST inner appeared before kind"));
                    }
                    self.state.process(&node).map_err(de::Error::custom)?;
                    processed = true;
                    let pushed = self.state.push_parent(&node).map_err(de::Error::custom)?;
                    self.state.depth = self
                        .state
                        .depth
                        .checked_add(1)
                        .ok_or_else(|| de::Error::custom("AST depth overflow"))?;
                    if self.state.depth > 512 {
                        return Err(de::Error::custom("AST depth exceeds 512-node safety cap"));
                    }
                    self.state.max_depth = self.state.max_depth.max(self.state.depth);
                    map.next_value_seed(ChildrenSeed { state: self.state })?;
                    self.state.depth -= 1;
                    if pushed {
                        self.state.parents.pop();
                        self.state.scope_kinds.pop();
                        self.state.scope_files.pop();
                    }
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        if !processed {
            if node.kind.is_empty() {
                // Clang uses kindless `inner` objects for base/type metadata. They cannot be
                // declarations because declaration identity is carried by a `*Decl` kind.
                return Ok(());
            }
            self.state.process(&node).map_err(de::Error::custom)?;
        }
        Ok(())
    }
}

struct ChildrenSeed<'state, 'paths> {
    state: &'state mut State<'paths>,
}

impl<'de> DeserializeSeed<'de> for ChildrenSeed<'_, '_> {
    type Value = ();
    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(ChildrenVisitor { state: self.state })
    }
}

struct ChildrenVisitor<'state, 'paths> {
    state: &'state mut State<'paths>,
}

impl<'de> Visitor<'de> for ChildrenVisitor<'_, '_> {
    type Value = ();
    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a Clang AST child array")
    }
    fn visit_seq<S>(self, mut sequence: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        while sequence
            .next_element_seed(NodeSeed { state: self.state })?
            .is_some()
        {}
        Ok(())
    }
}

pub fn run_clang(
    arguments: &[String],
    temporary: &Path,
    materialized: &Path,
    original: &Path,
    items: &mut Vec<Extracted>,
    coverage: &mut Vec<Coverage>,
    deadline: Instant,
) -> Result<(), String> {
    let stderr = tempfile::NamedTempFile::new_in(temporary)
        .map_err(|error| format!("cannot create Clang stderr capture: {error}"))?;
    let stderr_file = stderr
        .reopen()
        .map_err(|error| format!("cannot open Clang stderr capture: {error}"))?;
    let mut child = Command::new(CLANG)
        .args(arguments)
        .current_dir(temporary)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(stderr_file)
        .spawn()
        .map_err(|error| format!("cannot execute Clang: {error}"))?;
    let stdout = child.stdout.take().ok_or("Clang stdout pipe unavailable")?;
    let child = Arc::new(Mutex::new(child));
    let watchdog_thread = spawn_watchdog(Arc::clone(&child), deadline);
    let parse_result = parse_filtered_reader(
        FilteredAstReader::new(BufReader::with_capacity(64 * 1024, stdout), deadline),
        materialized,
        original,
        items,
        coverage,
    );
    if parse_result.is_err()
        && let Ok(mut child) = child.lock()
    {
        let _ = child.kill();
    }
    let watch_result = watchdog_thread
        .join()
        .map_err(|_| "Clang watchdog thread panicked".to_owned())??;
    let diagnostic = fs::read_to_string(stderr.path()).unwrap_or_default();
    if watch_result.timed_out {
        return Err("full AST extraction exceeded 300-second wall limit".to_owned());
    }
    if let Err(error) = parse_result {
        let suffix = if diagnostic.trim().is_empty() {
            String::new()
        } else {
            format!(": {}", diagnostic.trim())
        };
        return Err(format!("{error}{suffix}"));
    }
    if !watch_result.status.success() {
        return Err(format!(
            "Clang AST extraction failed: {}",
            diagnostic.trim()
        ));
    }
    Ok(())
}

pub fn run_clang_capture(
    arguments: &[String],
    temporary: &Path,
    deadline: Instant,
) -> Result<TextualAstCapture, String> {
    let stderr = tempfile::NamedTempFile::new_in(temporary)
        .map_err(|error| format!("cannot create Clang stderr capture: {error}"))?;
    let stderr_file = stderr
        .reopen()
        .map_err(|error| format!("cannot open Clang stderr capture: {error}"))?;
    let mut child = Command::new(CLANG)
        .args(arguments)
        .current_dir(temporary)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(stderr_file)
        .spawn()
        .map_err(|error| format!("cannot execute textual Clang: {error}"))?;
    let mut stdout = child.stdout.take().ok_or("Clang stdout pipe unavailable")?;
    let mut output = tempfile::NamedTempFile::new_in(temporary)
        .map_err(|error| format!("cannot create textual Clang AST capture: {error}"))?;
    let child = Arc::new(Mutex::new(child));
    let watchdog_thread = spawn_watchdog(Arc::clone(&child), deadline);
    let read_result = copy_bounded(
        &mut stdout,
        output.as_file_mut(),
        TEXTUAL_AST_CAPTURE_LIMIT_BYTES,
    );
    if read_result.is_err()
        && let Ok(mut child) = child.lock()
    {
        let _ = child.kill();
    }
    let watch_result = watchdog_thread
        .join()
        .map_err(|_| "textual Clang watchdog thread panicked".to_owned())??;
    let diagnostic = fs::read_to_string(stderr.path()).unwrap_or_default();
    if watch_result.timed_out {
        return Err("full AST extraction exceeded 300-second wall limit".into());
    }
    let bytes = read_result.map_err(|error| format!("cannot read textual Clang AST: {error}"))?;
    if !watch_result.status.success() {
        return Err(format!("textual Clang AST failed: {}", diagnostic.trim()));
    }
    Ok(TextualAstCapture {
        file: output,
        bytes,
    })
}

fn copy_bounded(
    reader: &mut impl Read,
    writer: &mut impl Write,
    limit: u64,
) -> Result<u64, String> {
    let mut buffer = [0u8; 16 * 1024];
    let mut written = 0u64;
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("stream read failed: {error}"))?;
        if count == 0 {
            return Ok(written);
        }
        let count = u64::try_from(count).map_err(|_| "textual AST read size overflow")?;
        written = written
            .checked_add(count)
            .ok_or("textual AST byte count overflow")?;
        if written > limit {
            return Err(format!(
                "textual Clang AST exceeded deterministic {limit}-byte capture limit"
            ));
        }
        let count = usize::try_from(count).map_err(|_| "textual AST write size overflow")?;
        writer
            .write_all(&buffer[..count])
            .map_err(|error| format!("capture write failed: {error}"))?;
    }
}

struct WatchResult {
    status: std::process::ExitStatus,
    timed_out: bool,
}

fn spawn_watchdog(
    child: Arc<Mutex<std::process::Child>>,
    deadline: Instant,
) -> JoinHandle<Result<WatchResult, String>> {
    thread::spawn(move || {
        loop {
            {
                let mut child = child
                    .lock()
                    .map_err(|_| "Clang child mutex was poisoned".to_owned())?;
                if let Some(status) = child
                    .try_wait()
                    .map_err(|error| format!("cannot poll Clang child: {error}"))?
                {
                    drop(child);
                    return Ok(WatchResult {
                        status,
                        timed_out: false,
                    });
                }
                if Instant::now() >= deadline {
                    child
                        .kill()
                        .map_err(|error| format!("cannot kill timed-out Clang child: {error}"))?;
                    let status = child
                        .wait()
                        .map_err(|error| format!("cannot reap timed-out Clang child: {error}"))?;
                    drop(child);
                    return Ok(WatchResult {
                        status,
                        timed_out: true,
                    });
                }
            }
            thread::sleep(Duration::from_millis(10));
        }
    })
}

struct FilteredAstReader<R> {
    source: R,
    deadline: Instant,
    pending: Vec<u8>,
    pending_offset: usize,
    depth: usize,
    string_state: StringState,
    document_state: DocumentState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StringState {
    Outside,
    Inside,
    Escaped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DocumentState {
    Empty,
    Objects,
    Finished,
}

impl<R> FilteredAstReader<R> {
    fn new(source: R, deadline: Instant) -> Self {
        Self {
            source,
            deadline,
            pending: br#"{"kind":"TranslationUnitDecl","inner":["#.to_vec(),
            pending_offset: 0,
            depth: 0,
            string_state: StringState::Outside,
            document_state: DocumentState::Empty,
        }
    }
}

impl<R: Read> FilteredAstReader<R> {
    fn refill(&mut self) -> std::io::Result<()> {
        self.pending.clear();
        self.pending_offset = 0;
        if self.document_state == DocumentState::Finished {
            return Ok(());
        }
        if Instant::now() >= self.deadline {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "full AST extraction exceeded 300-second wall limit",
            ));
        }
        let mut input = [0_u8; 8 * 1024];
        let count = self.source.read(&mut input)?;
        if count == 0 {
            if self.string_state != StringState::Outside || self.depth != 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "truncated filtered Clang AST object",
                ));
            }
            self.pending.extend_from_slice(b"]}");
            self.document_state = DocumentState::Finished;
            return Ok(());
        }
        for byte in &input[..count] {
            if self.string_state != StringState::Outside {
                self.pending.push(*byte);
                if self.string_state == StringState::Escaped {
                    self.string_state = StringState::Inside;
                } else if *byte == b'\\' {
                    self.string_state = StringState::Escaped;
                } else if *byte == b'"' {
                    self.string_state = StringState::Outside;
                }
                continue;
            }
            if *byte == b'"' && self.depth != 0 {
                self.string_state = StringState::Inside;
                self.pending.push(*byte);
                continue;
            }
            if self.depth == 0 {
                if byte.is_ascii_whitespace() {
                    continue;
                }
                if *byte != b'{' {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "filtered Clang AST contains a non-object top-level value",
                    ));
                }
                if self.document_state == DocumentState::Objects {
                    self.pending.push(b',');
                }
                self.document_state = DocumentState::Objects;
                self.depth = 1;
                self.pending.push(*byte);
                continue;
            }
            self.pending.push(*byte);
            match *byte {
                b'{' | b'[' => self.depth += 1,
                b'}' | b']' => {
                    self.depth = self.depth.checked_sub(1).ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "filtered Clang AST delimiter underflow",
                        )
                    })?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

impl<R: Read> Read for FilteredAstReader<R> {
    fn read(&mut self, target: &mut [u8]) -> std::io::Result<usize> {
        if target.is_empty() {
            return Ok(0);
        }
        while self.pending_offset == self.pending.len() {
            self.refill()?;
            if self.pending.is_empty() {
                return Ok(0);
            }
        }
        let count = target
            .len()
            .min(self.pending.len().saturating_sub(self.pending_offset));
        target[..count]
            .copy_from_slice(&self.pending[self.pending_offset..self.pending_offset + count]);
        self.pending_offset += count;
        Ok(count)
    }
}

fn parse_filtered_reader(
    reader: impl Read + Send,
    materialized: &Path,
    original: &Path,
    items: &mut Vec<Extracted>,
    coverage: &mut Vec<Coverage>,
) -> Result<(u64, usize), String> {
    parse_reader(reader, materialized, original, items, coverage)
}

fn parse_reader(
    reader: impl Read + Send,
    materialized: &Path,
    original: &Path,
    items: &mut Vec<Extracted>,
    coverage: &mut Vec<Coverage>,
) -> Result<(u64, usize), String> {
    thread::scope(|scope| {
        thread::Builder::new()
            .name("emel-model-ast-parser".into())
            .stack_size(16 * 1024 * 1024)
            .spawn_scoped(scope, move || {
                parse_reader_inner(reader, materialized, original, items, coverage)
            })
            .map_err(|error| format!("cannot spawn bounded AST parser: {error}"))?
            .join()
            .map_err(|_| "bounded AST parser thread panicked".to_owned())?
    })
}

fn parse_reader_inner(
    reader: impl Read,
    materialized: &Path,
    original: &Path,
    items: &mut Vec<Extracted>,
    coverage: &mut Vec<Coverage>,
) -> Result<(u64, usize), String> {
    let mut state = State {
        materialized,
        original,
        current_file: None,
        parents: Vec::new(),
        scope_kinds: Vec::new(),
        scope_files: Vec::new(),
        file_cache: BTreeMap::new(),
        items,
        coverage,
        nodes_seen: 0,
        depth: 0,
        max_depth: 0,
    };
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    deserializer.disable_recursion_limit();
    NodeSeed { state: &mut state }
        .deserialize(&mut deserializer)
        .map_err(|error| format!("malformed or partial Clang AST JSON: {error}"))?;
    deserializer
        .end()
        .map_err(|error| format!("trailing Clang AST JSON: {error}"))?;
    Ok((state.nodes_seen, state.max_depth))
}

pub fn validate_untrusted_filtered_ast(bytes: &[u8]) -> Result<(), String> {
    let temp = tempfile::tempdir()
        .map_err(|error| format!("cannot create AST validation directory: {error}"))?;
    let reader = FilteredAstReader::new(
        std::io::Cursor::new(bytes.to_vec()),
        Instant::now() + Duration::from_secs(1),
    );
    let mut items = Vec::new();
    let mut coverage = Vec::new();
    parse_filtered_reader(reader, temp.path(), temp.path(), &mut items, &mut coverage)?;
    Ok(())
}

fn normalize_owner(file: &str, materialized: &Path, original: &Path) -> Option<String> {
    let path = PathBuf::from(file);
    let relative = path
        .strip_prefix(materialized)
        .or_else(|_| path.strip_prefix(original))
        .ok()?;
    let normalized = normalize_relative_owner_path(relative)?;
    (normalized.starts_with("src/emel/model/") || normalized.starts_with("tests/model/"))
        .then_some(normalized)
}

pub fn normalize_relative_owner_path(relative: &Path) -> Option<String> {
    let mut parts = Vec::new();
    for component in relative.components() {
        match component {
            std::path::Component::Normal(part) => {
                let part = part.to_str()?;
                if part.contains(['/', '\\']) || part.is_empty() {
                    return None;
                }
                parts.push(part.to_owned());
            }
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                parts.pop()?;
            }
            std::path::Component::Prefix(_) | std::path::Component::RootDir => return None,
        }
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

pub fn canonicalize_source_roots(text: &str, materialized: &Path, original: &Path) -> String {
    let materialized = materialized.to_string_lossy();
    let original = original.to_string_lossy();
    let canonical = replace_source_root(text, &materialized);
    replace_source_root(&canonical, &original)
}

fn replace_source_root(text: &str, root: &str) -> String {
    if root.is_empty() {
        return text.to_owned();
    }
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some(relative) = text[cursor..].find(root) {
        let start = cursor + relative;
        let end = start + root.len();
        let before_is_boundary = text[..start].chars().next_back().is_none_or(|character| {
            !character.is_alphanumeric() && !matches!(character, '_' | '-' | '.' | '/' | '\\')
        });
        let after_is_boundary = text[end..]
            .chars()
            .next()
            .is_none_or(|character| matches!(character, '/' | '\\'));
        if before_is_boundary && after_is_boundary {
            output.push_str(&text[cursor..start]);
            output.push_str("<source-root>");
            cursor = end;
        } else {
            let next = start + root.chars().next().map_or(1, char::len_utf8);
            output.push_str(&text[cursor..next]);
            cursor = next;
        }
    }
    output.push_str(&text[cursor..]);
    output
}

fn classify_decl(
    kind: &str,
    name: &str,
    local: bool,
    externally_visible: bool,
    owner_kind: Option<&'static str>,
) -> Result<&'static str, String> {
    let lower = name.to_ascii_lowercase();
    let classified = if matches!(
        kind,
        "FunctionDecl"
            | "CXXMethodDecl"
            | "CXXConstructorDecl"
            | "CXXDestructorDecl"
            | "FunctionTemplateDecl"
    ) {
        if name == "process_event" || name == "main" {
            "runtime-entrypoint"
        } else if matches!(owner_kind, Some("guard" | "action")) {
            owner_kind.expect("matched semantic callable owner")
        } else if name.starts_with("guard_") || name.starts_with("is_") || name.starts_with("has_")
        {
            "guard"
        } else if name.starts_with("action_") || name.starts_with("effect_") {
            "action"
        } else {
            "algorithm"
        }
    } else if kind == "FieldDecl" {
        owner_kind
            .filter(|kind| matches!(*kind, "context" | "event" | "outcome" | "error"))
            .unwrap_or("type")
    } else if matches!(kind, "ParmVarDecl" | "DecompositionDecl" | "BindingDecl")
        || (kind == "VarDecl" && local)
    {
        "algorithm"
    } else if kind == "VarDecl" && externally_visible {
        "public-api"
    } else if kind == "VarDecl" {
        "algorithm"
    } else if matches!(
        kind,
        "CXXRecordDecl"
            | "RecordDecl"
            | "EnumDecl"
            | "TypedefDecl"
            | "TypeAliasDecl"
            | "ClassTemplateDecl"
            | "ClassTemplateSpecializationDecl"
            | "EnumConstantDecl"
    ) {
        if matches!(owner_kind, Some("guard" | "action")) {
            owner_kind.expect("matched semantic type owner")
        } else if lower.contains("error") {
            "error"
        } else if lower.ends_with("done") || lower.contains("outcome") {
            "outcome"
        } else if lower.contains("event") {
            "event"
        } else if lower.contains("context") {
            "context"
        } else {
            "type"
        }
    } else if kind == "StaticAssertDecl" {
        "assertion"
    } else if matches!(
        kind,
        "NamespaceDecl"
            | "NamespaceAliasDecl"
            | "UsingDecl"
            | "UsingDirectiveDecl"
            | "TypeAliasTemplateDecl"
            | "VarTemplateDecl"
            | "ConceptDecl"
    ) {
        "public-api"
    } else {
        return Err(format!("unsupported semantic declaration node: {kind}"));
    };
    Ok(classified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct GeneratedAst {
        remaining: usize,
        buffer: Vec<u8>,
        offset: usize,
        started: bool,
        finished: bool,
    }
    impl GeneratedAst {
        fn new(nodes: usize) -> Self {
            Self {
                remaining: nodes,
                buffer: Vec::new(),
                offset: 0,
                started: false,
                finished: false,
            }
        }
    }
    impl Read for GeneratedAst {
        fn read(&mut self, target: &mut [u8]) -> std::io::Result<usize> {
            if self.offset == self.buffer.len() {
                self.buffer = if !self.started {
                    self.started = true;
                    br#"{"kind":"TranslationUnitDecl","inner":["#.to_vec()
                } else if self.remaining > 0 {
                    self.remaining -= 1;
                    let comma = if self.remaining == 0 { "" } else { "," };
                    format!(r#"{{"kind":"NamespaceDecl","name":"external"}}{comma}"#).into_bytes()
                } else if !self.finished {
                    self.finished = true;
                    b"]}".to_vec()
                } else {
                    Vec::new()
                };
                self.offset = 0;
            }
            if self.buffer.is_empty() {
                return Ok(0);
            }
            let count = target.len().min(self.buffer.len() - self.offset);
            target[..count].copy_from_slice(&self.buffer[self.offset..self.offset + count]);
            self.offset += count;
            Ok(count)
        }
    }

    #[test]
    fn streams_large_ast_with_bounded_retained_state() {
        let temp = tempfile::tempdir().unwrap();
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        let (nodes, depth) = parse_reader(
            GeneratedAst::new(250_000),
            temp.path(),
            temp.path(),
            &mut items,
            &mut coverage,
        )
        .unwrap();
        assert_eq!(nodes, 250_001);
        assert!(depth <= 1);
        assert!(items.is_empty());
        assert!(coverage.is_empty());
    }

    #[test]
    fn rejects_truncated_ast() {
        let temp = tempfile::tempdir().unwrap();
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        assert!(
            parse_reader(
                &b"{\"kind\":\"TranslationUnitDecl\",\"inner\":[{"[..],
                temp.path(),
                temp.path(),
                &mut items,
                &mut coverage
            )
            .is_err()
        );
    }

    fn nested_ast(depth: usize) -> Vec<u8> {
        let mut ast = String::new();
        for _ in 0..depth {
            ast.push_str(r#"{"kind":"NamespaceDecl","name":"n","inner":["#);
        }
        ast.push_str(r#"{"kind":"NamespaceDecl","name":"leaf"}"#);
        for _ in 0..depth {
            ast.push_str("]}");
        }
        ast.into_bytes()
    }

    #[test]
    fn accepts_ast_depth_at_safety_cap() {
        let temp = tempfile::tempdir().unwrap();
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        let (_, depth) = parse_reader(
            &nested_ast(512)[..],
            temp.path(),
            temp.path(),
            &mut items,
            &mut coverage,
        )
        .unwrap();
        assert_eq!(depth, 512);
    }

    #[test]
    fn rejects_ast_depth_above_safety_cap() {
        let temp = tempfile::tempdir().unwrap();
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        let error = parse_reader(
            &nested_ast(513)[..],
            temp.path(),
            temp.path(),
            &mut items,
            &mut coverage,
        )
        .unwrap_err();
        assert!(error.contains("AST depth exceeds 512-node safety cap"));
    }

    #[test]
    fn filtered_stream_wraps_concatenated_objects() {
        let temp = tempfile::tempdir().unwrap();
        let input = br#"{"kind":"NamespaceDecl","name":"emel"}
{"kind":"NamespaceDecl","name":"emel_window_test"}"#;
        let reader = FilteredAstReader::new(&input[..], Instant::now() + Duration::from_secs(1));
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        let (nodes, _) =
            parse_filtered_reader(reader, temp.path(), temp.path(), &mut items, &mut coverage)
                .unwrap();
        assert_eq!(nodes, 3);
    }

    #[test]
    fn filtered_stream_accepts_empty_tu_output() {
        let temp = tempfile::tempdir().unwrap();
        let reader = FilteredAstReader::new(&b""[..], Instant::now() + Duration::from_secs(1));
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        let (nodes, _) =
            parse_filtered_reader(reader, temp.path(), temp.path(), &mut items, &mut coverage)
                .unwrap();
        assert_eq!(nodes, 1);
    }

    #[test]
    fn child_file_location_cannot_pollute_its_parent_scope_owner() {
        let temp = tempfile::tempdir().unwrap();
        let owner = temp.path().join("src/emel/model/actor.hpp");
        fs::create_dir_all(owner.parent().unwrap()).unwrap();
        fs::write(&owner, "struct actor { actor(); };\n").unwrap();
        let external = temp.path().join("external.hpp");
        let ast = format!(
            r#"{{"kind":"TranslationUnitDecl","inner":[{{
                "kind":"CXXRecordDecl",
                "loc":{{"file":{},"line":1,"col":8,"offset":7,"tokLen":5}},
                "range":{{
                    "begin":{{"file":{},"line":1,"col":1,"offset":0,"tokLen":6}},
                    "end":{{"line":1,"col":25,"offset":24,"tokLen":1}}
                }},
                "name":"actor",
                "inner":[
                    {{
                        "kind":"NamespaceDecl",
                        "loc":{{"file":{},"line":1,"col":1,"offset":0,"tokLen":1}},
                        "range":{{
                            "begin":{{"file":{},"line":1,"col":1,"offset":0,"tokLen":1}},
                            "end":{{"line":1,"col":1,"offset":0,"tokLen":1}}
                        }},
                        "name":"external"
                    }},
                    {{
                        "kind":"CXXConstructorDecl",
                        "loc":{{"line":1,"col":16,"offset":15,"tokLen":5}},
                        "range":{{
                            "begin":{{"line":1,"col":16,"offset":15,"tokLen":5}},
                            "end":{{"line":1,"col":22,"offset":21,"tokLen":1}}
                        }},
                        "name":"actor",
                        "type":{{"qualType":"void ()"}}
                    }}
                ]
            }}]}}"#,
            serde_json::to_string(&owner.to_string_lossy()).unwrap(),
            serde_json::to_string(&owner.to_string_lossy()).unwrap(),
            serde_json::to_string(&external.to_string_lossy()).unwrap(),
            serde_json::to_string(&external.to_string_lossy()).unwrap(),
        );
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        parse_reader(
            ast.as_bytes(),
            temp.path(),
            temp.path(),
            &mut items,
            &mut coverage,
        )
        .unwrap();
        let constructor = coverage
            .iter()
            .find(|row| row.kind == "CXXConstructorDecl")
            .expect("constructor remains bound to its record owner file");
        assert_eq!(constructor.path, "src/emel/model/actor.hpp");
        assert_eq!(constructor.name, "actor::actor(void ())");
    }

    #[test]
    fn json_redeclarations_keep_forward_member_and_canonical_definition() {
        let temp = tempfile::tempdir().unwrap();
        let owner = temp.path().join("src/emel/model/events.hpp");
        fs::create_dir_all(owner.parent().unwrap()).unwrap();
        fs::write(&owner, "struct bind_storage;\nstruct bind_storage {};\n").unwrap();
        let ast = format!(
            r#"{{"kind":"TranslationUnitDecl","inner":[
                {{
                    "id":"0x1",
                    "kind":"CXXRecordDecl",
                    "loc":{{"file":{},"line":1,"col":8,"offset":7,"tokLen":12}},
                    "range":{{
                        "begin":{{"file":{},"line":1,"col":1,"offset":0,"tokLen":6}},
                        "end":{{"line":1,"col":8,"offset":7,"tokLen":12}}
                    }},
                    "name":"bind_storage"
                }},
                {{
                    "id":"0x2",
                    "kind":"CXXRecordDecl",
                    "loc":{{"line":2,"col":8,"offset":28,"tokLen":12}},
                    "range":{{
                        "begin":{{"line":2,"col":1,"offset":21,"tokLen":6}},
                        "end":{{"line":2,"col":22,"offset":42,"tokLen":1}}
                    }},
                    "previousDecl":"0x1",
                    "name":"bind_storage",
                    "completeDefinition":true
                }}
            ]}}"#,
            serde_json::to_string(&owner.to_string_lossy()).unwrap(),
            serde_json::to_string(&owner.to_string_lossy()).unwrap(),
        );
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        parse_reader(
            ast.as_bytes(),
            temp.path(),
            temp.path(),
            &mut items,
            &mut coverage,
        )
        .unwrap();
        let names = coverage
            .iter()
            .filter(|row| row.kind == "CXXRecordDecl")
            .map(|row| row.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"bind_storage::<declaration@1:8>"));
        assert!(names.contains(&"bind_storage"));
        assert_eq!(
            items
                .iter()
                .filter(|item| item.kind == "type" && item.name.starts_with("bind_storage"))
                .count(),
            2
        );
    }

    #[test]
    fn filtered_stream_rejects_truncation() {
        let temp = tempfile::tempdir().unwrap();
        let reader = FilteredAstReader::new(
            &b"{\"kind\":\"NamespaceDecl\""[..],
            Instant::now() + Duration::from_secs(1),
        );
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        assert!(
            parse_filtered_reader(reader, temp.path(), temp.path(), &mut items, &mut coverage,)
                .is_err()
        );
    }

    #[test]
    fn filtered_stream_enforces_expired_deadline() {
        let mut reader = FilteredAstReader::new(
            &b"{}"[..],
            Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        );
        let mut byte = [0_u8; 128];
        let first = reader.read(&mut byte).unwrap();
        assert!(first > 0);
        assert_eq!(
            reader.read(&mut byte).unwrap_err().kind(),
            std::io::ErrorKind::TimedOut
        );
    }

    #[test]
    #[ignore = "subprocess helper for the watchdog regression"]
    fn watchdog_stall_helper() {
        std::thread::sleep(Duration::from_secs(10));
    }

    #[test]
    fn watchdog_kills_stall_unblocks_read_reaps_and_joins() {
        let executable = std::env::current_exe().unwrap();
        let mut child = Command::new(executable)
            .args([
                "--ignored",
                "--exact",
                "ast_stream::tests::watchdog_stall_helper",
                "--nocapture",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut stdout = child.stdout.take().unwrap();
        let child = Arc::new(Mutex::new(child));
        let started = Instant::now();
        let watchdog_thread = spawn_watchdog(
            Arc::clone(&child),
            Instant::now() + Duration::from_millis(100),
        );
        let mut output = Vec::new();
        stdout.read_to_end(&mut output).unwrap();
        let watch_result = watchdog_thread.join().unwrap().unwrap();
        assert!(watch_result.timed_out);
        assert!(!watch_result.status.success());
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(child.lock().unwrap().try_wait().unwrap().is_some());
    }

    #[test]
    fn overload_identity_includes_canonical_signature() {
        let first = Node {
            kind: "FunctionDecl".into(),
            name: "load".into(),
            ast_type: Some(AstType {
                qual_type: Some("void (int)".into()),
            }),
            ..Node::default()
        };
        let second = Node {
            ast_type: Some(AstType {
                qual_type: Some("void (float)".into()),
            }),
            ..first.clone()
        };
        assert_eq!(declaration_name(&first), "load(void (int))");
        assert_ne!(declaration_name(&first), declaration_name(&second));
    }

    #[test]
    fn callable_identity_keeps_structural_name_and_direct_signature_exactly_once() {
        for (kind, name, signature, expected) in [
            (
                "CXXConstructorDecl",
                "window_fixture",
                Some("void ()"),
                "window_fixture(void ())",
            ),
            (
                "CXXConstructorDecl",
                "window_fixture",
                Some("void (const window_fixture &)"),
                "window_fixture(void (const window_fixture &))",
            ),
            (
                "CXXDestructorDecl",
                "~window_fixture",
                Some("void () noexcept"),
                "~window_fixture(void () noexcept)",
            ),
            (
                "CXXMethodDecl",
                "operator=",
                Some("window_fixture &(const window_fixture &)"),
                "operator=(window_fixture &(const window_fixture &))",
            ),
            ("FunctionDecl", "run", Some("bool (int)"), "run(bool (int))"),
            (
                "FunctionTemplateDecl",
                "visit",
                None,
                "visit(FunctionTemplateDecl)",
            ),
        ] {
            assert_eq!(
                canonical_callable_identity(kind, name, signature).as_deref(),
                Some(expected)
            );
        }
        assert_eq!(
            canonical_callable_identity("FieldDecl", "run", Some("bool (int)")),
            None
        );
    }

    #[test]
    fn callable_identity_collisions_remain_separated_by_direct_signature() {
        let default =
            canonical_callable_identity("CXXConstructorDecl", "window_fixture", Some("void ()"))
                .unwrap();
        let copy = canonical_callable_identity(
            "CXXConstructorDecl",
            "window_fixture",
            Some("void (const window_fixture &)"),
        )
        .unwrap();
        let ordinary =
            canonical_callable_identity("FunctionDecl", "window_fixture", Some("void ()")).unwrap();
        assert_ne!(default, copy);
        assert_eq!(default, ordinary);
        assert!(default.starts_with("window_fixture("));
        assert!(!default.starts_with("void"));
    }

    #[test]
    fn redeclaration_identity_keeps_definition_canonical_and_declarations_exact() {
        let first = canonical_redeclaration_identity(
            "CXXRecordDecl",
            "emel::portable::event::bind_storage",
            false,
            58,
            8,
        );
        let repeated = canonical_redeclaration_identity(
            "CXXRecordDecl",
            "emel::portable::event::bind_storage",
            false,
            58,
            8,
        );
        let second = canonical_redeclaration_identity(
            "CXXRecordDecl",
            "emel::portable::event::bind_storage",
            false,
            95,
            8,
        );
        let definition = canonical_redeclaration_identity(
            "CXXRecordDecl",
            "emel::portable::event::bind_storage",
            true,
            138,
            8,
        );
        assert_eq!(first, repeated);
        assert_ne!(first, second);
        assert_eq!(definition, "emel::portable::event::bind_storage");
        assert_eq!(
            first,
            "emel::portable::event::bind_storage::<declaration@58:8>"
        );
    }

    #[test]
    fn owner_path_normalization_is_lexical_closed_and_portable() {
        assert_eq!(
            normalize_relative_owner_path(Path::new(
                "tests/model/loader/../../kernel/test_helpers.hpp"
            )),
            Some("tests/kernel/test_helpers.hpp".into())
        );
        assert_eq!(
            normalize_relative_owner_path(Path::new("tests/model/./loader/events.hpp")),
            Some("tests/model/loader/events.hpp".into())
        );
        assert!(normalize_relative_owner_path(Path::new("../tests/model/events.hpp")).is_none());
        assert!(normalize_relative_owner_path(Path::new("/tests/model/events.hpp")).is_none());
        assert!(normalize_relative_owner_path(Path::new("tests\\model/events.hpp")).is_none());
        assert!(normalize_relative_owner_path(Path::new(".")).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn owner_path_normalization_rejects_non_utf8_components() {
        use std::os::unix::ffi::OsStrExt as _;

        let path = Path::new(std::ffi::OsStr::from_bytes(b"tests/model/\xff.hpp"));
        assert!(normalize_relative_owner_path(path).is_none());
    }

    #[test]
    fn source_root_canonicalization_is_boundary_exact_and_preserves_other_bytes() {
        let materialized = Path::new("/tmp/run-a/source");
        let original = Path::new("/repo/emel.cpp");
        let input = "(lambda at /tmp/run-a/source/src/a.cpp:7:3); \
                     /repo/emel.cpp/tests/a.cpp; /tmp/run-a/source/src/b.cpp";
        assert_eq!(
            canonicalize_source_roots(input, materialized, original),
            "(lambda at <source-root>/src/a.cpp:7:3); \
             <source-root>/tests/a.cpp; <source-root>/src/b.cpp"
        );
        for near_prefix in [
            "prefix/tmp/run-a/source/src/a.cpp",
            "/tmp/run-a/source-near/src/a.cpp",
            "/repo/emel.cpp-near/tests/a.cpp",
        ] {
            assert_eq!(
                canonicalize_source_roots(near_prefix, materialized, original),
                near_prefix
            );
        }
    }

    #[test]
    fn source_root_canonicalization_stabilizes_json_and_text_lambda_signatures() {
        let first_root = Path::new("/tmp/first/source");
        let second_root = Path::new("/tmp/second/source");
        let original = Path::new("/repo/emel.cpp");
        let first = canonicalize_source_roots(
            "bool ((lambda at /tmp/first/source/src/emel/model/detail.cpp:172:11) &&)",
            first_root,
            original,
        );
        let second = canonicalize_source_roots(
            "bool ((lambda at /tmp/second/source/src/emel/model/detail.cpp:172:11) &&)",
            second_root,
            original,
        );
        assert_eq!(first, second);
        assert_eq!(
            first,
            "bool ((lambda at <source-root>/src/emel/model/detail.cpp:172:11) &&)"
        );
    }

    #[test]
    fn untyped_function_template_and_typed_child_have_distinct_exact_identities() {
        let template = Node {
            kind: "FunctionTemplateDecl".into(),
            name: "unwrap_runtime_event".into(),
            ..Node::default()
        };
        let child = Node {
            kind: "FunctionDecl".into(),
            name: "unwrap_runtime_event".into(),
            ast_type: Some(AstType {
                qual_type: Some("decltype(auto) (const runtime_event_type &) noexcept".into()),
            }),
            ..Node::default()
        };
        let template_identity =
            canonical_declaration_identity(&template.kind, &template.name, None, 41, 1);
        let overload_identity =
            canonical_declaration_identity(&template.kind, &template.name, None, 58, 1);
        assert_eq!(template_identity, "unwrap_runtime_event::<template@41:1>");
        assert_ne!(template_identity, overload_identity);
        assert_eq!(
            declaration_signature(&template).unwrap(),
            "FunctionTemplateDecl"
        );
        assert_eq!(
            declaration_name(&child),
            "unwrap_runtime_event(decltype(auto) (const runtime_event_type &) noexcept)"
        );
        let child_identity = declaration_name(&child);
        assert_ne!(template_identity, child_identity);
        let inventory_id = |name: String, payload: &str| {
            crate::scan::row(
                "cpp",
                &Extracted {
                    path: "src/emel/model/loader/actions.hpp".into(),
                    kind: "algorithm".into(),
                    name,
                    payload: payload.into(),
                    note: "test".into(),
                },
                "commit=source;model=model;tests=tests",
                "rust-tree",
            )
            .item_id
        };
        assert_ne!(
            inventory_id(template_identity, "FunctionTemplateDecl"),
            inventory_id(
                child_identity,
                "decltype(auto) (const runtime_event_type &) noexcept"
            )
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn json_template_parents_and_typed_children_all_keep_exact_identities() {
        let temp = tempfile::tempdir().unwrap();
        let owner = temp.path().join("src/emel/model/templates.hpp");
        fs::create_dir_all(owner.parent().unwrap()).unwrap();
        let source = "template <class T> int load(T); template <class T> int load(T*);";
        fs::write(&owner, source).unwrap();
        let first_template = 0_usize;
        let first_int = source.find("int").unwrap();
        let first_load = source.find("load").unwrap();
        let first_end = source.find(';').unwrap();
        let second_template = first_end + 2;
        let second_int = source[second_template..].find("int").unwrap() + second_template;
        let second_load = source[second_template..].find("load").unwrap() + second_template;
        let second_end = source[second_template..].find(';').unwrap() + second_template;
        let ast = format!(
            r#"{{"kind":"TranslationUnitDecl","inner":[
                {{
                    "kind":"FunctionTemplateDecl",
                    "loc":{{"file":{},"line":1,"col":{},"offset":{},"tokLen":4}},
                    "range":{{
                        "begin":{{"file":{},"line":1,"col":1,"offset":{},"tokLen":8}},
                        "end":{{"line":1,"col":{},"offset":{},"tokLen":1}}
                    }},
                    "name":"load",
                    "inner":[{{
                        "kind":"FunctionDecl",
                        "loc":{{"line":1,"col":{},"offset":{},"tokLen":4}},
                        "range":{{
                            "begin":{{"line":1,"col":{},"offset":{},"tokLen":3}},
                            "end":{{"line":1,"col":{},"offset":{},"tokLen":1}}
                        }},
                        "name":"load",
                        "type":{{"qualType":"int (T)"}}
                    }}]
                }},
                {{
                    "kind":"FunctionTemplateDecl",
                    "loc":{{"file":{},"line":1,"col":{},"offset":{},"tokLen":4}},
                    "range":{{
                        "begin":{{"file":{},"line":1,"col":{},"offset":{},"tokLen":8}},
                        "end":{{"line":1,"col":{},"offset":{},"tokLen":1}}
                    }},
                    "name":"load",
                    "inner":[{{
                        "kind":"FunctionDecl",
                        "loc":{{"line":1,"col":{},"offset":{},"tokLen":4}},
                        "range":{{
                            "begin":{{"line":1,"col":{},"offset":{},"tokLen":3}},
                            "end":{{"line":1,"col":{},"offset":{},"tokLen":1}}
                        }},
                        "name":"load",
                        "type":{{"qualType":"int (T *)"}}
                    }}]
                }}
            ]}}"#,
            serde_json::to_string(&owner.to_string_lossy()).unwrap(),
            first_load + 1,
            first_load,
            serde_json::to_string(&owner.to_string_lossy()).unwrap(),
            first_template,
            first_end + 1,
            first_end,
            first_load + 1,
            first_load,
            first_int + 1,
            first_int,
            first_end,
            first_end - 1,
            serde_json::to_string(&owner.to_string_lossy()).unwrap(),
            second_load + 1,
            second_load,
            serde_json::to_string(&owner.to_string_lossy()).unwrap(),
            second_template + 1,
            second_template,
            second_end + 1,
            second_end,
            second_load + 1,
            second_load,
            second_int + 1,
            second_int,
            second_end,
            second_end - 1,
        );
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        parse_reader(
            ast.as_bytes(),
            temp.path(),
            temp.path(),
            &mut items,
            &mut coverage,
        )
        .unwrap();
        let names = coverage
            .iter()
            .map(|row| row.name.as_str())
            .collect::<Vec<_>>();
        for expected in [
            "load::<template@1:24>",
            "load::<template@1:56>",
            "load(int (T))",
            "load(int (T *))",
        ] {
            assert!(
                names.contains(&expected),
                "missing {expected:?}; got {names:?}"
            );
        }
        assert_eq!(names.len(), 4);
    }

    fn streamed_enum_signature(
        fixed_underlying_type: Option<serde_json::Value>,
    ) -> Result<String, String> {
        let temp = tempfile::tempdir().unwrap();
        let owner = temp.path().join("src/emel/model/data.hpp");
        fs::create_dir_all(owner.parent().unwrap()).unwrap();
        let source = "enum class route : uint8_t { value };\n";
        fs::write(&owner, source).unwrap();
        let owner = owner.to_string_lossy().into_owned();
        let mut node = serde_json::json!({
            "id": "0x1",
            "kind": "EnumDecl",
            "loc": {
                "file": owner,
                "line": 1,
                "col": 12,
                "offset": 11,
                "tokLen": 5
            },
            "range": {
                "begin": {
                    "file": owner,
                    "line": 1,
                    "col": 1,
                    "offset": 0,
                    "tokLen": 4
                },
                "end": {
                    "line": 1,
                    "col": source.len(),
                    "offset": source.len() - 2,
                    "tokLen": 1
                }
            },
            "name": "route"
        });
        if let Some(fixed_underlying_type) = fixed_underlying_type {
            node["fixedUnderlyingType"] = fixed_underlying_type;
        }
        let encoded = serde_json::to_vec(&node).unwrap();
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        parse_reader(
            &encoded[..],
            temp.path(),
            temp.path(),
            &mut items,
            &mut coverage,
        )?;
        Ok(coverage[0].signature.clone())
    }

    #[test]
    fn streamed_enum_signature_preserves_fixed_typedef_and_builtin_spellings() {
        let typedef = streamed_enum_signature(Some(serde_json::json!({
            "qualType": "uint8_t",
            "desugaredQualType": "unsigned char"
        })))
        .unwrap();
        let builtin = streamed_enum_signature(Some(serde_json::json!({
            "qualType": "unsigned int"
        })))
        .unwrap();
        assert_eq!(typedef, "uint8_t");
        assert_eq!(builtin, "unsigned int");
    }

    #[test]
    fn streamed_unfixed_enum_uses_decl_kind_signature() {
        assert_eq!(streamed_enum_signature(None).unwrap(), "EnumDecl");
    }

    #[test]
    fn streamed_fixed_enum_rejects_missing_or_malformed_qual_type() {
        let missing = streamed_enum_signature(Some(serde_json::json!({}))).unwrap_err();
        assert!(missing.contains("fixedUnderlyingType lacks a string qualType"));

        let malformed = streamed_enum_signature(Some(serde_json::json!("uint8_t"))).unwrap_err();
        assert!(malformed.contains("malformed or partial Clang AST JSON"));
    }

    #[test]
    fn enum_is_semantic_parent_for_enumerators() {
        let temp = tempfile::tempdir().unwrap();
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        let mut state = State {
            materialized: temp.path(),
            original: temp.path(),
            current_file: None,
            parents: vec!["emel".into(), "model".into()],
            scope_kinds: vec![ScopeKind::Namespace(None), ScopeKind::Namespace(None)],
            scope_files: vec![None, None],
            file_cache: BTreeMap::new(),
            items: &mut items,
            coverage: &mut coverage,
            nodes_seen: 0,
            depth: 0,
            max_depth: 0,
        };
        let node = Node {
            kind: "EnumDecl".into(),
            name: "route".into(),
            ..Node::default()
        };
        assert!(state.push_parent(&node).unwrap());
        assert_eq!(state.parents.join("::"), "emel::model::route");
    }

    #[test]
    fn anonymous_namespace_parent_is_exact_distinct_and_named_namespace_is_unchanged() {
        let temp = tempfile::tempdir().unwrap();
        let owner = temp.path().join("src/emel/model/architecture/detail.cpp");
        fs::create_dir_all(owner.parent().unwrap()).unwrap();
        fs::write(&owner, "namespace emel {}\nnamespace {}\nnamespace {}\n").unwrap();
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        let mut state = State {
            materialized: temp.path(),
            original: temp.path(),
            current_file: Some(owner.to_string_lossy().into_owned()),
            parents: Vec::new(),
            scope_kinds: Vec::new(),
            scope_files: Vec::new(),
            file_cache: BTreeMap::new(),
            items: &mut items,
            coverage: &mut coverage,
            nodes_seen: 0,
            depth: 0,
            max_depth: 0,
        };
        assert!(
            state
                .push_parent(&Node {
                    kind: "NamespaceDecl".into(),
                    name: "emel".into(),
                    ..Node::default()
                })
                .unwrap()
        );
        assert_eq!(state.parents, ["emel"]);
        state.parents.clear();
        state.scope_kinds.clear();
        state.scope_files.clear();

        let anonymous = |line| Node {
            kind: "NamespaceDecl".into(),
            location: Some(Location {
                line: Some(line),
                col: Some(11),
                ..Location::default()
            }),
            ..Node::default()
        };
        assert!(state.push_parent(&anonymous(14)).unwrap());
        let first = state.parents.pop().unwrap();
        state.scope_kinds.pop();
        state.scope_files.pop();
        assert!(state.push_parent(&anonymous(80)).unwrap());
        let second = state.parents.pop().unwrap();
        state.scope_kinds.pop();
        state.scope_files.pop();
        assert_eq!(first, "<anonymous-NamespaceDecl@14:11>");
        assert_eq!(second, "<anonymous-NamespaceDecl@80:11>");
        assert_ne!(first, second);
        assert!(
            state
                .push_parent(&Node {
                    kind: "NamespaceDecl".into(),
                    ..Node::default()
                })
                .is_err()
        );
    }

    #[test]
    fn lexical_scopes_get_exact_distinct_parents() {
        let temp = tempfile::tempdir().unwrap();
        let owner = temp.path().join("src/emel/model/detail.cpp");
        fs::create_dir_all(owner.parent().unwrap()).unwrap();
        fs::write(&owner, "{}\n{}\n").unwrap();
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        let mut state = State {
            materialized: temp.path(),
            original: temp.path(),
            current_file: Some(owner.to_string_lossy().into_owned()),
            parents: vec!["emel".into()],
            scope_kinds: vec![ScopeKind::Namespace(None)],
            scope_files: vec![None],
            file_cache: BTreeMap::new(),
            items: &mut items,
            coverage: &mut coverage,
            nodes_seen: 0,
            depth: 0,
            max_depth: 0,
        };
        let scope = |line| Node {
            kind: "CompoundStmt".into(),
            range: Some(SourceRange {
                begin: Some(Location {
                    line: Some(line),
                    col: Some(1),
                    ..Location::default()
                }),
                end: None,
            }),
            ..Node::default()
        };
        assert!(state.push_parent(&scope(1)).unwrap());
        let first = state.parents.pop().unwrap();
        state.scope_kinds.pop();
        state.scope_files.pop();
        assert!(state.push_parent(&scope(2)).unwrap());
        let second = state.parents.pop().unwrap();
        state.scope_kinds.pop();
        state.scope_files.pop();
        assert_ne!(first, second);
    }

    #[test]
    fn same_spelling_record_and_variable_have_distinct_closed_kinds() {
        assert_eq!(
            classify_decl("CXXRecordDecl", "begin_bind", false, true, None).unwrap(),
            "type"
        );
        assert_eq!(
            classify_decl("VarDecl", "begin_bind", false, true, None).unwrap(),
            "public-api"
        );
    }

    #[test]
    fn classification_is_kind_first_and_callable_owner_aware() {
        assert_eq!(
            classify_decl("FunctionDecl", "publish_error", false, true, Some("action")).unwrap(),
            "action"
        );
        assert_eq!(
            classify_decl("FunctionDecl", "event_context", false, true, None).unwrap(),
            "algorithm"
        );
        assert_eq!(
            classify_decl("CXXRecordDecl", "LoadError", false, true, None).unwrap(),
            "error"
        );
        assert_eq!(
            classify_decl("CXXRecordDecl", "publish_error", false, true, Some("guard")).unwrap(),
            "guard"
        );
        assert_eq!(
            classify_decl("CXXMethodDecl", "operator()", false, true, Some("guard")).unwrap(),
            "guard"
        );
    }

    #[test]
    fn cpp_spelling_body_changes_hash_but_trivia_does_not() {
        fn hash(source: &str) -> String {
            let temp = tempfile::tempdir().unwrap();
            let owner = temp.path().join("src/emel/model/detail.cpp");
            fs::create_dir_all(owner.parent().unwrap()).unwrap();
            fs::write(&owner, source).unwrap();
            let file = owner.to_string_lossy().into_owned();
            let end = u64::try_from(source.len() - 1).unwrap();
            let location = Location {
                file: Some(file),
                line: Some(1),
                col: Some(1),
                offset: Some(0),
                tok_len: Some(1),
                spelling_loc: None,
            };
            let node = Node {
                kind: "FunctionDecl".into(),
                location: Some(location.clone()),
                range: Some(SourceRange {
                    begin: Some(location),
                    end: Some(Location {
                        offset: Some(end),
                        tok_len: Some(1),
                        ..Location::default()
                    }),
                }),
                name: "compute".into(),
                ast_type: Some(AstType {
                    qual_type: Some("int ()".into()),
                }),
                ..Node::default()
            };
            let mut items = Vec::new();
            let mut coverage = Vec::new();
            let mut state = State {
                materialized: temp.path(),
                original: temp.path(),
                current_file: None,
                parents: vec![],
                scope_kinds: vec![],
                scope_files: vec![],
                file_cache: BTreeMap::new(),
                items: &mut items,
                coverage: &mut coverage,
                nodes_seen: 0,
                depth: 0,
                max_depth: 0,
            };
            state.process(&node).unwrap();
            crate::hash::semantic_hash(&[&items[0].payload])
        }
        let first = hash("int compute(){return 1;}");
        let trivia = hash("int compute() { /* comment */ return 1 ; }");
        let changed = hash("int compute(){return 2;}");
        assert_eq!(first, trivia);
        assert_ne!(first, changed);
    }

    #[test]
    fn cpp_behavior_range_fails_closed_when_out_of_file() {
        let temp = tempfile::tempdir().unwrap();
        let owner = temp.path().join("src/emel/model/detail.cpp");
        fs::create_dir_all(owner.parent().unwrap()).unwrap();
        fs::write(&owner, "int f(){}\n").unwrap();
        let file = owner.to_string_lossy().into_owned();
        let node = Node {
            kind: "FunctionDecl".into(),
            location: Some(Location {
                file: Some(file),
                offset: Some(0),
                ..Location::default()
            }),
            range: Some(SourceRange {
                begin: Some(Location {
                    offset: Some(0),
                    ..Location::default()
                }),
                end: Some(Location {
                    offset: Some(1_000),
                    tok_len: Some(1),
                    ..Location::default()
                }),
            }),
            name: "f".into(),
            ..Node::default()
        };
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        let mut state = State {
            materialized: temp.path(),
            original: temp.path(),
            current_file: None,
            parents: vec![],
            scope_kinds: vec![],
            scope_files: vec![],
            file_cache: BTreeMap::new(),
            items: &mut items,
            coverage: &mut coverage,
            nodes_seen: 0,
            depth: 0,
            max_depth: 0,
        };
        assert!(state.process(&node).is_err());
    }

    #[test]
    fn namespace_variable_local_and_parameter_use_ast_scope_and_linkage() {
        assert_eq!(
            classify_decl("VarDecl", "actor", false, true, None).unwrap(),
            "public-api"
        );
        assert_eq!(
            classify_decl("VarDecl", "entry", true, true, None).unwrap(),
            "algorithm"
        );
        assert_eq!(
            classify_decl("ParmVarDecl", "event", true, true, None).unwrap(),
            "algorithm"
        );
        assert_eq!(
            classify_decl("VarDecl", "internal", false, false, None).unwrap(),
            "algorithm"
        );
    }

    #[test]
    fn cpp_parameters_and_local_declarations_are_covered_only_by_function_body() {
        let temp = tempfile::tempdir().unwrap();
        let owner = temp.path().join("src/emel/model/detail.cpp");
        fs::create_dir_all(owner.parent().unwrap()).unwrap();
        fs::write(
            &owner,
            "int f(int parameter){int local=parameter;return local;}\n",
        )
        .unwrap();
        let file = owner.to_string_lossy().into_owned();
        let mut items = Vec::new();
        let mut coverage = Vec::new();
        let mut state = State {
            materialized: temp.path(),
            original: temp.path(),
            current_file: Some(file.clone()),
            parents: vec!["f(int (int))".into()],
            scope_kinds: vec![ScopeKind::Function],
            scope_files: vec![None],
            file_cache: BTreeMap::new(),
            items: &mut items,
            coverage: &mut coverage,
            nodes_seen: 0,
            depth: 0,
            max_depth: 0,
        };
        for (kind, name) in [("ParmVarDecl", "parameter"), ("VarDecl", "local")] {
            state
                .process(&Node {
                    kind: kind.into(),
                    name: name.into(),
                    location: Some(Location {
                        file: Some(file.clone()),
                        line: Some(1),
                        col: Some(1),
                        offset: Some(0),
                        ..Location::default()
                    }),
                    ..Node::default()
                })
                .unwrap();
        }
        assert!(items.is_empty());
        assert!(coverage.is_empty());
    }

    #[test]
    fn cpp_closed_declaration_boundary_includes_owners_and_excludes_wrappers() {
        for included in [
            "NamespaceDecl",
            "NamespaceAliasDecl",
            "CXXRecordDecl",
            "EnumDecl",
            "FunctionDecl",
            "VarDecl",
            "FieldDecl",
            "EnumConstantDecl",
            "UsingDecl",
            "TypeAliasDecl",
            "StaticAssertDecl",
            "ConceptDecl",
        ] {
            assert!(!excluded_declaration_kind(included), "excluded {included}");
        }
        for excluded in [
            "ParmVarDecl",
            "BindingDecl",
            "TemplateTypeParmDecl",
            "NonTypeTemplateParmDecl",
            "TemplateTemplateParmDecl",
            "AccessSpecDecl",
            "EmptyDecl",
            "UsingShadowDecl",
            "LinkageSpecDecl",
            "FriendDecl",
        ] {
            assert!(excluded_declaration_kind(excluded), "included {excluded}");
        }
    }

    #[test]
    fn fields_inherit_only_semantic_actor_owners() {
        assert_eq!(
            classify_decl("FieldDecl", "status", false, true, Some("context")).unwrap(),
            "context"
        );
        assert_eq!(
            classify_decl("FieldDecl", "weights", false, true, Some("type")).unwrap(),
            "type"
        );
        assert_eq!(
            classify_decl("FieldDecl", "token", false, true, Some("event")).unwrap(),
            "event"
        );
    }
}
#[cfg(test)]
#[test]
fn textual_capture_accepts_exact_limit_and_rejects_first_oversize_byte() {
    let mut exact = std::io::Cursor::new(vec![b'x'; 32]);
    let mut exact_output = Vec::new();
    assert_eq!(copy_bounded(&mut exact, &mut exact_output, 32), Ok(32));
    assert_eq!(exact_output.len(), 32);

    let mut oversized = std::io::Cursor::new(vec![b'x'; 33]);
    let mut oversized_output = Vec::new();
    assert_eq!(
        copy_bounded(&mut oversized, &mut oversized_output, 32),
        Err("textual Clang AST exceeded deterministic 32-byte capture limit".into())
    );
    assert!(oversized_output.is_empty());
}
