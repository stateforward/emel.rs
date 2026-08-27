use std::collections::BTreeSet;

use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens as _;
use syn::spanned::Spanned as _;
use syn::visit::Visit as _;
use syn::{Attribute, Item, Visibility};

use crate::git::Blob;
use crate::hash::semantic_hash;
use crate::schema::{InventoryRow, VERSION};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Extracted {
    pub path: String,
    pub kind: String,
    pub name: String,
    pub payload: String,
    pub note: String,
}

pub fn rust(blobs: &[Blob]) -> Result<Vec<Extracted>, String> {
    let mut output = Vec::new();
    for blob in blobs {
        let source = std::str::from_utf8(&blob.bytes)
            .map_err(|error| format!("{} is not UTF-8: {error}", blob.path))?;
        if extension_is(&blob.path, "rs") {
            let file = syn::parse_file(source)
                .map_err(|error| format!("syn rejected {}: {error}", blob.path))?;
            rust_items(&blob.path, &[], &file.items, &mut output)?;
            RustInvocationVisitor {
                path: &blob.path,
                output: &mut output,
            }
            .visit_file(&file);
            scanners(&blob.path, source, "rust", &mut output)?;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_lines)]
fn rust_items(
    path: &str,
    parents: &[String],
    items: &[Item],
    output: &mut Vec<Extracted>,
) -> Result<(), String> {
    for item in items {
        match item {
            Item::Const(item) => emit_rust_item(
                path,
                parents,
                &item.ident.to_string(),
                "type",
                &item.vis,
                &item.attrs,
                &item.to_token_stream(),
                output,
            ),
            Item::Enum(item) => {
                let name = item.ident.to_string();
                emit_rust_item(
                    path,
                    parents,
                    &name,
                    classify_name(&name, "type"),
                    &item.vis,
                    &item.attrs,
                    &item.to_token_stream(),
                    output,
                );
                let mut owner = parents.to_vec();
                owner.push(name);
                for variant in &item.variants {
                    emit_rust_plain(
                        path,
                        &owner,
                        &variant.ident.to_string(),
                        classify_name(&variant.ident.to_string(), "type"),
                        &variant.to_token_stream(),
                        "syn-enum-variant",
                        output,
                    );
                    let mut variant_owner = owner.clone();
                    variant_owner.push(variant.ident.to_string());
                    rust_fields(path, &variant_owner, &variant.fields, output);
                }
            }
            Item::ExternCrate(item) => emit_rust_item(
                path,
                parents,
                &item.ident.to_string(),
                "public-api",
                &item.vis,
                &item.attrs,
                &item.to_token_stream(),
                output,
            ),
            Item::Fn(item) => {
                let name = item.sig.ident.to_string();
                emit_rust_item(
                    path,
                    parents,
                    &name,
                    classify_function(&name),
                    &item.vis,
                    &item.attrs,
                    &item.to_token_stream(),
                    output,
                );
            }
            Item::ForeignMod(item) => {
                let start = item.abi.extern_token.span.start();
                let owner_name = format!("extern@{}:{}", start.line, start.column + 1);
                emit_rust_plain(
                    path,
                    parents,
                    &owner_name,
                    "public-api",
                    &item.to_token_stream(),
                    "syn-foreign-mod",
                    output,
                );
                let mut owner = parents.to_vec();
                owner.push(owner_name);
                rust_foreign_items(path, &owner, &item.items, output)?;
            }
            Item::Impl(item) => rust_impl(path, parents, item, output)?,
            Item::Macro(item) => {
                if item.mac.path.is_ident("sml") {
                    sml_transition_rows(path, parents, &item.mac.tokens, output)?;
                } else {
                    let start = item.mac.path.span().start();
                    let name = format!(
                        "macro-{}@{}:{}",
                        display_tokens(&item.mac.path.to_token_stream().to_string()),
                        start.line,
                        start.column + 1
                    );
                    emit_rust_plain(
                        path,
                        parents,
                        &name,
                        "algorithm",
                        &item.to_token_stream(),
                        "syn-macro-item",
                        output,
                    );
                }
            }
            Item::Mod(item) => {
                let name = item.ident.to_string();
                if let Some((_, nested)) = &item.content {
                    let mut next = parents.to_vec();
                    next.push(name.clone());
                    rust_items(path, &next, nested, output)?;
                }
                emit_rust_item(
                    path,
                    parents,
                    &name,
                    "type",
                    &item.vis,
                    &item.attrs,
                    &item.to_token_stream(),
                    output,
                );
            }
            Item::Static(item) => emit_rust_item(
                path,
                parents,
                &item.ident.to_string(),
                "type",
                &item.vis,
                &item.attrs,
                &item.to_token_stream(),
                output,
            ),
            Item::Struct(item) => {
                let name = item.ident.to_string();
                emit_rust_item(
                    path,
                    parents,
                    &name,
                    classify_name(&name, "type"),
                    &item.vis,
                    &item.attrs,
                    &item.to_token_stream(),
                    output,
                );
                let mut owner = parents.to_vec();
                owner.push(name);
                rust_fields(path, &owner, &item.fields, output);
            }
            Item::Trait(item) => {
                let name = item.ident.to_string();
                emit_rust_item(
                    path,
                    parents,
                    &name,
                    "public-api",
                    &item.vis,
                    &item.attrs,
                    &item.to_token_stream(),
                    output,
                );
                let mut owner = parents.to_vec();
                owner.push(name);
                rust_trait_items(path, &owner, &item.items, output)?;
            }
            Item::TraitAlias(item) => emit_rust_item(
                path,
                parents,
                &item.ident.to_string(),
                "public-api",
                &item.vis,
                &item.attrs,
                &item.to_token_stream(),
                output,
            ),
            Item::Type(item) => emit_rust_item(
                path,
                parents,
                &item.ident.to_string(),
                classify_name(&item.ident.to_string(), "type"),
                &item.vis,
                &item.attrs,
                &item.to_token_stream(),
                output,
            ),
            Item::Union(item) => {
                let name = item.ident.to_string();
                emit_rust_item(
                    path,
                    parents,
                    &name,
                    "type",
                    &item.vis,
                    &item.attrs,
                    &item.to_token_stream(),
                    output,
                );
                let mut owner = parents.to_vec();
                owner.push(name);
                rust_named_fields(path, &owner, &item.fields.named, output);
            }
            Item::Use(item) => emit_rust_item(
                path,
                parents,
                &display_tokens(&item.tree.to_token_stream().to_string()),
                "public-api",
                &item.vis,
                &item.attrs,
                &item.to_token_stream(),
                output,
            ),
            Item::Verbatim(tokens) => {
                return Err(format!(
                    "unsupported verbatim Rust item in {path}: {}",
                    display_tokens(&tokens.to_string())
                ));
            }
            _ => return Err(format!("unsupported Rust item in {path}")),
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_rust_item(
    path: &str,
    parents: &[String],
    name: &str,
    mut kind: &str,
    visibility: &Visibility,
    attributes: &[Attribute],
    tokens: &TokenStream,
    output: &mut Vec<Extracted>,
) {
    if has_attribute(attributes, "test") {
        kind = "test";
    } else if matches!(visibility, Visibility::Public(_)) && kind == "type" {
        kind = "public-api";
    }
    emit_rust_plain(path, parents, name, kind, tokens, "syn-item", output);
}

fn emit_rust_plain(
    path: &str,
    parents: &[String],
    name: &str,
    kind: &str,
    tokens: &TokenStream,
    note: &str,
    output: &mut Vec<Extracted>,
) {
    output.push(Extracted {
        path: path.into(),
        kind: kind.into(),
        name: qualified(parents, name),
        payload: rust_token_payload(tokens),
        note: note.into(),
    });
}

fn rust_fields(path: &str, parents: &[String], fields: &syn::Fields, output: &mut Vec<Extracted>) {
    match fields {
        syn::Fields::Named(fields) => rust_named_fields(path, parents, &fields.named, output),
        syn::Fields::Unnamed(fields) => {
            for (index, field) in fields.unnamed.iter().enumerate() {
                emit_rust_plain(
                    path,
                    parents,
                    &format!("field-{index}"),
                    "type",
                    &field.to_token_stream(),
                    "syn-field",
                    output,
                );
            }
        }
        syn::Fields::Unit => {}
    }
}

fn rust_named_fields(
    path: &str,
    parents: &[String],
    fields: &syn::punctuated::Punctuated<syn::Field, syn::Token![,]>,
    output: &mut Vec<Extracted>,
) {
    for field in fields {
        let name = field
            .ident
            .as_ref()
            .expect("named field has an identifier")
            .to_string();
        emit_rust_plain(
            path,
            parents,
            &name,
            "type",
            &field.to_token_stream(),
            "syn-field",
            output,
        );
    }
}

fn rust_impl(
    path: &str,
    parents: &[String],
    item: &syn::ItemImpl,
    output: &mut Vec<Extracted>,
) -> Result<(), String> {
    let self_type = display_tokens(&item.self_ty.to_token_stream().to_string());
    let trait_name = item.trait_.as_ref().map_or_else(
        || "inherent".into(),
        |(polarity, path, _)| {
            let path = display_tokens(&path.to_token_stream().to_string());
            if polarity.is_some() {
                format!("not-{path}")
            } else {
                path
            }
        },
    );
    let parameters = display_tokens(&item.generics.params.to_token_stream().to_string());
    let where_clause = item
        .generics
        .where_clause
        .as_ref()
        .map_or_else(String::new, |clause| {
            display_tokens(&clause.to_token_stream().to_string())
        });
    let generics = format!("params={parameters};where={where_clause}");
    let name = format!("impl[{generics}]-{trait_name}-for-{self_type}");
    emit_rust_plain(
        path,
        parents,
        &name,
        "algorithm",
        &item.to_token_stream(),
        "syn-impl",
        output,
    );
    let mut owner = parents.to_vec();
    owner.push(name);
    for associated in &item.items {
        match associated {
            syn::ImplItem::Const(item) => emit_rust_item(
                path,
                &owner,
                &item.ident.to_string(),
                "type",
                &item.vis,
                &item.attrs,
                &item.to_token_stream(),
                output,
            ),
            syn::ImplItem::Fn(item) => {
                let name = item.sig.ident.to_string();
                emit_rust_item(
                    path,
                    &owner,
                    &name,
                    classify_function(&name),
                    &item.vis,
                    &item.attrs,
                    &item.to_token_stream(),
                    output,
                );
            }
            syn::ImplItem::Type(item) => emit_rust_item(
                path,
                &owner,
                &item.ident.to_string(),
                "type",
                &item.vis,
                &item.attrs,
                &item.to_token_stream(),
                output,
            ),
            syn::ImplItem::Macro(item) => {
                emit_associated_macro(path, &owner, &item.mac, &item.to_token_stream(), output);
            }
            syn::ImplItem::Verbatim(tokens) => {
                return Err(format!(
                    "unsupported verbatim impl item in {path}: {}",
                    display_tokens(&tokens.to_string())
                ));
            }
            _ => return Err(format!("unsupported impl item in {path}")),
        }
    }
    Ok(())
}

fn rust_trait_items(
    path: &str,
    parents: &[String],
    items: &[syn::TraitItem],
    output: &mut Vec<Extracted>,
) -> Result<(), String> {
    for item in items {
        match item {
            syn::TraitItem::Const(item) => emit_rust_plain(
                path,
                parents,
                &item.ident.to_string(),
                "type",
                &item.to_token_stream(),
                "syn-trait-item",
                output,
            ),
            syn::TraitItem::Fn(item) => {
                let name = item.sig.ident.to_string();
                emit_rust_plain(
                    path,
                    parents,
                    &name,
                    classify_function(&name),
                    &item.to_token_stream(),
                    "syn-trait-item",
                    output,
                );
            }
            syn::TraitItem::Type(item) => emit_rust_plain(
                path,
                parents,
                &item.ident.to_string(),
                "type",
                &item.to_token_stream(),
                "syn-trait-item",
                output,
            ),
            syn::TraitItem::Macro(item) => {
                emit_associated_macro(path, parents, &item.mac, &item.to_token_stream(), output);
            }
            syn::TraitItem::Verbatim(tokens) => {
                return Err(format!(
                    "unsupported verbatim trait item in {path}: {}",
                    display_tokens(&tokens.to_string())
                ));
            }
            _ => return Err(format!("unsupported trait item in {path}")),
        }
    }
    Ok(())
}

fn rust_foreign_items(
    path: &str,
    parents: &[String],
    items: &[syn::ForeignItem],
    output: &mut Vec<Extracted>,
) -> Result<(), String> {
    for item in items {
        match item {
            syn::ForeignItem::Fn(item) => emit_rust_item(
                path,
                parents,
                &item.sig.ident.to_string(),
                "public-api",
                &item.vis,
                &item.attrs,
                &item.to_token_stream(),
                output,
            ),
            syn::ForeignItem::Static(item) => emit_rust_item(
                path,
                parents,
                &item.ident.to_string(),
                "public-api",
                &item.vis,
                &item.attrs,
                &item.to_token_stream(),
                output,
            ),
            syn::ForeignItem::Type(item) => emit_rust_item(
                path,
                parents,
                &item.ident.to_string(),
                "public-api",
                &item.vis,
                &item.attrs,
                &item.to_token_stream(),
                output,
            ),
            syn::ForeignItem::Macro(item) => {
                emit_associated_macro(path, parents, &item.mac, &item.to_token_stream(), output);
            }
            syn::ForeignItem::Verbatim(tokens) => {
                return Err(format!(
                    "unsupported verbatim foreign item in {path}: {}",
                    display_tokens(&tokens.to_string())
                ));
            }
            _ => return Err(format!("unsupported foreign item in {path}")),
        }
    }
    Ok(())
}

fn emit_associated_macro(
    path: &str,
    parents: &[String],
    macro_: &syn::Macro,
    tokens: &TokenStream,
    output: &mut Vec<Extracted>,
) {
    let start = macro_.path.span().start();
    let name = format!(
        "macro-{}@{}:{}",
        display_tokens(&macro_.path.to_token_stream().to_string()),
        start.line,
        start.column + 1
    );
    emit_rust_plain(
        path,
        parents,
        &name,
        "algorithm",
        tokens,
        "syn-associated-macro",
        output,
    );
}

struct RustInvocationVisitor<'a> {
    path: &'a str,
    output: &'a mut Vec<Extracted>,
}

impl<'ast> syn::visit::Visit<'ast> for RustInvocationVisitor<'_> {
    fn visit_macro(&mut self, macro_: &'ast syn::Macro) {
        let Some(identifier) = macro_.path.segments.last().map(|segment| &segment.ident) else {
            return;
        };
        let name = identifier.to_string();
        if name.starts_with("assert") || name.starts_with("debug_assert") {
            let start = macro_.path.span().start();
            self.output.push(Extracted {
                path: self.path.into(),
                kind: "assertion".into(),
                name: format!("assertion@{}:{}", start.line, start.column + 1),
                payload: rust_token_payload(&macro_.to_token_stream()),
                note: "syn-assertion-macro".into(),
            });
        }
        syn::visit::visit_macro(self, macro_);
    }
}

fn has_attribute(attributes: &[Attribute], name: &str) -> bool {
    attributes
        .iter()
        .any(|attribute| attribute.path().is_ident(name))
}

fn classify_name(name: &str, fallback: &'static str) -> &'static str {
    let lowercase = name.to_ascii_lowercase();
    if lowercase.ends_with("error") || lowercase.contains("error") {
        "error"
    } else if lowercase.ends_with("done") || lowercase.ends_with("outcome") {
        "outcome"
    } else if lowercase.ends_with("event") {
        "event"
    } else if lowercase.ends_with("context") {
        "context"
    } else {
        fallback
    }
}

fn classify_function(name: &str) -> &'static str {
    if name.starts_with("guard_") || name.starts_with("is_") || name.starts_with("has_") {
        "guard"
    } else if name.starts_with("effect_")
        || name.starts_with("action_")
        || name.starts_with("enter_")
        || name.starts_with("exit_")
    {
        "action"
    } else if name == "main" || name == "process_event" || name.starts_with("process_") {
        "runtime-entrypoint"
    } else {
        "algorithm"
    }
}

fn qualified(parents: &[String], name: &str) -> String {
    if parents.is_empty() {
        name.to_owned()
    } else {
        format!("{}::{name}", parents.join("::"))
    }
}

fn sml_transition_rows(
    path: &str,
    parents: &[String],
    tokens: &TokenStream,
    output: &mut Vec<Extracted>,
) -> Result<(), String> {
    if collect_sml_transition_rows(path, parents, tokens, output)? == 0 {
        return Err(format!("sml macro in {path} has no transition rows"));
    }
    Ok(())
}

fn collect_sml_transition_rows(
    path: &str,
    parents: &[String],
    tokens: &TokenStream,
    output: &mut Vec<Extracted>,
) -> Result<usize, String> {
    let trees = tokens.clone().into_iter().collect::<Vec<_>>();
    let segments = split_token_rows(&trees);
    let rows = segments
        .iter()
        .filter(|segment| direct_has_leq(segment))
        .collect::<Vec<_>>();
    if !rows.is_empty() {
        for row in &rows {
            let first = row.first().ok_or("empty SML transition row")?;
            let start = first.span().start();
            let column = start
                .column
                .checked_add(1)
                .ok_or("SML source column overflow")?;
            let payload = rust_token_payload(&row_tokens(row));
            output.push(Extracted {
                path: path.into(),
                kind: "transition".into(),
                name: qualified(parents, &format!("transition-row@{}:{column}", start.line)),
                payload,
                note: "syn-sml-row".into(),
            });
        }
        return Ok(rows.len());
    }

    let mut count = 0usize;
    for tree in trees {
        if let TokenTree::Group(group) = tree {
            count = count
                .checked_add(collect_sml_transition_rows(
                    path,
                    parents,
                    &group.stream(),
                    output,
                )?)
                .ok_or("SML transition count overflow")?;
        }
    }
    Ok(count)
}

fn split_token_rows(trees: &[TokenTree]) -> Vec<Vec<TokenTree>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut angle_depth = 0usize;
    for (index, tree) in trees.iter().enumerate() {
        match tree {
            TokenTree::Punct(punctuation)
                if punctuation.as_char() == '<' && !next_is_punctuation(trees, index, '=') =>
            {
                angle_depth = angle_depth.saturating_add(1);
                row.push(tree.clone());
            }
            TokenTree::Punct(punctuation) if punctuation.as_char() == '>' && angle_depth > 0 => {
                angle_depth -= 1;
                row.push(tree.clone());
            }
            TokenTree::Punct(punctuation) if punctuation.as_char() == ',' && angle_depth == 0 => {
                if !row.is_empty() {
                    rows.push(std::mem::take(&mut row));
                }
            }
            _ => row.push(tree.clone()),
        }
    }
    if !row.is_empty() {
        rows.push(row);
    }
    rows
}

fn next_is_punctuation(trees: &[TokenTree], index: usize, expected: char) -> bool {
    trees.get(index + 1).is_some_and(
        |tree| matches!(tree, TokenTree::Punct(punctuation) if punctuation.as_char() == expected),
    )
}

fn direct_has_leq(trees: &[TokenTree]) -> bool {
    trees.windows(2).any(|pair| {
        matches!(
            (&pair[0], &pair[1]),
            (TokenTree::Punct(left), TokenTree::Punct(right))
                if left.as_char() == '<' && right.as_char() == '='
        )
    })
}

fn row_tokens(trees: &[TokenTree]) -> TokenStream {
    trees.iter().cloned().collect()
}

pub fn cpp_transition_rows(
    path: &str,
    source: &str,
    output: &mut Vec<Extracted>,
) -> Result<(), String> {
    const TABLE: &str = "sml::make_transition_table";
    let mut search_start = 0usize;
    while let Some(relative) = source[search_start..].find(TABLE) {
        let table_start = search_start + relative;
        let mut open = table_start + TABLE.len();
        while source
            .as_bytes()
            .get(open)
            .is_some_and(u8::is_ascii_whitespace)
        {
            open += 1;
        }
        if source.as_bytes().get(open) != Some(&b'(') {
            search_start = open;
            continue;
        }
        search_start = scan_cpp_transition_table(path, source, open, output)?;
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn scan_cpp_transition_table(
    path: &str,
    source: &str,
    open: usize,
    output: &mut Vec<Extracted>,
) -> Result<usize, String> {
    let bytes = source.as_bytes();
    let mut index = open + 1;
    let mut segment_start = index;
    let mut parentheses = 1usize;
    let mut brackets = 0usize;
    let mut braces = 0usize;
    let mut angles = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index = skip_line_comment(bytes, index + 2);
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            index = skip_block_comment(bytes, index + 2)
                .ok_or_else(|| format!("unterminated block comment in {path}"))?;
            continue;
        }
        if matches!(bytes[index], b'\'' | b'"') {
            index = skip_cpp_quote(bytes, index)
                .ok_or_else(|| format!("unterminated quoted literal in {path}"))?;
            continue;
        }
        match bytes[index] {
            b'(' => parentheses += 1,
            b')' if parentheses == 1 => {
                push_cpp_transition(path, source, segment_start, index, output)?;
                return Ok(index + 1);
            }
            b')' => parentheses -= 1,
            b'[' => brackets += 1,
            b']' if brackets > 0 => brackets -= 1,
            b'{' => braces += 1,
            b'}' if braces > 0 => braces -= 1,
            b'<' if parentheses == 1
                && brackets == 0
                && braces == 0
                && bytes.get(index + 1) != Some(&b'=') =>
            {
                angles += 1;
            }
            b'>' if angles > 0 => angles -= 1,
            b',' if parentheses == 1 && brackets == 0 && braces == 0 && angles == 0 => {
                push_cpp_transition(path, source, segment_start, index, output)?;
                segment_start = index + 1;
            }
            _ => {}
        }
        index += 1;
    }
    Err(format!("unterminated sml::make_transition_table in {path}"))
}

fn push_cpp_transition(
    path: &str,
    source: &str,
    start: usize,
    end: usize,
    output: &mut Vec<Extracted>,
) -> Result<(), String> {
    let raw = &source[start..end];
    let trimmed = raw.trim();
    let display = display_tokens(trimmed);
    if !display.contains("<=") || !display.contains("sml::state") {
        return Ok(());
    }
    let payload = canonical_tokens(trimmed)?;
    let leading = raw.len() - raw.trim_start().len();
    let absolute = start + leading;
    let (line, column) = line_column(source, absolute);
    output.push(Extracted {
        path: path.into(),
        kind: "transition".into(),
        name: format!("transition-row@{line}:{column}"),
        payload,
        note: "cpp-sml-row".into(),
    });
    Ok(())
}

fn line_column(source: &str, byte: usize) -> (usize, usize) {
    let before = &source[..byte];
    let line = before.bytes().filter(|byte| *byte == b'\n').count() + 1;
    let column = before
        .rsplit_once('\n')
        .map_or(before.len(), |(_, tail)| tail.len())
        + 1;
    (line, column)
}

const fn skip_line_comment(bytes: &[u8], mut index: usize) -> usize {
    while index < bytes.len() && bytes[index] != b'\n' {
        index += 1;
    }
    index
}

const fn skip_block_comment(bytes: &[u8], mut index: usize) -> Option<usize> {
    while index + 1 < bytes.len() {
        if bytes[index] == b'*' && bytes[index + 1] == b'/' {
            return Some(index + 2);
        }
        index += 1;
    }
    None
}

fn skip_cpp_quote(bytes: &[u8], start: usize) -> Option<usize> {
    let quote = bytes[start];
    let mut index = start + 1;
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            index = index.checked_add(2)?;
        } else if bytes[index] == quote {
            return Some(index + 1);
        } else {
            index += 1;
        }
    }
    None
}

pub fn cpp_test_assertion_invocations(
    path: &str,
    source: &str,
    output: &mut Vec<Extracted>,
) -> Result<(), String> {
    let bytes = source.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index = skip_line_comment(bytes, index + 2);
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            index = skip_block_comment(bytes, index + 2)
                .ok_or_else(|| format!("unterminated block comment in {path}"))?;
            continue;
        }
        if let Some(end) = cpp_raw_literal_end(bytes, index)? {
            index = end;
            continue;
        }
        if let Some((quote, _)) = cpp_quote_start(bytes, index) {
            index = skip_cpp_quote(bytes, quote)
                .ok_or_else(|| format!("unterminated quoted literal in {path}"))?;
            continue;
        }
        if !is_identifier_start(bytes[index]) {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        while index < bytes.len() && is_identifier_continue(bytes[index]) {
            index += 1;
        }
        let identifier = std::str::from_utf8(&bytes[start..index])
            .map_err(|error| format!("non-UTF-8 macro identifier in {path}: {error}"))?;
        let kind = if identifier.starts_with("TEST_CASE") {
            Some(("test", "test", "cpp-test-macro"))
        } else if identifier.starts_with("CHECK") || identifier.starts_with("REQUIRE") {
            Some(("assertion", "assertion", "cpp-assertion-macro"))
        } else {
            None
        };
        let Some((kind, label, note)) = kind else {
            continue;
        };
        let mut open = index;
        while bytes.get(open).is_some_and(u8::is_ascii_whitespace) {
            open += 1;
        }
        if bytes.get(open) != Some(&b'(') {
            continue;
        }
        let end = balanced_cpp_invocation_end(bytes, open)
            .ok_or_else(|| format!("unterminated {identifier} invocation in {path}"))?;
        let (line, column) = line_column(source, start);
        output.push(Extracted {
            path: path.into(),
            kind: kind.into(),
            name: format!("{label}@{line}:{column}"),
            payload: canonical_tokens(&source[start..end])?,
            note: note.into(),
        });
        index = end;
    }
    Ok(())
}

fn balanced_cpp_invocation_end(bytes: &[u8], open: usize) -> Option<usize> {
    let mut index = open + 1;
    let mut parentheses = 1usize;
    let mut brackets = 0usize;
    let mut braces = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index = skip_line_comment(bytes, index + 2);
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            index = skip_block_comment(bytes, index + 2)?;
            continue;
        }
        if let Ok(Some(end)) = source_literal_end(bytes, index) {
            index = end;
            continue;
        }
        if let Some((quote, _)) = cpp_quote_start(bytes, index) {
            index = skip_cpp_quote(bytes, quote)?;
            continue;
        }
        match bytes[index] {
            b'(' => parentheses += 1,
            b')' if parentheses == 1 && brackets == 0 && braces == 0 => {
                return Some(index + 1);
            }
            b')' => parentheses = parentheses.checked_sub(1)?,
            b'[' => brackets += 1,
            b']' => brackets = brackets.checked_sub(1)?,
            b'{' => braces += 1,
            b'}' => braces = braces.checked_sub(1)?,
            _ => {}
        }
        index += 1;
    }
    None
}

pub fn scanners(
    path: &str,
    source: &str,
    side: &str,
    output: &mut Vec<Extracted>,
) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    let bytes = source.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index = skip_line_comment(bytes, index + 2);
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            index = skip_block_comment(bytes, index + 2)
                .ok_or_else(|| format!("unterminated block comment in {path}"))?;
            continue;
        }
        if let Some(end) = source_literal_end(bytes, index)? {
            index = end;
            continue;
        }
        if let Some((quote, _)) = cpp_quote_start(bytes, index) {
            index = skip_cpp_quote(bytes, quote)
                .ok_or_else(|| format!("unterminated quoted literal in {path}"))?;
            continue;
        }
        if !is_identifier_start(bytes[index]) {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        while index < bytes.len() && is_identifier_continue(bytes[index]) {
            index += 1;
        }
        let identifier = std::str::from_utf8(&bytes[start..index])
            .map_err(|error| format!("non-UTF-8 semantic identifier in {path}: {error}"))?;
        let marker = if is_fixture_identifier(identifier) {
            Some(("fixture", "fixture"))
        } else if identifier == "process_event" || identifier == "main" {
            Some(("runtime-entrypoint", "entrypoint"))
        } else {
            None
        };
        let Some((kind, label)) = marker else {
            continue;
        };
        let end = semantic_span_end(bytes, index, path)?;
        let (line, column) = line_column(source, start);
        let name = format!("{label}@{line}:{column}");
        let payload = canonical_tokens(&source[start..end])?;
        let identity = format!("{side}:{path}:{kind}:{start}:{end}:{payload}");
        if seen.insert(identity) {
            output.push(Extracted {
                path: path.into(),
                kind: kind.into(),
                name,
                payload,
                note: "deterministic-scanner".into(),
            });
        }
    }
    Ok(())
}

fn is_fixture_identifier(identifier: &str) -> bool {
    let lower = identifier.to_ascii_lowercase();
    lower == "fixture"
        || lower.starts_with("fixture_")
        || lower.ends_with("_fixture")
        || lower.contains("_fixture_")
        || identifier.starts_with("Fixture")
        || identifier.ends_with("Fixture")
}

fn semantic_span_end(bytes: &[u8], identifier_end: usize, path: &str) -> Result<usize, String> {
    let mut cursor = skip_semantic_trivia(bytes, identifier_end, path)?;
    if bytes.get(cursor) == Some(&b'!') {
        cursor = skip_semantic_trivia(bytes, cursor + 1, path)?;
    }
    if bytes.get(cursor) == Some(&b'(') {
        cursor = balanced_cpp_invocation_end(bytes, cursor)
            .ok_or_else(|| format!("unterminated semantic invocation in {path}"))?;
        cursor = skip_semantic_trivia(bytes, cursor, path)?;
        if bytes.get(cursor) == Some(&b'{') {
            return balanced_brace_end(bytes, cursor)
                .ok_or_else(|| format!("unterminated semantic body in {path}"));
        }
        return Ok(cursor);
    }
    if bytes.get(cursor) == Some(&b'{') {
        return balanced_brace_end(bytes, cursor)
            .ok_or_else(|| format!("unterminated semantic body in {path}"));
    }
    while cursor < bytes.len() {
        if let Some(end) = source_literal_end(bytes, cursor)? {
            cursor = end;
            continue;
        }
        if bytes[cursor] == b'/' && bytes.get(cursor + 1) == Some(&b'/') {
            cursor = skip_line_comment(bytes, cursor + 2);
            continue;
        }
        if bytes[cursor] == b'/' && bytes.get(cursor + 1) == Some(&b'*') {
            cursor = skip_block_comment(bytes, cursor + 2)
                .ok_or_else(|| format!("unterminated block comment in {path}"))?;
            continue;
        }
        if let Some((quote, _)) = cpp_quote_start(bytes, cursor) {
            cursor = skip_cpp_quote(bytes, quote)
                .ok_or_else(|| format!("unterminated quoted literal in {path}"))?;
            continue;
        }
        if bytes[cursor] == b';' {
            return Ok(cursor + 1);
        }
        if bytes[cursor] == b'{' {
            return balanced_brace_end(bytes, cursor)
                .ok_or_else(|| format!("unterminated semantic body in {path}"));
        }
        cursor += 1;
    }
    Ok(identifier_end)
}

fn skip_semantic_trivia(bytes: &[u8], mut index: usize, path: &str) -> Result<usize, String> {
    loop {
        while bytes.get(index).is_some_and(u8::is_ascii_whitespace) {
            index += 1;
        }
        if bytes.get(index) == Some(&b'/') && bytes.get(index + 1) == Some(&b'/') {
            index = skip_line_comment(bytes, index + 2);
        } else if bytes.get(index) == Some(&b'/') && bytes.get(index + 1) == Some(&b'*') {
            index = skip_block_comment(bytes, index + 2)
                .ok_or_else(|| format!("unterminated block comment in {path}"))?;
        } else {
            return Ok(index);
        }
    }
}

fn balanced_brace_end(bytes: &[u8], open: usize) -> Option<usize> {
    let mut index = open + 1;
    let mut depth = 1usize;
    while index < bytes.len() {
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index = skip_line_comment(bytes, index + 2);
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            index = skip_block_comment(bytes, index + 2)?;
            continue;
        }
        if let Ok(Some(end)) = source_literal_end(bytes, index) {
            index = end;
            continue;
        }
        if let Some((quote, _)) = cpp_quote_start(bytes, index) {
            index = skip_cpp_quote(bytes, quote)?;
            continue;
        }
        match bytes[index] {
            b'{' => depth += 1,
            b'}' if depth == 1 => return Some(index + 1),
            b'}' => depth = depth.checked_sub(1)?,
            _ => {}
        }
        index += 1;
    }
    None
}

pub fn display_tokens(text: &str) -> String {
    let mut output = String::new();
    let mut characters = text.chars().peekable();
    let mut quoted = None;
    while let Some(character) = characters.next() {
        if let Some(quote) = quoted {
            output.push(character);
            if character == '\\' {
                if let Some(next) = characters.next() {
                    output.push(next);
                }
            } else if character == quote {
                quoted = None;
            }
            continue;
        }
        if character == '"' || character == '\'' {
            quoted = Some(character);
            output.push(character);
            continue;
        }
        if character == '/' && characters.peek() == Some(&'/') {
            characters.next();
            for next in characters.by_ref() {
                if next == '\n' {
                    break;
                }
            }
            continue;
        }
        if character == '/' && characters.peek() == Some(&'*') {
            characters.next();
            while let Some(next) = characters.next() {
                if next == '*' && characters.peek() == Some(&'/') {
                    characters.next();
                    break;
                }
            }
            continue;
        }
        if !character.is_whitespace() {
            output.push(character);
        }
    }
    output
}

/// Produces a trivia-free, token-kind-tagged, length-delimited C++ token stream.
pub fn canonical_tokens(text: &str) -> Result<String, String> {
    let spliced = splice_cpp_lines(text.as_bytes())?;
    let bytes = spliced.as_slice();
    let mut output = String::new();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index = skip_line_comment(bytes, index + 2);
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            index = skip_block_comment(bytes, index + 2)
                .ok_or("unterminated block comment while canonicalizing tokens")?;
            continue;
        }
        if let Some(end) = cpp_raw_literal_end(bytes, index)? {
            push_encoded(&mut output, "literal", &bytes[index..end]);
            index = end;
            continue;
        }
        if let Some((quote_index, _prefix_length)) = cpp_quote_start(bytes, index) {
            let end = skip_cpp_quote(bytes, quote_index)
                .ok_or("unterminated quoted literal while canonicalizing tokens")?;
            push_encoded(&mut output, "literal", &bytes[index..end]);
            index = end;
            continue;
        }
        if is_identifier_start(bytes[index]) {
            let start = index;
            index += 1;
            while index < bytes.len() && is_identifier_continue(bytes[index]) {
                index += 1;
            }
            push_encoded(&mut output, "identifier", &bytes[start..index]);
            continue;
        }
        if bytes[index].is_ascii_digit()
            || (bytes[index] == b'.' && bytes.get(index + 1).is_some_and(u8::is_ascii_digit))
        {
            let start = index;
            index += 1;
            while index < bytes.len() {
                let byte = bytes[index];
                if is_identifier_continue(byte)
                    || matches!(byte, b'.' | b'\'')
                    || (matches!(byte, b'+' | b'-')
                        && index > start
                        && matches!(bytes[index - 1], b'e' | b'E' | b'p' | b'P'))
                {
                    index += 1;
                } else {
                    break;
                }
            }
            push_encoded(&mut output, "number", &bytes[start..index]);
            continue;
        }
        let punctuation = CPP_PUNCTUATORS
            .iter()
            .find(|punctuation| bytes[index..].starts_with(punctuation.as_bytes()))
            .copied()
            .or_else(|| {
                bytes[index]
                    .is_ascii_punctuation()
                    .then(|| std::str::from_utf8(&bytes[index..=index]).expect("ASCII"))
            })
            .ok_or_else(|| format!("unsupported C++ token byte 0x{:02x}", bytes[index]))?;
        push_encoded(&mut output, "punct", punctuation.as_bytes());
        index += punctuation.len();
    }
    Ok(output)
}

/// Finds direct `struct sm;` declarations in qualified `emel::io::<component>` namespaces.
///
/// This deliberately uses a trivia-free token stream instead of text matching so comments,
/// literals, and declarations nested below another scope cannot satisfy the source oracle.
pub fn cpp_emel_io_sm_forward_declarations(source: &str) -> Result<Vec<String>, String> {
    let tokens = cpp_structural_tokens(source)?;
    let mut declarations = Vec::new();
    let mut index = 0usize;
    while index + 6 < tokens.len() {
        if tokens[index] != "namespace"
            || tokens[index + 1] != "emel"
            || tokens[index + 2] != "::"
            || tokens[index + 3] != "io"
            || tokens[index + 4] != "::"
            || !is_cpp_identifier_token(&tokens[index + 5])
            || tokens[index + 6] != "{"
        {
            index += 1;
            continue;
        }
        let component = tokens[index + 5].clone();
        let mut cursor = index + 7;
        let mut depth = 1usize;
        while cursor < tokens.len() && depth != 0 {
            if depth == 1
                && tokens.get(cursor).is_some_and(|token| token == "struct")
                && tokens.get(cursor + 1).is_some_and(|token| token == "sm")
                && tokens.get(cursor + 2).is_some_and(|token| token == ";")
            {
                declarations.push(format!("emel::io::{component}::sm"));
                cursor += 3;
                continue;
            }
            match tokens[cursor].as_str() {
                "{" => depth += 1,
                "}" => depth -= 1,
                _ => {}
            }
            cursor += 1;
        }
        if depth != 0 {
            return Err(format!(
                "unterminated emel::io::{component} namespace while scanning pinned source"
            ));
        }
        index = cursor;
    }
    Ok(declarations)
}

fn cpp_structural_tokens(source: &str) -> Result<Vec<String>, String> {
    let spliced = splice_cpp_lines(source.as_bytes())?;
    let bytes = spliced.as_slice();
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() {
            index += 1;
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index = skip_line_comment(bytes, index + 2);
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            index = skip_block_comment(bytes, index + 2)
                .ok_or("unterminated block comment while scanning C++ declarations")?;
            continue;
        }
        if let Some(end) = cpp_raw_literal_end(bytes, index)? {
            index = end;
            continue;
        }
        if let Some((quote_index, _)) = cpp_quote_start(bytes, index) {
            index = skip_cpp_quote(bytes, quote_index)
                .ok_or("unterminated quoted literal while scanning C++ declarations")?;
            continue;
        }
        if is_identifier_start(bytes[index]) {
            let start = index;
            index += 1;
            while index < bytes.len() && is_identifier_continue(bytes[index]) {
                index += 1;
            }
            tokens.push(
                std::str::from_utf8(&bytes[start..index])
                    .map_err(|error| format!("non-UTF-8 C++ identifier: {error}"))?
                    .to_owned(),
            );
            continue;
        }
        if bytes[index].is_ascii_digit()
            || (bytes[index] == b'.' && bytes.get(index + 1).is_some_and(u8::is_ascii_digit))
        {
            index += 1;
            while index < bytes.len()
                && (is_identifier_continue(bytes[index]) || matches!(bytes[index], b'.' | b'\''))
            {
                index += 1;
            }
            continue;
        }
        let punctuation = CPP_PUNCTUATORS
            .iter()
            .find(|punctuation| bytes[index..].starts_with(punctuation.as_bytes()))
            .copied()
            .or_else(|| {
                bytes[index]
                    .is_ascii_punctuation()
                    .then(|| std::str::from_utf8(&bytes[index..=index]).expect("ASCII"))
            })
            .ok_or_else(|| format!("unsupported C++ token byte 0x{:02x}", bytes[index]))?;
        tokens.push(punctuation.to_owned());
        index += punctuation.len();
    }
    Ok(tokens)
}

fn is_cpp_identifier_token(token: &str) -> bool {
    let bytes = token.as_bytes();
    bytes.first().is_some_and(|byte| is_identifier_start(*byte))
        && bytes
            .iter()
            .skip(1)
            .all(|byte| is_identifier_continue(*byte))
}

fn splice_cpp_lines(bytes: &[u8]) -> Result<Vec<u8>, String> {
    #[derive(Clone, Copy, Eq, PartialEq)]
    enum State {
        Outside,
        LineComment,
        BlockComment,
        Quote(u8),
    }
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    let mut state = State::Outside;
    while index < bytes.len() {
        if bytes[index] == b'\\' && bytes.get(index + 1) == Some(&b'\n') {
            index += 2;
            continue;
        } else if bytes[index] == b'\\'
            && bytes.get(index + 1) == Some(&b'\r')
            && bytes.get(index + 2) == Some(&b'\n')
        {
            index += 3;
            continue;
        }
        match state {
            State::Outside => {
                if let Some(end) = cpp_raw_literal_end(bytes, index)? {
                    output.extend_from_slice(&bytes[index..end]);
                    index = end;
                    continue;
                }
                if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
                    output.extend_from_slice(b"//");
                    index += 2;
                    state = State::LineComment;
                    continue;
                }
                if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
                    output.extend_from_slice(b"/*");
                    index += 2;
                    state = State::BlockComment;
                    continue;
                }
                if matches!(bytes[index], b'\'' | b'"') {
                    state = State::Quote(bytes[index]);
                }
            }
            State::LineComment if bytes[index] == b'\n' => state = State::Outside,
            State::BlockComment if bytes[index] == b'*' && bytes.get(index + 1) == Some(&b'/') => {
                output.extend_from_slice(b"*/");
                index += 2;
                state = State::Outside;
                continue;
            }
            State::Quote(_) if bytes[index] == b'\\' => {
                output.push(bytes[index]);
                if let Some(next) = bytes.get(index + 1) {
                    output.push(*next);
                    index += 2;
                    continue;
                }
                return Err("unterminated escape while splicing C++ source".into());
            }
            State::Quote(quote) if bytes[index] == quote => state = State::Outside,
            _ => {}
        }
        output.push(bytes[index]);
        index += 1;
    }
    Ok(output)
}

fn rust_token_payload(tokens: &TokenStream) -> String {
    fn append(output: &mut String, tokens: TokenStream) {
        for tree in tokens {
            match tree {
                TokenTree::Group(group) => {
                    let delimiter = match group.delimiter() {
                        proc_macro2::Delimiter::Parenthesis => b"parenthesis".as_slice(),
                        proc_macro2::Delimiter::Brace => b"brace".as_slice(),
                        proc_macro2::Delimiter::Bracket => b"bracket".as_slice(),
                        proc_macro2::Delimiter::None => b"none".as_slice(),
                    };
                    push_encoded(output, "group-open", delimiter);
                    append(output, group.stream());
                    push_encoded(output, "group-close", delimiter);
                }
                TokenTree::Ident(identifier) => {
                    push_encoded(output, "identifier", identifier.to_string().as_bytes());
                }
                TokenTree::Literal(literal) => {
                    push_encoded(output, "literal", literal.to_string().as_bytes());
                }
                TokenTree::Punct(punctuation) => {
                    let spacing = match punctuation.spacing() {
                        proc_macro2::Spacing::Alone => "alone",
                        proc_macro2::Spacing::Joint => "joint",
                    };
                    let value = format!("{}{spacing}", punctuation.as_char());
                    push_encoded(output, "punct", value.as_bytes());
                }
            }
        }
    }
    let mut output = String::new();
    append(&mut output, tokens.clone());
    output
}

fn push_encoded(output: &mut String, kind: &str, bytes: &[u8]) {
    use std::fmt::Write as _;
    write!(output, "{}:{}:", kind.len(), kind).expect("writing to String cannot fail");
    write!(output, "{}:", bytes.len()).expect("writing to String cannot fail");
    for byte in bytes {
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output.push(';');
}

const fn is_identifier_start(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphabetic() || !byte.is_ascii()
}

const fn is_identifier_continue(byte: u8) -> bool {
    is_identifier_start(byte) || byte.is_ascii_digit()
}

fn cpp_quote_start(bytes: &[u8], index: usize) -> Option<(usize, usize)> {
    for prefix in [&b"u8"[..], &b"u"[..], &b"U"[..], &b"L"[..], &b""[..]] {
        if bytes[index..].starts_with(prefix) {
            let quote = index + prefix.len();
            if matches!(bytes.get(quote), Some(b'\'' | b'"')) {
                if prefix.is_empty()
                    && bytes[quote] == b'\''
                    && bytes
                        .get(quote + 1)
                        .is_some_and(|byte| *byte == b'_' || byte.is_ascii_alphabetic())
                {
                    let mut end = quote + 2;
                    while bytes
                        .get(end)
                        .is_some_and(|byte| *byte == b'_' || byte.is_ascii_alphanumeric())
                    {
                        end += 1;
                    }
                    if bytes.get(end) != Some(&b'\'') {
                        return None;
                    }
                }
                return Some((quote, prefix.len()));
            }
        }
    }
    None
}

fn cpp_raw_literal_end(bytes: &[u8], index: usize) -> Result<Option<usize>, String> {
    for prefix in ["u8R\"", "uR\"", "UR\"", "LR\"", "R\""] {
        if !bytes[index..].starts_with(prefix.as_bytes()) {
            continue;
        }
        let delimiter_start = index + prefix.len();
        let mut open = delimiter_start;
        while open < bytes.len() && bytes[open] != b'(' {
            if open - delimiter_start >= 16
                || bytes[open].is_ascii_whitespace()
                || matches!(bytes[open], b')' | b'\\')
            {
                return Err("invalid C++ raw-string delimiter".into());
            }
            open += 1;
        }
        if open == bytes.len() {
            return Err("unterminated C++ raw-string delimiter".into());
        }
        let delimiter = &bytes[delimiter_start..open];
        let mut close = open + 1;
        while close < bytes.len() {
            if bytes[close] == b')'
                && bytes.get(close + 1..close + 1 + delimiter.len()) == Some(delimiter)
                && bytes.get(close + 1 + delimiter.len()) == Some(&b'"')
            {
                return Ok(Some(close + delimiter.len() + 2));
            }
            close += 1;
        }
        return Err("unterminated C++ raw string".into());
    }
    Ok(None)
}

fn source_literal_end(bytes: &[u8], index: usize) -> Result<Option<usize>, String> {
    if let Some(end) = cpp_raw_literal_end(bytes, index)? {
        return Ok(Some(end));
    }
    rust_raw_literal_end(bytes, index)
}

fn rust_raw_literal_end(bytes: &[u8], index: usize) -> Result<Option<usize>, String> {
    let mut cursor = index;
    if bytes.get(cursor) == Some(&b'b') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'r') {
        return Ok(None);
    }
    cursor += 1;
    let hashes_start = cursor;
    while bytes.get(cursor) == Some(&b'#') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'"') {
        return Ok(None);
    }
    let hashes = cursor - hashes_start;
    cursor += 1;
    while cursor < bytes.len() {
        if bytes[cursor] == b'"'
            && bytes
                .get(cursor + 1..cursor + 1 + hashes)
                .is_some_and(|suffix| suffix.iter().all(|byte| *byte == b'#'))
        {
            return Ok(Some(cursor + 1 + hashes));
        }
        cursor += 1;
    }
    Err("unterminated Rust raw literal".to_owned())
}

const CPP_PUNCTUATORS: &[&str] = &[
    "%:%:", "<=>", ">>=", "<<=", "->*", "...", "::", ".*", "->", "++", "--", "<<", ">>", "<=",
    ">=", "==", "!=", "&&", "||", "*=", "/=", "%=", "+=", "-=", "&=", "^=", "|=", "##", "<:", ":>",
    "<%", "%>", "%:",
];

fn extension_is(path: &str, extension: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .is_some_and(|actual| actual.eq_ignore_ascii_case(extension))
}

pub fn row(side: &str, extracted: &Extracted, source_tree: &str, rust_tree: &str) -> InventoryRow {
    let item_id = format!(
        "{side}:{}:{}:{}",
        extracted.path, extracted.kind, extracted.name
    );
    InventoryRow {
        item_id,
        side: side.into(),
        domain: "model".into(),
        kind: extracted.kind.clone(),
        owner_path: extracted.path.clone(),
        qualified_name: extracted.name.clone(),
        semantic_hash: semantic_hash(&[
            VERSION,
            side,
            &extracted.path,
            &extracted.kind,
            &extracted.name,
            &extracted.payload,
        ]),
        counterpart_id: String::new(),
        disposition: if side == "cpp" {
            "missing".into()
        } else {
            "rust-only".into()
        },
        approval_ref: String::new(),
        implementation_commit: String::new(),
        proof_status: "open".into(),
        proof_command: String::new(),
        proof_artifact: String::new(),
        fixture_id: String::new(),
        operand_hash: String::new(),
        source_tree: source_tree.into(),
        rust_tree: rust_tree.into(),
        notes: extracted.note.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scanner_hash_ignores_trivia() {
        assert_eq!(
            canonical_tokens("state <= event; // x"),
            canonical_tokens(" state<=event;")
        );
    }

    #[test]
    fn semantic_scanner_ignores_comments_strings_and_unrelated_identifiers() {
        let source = r##"
// fixture_in_comment(); process_event(comment);
const char * text = "fixture_in_string(); process_event(string)";
const char * raw = R"tag(fixture_in_raw(); process_event(raw))tag";
let rust_raw = r#"fixture_in_rust_raw(); process_event(raw)"#;
fn ordinary() { let fixtureless = 1; }
"##;
        let mut rows = Vec::new();
        scanners("src/sample.rs", source, "rust", &mut rows).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn semantic_scanner_captures_complete_bodies_and_body_changes() {
        let first = "fn fixture_loader() { process_event(Start { value: 1 }); }";
        let second = "fn fixture_loader() { process_event(Start { value: 2 }); }";
        let extract = |source: &str| {
            let mut rows = Vec::new();
            scanners("src/sample.rs", source, "rust", &mut rows).unwrap();
            rows
        };
        let first_rows = extract(first);
        let second_rows = extract(second);
        assert_eq!(first_rows.len(), 2);
        assert_eq!(second_rows.len(), 2);
        for kind in ["fixture", "runtime-entrypoint"] {
            let before = first_rows.iter().find(|row| row.kind == kind).unwrap();
            let after = second_rows.iter().find(|row| row.kind == kind).unwrap();
            assert_ne!(before.payload, after.payload);
        }
    }

    #[test]
    fn semantic_scanner_distinguishes_rust_lifetimes_from_char_literals() {
        let source = "fn main() { let value: &'static str = \"x\"; let quote = '\\''; }";
        let mut rows = Vec::new();
        scanners("src/main.rs", source, "rust", &mut rows).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kind, "runtime-entrypoint");
        assert!(!rows[0].payload.is_empty());
    }

    #[test]
    fn cpp_token_canonicalization_preserves_lexeme_boundaries() {
        assert_ne!(
            canonical_tokens("int x").unwrap(),
            canonical_tokens("intx").unwrap()
        );
        assert_ne!(
            canonical_tokens("return a").unwrap(),
            canonical_tokens("returna").unwrap()
        );
        assert_ne!(
            canonical_tokens("> >").unwrap(),
            canonical_tokens(">>").unwrap()
        );
        assert_eq!(
            canonical_tokens("int /* trivia */ x; // end\n return x;").unwrap(),
            canonical_tokens("int x; return x;").unwrap()
        );
    }

    #[test]
    fn cpp_token_canonicalization_applies_phase_two_line_splicing() {
        assert_eq!(
            canonical_tokens("ret\\\nurn x;").unwrap(),
            canonical_tokens("return x;").unwrap()
        );
        assert_eq!(
            canonical_tokens("// hidden \\\ncontinued\nreturn x;").unwrap(),
            canonical_tokens("return x;").unwrap()
        );
        assert_eq!(
            canonical_tokens("// hidden \\\r\ncontinued\r\nreturn x;").unwrap(),
            canonical_tokens("return x;").unwrap()
        );
        let delimiter_16 = "1234567890abcdef";
        let valid = format!("R\"{delimiter_16}(body){delimiter_16}\"");
        assert!(canonical_tokens(&valid).is_ok());
        let delimiter_17 = "1234567890abcdefg";
        let invalid = format!("R\"{delimiter_17}(body){delimiter_17}\"");
        assert!(canonical_tokens(&invalid).is_err());
    }

    // This regression executes the repository-pinned Apple Clang 16 binary contract. Keep it
    // mandatory in native macOS gates; pure raw-string canonicalization coverage above remains
    // cross-platform.
    #[cfg(target_os = "macos")]
    #[test]
    fn cpp_raw_string_restores_spliced_bytes_like_pinned_clang() {
        let source = "const char * value = R\"tag(before\\\nafter)tag\";\n";
        let without_splice = "const char * value = R\"tag(beforeafter)tag\";\n";
        assert_ne!(
            canonical_tokens(source).unwrap(),
            canonical_tokens(without_splice).unwrap()
        );
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("raw.cpp");
        std::fs::write(&path, source).unwrap();
        let output = std::process::Command::new(crate::clang::CLANG)
            .args(["-std=c++23", "-E", "-P"])
            .arg(&path)
            .output()
            .unwrap();
        assert!(output.status.success());
        let preprocessed = String::from_utf8(output.stdout).unwrap();
        assert!(preprocessed.contains("R\"tag(before\\\nafter)tag\""));
    }

    #[test]
    fn rust_token_canonicalization_preserves_tree_boundaries() {
        let separated = quote::quote!(let x = value > > other;);
        let joined = quote::quote!(letx = value >> other;);
        assert_ne!(rust_token_payload(&separated), rust_token_payload(&joined));
        let parsed_a: TokenStream = "fn f() { return 1; }".parse().unwrap();
        let parsed_b: TokenStream = "fn  f( ) {/* trivia */ return 1 ;}".parse().unwrap();
        assert_eq!(rust_token_payload(&parsed_a), rust_token_payload(&parsed_b));
    }

    #[test]
    fn syn_rejects_malformed_input() {
        let blobs = vec![Blob {
            path: "crates/emel-model/src/lib.rs".into(),
            bytes: b"fn {".to_vec(),
        }];
        assert!(rust(&blobs).is_err());
    }

    #[test]
    fn syn_impl_identity_is_structural_and_trivia_stable() {
        let blobs = vec![Blob {
            path: "crates/emel-model/src/lib.rs".into(),
            bytes: b"trait A {}\nstruct S;\nimpl A for S {}\nimpl S {}\n".to_vec(),
        }];
        let items = rust(&blobs).unwrap();
        let names = items
            .iter()
            .filter(|item| item.name.starts_with("impl["))
            .map(|item| item.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names.len(), 2);
        assert_ne!(names[0], names[1]);
        assert!(
            names
                .iter()
                .any(|name| name.contains("impl[params=;where=]-A-for-S"))
        );
        assert!(
            names
                .iter()
                .any(|name| name.contains("impl[params=;where=]-inherent-for-S"))
        );

        let shifted = rust(&[Blob {
            path: "crates/emel-model/src/lib.rs".into(),
            bytes: b"\n// trivia\ntrait A {}\nstruct S;\nimpl   A   for   S { }\nimpl S { }\n"
                .to_vec(),
        }])
        .unwrap();
        let shifted_names = shifted
            .iter()
            .filter(|item| item.name.starts_with("impl["))
            .map(|item| item.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, shifted_names);
    }

    #[test]
    fn syn_impl_identity_disambiguates_generics_where_and_self_type() {
        let items = rust(&[Blob {
            path: "crates/emel-model/src/lib.rs".into(),
            bytes: br"
trait A<T> {}
struct S<T>(T);
struct U<T>(T);
impl<T: Copy> A<T> for S<T> where T: Send {}
impl<T: Copy> A<T> for U<T> where T: Sync {}
"
            .to_vec(),
        }])
        .unwrap();
        let names = items
            .iter()
            .filter(|item| item.name.starts_with("impl["))
            .map(|item| item.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names.len(), 2);
        assert_ne!(names[0], names[1]);
        assert!(names.iter().any(|name| name.contains("Send")));
        assert!(names.iter().any(|name| name.contains("Sync")));
    }

    #[test]
    fn rust_function_bodies_and_individual_members_are_semantic_rows() {
        let source = br"
struct Pair { left: u8, right: u8 }
enum Route { First { code: u8 }, Second }
trait Execute { fn first(&self) -> u8; fn second(&self) { let _ = 2; } }
impl Pair { fn first(&self) -> u8 { 1 } fn second(&self) -> u8 { 2 } }
fn run() -> u8 { const LOCAL: u8 = 9; let value = LOCAL; value - 8 }
";
        let mut changed = source.to_vec();
        let position = changed
            .windows(b"value - 8".len())
            .position(|window| window == b"value - 8")
            .unwrap();
        changed[position + b"value - ".len()] = b'7';
        let extract = |bytes: &[u8]| {
            rust(&[Blob {
                path: "crates/emel-model/src/lib.rs".into(),
                bytes: bytes.to_vec(),
            }])
            .unwrap()
        };
        let rows = extract(source);
        let changed_rows = extract(&changed);
        let run = rows.iter().find(|row| row.name == "run").unwrap();
        let changed_run = changed_rows.iter().find(|row| row.name == "run").unwrap();
        assert_ne!(run.payload, changed_run.payload);
        assert!(!rows.iter().any(|row| row.name.ends_with("LOCAL")));
        for expected in [
            "Pair::left",
            "Pair::right",
            "Route::First",
            "Route::Second",
            "Route::First::code",
            "Execute::first",
            "Execute::second",
        ] {
            assert!(
                rows.iter().any(|row| row.name == expected),
                "missing {expected}"
            );
        }
        assert_eq!(
            rows.iter()
                .filter(|row| row.name.ends_with("::first") && row.name.contains("impl["))
                .count(),
            1
        );
        assert_eq!(
            rows.iter()
                .filter(|row| row.name.ends_with("::second") && row.name.contains("impl["))
                .count(),
            1
        );
    }

    #[test]
    fn rust_multiline_assertion_keeps_nested_full_invocation() {
        let source = br#"#[test]
fn verifies() {
    assert_eq!(
        compute((1, 2), [3, 4]),
        expected::<Pair<u8, u16>>(),
        "diagnostic {}",
        7,
    );
}
"#;
        let rows = rust(&[Blob {
            path: "crates/emel-model/src/tests.rs".into(),
            bytes: source.to_vec(),
        }])
        .unwrap();
        let assertions = rows
            .iter()
            .filter(|row| row.kind == "assertion")
            .collect::<Vec<_>>();
        assert_eq!(assertions.len(), 1);
        assert_eq!(assertions[0].name, "assertion@3:5");
        let expected: TokenStream = r#"assert_eq!(
        compute((1, 2), [3, 4]),
        expected::<Pair<u8, u16>>(),
        "diagnostic {}",
        7,
    )"#
        .parse()
        .unwrap();
        assert_eq!(assertions[0].payload, rust_token_payload(&expected));
    }

    #[test]
    fn cpp_multiline_tests_and_assertions_keep_nested_full_invocations() {
        let source = r#"TEST_CASE(
    "nested case",
    "[model]"
) {
    REQUIRE_MESSAGE(
        compare(std::pair<int, int>{1, 2}, values[3]),
        // The diagnostic line below remains part of the invocation.
        format("value {}", 3)
    );
}
"#;
        let mut rows = Vec::new();
        cpp_test_assertion_invocations("tests/model/example.cpp", source, &mut rows).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].name, "test@1:1");
        assert_eq!(rows[1].name, "assertion@5:5");
        let test_end = source.find(") {").unwrap() + 1;
        assert_eq!(
            rows[0].payload,
            canonical_tokens(&source[..test_end]).unwrap()
        );
        let assertion_start = source.find("REQUIRE_MESSAGE").unwrap();
        let assertion_end = balanced_cpp_invocation_end(
            source.as_bytes(),
            source[assertion_start..].find('(').unwrap() + assertion_start,
        )
        .unwrap();
        assert_eq!(
            rows[1].payload,
            canonical_tokens(&source[assertion_start..assertion_end]).unwrap()
        );
        assert_eq!(
            rows.iter().filter(|row| row.kind == "test").count(),
            source.matches("TEST_CASE").count()
        );
        assert_eq!(
            rows.iter().filter(|row| row.kind == "assertion").count(),
            source.matches("REQUIRE_MESSAGE").count()
        );
    }

    #[test]
    fn rust_sml_rows_are_multiline_nested_and_not_aggregate() {
        let blobs = vec![Blob {
            path: "crates/emel-model/src/sm.rs".into(),
            bytes: br#"sml! {
    Machine {
        "state_b"_s <= *"state_a"_s
            + completion<Start, Pair<u8, u16>>
            [guard((1, 2), |value| { value.0 <= value.1 })]
            / effect_first,

        X <= "state_b"_s
            + completion<Finish>(Pair(u8, u16))
            [guard_second] / effect_second,
    }
}
"#
            .to_vec(),
        }];
        let items = rust(&blobs).unwrap();
        let transitions = items
            .iter()
            .filter(|item| item.kind == "transition")
            .collect::<Vec<_>>();
        assert_eq!(transitions.len(), 2);
        assert_eq!(transitions[0].name, "transition-row@3:9");
        assert_eq!(transitions[1].name, "transition-row@8:9");
        assert!(transitions.iter().all(|item| item.note == "syn-sml-row"));
        let first: TokenStream = r#""state_b"_s <= *"state_a"_s
            + completion<Start, Pair<u8, u16>>
            [guard((1, 2), |value| { value.0 <= value.1 })]
            / effect_first"#
            .parse()
            .unwrap();
        assert_eq!(transitions[0].payload, rust_token_payload(&first));
        assert_ne!(transitions[0].payload, transitions[1].payload);
    }

    #[test]
    fn cpp_sml_rows_are_full_multiline_nested_expressions() {
        let source = r"auto table() {
  return sml::make_transition_table(
   sml::state<A> <= *sml::state<B>
       + sml::portable::event<Event<std::pair<int, int>>>
       // The next line remains part of this transition's semantic payload.
       [guard([](auto value) { return value.first <= value.second; })]
       / action([](int left, int right) { return left + right; })
 , sml::state<C> <= sml::state<A>
       + sml::completion<Event>(Payload{1, 2})
       [other_guard] / other_action
  );
}
";
        let mut items = Vec::new();
        cpp_transition_rows("src/emel/model/sm.hpp", source, &mut items).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].name, "transition-row@3:4");
        assert_eq!(items[1].name, "transition-row@8:4");
        assert!(items.iter().all(|item| item.note == "cpp-sml-row"));
        let first = r"sml::state<A> <= *sml::state<B>
       + sml::portable::event<Event<std::pair<int, int>>>
       // The next line remains part of this transition's semantic payload.
       [guard([](auto value) { return value.first <= value.second; })]
       / action([](int left, int right) { return left + right; })";
        assert_eq!(items[0].payload, canonical_tokens(first).unwrap());
        assert_eq!(
            items[0].payload,
            canonical_tokens(&first.replace(
                "       // The next line remains part of this transition's semantic payload.\n",
                ""
            ))
            .unwrap()
        );
    }

    #[test]
    fn emel_io_forward_scan_is_token_and_namespace_aware() {
        let source = r#"
// namespace emel::io::commented { struct sm; }
constexpr auto text = "namespace emel::io::quoted { struct sm; }";
constexpr auto raw = R"tag(namespace emel::io::raw { struct sm; })tag";
namespace emel::io::loader {
struct sm;
struct not_sm;
namespace nested { struct sm; }
}
namespace emel::io::read { struct sm; struct sm; }
namespace other::io::ignored { struct sm; }
"#;
        assert_eq!(
            cpp_emel_io_sm_forward_declarations(source).unwrap(),
            vec![
                "emel::io::loader::sm",
                "emel::io::read::sm",
                "emel::io::read::sm",
            ]
        );

        let unterminated = "namespace emel::io::loader { struct sm;";
        assert!(cpp_emel_io_sm_forward_declarations(unterminated).is_err());
    }
}
