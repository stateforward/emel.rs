use std::collections::BTreeSet;

pub const VERSION: &str = "emel-model-parity-inventory/v1";
pub const INVENTORY_HEADER: &str = "item_id\tside\tdomain\tkind\towner_path\tqualified_name\tsemantic_hash\tcounterpart_id\tdisposition\tapproval_ref\timplementation_commit\tproof_status\tproof_command\tproof_artifact\tfixture_id\toperand_hash\tsource_tree\trust_tree\tnotes";
pub const COVERAGE_HEADER: &str = "item_id\tunit_test\tparity_case\tparity_modes\tfuzz_target\tallocation_test\tbenchmark_case\tmax_cpp_ratio\truntime_entrypoint\tterminal_tree\tstatus";
pub const GAP_HEADER: &str = "item_id\tgap_kind\towner_lane\tblocker\tnext_proof\tstatus";

const KINDS: &[&str] = &[
    "file",
    "type",
    "event",
    "outcome",
    "error",
    "context",
    "guard",
    "action",
    "transition",
    "algorithm",
    "fixture",
    "test",
    "assertion",
    "runtime-entrypoint",
    "public-api",
];
const DISPOSITIONS: &[&str] = &[
    "exact",
    "approved-contract-delta",
    "rust-only",
    "missing",
    "partial",
];
const PROOF_STATUSES: &[&str] = &["open", "implemented", "proven"];
const COVERAGE_STATUSES: &[&str] = &["open", "implemented", "proven"];
const GAP_KINDS: &[&str] = &["unmatched", "unproven"];
const GAP_STATUSES: &[&str] = &["open", "blocked", "closed"];

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
/// One validated row from a parity inventory.
pub struct InventoryRow {
    pub item_id: String,
    pub side: String,
    pub domain: String,
    pub kind: String,
    pub owner_path: String,
    pub qualified_name: String,
    pub semantic_hash: String,
    pub counterpart_id: String,
    pub disposition: String,
    pub approval_ref: String,
    pub implementation_commit: String,
    pub proof_status: String,
    pub proof_command: String,
    pub proof_artifact: String,
    pub fixture_id: String,
    pub operand_hash: String,
    pub source_tree: String,
    pub rust_tree: String,
    pub notes: String,
}

impl InventoryRow {
    pub fn fields(&self) -> [&str; 19] {
        [
            &self.item_id,
            &self.side,
            &self.domain,
            &self.kind,
            &self.owner_path,
            &self.qualified_name,
            &self.semantic_hash,
            &self.counterpart_id,
            &self.disposition,
            &self.approval_ref,
            &self.implementation_commit,
            &self.proof_status,
            &self.proof_command,
            &self.proof_artifact,
            &self.fixture_id,
            &self.operand_hash,
            &self.source_tree,
            &self.rust_tree,
            &self.notes,
        ]
    }

    pub fn validate(&self) -> Result<(), String> {
        let expected_id = format!(
            "{}:{}:{}:{}",
            self.side, self.owner_path, self.kind, self.qualified_name
        );
        if self.item_id != expected_id || !matches!(self.side.as_str(), "cpp" | "rust") {
            return Err(format!("invalid item identity: {}", self.item_id));
        }
        if self.domain != "model" {
            return Err(format!("invalid inventory domain: {}", self.domain));
        }
        if !KINDS.contains(&self.kind.as_str()) {
            return Err(format!("unsupported item kind: {}", self.kind));
        }
        if !DISPOSITIONS.contains(&self.disposition.as_str()) {
            return Err(format!("unsupported disposition: {}", self.disposition));
        }
        if !PROOF_STATUSES.contains(&self.proof_status.as_str()) {
            return Err(format!("unsupported proof status: {}", self.proof_status));
        }
        if self.disposition == "approved-contract-delta" && self.approval_ref.is_empty() {
            return Err(format!("approved delta lacks approval: {}", self.item_id));
        }
        if self.disposition != "approved-contract-delta" && !self.approval_ref.is_empty() {
            return Err(format!("non-delta row carries approval: {}", self.item_id));
        }
        if (self.side == "cpp" && self.disposition == "rust-only")
            || (self.side == "rust" && self.disposition == "missing")
        {
            return Err(format!("side-incompatible disposition: {}", self.item_id));
        }
        if matches!(self.disposition.as_str(), "missing" | "rust-only")
            && !self.counterpart_id.is_empty()
        {
            return Err(format!(
                "unmatched disposition has counterpart: {}",
                self.item_id
            ));
        }
        if matches!(
            self.disposition.as_str(),
            "exact" | "approved-contract-delta" | "partial"
        ) && self.counterpart_id.is_empty()
        {
            return Err(format!(
                "matched disposition lacks counterpart: {}",
                self.item_id
            ));
        }
        if !is_lower_hex(&self.semantic_hash, 64) {
            return Err(format!("invalid semantic hash: {}", self.item_id));
        }
        validate_source_tree(&self.source_tree)?;
        if !is_lower_hex(&self.rust_tree, 40) {
            return Err(format!("invalid Rust tree: {}", self.item_id));
        }
        if !self.implementation_commit.is_empty() && !is_lower_hex(&self.implementation_commit, 40)
        {
            return Err(format!("invalid implementation commit: {}", self.item_id));
        }
        if !self.operand_hash.is_empty() && !is_lower_hex(&self.operand_hash, 64) {
            return Err(format!("invalid operand hash: {}", self.item_id));
        }
        if self.proof_status == "proven"
            && [
                &self.implementation_commit,
                &self.proof_command,
                &self.proof_artifact,
                &self.fixture_id,
                &self.operand_hash,
            ]
            .iter()
            .any(|field| field.is_empty())
        {
            return Err(format!("proven row lacks proof binding: {}", self.item_id));
        }
        if self.proof_status == "open"
            && [
                &self.implementation_commit,
                &self.proof_command,
                &self.proof_artifact,
                &self.fixture_id,
                &self.operand_hash,
            ]
            .iter()
            .any(|field| !field.is_empty())
        {
            return Err(format!("open row carries proof binding: {}", self.item_id));
        }
        if self
            .fields()
            .iter()
            .any(|field| field.contains(['\t', '\n', '\r']))
        {
            return Err(format!("TSV control character in {}", self.item_id));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoverageRow {
    pub item_id: String,
    pub terminal_tree: String,
    pub status: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GapRow {
    pub item_id: String,
    pub gap_kind: String,
    pub owner_lane: String,
    pub blocker: String,
    pub next_proof: String,
    pub status: String,
}

/// Parses and validates a complete inventory TSV, rejecting malformed and duplicate rows.
///
/// # Errors
///
/// Returns a diagnostic when UTF-8, the closed schema, vocabulary, or ID uniqueness is invalid.
pub fn parse_inventory_tsv(bytes: &[u8]) -> Result<Vec<InventoryRow>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| format!("invalid UTF-8 TSV: {error}"))?;
    let mut lines = text.lines();
    if lines.next() != Some(INVENTORY_HEADER) {
        return Err("inventory header mismatch".to_owned());
    }
    let mut rows = Vec::new();
    let mut ids = BTreeSet::new();
    for (line_index, line) in lines.enumerate() {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 19 {
            return Err(format!(
                "line {} has {} fields, expected 19",
                line_index + 2,
                fields.len()
            ));
        }
        let row = InventoryRow {
            item_id: fields[0].into(),
            side: fields[1].into(),
            domain: fields[2].into(),
            kind: fields[3].into(),
            owner_path: fields[4].into(),
            qualified_name: fields[5].into(),
            semantic_hash: fields[6].into(),
            counterpart_id: fields[7].into(),
            disposition: fields[8].into(),
            approval_ref: fields[9].into(),
            implementation_commit: fields[10].into(),
            proof_status: fields[11].into(),
            proof_command: fields[12].into(),
            proof_artifact: fields[13].into(),
            fixture_id: fields[14].into(),
            operand_hash: fields[15].into(),
            source_tree: fields[16].into(),
            rust_tree: fields[17].into(),
            notes: fields[18].into(),
        };
        row.validate()?;
        if !ids.insert(row.item_id.clone()) {
            return Err(format!("duplicate item ID: {}", row.item_id));
        }
        rows.push(row);
    }
    Ok(rows)
}

pub fn inventory_tsv(rows: &[InventoryRow]) -> Result<Vec<u8>, String> {
    let mut rows = rows.to_vec();
    rows.sort();
    let mut ids = BTreeSet::new();
    let mut output = String::from(INVENTORY_HEADER);
    output.push('\n');
    for row in rows {
        row.validate()?;
        if !ids.insert(row.item_id.clone()) {
            return Err(format!("duplicate item ID: {}", row.item_id));
        }
        output.push_str(&row.fields().join("\t"));
        output.push('\n');
    }
    Ok(output.into_bytes())
}

pub fn parse_coverage_tsv(bytes: &[u8]) -> Result<Vec<CoverageRow>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| format!("invalid UTF-8 TSV: {error}"))?;
    let mut lines = text.lines();
    if lines.next() != Some(COVERAGE_HEADER) {
        return Err("coverage header mismatch".into());
    }
    let mut ids = BTreeSet::new();
    let mut rows = Vec::new();
    for (index, line) in lines.enumerate() {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 11 {
            return Err(format!(
                "coverage line {} has incorrect field count",
                index + 2
            ));
        }
        if !ids.insert(fields[0].to_owned()) {
            return Err(format!("duplicate coverage item ID: {}", fields[0]));
        }
        if !COVERAGE_STATUSES.contains(&fields[10]) {
            return Err(format!("invalid coverage status: {}", fields[10]));
        }
        if !is_lower_hex(fields[9], 40) {
            return Err(format!("invalid coverage terminal tree: {}", fields[0]));
        }
        rows.push(CoverageRow {
            item_id: fields[0].into(),
            terminal_tree: fields[9].into(),
            status: fields[10].into(),
        });
    }
    Ok(rows)
}

pub fn parse_gap_tsv(bytes: &[u8]) -> Result<Vec<GapRow>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| format!("invalid UTF-8 TSV: {error}"))?;
    let mut lines = text.lines();
    if lines.next() != Some(GAP_HEADER) {
        return Err("gap header mismatch".into());
    }
    let mut ids = BTreeSet::new();
    let mut rows = Vec::new();
    for (index, line) in lines.enumerate() {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 6 {
            return Err(format!("gap line {} has incorrect field count", index + 2));
        }
        if !ids.insert(fields[0].to_owned()) {
            return Err(format!("duplicate gap item ID: {}", fields[0]));
        }
        if !GAP_KINDS.contains(&fields[1]) {
            return Err(format!("invalid gap kind: {}", fields[1]));
        }
        if !GAP_STATUSES.contains(&fields[5]) {
            return Err(format!("invalid gap status: {}", fields[5]));
        }
        if fields[2].is_empty() || fields[4].is_empty() {
            return Err(format!("gap lacks owner or next proof: {}", fields[0]));
        }
        rows.push(GapRow {
            item_id: fields[0].into(),
            gap_kind: fields[1].into(),
            owner_lane: fields[2].into(),
            blocker: fields[3].into(),
            next_proof: fields[4].into(),
            status: fields[5].into(),
        });
    }
    Ok(rows)
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn validate_source_tree(value: &str) -> Result<(), String> {
    let Some((commit, trees)) = value
        .strip_prefix("commit=")
        .and_then(|value| value.split_once(";model="))
    else {
        return Err("invalid source tree binding".into());
    };
    let Some((model, tests)) = trees.split_once(";tests=") else {
        return Err("invalid source tree binding".into());
    };
    if !is_lower_hex(commit, 40) || !is_lower_hex(model, 40) || !is_lower_hex(tests, 40) {
        return Err("invalid source tree hash".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_row() -> InventoryRow {
        InventoryRow {
            item_id: "cpp:src/emel/model/a.hpp:type:A".into(),
            side: "cpp".into(),
            domain: "model".into(),
            kind: "type".into(),
            owner_path: "src/emel/model/a.hpp".into(),
            qualified_name: "A".into(),
            semantic_hash: "a".repeat(64),
            counterpart_id: String::new(),
            disposition: "missing".into(),
            approval_ref: String::new(),
            implementation_commit: String::new(),
            proof_status: "open".into(),
            proof_command: String::new(),
            proof_artifact: String::new(),
            fixture_id: String::new(),
            operand_hash: String::new(),
            source_tree: format!(
                "commit={};model={};tests={}",
                "a".repeat(40),
                "b".repeat(40),
                "c".repeat(40)
            ),
            rust_tree: "d".repeat(40),
            notes: "test".into(),
        }
    }

    #[test]
    fn inventory_row_enforces_exact_identity_hash_and_side_rules() {
        assert!(valid_row().validate().is_ok());
        let mut row = valid_row();
        row.item_id.push_str("-wrong");
        assert!(row.validate().is_err());
        let mut row = valid_row();
        row.semantic_hash = "A".repeat(64);
        assert!(row.validate().is_err());
        let mut row = valid_row();
        row.disposition = "rust-only".into();
        assert!(row.validate().is_err());
        let mut row = valid_row();
        row.approval_ref = "not-a-delta".into();
        assert!(row.validate().is_err());
    }

    #[test]
    fn inventory_row_enforces_proof_bindings() {
        let mut proven = valid_row();
        proven.proof_status = "proven".into();
        assert!(proven.validate().is_err());
        proven.implementation_commit = "e".repeat(40);
        proven.proof_command = "cargo test".into();
        proven.proof_artifact = "proof.tsv".into();
        proven.fixture_id = "fixture".into();
        proven.operand_hash = "f".repeat(64);
        assert!(proven.validate().is_ok());
        let mut open = valid_row();
        open.proof_command = "hidden proof".into();
        assert!(open.validate().is_err());
    }

    #[test]
    fn coverage_and_gap_parsers_reject_duplicates_and_closed_vocabulary_drift() {
        let tree = "a".repeat(40);
        let coverage = format!(
            "{COVERAGE_HEADER}\nid\t\t\t\t\t\t\t\t\t{tree}\topen\nid\t\t\t\t\t\t\t\t\t{tree}\topen\n"
        );
        assert!(parse_coverage_tsv(coverage.as_bytes()).is_err());
        let invalid_coverage = format!("{COVERAGE_HEADER}\nid\t\t\t\t\t\t\t\t\t{tree}\tunknown\n");
        assert!(parse_coverage_tsv(invalid_coverage.as_bytes()).is_err());
        let gaps = format!(
            "{GAP_HEADER}\nid\tunmatched\tlane\tblocker\tproof\topen\nid\tunmatched\tlane\tblocker\tproof\topen\n"
        );
        assert!(parse_gap_tsv(gaps.as_bytes()).is_err());
        let invalid_gap = format!("{GAP_HEADER}\nid\tunknown\tlane\tblocker\tproof\tinvalid\n");
        assert!(parse_gap_tsv(invalid_gap.as_bytes()).is_err());
    }
}
