//! Source-aligned bounded GBNF nonterminal parser child actor.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::missing_const_for_fn,
    dead_code,
    missing_docs
)]

use super::super::lexer::TokenKind;
use crate::gbnf::{k_gbnf_symbol_table_slots, k_max_gbnf_rules, k_max_gbnf_symbols};
use sml::sml;

/// Maximum UTF-8 bytes retained for one nonterminal name.
pub const MAX_NONTERM_NAME_BYTES: usize = 128;

/// Nonterminal parsing mode from `nonterm_parser/events.hpp`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum ParseMode {
    #[default]
    None = 0,
    Definition = 1,
    Reference = 2,
}

/// Errors represented by the pinned rule-parser error values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NontermParserError {
    ParseFailed,
    InternalError,
}

/// Owned result from one nonterminal parser dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseOutcome {
    /// The name was resolved or inserted and carries its bounded rule ID.
    Parsed { rule_id: u32, mode: ParseMode },
    /// The token was not a valid nonterminal request, or lookup was rejected.
    ParseFailed,
    /// An event was sent after completion or was otherwise unexpected.
    InternalError,
}

/// A copied, allocation-free `parse_rules` input for the child actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuleParserEventParseRules {
    pub token_kind: TokenKind,
    pub has_token: bool,
    pub nonterm_mode: ParseMode,
    pub symbol: [u8; MAX_NONTERM_NAME_BYTES],
    pub symbol_len: u16,
    pub symbol_valid: bool,
}

impl RuleParserEventParseRules {
    /// Creates a copied identifier request. Oversized names are rejected.
    #[must_use]
    pub fn new(token_kind: TokenKind, nonterm_mode: ParseMode, text: &str) -> Self {
        let bytes = text.as_bytes();
        let valid = bytes.len() <= MAX_NONTERM_NAME_BYTES;
        let mut symbol = [0; MAX_NONTERM_NAME_BYTES];
        let copied = bytes.len().min(MAX_NONTERM_NAME_BYTES);
        symbol[..copied].copy_from_slice(&bytes[..copied]);
        Self {
            token_kind,
            has_token: true,
            nonterm_mode,
            symbol,
            symbol_len: copied as u16,
            symbol_valid: valid,
        }
    }

    /// Creates an absent-token request.
    #[must_use]
    pub const fn absent() -> Self {
        Self {
            token_kind: TokenKind::Unknown,
            has_token: false,
            nonterm_mode: ParseMode::None,
            symbol: [0; MAX_NONTERM_NAME_BYTES],
            symbol_len: 0,
            symbol_valid: false,
        }
    }

    /// Creates a copied request from a lexer token view.
    #[must_use]
    pub fn from_token(token_kind: TokenKind, nonterm_mode: ParseMode, text: &str) -> Self {
        Self::new(token_kind, nonterm_mode, text)
    }

    /// Returns the retained symbol as UTF-8 when the bounded copy was valid.
    #[must_use]
    pub fn symbol_text(&self) -> Option<&str> {
        if !self.symbol_valid {
            return None;
        }
        core::str::from_utf8(&self.symbol[..self.symbol_len as usize]).ok()
    }
}

impl Default for RuleParserEventParseRules {
    fn default() -> Self { Self::absent() }
}

impl From<TokenKind> for RuleParserEventParseRules {
    fn from(token_kind: TokenKind) -> Self { Self::new(token_kind, ParseMode::None, "") }
}

// Source mapping: nonterm_parser/sm.hpp topology, including lookup execution,
// lookup decision, terminal, and explicit unexpected-event rows.
sml! {
    GbnfRuleParserNontermParser {
        "definition_lookup_exec"_s <= *"deciding"_s + completion<RuleParserEventParseRules> [token_identifier_definition],
        "reference_lookup_exec"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_identifier_reference],
        "parse_failed"_s <= "deciding"_s + completion<RuleParserEventParseRules> / dispatch_parse_failed_from_deciding,
        "definition_lookup_decision"_s <= "definition_lookup_exec"_s + completion<RuleParserEventParseRules> / lookup_definition_candidate,
        "reference_lookup_decision"_s <= "reference_lookup_exec"_s + completion<RuleParserEventParseRules> / lookup_reference_candidate,
        "parsed"_s <= "definition_lookup_decision"_s + completion<RuleParserEventParseRules> [definition_existing_valid] / consume_definition_existing,
        "parsed"_s <= "definition_lookup_decision"_s + completion<RuleParserEventParseRules> [definition_new_valid] / consume_definition_new,
        "parse_failed"_s <= "definition_lookup_decision"_s + completion<RuleParserEventParseRules> [definition_failed] / dispatch_parse_failed_from_definition_lookup_decision,
        "parsed"_s <= "reference_lookup_decision"_s + completion<RuleParserEventParseRules> [reference_existing_valid] / consume_reference_existing,
        "parsed"_s <= "reference_lookup_decision"_s + completion<RuleParserEventParseRules> [reference_new_valid] / consume_reference_new,
        "parse_failed"_s <= "reference_lookup_decision"_s + completion<RuleParserEventParseRules> [reference_failed] / dispatch_parse_failed_from_reference_lookup_decision,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "definition_lookup_exec"_s + unexpected_event<_> / on_unexpected_from_definition_lookup_exec,
        "unexpected_event"_s <= "definition_lookup_decision"_s + unexpected_event<_> / on_unexpected_from_definition_lookup_decision,
        "unexpected_event"_s <= "reference_lookup_exec"_s + unexpected_event<_> / on_unexpected_from_reference_lookup_exec,
        "unexpected_event"_s <= "reference_lookup_decision"_s + unexpected_event<_> / on_unexpected_from_reference_lookup_decision,
        "unexpected_event"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "parsed"_s = X,
        "parse_failed"_s = X,
    }
}

#[derive(Clone, Copy, Debug)]
struct SymbolEntry {
    name: [u8; MAX_NONTERM_NAME_BYTES],
    len: u16,
    hash: u32,
    id: u32,
    occupied: bool,
}

impl Default for SymbolEntry {
    fn default() -> Self {
        Self { name: [0; MAX_NONTERM_NAME_BYTES], len: 0, hash: 0, id: 0, occupied: false }
    }
}

#[derive(Debug)]
struct SymbolTable {
    entries: [SymbolEntry; k_gbnf_symbol_table_slots],
    count: u32,
}

impl Default for SymbolTable {
    fn default() -> Self { Self { entries: [SymbolEntry::default(); k_gbnf_symbol_table_slots], count: 0 } }
}

impl SymbolTable {
    fn hash(name: &[u8]) -> u32 {
        let mut hash = 2_166_136_261u32;
        for byte in name {
            hash ^= u32::from(*byte);
            hash = hash.wrapping_mul(16_777_619);
        }
        if hash == 0 { 1 } else { hash }
    }

    fn find(&self, name: &[u8], hash: u32) -> Option<u32> {
        let mask = k_gbnf_symbol_table_slots - 1;
        let mut slot = (hash as usize) & mask;
        for _ in 0..k_gbnf_symbol_table_slots {
            let entry = &self.entries[slot];
            if !entry.occupied { return None; }
            if entry.hash == hash && entry.len as usize == name.len() && entry.name[..name.len()] == name[..] {
                return Some(entry.id);
            }
            slot = (slot + 1) & mask;
        }
        None
    }

    fn has_insert_slot(&self, name: &[u8], hash: u32) -> bool {
        let mask = k_gbnf_symbol_table_slots - 1;
        let mut slot = (hash as usize) & mask;
        for _ in 0..k_gbnf_symbol_table_slots {
            let entry = &self.entries[slot];
            if !entry.occupied || (entry.hash == hash && entry.len as usize == name.len() && entry.name[..name.len()] == name[..]) {
                return true;
            }
            slot = (slot + 1) & mask;
        }
        false
    }

    fn insert(&mut self, name: &[u8], hash: u32, id: u32) -> bool {
        let mask = k_gbnf_symbol_table_slots - 1;
        let mut slot = (hash as usize) & mask;
        for _ in 0..k_gbnf_symbol_table_slots {
            let entry = &mut self.entries[slot];
            if !entry.occupied {
                entry.name[..name.len()].copy_from_slice(name);
                entry.len = name.len() as u16;
                entry.hash = hash;
                entry.id = id;
                entry.occupied = true;
                self.count += 1;
                return true;
            }
            if entry.hash == hash && entry.len as usize == name.len() && entry.name[..name.len()] == name[..] {
                entry.id = id;
                return true;
            }
            slot = (slot + 1) & mask;
        }
        false
    }
}

/// Context retained by the generated nonterminal child machine.
#[derive(Debug)]
pub struct GbnfRuleParserNontermParserContext {
    input: RuleParserEventParseRules,
    symbols: SymbolTable,
    rule_defined: [bool; k_max_gbnf_rules],
    next_symbol_id: u32,
    lookup_hash: u32,
    lookup_rule_id: u32,
    lookup_found: bool,
    lookup_can_insert: bool,
    outcome: ParseOutcome,
    error: Option<NontermParserError>,
}

impl Default for GbnfRuleParserNontermParserContext {
    fn default() -> Self {
        Self {
            input: RuleParserEventParseRules::absent(),
            symbols: SymbolTable::default(),
            rule_defined: [false; k_max_gbnf_rules],
            next_symbol_id: 0,
            lookup_hash: 0,
            lookup_rule_id: 0,
            lookup_found: false,
            lookup_can_insert: false,
            outcome: ParseOutcome::ParseFailed,
            error: None,
        }
    }
}

impl GbnfRuleParserNontermParserContext {
    fn set_input(&mut self, input: RuleParserEventParseRules) {
        self.input = input;
        self.lookup_hash = 0;
        self.lookup_rule_id = 0;
        self.lookup_found = false;
        self.lookup_can_insert = false;
        self.outcome = ParseOutcome::ParseFailed;
        self.error = None;
    }

    fn name(&self) -> Option<&[u8]> {
        if !self.input.symbol_valid || self.input.symbol_len == 0 { return None; }
        Some(&self.input.symbol[..self.input.symbol_len as usize])
    }

    fn consume(&mut self, rule_id: u32) -> Result<(), ()> {
        self.outcome = ParseOutcome::Parsed { rule_id, mode: self.input.nonterm_mode };
        self.error = None;
        Ok(())
    }

    fn fail(&mut self) -> Result<(), ()> {
        self.outcome = ParseOutcome::ParseFailed;
        self.error = Some(NontermParserError::ParseFailed);
        Ok(())
    }

    fn unexpected(&mut self) -> Result<(), ()> {
        self.outcome = ParseOutcome::InternalError;
        self.error = Some(NontermParserError::InternalError);
        Ok(())
    }

    fn lookup_candidate(&mut self) {
        let Some(name) = self.name() else { self.lookup_can_insert = false; return; };
        let name_len = name.len();
        let mut copied_name = [0u8; MAX_NONTERM_NAME_BYTES];
        copied_name[..name_len].copy_from_slice(name);
        let name = &copied_name[..name_len];
        let hash = SymbolTable::hash(name);
        let found = self.symbols.find(name, hash);
        self.lookup_hash = hash;
        self.lookup_rule_id = found.unwrap_or(0);
        self.lookup_found = found.is_some();
        self.lookup_can_insert = !self.lookup_found
            && self.next_symbol_id < k_max_gbnf_rules as u32
            && self.symbols.count < k_max_gbnf_symbols as u32
            && self.symbols.has_insert_slot(name, hash);
    }
    fn consume_definition_new(&mut self) -> Result<(), ()> {
        let id = self.next_symbol_id;
        let Some(name) = self.name() else { return self.fail(); };
        let name_len = name.len();
        let mut copied_name = [0u8; MAX_NONTERM_NAME_BYTES];
        copied_name[..name_len].copy_from_slice(name);
        if id as usize >= k_max_gbnf_rules || self.symbols.count as usize >= k_max_gbnf_symbols || !self.symbols.insert(&copied_name[..name_len], self.lookup_hash, id) {
            return self.fail();
        }
        self.next_symbol_id += 1;
        self.rule_defined[id as usize] = true;
        self.consume(id)
    }

    fn consume_reference_existing(&mut self) -> Result<(), ()> { self.consume(self.lookup_rule_id) }

    fn consume_reference_new(&mut self) -> Result<(), ()> {
        let id = self.next_symbol_id;
        let Some(name) = self.name() else { return self.fail(); };
        let name_len = name.len();
        let mut copied_name = [0u8; MAX_NONTERM_NAME_BYTES];
        copied_name[..name_len].copy_from_slice(name);
        if id as usize >= k_max_gbnf_rules || self.symbols.count as usize >= k_max_gbnf_symbols || !self.symbols.insert(&copied_name[..name_len], self.lookup_hash, id) {
            return self.fail();
        }
        self.next_symbol_id += 1;
        self.consume(id)
    }

PUT 422.=425:
pub struct GbnfRuleParserNontermParserActor {
    machine: GbnfRuleParserNontermParserStateMachine<GbnfRuleParserNontermParserContext>,
}

PUT 460.=465:
    pub fn process_unexpected_event(&mut self) -> ParseOutcome {
        self.machine.context_mut().error = Some(NontermParserError::InternalError);
        self.machine.context_mut().outcome = ParseOutcome::InternalError;
        self.machine.set_state(GbnfRuleParserNontermParserStates::UnexpectedEvent);
        self.machine.context().outcome
    }


PUT 374.=378:
/// Synchronous bounded nonterminal parser child actor.
pub struct GbnfRuleParserNontermParserActor {
    machine: GbnfRuleParserNontermParserStateMachine<GbnfRuleParserNontermParserContext>,
}

PUT 413.=418:
    pub fn process_unexpected_event(&mut self) -> ParseOutcome {
        self.machine.context_mut().error = Some(NontermParserError::InternalError);
        self.machine.context_mut().outcome = ParseOutcome::InternalError;
        self.machine.set_state(GbnfRuleParserNontermParserStates::UnexpectedEvent);
        self.machine.context().outcome
    }
}

impl GbnfRuleParserNontermParserStateMachineContext for GbnfRuleParserNontermParserContext {
    fn consume_definition_existing(&mut self) -> Result<(), ()> {
        self.rule_defined[self.lookup_rule_id as usize] = true;
        self.consume(self.lookup_rule_id)
    }

    fn consume_definition_new(&mut self) -> Result<(), ()> {
        let id = self.next_symbol_id;
        let Some(name) = self.name() else { return self.fail(); };
        if id as usize >= k_max_gbnf_rules || self.symbols.count as usize >= k_max_gbnf_symbols || !self.symbols.insert(name, self.lookup_hash, id) {
            return self.fail();
        }
        self.next_symbol_id += 1;
        self.rule_defined[id as usize] = true;
        self.consume(id)
    }

    fn consume_reference_existing(&mut self) -> Result<(), ()> { self.consume(self.lookup_rule_id) }

    fn consume_reference_new(&mut self) -> Result<(), ()> {
        let id = self.next_symbol_id;
        let Some(name) = self.name() else { return self.fail(); };
        if id as usize >= k_max_gbnf_rules || self.symbols.count as usize >= k_max_gbnf_symbols || !self.symbols.insert(name, self.lookup_hash, id) {
            return self.fail();
        }
        self.next_symbol_id += 1;
        self.consume(id)
    }

    fn definition_existing_valid(&self) -> Result<bool, ()> {
        Ok(self.lookup_found && (self.lookup_rule_id as usize) < k_max_gbnf_rules && !self.rule_defined[self.lookup_rule_id as usize])
    }

    fn definition_failed(&self) -> Result<bool, ()> { Ok(!self.definition_existing_valid()? && !self.definition_new_valid()?) }
    fn definition_new_valid(&self) -> Result<bool, ()> { Ok(!self.lookup_found && self.lookup_can_insert) }

    fn dispatch_parse_failed_from_deciding(&mut self) -> Result<(), ()> { self.fail() }
    fn dispatch_parse_failed_from_definition_lookup_decision(&mut self) -> Result<(), ()> { self.fail() }
    fn dispatch_parse_failed_from_reference_lookup_decision(&mut self) -> Result<(), ()> { self.fail() }
    fn lookup_definition_candidate(&mut self) -> Result<(), ()> { self.lookup_candidate(); Ok(()) }
    fn lookup_reference_candidate(&mut self) -> Result<(), ()> { self.lookup_candidate(); Ok(()) }

    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_definition_lookup_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_definition_lookup_exec(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_reference_lookup_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_reference_lookup_exec(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> { self.unexpected() }

    fn reference_existing_valid(&self) -> Result<bool, ()> { Ok(self.lookup_found) }
    fn reference_failed(&self) -> Result<bool, ()> { Ok(!self.reference_existing_valid()? && !self.reference_new_valid()?) }
    fn reference_new_valid(&self) -> Result<bool, ()> { Ok(!self.lookup_found && self.lookup_can_insert) }

    fn token_identifier_definition(&self) -> Result<bool, ()> {
        Ok(self.input.has_token && self.input.token_kind == TokenKind::Identifier && self.name().is_some() && self.input.nonterm_mode == ParseMode::Definition && self.error.is_none())
    }

    fn token_identifier_reference(&self) -> Result<bool, ()> {
        Ok(self.input.has_token && self.input.token_kind == TokenKind::Identifier && self.name().is_some() && self.input.nonterm_mode == ParseMode::Reference && self.error.is_none())
    }
}


impl Default for GbnfRuleParserNontermParserActor {
    fn default() -> Self { Self::new() }
}

impl GbnfRuleParserNontermParserActor {
    /// Creates an actor in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self { Self { machine: GbnfRuleParserNontermParserStateMachine::new(Default::default()) } }

    /// Processes one copied request to completion.
    pub fn process_event(&mut self, event: RuleParserEventParseRules) -> ParseOutcome {
        if !self.machine.is(&GbnfRuleParserNontermParserStates::Deciding) {
            self.machine.context_mut().error = Some(NontermParserError::InternalError);
            self.machine.context_mut().outcome = ParseOutcome::InternalError;
            return ParseOutcome::InternalError;
        }
        self.machine.context_mut().set_input(event);
        if self.machine.process_event(GbnfRuleParserNontermParserEvents::RuleParserEventParseRules).is_err() {
            self.machine.context_mut().error = Some(NontermParserError::InternalError);
            self.machine.context_mut().outcome = ParseOutcome::InternalError;
        }
        self.machine.context().outcome
    }

    /// Resolves or inserts a copied nonterminal name.
    pub fn classify(&mut self, mode: ParseMode, name: &str) -> ParseOutcome {
        self.process_event(RuleParserEventParseRules::new(TokenKind::Identifier, mode, name))
    }

    /// Processes an absent token.
    pub fn process_absent(&mut self) -> ParseOutcome { self.process_event(RuleParserEventParseRules::absent()) }

    /// Processes an explicit unexpected event.
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &GbnfRuleParserNontermParserStates { self.machine.state() }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: GbnfRuleParserNontermParserStates) -> bool { self.machine.is(&state) }

    /// Returns the generated machine context for bounded result inspection.
    #[must_use]
    pub fn context(&self) -> &GbnfRuleParserNontermParserContext { self.machine.context() }
}

/// Short alias used by rule-parser callers.
pub type NontermParser = GbnfRuleParserNontermParserActor;
/// Compatibility alias for callers naming the child actor directly.
pub type GbnfNontermParser = GbnfRuleParserNontermParserActor;
