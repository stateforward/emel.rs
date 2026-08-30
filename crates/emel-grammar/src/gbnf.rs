//! Fixed-capacity GBNF grammar representation.
//!
//! This module mirrors the storage and lookup contract in the pinned
//! `emel.cpp/src/emel/gbnf/detail.hpp`.  The C++ pointer-bearing rule view is
//! represented with an optional borrowed slice so invalid rules cannot create
//! an invalid pointer while retaining the same null-and-zero result.

#![allow(non_camel_case_types)]

/// GBNF element type mappings from `llama.cpp`'s `llama_gretype`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum element_type {
    #[default]
    end = 0,
    alt = 1,
    rule_ref = 2,
    character = 3,
    char_not = 4,
    char_rng_upper = 5,
    char_alt = 6,
    char_any = 7,
    token = 8,
    token_not = 9,
}

/// GBNF AST element equivalent to `llama_grammar_element`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct element {
    /// Element kind.
    pub r#type: element_type,
    /// Unicode code point, rule ID, or token ID.
    pub value: u32,
}

/// Maximum number of rules stored in a grammar.
pub const k_max_gbnf_rules: usize = 2048;
/// Maximum number of elements stored across all rules.
pub const k_max_gbnf_elements: usize = 65_536;
/// Maximum number of elements in one rule.
pub const k_max_gbnf_rule_elements: usize = 4096;
/// Maximum number of symbols in the parser symbol table.
pub const k_max_gbnf_symbols: usize = 2048;
/// Number of slots in the parser symbol table.
pub const k_gbnf_symbol_table_slots: usize = 4096;

const _: () = assert!(k_gbnf_symbol_table_slots & (k_gbnf_symbol_table_slots - 1) == 0);
const _: () = assert!(k_max_gbnf_rule_elements <= k_max_gbnf_elements);

/// A borrowed view of one grammar rule.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct rule_view<'a> {
    /// Elements in the rule, or `None` for an invalid or empty rule.
    pub elements: Option<&'a [element]>,
    /// Number of elements in the rule.  Invalid and empty rules report zero.
    pub length: u32,
}

/// Fixed-capacity grammar storage.
#[derive(Debug)]
pub struct grammar {
    /// Contiguous storage for every rule's elements.
    pub elements: [element; k_max_gbnf_elements],
    /// Starting element offset for each rule ID.
    pub rule_offsets: [u32; k_max_gbnf_rules],
    /// Element count for each rule ID.
    pub rule_lengths: [u32; k_max_gbnf_rules],
    /// Number of currently defined rules.
    pub rule_count: u32,
    /// Number of currently used elements in [`Self::elements`].
    pub element_count: u32,
}

impl Default for grammar {
    fn default() -> Self {
        Self {
            elements: [element::default(); k_max_gbnf_elements],
            rule_offsets: [0; k_max_gbnf_rules],
            rule_lengths: [0; k_max_gbnf_rules],
            rule_count: 0,
            element_count: 0,
        }
    }
}

impl grammar {
    /// Clears active rule metadata and resets the used element count.
    pub fn reset(&mut self) {
        // Valid grammar states always keep rule_count within capacity.  The
        // upper bound also keeps reset safe if a caller has modified the
        // public counters before recovery.
        let active_rules = (self.rule_count as usize).min(k_max_gbnf_rules);
        self.rule_offsets[..active_rules].fill(0);
        self.rule_lengths[..active_rules].fill(0);
        self.rule_count = 0;
        self.element_count = 0;
    }

    /// Returns a borrowed rule view, or a null-equivalent empty view when the
    /// ID, length, or element range is invalid.
    #[must_use]
    pub fn rule(&self, rule_id: u32) -> rule_view<'_> {
        let index = rule_id as usize;
        if index >= self.rule_count as usize || index >= k_max_gbnf_rules {
            return rule_view::default();
        }

        let length = self.rule_lengths[index];
        if length == 0 {
            return rule_view::default();
        }

        let offset = self.rule_offsets[index] as usize;
        let length = length as usize;
        let element_count = self.element_count as usize;
        let Some(end) = offset.checked_add(length) else {
            return rule_view::default();
        };
        if end > element_count || end > k_max_gbnf_elements {
            return rule_view::default();
        }

        rule_view {
            elements: Some(&self.elements[offset..end]),
            length: length as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_type_values_match_pinned_mapping() {
        assert_eq!(element_type::end as u32, 0);
        assert_eq!(element_type::alt as u32, 1);
        assert_eq!(element_type::rule_ref as u32, 2);
        assert_eq!(element_type::character as u32, 3);
        assert_eq!(element_type::char_not as u32, 4);
        assert_eq!(element_type::char_rng_upper as u32, 5);
        assert_eq!(element_type::char_alt as u32, 6);
        assert_eq!(element_type::char_any as u32, 7);
        assert_eq!(element_type::token as u32, 8);
        assert_eq!(element_type::token_not as u32, 9);
    }

    #[test]
    fn rule_returns_borrowed_bounded_slice() {
        let mut grammar = grammar::default();
        grammar.elements[3] = element {
            r#type: element_type::character,
            value: 65,
        };
        grammar.elements[4] = element {
            r#type: element_type::token,
            value: 7,
        };
        grammar.rule_offsets[2] = 3;
        grammar.rule_lengths[2] = 2;
        grammar.rule_count = 3;
        grammar.element_count = 5;

        let view = grammar.rule(2);
        assert_eq!(view.length, 2);
        assert_eq!(view.elements, Some(&grammar.elements[3..5]));
    }

    #[test]
    fn invalid_rule_lookup_is_empty_for_id_length_and_bounds() {
        let mut grammar = grammar::default();
        grammar.rule_count = 1;
        grammar.element_count = 4;

        assert_eq!(grammar.rule(1), rule_view::default());
        assert_eq!(grammar.rule(u32::MAX), rule_view::default());

        grammar.rule_lengths[0] = 0;
        assert_eq!(grammar.rule(0), rule_view::default());

        grammar.rule_offsets[0] = 3;
        grammar.rule_lengths[0] = 2;
        assert_eq!(grammar.rule(0), rule_view::default());
    }

    #[test]
    fn reset_clears_active_metadata_and_counts() {
        let mut grammar = grammar::default();
        grammar.rule_offsets[0] = 8;
        grammar.rule_lengths[0] = 4;
        grammar.rule_count = 1;
        grammar.element_count = 12;

        grammar.reset();

        assert_eq!(grammar.rule_count, 0);
        assert_eq!(grammar.element_count, 0);
        assert_eq!(grammar.rule_offsets[0], 0);
        assert_eq!(grammar.rule_lengths[0], 0);
    }
}
