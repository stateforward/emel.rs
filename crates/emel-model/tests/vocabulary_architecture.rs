//! Source-level vocabulary actor architecture gates.

use allocation_counter as _;
use emel_gguf as _;
use emel_io as _;
use emel_model as _;
use emel_tensor as _;
use emel_token as _;
use sml as _;
use std::fs;
use std::path::PathBuf;

fn vocabulary_source(file: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(root.join("src/vocabulary").join(file))
        .expect("vocabulary source must remain available to the architecture gate")
}

fn hparam_source(file: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(root.join("src/loader/hparams").join(file))
        .expect("hparam source must remain available to the architecture gate")
}

#[test]
fn dispatch_local_lengths_are_not_actor_context_fields() {
    let source = vocabulary_source("mod.rs");
    let context = source
        .split_once("struct Context")
        .and_then(|(_, tail)| tail.split_once("pub struct Loader"))
        .map(|(context, _)| context)
        .expect("vocabulary context must remain directly inspectable");

    assert!(
        !context.contains("model_length:"),
        "model length is a dispatch-local result and must travel with the load event"
    );
    assert!(
        !context.contains("pre_length:"),
        "pre-profile length is a dispatch-local result and must travel with the load event"
    );
}

#[test]
fn materialization_phases_are_visible_in_the_state_machine() {
    let source = vocabulary_source("sm.rs");

    for phase in [
        "state_token_type_count",
        "state_tokens",
        "state_scores",
        "state_token_types",
        "state_merges",
        "state_charmap",
        "state_special_id",
        "state_flag",
        "state_patch",
    ] {
        assert!(
            source.contains(phase),
            "missing explicit vocabulary phase `{phase}`"
        );
    }

    let implementation = vocabulary_source("mod.rs");
    assert!(
        !implementation.contains("fn materialize("),
        "a monolithic materialization helper hides routing from the state machine"
    );
    for hidden_router in [
        "read_unsigned_alias",
        "required_string_array_key",
        "optional_string_array_key",
        "optional_float_array_key",
        "optional_integer_array_key",
    ] {
        assert!(
            !implementation.contains(hidden_router),
            "helper `{hidden_router}` hides alias or element-kind routing from SML"
        );
    }
}

#[test]
fn public_queries_delegate_runtime_choices_to_the_query_actor() {
    let implementation = vocabulary_source("mod.rs");
    let query_machine = vocabulary_source("query/sm.rs");

    assert!(implementation.contains("fn query<O: query::Operation>"));
    let query_implementation = vocabulary_source("query/mod.rs");
    for operation in ["WithInfo", "WithToken", "WithMerge", "WithCharmap"] {
        assert!(
            query_implementation.contains(&format!("Operation for {operation}")),
            "query operation `{operation}` must be a statically dispatched child operation"
        );
    }
    for guard in [
        "guard_not_loaded",
        "guard_loaded_valid",
        "guard_loaded_out_of_range",
    ] {
        assert!(
            query_machine.contains(guard),
            "query runtime choice `{guard}` is not visible in SML"
        );
    }
    assert!(query_machine.contains("unexpected_event<_>"));
}

#[test]
fn every_declared_state_has_explicit_unexpected_event_recovery() {
    let source = vocabulary_source("sm.rs");
    let mut states = Vec::new();
    for fragment in source.split('"').skip(1).step_by(2) {
        if fragment.starts_with("state_") && !states.contains(&fragment) {
            states.push(fragment);
        }
    }

    for state in states {
        let recovery = format!("\"{state}\"_s + unexpected_event<_>");
        assert!(
            source.contains(&recovery),
            "state `{state}` has no explicit unexpected-event recovery"
        );
    }
}

#[test]
fn scan_and_copy_helpers_do_not_publish_validation_policy() {
    let source = vocabulary_source("mod.rs");
    for helper in [
        "scan_string_array",
        "scan_byte_array",
        "scan_unsigned_array",
        "copy_tokens",
        "copy_scores",
        "copy_token_types",
        "copy_merges",
        "copy_charmap",
    ] {
        let marker = format!("fn {helper}");
        let body = source
            .split_once(&marker)
            .and_then(|(_, tail)| tail.split_once("\nfn "))
            .map(|(body, _)| body)
            .expect("bounded detail helper must remain directly inspectable");
        for hidden_policy in ["ErrorKind::", "Phase::", "MAX_"] {
            assert!(
                !body.contains(hidden_policy),
                "helper `{helper}` hides validation policy `{hidden_policy}`"
            );
        }
    }

    for helper in [
        "copy_tokens",
        "copy_scores",
        "copy_token_types",
        "copy_merges",
        "copy_charmap",
    ] {
        let marker = format!("fn {helper}");
        let body = source
            .split_once(&marker)
            .and_then(|(_, tail)| tail.split_once("\nfn "))
            .map(|(body, _)| body)
            .expect("copy helper must remain directly inspectable");
        assert!(
            body.contains(".copy") && body.contains(".set("),
            "copy helper `{helper}` must retain the raw typed child result"
        );
        assert!(
            !body.contains("map_err"),
            "copy helper `{helper}` must not collapse a typed child failure"
        );
    }

    let machine = vocabulary_source("sm.rs");
    for state in [
        "state_tokens_copy_decision",
        "state_scores_copy_decision",
        "state_token_types_copy_decision",
        "state_merges_copy_decision",
        "state_charmap_copy_decision",
    ] {
        assert!(
            machine.contains(state),
            "copy result state `{state}` must remain explicit in SML"
        );
    }
    for guard in [
        "guard_copy_success",
        "guard_copy_missing",
        "guard_copy_malformed",
        "guard_copy_wrong_kind",
        "guard_copy_range",
        "guard_copy_count",
        "guard_copy_query",
        "guard_copy_internal",
    ] {
        assert!(
            machine.contains(guard),
            "raw copy outcome `{guard}` must remain explicit in SML"
        );
    }
}

#[test]
fn actions_and_helpers_do_not_classify_query_failures() {
    for (owner, source) in [
        ("vocabulary", vocabulary_source("mod.rs")),
        ("hparams", hparam_source("mod.rs")),
    ] {
        for hidden_classifier in [
            "fn map_query",
            "fn string_error",
            "fn query_error_kind",
            "fn scan_error_kind",
            "fn scan_query_error",
        ] {
            assert!(
                !source.contains(hidden_classifier),
                "{owner} helper `{hidden_classifier}` hides externally observable error routing"
            );
        }

        for action in source.split("    fn effect_").skip(1) {
            let body_end = [
                "\n    fn ",
                "\n    fixed_error_effect!",
                "\n    raw_outcome_guard!",
                "\n    raw_hparam_guard!",
                "\n}",
            ]
            .into_iter()
            .filter_map(|marker| action.find(marker))
            .min()
            .unwrap_or(action.len());
            let body = &action[..body_end];
            assert!(
                !body.contains("QueryError::"),
                "{owner} effect classifies a raw QueryError instead of leaving it to SML guards"
            );
            let kinds = body.match_indices("ErrorKind::").count();
            assert!(
                kinds <= 1,
                "{owner} effect publishes multiple ErrorKind choices instead of one fixed outcome"
            );
        }
    }

    let vocabulary = vocabulary_source("mod.rs");
    let fixed_macro = vocabulary
        .split_once("macro_rules! fixed_error_effect")
        .and_then(|(_, tail)| tail.split_once("\n}\n"))
        .map(|(body, _)| body)
        .expect("fixed error callback macro must remain directly inspectable");
    for hidden_choice in ["match ", "if ", "QueryError::"] {
        assert!(
            !fixed_macro.contains(hidden_choice),
            "fixed error callback macro must not route on `{hidden_choice}`"
        );
    }
    for declaration in vocabulary.match_indices("fixed_error_effect!(") {
        let invocation = vocabulary[declaration.0..]
            .split_once(");")
            .map(|(body, _)| body)
            .expect("fixed error declaration must have a closed invocation");
        assert_eq!(
            invocation.match_indices("ErrorKind::").count(),
            1,
            "every fixed error declaration must publish exactly one compile-time ErrorKind"
        );
    }

    for (owner, source, macro_name) in [
        ("vocabulary", vocabulary, "raw_outcome_guard"),
        ("hparams", hparam_source("mod.rs"), "raw_hparam_guard"),
    ] {
        let marker = format!("macro_rules! {macro_name}");
        let body = source
            .split_once(&marker)
            .and_then(|(_, tail)| tail.split_once("\n}\n"))
            .map(|(body, _)| body)
            .expect("raw outcome guard macro must remain directly inspectable");
        assert!(body.contains("$pattern:pat"));
        assert!(body.contains("matches!(event.$field.get(), $pattern)"));
        for hidden_choice in ["ErrorKind::", "if "] {
            assert!(
                !body.contains(hidden_choice),
                "{owner} raw guard macro must not publish or route on `{hidden_choice}`"
            );
        }
    }
}

#[test]
fn hparam_child_outcomes_are_classified_only_by_sml() {
    let implementation = hparam_source("mod.rs");
    assert!(
        implementation.contains("event.scan.set(ScanOutcome::Query(error))"),
        "nonzero scanning must preserve the original query error"
    );
    assert!(
        implementation.contains(".array_visit") && implementation.contains(".set("),
        "flag visitors must retain their raw typed child result"
    );

    let machine = hparam_source("sm.rs");
    for guard in [
        "guard_unsigned_malformed",
        "guard_unsigned_wrong_kind",
        "guard_unsigned_count",
        "guard_unsigned_query_range",
        "guard_unsigned_query",
        "guard_unsigned_internal",
        "guard_scan_malformed",
        "guard_scan_wrong_kind",
        "guard_scan_count",
        "guard_scan_range",
        "guard_scan_query",
        "guard_scan_internal",
        "guard_array_visit_missing",
        "guard_array_visit_wrong_kind",
        "guard_array_visit_count",
        "guard_array_visit_malformed",
        "guard_array_visit_range",
        "guard_array_visit_query",
        "guard_array_visit_internal",
    ] {
        assert!(
            machine.contains(guard),
            "hparam outcome `{guard}` must remain explicit in SML"
        );
    }
}
use emel_kernels as _;
