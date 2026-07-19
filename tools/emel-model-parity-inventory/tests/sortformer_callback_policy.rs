//! Structural policy proof for the Sortformer GGUF callback boundary.

#[cfg(windows)]
use winapi_util as _;
#[cfg(windows)]
use windows_sys as _;
#[cfg(unix)]
use {cap_std as _, cap_tempfile as _, uuid as _};
use {
    emel_model_parity_inventory as _, proc_macro2 as _, quote as _, serde as _, serde_json as _,
    sha2 as _, shell_words as _, tempfile as _,
};

use syn::visit::{self, Visit};
use syn::{
    Expr, ExprCall, ExprClosure, ExprForLoop, ExprMethodCall, FnArg, ImplItemFn, ItemFn, Pat,
    PatType, Stmt, Type,
};

const ACTOR_SOURCE: &str = include_str!("../../../crates/emel-model/src/sortformer/actor.rs");

fn expression_path_is(expression: &Expr, expected: &[&str]) -> bool {
    let Expr::Path(path) = expression else {
        return false;
    };
    path.path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .eq(expected.iter().copied())
}

fn path_argument_is(expression: &Expr, expected: &str) -> bool {
    expression_path_is(expression, &[expected])
}

fn field_argument_is(expression: &Expr, base: &str, field: &str, mutable: bool) -> bool {
    let expression = if let Expr::Reference(reference) = expression {
        if reference.mutability.is_some() != mutable {
            return false;
        }
        reference.expr.as_ref()
    } else {
        return false;
    };
    let Expr::Field(field_expression) = expression else {
        return false;
    };
    path_argument_is(&field_expression.base, base)
        && matches!(&field_expression.member, syn::Member::Named(member) if member == field)
}

const fn closure_expression(expression: &Expr) -> &Expr {
    if let Expr::Block(block) = expression
        && let [Stmt::Expr(expression, None)] = block.block.stmts.as_slice()
    {
        return expression;
    }
    expression
}

fn closure_inputs_are(closure: &ExprClosure, expected: &[Option<&str>]) -> bool {
    closure.inputs.len() == expected.len()
        && closure
            .inputs
            .iter()
            .zip(expected)
            .all(|(pattern, expected)| match (pattern, expected) {
                (Pat::Wild(_), None) => true,
                (Pat::Type(PatType { pat, .. }), None) => matches!(pat.as_ref(), Pat::Wild(_)),
                (Pat::Ident(pattern), Some(expected)) => pattern.ident == *expected,
                (Pat::Type(PatType { pat, .. }), Some(expected)) => {
                    matches!(pat.as_ref(), Pat::Ident(pattern) if pattern.ident == *expected)
                }
                _ => false,
            })
}

fn name_length_callback_is_constrained(closure: &ExprClosure) -> bool {
    if !closure_inputs_are(closure, &[Some("name"), None, None]) {
        return false;
    }
    matches!(
        closure_expression(&closure.body),
        Expr::MethodCall(call)
            if call.method == "len"
                && call.args.is_empty()
                && path_argument_is(&call.receiver, "name")
    )
}

fn capture_callback_is_constrained(closure: &ExprClosure) -> bool {
    if !closure_inputs_are(
        closure,
        &[Some("name"), Some("descriptor"), Some("payload")],
    ) {
        return false;
    }
    let Expr::Call(call) = closure_expression(&closure.body) else {
        return false;
    };
    expression_path_is(&call.func, &["capture_source_observation"])
        && call.args.len() == 5
        && field_argument_is(&call.args[0], "storage", "observation_names", true)
        && path_argument_is(&call.args[1], "offset")
        && path_argument_is(&call.args[2], "name")
        && path_argument_is(&call.args[3], "descriptor")
        && path_argument_is(&call.args[4], "payload")
}

#[derive(Default)]
struct ActorItems<'ast> {
    load: Option<&'ast ImplItemFn>,
    capture: Option<&'ast ItemFn>,
}

impl<'ast> Visit<'ast> for ActorItems<'ast> {
    fn visit_impl_item_fn(&mut self, function: &'ast ImplItemFn) {
        if function.sig.ident == "load" {
            self.load = Some(function);
        }
        visit::visit_impl_item_fn(self, function);
    }

    fn visit_item_fn(&mut self, function: &'ast ItemFn) {
        if function.sig.ident == "capture_source_observation" {
            self.capture = Some(function);
        }
        visit::visit_item_fn(self, function);
    }
}

#[derive(Default)]
struct TensorCallbacks<'ast> {
    callbacks: Vec<&'ast ExprClosure>,
}

impl<'ast> Visit<'ast> for TensorCallbacks<'ast> {
    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        if expression_path_is(&call.func, &["WithTensor", "new"])
            && let Some(Expr::Closure(callback)) = call.args.iter().nth(1)
        {
            self.callbacks.push(callback);
        }
        visit::visit_expr_call(self, call);
    }
}

struct CaptureBodyPolicy {
    valid: bool,
}

impl CaptureBodyPolicy {
    const ALLOWED_METHODS: [&'static str; 6] = [
        "checked_add",
        "copy_from_slice",
        "get_mut",
        "is_empty",
        "len",
        "ok_or",
    ];
}

impl<'ast> Visit<'ast> for CaptureBodyPolicy {
    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        if !expression_path_is(&call.func, &["Ok"]) {
            self.valid = false;
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast ExprMethodCall) {
        if !Self::ALLOWED_METHODS
            .iter()
            .any(|allowed| call.method == *allowed)
        {
            self.valid = false;
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_closure(&mut self, _: &'ast ExprClosure) {
        self.valid = false;
    }

    fn visit_expr_macro(&mut self, _: &'ast syn::ExprMacro) {
        self.valid = false;
    }

    fn visit_expr_unsafe(&mut self, _: &'ast syn::ExprUnsafe) {
        self.valid = false;
    }
}

fn type_has_actor(type_: &Type) -> bool {
    struct ActorType {
        found: bool,
    }
    impl<'ast> Visit<'ast> for ActorType {
        fn visit_type_path(&mut self, path: &'ast syn::TypePath) {
            self.found |= path.path.segments.iter().any(|segment| {
                matches!(
                    segment.ident.to_string().as_str(),
                    "Catalog" | "GgufLoader" | "Loader" | "Sortformer"
                )
            });
            visit::visit_type_path(self, path);
        }
    }
    let mut visitor = ActorType { found: false };
    visitor.visit_type(type_);
    visitor.found
}

fn capture_signature_is_actor_free(function: &ItemFn) -> bool {
    function.sig.inputs.len() == 5
        && function.sig.inputs.iter().all(|argument| match argument {
            FnArg::Typed(argument) => !type_has_actor(&argument.ty),
            FnArg::Receiver(_) => false,
        })
}

#[derive(Default)]
struct Dispatches<'ast> {
    loader_with_tensor: Vec<&'ast ExprMethodCall>,
    catalog_find_tensor: Vec<&'ast ExprMethodCall>,
}

impl<'ast> Visit<'ast> for Dispatches<'ast> {
    fn visit_expr_method_call(&mut self, call: &'ast ExprMethodCall) {
        if call.method == "process_event" && call.args.len() == 1 {
            if path_argument_is(&call.receiver, "loader")
                && matches!(call.args.first(), Some(Expr::Call(event)) if expression_path_is(&event.func, &["WithTensor", "new"]))
            {
                self.loader_with_tensor.push(call);
            }
            if path_argument_is(&call.receiver, "catalog")
                && matches!(call.args.first(), Some(Expr::Call(event)) if expression_path_is(&event.func, &["FindTensor", "new"]))
            {
                self.catalog_find_tensor.push(call);
            }
        }
        visit::visit_expr_method_call(self, call);
    }
}

#[derive(Default)]
struct ForLoops<'ast> {
    loops: Vec<&'ast ExprForLoop>,
}

impl<'ast> Visit<'ast> for ForLoops<'ast> {
    fn visit_expr_for_loop(&mut self, expression: &'ast ExprForLoop) {
        self.loops.push(expression);
        visit::visit_expr_for_loop(self, expression);
    }
}

fn catalog_dispatch_follows_loader_return(load: &ImplItemFn) -> bool {
    let mut loops = ForLoops::default();
    loops.visit_block(&load.block);
    let mut matching_loops = 0usize;
    for expression in loops.loops {
        let mut loader_statement = None;
        let mut catalog_statement = None;
        let mut loader_calls = 0usize;
        let mut catalog_calls = 0usize;
        for (index, statement) in expression.body.stmts.iter().enumerate() {
            let mut dispatches = Dispatches::default();
            dispatches.visit_stmt(statement);
            loader_calls += dispatches.loader_with_tensor.len();
            catalog_calls += dispatches.catalog_find_tensor.len();
            if !dispatches.loader_with_tensor.is_empty() {
                loader_statement = Some(index);
            }
            if !dispatches.catalog_find_tensor.is_empty() {
                catalog_statement = Some(index);
            }
        }
        if loader_calls == 1 && catalog_calls == 1 {
            matching_loops += 1;
            if !matches!((loader_statement, catalog_statement), (Some(loader), Some(catalog)) if loader < catalog)
            {
                return false;
            }
        }
    }
    matching_loops == 1
}

fn actor_callback_policy(source: &str) -> Result<(), &'static str> {
    let syntax = syn::parse_file(source).map_err(|_| "actor source must parse")?;
    let mut items = ActorItems::default();
    items.visit_file(&syntax);
    let load = items.load.ok_or("Sortformer::load must exist")?;
    let capture = items
        .capture
        .ok_or("capture_source_observation must exist")?;

    let mut callbacks = TensorCallbacks::default();
    callbacks.visit_block(&load.block);
    if callbacks.callbacks.len() != 2
        || !name_length_callback_is_constrained(callbacks.callbacks[0])
        || !capture_callback_is_constrained(callbacks.callbacks[1])
    {
        return Err("WithTensor callbacks must keep their exact actor-free shape");
    }
    if !capture_signature_is_actor_free(capture) {
        return Err("capture helper types must not admit an actor");
    }
    let mut capture_policy = CaptureBodyPolicy { valid: true };
    capture_policy.visit_block(&capture.block);
    if !capture_policy.valid {
        return Err("capture helper must use only allocation-free primitive operations");
    }
    if !catalog_dispatch_follows_loader_return(load) {
        return Err("catalog dispatch must follow the completed Loader dispatch");
    }
    Ok(())
}

fn mutate_capture_callback_callee(source: &str, replacement: &str) -> String {
    let callback_start = source
        .find("|name: &[u8], descriptor, payload: &[u8]|")
        .expect("capture callback must exist");
    let (prefix, callback) = source.split_at(callback_start);
    let callback = callback.replacen("capture_source_observation", replacement, 1);
    format!("{prefix}{callback}")
}

#[test]
fn callback_ast_enforces_actor_free_capture_and_sequential_dispatch() {
    actor_callback_policy(ACTOR_SOURCE).unwrap();
}

#[test]
fn callback_ast_policy_rejects_forbidden_mutations() {
    let allocation = ACTOR_SOURCE.replacen(".copy_from_slice(name);", ".to_vec();", 1);
    assert!(actor_callback_policy(&allocation).is_err());

    let direct_dispatch = mutate_capture_callback_callee(ACTOR_SOURCE, "loader.process_event");
    assert!(actor_callback_policy(&direct_dispatch).is_err());

    let indirect_dispatch = mutate_capture_callback_callee(ACTOR_SOURCE, "dispatch_catalog");
    assert!(actor_callback_policy(&indirect_dispatch).is_err());

    let catalog_lookup = mutate_capture_callback_callee(ACTOR_SOURCE, "catalog.process_event");
    assert!(actor_callback_policy(&catalog_lookup).is_err());
}
