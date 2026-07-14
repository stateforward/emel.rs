# AGENTS.md

These rules define the engineering contract for `emel.rs`. They apply to the
entire Cargo workspace unless a more specific `AGENTS.md` exists below this
directory.

## stateforward-sml actor model

ALWAYS use the `stateforward-sml` crate through its `sml` package alias for
orchestration state machines.

ALWAYS keep dispatch run-to-completion (RTC) and single-writer per actor.

NEVER use `defer`, `process(...)`, `sml::utility::EventQueue`,
`sml::utility::EventQueues`, a mailbox, or any post-for-later mechanism in an
EMEL actor.

ALWAYS treat async dispatch semantically. An async `process_event` MAY remain
RTC when the actor or owning orchestrator drives and observes completion within
that dispatch boundary.

NEVER let futures, spawned tasks, scheduler work items, callbacks, or other
continuations escape an RTC boundary as hidden deferred work.

NEVER call an actor's own `process_event` from guards, actions, entry handlers,
or exit handlers.

ALWAYS model internal multi-step flows with `completion<TEvent>`, anonymous
transitions, entry actions, or explicit internal result events.

ALWAYS keep anonymous transition graphs acyclic or statically bounded.

ALWAYS propagate originating event data across internal phases with typed
completion transitions when the later phase needs it.

NEVER use completion or anonymous transitions as data-plane iteration loops,
including per-logit, per-token, per-tensor, or per-element scans.

ALWAYS keep completion chains phase-level, with a small statically bounded
number of transitions per top-level dispatch.

ALWAYS implement bulk numeric iteration in allocation-free action or detail
kernels within one transition per phase.

NEVER copy an event payload into context only to bridge internal phases.

ALWAYS keep guards pure predicates of `(event, context)` with no side effects.

ALWAYS keep actions bounded during dispatch.

ONLY allow an action to await already-submitted child actor dispatches when all
children join before the action returns, RTC is preserved, the same actor is
not re-entered, and no deferred work remains.

ALWAYS keep hot-path actions allocation-free.

ALWAYS perform permitted one-time construction or initialization allocation
before `process_event(...)` dispatch.

NEVER allocate during dispatch, including in guards, actions, entry or exit
handlers, and completion progress.

ALWAYS model runtime behavior choices as explicit guards or explicit choice
states and transitions.

NEVER hide runtime behavior selection in actions, context methods, state
machine wrappers, closures, or helpers called from them.

Treat runtime behavior selection semantically, not syntactically. It includes
helper-local branching, lookup tables, flag dispatch, dtype dispatch, backend
dispatch, modality dispatch, model-family dispatch, fallback selection,
block-kind selection, activation selection, skip or residual selection,
buffer-lane selection, callback selection, error-channel selection, and any
other runtime choice that changes the algorithm, path, variant, or externally
observable behavior.

NEVER put runtime control branching (`if`, `else`, `match`, boolean short
circuiting used as dispatch, or runtime-indexed handler tables) in actions,
context methods used as actions, or helpers called from actions.

NEVER put validation-path branching in actions or helpers called from actions.
Model validation outcomes with guards and explicit transitions.

NEVER disguise runtime branching as a single-pass loop or a runtime-indexed
candidate or handler array.

ALWAYS use loops in actions and detail code only for data-plane iteration with
monotonic progress and bounded work.

Compile-time selection through generics, const evaluation, and `cfg` is allowed
inside actions, context methods, and their helpers.

NEVER perform I/O waits, lock waits, blocking channel operations, or sleeps in
guards or actions.

ALWAYS inject time through event payloads. NEVER read wall-clock time in guards
or actions.

ALWAYS keep public events immutable and small. Prefer owned `Copy` payloads when
copies are cheap and borrowing would complicate the API.

Internal-only events MAY carry mutable borrows for synchronous same-RTC
handoff.

NEVER expose a mutable internal-event payload through public API types.

NEVER retain a mutable internal-event payload beyond its top-level dispatch.

NEVER put owning heap containers in events unless allocation is completed
before dispatch and the move is demonstrably safe for the hot path.

ALWAYS validate runtime event IDs before constructing an
`sml::utility::DispatchTable`.

ALWAYS give each component a local context and pass it to the generated state
machine constructor.

ALWAYS use generated state inspection such as `state()`, `states()`, `is(...)`,
or `visit_current_state(...)`.

NEVER allow re-entrancy into the same actor within one RTC chain.

ALWAYS keep synchronous cross-actor calls acyclic and deterministically
ordered.

Callbacks are allowed only for immediate synchronous replies within the same
RTC chain. Invoke them before dispatch returns, never retain them in context,
and never call `process_event` from a callback.

ALWAYS define explicit behavior for unexpected events. NEVER drop them
silently.

ALWAYS use `unexpected_event<T>` or `unexpected_event<_>` for unexpected-event
handling.

ALWAYS reproduce a reported bug with a failing test before implementing the
fix.

ALWAYS keep tracing deterministic, bounded, allocation-free on hot paths, and
disabled unless requested.

## architecture and composition

ALWAYS define orchestration tables with `sml!` and keep generated machine,
event, state, and context-trait types local to the owning component until the
implementation is ready as public API.

ALWAYS write transition rows in destination-first form:
`destination <= source + Event [guard] / action`.

NEVER introduce source-first rows in new or modified transition tables.

ALWAYS keep the destination state and `<=` on the same line.

ALWAYS organize large transition tables into visible phase sections separated
by blank lines or divider comments.

ALWAYS use one transition row per line and retain trailing commas.

Use `sml!` directly. NEVER hide transition rows behind additional macros or
code generation layers without explicit user approval.

ALWAYS keep canonical machine modules under the owning component's `sm` module.

ALWAYS map crate and module layout to ownership boundaries.

ALWAYS limit component modules to clear roles such as `any`, `context`,
`actions`, `guards`, `errors`, `events`, `sm`, or `detail` when the component is
large enough to require separation.

ALWAYS structure variant families as `<domain>/<component>/<variant>`, for
example `memory/coordinator/kv` or `text/tokenizer/preprocessor/bpe`.

ALWAYS colocate a machine definition, its context, guards, actions, and events
within the same component module tree.

NEVER place orchestration logic in data-only modules.

ALWAYS treat crates and top-level domain modules as ownership boundaries, not
naming decoration.

A domain implementation MAY depend inward on shared kernel, model, text,
speech, memory, I/O, tensor, or core primitives. Shared crates MUST NOT depend
on or name downstream runtime internals.

NEVER create top-level crates or runtime modules for model families unless the
user explicitly approves them. Keep a model family under the owning domain and
variant route.

NEVER expose model-family contracts from generic speech, text, generator,
recognizer, event, or context APIs.

ALWAYS construct variant contracts at the variant boundary after model-binding
validation.

ALWAYS name variant-specific routes with the variant or domain name when they
directly depend on variant-specific tokenizer, model, or detail contracts.

ALWAYS add or maintain dependency-boundary checks when adding or moving
domain-specific code. Use `cargo tree`, `rg`, and focused compile tests to prove
the intended direction of dependencies.

ALWAYS put runtime behavior choices in `sm.rs` transitions with predicates
implemented by guard callbacks.

NEVER put runtime behavior choices in action callbacks or `detail.rs` helpers.

ALWAYS treat guards as the home for runtime predicates that decide which
transition or behavior path is taken.

ALWAYS treat actions as bounded execution of an already-chosen behavior path.

ALWAYS use `state_`, `event_`, `guard_`, `effect_`, `enter_`, and `exit_`
prefixes for new transition-table aliases and callbacks when the generated API
permits those names.

NEVER rename established public state or event types solely to satisfy this
convention. Apply it to new symbols and code already being refactored.

ALWAYS keep `detail.rs` helpers private, non-routing, and non-orchestrating.

ONLY extract a detail helper when it is reused or when extraction materially
clarifies a bounded data-plane algorithm. Otherwise keep logic in its owner.

Helpers called from actions MUST NOT select behavior, route fallbacks, choose
modes, choose success or error paths, or decide what runs next.

If a helper inspects runtime dtype, backend support, architecture, model name,
tensor layout, modality, block flags, activation kind, or scratch lane to choose
which computation runs, that helper is choosing behavior and belongs behind
explicit state-machine guards and transitions.

Data-plane branching in detail code is allowed only within an already-chosen
algorithm for numeric work, bounds handling, padding, clamping, parsing, or
monotonic loop progress. It MUST NOT select an algorithm, variant, backend, or
behavior family.

Typed result events MAY carry a detail operation's success or error outcome
back to the machine. Every externally observable outcome MUST have an explicit
guard and transition; no catch-all fallback may conceal it.

Superficial relocation does not satisfy these rules. Moving a runtime choice
from an action into a helper, iterator combinator, lookup table, closure, or
context method is still hidden control logic.

Shared non-routing helpers SHOULD use truthful verbs such as `compute_`,
`validate_`, `bind_`, `scan_`, `append_`, or `reset_`. NEVER use routing or
selection verbs for non-guard helpers.

ALWAYS give each machine its own `process_event` wrapper and owned context.

Share behavior through ordinary private functions or explicitly shared generic
components, never through inheritance-like trait hierarchies that hide dispatch.

ALWAYS keep child-machine data owned by its parent and borrow parent-owned data
explicitly where composition requires it.

ALWAYS communicate between machines through events and explicit interfaces.

NEVER call another machine's actions or guards directly.

NEVER mutate another machine's context directly.

NEVER expose or consume another component's loader, context, storage, parsed
record, guard, action, or detail internals across crate boundaries. Cross-crate
runtime interaction MUST use the component's public actors and typed events.

The `emel-gguf` public surface MUST remain limited to its exported `Loader`
actor and `event` module. Other crates and tools MUST dispatch GGUF events to
the loader actor and MUST NOT receive visibility into the private loader
implementation. NEVER expose actor state inspection across crate boundaries;
consumers MUST infer progress from typed event outcomes.

ALWAYS dispatch cross-machine events through the owning machine's public
`process_event` wrapper.

ALWAYS keep operator arithmetic, lowering, packing, quantization,
dequantization, and backend-specific numeric work in `emel-kernels` or an
explicitly component-owned kernel module.

ALWAYS keep higher layers limited to orchestration, metadata shaping, buffer
binding, and dispatch into kernels.

NEVER add ad hoc compute, backend-specialized loops, lowering, or packing logic
to generators, planners, loaders, graph orchestration, wrappers, tests, or tools
to bridge a missing kernel path.

If ad hoc compute exists outside kernel ownership, migrate it into the owning
kernel surface instead of extending it.

ALWAYS land new or changed operators in kernel-owned modules first. Other
changes may wire, dispatch, validate, benchmark, or prove those kernels.

NEVER duplicate behavior used in more than one place; move it to the narrowest
shared owner that preserves dependency direction.

## events, outcomes, and errors

ALWAYS name trigger events with noun-like domain-action names and no `cmd_`
prefix.

ALWAYS give internal outcome events explicit `Done` and `Error` suffixes.

Internal outcome events MAY carry mutable borrows when they never cross the
component boundary and remain within one RTC dispatch.

ALWAYS use non-optional fields for required event data.

ONLY use `Option<T>` for semantically optional or nullable fields.

NEVER use `cmd_`-prefixed event names.

ALWAYS model failures with explicit error states and error outcome events.

NEVER add synthetic fault-injection controls to production events or actions.

NEVER add test-only fields or variants to production event types.

ALWAYS encode retries and one-shot attempts in transitions, not mutable context
flags.

NEVER mirror explicit state or event outcomes into redundant context status
fields.

NEVER retain per-invocation output borrows in machine context.

ALWAYS use typed Rust error enums and `Result` at fallible public boundaries.

NEVER collapse distinct observable failure classes into strings or booleans.

## context rules

ALWAYS define a component-local context type.

ALWAYS mutate context in actions or internal transitions when persistent actor
state must change.

NEVER mutate context in guards.

NEVER read or write context directly from state-machine wrapper methods to make
behavior choices.

ALWAYS keep context focused on actor-owned runtime data required across
top-level dispatch calls.

NEVER store dispatch-local data in context. This includes current event
borrows, output borrows, phase flags, step indexes, temporary counts, and
transient errors or status values.

ALWAYS pass per-dispatch data across phases with typed internal events or typed
completion payloads.

If a machine has no persistent actor-owned state, its context MUST be an empty
struct.

NEVER mirror event fields into context only for phase handoff.

NEVER store orchestration phase, attempt, or failure flags in context.

## API boundaries and safety

ALWAYS expose idiomatic, ownership-safe Rust APIs.

ALWAYS use fixed-width integers for serialized formats, stable wire contracts,
and foreign-function boundaries.

ALWAYS return typed `Result` values for recoverable failures.

ALWAYS keep public types free of implementation-only generated machine details.

NEVER expose mutable internal state through public references.

NEVER use `unsafe` unless a safe implementation cannot satisfy the required
contract and the user has approved the boundary.

Every `unsafe` block MUST document its safety invariants and have focused tests
that exercise the boundary.

The sole currently approved unsafe exception is the private target-gated
`crates/emel-io/src/mmap/platform/` module required for true native file-backed
mapping. This exception is not precedent or standing authority for unsafe code
in any other crate, module, actor, test, tool, or future implementation.

Within that one module, unsafe is limited to native Unix map, unmap, and advise
calls; native Windows mapping-handle, map, unmap, and prefetch calls; checked
pointer arithmetic; and one immutable `slice::from_raw_parts` conversion.
ALWAYS keep the crate deny-by-default for unsafe code and allow it only on that
private module.

The mmap exception requires private ownership of every native resource; no raw
pointer, native handle, mutable slice, or platform operation in public API; no
aliasing, re-entry, or escaped mapped view; exclusive actor access for release;
and an unchanged, untruncated mapped file for the complete mapping lifetime.
Every unsafe operation MUST state which invariant makes that exact operation
sound. Target-gated focused tests, hostile-input tests, teardown and partial-
release tests, and a fresh blind review MUST cover the boundary.

NEVER infer permission for a read-into-`Vec<u8>` fallback, another unsafe
module, a wider lint allowance, or a public raw-pointer API from this exception.
Any change to this exact unsafe surface or its invariants requires new explicit
user approval and an `AGENTS.md` update before source implementation.

ALWAYS isolate any foreign-function integration in a dedicated boundary crate
or module. Safe workspace crates MUST consume a safe wrapper.

## performance and allocation

ALWAYS treat performance as a top-level requirement.

NEVER use trait-object dynamic dispatch in inference hot paths unless benchmarks
justify it and the user approves it.

ALWAYS prefer static dispatch through generics, enums, and explicit state
transitions in hot paths.

NEVER allocate during inference, sampling, or actor dispatch hot paths.

NEVER use heap allocation by default when bounded stack, caller-provided, or
reusable owned storage is practical.

ALWAYS permit one-time heap allocation during construction, initialization,
model loading, or other non-hot-path setup when required.

ALWAYS reuse unavoidable heap allocations.

ALWAYS document the rationale for unavoidable hot-adjacent allocation.

ALWAYS keep telemetry optional and non-blocking.

NEVER use panics for expected control flow.

NEVER rely on unwinding inside state-machine guards or actions.

ALWAYS keep actor models independent. Do not share one generated machine
instance between actors unless the user explicitly authorizes it.

## naming, style, and portability

ALWAYS follow standard Rust naming: `snake_case` for functions, variables, and
modules; `PascalCase` for types and traits; and `SCREAMING_SNAKE_CASE` for
constants and statics.

ALWAYS use `cargo fmt` formatting and keep prose and comments near 100 columns
when practical.

ALWAYS keep code portable across Linux, macOS, and Windows.

NEVER use platform-specific APIs without an abstraction and a portable fallback
or an explicitly gated target implementation.

ALWAYS write repository shell scripts for portable Unix environments and keep
platform-specific runtime behavior inside Rust where it can be target-gated and
tested.

ALWAYS honor the workspace's `rust-version` and edition declarations.

## build, tests, and CI gates

ALWAYS work on a feature branch and submit changes through a pull request.

NEVER push directly to `main` unless the user explicitly requests it.

ALWAYS make atomic commits. Each commit MUST contain one coherent change with
its directly related tests, documentation, fixtures, and quality-gate updates.

NEVER mix unrelated refactors, features, tooling, generated artifacts, or
policy changes in the same commit.

ALWAYS leave every commit buildable and with its relevant tests passing so it
can be reviewed, reverted, or cherry-picked independently.

ALWAYS use the pinned Rust toolchains and locked dependency graph.

NEVER run more than one build, quality-gate, fuzz, or benchmark campaign at the
same time on a shared host unless their resource use is known to be isolated.

ALWAYS use ordinary Rust unit and integration tests.

ALWAYS use generated SML state inspection for machine assertions.

ALWAYS name integration test files by machine, component, or domain.

NEVER create monolithic test files.

ALWAYS scope each test file to one machine, system, or behavior family.

ALWAYS keep benchmark cases focused and dependency-light.

ALWAYS keep snapshot baselines under `snapshots/`.

FOR THE CURRENT `emel.cpp`-to-`emel.rs` port, until the user declares snapshot
baselines frozen, ALWAYS create or update the relevant parity and benchmark
baselines without requesting separate consent.

ALWAYS record the source identity, fixture or benchmark configuration, and
validation evidence for every created or updated baseline.

ALWAYS include a created or updated baseline in the same atomic commit as the
behavior, fixture, benchmark, or parity case that requires it.

AFTER the user declares snapshot baselines frozen, NEVER accept or update them
without explicit user consent.

ALWAYS hard-fail when a required quality tool is missing.

ALWAYS enforce line coverage of at least 90% and branch coverage of at least
50% on every completed production crate in the coverage surface.

ALWAYS expand `scripts/coverage.sh` when another scaffold crate becomes a
maintained implementation.

ALWAYS run the relevant local quality gates after an implementation change:

```sh
cargo fmt --all -- --check
cargo lint
cargo test-all
scripts/coverage.sh
scripts/paritychecker.sh --snapshot-only
```

Run the isolated fuzz checks when parser, loader, binary format, or unsafe input
handling changes:

```sh
cargo clippy --manifest-path fuzz/Cargo.toml --bins -- -D warnings
scripts/fuzz.sh --seconds 10
```

Run `scripts/bench.sh --snapshot` when a hot path or performance-sensitive
dependency changes.

NEVER weaken coverage, parity, benchmark, fuzz, formatting, lint, or
documentation gates to save time.

If a gate is irrelevant to a scoped change, it may be skipped locally, but any
relevant gate MUST run against the maintained implementation path before
closeout.

ALWAYS run the complete relevant gate set before milestone closeout, release
readiness, or cross-cutting changes.

## reference policy

ALWAYS treat maintained stateforward-sml machines under `crates/` as the single
source of truth for architecture and orchestration.

ALWAYS treat milestone claims about runtime, fixtures, contracts, parity,
benchmarks, or publication as source-backed claims rather than document-backed
claims.

ALWAYS trace the actual code path from the pinned fixture or model contract
through the maintained loader and builder into runtime, parity, benchmark, and
publication entry points before making a closeout claim.

NEVER mark a requirement or milestone satisfied from roadmap, state, summary,
verification, validation, or generated artifacts alone when the claim depends
on which runtime path or fixture actually executes.

If inspection shows that a synthetic, fabricated, tool-only, or test-only path
feeds a claimed maintained runtime, treat the claim as unsatisfied until the
maintained path is wired end to end.

ALWAYS use `crates/emel-gguf/src/loader` as the current architectural reference
for explicit state-machine orchestration until a more complete canonical domain
is established.

NEVER maintain parallel machine-definition specifications in documentation.

ALWAYS document state purpose, key invariants, guard semantics, and action side
effects next to the maintained implementation.

ALWAYS treat the pinned reference implementation as the functional logic
reference for allocator, file-format, kernel, and behavioral parity work.

NEVER copy reference control flow, branching structure, lifecycle semantics, or
orchestration decisions into EMEL runtime code.

ALWAYS define EMEL behavior and orchestration in stateforward-sml machines as
the source of truth.

ALWAYS port reference arithmetic, kernels, format interpretation, and
instruction behavior into EMEL-owned Rust when implementing equivalent
functionality.

ALWAYS preserve or improve performance when porting reference logic.

ALWAYS implement equivalent functionality natively without runtime linkage to
llama.cpp or ggml.

NEVER link an EMEL crate against llama.cpp or ggml. Reference integration is
allowed only in dedicated benchmark and parity tooling.

ALWAYS confine llama.cpp and ggml execution to the explicit reference-side
process under `tools/llama-gguf-reference` or another user-approved reference
tool.

NEVER let the EMEL side of a benchmark or parity harness call into, initialize
from, or depend on reference-owned vocabulary, tokenizer, formatter, model,
tensor metadata, runtime, context, or cache state.

NEVER share reference-created state or objects with the EMEL side of a
benchmark or parity harness.

ALWAYS keep benchmark and parity harnesses split into two visible lanes: an
EMEL-owned lane using only workspace crates and a reference lane used only to
produce comparison results.

NEVER let parity or benchmark tools call private actor actions, guards, context
methods, or detail helpers directly. Drive behavior through public loader or
machine interfaces.

ALWAYS prefer porting exact reference arithmetic and operand handling into
EMEL-owned crates when the user requests parity, benchmarking, or performance
comparison.

ALWAYS treat kernel parity as incomplete until the EMEL-owned kernel consumes
the same effective operand format as the reference path.

NEVER replace a missing native packed or quantized hot-path kernel with a
dequantize-to-`f32` fallback unless the user explicitly approves that fallback
as an interim milestone.

If the user approves an interim fallback, label it `interim` in code, tests,
milestone documentation, and status updates, and state the exact operand or
kernel path still missing.

NEVER present parity or benchmark results as kernel parity when EMEL and the
reference implementation execute materially different operand pipelines.

ALWAYS ask before landing an implementation that changes a milestone's
performance contract by substituting a simpler fallback kernel, helper backend,
or tool-local scaffold for the intended runtime path.

For quantized inference work, done means:

1. EMEL-owned code in a maintained crate.
2. No tool-only compute fallback.
3. No whole-tensor dequantize-to-`f32` substitution in the hot path.
4. Benchmark claims based on the same effective operand class as the reference
   path.

When an implementation is narrower than the user's stated goal, stop and get
explicit approval before proceeding, even if the narrower implementation is
faster to complete.
