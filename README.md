# emel.rs

Rust port of the companion `emel.cpp` project, organized as independently buildable crates around the original system's stable domains.

## Crates

<!-- workspace members declared in Cargo.toml -->

| Crate | Responsibility |
| --- | --- |
| `emel-core` | Shared model metadata, errors, and domain types. |
| `emel-tensor` | Tensor contracts and quantization primitives. |
| `emel-kernels` | Portable and architecture-specific numerical kernels. |
| `emel-io` | Model-source and memory-mapping interfaces. |
| `emel-grammar` | GBNF parsing and constrained-generation contracts. |
| `emel-batch` | Batch planning and scheduling. |
| `emel-embeddings` | Embedding-model and generator components. |
| `emel-gguf` | GGUF container parsing and loading. |
| `emel-graph` | Execution-graph allocation, assembly, processing, and tensor binding. |
| `emel-logits` | Logit validation and sampling. |
| `emel-memory` | KV, recurrent, streaming, and hybrid memory management. |
| `emel-model` | Model architectures, family bindings, and lifecycle loading. |
| `emel-token` | Token representations and batching. |
| `emel-text` | Text-generation state machines, backed by `stateforward-sml`. |
| `emel-speech` | Speech-model components. |
| `emel-diarization` | Speaker-diarization components. |
| `emel` | Feature-gated public façade. |
| `emel-inspect` | Model-inspection command-line tool. |
| `emel-bench` | Dependency-light performance snapshot runner. |
| `emel-gguf-parity` | Differential GGUF loader runner against pinned llama.cpp. |

## Development

<!-- Cargo aliases declared in .cargo/config.toml -->

```sh
cargo check-all
cargo lint
cargo test-all
```

<!-- benchmark, parity, fuzz, and coverage commands implemented by scripts/bench.sh, scripts/paritychecker.sh, scripts/fuzz.sh, and scripts/coverage.sh -->

`scripts/paritychecker.sh` refreshes the checked-in snapshot from the pinned
llama.cpp reference, performs live parity, and verifies the resulting snapshot
by default. Use `--snapshot-only` for the fast Rust-only gate without a C++ build,
or the `--no-snapshot`, `--no-live`, and `--no-update` switches to select phases.

`scripts/bench.sh` runs the dependency-light benchmark runner. Add `--snapshot`
to compare with the current architecture baseline, or `--snapshot --update` to
refresh it after an intentional performance change.

`scripts/fuzz.sh` regenerates a shared corpus from the llama.cpp parity fixtures
and smoke-tests all isolated `cargo-fuzz` GGUF targets for ten seconds each. It
requires nightly Rust and `cargo-fuzz`; use `--target`, `--seconds`, or
`--build-only` to narrow a local run. Pull requests run the smoke suite, while a
separate weekly workflow preserves the evolving corpus and runs longer campaigns.

`scripts/coverage.sh` enforces at least 90% line coverage and 50% branch coverage
for the completed `emel-gguf` production sources. Generated `sm.rs` state-machine
plumbing is excluded. Branch instrumentation uses the pinned nightly toolchain
and `cargo-llvm-cov` version named by the script; CI stores the JSON report as an
artifact. Raise or broaden the gate as additional scaffold crates are ported.

`sml.rs` remains separately versioned and is consumed from crates.io as `stateforward-sml`.
