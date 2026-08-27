<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Proof Supplied

* claim_scope=kernel-flash-attn-live-observer
* proof_supplied=live pinned C++ x86 and public Rust root `Kernel` observers matched byte-for-byte; canonical and invalid-scale output bits matched; snapshot-only replay passed; `git diff --check` passed
* proof_artifact=.artifacts/kernel-goal/flash-attn-live/report.md
* output_sha256=c10fbc7ceb95e3719feb77e29f06107bb3bb1c03e650a9671b3bf2a779d7c218
* source_commit=843a117386ef17dc5a50549bbfc821074c2141d6
* source_sml_commit=49207123cd3f39767764bae774932cb48623f92f
* residual=exact_live_differential;simd_target_routes;flash_attn_back;masked_causal_execution;non-f16-kv;raw-op-parameter-byte-provenance;f16-multi-dimension-byte-alignment-safe-subset
* proof_not_claimed=SIMD target routes, flash_attn_back, masked causal execution, non-F16 K/V, raw op-parameter byte provenance, F16 multi-dimension byte-alignment-safe subset, full workspace quality gates, commit, push, PR, or merge
* next_owner=root-orchestrator
