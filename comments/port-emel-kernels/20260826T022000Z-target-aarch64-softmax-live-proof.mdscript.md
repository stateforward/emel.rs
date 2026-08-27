<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Proof Supplied

* claim_scope=target-aarch64-softmax-live-observer
* proof_supplied=live pinned C++ AArch64 `sm` and public Rust `TargetAarch64Kernel` byte match for dense F32 `op_soft_max`; typed `InvalidView`, `ShapeMismatch`, and `UnexpectedEvent` rejection with sentinel preservation; snapshot-only replay; clippy `-D warnings`; scoped fmt; `git diff --check`
* proof_artifact=.artifacts/kernel-goal/target-softmax-live/report.md
* output_sha256=1f236c78ece96c47c98524214f4d486cfebe9093ebbe79f0852eb9b462860a30
* source_commit=843a117386ef17dc5a50549bbfc821074c2141d6
* source_sml_commit=49207123cd3f39767764bae774932cb48623f92f
* residual=soft_max_back
* proof_not_claimed=sparse live observer, x86 live target, full workspace quality gates, commit, push, PR, or merge
* next_owner=root-orchestrator
