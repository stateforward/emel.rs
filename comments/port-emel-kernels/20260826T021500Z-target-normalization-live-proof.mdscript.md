<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Proof Supplied

* claim_scope=target-aarch64-normalization-live-observer
* proof_supplied=live pinned C++ AArch64 `sm` and public Rust `TargetAarch64Kernel` byte match for dense F32 `norm` and `rms_norm`; typed `InvalidParameters`, `InvalidView`, and `ShapeMismatch` rejection with sentinel preservation; snapshot-only replay; clippy `-D warnings`; scoped fmt; `git diff --check`
* proof_artifact=.artifacts/kernel-goal/target-normalization-live/report.md
* output_sha256=88fc92695f561b59154ee9a27ad99ab8a46a82caaa1ed60badfd1e5ab49579e9
* proof_not_claimed=full workspace quality gates, SIMD-specific implementation, rms_norm_back, group_norm, commit, push, PR, merge
* next_owner=root-orchestrator
