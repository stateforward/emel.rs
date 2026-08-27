<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Proof Supplied

* claim_scope=target-aarch64-im2col-live-observer
* proof_supplied=live pinned C++ AArch64 `sm` and public Rust `TargetAarch64Kernel::process_im2col` byte match for dense F32 1-D `zero_padding`; typed `InvalidParameters` and `ShapeMismatch` rejection with sentinel preservation; snapshot-only replay; clippy `-D warnings`; scoped fmt; `git diff --check`
* proof_artifact=.artifacts/kernel-goal/target-im2col-live/report.md
* output_sha256=102bcd643c0ce708c233c6c78dacc04c1561098f7ed7567f1602e1341f981f42
* source_spans=src/emel/kernel/aarch64/sm.hpp:821-833; src/emel/kernel/detail.hpp:4858-4960
* proof_not_claimed=full workspace quality gates, SIMD-specific implementation, im2col_back, im2col_3d, full reference fixture differential, commit, push, PR, merge
* next_owner=root-orchestrator
