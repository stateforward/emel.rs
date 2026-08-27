---
lane_id: target-aarch64-gemv-live
owner_role: implementer
parent_agent: root-orchestrator
claim_scope: target-aarch64-f32-gemv-live-observer
---

stop_reason=proof-complete
proof_supplied=live pinned C++ aarch64::sm and public TargetAarch64Kernel observers matched byte-for-byte for dense contiguous F32 GEMV m=5,k=17, including body/tail output bits; zero-count and dimension-mismatch rejection preserved sentinels; snapshot-only replay passed; focused fmt/test/clippy, emel-kernels target router tests, diff-check, and forbidden-surface scan passed
proof_not_claimed=n_gt_1 dense matrix, noncontiguous/strided layouts, other dtypes, x86 live target, or full crate parity
artifacts=.artifacts/kernel-goal/target-f32-gemv-aarch64-live/report.md,.artifacts/kernel-goal/target-f32-gemv-aarch64-live/cpp.out,.artifacts/kernel-goal/target-f32-gemv-aarch64-live/rust.out
changed=tools/emel-kernel-target-f32-gemv-aarch64-parity,tools/emel-kernel-target-f32-gemv-aarch64-reference,scripts/kernel-target-f32-gemv-aarch64-parity.sh,snapshots/parity/kernel-target-f32-gemv-aarch64/manifest.txt
crate_edits=none
unsafe=none
commit=not authorized and not created
next_owner=root-orchestrator
