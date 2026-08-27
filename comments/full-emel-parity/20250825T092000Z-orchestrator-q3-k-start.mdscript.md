---
kind: coordination
task_id: full-emel-model-parity
lane_id: full-emel-model-parity-quantized-q3-k
status: active
owner: /root/q3_k_matmul_lane
source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6
artifact_root: /Users/gabrielwillen/VSCode/stateforward/emel/emel.rs/.artifacts/kernel-goal/quantized-q3-k
---

Q4_1, Q5_0, and Q2_K regular routes already exist. The next bounded
source-backed gap is regular native Q3_K `op_mul_mat`, using the pinned Q8_K
operand pipeline and the existing packed dot primitive. Q2_K’s missing focused
tests remain an explicit coverage residual and must not be silently treated as
parity proof.
