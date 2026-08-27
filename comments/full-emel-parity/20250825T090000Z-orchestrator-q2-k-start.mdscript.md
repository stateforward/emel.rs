---
kind: coordination
task_id: full-emel-model-parity
lane_id: full-emel-model-parity-quantized-q2-k
status: active
owner: /root/q2_k_matmul_lane
source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6
artifact_root: /Users/gabrielwillen/VSCode/stateforward/emel/emel.rs/.artifacts/kernel-goal/quantized-q2-k
---

Q4_1 and Q5_0 regular routes are already maintained and audited. The next
bounded executable gap is regular native Q2_K `op_mul_mat`, using the pinned
Q8_K operand pipeline and existing packed dot primitive. The lane must keep
behavior choices in explicit SML guards/transitions and must not substitute a
whole-tensor dequantized F32 path.
