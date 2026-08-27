<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Resume Goal

* continue the pinned kernel port at [Hot Path](#hot-path)
* owner_role=root-orchestrator
* source_of_truth=emel.cpp 843a117386ef17dc5a50549bbfc821074c2141d6 and crates/emel-kernels
* stop_condition=all 22 inventory counterparts are maintained, split-reference parity and benchmark snapshots are proven, and the required workspace gates pass

## Hot Path

* run `mdscript-exec comments/port-emel-kernels/20260827T024907Z-x86-conv-transpose-f32-implementation.mdscript.md#next-steps`

## Done So Far

* implemented the x86_64 dense F32 `op_conv_transpose_1d` route in safe Rust through Pulp's AVX2/FMA capability boundary
* added a public typed router event, SML forwarding transition, child actor, guards, action, unexpected-event route, focused tests, and allocation-free dispatch check
* removed an incomplete x86 reference-tool scaffold from the active tool tree before it could be mistaken for parity evidence
* verified package formatting, host lint, host tests, x86 cross-target compilation/clippy, and a focused x86-target test binary; its AVX2/FMA-gated cases execute only on a capable x86_64 host and therefore covered local compile/runtime-entry behavior without claiming live differential parity

## Next Steps

* implement a complete split reference/Rust observer pair for the x86 route on a host that can execute AVX2/FMA
* commit its live parity snapshot under `snapshots/parity`
* add a reproducible benchmark lane and committed benchmark snapshot before closeout for this operation

## Proof Supplied

* proof_supplied=focused implementation proof: source identity, valid F32 layouts, invalid-shape rejection without output mutation, explicit unexpected event, and allocation-free composed dispatch tests; cross-target compile and clippy gates
* proof_not_claimed=live C++ reference differential, committed x86 parity snapshot, performance benchmark snapshot, full workspace gate rerun after this slice, coverage delta, fuzz result, commit, PR, or merge
* residual=F16 weights, strided tensor metadata, shared scalar transition, other unpinned x86 operators, and all remaining partial inventory rows
* next_owner=root-orchestrator
