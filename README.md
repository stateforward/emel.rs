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

## Development

<!-- Cargo aliases declared in .cargo/config.toml -->

```sh
cargo check-all
cargo lint
cargo test-all
```

`sml.rs` remains a separately versioned workspace. During the port, this workspace uses its local path dependency; release builds should replace it with the published `stateforward-sml` version.
