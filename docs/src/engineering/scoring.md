# Scoring and batched decoding

typed-lm reads a decision from one position of one forward pass. This page
describes how the forward pass is organized and how the answers are calibrated.

## Shared prefill and broadcast

The evaluator prefills the fixed system context into a KV-cache once at startup.
Each request then:

1. tokenizes `system + state` and prefills the common prefix;
2. **broadcasts** the cache on the attention batch dimension;
3. evaluates each question suffix in a **single batched forward pass**;
4. reads each label token's logit at the last position.

```mermaid
---
accTitle: Batched decoding
accDescr: The common prefix is prefilled once, the cache is broadcast across the batch, and every question suffix is decoded together.
---
flowchart LR
  prefix["prefill common prefix"]:::accent
  cache["KV-cache"]:::warning
  broadcast["broadcast on batch dim"]:::primary
  q1["suffix 1"]:::primary
  q2["suffix 2"]:::primary
  qn["suffix n"]:::primary
  batch["one batched forward pass"]:::success
  logits["label logits at last position"]:::success

  prefix --> cache --> broadcast
  broadcast --> q1
  broadcast --> q2
  broadcast --> qn
  q1 --> batch
  q2 --> batch
  qn --> batch
  batch --> logits

  classDef primary fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:1.5px
  classDef accent fill:#dbeafe,stroke:#2563eb,color:#0c4a6e,stroke-width:1.5px
  classDef success fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px
  classDef warning fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:1.5px
```

## Calibration

| Type | Calibration |
|---|---|
| `noul` | Binary softmax over the yes/no labels |
| `choice` | Temperature-scaled restricted softmax over the option labels |
| `score` | Temperature-scaled restricted softmax over the level labels |

`score` is the expected value over the level indices. `confidence` is derived from
how concentrated the distribution is.

## Vendored parallel forward

`typed-lm-serve/src/infrastructure/parallel_llama.rs` is a vendored,
broadcastable dense decoder implementation covering every supported family. It is
validated against upstream by `#[ignore]` equivalence tests.

GGUF-quantized checkpoints use a Qwen2-only path
(`parallel_quantized_qwen2.rs`); any other family served from GGUF is rejected with
an actionable message.

The fused CPU flash attention is used automatically on CPU and keeps GQA grouped.

## Next steps

- [Session prefix cache](./session-cache.md) — skipping the prefix pass.
- [Benchmarks](./benchmarks.md) — measured latency.
- [Architecture](./architecture.md) — the module layout.
