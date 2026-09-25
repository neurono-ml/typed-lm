# Supported architectures

The architecture is detected automatically from the `model_type` field in
`config.json`; no flag is needed.

## Supported dense families

| Family | `model_type` |
|---|---|
| Llama | `llama` |
| Qwen2 | `qwen2` |
| Qwen3 | `qwen3` |
| Mistral | `mistral` |
| Gemma | `gemma` |
| Gemma2 | `gemma2` |
| Gemma3 | `gemma3` |

## Rejected families

Mixture-of-Experts and multi-head-latent-attention families are **not supported**
and are rejected at load time with an actionable error:

| Rejected `model_type` | Reason |
|---|---|
| `mixtral` | Mixture-of-Experts |
| `qwen3_moe` | Mixture-of-Experts |
| `deepseek_v2` (also `deepseek2`) | Multi-head latent attention |
| `deepseek_v3` | Multi-head latent attention |

DeepSeek is therefore excluded.

## Dense versus GGUF

Dense safetensors, PyTorch and NumPy checkpoints of any of the seven families are
served. **GGUF-quantized serving is Qwen2-only**: a GGUF checkpoint declaring
another architecture is rejected, and a non-Qwen2 model must be converted to a
dense format first.

```mermaid
---
accTitle: Architecture detection and compatibility
accDescr: model_type selects the dense family; MoE and MLA families are rejected, and GGUF serving is restricted to Qwen2.
---
flowchart TB
  config["config.json model_type"]:::neutral
  dense{"dense family?"}:::warning
  family["llama · qwen2 · qwen3<br/>mistral · gemma · gemma2 · gemma3"]:::success
  moe["mixtral · qwen3_moe<br/>deepseek_v2 · deepseek_v3"]:::danger
  gguf{"GGUF and qwen2?"}:::warning
  served["served"]:::success
  rejected["rejected"]:::danger

  config --> dense
  dense -- "yes" --> family --> gguf
  dense -- "no" --> moe --> rejected
  gguf -- "yes" --> served
  gguf -- "no (dense only)" --> family

  classDef success fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px
  classDef danger fill:#fee2e2,stroke:#dc2626,color:#7f1d1d,stroke-width:1.5px
  classDef warning fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:1.5px
  classDef neutral fill:#f4f4f5,stroke:#a1a1aa,color:#18181b,stroke-width:1.5px
```

## Related

- [Running the server](../guides/running.md) — layouts and weight kinds.
- [Choosing an architecture](../training/architecture.md) — geometry for training.
