# Training overview

`typed-lm-trainer` is a binary with two subcommands:

- `train` — trains with one of four methods (`lora`, `qlora`, `full`,
  `from-scratch`), either an adapter over a frozen base checkpoint or the full set
  of parameters;
- `quantize` — applies **post-training quantization (PTQ)** to FP8/FP4 and merges
  an optional adapter first.

Both optimize the **cross-entropy at the decision position** — the last prompt
token, restricted to the candidate labels — exactly the position the server reads
at inference. Training therefore tunes the behavior the API actually uses.

## The pipeline

```mermaid
---
accTitle: Training and serving pipeline
accDescr: A dataset and a base checkpoint are used to train an adapter or a full checkpoint, optionally quantized, then served through the API.
---
flowchart LR
  dataset["dataset<br/>state + questions + answer"]:::neutral
  checkpoint["base checkpoint"]:::accent
  config["run configuration<br/>CLI or TOML"]:::warning
  train["train<br/>lora · qlora · full · from-scratch"]:::primary
  artifact["artifact<br/>adapter or checkpoint"]:::success
  quantize["quantize<br/>fp8 · fp4"]:::accent
  serve["typed-lm-serve"]:::success

  dataset --> train
  checkpoint --> train
  config --> train
  train --> artifact --> serve
  artifact --> quantize --> serve

  classDef primary fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:1.5px
  classDef accent fill:#dbeafe,stroke:#2563eb,color:#0c4a6e,stroke-width:1.5px
  classDef success fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px
  classDef warning fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:1.5px
  classDef neutral fill:#f4f4f5,stroke:#a1a1aa,color:#18181b,stroke-width:1.5px
```

## Training methods

The `--method` flag selects how the model starts and which parameters it trains:

| `--method` | Base origin | Trainable parameters | Output artifacts |
|---|---|---|---|
| `lora` (default) | checkpoint, frozen | LoRA `A`/`B` only | `adapter.safetensors` + `adapter_config.json` |
| `qlora` | checkpoint, quantized (dequantized on load), frozen | LoRA `A`/`B` only | `adapter.safetensors` + `adapter_config.json` |
| `full` | checkpoint | every parameter | `model.safetensors` + `config.json` + `tokenizer.json` |
| `from-scratch` | random initialization | every parameter | `model.safetensors` + `config.json` + `tokenizer.json` |

`lora` and `qlora` are the adapter methods: the base is never duplicated, and only
the adapter tensors are stored. `full` and `from-scratch` train every parameter
and write a **complete dense checkpoint** that `typed-lm-serve` serves directly.

```mermaid
---
accTitle: What each training method trains
accDescr: Adapter methods freeze the base and train LoRA tensors; full methods train every parameter.
---
flowchart TB
  subgraph adapter["Adapter methods (lora, qlora)"]
    base1["frozen base<br/>never duplicated"]:::accent
    lora["LoRA A/B<br/>trained"]:::primary
    out1["adapter.safetensors"]:::success
    base1 --> lora --> out1
  end

  subgraph fullmethod["Full methods (full, from-scratch)"]
    base2["base or random init"]:::accent
    all["every parameter<br/>trained"]:::primary
    out2["model.safetensors<br/>+ config + tokenizer"]:::success
    base2 --> all --> out2
  end

  classDef primary fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:1.5px
  classDef accent fill:#dbeafe,stroke:#2563eb,color:#0c4a6e,stroke-width:1.5px
  classDef success fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px
```

## Prerequisites

- Rust stable.
- A local base checkpoint (the trainer requires a path on disk; Hub identifiers
  must be downloaded first — the server loader can prefetch them).
- Optional: a CUDA build for GPU training.

```bash
cargo build --release -p typed-lm-trainer              # CPU
cargo build --release -p typed-lm-trainer --features cuda   # GPU
```

## Device selection

Every subcommand accepts `--device`:

| Value | Meaning |
|---|---|
| `auto` (default) | CUDA when available, otherwise CPU |
| `cpu` | Force the CPU |
| `cuda` | Force the first CUDA device (fails when no GPU is present) |

Training keeps master weights and optimizer state in **F32** on every device and
uses **BF16** only as the compute dtype on GPU (`PrecisionPolicy`), so adapter
quality is functionally equivalent between CPU and CUDA.

## Where to start

- [Preparing datasets](./datasets.md) — the Jev-native format.
- [Configuring a run](./configuration.md) — CLI flags and the TOML file.
- [Training LoRA and QLoRA adapters](./lora-qlora.md) — the common path.
- [Quantization (FP8 and FP4)](./quantization.md) — shrink the artifact.
