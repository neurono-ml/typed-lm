# Full fine-tuning

`--method full` keeps the existing checkpoint weights but makes **every** parameter
trainable: embeddings, norms, all attention and MLP projections, and the
language-model head. It reuses the same forward pass and decision-position loss as
the adapter methods.

## When should I use full fine-tuning?

- You need more capacity than LoRA provides and have a large, clean dataset.
- You can afford the memory: every parameter keeps an F32 master weight plus AdamW
  moments.
- You want a complete, self-contained checkpoint rather than an adapter.

If memory or compute is a constraint, prefer [LoRA or QLoRA](./lora-qlora.md).

## Run it

The geometry is read from the checkpoint `config.json`, so no geometry flags are
required:

```bash
cargo run --release -p typed-lm-trainer -- train \
  --method full \
  --model-id /path/to/local/checkpoint \
  --dataset resources/dataset.jsonl \
  --output-directory output/full \
  --epochs 3 --batch-size 4 --learning-rate 1e-4
```

```mermaid
---
accTitle: Full fine-tuning
accDescr: Every parameter receives gradients and the run writes a complete dense checkpoint.
---
flowchart LR
  checkpoint["base checkpoint"]:::accent
  all["every parameter<br/>trainable"]:::primary
  forward["forward to<br/>decision position"]:::accent
  loss["restricted cross-entropy"]:::warning
  optimizer["AdamW update<br/>all parameters"]:::primary
  output["model.safetensors<br/>+ config + tokenizer"]:::success

  checkpoint --> all --> forward --> loss --> optimizer --> all
  optimizer --> output

  classDef primary fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:1.5px
  classDef accent fill:#dbeafe,stroke:#2563eb,color:#0c4a6e,stroke-width:1.5px
  classDef success fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px
  classDef warning fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:1.5px
```

## Output

```text
output/full/
├── model.safetensors   # canonical Hugging Face tensor names, all parameters
├── config.json         # reparseable model configuration
└── tokenizer.json      # copy of the checkpoint tokenizer
```

This directory is a complete checkpoint. Point `typed-lm-serve` at it with no
merge step.

> **Cost.** Every parameter keeps an F32 master weight plus AdamW moments, so full
> training uses much more memory and compute than LoRA. Use it when the task
> justifies it.

## Next steps

- [Training from scratch](./from-scratch.md) — train from random weights.
- [Quantization (FP8 and FP4)](./quantization.md) — shrink the full checkpoint.
- [Serving a trained artifact](./serving-artifacts.md) — serve it directly.
