# Quick start

This guide takes you from zero to a typed answer, then trains and serves a LoRA
adapter. It assumes a Rust toolchain; no GPU is required for the first steps.

## 1. Install

```bash
# Server and trainer from crates.io (CPU build).
cargo install typed-lm-serve
cargo install typed-lm-trainer

# Or build from the workspace.
cargo build --release --workspace
```

For a CUDA build, add `--features cuda`; on Apple Silicon, add `--features metal`.
See [Running the server](./guides/running.md) for the full matrix.

## 2. Start the server

```bash
# Downloads the default model (Qwen/Qwen2.5-1.5B-Instruct) on first startup.
typed-lm-serve --context-path resources/memory.md
```

Verify it is up:

```bash
curl -s http://127.0.0.1:8080/health/live
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
```

The server listens on `0.0.0.0:8080` by default.

## 3. Ask your first questions

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_mixed.json
```

You receive typed answers: a probability for `noul`, a winning label with a
distribution for `choice`, and an expected value with a legend for `score`.

```json
{
  "model": "typed-lm",
  "answers": {
    "refund_eligible": { "type": "noul", "noul": 0.87 },
    "responsible_department": {
      "type": "choice",
      "choice": "logistics",
      "probabilities": { "billing": 0.05, "logistics": 0.9, "product_support": 0.05 },
      "confidence": 0.85
    }
  },
  "usage": { "input_tokens": 512, "output_tokens": 4 }
}
```

The complete contract is in [Calling the API](./guides/api.md).

## 4. Train a LoRA adapter

```bash
typed-lm-trainer train \
  --model-id /path/to/local/checkpoint \
  --dataset resources/dataset.jsonl \
  --output-directory output/train \
  --method lora --epochs 3 --batch-size 4 --learning-rate 1e-4
```

The dataset format and every flag are documented in
[Preparing datasets](./training/datasets.md) and
[Training LoRA and QLoRA adapters](./training/lora-qlora.md).

## 5. Quantize and serve

```bash
typed-lm-trainer quantize \
  --model-id /path/to/local/checkpoint \
  --adapter-directory output/train \
  --quantization fp8 --output-directory output/quantized

# The quantized directory holds weights only; add the base metadata.
cp /path/to/local/checkpoint/config.json    output/quantized/
cp /path/to/local/checkpoint/tokenizer.json output/quantized/

typed-lm-serve --model-id output/quantized
```

See [Quantization (FP8 and FP4)](./training/quantization.md) and
[Serving a trained artifact](./training/serving-artifacts.md).

<div class="sk-box sk-box--info">
<strong>No weights yet?</strong> You can validate the whole pipeline end to end with a
tiny dummy checkpoint: <code>cargo test -p typed-lm-serve --test end_to_end</code>.
</div>

## Where to go next

- [System One decisions](./concepts/system-one.md) — the mental model.
- [Questions (primitives)](./concepts/primitives.md) — noul, choice and score.
- [Training overview](./training/index.md) — the full training tutorial.
- [HTTP API reference](./reference/api.md) — every route and field.
