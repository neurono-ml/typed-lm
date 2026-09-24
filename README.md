# typed-lm

Rust monorepo for **deterministic inference** and **adapter training** of
Llama/Qwen2-style models, part of the **Sciencekit** ecosystem. Instead of
autoregressive text generation, the server classifies answers in a **single
forward pass**: each question is answered from the logits of a local model run
with [Candle](https://github.com/huggingface/candle). The trainer produces
LoRA/QLoRA adapters and quantized artifacts (FP8/FP4) that the server consumes
directly.

The HTTP API is compatible with the **Jev (TypeSafe AI)** format: the client
sends `state` (case facts) and `questions` (`noul`/`choice`/`score` with
instructions and criteria) and receives typed answers — no free text.

## Documentation

| Guide | Contents |
|---|---|
| [`docs/running.md`](docs/running.md) | Build and run the server: CPU/GPU, flags, layouts, session cache. |
| [`docs/api.md`](docs/api.md) | Jev contract, routes, response shapes and `curl` examples. |
| [`docs/training.md`](docs/training.md) | Dataset format, LoRA/QLoRA training, FP8/FP4 quantization. |

Additional references:

- [`typed-lm-trainer/README.md`](typed-lm-trainer/README.md) — trainer
  subcommands, flags and PTQ details.
- [`examples/README.md`](examples/README.md) — request examples and `curl` calls.
- [`example.http`](example.http) — the same requests as an HTTP file
  (VS Code REST Client).
- [`AGENTS.md`](AGENTS.md) — project rules for contributors and agents.

## Workspace

| Crate | Role | Type |
|---|---|---|
| `typed-lm-common` | Jev contract, labels, prompt rendering, checkpoint detection, device/dtype, quantization | lib |
| `typed-lm-serve` | Jev-compatible Actix server (binary, no subcommand) | bin |
| `typed-lm-trainer` | LoRA/QLoRA fine-tuning and post-training quantization (subcommands `train`/`quantize`) | bin + lib |

```bash
cargo build --workspace
cargo run -p typed-lm-serve -- --help
cargo run -p typed-lm-trainer -- --help
```

## Quickstart

```bash
# Server (downloads the default model on first startup).
cargo run -p typed-lm-serve

# A typed request.
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_mixed.json

# Train a LoRA adapter, then quantize it to FP8.
cargo run -p typed-lm-trainer -- train \
  --model-id /path/to/local/checkpoint \
  --dataset resources/dataset.jsonl \
  --output-directory output/train --method lora \
  --epochs 3 --batch-size 4 --learning-rate 1e-4

cargo run -p typed-lm-trainer -- quantize \
  --model-id /path/to/local/checkpoint \
  --adapter-directory output/train \
  --quantization fp8 --output-directory output/quantized
```

See [`docs/running.md`](docs/running.md) for server details,
[`docs/api.md`](docs/api.md) for the full contract and
[`docs/training.md`](docs/training.md) for the training/quantization pipeline.

## Routes

`POST /v1/systemone`, `GET /v1/models`, `GET /health`, `GET /health/live`.
The request accepts `noul` (boolean decision), `choice` (best option from a
restricted set) and `score` (continuous value over levels) questions, combinable
in one call. Invalid bodies return `422`, unknown models `404` and inference
failures `500`, all with the `{"error": {"message": "..."}}` envelope. The
complete shapes are in [`docs/api.md`](docs/api.md).

## Acceleration

- **CPU** — GGUF `Q4_K_M` (roughly half the per-request cost of dense `F32`) plus
  the `mkl` feature and the fused CPU flash attention.
- **GPU** — `--features cuda` selects F16 weights via `--model-dtype auto`.
- **Training** — `PrecisionPolicy { master: F32, compute: F32(CPU)/BF16(GPU),
  reduction: F32 }` keeps master weights and optimizer state in F32 on every
  device; parity is functional, not of speed.

Measured latency tables and the devcontainer/CUDA setup are in
[`docs/running.md`](docs/running.md) and the original benchmark notes further
below.

## Tests

```bash
cargo test --workspace                       # unit + integration, no download
cargo test --workspace -- --ignored --nocapture   # live tests (real weights)
cargo clippy --workspace --all-targets
cargo fmt --check
```

- `cargo test --workspace` covers probability calibration, labels, prompt
  rendering, session-cache eviction, CPU flash attention vs a matmul/softmax
  reference, FP8/FP4 quantization, dataset/collate, LoRA and the training loop
  (dummies), API integration with a `MockEvaluator`, and a **binary-level E2E**
  (`typed-lm-serve/tests/end_to_end.rs`: `train` → `quantize` → `serve` over
  HTTP) with no weight download.
- `cargo test --workspace -- --ignored` runs the **live** tests marked
  `#[ignore]`: real-weight equivalence with upstream, session-cache gain,
  latency benchmarks, FP8/FP4 artifact loading, and **GPU training/quantization**
  (`typed-lm-trainer/tests/live_gpu_e2e.rs`). Do not run these in CI.
- Do not pass `--all-features` on Linux (the Metal feature requires macOS).

A reproducible CPU E2E script lives in `temporary/e2e/run_e2e_cpu.sh`.

## Architecture

- **HTTP (`typed-lm-serve/src/api/`)**: `actix-web` with `web::scope("/v1")`
  (`POST /v1/systemone`, `GET /v1/models`) and `GET /health`, `GET /health/live`.
  Handlers receive a `SharedState` (evaluator + model name + context name +
  startup time) via `web::Data`.
- **Shared contract (`typed-lm-common`)**: request DTOs, label arithmetic and
  prompt rendering live in the common crate so server and trainer produce
  byte-identical prompts and agree on the decision position.
- **Scoring (Candle)**: `CandleEvaluator` prefills the fixed system context into
  a KV-cache **once at startup**. Each request does a *single shared prefill* of
  the tokens common to the questions, **broadcasts** the cache on the attention
  batch dimension, and evaluates each question suffix in **a single batched
  forward pass**. The state prefix (`system + state`) is retained in a bounded
  LRU. It reads each label token's logit (`A`, `B`, …) at the last position and
  calibrates the distribution (binary softmax for `noul`, temperature for
  `choice`/`score`).
- **Parallel forward (`typed-lm-serve/src/infrastructure/parallel_llama.rs`)**:
  a vendored, broadcastable Llama implementation, validated against upstream
  `Llama`/`Qwen2` by `#[ignore]` equivalence tests.
- **Training (`typed-lm-trainer/src/`)**: `dataset` (discovery/record/loader/
  collate), `model` (precision/LoRA/differentiable forward/weight loading),
  `training` (loss/optimizer/checkpoint/loop) and `quantization` (export).
- **Context (`ContextProvider`)**: currently `FileContextProvider` (for example
  `--context-path resources/memory.md`); the interface allows swapping the source
  for retrieval (RAG) later without changing handlers, evaluator or API.

## CPU benchmark notes

`reports_latency_breakdown` (release, Qwen2.5-1.5B dense, `F32`):

| Prefix | Stage | Baseline | + CPU flash | + MKL |
|---|---|---|---|---|
| 64 | prefill | 3.13 s | 2.34 s | **0.52 s** |
| 256 | prefill | 8.65 s | 4.97 s | **1.69 s** |
| 1024 | prefill | 28.23 s | 20.66 s | **13.13 s** |
| 64 | 5 batched suffixes | 1.44 s | 1.33 s | **0.25 s** |
| 256 | 5 batched suffixes | 2.47 s | 1.98 s | **0.35 s** |
| 1024 | 5 batched suffixes | 4.74 s | 4.59 s | **2.57 s** |
| 64 | single next token | 655 ms | 699 ms | **159 ms** |
| 256 | single next token | 811 ms | 347 ms | **175 ms** |
| 1024 | single next token | 815 ms | 545 ms | **300 ms** |

## GPU benchmark notes

`reports_latency_breakdown` (release, Qwen2.5-1.5B, `F16`, RTX 3070):

| Prefix | prefill | 5 batched suffixes | single next token |
|---|---|---|---|
| 64 | 14 ms | 36 ms | 52 ms |
| 256 | 31 ms | 81 ms | 65 ms |
| 1024 | 154 ms | 379 ms | 64 ms |
