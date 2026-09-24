# manaca-jev-like

Deterministic inference server in Rust, part of the **Sciencekit** ecosystem.
It replaces autoregressive text generation with classification in a
*single forward pass*: each question is answered from the *logits* of a
Llama-style model (e.g.: Manacá-1B) executed locally with
[Candle](https://github.com/huggingface/candle).

The HTTP API is compatible with the **Jev (TypeSafe AI)** format: the client sends
`state` (the case facts) and `questions` (noul/choice/score with instructions and
criteria) and receives typed answers — no free text.

## Routes

### `POST /v1/systemone`

Evaluates one or more questions (`noul`, `choice` and `score` can be combined
in the same request). The `model` field must be the served name (see
`--served-model-name`) or any alias with the `jev-` prefix. Validation is
structural: `questions` cannot be empty, `choice` requires at least one
criterion and `score` requires 2 to 10 levels.

Request (see `examples/request_mixed.json`):

```json
{
  "model": "jev-latest",
  "state": "Order #7710 arrived with a smashed box and a cracked vase inside. Delivery was 3 days ago and the customer asks what to do next.",
  "questions": {
    "refund_eligible": {
      "type": "noul",
      "instructions": "The customer is eligible for a full refund under the store policy."
    },
    "responsible_department": {
      "type": "choice",
      "instructions": "Which department should handle this case?",
      "criteria": {
        "billing": "Double charges and payment errors",
        "logistics": "Damaged, lost, or late shipments",
        "product_support": "Defective-item troubleshooting, replacements, and setup help"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this case?",
      "criteria": ["Routine", "Urgent", "Emergency"]
    }
  }
}
```

Response (format; values vary by model and context):

```json
{
  "model": "jev-latest",
  "answers": {
    "refund_eligible": { "type": "noul", "noul": 0.87 },
    "responsible_department": {
      "type": "choice",
      "choice": "logistics",
      "probabilities": { "billing": 0.05, "logistics": 0.9, "product_support": 0.05 },
      "confidence": 0.85
    },
    "urgency": {
      "type": "score",
      "score": 1.2,
      "legend": { "0": "Routine", "1": "Urgent", "2": "Emergency" },
      "probabilities": { "0": 0.2, "1": 0.4, "2": 0.4 },
      "confidence": 0.2
    }
  },
  "usage": { "input_tokens": 512, "output_tokens": 4 }
}
```

Semantics per type:

- `noul`: probability of the affirmative answer in the `noul` field (0.0 to 1.0).
- `choice`: winning label in `choice`, distribution in `probabilities` and
  `confidence`.
- `score`: expected value over the levels in `score`, index-to-name legend in
  `legend`, distribution in `probabilities` and `confidence`.

Errors follow the `{"error": {"message": "..."}}` envelope: invalid body or
out-of-contract question returns `422`, unknown model returns `404` and
inference failure returns `500`.

### `GET /v1/models`

Lists the served model plus the `jev-latest` alias:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

```json
{
  "object": "list",
  "data": [{ "id": "jev-latest", "object": "model", "owned_by": "manaca" }],
  "models": [{ "name": "jev-latest", "description": "Jev-compatible model served from context '...'", "release_date": "unknown" }]
}
```

### `GET /health` and `GET /health/live`

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/health/live
```

`/health` returns `{"status": "ok", "startup_seconds": 12.3}` (model load time
at startup); `/health/live` returns `{"status": "ok"}` and does not depend
on the model.

## Quickstart

Prerequisites: stable Rust (the binary is called `manaca-typed`, see
`Cargo.toml`).

```bash
cargo build
cargo run -- serve
```

The server listens on `0.0.0.0:8080` by default (access it via
`http://127.0.0.1:8080`). On first startup it downloads the default model weights
to the local Hugging Face cache.

Example requests (server running in another terminal):

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_noul.json

curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_mixed.json

curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_context.json
```

The examples use the fictional GreenLeaf store facts described in
`examples/README.md`. Without a context (`--context-path` missing) the evaluator
receives an empty context; to ground answers in the example facts:

```bash
cargo run -- serve --context-path resources/memory.md
```

## Configuration

Every `serve` option can also come from an environment variable
(CLI flag > env > default). Verified against `cargo run -- serve --help`:

| CLI Flag | Environment Variable | Default |
|---|---|---|
| `--host` | `HOST` | `0.0.0.0` |
| `--port` | `PORT` | `8080` |
| `--model-id` | `MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `--model-revision` | `MODEL_REVISION` | `main` |
| `--weights-file` | `WEIGHTS_FILE` | (auto-detected) |
| `--tokenizer-file` | `TOKENIZER_FILE` | (next to the weights) |
| `--config-file` | `CONFIG_FILE` | (next to the weights) |
| `--context-path` | `CONTEXT_PATH` | (missing = empty context) |
| `--served-model-name` | `SERVED_MODEL_NAME` | `jev-latest` |
| `--model-dtype` | `MODEL_DTYPE` | `auto` (F32 on CPU, F16 on CUDA/Metal) |
| `--session-cache-entries` | `SESSION_CACHE_ENTRIES` | `16` |
| `--session-cache-tokens` | `SESSION_CACHE_TOKENS` | `32768` |
| `--hf-token` | `HF_TOKEN` | (missing) |

### Session prefix cache

The expensive part of a request is the forward pass over the *state* prefix
(the system context plus the request `state`). The server tokenizes
`system + state` once and retains the resulting key/value cache in a bounded
least-recently-used cache keyed by a canonical hash of the state. A state that
reappears across requests therefore skips that forward pass. The cache is
bounded by both the number of entries (`--session-cache-entries`) and the total
number of cached tokens (`--session-cache-tokens`); the least recently used
entries are evicted first. Setting either bound to `0` disables session
caching. The cached key/value cache is never mutated: every request clones it
before use.

Example with environment:

```bash
PORT=9090 MODEL_ID=recogna-nlp/bode-1b-instruct cargo run -- serve
```

### Restricted (gated) models and `HF_TOKEN`

The default model (`menezesbruno/manaca-1b-base`) is public and requires no
authentication. If you switch to a restricted-access model via
`--model-id` (for example `recogna-nlp/bode-1b-instruct`), accept the terms of use
on the model page on Hugging Face and export a token with read permission before
starting the server:

```bash
HF_TOKEN=hf_your_token_here cargo run -- serve --model-id recogna-nlp/bode-1b-instruct
```

Without the token, the weight download fails with a `401` error
(`failed to download ... status code 401`) — expected behavior, not a
bug. Details and more examples in `examples/README.md`.

At startup the repository is inspected before downloading: if it does not
expose `config.json`, `tokenizer.json` and `model.safetensors`, the server
reports it as *not a servable checkpoint* instead of a misleading `404` or
gating hint. This is the case for code-only repositories that share a model
name, such as `harshatheg/Qwen-2.5-1B-RLCD` (the parallel-constrained-decoding
demo source, whose checkpoint is Qwen2.5/MLX and is not loadable here).

### Supported models, layouts and architectures

The checkpoint is detected automatically:

- **Layouts**: safetensors (single file or sharded), GGUF (dense or
  GGML-quantized, e.g. `Q4_K_M`), PyTorch `.pth`/`.bin` and NumPy `.npz`.
- **Architectures**: Llama and Qwen2.
- **Weight kinds**: full precision (`BF16`/`F16`/`F32`) and GGML-quantized
  (GGUF). `FP8`/`F8_E4M3`, `GPTQ` and `AWQ` are rejected at startup with a
  descriptive error instead of panicking inside `Llama::load`. For example
  `liodon-ai/manaca-1b-base-FP8` (168 `F8_E4M3` tensors) is not loadable;
  supporting `FP8` would require dequantizing at load time.

```bash
cargo run -- serve --model-id menezesbruno/manaca-1b-base --served-model-name manaca
```

## Acceleration

### CPU

For CPU inference the recommended mode is a GGUF `Q4_K_M` checkpoint (roughly
halves the per-request cost of the `F32` dense path) combined with the `mkl`
feature (Intel MKL BLAS) and the built-in fused CPU flash attention. The
vendored attention uses `candle_nn::attention::flash_attn` on the CPU
automatically — no flag needed — and keeps grouped query attention grouped.
`--model-dtype` stays `auto` (F32 on CPU).

```bash
cargo build --release --features mkl
# GGUF Q4_K_M (tokenizer borrowed from the full-precision repo):
TOKENIZER=$(find ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct \
  -name tokenizer.json | head -1)
cargo run --release --features mkl -- serve \
  --model-id Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --weights-file qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --tokenizer-file "$TOKENIZER" \
  --context-path resources/memory.md
```

`.cargo/config.toml` already sets `target-cpu=native`; only build and run on the
same machine (remove it when cross-compiling).

#### Measured CPU gains

`reports_latency_breakdown` (release, Qwen2.5-1.5B dense, `F32`) over the same
prefill/suffix/decode workload:

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

The session cache removes the state-prefix prefill from repeated requests over
the same state; combined with MKL, short-context requests land in the hundreds
of milliseconds (see `session_cache_reuses_state_prefix_and_preserves_answers`).

> `intel-mkl-src` 0.8.1 bundles Intel MKL 2020.1, which does not export the
> half-precision `hgemm_` symbol that `candle-core`'s `mkl` feature references
> (the CPU path always computes in `F32`, but the reference still breaks the
> link). `src/infrastructure/mkl_f16_shim.rs` supplies an `hgemm_` built on
> MKL's `sgemm_`, so the `mkl` feature links and half-precision calls remain
> correct.

### GPU (CUDA) via devcontainer

The host needs the NVIDIA driver and the NVIDIA container toolkit
(`nvidia-ctk runtime configure --runtime=docker`). The devcontainer installs
the CUDA toolkit (`nvcc`) through the `nvidia-cuda` feature and requests the GPU
in `docker-compose.yml`, so the host needs no CUDA toolkit of its own:

```bash
cargo build --release --features cuda
cargo run --release --features cuda -- serve \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --context-path resources/memory.md
```

`auto` selects `F16` weights on CUDA. Verify the GPU is visible with
`nvidia-smi` inside the container (`nvcc` is put on `PATH` automatically).

#### Measured GPU latency

`reports_latency_breakdown` (release, Qwen2.5-1.5B, `F16`, RTX 3070):

| Prefix | prefill | 5 batched suffixes | single next token |
|---|---|---|---|
| 64 | 14 ms | 36 ms | 52 ms |
| 256 | 31 ms | 81 ms | 65 ms |
| 1024 | 154 ms | 379 ms | 64 ms |

## Tests

```bash
cargo test
cargo test -- --ignored --nocapture
cargo clippy --all-targets
cargo fmt --check
```

- `cargo test`: unit tests (probability calibration, labels, prompt rendering,
  session-cache eviction, CPU flash attention against a matmul reference) and
  API integration via `actix_web::test` with a mocked evaluator
  (`MockEvaluator`), without downloading weights.
- `cargo test -- --ignored --nocapture`: *live* tests, marked with `#[ignore]`;
  download the real weights once. They validate the vendored model against
  upstream (`vendored_forward_matches_candle_llama`,
  `vendored_forward_matches_candle_qwen2`), the batched broadcast against
  sequential scoring (`batched_suffixes_match_sequential`), the structured
  tokenization against the monolithic prompt
  (`prefix_reuse_matches_monolithic_forward`), the session cache correctness
  and gain (`session_cache_reuses_state_prefix_and_preserves_answers`) and the
  latency breakdown (`reports_latency_breakdown`). Do not run in CI.
- `cargo clippy --all-targets` / `cargo fmt --check`: lint and formatting. Do
  not use `--all-features` on Linux (the Metal feature needs macOS).

## Architecture

- **HTTP (`src/api/`)**: `actix-web` with `web::scope("/v1")` (`POST
  /v1/systemone`, `GET /v1/models`) and the `GET /health` and
  `GET /health/live` routes. *Handlers* receive a `SharedState` (evaluator +
  model name + context name + startup time) via `web::Data`.
- **Orchestration (Rig)**: the evaluator builds the conversation history with
  `rig-core` types — the loaded context as a `system` message and the
  `state` + question text as a `user` message — and renders the single prompt
  evaluated by the model. Rig does not expose logprobs: it organizes, it does not
  score.
- **Scoring (Candle)**: `CandleEvaluator` prefills the fixed system context
  into a KV-cache **once at startup**. Each request then performs a *single
  shared prefill* of the tokens common to all its questions (the `state`
  prefix plus the common question prefix). That shared cache is **broadcast
  across the attention batch dimension** and every question suffix is evaluated
  in a **single batched forward pass** — the common prefix is never recomputed
  per question. The state prefix (`system + state`) is additionally retained in
  a bounded LRU cache keyed by a hash of the state, so a reappearing state skips
  its forward pass entirely. It reads the *logit* of each answer-label token
  (`A`, `B`, …) at each row's last position and calibrates the distribution
  (binary softmax for `noul`, temperature softmax for `choice`/`score`). This is
  the "parallel evaluation via KV-cache broadcasting" pattern; the broadcast
  cache is discarded after the batched pass and the cached prefix is never
  mutated.
- **Parallel forward (`src/infrastructure/parallel_llama.rs`)**: a vendored,
  batch-broadcastable Llama implementation. Upstream `candle-transformers`
  keeps its KV-cache private and only returns last-position logits, which
  prevents both broadcasting and per-row collection; the vendored variant
  exposes `broadcast_batch` and `logits_from_hidden_at_positions`. On the CPU
  it runs the fused flash-style attention kernel, keeping grouped query
  attention grouped; on accelerators it keeps the matmul/softmax path. It is
  validated against upstream `Llama`/`Qwen2` by ignored equivalence tests.
- **Context (`ContextProvider`)**: currently implemented as
  `FileContextProvider` (e.g.: `--context-path resources/memory.md`); the
  interface allows swapping the source for retrieval (RAG) in the future without
  changing *handlers*, evaluator or API.
