# API usage examples

Request-body examples for `POST /v1/systemone`.
The answers depend on the facts in `resources/memory.md`
(the fictional GreenLeaf store).

## Starting the server

```bash
cargo run -p typed-lm-serve
```

The server starts by default at `http://127.0.0.1:8080`
without a context (no `--context-path`). Pass
`--context-path resources/memory.md` (or `CONTEXT_PATH`)
to anchor the answers on the example facts.

## Gated models (restricted access)

The default model (`Qwen/Qwen2.5-1.5B-Instruct`) is public
and requires no authentication. If you switch to a gated
model via `--model-id` (for example
`recogna-nlp/bode-1b-instruct`), you must accept the terms
of use on the model page on Hugging Face and export a token
with read permission before starting the server:

```bash
HF_TOKEN=hf_your_token_here cargo run -p typed-lm-serve -- --model-id recogna-nlp/bode-1b-instruct
```

Without `HF_TOKEN`, the weight download fails with a `401`
error (`failed to download ... status code 401`) — that is the
expected behaviour, not a bug.

## Environment variables

Every `typed-lm-serve` option can also come from the environment
(CLI flag > env > default): `HOST`, `PORT`,
`MODEL_ID`, `CONTEXT_PATH`,
`SERVED_MODEL_NAME` (and `HF_TOKEN` for `--hf-token`).

```bash
PORT=9090 MODEL_ID=recogna-nlp/bode-1b-instruct cargo run -p typed-lm-serve
```

## `curl` requests

Simple boolean (duplicate-charge refund eligibility):

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_noul.json
```

Boolean + routing + urgency (item damaged in transit):

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_mixed.json
```

Complex questions anchored on memory (defective medical-use
air purifier: eligibility, responsible department, urgency
and gift-card rule):

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_context.json
```

## Live integration test (requires real weights)

The live test loads the real model and validates answers anchored
on memory. It is marked `#[ignore]` and does **not** run in CI.
To run it (downloads the weights once into the local cache):

```bash
cargo test --workspace -- --ignored
```
