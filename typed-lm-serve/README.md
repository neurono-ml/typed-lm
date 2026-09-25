# typed-lm-serve

Jev-compatible, single-forward-pass semantic routing server for dense decoder
models, built on [Candle](https://github.com/huggingface/candle).

The server answers typed questions (`noul`, `choice`, `score`) from the logits
of a local checkpoint in one forward pass — it never generates free text. The
main route is `POST /v1/systemone`; `GET /v1/models`, `GET /health` and
`GET /health/live` complete the API.

## Install

```bash
# CPU build (default).
cargo install typed-lm-serve

# CUDA build (requires the CUDA toolkit).
cargo install typed-lm-serve --features cuda

# Apple Silicon GPU build (macOS, Metal).
cargo install typed-lm-serve --features metal
```

## Run

```bash
typed-lm-serve --help
typed-lm-serve
```

See the [workspace README](https://github.com/neurono-ml/typed-lm) and
[docs/running.md](https://github.com/neurono-ml/typed-lm/blob/main/docs/running.md)
for flags, layouts and the full Jev contract.

## License

Apache-2.0. See [LICENSE](../LICENSE).
