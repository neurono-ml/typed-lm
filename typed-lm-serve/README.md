# typed-lm-serve

[![Docs](https://img.shields.io/badge/docs-neurono--ml.github.io-6d28d9)](https://neurono-ml.github.io/typed-lm/)

Jev-compatible, single-forward-pass semantic routing server for dense decoder
models, built on [Candle](https://github.com/huggingface/candle).

The server answers typed questions (`noul`, `choice`, `score`) from the logits
of a local checkpoint in one forward pass — it never generates free text. The
main route is `POST /v1/systemone`; `GET /v1/models`, `GET /health` and
`GET /health/live` complete the API.

> **📖 Full documentation:** <https://neurono-ml.github.io/typed-lm/>
> Quick start: <https://neurono-ml.github.io/typed-lm/quickstart.html> ·
> API: <https://neurono-ml.github.io/typed-lm/guides/api.html> ·
> Running: <https://neurono-ml.github.io/typed-lm/guides/running.html>

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

See the [documentation site](https://neurono-ml.github.io/typed-lm/) and
[Running the server](https://neurono-ml.github.io/typed-lm/guides/running.html)
for flags, layouts and the full Jev contract.

## License

Apache-2.0. See [LICENSE](../LICENSE).
