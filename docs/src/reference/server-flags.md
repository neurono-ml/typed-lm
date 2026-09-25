# Server flags

Every flag also reads an environment variable; precedence is
**CLI flag > environment variable > default**.

| CLI flag | Environment variable | Default | Description |
|---|---|---|---|
| `--host` | `HOST` | `0.0.0.0` | Bind address |
| `--port` | `PORT` | `8080` | Bind port |
| `--model-id` | `MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | Hub id or local checkpoint path |
| `--model-revision` | `MODEL_REVISION` | `main` | Hub revision |
| `--weights-file` | `WEIGHTS_FILE` | auto-detected | Explicit weights file |
| `--tokenizer-file` | `TOKENIZER_FILE` | next to the weights | Tokenizer override |
| `--config-file` | `CONFIG_FILE` | next to the weights | Config override |
| `--context-path` | `CONTEXT_PATH` | missing = empty context | System context file |
| `--served-model-name` | `SERVED_MODEL_NAME` | `typed-lm` | Name clients request |
| `--model-dtype` | `MODEL_DTYPE` | `auto` | `auto` (F32 CPU, F16 CUDA/Metal) |
| `--session-cache-entries` | `SESSION_CACHE_ENTRIES` | `16` | Max cached prefixes |
| `--session-cache-tokens` | `SESSION_CACHE_TOKENS` | `32768` | Max cached tokens |
| `--hf-token` | `HF_TOKEN` | missing | Token for gated models |

## Examples

```bash
# Environment only.
PORT=9090 MODEL_ID=recogna-nlp/bode-1b-instruct typed-lm-serve

# CPU with MKL and a memory context.
typed-lm-serve --features mkl -- \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --context-path resources/memory.md
```

## Related

- [Running the server](../guides/running.md) — the guided version.
- [Deploying and operating](../guides/operations.md) — sizing and probes.
