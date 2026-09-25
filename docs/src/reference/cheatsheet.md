# CLI cheat sheet

One-page reference for the most common commands.

## Build

```bash
cargo build --release --workspace                      # everything, CPU
cargo build --release -p typed-lm-serve --features mkl # server, CPU + MKL
cargo build --release -p typed-lm-serve --features cuda # server, CUDA
```

## Serve

```bash
typed-lm-serve                                         # default model
typed-lm-serve --context-path resources/memory.md      # with context
typed-lm-serve --model-id output/quantized             # serve an artifact
```

## Ask

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request_mixed.json

curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/health/live
```

## Train

```bash
# LoRA adapter.
typed-lm-trainer train \
  --model-id /path/to/local/checkpoint \
  --dataset resources/dataset.jsonl \
  --output-directory output/train \
  --method lora --epochs 3 --batch-size 4 --learning-rate 1e-4

# From scratch (explicit geometry + tokenizer).
typed-lm-trainer train \
  --method from-scratch --architecture qwen2 \
  --vocab-size 151936 --hidden-size 512 --intermediate-size 2048 \
  --num-hidden-layers 8 --num-attention-heads 8 --num-key-value-heads 4 \
  --max-position-embeddings 1024 \
  --tokenizer-file /path/to/tokenizer.json \
  --dataset resources/dataset.jsonl --output-directory output/scratch \
  --seed 42 --epochs 3 --batch-size 4 --learning-rate 1e-4
```

## Quantize

```bash
typed-lm-trainer quantize \
  --model-id /path/to/local/checkpoint \
  --adapter-directory output/train \
  --quantization fp8 --output-directory output/quantized
```

## Test

```bash
cargo test --workspace                                 # unit + integration
cargo test --workspace -- --ignored --nocapture        # live tests (weights)
cargo test -p typed-lm-serve --test end_to_end          # train -> quantize -> serve
```

## Related

- [Server flags](./server-flags.md)
- [Trainer flags](./trainer-flags.md)
- [Configuration file (TOML)](./configuration-file.md)
