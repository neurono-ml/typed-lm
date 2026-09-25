# Trainer flags

`typed-lm-trainer` has two subcommands: `train` and `quantize`. Both accept
`--device`. Values resolve with the precedence **CLI flag > TOML key > default**.

## `train`

| Flag | Description | Default |
|---|---|---|
| `--model-id` | Local base checkpoint (directory) | `Qwen/Qwen2.5-1.5B-Instruct` |
| `--dataset` | Dataset file or directory | required |
| `--output-directory` | Destination | `output/train` |
| `--method` | `lora`, `qlora`, `full` or `from-scratch` | `lora` |
| `--seed` | Deterministic initialization seed (from-scratch) | `42` |
| `--configuration-file` | Optional TOML; explicit CLI flags win | — |
| `--tokenizer-file` | `tokenizer.json` for from-scratch | — |
| `--lora-rank` / `--lora-alpha` | LoRA rank and alpha | `16` / `32` |
| `--lora-dropout` | Adapter dropout | `0` |
| `--epochs` | Epochs | `3` |
| `--batch-size` | Batch per step | `4` |
| `--gradient-accumulation-steps` | Micro-batches per step | `1` |
| `--learning-rate` | Peak LR (warmup + cosine) | `1e-4` |
| `--warmup-steps` | Warmup steps | `10` |
| `--weight-decay` | AdamW weight decay | `0` |
| `--maximum-gradient-norm` | Gradient-norm clipping | `1` |
| `--max-sequence-length` | Maximum prompt length; longer items skipped | `1024` |
| `--minimum-improvement` | Improvement that resets patience | `0` |
| `--early-stop-patience` | Epochs without improvement before stopping (`0` disables) | `0` |
| `--quantization` | `none`, `fp8` or `fp4` | `none` |
| `--quantization-mode` | `post-training` or `training` | `post-training` |
| `--device` | `auto`, `cpu` or `cuda` | `auto` |

### Geometry flags (`full` / `from-scratch`)

`--architecture` (`llama`, `qwen2`, `qwen3`, `mistral`, `gemma`, `gemma2`,
`gemma3`), `--vocab-size`, `--hidden-size`, `--intermediate-size`,
`--num-hidden-layers`, `--num-attention-heads`, `--head-dim`,
`--num-key-value-heads`, `--max-position-embeddings`, `--rope-theta`,
`--rms-norm-eps`, `--tie-word-embeddings`, `--attention-bias`, `--sliding-window`,
`--sliding-window-pattern`, `--rope-local-base-frequency`,
`--query-pre-attention-scalar`, `--logit-softcapping`,
`--attention-logit-softcapping`.

## `quantize`

| Flag | Description | Default |
|---|---|---|
| `--model-id` | Dense checkpoint directory | required |
| `--adapter-directory` | Optional adapter to merge before quantizing | — |
| `--quantization` | `none`, `fp8` or `fp4` | `fp8` |
| `--output-directory` | Destination | `output/quantized` |
| `--device` | `auto`, `cpu` or `cuda` | `auto` |

## Examples

```bash
# Train a LoRA adapter.
typed-lm-trainer train \
  --model-id /path/to/local/checkpoint \
  --dataset resources/dataset.jsonl \
  --output-directory output/train \
  --method lora --epochs 3 --batch-size 4 --learning-rate 1e-4

# Quantize while merging the adapter.
typed-lm-trainer quantize \
  --model-id /path/to/local/checkpoint \
  --adapter-directory output/train \
  --quantization fp8 --output-directory output/quantized
```

## Related

- [Training LoRA and QLoRA adapters](../training/lora-qlora.md)
- [Configuration file (TOML)](./configuration-file.md)
- [CLI cheat sheet](./cheatsheet.md)
