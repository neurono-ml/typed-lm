# typed-lm-trainer

[![Docs](https://img.shields.io/badge/docs-neurono--ml.github.io-6d28d9)](https://neurono-ml.github.io/typed-lm/)

**LoRA/QLoRA** fine-tuning, **full-parameter** training (`full` and
`from-scratch`) and **post-training quantization** (**PTQ FP8/FP4**) for the
models served by `typed-lm-serve`. The trainer optimizes the **cross-entropy at
the decision position** — the last token of the prompt, restricted to the
candidate labels — which is exactly the position the server reads at inference
time, ensuring the adapter tunes the behaviour the API actually uses.

> **📖 Full documentation:** <https://neurono-ml.github.io/typed-lm/>
> Training tutorial: <https://neurono-ml.github.io/typed-lm/training/index.html> ·
> Preparing datasets: <https://neurono-ml.github.io/typed-lm/training/datasets.html> ·
> Configuration (TOML): <https://neurono-ml.github.io/typed-lm/reference/configuration-file.html>

## Subcommands

```bash
cargo run -p typed-lm-trainer -- train --help
cargo run -p typed-lm-trainer -- quantize --help
```

## Dataset format (Jev-native)

Each record mirrors the Jev request contract (`state` + a `questions` map) and
adds an `answer` field to every question. The `answer` is **semantic**:

- `noul` → `"yes"` / `"no"` (or the labels declared in `criteria`);
- `choice` → the option name;
- `score` → the level name.

The trainer maps the `answer` to the spreadsheet label (`A`, `B`, …) that the
server scores.

Accepted files: `.jsonl` (one record per line) or `.json` (a single object or an
array). A directory is scanned recursively.

```jsonl
{"state": "charged twice", "questions": {"refund": {"type": "noul", "instructions": "Refund?", "answer": "yes"}, "dept": {"type": "choice", "instructions": "Dept?", "criteria": {"billing": "Payments", "technical": "Bugs"}, "answer": "technical"}}}
{"state": "package arrived broken", "questions": {"urg": {"type": "score", "instructions": "Urgent?", "criteria": ["Routine", "Urgent", "Emergency"], "answer": "Urgent"}}}
```

An example lives in `resources/dataset.jsonl`.

## Training methods

`--method` selects the trained slice and the produced artifact:

| `--method` | Trainable parameters | Output in `--output-directory` |
|---|---|---|
| `lora` | LoRA adapters over a frozen dense checkpoint | `adapter.safetensors` + `adapter_config.json` |
| `qlora` | LoRA adapters over a quantized base checkpoint (dequantized on load) | `adapter.safetensors` + `adapter_config.json` |
| `full` | Every parameter, starting from an existing checkpoint | `model.safetensors` + `config.json` + `tokenizer.json` |
| `from-scratch` | Every parameter, starting from randomly initialized weights | `model.safetensors` + `config.json` + `tokenizer.json` |

`full` and `from-scratch` use `save_full_checkpoint` and write a complete
checkpoint (dense weights + `config.json` + `tokenizer.json`). That artifact is
**serveable directly** by `typed-lm-serve`, with no adapter merge step. The
`full`/`from-scratch` methods require the model geometry: supply it from a
checkpoint (`--model-id` with a `config.json`) or through the geometry flags
below.

## Training an adapter

```bash
cargo run -p typed-lm-trainer -- train \
  --model-id /path/to/local/checkpoint \
  --dataset resources/dataset.jsonl \
  --output-directory output/train \
  --method lora \
  --lora-rank 16 --lora-alpha 32 \
  --epochs 3 --batch-size 4 --learning-rate 1e-4
```

Main flags:

| Flag | Description | Default |
|---|---|---|
| `--model-id` | Local base checkpoint (directory) | `Qwen/Qwen2.5-1.5B-Instruct` |
| `--dataset` | Dataset file or directory (or TOML `[dataset] path`) | (required) |
| `--output-directory` | Destination for the adapter or the complete checkpoint | `output/train` |
| `--method` | `lora`, `qlora`, `full` or `from-scratch` | `lora` |
| `--lora-rank` / `--lora-alpha` | LoRA rank and alpha (scale `alpha/rank`) | `16` / `32` |
| `--lora-dropout` | Adapter dropout | `0` |
| `--epochs` | Epochs | `3` |
| `--batch-size` | Batch per step (items sharing a state are bucketed) | `4` |
| `--gradient-accumulation-steps` | Accumulated micro-batches | `1` |
| `--learning-rate` | Peak LR (warmup + cosine) | `1e-4` |
| `--warmup-steps` | Warmup steps | `10` |
| `--weight-decay` | AdamW weight decay | `0` |
| `--maximum-gradient-norm` | Global gradient-norm clipping | `1` |
| `--max-sequence-length` | Maximum prompt length; longer items are skipped | `1024` |
| `--minimum-improvement` | Minimum improvement that resets patience | `0` |
| `--early-stop-patience` | Epochs without improvement before stopping (`0` disables) | `0` |
| `--quantization` | `none`, `fp8` or `fp4` | `none` |
| `--quantization-mode` | `post-training` or `training` | `post-training` |
| `--device` | `auto` (CUDA > Metal > CPU), `cpu` or `cuda` | `auto` |
| `--seed` | Deterministic initialization seed (`from-scratch`) | `42` |
| `--tokenizer-file` | `tokenizer.json` used by `from-scratch` (no checkpoint, no tokenizer) | (none) |
| `--configuration-file` | Optional TOML; explicit CLI flags take precedence | (none) |

For `lora`/`qlora` the output in `--output-directory` is `adapter.safetensors` +
`adapter_config.json`; for `full`/`from-scratch` it is a complete checkpoint
(`model.safetensors` + `config.json` + `tokenizer.json`).

### Model geometry (`full` / `from-scratch`)

Used when the geometry does not come from a checkpoint `config.json`. All are
optional on the CLI (default `none`) and can come from the TOML file:

| Flag | Description |
|---|---|
| `--architecture` | Family (`llama`, `qwen2`, `qwen3`, `mistral`, `gemma`, `gemma2`, `gemma3`) |
| `--hidden-size` | Hidden dimension |
| `--intermediate-size` | Feed-forward intermediate dimension |
| `--num-hidden-layers` | Number of transformer blocks |
| `--num-attention-heads` | Number of query heads |
| `--head-dim` | Head dimension (default: `hidden_size / num_attention_heads`) |
| `--num-key-value-heads` | Number of key/value heads (GQA) |
| `--vocab-size` | Vocabulary size |
| `--max-position-embeddings` | Maximum sequence length |
| `--rope-theta` | Rotary embedding base frequency |
| `--rms-norm-eps` | RMS normalization epsilon |
| `--tie-word-embeddings` | Input and output embeddings share weights |
| `--attention-bias` | Attention projections carry a bias |
| `--sliding-window` | Sliding-window size |
| `--sliding-window-pattern` | Gemma3 global/local alternation |
| `--rope-local-base-frequency` | Gemma3 local RoPE base frequency |
| `--query-pre-attention-scalar` | Gemma2/Gemma3 attention scaling denominator |
| `--logit-softcapping` | Gemma2/Gemma3 `final_logit_softcapping` |
| `--attention-logit-softcapping` | Gemma2/Gemma3 `attn_logit_softcapping` |

Fields left unset take the family default (`head_dim` derivation, soft-caps,
Gemma3 local RoPE and window, attention bias), so the emitted `config.json` is
always serveable by `typed-lm-serve`.

The complete TOML reference (`[run]`, `[model]`, `[initialization]`,
`[dataset]`, `[tokenizer]`) is in the
[configuration file reference](https://neurono-ml.github.io/typed-lm/reference/configuration-file.html).

## Training from scratch (`from-scratch`)

```bash
cargo run -p typed-lm-trainer -- train \
  --dataset resources/dataset.jsonl \
  --output-directory output/scratch \
  --method from-scratch \
  --architecture qwen2 \
  --hidden-size 512 --intermediate-size 2048 \
  --num-hidden-layers 8 --num-attention-heads 8 --num-key-value-heads 2 \
  --vocab-size 151936 --max-position-embeddings 1024 \
  --tokenizer-file /path/to/tokenizer.json \
  --seed 42 --device auto
```

Without `--model-id`, the prompt format (spreadsheet label) is fixed by
`--architecture`; with no checkpoint, the tokenizer must be supplied via
`--tokenizer-file` (or `[tokenizer] file` in the TOML).

## Quantize (PTQ)

```bash
cargo run -p typed-lm-trainer -- quantize \
  --model-id /path/to/local/checkpoint \
  --adapter-directory output/train \
  --quantization fp8 \
  --output-directory output/quantized
```

- `--adapter-directory` is optional: when supplied, the adapter is merged into
  the base weights before quantization; when omitted, the base checkpoint is
  quantized as-is.
- `--model-id` accepts any dense checkpoint directory, including one written by
  `full`/`from-scratch`, so a from-scratch artifact can be quantized directly.
- Output: `model.safetensors` + `quantization_config.json`.
- `fp8` uses `F8_E4M3` tensors with per-channel scaling (`*_scale`).
- `fp4` (MXFP4) writes E2M1 nibbles packed into `U8` plus `F8E8M0` exponents
  (`*_scale`), because safetensors/Candle cannot convert `F4`. The loader
  dequantizes both formats to dense F32 on load.

## CPU/CUDA parity

`PrecisionPolicy { master: F32, compute: F32(CPU)/BF16(GPU), reduction: F32 }`:
master weights and optimizer stay in F32 on both devices, BF16 is only a
compute dtype on GPU, and reductions are always F32. Adapter quality is
device-independent; parity is functional, not of speed.

## Tests

```bash
cargo test -p typed-lm-trainer
cargo test -p typed-lm-trainer -- --ignored   # real-weight cases
```

The E2E tests (dummy overfit, FP8/FP4 export) run on CPU with a tiny checkpoint
in `tests/support/`, with no download. The live GPU test
(`tests/live_gpu_e2e.rs`) downloads a real checkpoint, trains LoRA on CUDA and
exports FP8:

```bash
cargo test -p typed-lm-trainer --features cuda --test live_gpu_e2e -- --ignored --nocapture
```

Detailed documentation is in the
[training tutorial](https://neurono-ml.github.io/typed-lm/training/index.html).
