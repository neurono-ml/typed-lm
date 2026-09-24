# Training and quantization

`typed-lm-trainer` is a binary with two subcommands:

- `train` — trains with one of four methods (`lora`, `qlora`, `full`,
  `from-scratch`), either an adapter over a frozen base checkpoint or the full
  set of parameters;
- `quantize` — applies **post-training quantization (PTQ)** to FP8/FP4 and merges
  an optional adapter first.

Both optimize the **cross-entropy at the decision position** (the last prompt
token, restricted to the candidate labels) — exactly the position the server
reads at inference — so the training tunes the behavior the API actually uses.

## Training methods

The `--method` flag selects how the model starts and which parameters it
trains:

| `--method` | Base origin | Trainable parameters | Output artifacts |
|---|---|---|---|
| `lora` (default) | checkpoint, frozen | LoRA `A`/`B` only | `adapter.safetensors` + `adapter_config.json` |
| `qlora` | checkpoint, quantized (dequantized on load), frozen | LoRA `A`/`B` only | `adapter.safetensors` + `adapter_config.json` |
| `full` | checkpoint | every parameter | `model.safetensors` + `config.json` + `tokenizer.json` |
| `from-scratch` | random initialization | every parameter | `model.safetensors` + `config.json` + `tokenizer.json` |

`lora` and `qlora` are the adapter methods: the base is never duplicated, and
only the adapter tensors are stored. `full` and `from-scratch` train every
parameter and write a **complete dense checkpoint** that `typed-lm-serve` serves
directly.

## Prerequisites

- Rust stable.
- A local base checkpoint (the trainer requires a path on disk; Hub identifiers
  must be downloaded first — the server loader can prefetch them).
- Optional: a CUDA build for GPU training.

Build variants:

```bash
cargo build --release -p typed-lm-trainer            # CPU
cargo build --release -p typed-lm-trainer --features cuda   # GPU
```

## Device selection

Every subcommand accepts `--device`:

| Value | Meaning |
|---|---|
| `auto` (default) | CUDA when available, otherwise CPU |
| `cpu` | Force the CPU |
| `cuda` | Force the first CUDA device (fails when no GPU is present) |

Training keeps master weights and optimizer state in **F32** on every device and
uses **BF16** only as the compute dtype on GPU (`PrecisionPolicy`), so adapter
quality is functionally equivalent between CPU and CUDA.

## Dataset format (Jev-native)

Each record mirrors the Jev request contract (`state` + a map of `questions`) and
adds an `answer` to every question. Every record must be a JSON **object** with:

- `state` — a string (free text) or any JSON value (structured state);
- `questions` — a non-empty object mapping a question name to its definition;
- each question — the serving-contract shape (`type` + `instructions`, plus
  `criteria` where required) **plus** an `answer` string.

Unknown keys are rejected: a question is the contract shape plus `answer` only.

### Accepted files

| Input | Description |
|---|---|
| `.jsonl` | One JSON object per line; blank lines are skipped. |
| `.json` (object) | A single record object. |
| `.json` (array) | An array of record objects. |
| directory | Scanned **recursively** for `.jsonl`/`.json`; files are sorted. |

The file extension is case-insensitive. Errors name the file, and the line for
JSONL, so a malformed dataset is fixable without guesswork.

### Question shapes and `answer`

The `answer` is **semantic** and must be one of the question's candidates:

- `noul` → `"yes"` / `"no"`, or the labels declared in `criteria`
  (`{"yes": "...", "no": "..."}`; default `"Yes"`/`"No"`);
- `choice` → the option **name** (a key of `criteria`);
- `score` → the level **name** (an element of the `criteria` array).

`criteria` per type:

- `noul` — optional object `{"yes": "...", "no": "..."}` (also accepts the
  aliases `true`/`false`);
- `choice` — required object mapping each option name to a description (the
  value may be `null` when no description is needed);
- `score` — required array of 2 to 10 level names, in increasing order.

### JSONL example (all three question types)

```jsonl
{"state": "charged twice", "questions": {"refund": {"type": "noul", "instructions": "Refund?", "answer": "yes"}, "dept": {"type": "choice", "instructions": "Dept?", "criteria": {"billing": "Payments", "technical": "Bugs"}, "answer": "technical"}, "urg": {"type": "score", "instructions": "Urgent?", "criteria": ["Routine", "Urgent", "Emergency"], "answer": "Urgent"}}}
{"state": "package arrived broken", "questions": {"refund": {"type": "noul", "instructions": "Refund?", "answer": "no"}, "urg": {"type": "score", "instructions": "Urgent?", "criteria": ["Routine", "Urgent", "Emergency"], "answer": "Emergency"}}}
```

A ready example lives in `resources/dataset.jsonl`.

### Single-record JSON example

```json
{
  "state": "charged twice",
  "questions": {
    "refund": { "type": "noul", "instructions": "Refund?", "answer": "yes" },
    "dept": {
      "type": "choice",
      "instructions": "Dept?",
      "criteria": { "billing": "Payments", "technical": "Bugs" },
      "answer": "technical"
    },
    "urg": {
      "type": "score",
      "instructions": "Urgent?",
      "criteria": ["Routine", "Urgent", "Emergency"],
      "answer": "Urgent"
    }
  }
}
```

### Array-of-records JSON example

```json
[
  {
    "state": "charged twice",
    "questions": {
      "dept": {
        "type": "choice",
        "instructions": "Dept?",
        "criteria": { "billing": null, "technical": null },
        "answer": "billing"
      }
    }
  },
  {
    "state": "a dependant's medical device stopped working",
    "questions": {
      "urg": {
        "type": "score",
        "instructions": "Urgent?",
        "criteria": ["Routine", "Urgent", "Emergency"],
        "answer": "Emergency"
      }
    }
  }
]
```

### Custom `noul` labels

`noul` accepts custom yes/no labels via `criteria`; the `answer` may use either
the label or the plain `yes`/`no` token:

```json
{
  "state": "the customer requests a manager review",
  "questions": {
    "approved": {
      "type": "noul",
      "instructions": "The request is approved.",
      "criteria": { "yes": "Approved", "no": "Rejected" },
      "answer": "Rejected"
    }
  }
}
```

### Answer labels

The trainer maps the semantic `answer` to the spreadsheet label (`A`, `B`, …)
the server scores:

| Type | Candidate order | Example `answer` → label |
|---|---|---|
| `noul` | `yes`, `no` | `yes` → `A`, `no` → `B` |
| `choice` | option names sorted lexicographically | `technical` → `B` (with `billing`, `technical`) |
| `score` | declared level order | `Urgent` → `B` (with `Routine`, `Urgent`, `Emergency`) |

## Train an adapter

```bash
cargo run --release -p typed-lm-trainer -- train \
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
| `--dataset` | Dataset file or directory | required |
| `--output-directory` | Adapter destination | `output/train` |
| `--method` | `lora`, `qlora`, `full` or `from-scratch` | `lora` |
| `--seed` | Initialization seed for `from-scratch` | `42` |
| `--configuration-file` | Optional TOML file; explicit CLI flags win | — |
| `--tokenizer-file` | `tokenizer.json` for `from-scratch` (checkpoint methods read it from the checkpoint) | — |
| `--lora-rank` / `--lora-alpha` | LoRA rank and alpha (scale `alpha/rank`) | `16` / `32` |
| `--lora-dropout` | Adapter dropout | `0` |
| `--epochs` | Epochs | `3` |
| `--batch-size` | Batch per step (items sharing a state are bucketed) | `4` |
| `--gradient-accumulation-steps` | Micro-batches accumulated before a step | `1` |
| `--learning-rate` | Peak LR (warmup + cosine decay) | `1e-4` |
| `--warmup-steps` | Warmup steps | `10` |
| `--weight-decay` | AdamW weight decay | `0` |
| `--maximum-gradient-norm` | Global gradient-norm clipping | `1` |
| `--max-sequence-length` | Maximum prompt length; longer items are skipped | `1024` |
| `--minimum-improvement` | Minimum improvement that resets patience | `0` |
| `--early-stop-patience` | Epochs without improvement before stopping (`0` disables) | `0` |
| `--quantization` | `none`, `fp8` or `fp4` | `none` |
| `--quantization-mode` | `post-training` or `training` | `post-training` |
| `--device` | `auto`, `cpu` or `cuda` | `auto` |

Output in `--output-directory`: `adapter.safetensors` + `adapter_config.json`.

### QLoRA

`--method qlora --quantization fp4 --quantization-mode training` keeps the base
quantized (dequantized on load) throughout training.

### Adapter output

Written to `--output-directory`:

```
output/train/
├── adapter.safetensors      # LoRA tensors only (base is never duplicated)
└── adapter_config.json      # rank, alpha, source model, architecture
```

`adapter_config.json`:

```json
{
  "rank": 16,
  "alpha": 32.0,
  "model_identifier": "/path/to/local/checkpoint",
  "architecture": "llama"
}
```

The adapter tensor names are `model.layers.{index}.<projection>.lora_a` and
`.lora_b` for `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and
`down_proj`. Only the adapter is stored — the frozen base is never copied.

## Full fine-tuning (`--method full`)

`--method full` keeps the existing checkpoint weights but makes **every**
parameter trainable (embeddings, norms, all attention and MLP projections, and
the language-model head). It reuses the same forward pass and decision-position
loss as the adapter methods. The geometry is read from the checkpoint
`config.json`, so no geometry flags are required:

```bash
cargo run --release -p typed-lm-trainer -- train \
  --method full \
  --model-id /path/to/local/checkpoint \
  --dataset resources/dataset.jsonl \
  --output-directory output/full \
  --epochs 3 --batch-size 4 --learning-rate 1e-4
```

## From scratch / full training

`--method from-scratch` initializes **random weights** over an explicit geometry
and trains **all** parameters. Because there is no checkpoint to read the shape
or the tokenizer from, both must be supplied:

- the geometry via the flags (`--architecture`, `--vocab-size`, `--hidden-size`,
  `--num-hidden-layers`, `--num-attention-heads`, …) or the TOML `[model]`
  section;
- a tokenizer via `--tokenizer-file` (or the TOML `[tokenizer] file` key).

Initialization is deterministic and reproducible from `--seed` (default `42`);
RMSNorm weights are set to `1.0` and biases to `0.0`.

```bash
cargo run --release -p typed-lm-trainer -- train \
  --method from-scratch \
  --architecture qwen2 \
  --vocab-size 151936 \
  --hidden-size 512 \
  --intermediate-size 2048 \
  --num-hidden-layers 8 \
  --num-attention-heads 8 \
  --num-key-value-heads 4 \
  --max-position-embeddings 1024 \
  --tokenizer-file /path/to/tokenizer.json \
  --dataset resources/dataset.jsonl \
  --output-directory output/scratch \
  --seed 42 \
  --epochs 3 --batch-size 4 --learning-rate 1e-4
```

The run writes a **complete dense checkpoint** into `--output-directory`:

```
output/scratch/
├── model.safetensors   # canonical Hugging Face tensor names, all parameters
├── config.json         # reparseable model configuration
└── tokenizer.json      # copy of the tokenizer passed with --tokenizer-file
```

This directory is a fully resolved checkpoint: point `typed-lm-serve` at it
without any further copy or merge step:

```bash
cargo run --release -p typed-lm-serve -- --model-id output/scratch
```

> **Not a general-purpose assistant.** A from-scratch model is trained only on
> the Jev decision objective over your dataset (cross-entropy at the decision
> position), not with general causal-LM pretraining. It does **not** acquire
> language understanding: it validates the architecture, the dataset and the
> training pipeline end to end (and can be served), but it is not a
> general-purpose assistant. Use a pretrained checkpoint (`lora`, `qlora` or
> `full`) for real routing quality.

## Configuration file (TOML)

Any `train` invocation can be shortened with `--configuration-file <path.toml>`.
Values resolve with the precedence **CLI flag > TOML key > default**: an explicit
flag always wins, an absent flag takes the TOML value, and an absent key takes
the built-in default. Section headers are `[run]`, `[model]`,
`[initialization]`, `[dataset]` and `[tokenizer]`.

```bash
cargo run --release -p typed-lm-trainer -- train \
  --configuration-file training.toml \
  --output-directory output/override   # explicit flag beats the TOML value
```

See [Configuration file reference](configuration-file.md) for every key, its CLI
flag and a worked precedence example.

## Quantize (PTQ)

```bash
cargo run --release -p typed-lm-trainer -- quantize \
  --model-id /path/to/local/checkpoint \
  --adapter-directory output/train \
  --quantization fp8 \
  --output-directory output/quantized
```

- `--adapter-directory` is optional: when given, the adapter is merged into the
  base weights before quantization.
- Output: `model.safetensors` + `quantization_config.json`.
- `fp8` uses `F8_E4M3` tensors with per-channel scaling (`*_scale`).
- `fp4` (MXFP4) writes E2M1 nibbles packed into `U8` plus `F8E8M0` exponents
  (`*_scale`), because safetensors/Candle cannot convert `F4` directly. Both
  formats are dequantized to dense F32 on load.
- Quantization is a host-side operation: the merged weights are staged through
  the CPU (candle has no CUDA kernel for the FP8 cast) and the artifact is
  written CPU-resident.

### Quantized output

Written to `--output-directory`:

```
output/quantized/
├── model.safetensors         # dense, FP8 or packed FP4 weights (+ metadata)
└── quantization_config.json  # scheme and block size
```

`quantization_config.json`:

```json
{
  "scheme": "fp8",
  "block_size": 32
}
```

Stored tensor layout per scheme:

- `none` — dense `F32` `model.safetensors`, no extra tensors.
- `fp8` — `F8_E4M3` weights, each paired with a per-channel
  `<weight>_scale` tensor.
- `fp4` (MXFP4) — E2M1 nibbles packed into `U8` (`<weight>`), a `U8`
  `<weight>_scale` exponent tensor (F8E8M0) and a `<weight>_shape` tensor
  recording the original shape (the packed tensor is flat).

The `quantize` output holds weights only; copy the base `config.json` and
`tokenizer.json` next to it before serving (see below).

## Serving the artifact

The trainer's `quantize` output contains only the weights. Copy the base
`config.json` and `tokenizer.json` next to `model.safetensors` so the directory
resolves as a complete checkpoint, then point the server at it:

```bash
cp /path/to/local/checkpoint/config.json    output/quantized/
cp /path/to/local/checkpoint/tokenizer.json output/quantized/

cargo run --release -p typed-lm-serve --features cuda -- \
  --model-id output/quantized \
  --context-path resources/memory.md
```

The server detects the FP8/FP4 artifact and dequantizes it on load.

## Tests

```bash
# Fast, no download: dummy checkpoint + library pipeline.
cargo test -p typed-lm-trainer

# Binary-level E2E (train -> quantize -> serve over HTTP) on CPU.
cargo test -p typed-lm-serve --test end_to_end

# Live GPU E2E: real checkpoint, LoRA on CUDA, FP8 PTQ (needs network + GPU).
cargo test -p typed-lm-trainer --features cuda --test live_gpu_e2e -- --ignored --nocapture
```

A reproducible script that runs the CPU E2E end to end lives in
`temporary/e2e/run_e2e_cpu.sh`.

## Next steps

- [Running the server](running.md) — serve the trained/quantized artifact.
- [Calling the API](api.md) — the Jev contract and `curl` examples.
