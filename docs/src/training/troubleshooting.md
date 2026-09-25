# Troubleshooting

This page collects the errors you are most likely to hit and what they mean.

## Dataset errors

A malformed dataset names the file and, for JSONL, the line.

| Symptom | Likely cause | Fix |
|---|---|---|
| `unknown field ...` | A question carries a key outside the contract plus `answer` | Remove extra keys |
| `invalid type` | A field has the wrong type (for example `epochs = "three"`) | Match the documented type |
| Empty or missing `questions` | A record without questions | Ensure `questions` is a non-empty object |
| `choice` rejected | No `criteria`, or `criteria` empty | Declare at least one option |
| `score` rejected | Fewer than 2 or more than 10 levels | Use 2 to 10 ordered levels |
| `answer` not matched | The `answer` is not a declared candidate | Use a level name or option key |
| Item skipped | Prompt longer than `--max-sequence-length` | Raise the limit or shorten states |

See [Preparing datasets](./datasets.md) for the exact shapes.

## Configuration errors

```text
configuration file error in 'training.toml': unknown field `epocs`, expected one of ...
configuration file error in 'training.toml': invalid type: string "three", expected usize ...
```

Unknown keys and wrong types are rejected rather than ignored. A missing file is
reported as an I/O error carrying the path. Remember the precedence:
**CLI flag > TOML key > default**.

## Model loading errors

| Symptom | Meaning | Fix |
|---|---|---|
| `401` on download | Gated model without a token | Accept the terms and set `HF_TOKEN` |
| MoE family rejected | `mixtral`, `qwen3_moe`, `deepseek_v2`/`deepseek_v3` | Not supported; use a dense family |
| GGUF non-Qwen2 rejected | GGUF serving is Qwen2-only | Convert to dense or use a Qwen2 GGUF |
| `GPTQ`/`AWQ` rejected | Unsupported quantization | Convert to FP8/FP4 or dense |

## API errors

Errors use the envelope `{"error": {"message": "..."}}`.

| Status | Situation |
|---|---|
| `422` | Malformed body or a question outside the contract |
| `404` | Unknown model name |
| `500` | Inference failure |

## Training misbehaves

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss does not move | Learning rate too small, or label imbalance | Raise the LR; rebalance the dataset |
| Perfect train loss, poor serving | Overfitting or leakage | Add data, lower rank, remove answers from states |
| Out of memory (full/from-scratch) | Every parameter keeps F32 master + moments | Use LoRA/QLoRA or a smaller geometry |
| Adapter has no effect | The adapter was not merged before serving | Run `quantize --adapter-directory` |

## Where to look next

- [Training overview](./index.md) — the pipeline.
- [Serving a trained artifact](./serving-artifacts.md) — artifact resolution.
- [FAQ](../community/faq.md) — common questions.
