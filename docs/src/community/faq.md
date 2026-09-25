# FAQ

## What is typed-lm, in one sentence?

A Rust server and trainer that turn dense decoder models into a typed,
single-forward-pass semantic routing API compatible with the Jev contract.

## Does it generate text?

No. Every answer is a typed value extracted from the model's logits at a single
decision position. There is no free-text generation.

## Which models can it serve?

Dense decoder families detected from `model_type`: Llama, Qwen2, Qwen3, Mistral,
Gemma, Gemma2 and Gemma3. MoE and MLA families (`mixtral`, `qwen3_moe`,
`deepseek_v2`, `deepseek_v3`) are rejected.

## Can I serve GGUF?

Yes, for **Qwen2** only. Other architectures served from GGUF are rejected; convert
them to a dense format first.

## Do I need a GPU?

No. The CPU path works, and a GGUF `Q4_K_M` checkpoint with the `mkl` feature is
the recommended CPU mode. CUDA and Metal are optional accelerators.

## How do I add a system context?

Start the server with `--context-path resources/memory.md`. The context is
prefilled once and prepended to every state.

## Why is the same state fast the second time?

The session prefix cache stores the KV-cache of `system + state` in a bounded LRU
keyed by a state hash. Repeated states skip the prefill.

## What is the difference between `choice` and `score`?

`choice` picks one of a closed set; `score` rates the state on ordered levels and
returns an expected value. Use `noul` for booleans.

## What does `confidence` mean?

How concentrated the distribution is. It tells you whether to act on the answer.
`noul` has no separate confidence; its value is the calibrated probability.

## How do I train on my own labels?

Write a Jev-native dataset (`state` + `questions` + `answer`) and run the trainer
with `--method lora`. See [Preparing datasets](../training/datasets.md).

## Why is my from-scratch model not smart?

A from-scratch run trains only the decision objective, not general language
understanding. Use a pretrained checkpoint for real routing quality.

## Can I serve a LoRA adapter directly?

No. Merge it first, for example with `quantize --adapter-directory`, then serve the
resulting checkpoint.

## Which format should I quantize to?

`fp8` for a good size/accuracy balance; `fp4` when memory is tight. Both are
dequantized to dense F32 on load.

## Where is the API reference?

[HTTP API reference](../reference/api.md) and
[Calling the API](../guides/api.md).

## How is the site indexed for AI assistants?

See [Search and AI indexing](./ai-indexing.md).

## Next steps

- [What is typed-lm?](../introduction.md)
- [Training overview](../training/index.md)
- [Troubleshooting](../training/troubleshooting.md)
