# Summary

## Introduction

- [Home](index.md)
- [What is typed-lm?](introduction.md)
- [Quick start](quickstart.md)

## Concepts

- [System One decisions](concepts/system-one.md)
- [State](concepts/state.md)
- [Questions (primitives)](concepts/primitives.md)
  - [Choice](concepts/choice.md)
  - [Score](concepts/score.md)
  - [Noul](concepts/noul.md)
- [Confidence](concepts/confidence.md)
- [Designing with typed decisions](concepts/how-to-build.md)
- [typed-lm and Jev](concepts/comparison-with-jev.md)

## Guides

- [Calling the API](guides/api.md)
- [Running the server](guides/running.md)
- [Deploying and operating](guides/operations.md)
- [Patterns](guides/patterns.md)
  - [Confidence-gated routing](guides/patterns/confidence-routing.md)
  - [Composite scoring](guides/patterns/composite-scoring.md)
  - [Intent routing](guides/patterns/intent-routing.md)
  - [Speculative fan-out](guides/patterns/fan-out.md)
- [Cookbooks](guides/cookbooks.md)
  - [Customer-support routing](guides/cookbooks/support-routing.md)
  - [Content-moderation guardrails](guides/cookbooks/moderation-guardrails.md)
  - [Passage re-ranking](guides/cookbooks/reranking.md)
  - [Document classification with confidence](guides/cookbooks/classification-confidence.md)

## Training tutorial

- [Training overview](training/index.md)
- [Preparing datasets](training/datasets.md)
- [Configuring a run (CLI and TOML)](training/configuration.md)
- [Choosing an architecture](training/architecture.md)
- [Training LoRA and QLoRA adapters](training/lora-qlora.md)
- [Full fine-tuning](training/full.md)
- [Training from scratch](training/from-scratch.md)
- [Quantization (FP8 and FP4)](training/quantization.md)
- [Serving a trained artifact](training/serving-artifacts.md)
- [Troubleshooting](training/troubleshooting.md)

## Reference

- [HTTP API reference](reference/api.md)
- [Server flags](reference/server-flags.md)
- [Trainer flags](reference/trainer-flags.md)
- [Configuration file (TOML)](reference/configuration-file.md)
- [Supported architectures](reference/architectures.md)
- [Artifact formats](reference/artifacts.md)
- [CLI cheat sheet](reference/cheatsheet.md)

## Engineering

- [Architecture](engineering/architecture.md)
- [Scoring and batched decoding](engineering/scoring.md)
- [Session prefix cache](engineering/session-cache.md)
- [Benchmarks](engineering/benchmarks.md)
- [Testing](engineering/testing.md)

## Community

- [Contributing](community/contributing.md)
- [FAQ](community/faq.md)
- [Search and AI indexing](community/ai-indexing.md)
- [License](community/license.md)
