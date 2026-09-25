# typed-lm-common

[![Docs](https://img.shields.io/badge/docs-neurono--ml.github.io-6d28d9)](https://neurono-ml.github.io/typed-lm/)

Shared library for the [`typed-lm`](https://github.com/neurono-ml/typed-lm)
workspace. It defines everything the server and the trainer must agree on:

- the **Jev contract** (`noul`/`choice`/`score` request and response types);
- label arithmetic and prompt rendering;
- checkpoint and architecture detection for the dense decoder families
  (Llama, Qwen2, Qwen3, Mistral, Gemma, Gemma2, Gemma3);
- device/dtype resolution (`PrecisionPolicy`);
- FP8/FP4 quantization helpers and tokenizer utilities.

Most users depend on `typed-lm-serve` or `typed-lm-trainer` rather than this
crate directly.

> **📖 Full documentation:** <https://neurono-ml.github.io/typed-lm/>

## License

Apache-2.0. See [LICENSE](../LICENSE).
