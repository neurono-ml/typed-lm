# Guidelines for AI Agents (AGENTS.md)

## 1. Project Context
This project is a Rust monorepo for deterministic inference and training, focused on MLOps. It replaces the autoregressive text generation of traditional LLMs with a single-forward-pass classification architecture, using dense models from the **Llama, Qwen2, Qwen3, Mistral, Gemma, Gemma2 and Gemma3** families (for example Manacá-1B, Qwen2.5-1.5B-Instruct, the default model), detected automatically from the `model_type` field in `config.json` through the Hugging Face `candle` framework. MoE/MLA families (`mixtral`, `qwen3_moe`, `deepseek_v2`, `deepseek_v3`) are rejected with an actionable message.

The goal is to provide an ultra-low-latency API for semantic routing, strictly compatible with the **Jev** (TypeSafe AI) API specification, plus a LoRA/QLoRA/full/from-scratch trainer and a quantization (FP8/FP4) pipeline whose artifacts the server consumes directly.

## 2. Architecture and Technology Stack
*   **Language:** Rust (Edition 2021).
*   **Workspace:** `typed-lm` with three members:
    *   `typed-lm-common` (lib) — Jev contract, labels, prompt rendering, checkpoint and architecture detection, dense-architecture traits, device/dtype (`PrecisionPolicy`), quantization (FP8/FP4), tokenizer.
    *   `typed-lm-serve` (bin) — Actix server; a binary **with no subcommand** (top-level flags).
    *   `typed-lm-trainer` (bin+lib) — subcommands `train` and `quantize`.
*   **Web Server:** `actix-web` with asynchronous concurrency managed by `tokio`.
*   **Inference Engine:** `candle-core`, `candle-nn`, `candle-transformers`. The parametrized dense forward covers all seven dense families (Llama, Qwen2, Qwen3, Mistral, Gemma, Gemma2, Gemma3); the GGUF/GGML-quantized path implements **only Qwen2** and rejects the other architectures at load time.
*   **Training methods (`train`):** `--method lora|qlora|full|from-scratch`. `lora`/`qlora` train adapters over a frozen checkpoint; `full` tunes every parameter from a checkpoint; `from-scratch` initializes every parameter randomly (deterministic through `--seed`). An optional TOML configuration (`--configuration-file`) supplies any parameter with **CLI > TOML > default** precedence; `from-scratch` requires an explicit geometry (`[model]` or flags) and a `tokenizer.json` (`[tokenizer] file` or `--tokenizer-file`).
*   **Module Layout:**
    *   `typed-lm-common/src/architecture_traits.rs` — per-family dense capabilities (attention bias, explicit `head_dim`, sliding window, logit soft-capping, RMSNorm offset, embedding scale, local RoPE).
    *   `typed-lm-serve/src/api/` — response DTOs, errors, routes and Actix handlers.
    *   `typed-lm-serve/src/domain/` — the `Evaluator`/`MockEvaluator` trait.
    *   `typed-lm-serve/src/infrastructure/` — Candle: checkpoint loading, tokenizer, vendored parallel forward and the real evaluator.
    *   `typed-lm-serve/src/config/`, `typed-lm-serve/src/bootstrap/` — CLI and startup/server.
    *   `typed-lm-trainer/src/{dataset,model,training,quantization}/` — training and PTQ pipeline; `model/` includes `initialization`, `trainable_dense`, `trainable_full`, `trainable_linear` and `trainable_rms_norm`, plus the LoRA adapters.
    *   `typed-lm-trainer/src/configuration_file.rs`, `typed-lm-trainer/src/configuration_resolution.rs` — TOML schema and **CLI > TOML > default** resolution.
*   **State Management:** The model, the tokenizer and the `base_cache` (KV-cache of the system prompt) must be loaded once at application startup and shared with the `actix-web` workers through `actix_web::web::Data`. The per-request mutable cache must always be a clone of the base cache.
*   **Supported formats:** safetensors (single or sharded), GGUF (dense and GGML-quantized, e.g. `Q4_K_M`), PyTorch `.pth`/`.bin` and NumPy `.npz`. **FP8 (`F8_E4M3`/`F8_E5M2`) and FP4 (MXFP4) are supported via dequantization at load** to dense F32; `GPTQ`/`AWQ` are rejected with a clear message.
*   **CPU:** the fused `candle-nn` flash attention is used automatically on CPU (keeps GQA grouped); `--features mkl` enables Intel MKL BLAS. GGUF `Q4_K_M` is the recommended CPU mode.
*   **GPU:** `--features cuda` (automatic F16). The devcontainer installs the CUDA toolkit through the `nvidia-cuda` feature and reserves the GPU in `docker-compose.yml`; the host only needs the driver and the NVIDIA container toolkit.
*   **Training precision:** `PrecisionPolicy { master: F32, compute: F32(CPU)/BF16(GPU), reduction: F32 }` — master weight and optimizer in F32; BF16 only as compute on GPU. Functional CPU/CUDA parity, not of speed.
*   **Session cache:** the `system + state` prefix is retained in an LRU cache (by canonical hash of the state, bounded by number of entries and total tokens). The stored value is always a clone; the retained cache is never mutated.

## 3. API Contract (Jev Compatibility)
The application does not generate free text. It exposes routes that return structured types based on direct logit extraction. The main route is `POST /v1/systemone`, which accepts a JSON payload with `model`, `state` and `questions` (a map of questions).

Each question is typed in one of the formats below and can be combined in the same request:

*   **`noul`:** boolean decision → `{"type": "noul", "noul": 0.0 to 1.0}`.
*   **`choice`:** selects the best option from a restricted set → `{"type": "choice", "choice": "String", "probabilities": {...}, "confidence": 0.0 to 1.0}`.
*   **`score`:** continuous score mapped from vocabulary positions → `{"type": "score", "score": f32, "legend": {...}, "probabilities": {...}, "confidence": 0.0 to 1.0}`.

The API is complemented by `GET /v1/models`, `GET /health` and `GET /health/live`.

## 4. Code Rules (Unbreakable)
1.  **Code and Documentation Always in English (no exception):** **All** content versioned in the repository must be in English — with no exception. This includes:
    *   **Code:** variables, functions, structs, enums, modules, traits, comments, doc-comments (`///`, `//!`) and `tracing`/error messages.
    *   **Commit messages** and Pull Request descriptions.
    *   **Documentation and text files:** `README.md` (root and per crate), every `docs/*.md`, `examples/README.md`, `AGENTS.md`, example files (`*.http`, `*.jsonl`, resource `*.md`), comments in the devcontainer `*.toml`/`*.yml` and any other versioned artifact.
    *   **Identifiers of example data** (states, questions, answers) used in tests and fixtures.
    *   The **only** exception is direct chat communication with the maintainer, which may be in Portuguese. None of it may leak into a repository file.
    *   Before finishing any task, review the touched files for Portuguese text (e.g. `grep -rnE "[àáâãéêíóôõúçÀÁÂÃÉÊÍÓÔÕÚÇ]"` and keywords such as `não`, `para`, `servidor`, `modelo`, `arquivo`, `configuração`, `treino`, `requisição`, `exemplo`) and translate whatever you find.
    *   When creating a new file, write it directly in English; never translate afterwards.
2.  **Absolute Ban on Abbreviations:** The agent **must never** use abbreviations in variables, functions, structs or modules.
    *   *Wrong:* `calc_prob`, `ctx`, `req`, `init_kv`.
    *   *Correct:* `calculate_probability`, `context`, `request`, `initialize_key_value_cache`.
3.  **Cache Isolation:** `cache_base` must never be mutated during a user request. The route must clone the cache, run the forward pass from the system sequence length (`system_sequence_length`) and drop the clone at the end of the scope. The session cache (LRU by state hash) also only stores and returns clones; the context `cache_base` and every retained prefix remain immutable.
4.  **Error Handling:** Do not use `unwrap()` or `expect()` in production code. Map the Candle and Actix errors to a custom error struct that returns a standardized `HttpResponse::InternalServerError`.
5.  **Total Ban on `unwrap()`/`expect()` (including in tests):** No file under `src/` (including `#[cfg(test)]`, mocks, helpers and internal examples) may contain `.unwrap()`, `.expect(`, `.unwrap_err()` or `.expect_err()`. Methods that do not panic (`unwrap_or`, `unwrap_or_else`, `unwrap_or_default`) are allowed. In tests, functions must return `anyhow::Result<()>` (or `Result<_, EvaluationError>`) and propagate with `?`; error cases must be checked with `assert!(result.is_err())` + `let Err(error) = result else { return Ok(()); }`, and `Option` values with `ok_or_else(|| anyhow::anyhow!(...))?` or `unwrap_or`/`unwrap_or_default`. Validate with `grep -rn "unwrap()\|\.expect(\|unwrap_err" typed-lm-*/src --include="*.rs"` returning empty (also applies to `tests/`).
6.  **Deterministic Initialization (From-Scratch):** The weight initialization of `--method from-scratch` must be reproducible and **without a new RNG dependency**: use the project's own LCG/Box-Muller generator (`typed-lm-trainer/src/model/initialization.rs`), seedable through `--seed`. The same `(config, configuration, seed)` triple must produce byte-identical tensors. `from-scratch` requires an explicit geometry (`--architecture`/`--hidden-size`/... flags or the TOML `[model]` section) and a `tokenizer.json` (`--tokenizer-file` or `[tokenizer] file`), because there is no checkpoint to read them from.
7.  **Logging Exclusively via `tracing`:** It is forbidden to use `println!`, `eprintln!`, `print!`, `dbg!` or any direct print macro under `src/` and `tests/` (including tests, internal examples and benchmarks). Every log must go through the `tracing` crate (`tracing::info!`, `tracing::warn!`, `tracing::error!`, `tracing::debug!`, `tracing::trace!`), with structured fields where it makes sense. `tracing` is initialized at application startup; test output that needs visibility must use a log level where applicable. Validate with `grep -rn "println!\|eprintln!\|print!\|dbg!" typed-lm-*/src typed-lm-*/tests` returning empty.
8.  **GPU Always via Devcontainer:** Every execution that requires a GPU (`--features cuda` build, CUDA `#[ignore]` tests, GPU training/quantization, GPU benchmarks) must run **inside the devcontainer** (`.devcontainer/`), which reserves the GPU in `docker-compose.yml` and installs the CUDA toolkit. Never assume CUDA on the host and never run `cargo` with `--features cuda` outside the devcontainer. CPU commands (fmt, clippy, default `cargo test`) run on the host normally. To run something in the devcontainer use the `devcontainer up` / `devcontainer exec` commands (or the `devcontainers_*` MCP); the workspace inside the container is `/workspaces/typed-lm`.
9.  **Two Long-Lived Branches, Never Merged:** The repository has exactly two long-lived branches and they must **never** be merged into each other:
    *   **`main`** owns the **code**: the Rust crates, `Cargo.toml`/`Cargo.lock`, tests, examples (`examples/`, `example.http`), CI/release workflows and the release tags. All code changes, version bumps and releases land here.
    *   **`docs/gh-pages`** owns the **documentation**: the mdBook sources under `docs/` (`book.toml`, `src/`, `skin/`) and the `Deploy documentation` workflow. It is the GitHub Pages source branch.
    *   A code change must be committed on `main` and, when it affects the documented behavior, mirrored as a **separate documentation commit on `docs/gh-pages`** — never by merging one branch into the other. Do not cherry-pick documentation commits into `main`, and do not merge `main` into `docs/gh-pages`.
    *   Never open a pull request that merges `docs/gh-pages` into `main` (or the reverse). If a shared file such as a crate `README.md` must change, apply the same edit independently on each branch.
    *   The one exception is `.github/workflows/deploy-documentation.yml`: it must exist on **both** branches, because GitHub only registers a `push` workflow trigger from the default branch.
10. **Model Name is `typed-lm`, Never `jev-latest`:** The served model name defaults to **`typed-lm`**. Do **not** use `jev-latest` (or any `jev-` alias) as a model name in code, tests, fixtures, examples or documentation. The term **Jev** refers exclusively to the **TypeSafe AI product** and may appear only when referring to it (for example, the Jev-compatible HTTP contract and the "typed-lm and Jev" comparison). `is_supported_model` accepts only the exact served model name.
11. **Releases and Container Builds Happen in GitHub Actions, Never Locally:** Every artifact that is published is built by a workflow on GitHub — never by hand on a workstation or in the devcontainer:
    *   **Crates:** `cargo publish` runs only from the `publish-crates` job in `.github/workflows/release.yml`, triggered by a version tag. Do **not** run `cargo publish` locally.
    *   **Container images:** the CPU and CUDA images are built and pushed only by the `docker` and `docker-cuda` jobs, using `docker/build-push-action`. Do **not** `docker push` an image built locally.
    *   **Release binaries:** the tar.gz archives are produced only by the `build-cpu`, `build-cuda` and `build-metal` jobs.
    *   Local `docker build` and `cargo build` are for **verification only** (checking that a change compiles, that an image starts, that a Dockerfile's layers resolve). They must never be the source of a published artifact.
    *   To ship a change, bump the version in `Cargo.toml` and push a tag; the `Release` workflow does the rest. Any manual publication is a process violation.

## 5. Testing Guidelines (Mandatory)
No code, route or function may be produced without its automated test. The agent must adopt the TDD (Test-Driven Development) methodology in its responses.

1.  **Unit Tests:** For the mathematical logic of probability extraction and calibration (restricted softmax) from simulated tensors (dummy tensors). Also include numerical equivalence tests for the vendored paths: the CPU flash attention against a matmul/softmax reference (with GQA and causal offset) and the structured tokenization against the monolithic prompt.
2.  **API Integration Tests:** Use `actix_web::test` to create local instances of the `App` service and guarantee that the input and output JSON payloads match the Jev contract exactly.
3.  **Injectable Mocks:** To avoid downloading heavy models during CI/CD tests, create an abstraction (trait) for the evaluator. The Actix layer must be testable through a simulated model (mock) struct that returns predictable logits.
4.  **Live Tests (`#[ignore]`):** Cases that require real weights (upstream equivalence, session-cache gain, latency benchmark) are marked `#[ignore]` and never run in CI.

## 6. Agent Workflow
When instructed to create a new feature:
1. Start by designing the data types (Request and Response structs).
2. Write the Actix integration test for the route (which will initially fail).
3. Implement the tensor-handling logic (with unit tests).
4. Wire everything into the Actix handler until the tests pass.
