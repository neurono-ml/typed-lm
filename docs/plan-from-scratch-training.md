# Plan — Training from scratch (full-parameter, Jev decision objective)

> Status: **proposal** · Scope confirmed with the maintainer:
> train **from random initialization** targeting the **same Jev decision
> objective** the project already uses (single forward pass, cross-entropy at
> the decision position). General causal-LM pretraining is **out of scope**.
> Exposed as a **new method** on the existing `train` subcommand.

## 1. Goal and non-goals

**Goal.** Let `typed-lm-trainer train` initialize a model from scratch (random
weights over a chosen geometry) and train **all** parameters — embeddings,
norms, attention and MLP projections, and the language-model head — on a
Jev-native dataset, using the existing decision-position loss. The result is a
full dense checkpoint (`model.safetensors` + `config.json` + `tokenizer.json`)
that `typed-lm-serve` can serve directly.

**Non-goals.**

- General causal language-model pretraining (next-token over arbitrary corpora).
  Out of scope for this plan; would be a separate pipeline (full-sequence loss,
  corpus tokenizer, much larger compute budget).
- Distributed/multi-GPU training.
- New architectures beyond Llama/Qwen2 (the existing forward already covers both).

**Why this is feasible with today's code.** The differentiable full-sequence
forward already exists (`TrainableLlama`), with a hand-rolled differentiable
RMSNorm and softmax because candle's fused kernels have no backward on CPU
(verified: `rms_norm` uses `apply_op2_no_bwd`; `softmax_last_dim` is
non-differentiable on CPU). `Op::IndexSelect` **does** have a backward
(`index_add`), so a trainable embedding table works. The only structural change
is *which tensors are `Var`s*: today only the LoRA `A`/`B`; from-scratch requires
every weight to be trainable.

## 2. Current state (what is frozen today)

- `FrozenBase` (`model/weight_loading.rs`) stores every base weight as a plain
  `Tensor` — never a `Var`, so the optimizer never sees it.
- `LoRALinear` (`model/lora.rs`) wraps a frozen base `Tensor` with trainable
  `lora_a`/`lora_b` `Var`s; the base gradient path is deliberately cut.
- `TrainableLlama` (`model/trainable_llama.rs`) builds the whole tree from a
  `VarBuilder::from_tensors(base)` (frozen) + a `VarBuilder::from_varmap`
  (adapters). Embeddings (`token_embedding`, `language_model_head`) and
  `FrozenRmsNorm.weight` are frozen `Tensor`s.
- `TrainableModel::variables()` returns only `adapter_variables()`.
- `train` saves an adapter (`adapter.safetensors` + `adapter_config.json`).
- CLI method parsing is `TrainingMethod::{Lora, QLoRa}` in `cli.rs`.

## 3. Design

### 3.1 Training methods matrix

| `--method` | Base weights | Trainable params | Output |
|---|---|---|---|
| `lora` (current) | checkpoint, frozen | LoRA `A`/`B` | `adapter.safetensors` |
| `qlora` (current) | quantized base, frozen | LoRA `A`/`B` | `adapter.safetensors` |
| `full` (**new**) | checkpoint, **trainable** | all weights | full `model.safetensors` |
| `from-scratch` (**new**) | **random init** | all weights | full `model.safetensors` |

Both new methods share one code path; they differ only in the source of the
initial weights (checkpoint vs. random). Internally they are one enum value with
a boolean `random_initialization`, or two method values mapping to the same
builder. `--method full` on an existing checkpoint is a natural bonus and costs
almost nothing extra.

### 3.2 Model construction — one trainable-weight path

Introduce a builder that materializes **all** weights as `Var`s in the shared
`VarMap`, keyed by their canonical checkpoint names, while keeping the exact
same forward math. Two ways to do it; **prefer A**:

- **A (recommended): extend `TrainableLlama` with a weight source.** Add a
  `WeightSource` that, for each canonical name, returns either the frozen base
  tensor (LoRA modes) or a `Var` from the `VarMap` (full/from-scratch). The
  block/attention/MLP structs become generic over a projection abstraction:
  - `LoRALinear` (frozen base + `A`/`B`) — unchanged for LoRA/QLoRA;
  - `TrainableLinear` (single `Var` weight + optional `Var` bias) — new, for
    full/from-scratch.
  - `TrainableRmsNorm` (weight is a `Var`, differentiable formula from the
    existing `FrozenRmsNorm`) — new.
  - Embedding and head: `index_select`/`matmul` on `Var`s (differentiable via
    `IndexSelect`/`Matmul` backward).
- **B (fallback): a parallel `TrainableFullModel` builder** reusing the block
  forward functions but with `Var`-backed weights. More duplication; only if A's
  generics get unwieldy.

`adapter_variables()` stays for LoRA; a new `all_trainable_variables()` collects
every `Var`. `TrainableModel::variables()` returns the right set based on the
method.

### 3.3 Random initialization (from-scratch)

Add `model/initialization.rs`:

- Deterministic, seedable initializer (a small LCG like the test helpers, no new
  RNG dependency) so runs are reproducible from `--seed`.
- Per-tensor default initialization:
  - embeddings and linear weights: normal `mean 0, std = 0.02` (Llama-style
    `initializer_range`), or scaled `1/sqrt(fan_in)`;
  - RMSNorm weights: `1.0`;
  - biases: `0.0` (Qwen2 q/k/v);
  - `lm_head`: normal, tied to the embedding when `tie_word_embeddings`.
- Requires an explicit geometry, given either by a config file or by flags
  (`--hidden-size`, `--num-hidden-layers`, `--num-attention-heads`,
  `--num-key-value-heads`, `--intermediate-size`, `--vocab-size`,
  `--max-position-embeddings`, `--rope-theta`, `--architecture`). A `config.json`
  is written to the output directory so the checkpoint is self-describing.

### 3.4 CLI

Extend `TrainingMethod` (or a new `--method full|from-scratch`) in
`typed-lm-trainer/src/cli.rs`:

- `--method lora|qlora|full|from-scratch` (default stays `lora`).
- From-scratch geometry flags (grouped): either `--config <path>` **or** the
  explicit size flags above. Mutually exclusive with a required `--model-id` for
  the other methods; `--model-id` is **ignored/forbidden** for `from-scratch`.
- `--seed <u64>` (new, applies to from-scratch initialization).
- Full/from-scratch ignore `--adapter-directory` semantics of quantize; the
  quantize subcommand must accept a full-checkpoint directory as `--model-id`
  (it already loads any resolved checkpoint, so this works as-is).

### 3.5 Saving the result — full checkpoint

Add `training/checkpoint_saving.rs` (or extend `checkpoint.rs`):

- `save_full_checkpoint(variable_map, config, output_directory)`:
  - writes `model.safetensors` with **canonical names** (`model.embed_tokens.weight`,
    `lm_head.weight`, `model.norm.weight`, `model.layers.N.*`), so the serving
    loader resolves it unchanged;
  - writes `config.json` (serialize `ParallelModelConfig` back to a
    HuggingFace-shaped config);
  - copies/links `tokenizer.json` into the output directory.
- `quantize` then works on the from-scratch checkpoint directly (merge is a
  no-op when no adapter is supplied).

### 3.6 Prompt/loss reuse

No change to `dataset/` or the loss: tokenization already builds the exact
serving prompt and pins the decision position; `decision_loss` is unchanged. The
trainer must still be given a tokenizer for from-scratch (the tokenizer defines
the vocabulary). So `from-scratch` requires `--tokenizer-file` (or `--config` +
`--tokenizer-file`).

## 4. Work breakdown (waves, TDD)

Each wave follows the repo rule: write the failing test first, then implement,
then `cargo fmt`/`clippy`/`test` + the `unwrap()` grep must stay empty.

- **W0 — Types and tests scaffolding (no behavior).**
  - Add `TrainableMethod`/method parsing tests in `cli.rs` (red).
  - Decide and document the from-scratch geometry flags; parse-only tests.
  - Extend `TrainerError` with a `Initialization` variant if needed + tests.

- **W1 — Trainable projections and norms.**
  - `TrainableLinear` (weight `Var` + optional bias `Var`) with unit tests
    (forward equals a dense matmul; gradients reach the weight).
  - `TrainableRmsNorm` with a numerical-parity test against `FrozenRmsNorm` and a
    gradient-existence test.
  - `all_trainable_variables()` tests.

- **W2 — Model builder over a weight source.**
  - Refactor `TrainableLlama` to a `WeightSource` (frozen vs. `Var`) — keep the
    existing LoRA tests green throughout.
  - From-scratch build test: model with random `Var` weights whose forward is
    finite and whose `variables()` count equals the expected tensor count.

- **W3 — Random initialization.**
  - `model/initialization.rs` with deterministic-seed tests (same seed → same
    tensor; different seed → different tensor; norm weights are 1; biases 0).

- **W4 — Full/from-scratch training wiring in `main.rs`.**
  - Resolve geometry (`--config` or size flags), build the model from random
    `Var`s or a checkpoint, train with the existing loop.
  - Integration test (CPU, dummy geometry): from-scratch overfit on a tiny
    dataset reduces the loss and writes a full checkpoint that
    `LocalCheckpointResolver` resolves as dense.

- **W5 — Saving the full checkpoint.**
  - `save_full_checkpoint` + `config.json` serialization + tokenizer copy.
  - Test: save → resolve → `LanguageModel`-style load of the saved directory
    (or at least `FrozenBase::from_checkpoint` on it) succeeds.

- **W6 — CLI, docs and end-to-end.**
  - CLI flags, help text, mutual-exclusion validation tests.
  - Extend `typed-lm-serve/tests/end_to_end.rs` (or add a sibling) with a
    `from-scratch → serve` variant using a dummy geometry and no download.
  - Docs: `docs/training.md` section "Training from scratch"; README pointer;
    trainer README flag table.
  - Optional `#[ignore]` live GPU test: from-scratch on CUDA for a few steps.

## 5. Interfaces (sketches)

```rust
// model/trainable_llama.rs (new)
pub enum WeightSource<'a> {
    Frozen(&'a HashMap<String, Tensor>),
    Trainable(&'a VarBuilder<'a>), // every canonical name becomes a Var
}

pub struct TrainableLinear {
    weight: Var,
    bias: Option<Var>,
}

impl TrainableLinear {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor>;
    pub fn weight(&self) -> &Var;
}

pub struct TrainableRmsNorm {
    weight: Var,
    eps: f64,
}
```

```rust
// model/initialization.rs (new)
pub struct InitializationConfig {
    pub seed: u64,
    pub initializer_range: f32, // default 0.02
}

pub fn initialize_tensor(
    name: &str,
    shape: &[usize],
    config: &InitializationConfig,
) -> Result<Tensor>;
```

```rust
// training/checkpoint.rs (new)
pub fn save_full_checkpoint(
    variable_map: &VarMap,
    configuration: &ParallelModelConfig,
    tokenizer_source: &Path,
    output_directory: &Path,
) -> anyhow::Result<PathBuf>;
```

```rust
// cli.rs (extended)
pub enum TrainingMethod { Lora, QLoRa, Full, FromScratch }
// + from-scratch geometry args grouped in `TrainArguments`
```

## 6. Risks and mitigations

1. **Differentiability gaps in candle 0.11.** Fused RMSNorm/softmax are
   non-differentiable on CPU (confirmed). Mitigation: reuse the existing
   primitive reimplementations; add gradient-existence tests for every new
   trainable op before relying on it.
2. **Memory / optimizer state.** Full training keeps F32 master weights +
   AdamW moments for *every* parameter (≫ LoRA). Mitigation: document the cost,
   keep BF16 compute on GPU via `PrecisionPolicy`, and set expectations that
   from-scratch is for small geometries / prototyping, not 1B+ bases.
3. **Random init is not a language model.** A from-scratch model trained only on
   the decision objective will not "understand" text; it fits the dataset. This
   is expected and must be stated in the docs (it validates architecture,
   pipeline and dataset, it does not produce a general assistant).
4. **Regression risk in `TrainableLlama` refactor.** The whole existing test
   suite depends on it. Mitigation: keep the LoRA path behavior byte-identical,
   run `cargo test --workspace` at every wave, and add parity tests frozen vs.
   trainable paths.
5. **Config round-trip fidelity.** `ParallelModelConfig` → `config.json` must
   reproduce a HuggingFace-shaped config the resolver accepts. Mitigation: a
   round-trip test (`from_json_for_architecture` after serialization equals the
   original).
6. **Reproducibility.** No new RNG dependency: seed a local deterministic LCG.
   Test that the same seed yields identical weights.
7. **No abbreviations / no unwrap.** All new identifiers spelled out; tests
   return `Result` and propagate with `?`; the `unwrap()/expect(` grep must stay
   empty (CI gate).

## 7. Definition of Done

- [ ] `--method full` fine-tunes all parameters from an existing checkpoint.
- [ ] `--method from-scratch` trains from random weights over an explicit
      geometry and writes a complete checkpoint.
- [ ] The written checkpoint resolves and loads through the existing
      `typed-lm-serve` loader (dense safetensors + config + tokenizer).
- [ ] `quantize` works on a from-scratch checkpoint (FP8/FP4) and the API serves
      it (covered by an E2E test).
- [ ] Unit tests: trainable linear/norm forward parity + gradient existence,
      deterministic initialization, config round-trip.
- [ ] Integration test: from-scratch overfit reduces the loss (CPU, no download).
- [ ] `cargo fmt --check`, `cargo clippy --workspace --all-targets`,
      `cargo test --workspace` green; `unwrap()/expect(` grep empty.
- [ ] Docs updated (`docs/training.md`, trainer README, README pointer) with the
      from-scratch caveats and examples.

## 8. Effort estimate (rough)

| Wave | Focus | Relative size |
|---|---|---|
| W0 | CLI/types/tests scaffolding | S |
| W1 | Trainable linear + norm | M |
| W2 | Model builder over weight source | L |
| W3 | Random initialization | S |
| W4 | Training wiring | M |
| W5 | Full checkpoint saving | M |
| W6 | CLI + docs + E2E | M |

The largest risk and the largest single wave is **W2** (making the model tree
trainable without regressing the LoRA path). Everything else reuses existing,
tested machinery.
