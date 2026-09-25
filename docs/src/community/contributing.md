# Contributing

typed-lm is open source under Apache-2.0. Contributions are welcome — code, docs,
datasets and prompts alike.

## Project rules

- **Code and documentation are always in English** — identifiers, comments,
  commit messages and files.
- **No abbreviations** in names: use `calculate_probability`, not `calc_prob`.
- **No `unwrap()` or `expect()`**, including in tests; propagate errors with `?`.
- **Logging only through `tracing`** — no `println!` or `dbg!`.
- **GPU work runs in the devcontainer**, never on a bare host with `--features cuda`.

The full rules live in `AGENTS.md` at the repository root.

## Workflow

1. Design the request and response types first.
2. Write the integration test that will initially fail.
3. Implement the logic with unit tests.
4. Wire it into the handler until the tests pass.

## Before you open a pull request

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

<div class="sk-box sk-box--info">
<strong>This book is part of the repository.</strong> Documentation changes follow the
same rules as code. The source lives under <code>docs/src</code> on the
<code>docs/gh-pages</code> branch and is published with mdBook.
</div>

## Where to help

- **Algorithms and backends** — new architectures, quantization formats, CUDA
  kernels.
- **Datasets and prompts** — examples that show the primitives in real domains.
- **Documentation** — tutorials, cookbooks and fixes.

## Next steps

- [Testing](../engineering/testing.md) — the test layers.
- [Architecture](../engineering/architecture.md) — the module map.
