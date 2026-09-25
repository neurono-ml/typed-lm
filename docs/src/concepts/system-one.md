# System One decisions

A traditional language model answers by generating text token by token. The
answer *is* the generation. typed-lm takes a different route: it runs the model
**once**, looks at a single position, and returns a typed decision. This chapter
explains why that matters and how the scoring works.

## Generation versus decision

Text generation and a typed decision optimize different things. Generation must
produce a fluent, long sequence; a decision only needs the next-token
distribution at one position to be calibrated over a small set of labels.

```mermaid
---
accTitle: Text generation compared with a typed decision
accDescr: Generation decodes many tokens over many steps; a typed decision reads one position and calibrates a small set of labels.
---
flowchart LR
  subgraph gen["Autoregressive generation"]
    direction TB
    prompt1["prompt"]:::neutral
    token1["sample token 1"]:::warning
    token2["sample token 2"]:::warning
    tokenN["... token N"]:::warning
    text["free text"]:::danger
  end

  subgraph dec["typed-lm decision"]
    direction TB
    prompt2["state + question"]:::neutral
    forward["one forward pass"]:::accent
    logits["label logits"]:::primary
    typed["typed answer<br/>+ distribution"]:::success
  end

  prompt1 --> token1 --> token2 --> tokenN --> text
  prompt2 --> forward --> logits --> typed

  classDef primary fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:1.5px
  classDef accent fill:#dbeafe,stroke:#2563eb,color:#0c4a6e,stroke-width:1.5px
  classDef success fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px
  classDef warning fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:1.5px
  classDef danger fill:#fee2e2,stroke:#dc2626,color:#7f1d1d,stroke-width:1.5px
  classDef neutral fill:#f4f4f5,stroke:#a1a1aa,color:#18181b,stroke-width:1.5px
```

## The decision position

Every question is rendered into a prompt that ends with a small set of candidate
labels. The model reads the logits of those labels at the **last position** — the
decision position — and converts them into a distribution:

- **Noul** applies a binary softmax over the yes/no labels.
- **Choice** applies a restricted softmax over the option labels.
- **Score** applies a temperature-scaled softmax over the ordered levels.

Because the label set is closed, the model can never emit an answer outside it.
There is no parsing step and no hallucinated option.

## One shared prefill, many questions

All questions in a request share the same state. typed-lm tokenizes the common
prefix once, prefills it, and then evaluates each question suffix in a **single
batched forward pass**. Adding a question adds work proportional to the suffix,
not to the whole prompt.

```mermaid
---
accTitle: Shared prefill and batched question suffixes
accDescr: The common state prefix is prefilled once; each question suffix is evaluated in one batch and only the last position is read.
---
flowchart LR
  prefix["system + state<br/>common prefix"]:::accent
  q1["question 1 suffix"]:::primary
  q2["question 2 suffix"]:::primary
  q3["question 3 suffix"]:::primary
  batch["one batched forward pass"]:::success
  answers["independent typed answers"]:::success

  prefix --> batch
  q1 --> batch
  q2 --> batch
  q3 --> batch
  batch --> answers

  classDef primary fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:1.5px
  classDef accent fill:#dbeafe,stroke:#2563eb,color:#0c4a6e,stroke-width:1.5px
  classDef success fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px
```

## Atomic questions, composed in code

A System One question works best when it is narrow and self-contained. If a
question would require extended reasoning or mixes several independent factors,
split it. Ask each factor separately and combine the results with logic in your
code. When priorities change, you change a coefficient instead of rewriting a
prompt.

## What this buys you

- **Determinism** — the same state, questions and model produce the same answers.
- **Latency** — one forward pass per request, not one per generated token.
- **Safety** — answers are always inside the declared label set.
- **Calibration** — the returned distribution is what the training optimized.

## Next steps

- [State](./state.md) — what the model evaluates.
- [Questions (primitives)](./primitives.md) — the three question types.
- [Scoring and batched decoding](../engineering/scoring.md) — the implementation.
