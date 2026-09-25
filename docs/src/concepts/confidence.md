# Confidence

**Confidence** tells your code how certain the model is about a `choice` or
`score` answer. It is a second axis: the answer says *what*, confidence says
*whether to act on it*.

## Which answers carry confidence?

- `choice` — returns `confidence` alongside `choice` and `probabilities`.
- `score` — returns `confidence` alongside `score`, `legend` and `probabilities`.
- `noul` — does not return a separate confidence; the `noul` value is itself the
  calibrated affirmative probability.

## How is confidence defined?

Confidence is derived from how concentrated the restricted distribution is. A
distribution with one dominant option is confident; a flat distribution is not.
A perfectly uniform distribution over two options is the least confident case.

```mermaid
---
accTitle: Confidence reflects distribution concentration
accDescr: A peaked choice distribution yields high confidence; a flat distribution yields low confidence.
---
flowchart LR
  subgraph peaked["Peaked distribution"]
    p["logistics 0.95<br/>billing 0.03<br/>product_support 0.02"]:::success
    ph["confidence 0.90"]:::success
  end

  subgraph flat["Flat distribution"]
    f["logistics 0.35<br/>billing 0.33<br/>product_support 0.32"]:::warning
    fh["confidence 0.02"]:::warning
  end

  p --> ph
  f --> fh

  classDef success fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px
  classDef warning fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:1.5px
```

## Why does confidence matter?

High accuracy overall can still hide unreliable individual answers. Confidence
lets you separate the two:

- **Answer** — the best option or the expected score.
- **Confidence** — whether that answer is trustworthy enough to automate.

A common pattern is a three-way decision: act automatically above a high
threshold, review in a middle band, and reject or escalate below a low one.

## How do I use it architecturally?

```mermaid
---
accTitle: Confidence-gated routing
accDescr: A choice answer is routed to automatic handling, human review or rejection depending on its confidence.
---
flowchart TB
  request["state + choice question"]:::neutral
  model["typed-lm"]:::accent
  answer["choice + confidence"]:::primary
  gate{"confidence?"}:::warning
  auto["automatic action"]:::success
  review["human review"]:::warning
  reject["reject / escalate"]:::danger

  request --> model --> answer --> gate
  gate -- "high" --> auto
  gate -- "medium" --> review
  gate -- "low" --> reject

  classDef primary fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:1.5px
  classDef accent fill:#dbeafe,stroke:#2563eb,color:#0c4a6e,stroke-width:1.5px
  classDef success fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px
  classDef warning fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:1.5px
  classDef danger fill:#fee2e2,stroke:#dc2626,color:#7f1d1d,stroke-width:1.5px
  classDef neutral fill:#f4f4f5,stroke:#a1a1aa,color:#18181b,stroke-width:1.5px
```

## Calibration

Confidence is only useful if it is calibrated by training. Because the trainer
optimizes the same decision-position cross-entropy the server reads, the
distribution you see at inference is the one the model was tuned to produce. See
[Training overview](../training/index.md).

## Best practices

- **Do not hardcode one threshold for everything.** Different questions have
  different error costs.
- **Measure on your data.** Pick thresholds from a held-out set, not by intuition.
- **Keep the raw distribution.** Even when you act automatically, log
  `probabilities` so you can audit and recalibrate later.
- **Prefer more specific questions.** A vague question produces a flat
  distribution; a focused one produces a confident answer.

## Next steps

- [Patterns](../guides/patterns.md) — confidence-gated routing and more.
- [Choice](./choice.md) and [Score](./score.md) — where confidence appears.
- [Training overview](../training/index.md) — how calibration is produced.
