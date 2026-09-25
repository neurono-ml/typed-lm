# Composite scoring

A single score for a multi-factor judgment is unreliable. Decompose the judgment
into atomic scores and combine them with weights you control in code.

## The pattern

1. Identify the independent dimensions of the judgment.
2. Ask one `score` question per dimension.
3. Combine the scores with your own weights and thresholds.

```mermaid
---
accTitle: Composite scoring
accDescr: Several atomic score questions are combined with code-owned weights into a single composite decision.
---
flowchart LR
  state["state"]:::neutral
  s1["market score"]:::primary
  s2["feasibility score"]:::accent
  s3["differentiation score"]:::success
  weights["your weights"]:::warning
  composite["composite score"]:::success
  decision["decision"]:::success

  state --> s1 --> composite
  state --> s2 --> composite
  state --> s3 --> composite
  weights --> composite
  composite --> decision

  classDef primary fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:1.5px
  classDef accent fill:#dbeafe,stroke:#2563eb,color:#0c4a6e,stroke-width:1.5px
  classDef success fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px
  classDef warning fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:1.5px
  classDef neutral fill:#f4f4f5,stroke:#a1a1aa,color:#18181b,stroke-width:1.5px
```

## Example request

```json
{
  "model": "jev-latest",
  "state": "A pitch for a subscription service for weekly meal planning.",
  "questions": {
    "market": {
      "type": "score",
      "instructions": "How large and reachable is the target market?",
      "criteria": ["Very small", "Small", "Moderate", "Large", "Very large"]
    },
    "feasibility": {
      "type": "score",
      "instructions": "How feasible is the technical build?",
      "criteria": ["Very hard", "Hard", "Moderate", "Easy", "Trivial"]
    },
    "differentiation": {
      "type": "score",
      "instructions": "How differentiated is this from existing products?",
      "criteria": ["Commodity", "Weak", "Moderate", "Strong", "Unique"]
    }
  }
}
```

## Combining in code

```python
answers = response["answers"]
composite = (
    0.5 * answers["market"]["score"]
    + 0.3 * answers["feasibility"]["score"]
    + 0.2 * answers["differentiation"]["score"]
)
```

When priorities change, change the coefficients — not the prompts.

## Best practices

- Normalize scores before combining if the scales differ in length.
- Weight by business impact, not by how easy a dimension is to score.
- Gate the composite with confidence when a dimension is uncertain.

## Next steps

- [Confidence-gated routing](./confidence-routing.md) — gate on certainty.
- [Score](../../concepts/score.md) — the primitive.
