# Confidence-gated routing

Use the answer for *what* and confidence for *whether to act*. A single `choice`
question can drive a three-way decision: act automatically, route to review, or
reject.

## The pattern

1. Ask one `choice` question.
2. Read `choice` and `confidence`.
3. Route on confidence:
   - high confidence → act automatically,
   - medium confidence → send to a human,
   - low confidence → reject or escalate.

```mermaid
---
accTitle: Confidence-gated routing
accDescr: A single choice answer is routed to automation, review or rejection based on its confidence value.
---
flowchart LR
  request["incoming request"]:::neutral
  choice["choice question"]:::accent
  answer["choice + confidence"]:::primary
  gate{"confidence band?"}:::warning
  auto["automatic action"]:::success
  review["human review"]:::warning
  reject["reject / escalate"]:::danger

  request --> choice --> answer --> gate
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

## Example request

```json
{
  "model": "typed-lm",
  "state": "The user asks whether the annual plan can be cancelled mid-cycle.",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Which intent best describes the user request?",
      "criteria": {
        "billing": "Payments, invoices and refunds",
        "account": "Login, profile and cancellation",
        "technical": "Product errors and setup"
      }
    }
  }
}
```

## Combining in code

```python
answer = response["answers"]["intent"]
confidence = answer["confidence"]

if confidence >= 0.85:
    dispatch[answer["choice"]](request)
elif confidence >= 0.45:
    queue_for_human(request, suggested=answer["choice"])
else:
    escalate(request, reason="low_confidence")
```

## Why it works

The distribution is calibrated by training on the same decision position the
server reads. A focused question produces a peaked distribution; a vague one
produces a flat distribution you should not automate.

## Best practices

- Calibrate the thresholds on a held-out set, not by intuition.
- Use different thresholds per question when error costs differ.
- Always log the raw `probabilities` for later recalibration.

## Next steps

- [Composite scoring](./composite-scoring.md) — combine several decisions.
- [Confidence](../../concepts/confidence.md) — the underlying concept.
