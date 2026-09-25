# Customer-support routing

## Problem

A support inbox receives order complaints. Each ticket must be triaged into a
refund decision, an owning department and an urgency level, before a human or a
downstream system acts.

## Request

```json
{
  "model": "typed-lm",
  "state": "Order #7710 arrived with a smashed box and a cracked vase inside. Delivery was 3 days ago and the customer asks what to do next.",
  "questions": {
    "refund_eligible": {
      "type": "noul",
      "instructions": "The customer is eligible for a full refund under the store policy."
    },
    "responsible_department": {
      "type": "choice",
      "instructions": "Which department should handle this case?",
      "criteria": {
        "billing": "Double charges and payment errors",
        "logistics": "Damaged, lost, or late shipments",
        "product_support": "Defective-item troubleshooting, replacements, and setup help"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this case?",
      "criteria": ["Routine", "Urgent", "Emergency"]
    }
  }
}
```

## Response

```json
{
  "model": "typed-lm",
  "answers": {
    "refund_eligible": { "type": "noul", "noul": 0.87 },
    "responsible_department": {
      "type": "choice",
      "choice": "logistics",
      "probabilities": { "billing": 0.05, "logistics": 0.9, "product_support": 0.05 },
      "confidence": 0.85
    },
    "urgency": {
      "type": "score",
      "score": 1.2,
      "legend": { "0": "Routine", "1": "Urgent", "2": "Emergency" },
      "probabilities": { "0": 0.2, "1": 0.4, "2": 0.4 },
      "confidence": 0.2
    }
  },
  "usage": { "input_tokens": 512, "output_tokens": 4 }
}
```

## Code

```python
answers = response["answers"]

refund = answers["refund_eligible"]["noul"] >= 0.8
department = answers["responsible_department"]
urgency = answers["urgency"]["score"]

if department["confidence"] < 0.5:
    queue_for_human(ticket, suggested=department["choice"])
elif refund and urgency >= 1.5:
    open_priority_case(ticket, department["choice"])
else:
    route(ticket, department["choice"])
```

## Why it works

Three atomic questions share one state and one forward pass. The refund decision
is a boolean, the routing is a closed classification and the urgency is an
ordered score — each in the primitive that fits it.

## Next steps

- [Confidence-gated routing](../patterns/confidence-routing.md) — the gate used here.
- [Noul](../../concepts/noul.md) — thresholding the refund probability.
