# Document classification with confidence

## Problem

A pipeline classifies documents into a taxonomy. When the model is unsure, it
should report a broader parent category rather than a wrong leaf.

## Request

```json
{
  "model": "jev-latest",
  "state": "The filing describes a quarterly dividend distribution to common shareholders.",
  "questions": {
    "category": {
      "type": "choice",
      "instructions": "Which category best describes this document?",
      "criteria": {
        "finance": "Financial instruments, dividends and accounting",
        "legal": "Contracts, litigation and compliance",
        "engineering": "Product design and technical documentation"
      }
    },
    "subcategory": {
      "type": "choice",
      "instructions": "Which finance subcategory best describes this document?",
      "criteria": {
        "finance.dividends": "Dividend distributions",
        "finance.reporting": "Accounting and reporting",
        "finance.tax": "Tax matters"
      }
    }
  }
}
```

## Response

```json
{
  "answers": {
    "category": {
      "type": "choice",
      "choice": "finance",
      "probabilities": { "finance": 0.96, "legal": 0.03, "engineering": 0.01 },
      "confidence": 0.91
    },
    "subcategory": {
      "type": "choice",
      "choice": "finance.dividends",
      "probabilities": { "finance.dividends": 0.62, "finance.reporting": 0.25, "finance.tax": 0.13 },
      "confidence": 0.28
    }
  }
}
```

## Code

```python
answers = response["answers"]
parent = answers["category"]
leaf = answers["subcategory"]

if parent["confidence"] < 0.5:
    label = "unclassified"
elif leaf["confidence"] < 0.5:
    label = parent["choice"]          # fall back to the parent category
else:
    label = leaf["choice"]
```

## Why it works

A hierarchy is safer with confidence: a confident parent and an unsure leaf resolve
to the parent, so the system is never forced to commit to a wrong leaf. Both
questions are answered in one request.

## Best practices

- Classify coarse first, then refine, rather than one giant flat taxonomy.
- Prefix leaf names with the parent for readable audit logs.
- Re-check uncertain leaves with a second, narrower question if accuracy matters.

## Next steps

- [Confidence](../../concepts/confidence.md) — interpreting the value.
- [Intent routing](../patterns/intent-routing.md) — a sibling classification pattern.
