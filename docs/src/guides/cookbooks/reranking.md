# Passage re-ranking

## Problem

A retrieval step returns a shortlist of candidate passages. Before the passages
reach an answering model, they must be re-ranked by relevance to the query.

## Request

```json
{
  "model": "typed-lm",
  "state": {
    "query": "What is the refund window for damaged items?",
    "passage": "Defective or damaged items are eligible for a full refund within 30 days of delivery."
  },
  "questions": {
    "relevant": {
      "type": "score",
      "instructions": "How relevant is the passage to the query?",
      "criteria": ["Irrelevant", "Weak", "Partial", "Relevant", "Directly answers"]
    }
  }
}
```

Run one request per query-passage pair, or pack several passages into one request
by making each question's `instructions` reference a different passage id.

## Response

```json
{
  "answers": {
    "relevant": {
      "type": "score",
      "score": 3.8,
      "legend": { "0": "Irrelevant", "1": "Weak", "2": "Partial", "3": "Relevant", "4": "Directly answers" },
      "probabilities": { "0": 0.01, "1": 0.03, "2": 0.12, "3": 0.57, "4": 0.27 },
      "confidence": 0.3
    }
  }
}
```

## Code

```python
scored = []
for passage in shortlist:
    response = ask(state={"query": query, "passage": passage})
    scored.append((passage, response["answers"]["relevant"]["score"]))

ranked = [p for p, _ in sorted(scored, key=lambda item: item[1], reverse=True)]
top = ranked[:3]
```

## Why it works

Relevance is an ordered judgment, so a `score` fits better than a boolean. Scoring
each pair in isolation avoids context rot across candidates, and the expected
value gives a stable sort key.

## Best practices

- Keep the rubric small and ordered; five levels is usually enough.
- Re-rank a shortlist (10 to 50), not the whole corpus.
- Combine the score with confidence to decide how many passages to keep.

## Next steps

- [Score](../../concepts/score.md) — the primitive.
- [Speculative fan-out](../patterns/fan-out.md) — score many candidates in one call.
