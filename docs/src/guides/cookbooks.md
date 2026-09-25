# Cookbooks

Cookbooks are end-to-end recipes: a concrete problem, the request, the response,
and the code that acts on it.

- [Customer-support routing](./cookbooks/support-routing.md) — refund eligibility,
  department routing and urgency in one call.
- [Content-moderation guardrails](./cookbooks/moderation-guardrails.md) — screen
  messages with probabilities and severity.
- [Passage re-ranking](./cookbooks/reranking.md) — score a shortlist and keep the
  best passages.
- [Document classification with confidence](./cookbooks/classification-confidence.md)
  — classify into a hierarchy and fall back when unsure.

## How to read a cookbook

Each recipe follows the same structure:

1. **Problem** — the business situation.
2. **Request** — the exact `POST /v1/systemone` body.
3. **Response** — a representative typed answer.
4. **Code** — the logic that turns the answer into an action.

Adapt the thresholds and label sets to your domain; the shapes stay the same.

## Next steps

- [Patterns](../guides/patterns.md) — the reusable architectures behind these recipes.
- [Training overview](../training/index.md) — teach the model your labels.
