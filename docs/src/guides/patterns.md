# Patterns

Patterns are reusable architectures for building systems with typed-lm. Each one
keeps the control in your code and gives the model a narrow, structured decision.

- [Confidence-gated routing](./patterns/confidence-routing.md) — use confidence as
  a second axis to decide whether to act.
- [Composite scoring](./patterns/composite-scoring.md) — break a complex judgment
  into atomic scores and combine them with weights you control.
- [Intent routing](./patterns/intent-routing.md) — classify an incoming request
  and route it to the right handler.
- [Speculative fan-out](./patterns/fan-out.md) — ask many questions in one call
  and let your code decide what matters.

## Shared principles

- **One decision per question.** Atomic questions are more reliable than compound
  ones.
- **Closed answer sets.** Declare the candidates in `criteria`.
- **Combine in code.** Weights, thresholds and rules are yours to test and change.
- **Read confidence.** For choice and score, confidence tells you whether to act.

## Next steps

- [Cookbooks](../guides/cookbooks.md) — end-to-end examples.
- [Designing with typed decisions](../concepts/how-to-build.md) — the design guide.
