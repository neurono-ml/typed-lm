# HTTP API reference

Compact reference for the Jev-compatible HTTP API. For a guided version with
examples, see [Calling the API](../guides/api.md).

Base URL (default): `http://127.0.0.1:8080`.

## Routes

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/systemone` | Evaluate a `state` against typed questions. |
| `GET` | `/v1/models` | List the served model. |
| `GET` | `/health` | Readiness and model startup time. |
| `GET` | `/health/live` | Liveness; independent of the model. |

## `POST /v1/systemone` request

| Field | Type | Required | Notes |
|---|---|---|---|
| `model` | string | yes | Served name (default `typed-lm`) |
| `state` | string or JSON | yes | The facts of the case |
| `questions` | object | yes | Non-empty map of question name to question |

### Question types

| `type` | Required fields | Candidates |
|---|---|---|
| `noul` | `instructions` | Optional `criteria` `{"yes","no"}` |
| `choice` | `instructions`, `criteria` (object) | One entry per option |
| `score` | `instructions`, `criteria` (array) | 2 to 10 levels, increasing |

## `POST /v1/systemone` response

| Field | Type | Notes |
|---|---|---|
| `model` | string | Echoes the resolved model |
| `answers` | object | One entry per question name |
| `usage.input_tokens` | integer | Prompt tokens |
| `usage.output_tokens` | integer | Decision positions read |

### Answer shapes

| `type` | Fields |
|---|---|
| `noul` | `noul` (0.0 to 1.0) |
| `choice` | `choice`, `probabilities`, `confidence` |
| `score` | `score`, `legend`, `probabilities`, `confidence` |

## Status codes

| Status | Situation |
|---|---|
| `200` | Success |
| `404` | Unknown model |
| `422` | Malformed body or question outside the contract |
| `500` | Inference failure |

Error envelope:

```json
{ "error": { "message": "..." } }
```

## `GET /v1/models`

```json
{
  "object": "list",
  "data": [{ "id": "typed-lm", "object": "model", "owned_by": "typed-lm" }],
  "models": [
    { "name": "typed-lm", "description": "typed-lm model served from context '...'", "release_date": "unknown" }
  ]
}
```

## `GET /health`

```json
{ "status": "ok", "startup_seconds": 12.3 }
```

## `GET /health/live`

```json
{ "status": "ok" }
```

## Next steps

- [Calling the API](../guides/api.md) — examples.
- [Server flags](./server-flags.md) — configuration.
