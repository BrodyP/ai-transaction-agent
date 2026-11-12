# ai-transaction-agent

An HTTP service that scores payment transactions for risk with Claude. Give it
a transaction — amount, merchant, status, how many times it has been retried —
and it returns a risk level, the signals it decided on, and how sure it is.

```
POST /v1/transactions/analyze
{
  "transaction_id": "tx_1001",
  "amount": 50000,
  "currency": "THB",
  "merchant": "UNKNOWN_SHOP",
  "status": "FAILED",
  "retry_count": 8
}

200 OK
{
  "transaction_id": "tx_1001",
  "risk_level": "HIGH",
  "reason": "High amount and repeated failures",
  "confidence": 0.92
}
```

## Why it is built this way

The model's answer is constrained by a JSON schema (Anthropic's structured
outputs), not by asking politely for JSON in the prompt. The `risk_level` enum
in that schema is generated from `model.RiskLevels`, so the model physically
cannot return a level the rest of the system does not understand — and adding
one (`CRITICAL`, say) is a one-line change in one place.

Batch results are realigned against the input, so every transaction you send
gets exactly one assessment back, in the order you sent it. A transaction the
model skipped or answered unusably comes back as `HIGH` at confidence `0` with
a reason saying so. That is deliberate: a missing assessment is an unanswered
question, and in a risk pipeline an unanswered question belongs in a review
queue rather than waved through as `LOW`.

Transaction `status` is *not* validated against a fixed list. A payment
processor will eventually send a status nobody here anticipated, and rejecting
the request would be worse than letting the model reason about the string.
`model.go` documents the statuses the prompt is written to expect.

## Layout

| Path | What lives there |
| --- | --- |
| `cmd/api` | Entry point: config from env, wiring, graceful shutdown |
| `internal/handler` | HTTP routing, decoding, status-code mapping |
| `internal/service` | Prompt construction, the JSON schema, result alignment |
| `internal/llm` | Anthropic Messages API wrapper — prompt + schema in, JSON out |
| `internal/model` | Domain types: `Transaction`, `RiskAssessment`, `RiskLevel` |

Each layer depends only on the one below it, and each consumer declares the
narrow interface it needs (`structuredCaller`, `analyzer`), so the LLM can be
faked in tests without touching the network.

## Running it

```bash
cp .env.example .env   # then put your key in it
export $(grep -v '^#' .env | xargs)
go run ./cmd/api
```

The service refuses to start without `ANTHROPIC_API_KEY`.

## Endpoints

### `GET /healthz`

Liveness check. Returns `{"status":"ok"}`.

### `POST /v1/transactions/analyze`

Scores one transaction. Body is a single transaction object:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `transaction_id` | string | yes | Echoed back in the response |
| `amount` | number | yes | Major currency units (baht, not satang); must not be negative |
| `currency` | string | yes | 3-letter ISO code |
| `merchant` | string | no | An empty or placeholder name is itself a risk signal |
| `status` | string | yes | `PENDING`, `SUCCESS`, `FAILED`, `DECLINED`, `REVERSED`, `CHARGEBACK`, or anything else your processor emits |
| `retry_count` | integer | yes | Must not be negative |

Unknown fields are rejected, so a typo in a field name fails loudly instead of
being silently ignored. `currency` and `status` are upper-cased before the
model sees them.

### `POST /v1/transactions/analyze-batch`

Scores up to 50 transactions in a single model call — much cheaper and faster
than 50 separate requests, since the system prompt is sent once.

```bash
curl -s localhost:8080/v1/transactions/analyze-batch \
  -H 'Content-Type: application/json' \
  -d '{"transactions": [
        {"transaction_id":"tx_1001","amount":50000,"currency":"THB","merchant":"UNKNOWN_SHOP","status":"FAILED","retry_count":8},
        {"transaction_id":"tx_1002","amount":120,"currency":"THB","merchant":"7-Eleven","status":"SUCCESS","retry_count":0}
      ]}'
```

Responds with `{"assessments": [...]}`, one entry per input in input order.

## Response fields

| Field | Meaning |
| --- | --- |
| `transaction_id` | The id you sent |
| `risk_level` | `LOW`, `MEDIUM` or `HIGH` |
| `reason` | One short phrase naming the signals that decided it |
| `confidence` | 0.0–1.0 |

## What the model weighs

Spelled out in the system prompt in `internal/service/classifier.go`:

- **`retry_count`** — a few retries are normal; 5+ failed attempts is the
  signature of card testing.
- **`status`** — `SUCCESS` is neutral, `FAILED`/`DECLINED` raise risk,
  `REVERSED`/`CHARGEBACK` are serious.
- **`amount` against `currency`** — judged per currency, not against one global
  threshold. 50,000 THB is a large consumer payment; 50,000 JPY is ordinary.
- **`merchant`** — a placeholder or test-looking name means the counterparty
  cannot be verified.
- **Combinations** — a large amount at an unverifiable merchant that keeps
  failing is worse than the sum of the three.

## Status codes

| Code | When |
| --- | --- |
| 400 | Malformed or unparseable JSON body |
| 422 | Valid JSON, invalid transaction — or the model declined the payload |
| 502 | The model call failed |
| 504 | The model call exceeded `LLM_TIMEOUT` |

## Configuration

See [`.env.example`](.env.example). `CLAUDE_MODEL` and `CLAUDE_EFFORT` are the
two cost levers: `claude-haiku-5-5` at `low` effort is the cheap end,
`claude-opus-5-5` at `high` the accurate end.

## Development

```bash
go build ./...
go vet ./...
go test -race -cover ./...
```

Tests cover the handler and service layers against a fake LLM — the request and
response contract, the status-code mapping, result alignment and the
fail-closed placeholder. `internal/llm` is not covered: exercising it means a
real API call.

## Not built yet

- No persistence — assessments are returned, never stored.
- No auth on the endpoints, and no rate limiting.
- No retry on a transient model error; the SDK's own 2 retries are all there is.
- Risk is judged per transaction in isolation. Velocity across a card, a device
  or an account — the signal most fraud systems lean on hardest — needs history
  this service does not keep.
