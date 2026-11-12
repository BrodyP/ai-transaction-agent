# ai-transaction-agent

An HTTP service that scores payment transactions for risk with an LLM — Claude
or an OpenAI model, chosen by one environment variable. Give it a transaction:
amount, merchant, status, how many times it has been retried. It returns a risk
level, the signals it decided on, and how sure it is.

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

The model's answer is constrained by a JSON schema — Anthropic's structured
outputs or OpenAI's strict mode — not by asking politely for JSON in the
prompt. The `risk_level` enum in that schema is generated from
`model.RiskLevels`, so the model physically cannot return a level the rest of
the system does not understand, and adding one (`CRITICAL`, say) is a one-line
change in one place.

The same schema is sent to both providers unchanged. That works because
`StructuredRequest` commits to the intersection of what the two accept: a root
object with `additionalProperties: false` and every property listed in
`required`. Only one field differs — OpenAI wants the schema to have a name —
so the contract carries a `SchemaName` that Anthropic ignores.

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
| `internal/llm` | Provider wrappers — prompt + schema in, JSON out (Anthropic, OpenAI) |
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

The service refuses to start without the key for the selected provider, so a
missing key fails at boot rather than on the first request.

## Switching providers

`LLM_PROVIDER` picks the backend; nothing else changes.

```bash
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... go run ./cmd/api
LLM_PROVIDER=openai    OPENAI_API_KEY=sk-proj-... LLM_MODEL=gpt-5-nano go run ./cmd/api
```

Only `internal/llm` knows which provider is in play. The handler, service,
prompt, schema and tests are provider-agnostic — the service depends on a
one-method `structuredCaller` interface, so adding a third provider means
adding one file to `internal/llm` and a case to its `New`.

The honest caveat: a prompt tuned against one provider does not necessarily
score the same on the other. The code switches cleanly; the prompt quality does
not transfer for free. Treat a provider switch as something to re-measure, not
a config flip.

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

See [`.env.example`](.env.example). `LLM_MODEL` and `LLM_EFFORT` are the two
cost levers, and both providers accept the same effort names (`low` through
`max`). The cheap end is `claude-haiku-5-5` or `gpt-5-nano` at `low` effort;
the accurate end is `claude-opus-5-5` or `gpt-6-astra` at `high`.

Note that the per-provider defaults are not matched tiers — Anthropic defaults
to a frontier model, OpenAI to a mid-tier one — so set `LLM_MODEL` explicitly
when comparing them.

## Development

```bash
go build ./...
go vet ./...
go test -race -cover ./...
```

Requires Go 1.25. `github.com/openai/openai-go/v3` is pinned to v3.64.2
deliberately: from v3.75.0 it requires Go 1.26, which pulls `golang.org/x/sync`
v0.23.0 and forces the whole module to 1.26. Bump the pin and the `go`
directive together, or `go test -cover` fails on the packages without tests.

Tests cover the handler and service layers against a fake LLM — the request and
response contract, the status-code mapping, result alignment and the
fail-closed placeholder — plus provider selection and request validation in
`internal/llm`. What is not covered is either provider's actual network call;
that needs a real API key.

## Not built yet

- No persistence — assessments are returned, never stored.
- No auth on the endpoints, and no rate limiting.
- No retry on a transient model error; the SDK's own 2 retries are all there is.
- Risk is judged per transaction in isolation. Velocity across a card, a device
  or an account — the signal most fraud systems lean on hardest — needs history
  this service does not keep.
