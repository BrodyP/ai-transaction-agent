# ai-transaction-agent

An HTTP service that classifies bank and card transactions with Claude. Give it
a raw transaction — the kind of line a bank feed produces, full of processor
prefixes and store numbers — and it returns a category, a clean merchant name,
a confidence score, and a flag for whether a human should look at it.

```
POST /v1/transactions/classify
{ "id": "tx_1", "date": "2026-11-10T08:14:00Z",
  "description": "SQ *BLUE BOTTLE 00042 SF", "amount": 6.75,
  "currency": "USD", "direction": "debit" }

200 OK
{ "transaction_id": "tx_1", "category": "dining",
  "normalized_name": "Blue Bottle Coffee", "confidence": 0.94,
  "recurring": false, "needs_review": false,
  "reasoning": "Square prefix on a known coffee roaster at a cafe price point.",
  "suggested_tags": ["coffee", "cafe"] }
```

## Why it is built this way

The model's answer is constrained by a JSON schema (Anthropic's structured
outputs), not by asking politely for JSON in the prompt. The category enum in
that schema is generated from `model.Categories`, so the model physically
cannot return a category the rest of the system does not understand — and
adding a category is a one-line change in one place.

Results are also realigned against the input: every transaction you send gets
exactly one classification back, in the order you sent it. Anything the model
skipped or answered unusably comes back as a `needs_review` placeholder instead
of silently disappearing from the response.

## Layout

| Path | What lives there |
| --- | --- |
| `cmd/api` | Entry point: config from env, wiring, graceful shutdown |
| `internal/handler` | HTTP routing, decoding, status-code mapping |
| `internal/service` | Prompt construction, the JSON schema, result alignment |
| `internal/llm` | Anthropic Messages API wrapper — prompt + schema in, JSON out |
| `internal/model` | Domain types: `Transaction`, `Classification`, `Category` |

Each layer depends only on the one below it, and each consumer declares the
narrow interface it needs (`structuredCaller`, `classifier`), so the LLM can be
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

### `POST /v1/transactions/classify`

Classifies one transaction. Body is a single transaction object:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `id` | string | yes | Echoed back as `transaction_id` |
| `date` | RFC 3339 timestamp | yes | |
| `description` | string | yes* | *either this or `merchant` |
| `merchant` | string | no | |
| `amount` | number | yes | Must be non-zero |
| `currency` | string | yes | 3-letter ISO code |
| `direction` | string | yes | `debit` or `credit` |
| `account` | string | no | |

Unknown fields are rejected, so a typo in a field name fails loudly instead of
being silently ignored.

### `POST /v1/transactions/classify-batch`

Classifies up to 50 transactions in a single model call — much cheaper and
faster than 50 separate requests, since the system prompt is sent once.

```bash
curl -s localhost:8080/v1/transactions/classify-batch \
  -H 'Content-Type: application/json' \
  -d '{"transactions": [
        {"id":"tx_1","date":"2026-11-10T08:14:00Z","description":"SQ *BLUE BOTTLE 00042 SF","amount":6.75,"currency":"USD","direction":"debit"},
        {"id":"tx_2","date":"2026-11-01T00:00:00Z","description":"NETFLIX.COM","amount":15.49,"currency":"USD","direction":"debit"}
      ]}'
```

Responds with `{"classifications": [...]}`, one entry per input in input order.

## Response fields

| Field | Meaning |
| --- | --- |
| `transaction_id` | The id you sent |
| `category` | One of the 15 categories in `internal/model/transaction.go` |
| `normalized_name` | Merchant name with processor noise, store numbers and city codes stripped |
| `confidence` | 0.0–1.0 |
| `recurring` | Looks like a subscription or scheduled payment |
| `needs_review` | Set by the model, and forced on whenever `confidence < 0.7` |
| `reasoning` | One sentence naming the deciding signal |
| `suggested_tags` | Up to three search keywords; omitted when nothing applies |

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

## Not built yet

- No tests. The interfaces are in place for them (`structuredCaller` and
  `classifier` are both fakeable) but none are written.
- No persistence — classifications are returned, never stored.
- No auth on the endpoints, and no rate limiting.
- No retry on a transient model error; the SDK's own 2 retries are all there is.
