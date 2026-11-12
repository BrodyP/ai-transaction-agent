// Package service holds the risk analysis logic: it turns domain objects into
// a prompt plus schema, and the model's JSON back into domain objects.
package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/BrodyP/ai-transaction-agent/internal/llm"
	"github.com/BrodyP/ai-transaction-agent/internal/model"
)

// MaxBatchSize is the largest batch AnalyzeBatch accepts in one call.
const MaxBatchSize = 50

const systemPrompt = `You are a payment risk analysis engine. For each
transaction you receive, you judge how likely it is to be fraudulent, abusive,
or otherwise worth a human's attention, and you return a risk level, a short
reason, and your confidence.

Signals to weigh:

- retry_count. A handful of retries is normal; many retries on a transaction
  that never succeeds is the signature of card testing or credential stuffing.
  Treat 5 or more failed attempts as a strong signal on its own.
- status. SUCCESS is neutral. FAILED and DECLINED raise risk, especially with
  retries. REVERSED and CHARGEBACK are serious — a chargeback means a customer
  already disputed the payment.
- amount, read against the currency. Judge whether the amount is large for a
  consumer payment in that currency, not against a single global threshold.
  50,000 THB is a large consumer payment; 50,000 JPY is an ordinary one.
- merchant. A placeholder, test-looking, empty, or meaningless merchant name
  (UNKNOWN_SHOP, TEST, ASDF, a bare id) means the counterparty cannot be
  verified, which raises risk. A recognizable, established merchant lowers it.
- Combinations matter more than any single field. A large amount at an
  unverifiable merchant that keeps failing is worse than the sum of the three.

Rules:

- risk_level is exactly one of LOW, MEDIUM or HIGH.
- reason is one short phrase naming the signals you actually decided on, in
  plain language a fraud analyst would read in a queue. Name the signals, not
  your reasoning process. Good: "High amount and repeated failures". Bad: "I
  considered the amount and then looked at the retries".
- confidence is 0.0 to 1.0 and describes how sure you are of the level. Be
  honest: a transaction with no strong signal either way is LOW at modest
  confidence, not LOW at 0.99.
- Judge only on the fields you were given. Do not invent history, a cardholder,
  a location or a device.
- Return exactly one entry per transaction_id you were given.`

// Analyzer scores transactions for risk using an LLM.
type Analyzer struct {
	llm structuredCaller
}

// structuredCaller is the slice of the LLM client this service needs. Keeping
// it here rather than in the llm package lets tests substitute a fake.
type structuredCaller interface {
	Structured(ctx context.Context, req llm.StructuredRequest, out any) error
}

// NewAnalyzer wires an Analyzer to an LLM client.
func NewAnalyzer(client structuredCaller) *Analyzer {
	return &Analyzer{llm: client}
}

// ErrEmptyBatch is returned when there is nothing to analyze.
var ErrEmptyBatch = errors.New("service: no transactions to analyze")

// Analyze scores a single transaction.
func (a *Analyzer) Analyze(ctx context.Context, tx model.Transaction) (model.RiskAssessment, error) {
	results, err := a.AnalyzeBatch(ctx, []model.Transaction{tx})
	if err != nil {
		return model.RiskAssessment{}, err
	}
	if len(results) == 0 {
		return model.RiskAssessment{}, fmt.Errorf("service: no assessment returned for transaction %q", tx.TransactionID)
	}
	return results[0], nil
}

// AnalyzeBatch scores up to MaxBatchSize transactions in one model call.
// Results come back in the same order as the input; a transaction the model
// skipped is returned flagged HIGH at zero confidence rather than dropped, so
// a gap in the model's answer fails closed instead of passing silently.
func (a *Analyzer) AnalyzeBatch(ctx context.Context, txs []model.Transaction) ([]model.RiskAssessment, error) {
	if len(txs) == 0 {
		return nil, ErrEmptyBatch
	}
	if len(txs) > MaxBatchSize {
		return nil, fmt.Errorf("service: batch of %d exceeds the limit of %d", len(txs), MaxBatchSize)
	}

	normalized := make([]model.Transaction, 0, len(txs))
	for _, tx := range txs {
		tx = tx.Normalized()
		if err := tx.Validate(); err != nil {
			return nil, fmt.Errorf("service: transaction %q is invalid: %w", tx.TransactionID, err)
		}
		normalized = append(normalized, tx)
	}

	prompt, err := buildPrompt(normalized)
	if err != nil {
		return nil, err
	}

	var envelope struct {
		Assessments []model.RiskAssessment `json:"assessments"`
	}
	req := llm.StructuredRequest{
		System:     systemPrompt,
		Prompt:     prompt,
		Schema:     assessmentSchema(),
		SchemaName: "risk_assessments",
	}
	if err := a.llm.Structured(ctx, req, &envelope); err != nil {
		return nil, fmt.Errorf("service: analyzing %d transaction(s): %w", len(normalized), err)
	}

	return align(normalized, envelope.Assessments), nil
}

// buildPrompt renders the transactions as JSON so the model sees exactly the
// fields it is asked to reason about.
func buildPrompt(txs []model.Transaction) (string, error) {
	payload, err := json.MarshalIndent(txs, "", "  ")
	if err != nil {
		return "", fmt.Errorf("service: encoding transactions for the prompt: %w", err)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Assess these %d transaction(s). Return one entry per transaction_id.\n\n", len(txs))
	b.WriteString("<transactions>\n")
	b.Write(payload)
	b.WriteString("\n</transactions>")
	return b.String(), nil
}

// align reorders the model's output to match the input and fills in anything
// the model left out. The placeholder is HIGH, not LOW: a missing assessment
// is an unanswered question, and in a risk pipeline that belongs in a review
// queue rather than waved through.
func align(txs []model.Transaction, got []model.RiskAssessment) []model.RiskAssessment {
	byID := make(map[string]model.RiskAssessment, len(got))
	for _, assessment := range got {
		byID[assessment.TransactionID] = assessment
	}

	out := make([]model.RiskAssessment, 0, len(txs))
	for _, tx := range txs {
		assessment, ok := byID[tx.TransactionID]
		if !ok || !assessment.RiskLevel.Valid() {
			out = append(out, model.RiskAssessment{
				TransactionID: tx.TransactionID,
				RiskLevel:     model.RiskHigh,
				Reason:        "No assessment returned by the model; flagged for manual review",
				Confidence:    0,
			})
			continue
		}
		assessment.TransactionID = tx.TransactionID
		out = append(out, assessment)
	}
	return out
}

// assessmentSchema is the JSON schema the model's response is validated
// against. The risk level enum is generated from model.RiskLevels so the two
// can never drift apart.
func assessmentSchema() map[string]any {
	levels := make([]any, 0, len(model.RiskLevels))
	for _, level := range model.RiskLevels {
		levels = append(levels, string(level))
	}

	return map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"assessments"},
		"properties": map[string]any{
			"assessments": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type":                 "object",
					"additionalProperties": false,
					"required":             []any{"transaction_id", "risk_level", "reason", "confidence"},
					"properties": map[string]any{
						"transaction_id": map[string]any{
							"type":        "string",
							"description": "the id of the transaction being assessed",
						},
						"risk_level": map[string]any{
							"type": "string",
							"enum": levels,
						},
						"reason": map[string]any{
							"type":        "string",
							"description": "one short phrase naming the deciding signals",
						},
						"confidence": map[string]any{
							"type":    "number",
							"minimum": 0,
							"maximum": 1,
						},
					},
				},
			},
		},
	}
}
