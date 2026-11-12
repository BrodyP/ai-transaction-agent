// Package service holds the transaction classification logic: it turns
// domain objects into a prompt plus schema, and the model's JSON back into
// domain objects.
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

// MaxBatchSize is the largest batch ClassifyBatch accepts in one call.
const MaxBatchSize = 50

const systemPrompt = `You are a transaction classification engine for a personal finance app.

For each transaction you receive, you decide:
- category: the single best fit from the allowed list.
- normalized_name: the merchant's clean, human-readable name, stripped of
  payment-processor noise, store numbers, city codes and reference ids.
  Example: "SQ *BLUE BOTTLE 00042 SF" becomes "Blue Bottle Coffee".
- confidence: 0.0 to 1.0, how sure you are of the category.
- recurring: true when the description looks like a subscription or a regular
  scheduled payment (rent, insurance, utility bill, streaming service).
- needs_review: true when confidence is below 0.7, when the description is too
  opaque to classify, or when the amount looks inconsistent with the category.
- reasoning: one short sentence naming the signal you classified on.
- suggested_tags: up to three lowercase keywords that would help a user search
  for this transaction later. Omit when nothing useful applies.

Rules:
- A credit is income only when it looks like salary, interest or a payout.
  Refunds and payments between a user's own accounts are transfer.
- Never guess a merchant name you cannot see evidence for in the input. Echo
  the cleaned description instead and set needs_review.
- Classify every transaction you are given, keyed by the id you were given.`

// Classifier turns raw transactions into classifications using an LLM.
type Classifier struct {
	llm structuredCaller
}

// structuredCaller is the slice of the LLM client this service needs. Keeping
// it here rather than in the llm package lets tests substitute a fake.
type structuredCaller interface {
	Structured(ctx context.Context, req llm.StructuredRequest, out any) error
}

// NewClassifier wires a Classifier to an LLM client.
func NewClassifier(client structuredCaller) *Classifier {
	return &Classifier{llm: client}
}

// ErrEmptyBatch is returned when there is nothing to classify.
var ErrEmptyBatch = errors.New("service: no transactions to classify")

// Classify classifies a single transaction.
func (c *Classifier) Classify(ctx context.Context, tx model.Transaction) (model.Classification, error) {
	results, err := c.ClassifyBatch(ctx, []model.Transaction{tx})
	if err != nil {
		return model.Classification{}, err
	}
	if len(results) == 0 {
		return model.Classification{}, fmt.Errorf("service: model returned no classification for transaction %q", tx.ID)
	}
	return results[0], nil
}

// ClassifyBatch classifies up to MaxBatchSize transactions in one model call.
// Results come back in the same order as the input; a transaction the model
// skipped is returned as a needs_review placeholder rather than dropped.
func (c *Classifier) ClassifyBatch(ctx context.Context, txs []model.Transaction) ([]model.Classification, error) {
	if len(txs) == 0 {
		return nil, ErrEmptyBatch
	}
	if len(txs) > MaxBatchSize {
		return nil, fmt.Errorf("service: batch of %d exceeds the limit of %d", len(txs), MaxBatchSize)
	}
	for _, tx := range txs {
		if err := tx.Validate(); err != nil {
			return nil, fmt.Errorf("service: transaction %q is invalid: %w", tx.ID, err)
		}
	}

	prompt, err := buildPrompt(txs)
	if err != nil {
		return nil, err
	}

	var envelope struct {
		Classifications []model.Classification `json:"classifications"`
	}
	req := llm.StructuredRequest{
		System: systemPrompt,
		Prompt: prompt,
		Schema: classificationSchema(),
	}
	if err := c.llm.Structured(ctx, req, &envelope); err != nil {
		return nil, fmt.Errorf("service: classifying %d transaction(s): %w", len(txs), err)
	}

	return align(txs, envelope.Classifications), nil
}

// buildPrompt renders the transactions as JSON so the model sees exactly the
// fields it is asked to reason about.
func buildPrompt(txs []model.Transaction) (string, error) {
	payload, err := json.MarshalIndent(txs, "", "  ")
	if err != nil {
		return "", fmt.Errorf("service: encoding transactions for the prompt: %w", err)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Classify these %d transaction(s). Return one entry per id.\n\n", len(txs))
	b.WriteString("<transactions>\n")
	b.Write(payload)
	b.WriteString("\n</transactions>")
	return b.String(), nil
}

// align reorders the model's output to match the input and fills in anything
// the model left out, so callers always get one result per transaction.
func align(txs []model.Transaction, got []model.Classification) []model.Classification {
	byID := make(map[string]model.Classification, len(got))
	for _, cl := range got {
		byID[cl.TransactionID] = cl
	}

	out := make([]model.Classification, 0, len(txs))
	for _, tx := range txs {
		cl, ok := byID[tx.ID]
		if !ok || !cl.Category.Valid() {
			out = append(out, model.Classification{
				TransactionID:  tx.ID,
				Category:       model.CategoryOther,
				NormalizedName: fallbackName(tx),
				Confidence:     0,
				NeedsReview:    true,
				Reasoning:      "the model returned no usable classification for this transaction",
			})
			continue
		}
		cl.TransactionID = tx.ID
		if cl.Confidence < 0.7 {
			cl.NeedsReview = true
		}
		out = append(out, cl)
	}
	return out
}

func fallbackName(tx model.Transaction) string {
	if tx.Merchant != "" {
		return tx.Merchant
	}
	return tx.Description
}

// classificationSchema is the JSON schema the model's response is validated
// against. The category enum is generated from model.Categories so the two
// can never drift apart.
func classificationSchema() map[string]any {
	categories := make([]any, 0, len(model.Categories))
	for _, cat := range model.Categories {
		categories = append(categories, string(cat))
	}

	return map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"classifications"},
		"properties": map[string]any{
			"classifications": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type":                 "object",
					"additionalProperties": false,
					"required": []any{
						"transaction_id", "category", "normalized_name",
						"confidence", "recurring", "needs_review", "reasoning",
					},
					"properties": map[string]any{
						"transaction_id": map[string]any{
							"type":        "string",
							"description": "the id of the transaction being classified",
						},
						"category": map[string]any{
							"type": "string",
							"enum": categories,
						},
						"normalized_name": map[string]any{
							"type":        "string",
							"description": "clean human-readable merchant name",
						},
						"confidence": map[string]any{
							"type":    "number",
							"minimum": 0,
							"maximum": 1,
						},
						"recurring":    map[string]any{"type": "boolean"},
						"needs_review": map[string]any{"type": "boolean"},
						"reasoning": map[string]any{
							"type":        "string",
							"description": "one short sentence naming the deciding signal",
						},
						"suggested_tags": map[string]any{
							"type":     "array",
							"maxItems": 3,
							"items":    map[string]any{"type": "string"},
						},
					},
				},
			},
		},
	}
}
