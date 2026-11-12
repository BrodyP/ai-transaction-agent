// Package model holds the domain types shared by the HTTP, service and LLM layers.
package model

import (
	"errors"
	"fmt"
	"slices"
	"strings"
)

// RiskLevel is how risky the agent judged a transaction to be.
type RiskLevel string

const (
	RiskLow    RiskLevel = "LOW"
	RiskMedium RiskLevel = "MEDIUM"
	RiskHigh   RiskLevel = "HIGH"
)

// RiskLevels is the closed set the model may pick from. The LLM layer turns
// this into a JSON schema enum, so the model cannot return a level the rest of
// the system does not know about. Adding a level (CRITICAL, say) means adding
// it here and nowhere else.
var RiskLevels = []RiskLevel{RiskLow, RiskMedium, RiskHigh}

// Valid reports whether r is one of the known risk levels.
func (r RiskLevel) Valid() bool {
	return slices.Contains(RiskLevels, r)
}

// Known transaction statuses. Status is deliberately NOT validated against
// this list — a payment processor may send a status we have not seen, and
// rejecting it would be worse than letting the model reason about it. The list
// documents what the prompt is written to expect.
const (
	StatusPending    = "PENDING"
	StatusSuccess    = "SUCCESS"
	StatusFailed     = "FAILED"
	StatusDeclined   = "DECLINED"
	StatusReversed   = "REVERSED"
	StatusChargeback = "CHARGEBACK"
)

// Transaction is one payment event submitted for risk analysis.
type Transaction struct {
	TransactionID string  `json:"transaction_id"`
	Amount        float64 `json:"amount"`
	Currency      string  `json:"currency"`
	Merchant      string  `json:"merchant"`
	Status        string  `json:"status"`
	RetryCount    int     `json:"retry_count"`
}

// Validate checks the fields the analyzer needs. Amount is in major currency
// units (baht, not satang).
func (t Transaction) Validate() error {
	var errs []error
	if strings.TrimSpace(t.TransactionID) == "" {
		errs = append(errs, errors.New("transaction_id is required"))
	}
	if t.Amount < 0 {
		errs = append(errs, errors.New("amount must not be negative"))
	}
	if len(t.Currency) != 3 {
		errs = append(errs, errors.New("currency must be a 3-letter ISO code"))
	}
	if strings.TrimSpace(t.Status) == "" {
		errs = append(errs, errors.New("status is required"))
	}
	if t.RetryCount < 0 {
		errs = append(errs, errors.New("retry_count must not be negative"))
	}
	return errors.Join(errs...)
}

// Normalized returns a copy with the free-text fields cleaned up, so the model
// sees "FAILED" whether the caller sent "failed" or " Failed ".
func (t Transaction) Normalized() Transaction {
	t.TransactionID = strings.TrimSpace(t.TransactionID)
	t.Currency = strings.ToUpper(strings.TrimSpace(t.Currency))
	t.Status = strings.ToUpper(strings.TrimSpace(t.Status))
	t.Merchant = strings.TrimSpace(t.Merchant)
	return t
}

// RiskAssessment is the agent's verdict on a single transaction.
type RiskAssessment struct {
	TransactionID string    `json:"transaction_id"`
	RiskLevel     RiskLevel `json:"risk_level"`
	Reason        string    `json:"reason"`
	Confidence    float64   `json:"confidence"`
}

// String renders an assessment for logs.
func (a RiskAssessment) String() string {
	return fmt.Sprintf("%s=%s (%.2f): %s", a.TransactionID, a.RiskLevel, a.Confidence, a.Reason)
}
