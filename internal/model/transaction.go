// Package model holds the domain types shared by the HTTP, service and LLM layers.
package model

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
)

// Direction says which way money moved on a transaction.
type Direction string

const (
	DirectionDebit  Direction = "debit"  // money left the account
	DirectionCredit Direction = "credit" // money came into the account
)

// Category is the spending bucket a transaction is classified into.
type Category string

const (
	CategoryGroceries     Category = "groceries"
	CategoryDining        Category = "dining"
	CategoryTransport     Category = "transport"
	CategoryShopping      Category = "shopping"
	CategoryEntertainment Category = "entertainment"
	CategoryUtilities     Category = "utilities"
	CategoryHousing       Category = "housing"
	CategoryHealth        Category = "health"
	CategoryEducation     Category = "education"
	CategoryTravel        Category = "travel"
	CategorySubscription  Category = "subscription"
	CategoryFees          Category = "fees"
	CategoryTransfer      Category = "transfer"
	CategoryIncome        Category = "income"
	CategoryOther         Category = "other"
)

// Categories is the closed set the model is allowed to pick from. The LLM
// layer turns this into a JSON schema enum, so the model can never invent a
// category that the rest of the system does not know about.
var Categories = []Category{
	CategoryGroceries,
	CategoryDining,
	CategoryTransport,
	CategoryShopping,
	CategoryEntertainment,
	CategoryUtilities,
	CategoryHousing,
	CategoryHealth,
	CategoryEducation,
	CategoryTravel,
	CategorySubscription,
	CategoryFees,
	CategoryTransfer,
	CategoryIncome,
	CategoryOther,
}

// Valid reports whether c is one of the known categories.
func (c Category) Valid() bool {
	return slices.Contains(Categories, c)
}

// Transaction is one raw bank or card entry waiting to be classified.
type Transaction struct {
	ID          string    `json:"id"`
	Date        time.Time `json:"date"`
	Description string    `json:"description"`
	Merchant    string    `json:"merchant,omitempty"`
	Amount      float64   `json:"amount"`
	Currency    string    `json:"currency"`
	Direction   Direction `json:"direction"`
	Account     string    `json:"account,omitempty"`
}

// Validate checks the fields the classifier actually needs.
func (t Transaction) Validate() error {
	var errs []error
	if strings.TrimSpace(t.ID) == "" {
		errs = append(errs, errors.New("id is required"))
	}
	if strings.TrimSpace(t.Description) == "" && strings.TrimSpace(t.Merchant) == "" {
		errs = append(errs, errors.New("either description or merchant is required"))
	}
	if t.Amount == 0 {
		errs = append(errs, errors.New("amount must be non-zero"))
	}
	if len(t.Currency) != 3 {
		errs = append(errs, errors.New("currency must be a 3-letter ISO code"))
	}
	switch t.Direction {
	case DirectionDebit, DirectionCredit:
	case "":
		errs = append(errs, errors.New("direction is required"))
	default:
		errs = append(errs, fmt.Errorf("direction %q must be %q or %q", t.Direction, DirectionDebit, DirectionCredit))
	}
	return errors.Join(errs...)
}

// Classification is what the agent decided about a single transaction.
type Classification struct {
	TransactionID  string   `json:"transaction_id"`
	Category       Category `json:"category"`
	NormalizedName string   `json:"normalized_name"`
	Confidence     float64  `json:"confidence"`
	Recurring      bool     `json:"recurring"`
	NeedsReview    bool     `json:"needs_review"`
	Reasoning      string   `json:"reasoning"`
	SuggestedTags  []string `json:"suggested_tags,omitempty"`
}
