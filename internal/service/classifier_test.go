package service

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/BrodyP/ai-transaction-agent/internal/llm"
	"github.com/BrodyP/ai-transaction-agent/internal/model"
)

// fakeCaller stands in for the LLM. It records the request it was given and
// replies with canned JSON, so the service's prompt building and result
// alignment can be tested without touching the network.
type fakeCaller struct {
	reply string
	err   error
	got   llm.StructuredRequest
}

func (f *fakeCaller) Structured(_ context.Context, req llm.StructuredRequest, out any) error {
	f.got = req
	if f.err != nil {
		return f.err
	}
	return json.Unmarshal([]byte(f.reply), out)
}

func sampleTx() model.Transaction {
	return model.Transaction{
		TransactionID: "tx_1001",
		Amount:        50000,
		Currency:      "THB",
		Merchant:      "UNKNOWN_SHOP",
		Status:        "FAILED",
		RetryCount:    8,
	}
}

func TestAnalyze(t *testing.T) {
	t.Parallel()

	fake := &fakeCaller{reply: `{"assessments":[{
		"transaction_id":"tx_1001","risk_level":"HIGH",
		"reason":"High amount and repeated failures","confidence":0.92}]}`}

	got, err := NewAnalyzer(fake).Analyze(context.Background(), sampleTx())
	if err != nil {
		t.Fatalf("Analyze() error = %v", err)
	}

	want := model.RiskAssessment{
		TransactionID: "tx_1001",
		RiskLevel:     model.RiskHigh,
		Reason:        "High amount and repeated failures",
		Confidence:    0.92,
	}
	if got != want {
		t.Errorf("Analyze() = %+v, want %+v", got, want)
	}
}

func TestAnalyzeBatchAlignment(t *testing.T) {
	t.Parallel()

	txs := []model.Transaction{
		{TransactionID: "tx_1", Amount: 10, Currency: "THB", Status: "SUCCESS"},
		{TransactionID: "tx_2", Amount: 20, Currency: "THB", Status: "FAILED", RetryCount: 9},
		{TransactionID: "tx_3", Amount: 30, Currency: "THB", Status: "SUCCESS"},
	}

	tests := []struct {
		name       string
		reply      string
		wantLevels []model.RiskLevel
		wantConf   []float64
	}{
		{
			name: "out of order is realigned to input order",
			reply: `{"assessments":[
				{"transaction_id":"tx_3","risk_level":"LOW","reason":"c","confidence":0.8},
				{"transaction_id":"tx_1","risk_level":"MEDIUM","reason":"a","confidence":0.7},
				{"transaction_id":"tx_2","risk_level":"HIGH","reason":"b","confidence":0.9}]}`,
			wantLevels: []model.RiskLevel{model.RiskMedium, model.RiskHigh, model.RiskLow},
			wantConf:   []float64{0.7, 0.9, 0.8},
		},
		{
			name: "skipped transaction fails closed as HIGH",
			reply: `{"assessments":[
				{"transaction_id":"tx_1","risk_level":"LOW","reason":"a","confidence":0.8},
				{"transaction_id":"tx_3","risk_level":"LOW","reason":"c","confidence":0.8}]}`,
			wantLevels: []model.RiskLevel{model.RiskLow, model.RiskHigh, model.RiskLow},
			wantConf:   []float64{0.8, 0, 0.8},
		},
		{
			name: "unknown risk level fails closed as HIGH",
			reply: `{"assessments":[
				{"transaction_id":"tx_1","risk_level":"LOW","reason":"a","confidence":0.8},
				{"transaction_id":"tx_2","risk_level":"CATASTROPHIC","reason":"b","confidence":0.9},
				{"transaction_id":"tx_3","risk_level":"LOW","reason":"c","confidence":0.8}]}`,
			wantLevels: []model.RiskLevel{model.RiskLow, model.RiskHigh, model.RiskLow},
			wantConf:   []float64{0.8, 0, 0.8},
		},
		{
			name:       "empty reply flags every transaction",
			reply:      `{"assessments":[]}`,
			wantLevels: []model.RiskLevel{model.RiskHigh, model.RiskHigh, model.RiskHigh},
			wantConf:   []float64{0, 0, 0},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, err := NewAnalyzer(&fakeCaller{reply: tc.reply}).AnalyzeBatch(context.Background(), txs)
			if err != nil {
				t.Fatalf("AnalyzeBatch() error = %v", err)
			}
			if len(got) != len(txs) {
				t.Fatalf("AnalyzeBatch() returned %d assessments, want %d", len(got), len(txs))
			}
			for i, assessment := range got {
				if assessment.TransactionID != txs[i].TransactionID {
					t.Errorf("assessment[%d].TransactionID = %q, want %q", i, assessment.TransactionID, txs[i].TransactionID)
				}
				if assessment.RiskLevel != tc.wantLevels[i] {
					t.Errorf("assessment[%d].RiskLevel = %q, want %q", i, assessment.RiskLevel, tc.wantLevels[i])
				}
				if assessment.Confidence != tc.wantConf[i] {
					t.Errorf("assessment[%d].Confidence = %v, want %v", i, assessment.Confidence, tc.wantConf[i])
				}
			}
		})
	}
}

func TestAnalyzeBatchRejectsBadInput(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		txs     []model.Transaction
		wantErr error
	}{
		{name: "empty batch", txs: nil, wantErr: ErrEmptyBatch},
		{name: "over the batch limit", txs: make([]model.Transaction, MaxBatchSize+1)},
		{name: "invalid transaction", txs: []model.Transaction{{TransactionID: "", Currency: "THB", Status: "FAILED"}}},
		{name: "negative retry count", txs: []model.Transaction{{TransactionID: "tx_1", Currency: "THB", Status: "FAILED", RetryCount: -1}}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			fake := &fakeCaller{reply: `{"assessments":[]}`}
			_, err := NewAnalyzer(fake).AnalyzeBatch(context.Background(), tc.txs)
			if err == nil {
				t.Fatal("AnalyzeBatch() error = nil, want an error")
			}
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Errorf("AnalyzeBatch() error = %v, want %v", err, tc.wantErr)
			}
			if fake.got.Prompt != "" {
				t.Error("AnalyzeBatch() called the model despite invalid input")
			}
		})
	}
}

func TestAnalyzeBatchPropagatesLLMError(t *testing.T) {
	t.Parallel()

	fake := &fakeCaller{err: llm.ErrRefused}
	_, err := NewAnalyzer(fake).AnalyzeBatch(context.Background(), []model.Transaction{sampleTx()})
	if !errors.Is(err, llm.ErrRefused) {
		t.Errorf("AnalyzeBatch() error = %v, want it to wrap %v", err, llm.ErrRefused)
	}
}

// The prompt is what the model actually reasons over, so check the fields the
// system prompt tells it to weigh all reach it, normalized.
func TestPromptCarriesTheTransaction(t *testing.T) {
	t.Parallel()

	fake := &fakeCaller{reply: `{"assessments":[]}`}
	tx := sampleTx()
	tx.Status = " failed " // normalization should upper-case and trim this
	tx.Currency = "thb"

	if _, err := NewAnalyzer(fake).AnalyzeBatch(context.Background(), []model.Transaction{tx}); err != nil {
		t.Fatalf("AnalyzeBatch() error = %v", err)
	}

	for _, want := range []string{"tx_1001", "50000", "THB", "UNKNOWN_SHOP", "FAILED", "8"} {
		if !strings.Contains(fake.got.Prompt, want) {
			t.Errorf("prompt is missing %q:\n%s", want, fake.got.Prompt)
		}
	}
	if strings.Contains(fake.got.Prompt, " failed ") {
		t.Error("prompt carries the un-normalized status")
	}
	if fake.got.System == "" {
		t.Error("request carries no system prompt")
	}
}

// The schema is the only thing stopping the model returning a level the rest
// of the system cannot handle, so assert it is generated from model.RiskLevels.
func TestSchemaEnumMatchesDomain(t *testing.T) {
	t.Parallel()

	raw, err := json.Marshal(assessmentSchema())
	if err != nil {
		t.Fatalf("marshalling schema: %v", err)
	}

	var schema struct {
		Properties struct {
			Assessments struct {
				Items struct {
					Required   []string `json:"required"`
					Properties struct {
						RiskLevel struct {
							Enum []string `json:"enum"`
						} `json:"risk_level"`
					} `json:"properties"`
				} `json:"items"`
			} `json:"assessments"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(raw, &schema); err != nil {
		t.Fatalf("unmarshalling schema: %v", err)
	}

	gotEnum := schema.Properties.Assessments.Items.Properties.RiskLevel.Enum
	if len(gotEnum) != len(model.RiskLevels) {
		t.Fatalf("schema enum has %d levels, model.RiskLevels has %d", len(gotEnum), len(model.RiskLevels))
	}
	for i, level := range model.RiskLevels {
		if gotEnum[i] != string(level) {
			t.Errorf("schema enum[%d] = %q, want %q", i, gotEnum[i], level)
		}
	}

	wantRequired := []string{"transaction_id", "risk_level", "reason", "confidence"}
	got := schema.Properties.Assessments.Items.Required
	if len(got) != len(wantRequired) {
		t.Fatalf("schema requires %v, want %v", got, wantRequired)
	}
	for i, field := range wantRequired {
		if got[i] != field {
			t.Errorf("schema required[%d] = %q, want %q", i, got[i], field)
		}
	}
}
