package handler

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/BrodyP/ai-transaction-agent/internal/llm"
	"github.com/BrodyP/ai-transaction-agent/internal/model"
	"github.com/BrodyP/ai-transaction-agent/internal/service"
)

// fakeAnalyzer stands in for service.Analyzer.
type fakeAnalyzer struct {
	assessment model.RiskAssessment
	err        error
	gotTx      model.Transaction
}

func (f *fakeAnalyzer) Analyze(_ context.Context, tx model.Transaction) (model.RiskAssessment, error) {
	f.gotTx = tx
	return f.assessment, f.err
}

func (f *fakeAnalyzer) AnalyzeBatch(_ context.Context, txs []model.Transaction) ([]model.RiskAssessment, error) {
	if f.err != nil {
		return nil, f.err
	}
	out := make([]model.RiskAssessment, 0, len(txs))
	for range txs {
		out = append(out, f.assessment)
	}
	return out, nil
}

// The exact request and response shape the service is specified to speak.
func TestAnalyzeContract(t *testing.T) {
	t.Parallel()

	fake := &fakeAnalyzer{assessment: model.RiskAssessment{
		TransactionID: "tx_1001",
		RiskLevel:     model.RiskHigh,
		Reason:        "High amount and repeated failures",
		Confidence:    0.92,
	}}

	body := `{
		"transaction_id": "tx_1001",
		"amount": 50000,
		"currency": "THB",
		"merchant": "UNKNOWN_SHOP",
		"status": "FAILED",
		"retry_count": 8
	}`

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/transactions/analyze", strings.NewReader(body))
	NewTransaction(fake, nil).Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d (body: %s)", rec.Code, http.StatusOK, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}

	// Compare as decoded JSON so field order does not matter, but key names do.
	var got, want map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	if err := json.Unmarshal([]byte(`{
		"transaction_id": "tx_1001",
		"risk_level": "HIGH",
		"reason": "High amount and repeated failures",
		"confidence": 0.92
	}`), &want); err != nil {
		t.Fatalf("decoding expectation: %v", err)
	}
	if len(got) != len(want) {
		t.Errorf("response has keys %v, want %v", keys(got), keys(want))
	}
	for key, wantVal := range want {
		if gotVal, ok := got[key]; !ok {
			t.Errorf("response is missing %q", key)
		} else if gotVal != wantVal {
			t.Errorf("response[%q] = %v, want %v", key, gotVal, wantVal)
		}
	}

	// The handler must pass the request through unmangled.
	if fake.gotTx.RetryCount != 8 || fake.gotTx.Amount != 50000 || fake.gotTx.Status != "FAILED" {
		t.Errorf("analyzer received %+v", fake.gotTx)
	}
}

func TestAnalyzeStatusCodes(t *testing.T) {
	t.Parallel()

	valid := `{"transaction_id":"tx_1","amount":1,"currency":"THB","merchant":"m","status":"FAILED","retry_count":0}`

	tests := []struct {
		name     string
		path     string
		body     string
		err      error
		wantCode int
	}{
		{name: "ok", path: "/v1/transactions/analyze", body: valid, wantCode: http.StatusOK},
		{name: "malformed json", path: "/v1/transactions/analyze", body: `{"transaction_id":`, wantCode: http.StatusBadRequest},
		{name: "unknown field", path: "/v1/transactions/analyze", body: `{"transaction_idd":"tx_1"}`, wantCode: http.StatusBadRequest},
		{
			name:     "missing required fields",
			path:     "/v1/transactions/analyze",
			body:     `{"transaction_id":"","amount":1,"currency":"TH","merchant":"m","status":"","retry_count":0}`,
			wantCode: http.StatusUnprocessableEntity,
		},
		{
			name:     "negative retry count",
			path:     "/v1/transactions/analyze",
			body:     `{"transaction_id":"tx_1","amount":1,"currency":"THB","merchant":"m","status":"FAILED","retry_count":-2}`,
			wantCode: http.StatusUnprocessableEntity,
		},
		{name: "model refusal", path: "/v1/transactions/analyze", body: valid, err: llm.ErrRefused, wantCode: http.StatusUnprocessableEntity},
		{name: "rate limited", path: "/v1/transactions/analyze", body: valid, err: llm.ErrRateLimited, wantCode: http.StatusServiceUnavailable},
		{name: "quota exhausted", path: "/v1/transactions/analyze", body: valid, err: llm.ErrQuotaExhausted, wantCode: http.StatusServiceUnavailable},
		{name: "provider unavailable", path: "/v1/transactions/analyze", body: valid, err: llm.ErrUnavailable, wantCode: http.StatusServiceUnavailable},
		{name: "timeout", path: "/v1/transactions/analyze", body: valid, err: context.DeadlineExceeded, wantCode: http.StatusGatewayTimeout},
		{name: "model failure", path: "/v1/transactions/analyze", body: valid, err: errSentinel, wantCode: http.StatusBadGateway},
		{name: "empty batch", path: "/v1/transactions/analyze-batch", body: `{"transactions":[]}`, wantCode: http.StatusUnprocessableEntity},
		{
			name:     "batch rejected by service",
			path:     "/v1/transactions/analyze-batch",
			body:     `{"transactions":[` + valid + `]}`,
			err:      service.ErrEmptyBatch,
			wantCode: http.StatusUnprocessableEntity,
		},
		{name: "batch ok", path: "/v1/transactions/analyze-batch", body: `{"transactions":[` + valid + `]}`, wantCode: http.StatusOK},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			fake := &fakeAnalyzer{err: tc.err}
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			NewTransaction(fake, discardLogger()).Routes().ServeHTTP(rec, req)

			if rec.Code != tc.wantCode {
				t.Errorf("status = %d, want %d (body: %s)", rec.Code, tc.wantCode, strings.TrimSpace(rec.Body.String()))
			}
		})
	}
}

func TestBatchResponseIsEnveloped(t *testing.T) {
	t.Parallel()

	fake := &fakeAnalyzer{assessment: model.RiskAssessment{TransactionID: "tx_1", RiskLevel: model.RiskLow}}
	body := `{"transactions":[
		{"transaction_id":"tx_1","amount":1,"currency":"THB","merchant":"m","status":"SUCCESS","retry_count":0},
		{"transaction_id":"tx_2","amount":2,"currency":"THB","merchant":"m","status":"SUCCESS","retry_count":0}]}`

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/transactions/analyze-batch", strings.NewReader(body))
	NewTransaction(fake, nil).Routes().ServeHTTP(rec, req)

	var got struct {
		Assessments []model.RiskAssessment `json:"assessments"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	if len(got.Assessments) != 2 {
		t.Errorf("got %d assessments, want 2", len(got.Assessments))
	}
}

func TestHealthz(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	NewTransaction(&fakeAnalyzer{}, nil).Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	if got := strings.TrimSpace(rec.Body.String()); got != `{"status":"ok"}` {
		t.Errorf("body = %s", got)
	}
}

// An upstream failure must become a 503 that names an actionable cause, pass on
// the provider's Retry-After, and keep the provider's own message — which can
// name the account — out of the response body.
func TestUpstreamErrorResponse(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		err            error
		wantRetryAfter string
		wantInBody     string
	}{
		{
			name: "quota with retry hint",
			err: &llm.UpstreamError{
				Kind:       llm.ErrQuotaExhausted,
				Provider:   llm.ProviderOpenAI,
				StatusCode: http.StatusTooManyRequests,
				RetryAfter: 30 * time.Second,
				Err:        errors.New("You have no credits remaining for org-secret-name"),
			},
			wantRetryAfter: "30",
			wantInBody:     "credit or quota",
		},
		{
			name: "rate limit rounds a fractional hint up",
			err: &llm.UpstreamError{
				Kind:       llm.ErrRateLimited,
				Provider:   llm.ProviderAnthropic,
				StatusCode: http.StatusTooManyRequests,
				RetryAfter: 1500 * time.Millisecond,
			},
			wantRetryAfter: "2",
			wantInBody:     "rate limiting",
		},
		{
			name: "unavailable without a hint sends no header",
			err: &llm.UpstreamError{
				Kind:       llm.ErrUnavailable,
				Provider:   llm.ProviderAnthropic,
				StatusCode: 529,
			},
			wantRetryAfter: "",
			wantInBody:     "temporarily unavailable",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/v1/transactions/analyze",
				strings.NewReader(`{"transaction_id":"tx_1","amount":1,"currency":"THB","merchant":"m","status":"FAILED","retry_count":0}`))
			NewTransaction(&fakeAnalyzer{err: tc.err}, discardLogger()).Routes().ServeHTTP(rec, req)

			if rec.Code != http.StatusServiceUnavailable {
				t.Errorf("status = %d, want %d", rec.Code, http.StatusServiceUnavailable)
			}
			if got := rec.Header().Get("Retry-After"); got != tc.wantRetryAfter {
				t.Errorf("Retry-After = %q, want %q", got, tc.wantRetryAfter)
			}
			body := rec.Body.String()
			if !strings.Contains(body, tc.wantInBody) {
				t.Errorf("body = %s, missing %q", strings.TrimSpace(body), tc.wantInBody)
			}
			if strings.Contains(body, "org-secret-name") {
				t.Error("the provider's own error message leaked into the response body")
			}
		})
	}
}
