// Package handler exposes the risk analyzer over HTTP.
package handler

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"math"
	"net/http"
	"strconv"

	"github.com/BrodyP/ai-transaction-agent/internal/llm"
	"github.com/BrodyP/ai-transaction-agent/internal/model"
	"github.com/BrodyP/ai-transaction-agent/internal/service"
)

// maxBodyBytes caps a request body so a huge payload cannot exhaust memory.
const maxBodyBytes = 1 << 20 // 1 MiB

// analyzer is the slice of service.Analyzer this handler needs.
type analyzer interface {
	Analyze(ctx context.Context, tx model.Transaction) (model.RiskAssessment, error)
	AnalyzeBatch(ctx context.Context, txs []model.Transaction) ([]model.RiskAssessment, error)
}

// Transaction serves the transaction risk analysis endpoints.
type Transaction struct {
	analyzer analyzer
	log      *slog.Logger
}

// NewTransaction wires a handler to an analyzer.
func NewTransaction(a analyzer, log *slog.Logger) *Transaction {
	if log == nil {
		log = slog.Default()
	}
	return &Transaction{analyzer: a, log: log}
}

// Routes returns the handler's routes mounted on a fresh mux.
func (h *Transaction) Routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", h.health)
	mux.HandleFunc("POST /v1/transactions/analyze", h.analyzeOne)
	mux.HandleFunc("POST /v1/transactions/analyze-batch", h.analyzeBatch)
	return mux
}

func (h *Transaction) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (h *Transaction) analyzeOne(w http.ResponseWriter, r *http.Request) {
	var tx model.Transaction
	if err := decodeJSON(w, r, &tx); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := tx.Normalized().Validate(); err != nil {
		writeError(w, http.StatusUnprocessableEntity, err.Error())
		return
	}

	assessment, err := h.analyzer.Analyze(r.Context(), tx)
	if err != nil {
		h.writeAnalyzeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, assessment)
}

func (h *Transaction) analyzeBatch(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Transactions []model.Transaction `json:"transactions"`
	}
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Transactions) == 0 {
		writeError(w, http.StatusUnprocessableEntity, "transactions must not be empty")
		return
	}

	assessments, err := h.analyzer.AnalyzeBatch(r.Context(), req.Transactions)
	if err != nil {
		h.writeAnalyzeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"assessments": assessments})
}

// writeAnalyzeError maps an analyzer error onto a status code. Everything the
// caller could have fixed is a 4xx; anything else is logged and reported as a
// 5xx without leaking internals.
//
// A provider that throttles us, runs out of credit or falls over is a 503
// rather than a 429: the caller did not send too many requests, this service
// did, so blaming them would be wrong. Each of those carries a reason the
// caller can act on and, when the provider offered one, a Retry-After.
func (h *Transaction) writeAnalyzeError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, service.ErrEmptyBatch):
		writeError(w, http.StatusUnprocessableEntity, err.Error())
	case errors.Is(err, llm.ErrRefused):
		writeError(w, http.StatusUnprocessableEntity, "the model declined to assess this payload")
	case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
		writeError(w, http.StatusGatewayTimeout, "risk analysis timed out")
	case errors.Is(err, llm.ErrRateLimited):
		h.writeUnavailable(w, err, "the model provider is rate limiting this service")
	case errors.Is(err, llm.ErrQuotaExhausted):
		h.writeUnavailable(w, err, "the model provider reports no remaining credit or quota for this service")
	case errors.Is(err, llm.ErrUnavailable):
		h.writeUnavailable(w, err, "the model provider is temporarily unavailable")
	default:
		h.log.Error("risk analysis failed", "error", err)
		writeError(w, http.StatusBadGateway, "risk analysis failed")
	}
}

// writeUnavailable reports an upstream problem as 503, passing on the
// provider's Retry-After when it gave one. The full provider error goes to the
// log, never to the caller — a quota message naming the account is for whoever
// operates this service, not for whoever called it.
func (h *Transaction) writeUnavailable(w http.ResponseWriter, err error, message string) {
	h.log.Error("model provider unavailable", "error", err)

	var upstream *llm.UpstreamError
	if errors.As(err, &upstream) && upstream.RetryAfter > 0 {
		secs := int(math.Ceil(upstream.RetryAfter.Seconds()))
		w.Header().Set("Retry-After", strconv.Itoa(secs))
	}
	writeError(w, http.StatusServiceUnavailable, message)
}

func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return errors.New("invalid JSON body: " + err.Error())
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(body); err != nil {
		slog.Error("writing response body", "error", err)
	}
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}
