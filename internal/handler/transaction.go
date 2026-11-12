// Package handler exposes the classifier over HTTP.
package handler

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"

	"github.com/BrodyP/ai-transaction-agent/internal/llm"
	"github.com/BrodyP/ai-transaction-agent/internal/model"
	"github.com/BrodyP/ai-transaction-agent/internal/service"
)

// maxBodyBytes caps a request body so a huge payload cannot exhaust memory.
const maxBodyBytes = 1 << 20 // 1 MiB

// classifier is the slice of service.Classifier this handler needs.
type classifier interface {
	Classify(ctx context.Context, tx model.Transaction) (model.Classification, error)
	ClassifyBatch(ctx context.Context, txs []model.Transaction) ([]model.Classification, error)
}

// Transaction serves the transaction classification endpoints.
type Transaction struct {
	classifier classifier
	log        *slog.Logger
}

// NewTransaction wires a handler to a classifier.
func NewTransaction(c classifier, log *slog.Logger) *Transaction {
	if log == nil {
		log = slog.Default()
	}
	return &Transaction{classifier: c, log: log}
}

// Routes returns the handler's routes mounted on a fresh mux.
func (h *Transaction) Routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", h.health)
	mux.HandleFunc("POST /v1/transactions/classify", h.classifyOne)
	mux.HandleFunc("POST /v1/transactions/classify-batch", h.classifyBatch)
	return mux
}

func (h *Transaction) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (h *Transaction) classifyOne(w http.ResponseWriter, r *http.Request) {
	var tx model.Transaction
	if err := decodeJSON(w, r, &tx); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := tx.Validate(); err != nil {
		writeError(w, http.StatusUnprocessableEntity, err.Error())
		return
	}

	result, err := h.classifier.Classify(r.Context(), tx)
	if err != nil {
		h.writeClassifyError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (h *Transaction) classifyBatch(w http.ResponseWriter, r *http.Request) {
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

	results, err := h.classifier.ClassifyBatch(r.Context(), req.Transactions)
	if err != nil {
		h.writeClassifyError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"classifications": results})
}

// writeClassifyError maps a classifier error onto a status code. Everything
// the caller could have fixed is a 4xx; anything else is logged and reported
// as a 5xx without leaking internals.
func (h *Transaction) writeClassifyError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, service.ErrEmptyBatch):
		writeError(w, http.StatusUnprocessableEntity, err.Error())
	case errors.Is(err, llm.ErrRefused):
		writeError(w, http.StatusUnprocessableEntity, "the model declined to classify this payload")
	case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
		writeError(w, http.StatusGatewayTimeout, "classification timed out")
	default:
		h.log.Error("classification failed", "error", err)
		writeError(w, http.StatusBadGateway, "classification failed")
	}
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
