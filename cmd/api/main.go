// Command api serves the transaction classification HTTP API.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/BrodyP/ai-transaction-agent/internal/handler"
	"github.com/BrodyP/ai-transaction-agent/internal/llm"
	"github.com/BrodyP/ai-transaction-agent/internal/service"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: logLevel()}))
	slog.SetDefault(log)

	if err := run(log); err != nil {
		log.Error("server exited with an error", "error", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger) error {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		return errors.New("ANTHROPIC_API_KEY is not set")
	}

	llmClient := llm.New(
		llm.WithModel(os.Getenv("CLAUDE_MODEL")),
		llm.WithEffort(os.Getenv("CLAUDE_EFFORT")),
		llm.WithTimeout(envDuration("LLM_TIMEOUT", 60*time.Second)),
	)
	classifier := service.NewClassifier(llmClient)
	routes := handler.NewTransaction(classifier, log).Routes()

	addr := ":" + env("PORT", "8080")
	srv := &http.Server{
		Addr:              addr,
		Handler:           routes,
		ReadHeaderTimeout: 10 * time.Second,
		// Generous, because a batch classification waits on the model.
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Shut down cleanly on SIGINT/SIGTERM so in-flight requests finish.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		log.Info("listening", "addr", addr, "model", llmClient.Model())
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		log.Info("shutdown signal received, draining connections")
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		return err
	}
	log.Info("shutdown complete")
	return nil
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envDuration(key string, fallback time.Duration) time.Duration {
	raw := os.Getenv(key)
	if raw == "" {
		return fallback
	}
	d, err := time.ParseDuration(raw)
	if err != nil {
		slog.Warn("ignoring unparseable duration", "key", key, "value", raw)
		return fallback
	}
	return d
}

func logLevel() slog.Level {
	var level slog.Level
	if raw := os.Getenv("LOG_LEVEL"); raw != "" {
		if err := level.UnmarshalText([]byte(raw)); err != nil {
			// Nothing is configured yet, so this one goes to the bare logger.
			slog.Warn("ignoring unknown LOG_LEVEL, using info", "value", strconv.Quote(raw))
			return slog.LevelInfo
		}
	}
	return level
}
