// Package llm wraps a chat-completion provider behind a small,
// domain-agnostic interface: send a prompt plus a JSON schema, get validated
// JSON back. Anthropic and OpenAI implementations live alongside this file and
// are selected by WithProvider.
package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

// Provider names a supported backend.
type Provider string

const (
	ProviderAnthropic Provider = "anthropic"
	ProviderOpenAI    Provider = "openai"
)

// Default models per provider. These are not the same capability tier — the
// Anthropic default is a frontier model, the OpenAI default a mid-tier one —
// so set the model explicitly before comparing the two.
const (
	DefaultAnthropicModel = "claude-opus-5-5"
	DefaultOpenAIModel    = "gpt-5.4-mini"
)

// ErrRefused is returned when the provider's safety systems decline the
// request rather than answering it.
var ErrRefused = errors.New("llm: request refused by the model")

// ErrUnknownProvider is returned by New for a provider it does not implement.
var ErrUnknownProvider = errors.New("llm: unknown provider")

// Client sends a prompt and decodes a schema-constrained JSON response.
type Client interface {
	// Structured sends req and unmarshals the JSON response into out.
	Structured(ctx context.Context, req StructuredRequest, out any) error
	// Model reports the model id this client sends requests to.
	Model() string
	// Provider reports which backend this client talks to.
	Provider() Provider
}

// StructuredRequest is one prompt that must come back as JSON matching Schema.
type StructuredRequest struct {
	// System is the system prompt. Both providers cache it as a prefix, so
	// keep it stable across requests and put per-request detail in Prompt.
	System string
	// Prompt is the user message.
	Prompt string
	// Schema is the JSON schema the response must conform to. The top level
	// must be an object schema with "additionalProperties": false and every
	// property listed in "required" — the intersection of what Anthropic's
	// structured outputs and OpenAI's strict mode accept.
	Schema map[string]any
	// SchemaName identifies the schema. OpenAI requires one and allows only
	// [a-zA-Z0-9_-]; Anthropic ignores it. Defaults to "response".
	SchemaName string
}

// config holds the settings shared by every provider.
type config struct {
	provider  Provider
	model     string
	effort    string
	maxTokens int64
	timeout   time.Duration
	apiKey    string
}

// Option configures a Client.
type Option func(*config)

// WithProvider selects the backend. Defaults to Anthropic.
func WithProvider(provider string) Option {
	return func(c *config) {
		if provider != "" {
			c.provider = Provider(strings.ToLower(strings.TrimSpace(provider)))
		}
	}
}

// WithModel overrides the model id. Empty keeps the provider's default.
func WithModel(model string) Option {
	return func(c *config) {
		if model != "" {
			c.model = model
		}
	}
}

// WithEffort sets how much reasoning effort the model spends per response.
// Both providers accept low, medium, high, xhigh and max.
func WithEffort(effort string) Option {
	return func(c *config) {
		if effort != "" {
			c.effort = effort
		}
	}
}

// WithMaxTokens caps the response length, reasoning tokens included.
func WithMaxTokens(maxTokens int64) Option {
	return func(c *config) {
		if maxTokens > 0 {
			c.maxTokens = maxTokens
		}
	}
}

// WithTimeout bounds how long a single request may take.
func WithTimeout(d time.Duration) Option {
	return func(c *config) {
		if d > 0 {
			c.timeout = d
		}
	}
}

// WithAPIKey sets the key explicitly. Without it each SDK reads its own
// environment variable (ANTHROPIC_API_KEY or OPENAI_API_KEY).
func WithAPIKey(key string) Option {
	return func(c *config) {
		if key != "" {
			c.apiKey = key
		}
	}
}

// New builds a Client for the configured provider.
func New(opts ...Option) (Client, error) {
	cfg := config{
		provider:  ProviderAnthropic,
		effort:    "medium",
		maxTokens: 4096,
		timeout:   60 * time.Second,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	switch cfg.provider {
	case ProviderAnthropic:
		return newAnthropic(cfg), nil
	case ProviderOpenAI:
		return newOpenAI(cfg), nil
	default:
		return nil, fmt.Errorf("%w: %q (want %q or %q)", ErrUnknownProvider, cfg.provider, ProviderAnthropic, ProviderOpenAI)
	}
}

// validate checks a request before it costs a network round trip.
func (r StructuredRequest) validate() error {
	if r.Prompt == "" {
		return errors.New("llm: prompt is required")
	}
	if len(r.Schema) == 0 {
		return errors.New("llm: schema is required")
	}
	return nil
}

// name returns the schema name to send, falling back to a valid default.
func (r StructuredRequest) name() string {
	if r.SchemaName != "" {
		return r.SchemaName
	}
	return "response"
}
