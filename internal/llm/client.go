// Package llm wraps the Anthropic Messages API behind a small,
// domain-agnostic interface: send a prompt plus a JSON schema, get validated
// JSON back.
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// DefaultModel is the model every request uses unless WithModel overrides it.
const DefaultModel anthropic.Model = "claude-opus-5-5"

// Client talks to the Claude Messages API.
type Client struct {
	api       anthropic.Client
	model     anthropic.Model
	effort    anthropic.OutputConfigEffort
	maxTokens int64
	timeout   time.Duration
}

// Option configures a Client.
type Option func(*Client)

// WithModel overrides the model id used for every request.
func WithModel(model string) Option {
	return func(c *Client) {
		if model != "" {
			c.model = anthropic.Model(model)
		}
	}
}

// WithEffort sets how much effort the model spends on each response. Valid
// values are low, medium, high, xhigh and max.
func WithEffort(effort string) Option {
	return func(c *Client) {
		if effort != "" {
			c.effort = anthropic.OutputConfigEffort(effort)
		}
	}
}

// WithMaxTokens caps the response length.
func WithMaxTokens(maxTokens int64) Option {
	return func(c *Client) {
		if maxTokens > 0 {
			c.maxTokens = maxTokens
		}
	}
}

// WithTimeout bounds how long a single request may take.
func WithTimeout(d time.Duration) Option {
	return func(c *Client) {
		if d > 0 {
			c.timeout = d
		}
	}
}

// WithAPIKey sets the API key explicitly. Without it the SDK falls back to
// ANTHROPIC_API_KEY, then to the credentials stored by `ant auth login`.
func WithAPIKey(key string) Option {
	return func(c *Client) {
		if key != "" {
			c.api = anthropic.NewClient(option.WithAPIKey(key))
		}
	}
}

// New builds a Client. Options are applied in order, so WithAPIKey must come
// before any option that mutates the underlying SDK client.
func New(opts ...Option) *Client {
	c := &Client{
		api:       anthropic.NewClient(),
		model:     DefaultModel,
		effort:    anthropic.OutputConfigEffortMedium,
		maxTokens: 4096,
		timeout:   60 * time.Second,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Model reports the model id this client sends requests to.
func (c *Client) Model() string { return string(c.model) }

// StructuredRequest is one prompt that must come back as JSON matching Schema.
type StructuredRequest struct {
	// System is the system prompt. It is sent as a cacheable prefix, so keep
	// it stable across requests and put per-request detail in Prompt.
	System string
	// Prompt is the user message.
	Prompt string
	// Schema is the JSON schema the response must conform to. The top level
	// must be an object schema with "additionalProperties": false.
	Schema map[string]any
}

// ErrRefused is returned when Claude's safety classifiers decline the request.
var ErrRefused = errors.New("llm: request refused by the model")

// Structured sends req and unmarshals the JSON response into out.
func (c *Client) Structured(ctx context.Context, req StructuredRequest, out any) error {
	if req.Prompt == "" {
		return errors.New("llm: prompt is required")
	}
	if len(req.Schema) == 0 {
		return errors.New("llm: schema is required")
	}

	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	params := anthropic.MessageNewParams{
		Model:     c.model,
		MaxTokens: c.maxTokens,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(req.Prompt)),
		},
		OutputConfig: anthropic.OutputConfigParam{
			Effort: c.effort,
			Format: anthropic.JSONOutputFormatParam{Schema: req.Schema},
		},
	}
	if req.System != "" {
		params.System = []anthropic.TextBlockParam{{
			Text:         req.System,
			CacheControl: anthropic.NewCacheControlEphemeralParam(),
		}}
	}

	resp, err := c.api.Messages.New(ctx, params)
	if err != nil {
		return fmt.Errorf("llm: messages request failed: %w", err)
	}

	if resp.StopReason == anthropic.StopReasonRefusal {
		return fmt.Errorf("%w (category %q): %s", ErrRefused, resp.StopDetails.Category, resp.StopDetails.Explanation)
	}
	if resp.StopReason == anthropic.StopReasonMaxTokens {
		return fmt.Errorf("llm: response truncated at max_tokens=%d, raise it for this payload", c.maxTokens)
	}

	var payload strings.Builder
	for _, block := range resp.Content {
		if text, ok := block.AsAny().(anthropic.TextBlock); ok {
			payload.WriteString(text.Text)
		}
	}
	if payload.Len() == 0 {
		return errors.New("llm: response contained no text block")
	}

	if err := json.Unmarshal([]byte(payload.String()), out); err != nil {
		return fmt.Errorf("llm: decoding structured response: %w", err)
	}
	return nil
}
