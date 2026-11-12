package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/shared"
)

// anthropicClient talks to the Claude Messages API.
type anthropicClient struct {
	api anthropic.Client
	cfg config
}

func newAnthropic(cfg config) *anthropicClient {
	if cfg.model == "" {
		cfg.model = DefaultAnthropicModel
	}

	opts := []option.RequestOption{}
	if cfg.apiKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.apiKey))
	}
	return &anthropicClient{api: anthropic.NewClient(opts...), cfg: cfg}
}

func (c *anthropicClient) Model() string      { return c.cfg.model }
func (c *anthropicClient) Provider() Provider { return ProviderAnthropic }

// Structured sends req and unmarshals the JSON response into out.
func (c *anthropicClient) Structured(ctx context.Context, req StructuredRequest, out any) error {
	if err := req.validate(); err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(ctx, c.cfg.timeout)
	defer cancel()

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(c.cfg.model),
		MaxTokens: c.cfg.maxTokens,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(req.Prompt)),
		},
		OutputConfig: anthropic.OutputConfigParam{
			Effort: anthropic.OutputConfigEffort(c.cfg.effort),
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
		if upstream := classifyAnthropic(err); upstream != nil {
			return upstream
		}
		return fmt.Errorf("llm/anthropic: messages request failed: %w", err)
	}

	switch resp.StopReason {
	case anthropic.StopReasonRefusal:
		return fmt.Errorf("%w (category %q): %s", ErrRefused, resp.StopDetails.Category, resp.StopDetails.Explanation)
	case anthropic.StopReasonMaxTokens:
		return fmt.Errorf("llm/anthropic: response truncated at max_tokens=%d, raise it for this payload", c.cfg.maxTokens)
	}

	var payload strings.Builder
	for _, block := range resp.Content {
		if text, ok := block.AsAny().(anthropic.TextBlock); ok {
			payload.WriteString(text.Text)
		}
	}
	if payload.Len() == 0 {
		return errors.New("llm/anthropic: response contained no text block")
	}

	if err := json.Unmarshal([]byte(payload.String()), out); err != nil {
		return fmt.Errorf("llm/anthropic: decoding structured response: %w", err)
	}
	return nil
}

// classifyAnthropic maps an SDK error onto one of the package's upstream
// sentinels, or returns nil when it is some other failure the caller should
// see verbatim.
func classifyAnthropic(err error) *UpstreamError {
	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) {
		return nil
	}

	var kind error
	switch {
	case apiErr.Type() == shared.ErrorTypeBillingError:
		kind = ErrQuotaExhausted
	case apiErr.Type() == shared.ErrorTypeRateLimitError, apiErr.StatusCode == http.StatusTooManyRequests:
		kind = ErrRateLimited
	case apiErr.Type() == shared.ErrorTypeOverloadedError,
		apiErr.StatusCode == http.StatusServiceUnavailable,
		apiErr.StatusCode == 529:
		kind = ErrUnavailable
	default:
		return nil
	}

	return &UpstreamError{
		Kind:       kind,
		Provider:   ProviderAnthropic,
		StatusCode: apiErr.StatusCode,
		RetryAfter: RetryAfterFrom(apiErr.Response),
		Err:        err,
	}
}
