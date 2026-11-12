package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

// openAIClient talks to the OpenAI Responses API.
//
// The schema in a StructuredRequest is sent unchanged: strict mode accepts the
// same object-with-additionalProperties-false shape Anthropic wants, and it
// also honours minimum/maximum on numbers, so no translation is needed. Its
// extra rules — the root must be an object, every property must appear in
// "required" — are already part of the StructuredRequest contract.
type openAIClient struct {
	api openai.Client
	cfg config
}

func newOpenAI(cfg config) *openAIClient {
	if cfg.model == "" {
		cfg.model = DefaultOpenAIModel
	}

	opts := []option.RequestOption{}
	if cfg.apiKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.apiKey))
	}
	return &openAIClient{api: openai.NewClient(opts...), cfg: cfg}
}

func (c *openAIClient) Model() string      { return c.cfg.model }
func (c *openAIClient) Provider() Provider { return ProviderOpenAI }

// Structured sends req and unmarshals the JSON response into out.
func (c *openAIClient) Structured(ctx context.Context, req StructuredRequest, out any) error {
	if err := req.validate(); err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(ctx, c.cfg.timeout)
	defer cancel()

	params := responses.ResponseNewParams{
		Model:           shared.ResponsesModel(c.cfg.model),
		MaxOutputTokens: openai.Int(c.cfg.maxTokens),
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String(req.Prompt),
		},
		Reasoning: shared.ReasoningParam{
			Effort: shared.ReasoningEffort(c.cfg.effort),
		},
		Text: responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
					Name:   req.name(),
					Schema: req.Schema,
					Strict: openai.Bool(true),
				},
			},
		},
	}
	if req.System != "" {
		params.Instructions = openai.String(req.System)
	}

	resp, err := c.api.Responses.New(ctx, params)
	if err != nil {
		return fmt.Errorf("llm/openai: responses request failed: %w", err)
	}

	// A refusal arrives as a content part, not an error, so check for one
	// before trusting the output text.
	if refusal := firstRefusal(resp); refusal != "" {
		return fmt.Errorf("%w: %s", ErrRefused, refusal)
	}
	if resp.Status == responses.ResponseStatusIncomplete {
		return fmt.Errorf("llm/openai: response incomplete (%s), raise max_tokens for this payload",
			resp.IncompleteDetails.Reason)
	}

	payload := strings.TrimSpace(resp.OutputText())
	if payload == "" {
		return errors.New("llm/openai: response contained no output text")
	}

	if err := json.Unmarshal([]byte(payload), out); err != nil {
		return fmt.Errorf("llm/openai: decoding structured response: %w", err)
	}
	return nil
}

// firstRefusal returns the explanation from the first refusal content part, or
// an empty string when the model did not refuse.
func firstRefusal(resp *responses.Response) string {
	for _, item := range resp.Output {
		for _, content := range item.Content {
			if content.Type == "refusal" {
				return content.Refusal
			}
		}
	}
	return ""
}
