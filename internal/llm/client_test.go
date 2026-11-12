package llm

import (
	"errors"
	"testing"
)

func TestNewSelectsProvider(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		provider     string
		model        string
		wantProvider Provider
		wantModel    string
		wantErr      error
	}{
		{name: "default is anthropic", wantProvider: ProviderAnthropic, wantModel: DefaultAnthropicModel},
		{name: "anthropic", provider: "anthropic", wantProvider: ProviderAnthropic, wantModel: DefaultAnthropicModel},
		{name: "openai", provider: "openai", wantProvider: ProviderOpenAI, wantModel: DefaultOpenAIModel},
		{name: "case and space insensitive", provider: " OpenAI ", wantProvider: ProviderOpenAI, wantModel: DefaultOpenAIModel},
		{name: "explicit model wins", provider: "openai", model: "gpt-5-nano", wantProvider: ProviderOpenAI, wantModel: "gpt-5-nano"},
		{name: "unknown provider", provider: "gemini", wantErr: ErrUnknownProvider},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// A fake key keeps the SDKs from reading the ambient environment.
			got, err := New(
				WithProvider(tc.provider),
				WithModel(tc.model),
				WithAPIKey("test-key-not-used"),
			)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("New() error = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			if got.Provider() != tc.wantProvider {
				t.Errorf("Provider() = %q, want %q", got.Provider(), tc.wantProvider)
			}
			if got.Model() != tc.wantModel {
				t.Errorf("Model() = %q, want %q", got.Model(), tc.wantModel)
			}
		})
	}
}

// Both providers must reject a bad request before spending a round trip, and
// they must agree on which requests are bad.
func TestStructuredRejectsBadRequest(t *testing.T) {
	t.Parallel()

	schema := map[string]any{"type": "object", "additionalProperties": false}

	requests := []struct {
		name string
		req  StructuredRequest
	}{
		{name: "no prompt", req: StructuredRequest{Schema: schema}},
		{name: "no schema", req: StructuredRequest{Prompt: "hello"}},
	}

	for _, provider := range []string{"anthropic", "openai"} {
		for _, tc := range requests {
			t.Run(provider+"/"+tc.name, func(t *testing.T) {
				t.Parallel()

				client, err := New(WithProvider(provider), WithAPIKey("test-key-not-used"))
				if err != nil {
					t.Fatalf("New() error = %v", err)
				}
				var out map[string]any
				if err := client.Structured(t.Context(), tc.req, &out); err == nil {
					t.Error("Structured() error = nil, want a validation error")
				}
			})
		}
	}
}

func TestSchemaNameDefaults(t *testing.T) {
	t.Parallel()

	if got := (StructuredRequest{}).name(); got != "response" {
		t.Errorf("name() = %q, want %q", got, "response")
	}
	if got := (StructuredRequest{SchemaName: "risk"}).name(); got != "risk" {
		t.Errorf("name() = %q, want %q", got, "risk")
	}
}
