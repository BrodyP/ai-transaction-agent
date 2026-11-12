package llm

import (
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestUpstreamErrorMatchesKindAndCause(t *testing.T) {
	t.Parallel()

	cause := errors.New("429 Too Many Requests")
	err := error(&UpstreamError{
		Kind:       ErrQuotaExhausted,
		Provider:   ProviderOpenAI,
		StatusCode: http.StatusTooManyRequests,
		RetryAfter: 30 * time.Second,
		Err:        cause,
	})

	if !errors.Is(err, ErrQuotaExhausted) {
		t.Error("errors.Is did not match the kind")
	}
	if !errors.Is(err, cause) {
		t.Error("errors.Is did not match the underlying cause")
	}
	if errors.Is(err, ErrRateLimited) {
		t.Error("errors.Is matched the wrong kind")
	}

	var upstream *UpstreamError
	if !errors.As(err, &upstream) {
		t.Fatal("errors.As did not recover the UpstreamError")
	}
	if upstream.RetryAfter != 30*time.Second {
		t.Errorf("RetryAfter = %v, want 30s", upstream.RetryAfter)
	}

	// The message has to name the provider, the kind and the hint, because it
	// is what lands in the operator's log.
	for _, want := range []string{"openai", "quota", "429", "30s"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("Error() = %q, missing %q", err.Error(), want)
		}
	}
}

func TestUpstreamErrorWithoutCause(t *testing.T) {
	t.Parallel()

	err := error(&UpstreamError{Kind: ErrUnavailable, Provider: ProviderAnthropic, StatusCode: 529})
	if !errors.Is(err, ErrUnavailable) {
		t.Error("errors.Is did not match the kind")
	}
}

func TestRetryAfterFrom(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		header string
		// future puts a real timestamp 90s ahead in the header instead of
		// using the literal above, and asserts a positive duration.
		future bool
		want   time.Duration
	}{
		{name: "absent", header: "", want: 0},
		{name: "seconds", header: "42", want: 42 * time.Second},
		{name: "zero seconds", header: "0", want: 0},
		{name: "negative seconds", header: "-5", want: 0},
		{name: "whitespace padded", header: "  7  ", want: 7 * time.Second},
		{name: "garbage", header: "soon", want: 0},
		{name: "past http date", header: "Mon, 02 Jan 2006 15:04:05 GMT", want: 0},
		{name: "future http date", future: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp := &http.Response{Header: http.Header{}}
			switch {
			case tc.future:
				resp.Header.Set("Retry-After", time.Now().Add(90*time.Second).UTC().Format(http.TimeFormat))
			case tc.header != "":
				resp.Header.Set("Retry-After", tc.header)
			}

			got := RetryAfterFrom(resp)
			if tc.future {
				if got <= 0 || got > 2*time.Minute {
					t.Errorf("RetryAfterFrom() = %v, want a positive duration under 2m", got)
				}
				return
			}
			if got != tc.want {
				t.Errorf("RetryAfterFrom() = %v, want %v", got, tc.want)
			}
		})
	}

	if got := RetryAfterFrom(nil); got != 0 {
		t.Errorf("RetryAfterFrom(nil) = %v, want 0", got)
	}
}
