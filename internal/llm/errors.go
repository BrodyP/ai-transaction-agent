package llm

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Failures a caller can act on differently. A provider reports all three as an
// HTTP error, but they call for different handling: wait and retry, top up the
// account, or wait for the provider to recover.
var (
	// ErrRateLimited means the provider throttled us. Retrying later works.
	ErrRateLimited = errors.New("llm: rate limited by the provider")
	// ErrQuotaExhausted means the account is out of credit or quota.
	// Retrying never helps until somebody pays or raises the limit.
	ErrQuotaExhausted = errors.New("llm: provider quota or credit exhausted")
	// ErrUnavailable means the provider is overloaded or down.
	ErrUnavailable = errors.New("llm: provider temporarily unavailable")
)

// UpstreamError is a provider failure classified into one of the sentinels
// above, carrying the provider's own retry hint when it gave one.
//
// errors.Is matches both the sentinel and the underlying SDK error, so callers
// can branch on the kind and still reach the original for logging.
type UpstreamError struct {
	Kind       error    // ErrRateLimited, ErrQuotaExhausted or ErrUnavailable
	Provider   Provider // which backend failed
	StatusCode int      // the provider's HTTP status
	RetryAfter time.Duration
	Err        error // the SDK error this was classified from
}

func (e *UpstreamError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "llm/%s: %s (HTTP %d)", e.Provider, e.Kind, e.StatusCode)
	if e.RetryAfter > 0 {
		fmt.Fprintf(&b, ", retry after %s", e.RetryAfter)
	}
	if e.Err != nil {
		fmt.Fprintf(&b, ": %v", e.Err)
	}
	return b.String()
}

// Unwrap exposes both the kind and the SDK error to errors.Is and errors.As.
func (e *UpstreamError) Unwrap() []error {
	if e.Err == nil {
		return []error{e.Kind}
	}
	return []error{e.Kind, e.Err}
}

// RetryAfterFrom reads a Retry-After header, which may be a number of seconds
// or an HTTP date. It returns 0 when the header is absent or unparseable, so a
// caller can treat 0 as "no hint from the provider".
func RetryAfterFrom(resp *http.Response) time.Duration {
	if resp == nil {
		return 0
	}
	raw := strings.TrimSpace(resp.Header.Get("Retry-After"))
	if raw == "" {
		return 0
	}
	if secs, err := strconv.Atoi(raw); err == nil {
		if secs < 0 {
			return 0
		}
		return time.Duration(secs) * time.Second
	}
	if when, err := http.ParseTime(raw); err == nil {
		if d := time.Until(when); d > 0 {
			return d
		}
	}
	return 0
}
