package handler

import (
	"errors"
	"io"
	"log/slog"
	"sort"
)

// errSentinel is an error with no special meaning to the handler, used to
// exercise the catch-all 502 branch.
var errSentinel = errors.New("the model exploded")

// discardLogger keeps expected-error test cases from writing to stderr.
func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func keys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
