"""Retry and fall back when Gemini is overloaded (HTTP 503) or rate limited (429)."""

import time

from google.genai import errors

RETRYABLE = {429, 500, 503, 504}


class GeminiBusyError(RuntimeError):
    """Gemini stayed unavailable after every retry and fallback model."""


def call_with_retry(call, models: list[str], attempts: int = 3, delay: float = 2.0,
                    sleep=time.sleep):
    """Run call(model) for each model in turn; retry temporary errors with growing waits.

    Errors that retrying cannot fix (bad API key, bad request) are raised immediately."""
    last = None
    for model in models:
        for attempt in range(attempts):
            try:
                return call(model)
            except errors.APIError as e:
                if e.code not in RETRYABLE:
                    raise
                last = e
                if attempt < attempts - 1:
                    sleep(delay * 2 ** attempt)      # 2s, 4s, ...
    raise GeminiBusyError(
        f"Gemini is busy right now (tried {', '.join(models)}). Please try again in a minute."
    ) from last
