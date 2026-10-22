"""OpenRouter access through the standard OpenAI SDK, plus retry and fallback."""

import time

import openai

BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE = {408, 429, 500, 502, 503, 504}


class OpenRouterBusyError(RuntimeError):
    """The models stayed unavailable after every retry and fallback."""


def make_client(api_key: str, site_url: str | None = None,
                app_name: str | None = None) -> openai.OpenAI:
    """HTTP-Referer and X-Title are optional: OpenRouter uses them to credit your app."""
    headers = {}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name
    return openai.OpenAI(api_key=api_key, base_url=BASE_URL, default_headers=headers,
                         max_retries=0, timeout=60)   # we do our own retries below


def _temporary(e: Exception) -> bool:
    if isinstance(e, openai.APIStatusError):
        return e.status_code in RETRYABLE
    return isinstance(e, (openai.APIConnectionError, openai.APITimeoutError))


def call_with_retry(call, models: list[str], attempts: int = 3, delay: float = 2.0,
                    sleep=time.sleep):
    """Run call(model) for each model in turn; retry temporary errors with growing waits.

    Errors that retrying cannot fix (bad API key, bad request) are raised immediately."""
    last = None
    for model in models:
        for attempt in range(attempts):
            try:
                return call(model)
            except openai.OpenAIError as e:
                if not _temporary(e):
                    raise
                last = e
                if attempt < attempts - 1:
                    sleep(delay * 2 ** attempt)      # 2s, 4s, ...
    raise OpenRouterBusyError(
        f"The AI service is busy right now (tried {', '.join(models)}). "
        "Please try again in a minute."
    ) from last
