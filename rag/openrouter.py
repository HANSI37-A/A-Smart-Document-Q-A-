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


QUOTA_CODES = {"insufficient_quota", "credit_balance_exhausted"}


def _out_of_credit(e: Exception) -> bool:
    """A 429 that means the account has no credit left. Waiting does not fix that."""
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        body = body.get("error", body)     # some providers nest the error object
    return (isinstance(e, openai.RateLimitError) and isinstance(body, dict)
            and (body.get("code") in QUOTA_CODES or body.get("type") in QUOTA_CODES))


def _temporary(e: Exception) -> bool:
    if _out_of_credit(e):
        return False
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
                if _out_of_credit(e):      # every model and retry would fail the same way
                    raise OpenRouterBusyError(
                        "The AI account has no credit left. Add credit to your account "
                        "(for OpenAI: platform.openai.com/settings/organization/billing) "
                        "and try again.") from e
                if not _temporary(e):
                    raise
                last = e
                if attempt < attempts - 1:
                    sleep(delay * 2 ** attempt)      # 2s, 4s, ...
    raise OpenRouterBusyError(
        f"The AI service is busy right now (tried {', '.join(models)}). "
        "Please try again in a minute."
    ) from last
