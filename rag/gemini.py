"""Google Gemini access through its OpenAI-compatible endpoint.

The generator, the follow-up rewriter, the image judge and the vision describer only call
`client.chat.completions.create(model=..., max_tokens=..., messages=...)`, so a client for
Gemini slots in without changing any of them. Retry and fallback (`call_with_retry` in
rag/openrouter.py) work unchanged: the SDK raises the same errors.
"""

from types import SimpleNamespace

import openai

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Gemini's newer models "think" before they answer, and the thinking tokens count against
# max_tokens. A small limit (the follow-up rewriter asks for 256) could be used up by
# thinking and leave an empty reply, so every request gets this much extra room.
THINKING_HEADROOM = 1024


class GeminiClient:
    """An OpenAI-SDK client for Gemini that looks like `openai.OpenAI` to the rest of the app."""

    def __init__(self, api_key: str, reasoning_effort: str | None = None,
                 headroom: int = THINKING_HEADROOM, client=None):
        self.reasoning_effort = reasoning_effort or None
        self.headroom = headroom
        self._client = client or openai.OpenAI(
            api_key=api_key, base_url=GEMINI_BASE_URL,
            max_retries=0, timeout=60)   # we do our own retries (call_with_retry)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if kwargs.get("max_tokens"):
            kwargs["max_tokens"] += self.headroom
        if self.reasoning_effort:
            kwargs.setdefault("reasoning_effort", self.reasoning_effort)
        return self._client.chat.completions.create(**kwargs)


def make_gemini_client(api_key: str, reasoning_effort: str | None = None) -> GeminiClient:
    """`reasoning_effort` (for example "low") is only sent when set; the default is to leave
    the model's own setting alone."""
    return GeminiClient(api_key, reasoning_effort)
