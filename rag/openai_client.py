"""OpenAI as the provider that answers questions (and hears and speaks, see rag/voice.py).

OpenAI is what the OpenAI SDK talks to by default, so this is a thin wrapper that exposes the
same `client.chat.completions.create(model, max_tokens, messages)` call the generator,
rewriter, image judge and vision describer already use. Nothing else needs to change.
"""

import re
from types import SimpleNamespace

import openai

# gpt-5 and the o-series think before answering, and the thinking counts against the limit.
_REASONING_MODEL = re.compile(r"^(gpt-5|o\d)", re.IGNORECASE)
THINKING_HEADROOM = 1024


class OpenAIClient:
    """An OpenAI SDK client with the two differences newer OpenAI models need."""

    def __init__(self, api_key: str, reasoning_effort: str | None = None,
                 headroom: int = THINKING_HEADROOM, client=None):
        self._client = client or openai.OpenAI(api_key=api_key, max_retries=0, timeout=60)
        self.reasoning_effort = reasoning_effort
        self.headroom = headroom
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @property
    def audio(self):
        """Speech to text and text to speech (used by rag/voice.py)."""
        return self._client.audio

    def _create(self, **kwargs):
        reasoning = bool(_REASONING_MODEL.match(str(kwargs.get("model", ""))))
        limit = kwargs.pop("max_tokens", None)
        if limit:
            # newer models reject max_tokens; max_completion_tokens is accepted by all of them
            kwargs["max_completion_tokens"] = limit + (self.headroom if reasoning else 0)
        if reasoning and self.reasoning_effort:
            kwargs.setdefault("reasoning_effort", self.reasoning_effort)
        return self._client.chat.completions.create(**kwargs)


def make_openai_client(api_key: str, reasoning_effort: str | None = None) -> OpenAIClient:
    """max_retries=0 because call_with_retry (rag/openrouter.py) does the retrying."""
    return OpenAIClient(api_key, reasoning_effort)
