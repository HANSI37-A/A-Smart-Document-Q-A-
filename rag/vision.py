"""Turn an image of a chart or figure into searchable text using a vision model."""

import base64
from typing import Protocol

from rag.openrouter import call_with_retry

PROMPT = (
    "This image comes from a page of a PDF document. Describe it so that someone who "
    "cannot see it could answer questions about it. Include the type of visual, its "
    "title, axis labels and units, legend entries, the key numbers, and the main trends "
    "or comparisons. Transcribe any visible text. Do not guess beyond what is shown. "
    "If it is only decoration (a logo, a divider, a plain photo with no information), "
    "reply with exactly: DECORATIVE"
)

DECORATIVE = "DECORATIVE"


class VisionDescriber(Protocol):
    """Anything with this method works, which lets tests use a fake."""
    def describe(self, png: bytes) -> str: ...


class OpenRouterVisionDescriber:
    """Any vision-capable model on OpenRouter (GPT-4o, Claude, ...)."""

    def __init__(self, client, model: str = "openai/gpt-4o-mini", max_tokens: int = 2048,
                 fallback_model: str | None = None):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.fallback_model = fallback_model

    def describe(self, png: bytes) -> str:
        data_url = "data:image/png;base64," + base64.standard_b64encode(png).decode()
        response = call_with_retry(
            lambda model: self.client.chat.completions.create(
                model=model,
                max_tokens=self.max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
            ),
            [m for m in (self.model, self.fallback_model) if m],
        )
        return (response.choices[0].message.content or "").strip()
