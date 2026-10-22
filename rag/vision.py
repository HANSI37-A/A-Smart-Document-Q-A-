"""Turn an image of a chart or figure into searchable text using a vision model."""

import base64
from typing import Protocol

from google.genai import types

from rag.gemini_retry import call_with_retry

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


class GeminiVisionDescriber:
    def __init__(self, client, model: str = "gemini-3.8-flash", max_tokens: int = 2048,
                 fallback_model: str | None = None):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.fallback_model = fallback_model

    def describe(self, png: bytes) -> str:
        response = call_with_retry(
            lambda model: self.client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=png, mime_type="image/png"), PROMPT],
                config=types.GenerateContentConfig(max_output_tokens=self.max_tokens),
            ),
            [m for m in (self.model, self.fallback_model) if m],
        )
        return (response.text or "").strip()


class ClaudeVisionDescriber:
    """Same job with Claude. Needs `pip install anthropic`; pass an anthropic.Anthropic."""

    def __init__(self, client, model: str = "claude-haiku-5-5", max_tokens: int = 800):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def describe(self, png: bytes) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": base64.standard_b64encode(png).decode()}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()
