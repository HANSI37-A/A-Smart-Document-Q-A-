"""Turn an image of a chart or figure into searchable text using a vision model."""

import base64
import io
from typing import Protocol

from PIL import Image

from rag.openrouter import call_with_retry

PROMPT = (
    "This image was cut out of a page of a PDF document. Describe it so that someone who "
    "cannot see it could answer questions about it. Include the type of visual, its "
    "title, axis labels and units, legend entries, the key numbers, and the main trends "
    "or comparisons. Transcribe any visible text. Do not guess beyond what is shown. "
    "If it carries no information (a logo, an icon, a divider, a background pattern, "
    "a decorative or stock photo), reply with exactly: DECORATIVE"
)

DECORATIVE = "DECORATIVE"
MAX_CONTEXT_CHARS = 1500


class VisionDescriber(Protocol):
    """Anything with this method works, which lets tests use a fake."""
    def describe(self, image: Image.Image, context: str = "") -> str: ...


def image_to_data_url(image: Image.Image, max_side: int = 1024) -> str:
    """PNG data URL, shrunk so the longest side is at most max_side (fewer image tokens)."""
    image = image.convert("RGB")
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.standard_b64encode(buf.getvalue()).decode()


class OpenRouterVisionDescriber:
    """Any vision-capable model on OpenRouter (GPT-4o, Claude, ...)."""

    def __init__(self, client, model: str = "openai/gpt-4o-mini", max_tokens: int = 2048,
                 fallback_model: str | None = None):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.fallback_model = fallback_model

    def describe(self, image: Image.Image, context: str = "") -> str:
        text = PROMPT
        if context.strip():
            # The image is cropped, so the page title and caption are no longer in it.
            text += ("\n\nText from the same page, for context (titles, captions):\n"
                     + context.strip()[:MAX_CONTEXT_CHARS])
        data_url = image_to_data_url(image, max_side=1536)
        response = call_with_retry(
            lambda model: self.client.chat.completions.create(
                model=model,
                max_tokens=self.max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": text},
                    ],
                }],
            ),
            [m for m in (self.model, self.fallback_model) if m],
        )
        return (response.choices[0].message.content or "").strip()
