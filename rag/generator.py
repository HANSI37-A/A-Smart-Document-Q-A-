"""Build the prompt from retrieved chunks and ask the model (through OpenRouter)."""

import re
from pathlib import Path

import openai
from PIL import Image

from rag.models import SearchResult, Turn
from rag.openrouter import call_with_retry
from rag.vision import image_to_data_url


SYSTEM_PROMPT = """
You are a helpful document question-answering assistant.

Answer the user's question concisely, using only the provided document sources.
If the sources do not contain the answer, reply exactly:
I couldn't find that in the documents.

Sources may be plain text, Markdown tables (type="table") or written descriptions
of charts and images (type="figure"); use all of them.

Some figures may also be attached as images. Each image belongs to the numbered
source named in <images>. When an attached image helps to answer, explain what it
shows (for example the trend, the key values or the parts of a diagram) and cite
its source number.

Earlier messages in the conversation are only there to understand what the user is
referring to. Facts in your answer must come from the sources.

Cite every factual claim with its source number, such as [1] or [2].
Do not invent information or citations.
"""


def build_prompt(question: str, results: list[SearchResult],
                 images: list[SearchResult] = ()) -> str:
    blocks = []

    for n, result in enumerate(results, start=1):
        kind = "" if result.chunk.kind == "text" else f' type="{result.chunk.kind}"'
        blocks.append(
            f'<source id="{n}" file="{result.chunk.source}" '
            f'page="{result.chunk.page}"{kind}>\n'
            f"{result.chunk.text}\n"
            f"</source>"
        )

    sources = "\n".join(blocks) if blocks else "(no sources found)"

    attached = ""
    if images:
        number = {r.chunk.id: n for n, r in enumerate(results, start=1)}
        lines = [f"Image {i}: the figure of source [{number[img.chunk.id]}] "
                 f"({img.chunk.source}, page {img.chunk.page})"
                 for i, img in enumerate(images, start=1) if img.chunk.id in number]
        attached = "<images>\n" + "\n".join(lines) + "\n</images>\n\n"

    return (
        f"<sources>\n{sources}\n</sources>\n\n"
        f"{attached}"
        f"Question: {question}"
    )


def find_citations(answer: str, n_sources: int) -> list[int]:
    numbers = {
        int(n)
        for n in re.findall(r"\[(\d+)\]", answer)
    }

    return sorted(n for n in numbers if 1 <= n <= n_sources)


def _load_images(images: list[SearchResult]) -> list[tuple[SearchResult, str]]:
    """(figure, data URL) for every figure whose saved picture can still be opened."""
    loaded = []
    for img in images:
        path = Path(img.chunk.image_path) if img.chunk.image_path else None
        if path is None or not path.is_file():
            continue
        try:
            with Image.open(path) as picture:
                loaded.append((img, image_to_data_url(picture)))
        except OSError:
            continue
    return loaded


class OpenRouterGenerator:
    def __init__(
        self,
        client: openai.OpenAI,
        model: str = "openai/gpt-4o-mini",
        max_tokens: int = 2048,
        fallback_model: str | None = None,
        vision_model: str | None = None,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.fallback_model = fallback_model
        self.vision_model = vision_model or model   # used when images are attached

    def generate(
        self,
        question: str,
        results: list[SearchResult],
        history: list[Turn] = (),
        images: list[SearchResult] = (),
    ) -> str:
        ids = {r.chunk.id for r in results}
        loaded = [(img, url) for img, url in _load_images(list(images)) if img.chunk.id in ids]
        prompt = build_prompt(question, results, [img for img, _ in loaded])

        if loaded:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": url}} for _, url in loaded
            ]
            main = self.vision_model
        else:
            content = prompt      # unchanged plain-text request when there are no images
            main = self.model

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": t.role, "content": t.content} for t in history]
        messages.append({"role": "user", "content": content})

        response = call_with_retry(
            lambda model: self.client.chat.completions.create(
                model=model,
                max_tokens=self.max_tokens,
                messages=messages,
            ),
            [m for m in (main, self.fallback_model) if m],
        )
        # content is None when the reply was blocked or cut off before any answer was written
        return response.choices[0].message.content or "I couldn't find that in the documents."
