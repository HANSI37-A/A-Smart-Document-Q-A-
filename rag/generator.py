
"""Build the prompt from retrieved chunks and ask Gemini."""

import re

from google import genai
from google.genai import types

from rag.gemini_retry import call_with_retry
from rag.models import SearchResult


SYSTEM_PROMPT = """
You are a helpful document question-answering assistant.

Answer the user's question concisely, using only the provided document sources.
If the sources do not contain the answer, reply exactly:
I couldn't find that in the documents.

Sources may be plain text, Markdown tables (type="table") or written descriptions
of charts and images (type="figure"); use all of them.

Cite every factual claim with its source number, such as [1] or [2].
Do not invent information or citations.
"""


def build_prompt(question: str, results: list[SearchResult]) -> str:
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

    return (
        f"<sources>\n{sources}\n</sources>\n\n"
        f"Question: {question}"
    )


def find_citations(answer: str, n_sources: int) -> list[int]:
    numbers = {
        int(n)
        for n in re.findall(r"\[(\d+)\]", answer)
    }

    return sorted(n for n in numbers if 1 <= n <= n_sources)


class GeminiGenerator:
    def __init__(
        self,
        client: genai.Client,
        model: str = "gemini-3.8-flash",
        max_tokens: int = 2048,  # Gemini's hidden "thinking" tokens count toward this limit
        fallback_model: str | None = None,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.fallback_model = fallback_model

    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        prompt = build_prompt(question, results)
        response = call_with_retry(
            lambda model: self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=self.max_tokens,
                ),
            ),
            [m for m in (self.model, self.fallback_model) if m],
        )

        # text is None when the reply was blocked or cut off before any answer was written
        return response.text or "I couldn't find that in the documents."
