
"""Build the prompt from retrieved chunks and ask Claude."""

import re

import anthropic

from rag.models import SearchResult


SYSTEM_PROMPT = """
You are a helpful document question-answering assistant.

Answer the user's question using only the provided document sources.
If the sources do not contain enough information, clearly say that you
could not find the answer in the provided documents.

Include citations for factual claims using the source number format,
such as [1] or [2].
Do not invent information or citations.
"""


def build_prompt(question: str, results: list[SearchResult]) -> str:
    blocks = []

    for n, result in enumerate(results, start=1):
        blocks.append(
            f'<source id="{n}" file="{result.chunk.source}" '
            f'page="{result.chunk.page}">\n'
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


class ClaudeGenerator:
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-haiku-5-5",
        max_tokens: int = 800,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_prompt(question, results),
                }
            ],
        )

        return "".join(
            block.text
            for block in response.content
            if block.type == "text"
        )
