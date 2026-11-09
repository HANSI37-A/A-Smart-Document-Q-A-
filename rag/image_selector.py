"""Decide which figures (if any) are worth showing for a question.

Not every question needs an image. The selector works in two stages:

1. Similarity (free): every candidate figure has an embedded caption. Figures whose
   caption is not close to the question are dropped.
2. Judge (one small text-only LLM call): the remaining captions are shown to a model
   that picks only the figures that really help to answer. "None" is a valid answer.

Candidates are figure chunks that were retrieved directly, plus figures on the same
pages as the retrieved text (a paragraph often refers to the chart next to it)."""

import json
import math
import re
from pathlib import Path
from typing import Protocol

from rag.models import SearchResult
from rag.openrouter import call_with_retry
from rag.vector_store import VectorStore

MAX_CANDIDATES = 4        # captions shown to the judge
CONFIDENT_SCORE = 0.6     # without a judge, only a figure this similar is shown


class ImageJudge(Protocol):
    """Return the positions (0-based) of the candidates worth showing."""
    def pick(self, question: str, candidates: list[SearchResult]) -> list[int]: ...


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class ImageSelector:
    def __init__(self, store: VectorStore, judge: ImageJudge | None = None,
                 min_score: float = 0.45, max_images: int = 2):
        self.store = store
        self.judge = judge
        self.min_score = min_score
        self.max_images = max_images

    def candidates(self, query_vec: list[float],
                   results: list[SearchResult]) -> list[SearchResult]:
        """Figures related to the retrieved chunks, scored against the question, best first."""
        direct = [r.chunk for r in results if r.chunk.kind == "figure"]
        vectors = self.store.embeddings_for([c.id for c in direct])
        pool = {c.id: (c, vectors[c.id]) for c in direct if c.id in vectors}
        pages = sorted({(r.chunk.doc_id, r.chunk.page) for r in results})
        for chunk, vec in self.store.figures_on_pages(pages):
            pool.setdefault(chunk.id, (chunk, vec))

        scored = [SearchResult(chunk, round(cosine(query_vec, vec), 4))
                  for chunk, vec in pool.values()
                  if chunk.image_path and Path(chunk.image_path).is_file()]
        return sorted(scored, key=lambda r: r.score, reverse=True)

    def select(self, question: str, query_vec: list[float], results: list[SearchResult],
               use_judge: bool = True) -> list[SearchResult]:
        likely = [c for c in self.candidates(query_vec, results)
                  if c.score >= self.min_score][:MAX_CANDIDATES]
        if not likely:
            return []                                   # no image, and no LLM call
        if use_judge and self.judge is not None:
            try:
                picked = self.judge.pick(question, likely)
                chosen = [likely[i] for i in dict.fromkeys(picked) if 0 <= i < len(likely)]
                return chosen[:self.max_images]
            except Exception:
                pass                                    # judge failed: be conservative
        return [c for c in likely if c.score >= CONFIDENT_SCORE][:1]


JUDGE_PROMPT = """
You choose which document figures should be shown to help answer a question.

You get the question and a numbered list of figure descriptions (F1, F2, ...).
Pick a figure only if looking at it directly helps to answer this question:
it shows the data, diagram or picture the question is about.
Do not pick figures that are only loosely related, decorative, or logos.
Most questions about policies, definitions or plain facts need no figure.

Reply with JSON only, for example {"use": ["F2"]} or {"use": []}.
"""


def parse_judge_reply(reply: str, n: int) -> list[int]:
    """Read {"use": ["F2", ...]} leniently; anything unreadable counts as no figure."""
    labels = []
    match = re.search(r"\{.*\}", reply or "", re.S)
    if match:
        try:
            labels = [str(x) for x in json.loads(match.group(0)).get("use", [])]
        except (ValueError, AttributeError):
            labels = []
    else:
        labels = re.findall(r"F\d+", reply or "")
    picked = []
    for label in labels:
        digits = re.sub(r"\D", "", label)
        if digits and 1 <= int(digits) <= n:
            picked.append(int(digits) - 1)
    return list(dict.fromkeys(picked))


class OpenRouterImageJudge:
    def __init__(self, client, model: str = "openai/gpt-4o-mini",
                 fallback_model: str | None = None, max_tokens: int = 512):
        self.client = client
        self.model = model
        self.fallback_model = fallback_model
        self.max_tokens = max_tokens

    def pick(self, question: str, candidates: list[SearchResult]) -> list[int]:
        listing = "\n".join(
            f"F{i}. ({c.chunk.source}, page {c.chunk.page}) {c.chunk.text}"
            for i, c in enumerate(candidates, start=1)
        )
        response = call_with_retry(
            lambda model: self.client.chat.completions.create(
                model=model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": f"Question: {question}\n\nFigures:\n{listing}"},
                ],
            ),
            [m for m in (self.model, self.fallback_model) if m],
        )
        return parse_judge_reply(response.choices[0].message.content or "", len(candidates))
