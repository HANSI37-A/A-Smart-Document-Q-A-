"""Web fallback: answer from a Google search when the documents do not have the answer.

Uses Gemini's built-in "Grounding with Google Search" tool through the native
generateContent endpoint (the OpenAI-compatible endpoint used for answers does not
return the grounding sources). Like the other engines, it sits behind a small Protocol
(see rag.pipeline.WebSearcher), so the pipeline and the tests only depend on `search()`.
"""

import time
from dataclasses import dataclass

import httpx

from rag.models import Chunk, SearchResult, Turn
from rag.openrouter import RETRYABLE

GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

WEB_SYSTEM_PROMPT = """
You are a helpful assistant. The user's own documents did not contain the answer, so
answer the question using Google Search. Be concise and factual. If the search results
do not answer the question, say so plainly. Earlier messages in the conversation are only
there to understand what the user is referring to.
"""


class WebSearchError(Exception):
    """The web search failed or found nothing usable (the message is safe to show)."""


@dataclass
class WebAnswer:
    text: str                    # the answer, with [n] markers pointing into `sources`
    sources: list[SearchResult]  # kind="web" results; chunk.url is the link


class GeminiWebSearcher:
    def __init__(self, api_key: str, model: str = "gemini-3.8-flash", post=None,
                 attempts: int = 3, delay: float = 1.0, sleep=time.sleep, max_sources: int = 5):
        self._api_key = api_key
        self.model = model
        self._post = post or httpx.post     # tests pass a fake with the same signature
        self.attempts, self.delay, self._sleep = attempts, delay, sleep
        self.max_sources = max_sources

    def search(self, question: str, history: list[Turn] = ()) -> WebAnswer:
        contents = [{"role": "model" if t.role == "assistant" else "user",
                     "parts": [{"text": t.content}]} for t in history]
        contents.append({"role": "user", "parts": [{"text": question}]})
        body = {
            "systemInstruction": {"parts": [{"text": WEB_SYSTEM_PROMPT.strip()}]},
            "contents": contents,
            "tools": [{"google_search": {}}],
        }
        try:
            payload = self._send(body).json()
        except ValueError as e:
            raise WebSearchError("Google Search sent back something that is not JSON.") from e
        return parse_grounded_answer(payload, self.max_sources)

    def _send(self, body: dict):
        """POST with a few retries for busy servers; never puts the key in an error message."""
        url = GEMINI_GENERATE_URL.format(model=self.model)
        error = "unknown error"
        for attempt in range(self.attempts):
            try:
                response = self._post(url, json=body, timeout=60,
                                      headers={"x-goog-api-key": self._api_key})
            except httpx.TransportError as e:
                error = f"could not reach Gemini ({type(e).__name__})"
            else:
                if response.status_code < 300:
                    return response
                error = f"HTTP {response.status_code}: {_error_message(response)}"
                if response.status_code not in RETRYABLE:
                    break
            if attempt < self.attempts - 1:
                self._sleep(self.delay * 2 ** attempt)
        raise WebSearchError(f"Google Search failed ({error}).")


def parse_grounded_answer(payload: dict, max_sources: int = 5) -> WebAnswer:
    """Text, web sources and [n] citations from a grounded generateContent response."""
    try:
        candidate = payload["candidates"][0]
    except (KeyError, IndexError, TypeError):
        raise WebSearchError("Google Search returned no answer.") from None
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise WebSearchError("Google Search returned no answer.")

    meta = candidate.get("groundingMetadata") or {}
    # number each distinct page once; remember where every grounding chunk ended up
    sources: list[SearchResult] = []
    number_of: dict[int, int] = {}
    by_url: dict[str, int] = {}
    for i, chunk in enumerate(meta.get("groundingChunks") or []):
        web = chunk.get("web") or {}
        url = web.get("uri")
        if not url:
            continue
        if url not in by_url:
            if len(sources) >= max_sources:
                continue
            title = web.get("title") or url
            sources.append(SearchResult(
                Chunk(id=url, text=title, source=title, page=0, doc_id="web",
                      kind="web", url=url),
                score=1.0))
            by_url[url] = len(sources)
        number_of[i] = by_url[url]
    if not sources:
        raise WebSearchError("Google Search found no sources for this question.")
    return WebAnswer(add_citations(text, meta.get("groundingSupports") or [], number_of), sources)


def add_citations(text: str, supports: list[dict], number_of: dict[int, int]) -> str:
    """Insert [n] after each supported segment. Gemini's endIndex counts UTF-8 bytes."""
    data = text.encode("utf-8")
    inserts: dict[int, list[int]] = {}
    for support in supports:
        end = (support.get("segment") or {}).get("endIndex")
        numbers = sorted({number_of[i] for i in support.get("groundingChunkIndices") or []
                          if i in number_of})
        if end is None or not numbers or not 0 <= end <= len(data):
            continue
        inserts.setdefault(end, [])
        inserts[end] += [n for n in numbers if n not in inserts[end]]
    for end in sorted(inserts, reverse=True):     # back to front keeps earlier offsets valid
        marks = "".join(f"[{n}]" for n in inserts[end]).encode()
        data = data[:end] + marks + data[end:]
    return data.decode("utf-8", errors="replace")


def _error_message(response) -> str:
    try:
        message = response.json()["error"]["message"]
    except Exception:
        message = getattr(response, "text", "") or "no details"
    return str(message)[:200]
