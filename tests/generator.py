"""Tests for prompt building and the Gemini call.

The Gemini tests use a fake client, so no API key or internet is needed.
They still check exactly what our code would send to the API."""
from types import SimpleNamespace

import pytest

from rag.generator import SYSTEM_PROMPT, GeminiGenerator, build_prompt, find_citations
from rag.models import Chunk, SearchResult


def result(text, source="h.pdf", page=1, score=0.8):
    return SearchResult(Chunk(id=f"x-p{page}", text=text, source=source, page=page, doc_id="x"),
                        score)


def test_build_prompt_numbers_sources():
    prompt = build_prompt("How much leave?", [result("Leave is 14 days.", page=2),
                                              result("Sick leave is 7 days.", page=2)])
    assert '<source id="1" file="h.pdf" page="2">' in prompt
    assert '<source id="2"' in prompt
    assert "Leave is 14 days." in prompt
    assert prompt.endswith("Question: How much leave?")


def test_find_citations_ignores_invalid_numbers():
    assert find_citations("A [1]. B [3][1]. C [9].", n_sources=3) == [1, 3]
    assert find_citations("No citations here.", n_sources=3) == []


class FakeClient:
    """Mimics client.models.generate_content and records what was sent."""

    def __init__(self, text=None, error=None):
        self.sent = {}
        self._text, self._error = text, error
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.sent = kwargs
        if self._error:
            raise self._error
        return SimpleNamespace(text=self._text)


def test_gemini_generator_sends_correct_request():
    client = FakeClient("You get 14 days of annual leave [1].")
    gen = GeminiGenerator(client)
    answer = gen.generate("How much leave?", [result("Leave is 14 days.")])

    assert answer == "You get 14 days of annual leave [1]."
    assert client.sent["model"] == "gemini-2.5-flash"
    assert client.sent["config"].system_instruction == SYSTEM_PROMPT
    assert client.sent["config"].max_output_tokens == 800
    assert "Leave is 14 days." in client.sent["contents"]


def test_empty_response_returns_empty_string():
    assert GeminiGenerator(FakeClient(None)).generate("Q?", [result("text")]) == ""


def test_api_errors_propagate():
    gen = GeminiGenerator(FakeClient(error=RuntimeError("API key not valid")))
    with pytest.raises(RuntimeError):
        gen.generate("Q?", [result("text")])
