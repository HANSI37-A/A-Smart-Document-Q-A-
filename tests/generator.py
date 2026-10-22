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
    assert client.sent["model"] == gen.model
    assert client.sent["config"].system_instruction == SYSTEM_PROMPT
    assert client.sent["config"].max_output_tokens == gen.max_tokens
    assert "Leave is 14 days." in client.sent["contents"]


def test_empty_response_falls_back_to_not_found():
    answer = GeminiGenerator(FakeClient(None)).generate("Q?", [result("text")])
    assert answer == "I couldn't find that in the documents."


def test_api_errors_propagate():
    gen = GeminiGenerator(FakeClient(error=RuntimeError("API key not valid")))
    with pytest.raises(RuntimeError):
        gen.generate("Q?", [result("text")])


# ---------------- overload handling ----------------

def api_error(code):
    from google.genai import errors
    return errors.APIError(code, {"error": {"code": code, "message": "overloaded"}})


class FlakyClient:
    """Fails with `code` for the first `fail_times` calls, then answers."""

    def __init__(self, fail_times, code=503):
        self.calls = []
        self._fail, self._code = fail_times, code
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs["model"])
        if len(self.calls) <= self._fail:
            raise api_error(self._code)
        return SimpleNamespace(text="Answer [1].")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("rag.gemini_retry.time.sleep", lambda s: None)


def test_temporary_503_is_retried():
    client = FlakyClient(fail_times=2)
    assert GeminiGenerator(client, "main").generate("Q?", [result("t")]) == "Answer [1]."
    assert client.calls == ["main", "main", "main"]


def test_falls_back_to_second_model_when_main_stays_busy():
    client = FlakyClient(fail_times=3)
    gen = GeminiGenerator(client, "main", fallback_model="backup")
    assert gen.generate("Q?", [result("t")]) == "Answer [1]."
    assert client.calls == ["main", "main", "main", "backup"]


def test_gives_a_friendly_error_when_everything_is_busy():
    from rag.gemini_retry import GeminiBusyError
    gen = GeminiGenerator(FlakyClient(fail_times=99), "main", fallback_model="backup")
    with pytest.raises(GeminiBusyError, match="try again"):
        gen.generate("Q?", [result("t")])


def test_bad_api_key_is_not_retried():
    from google.genai import errors
    client = FlakyClient(fail_times=99, code=400)
    with pytest.raises(errors.APIError):
        GeminiGenerator(client, "main", fallback_model="backup").generate("Q?", [result("t")])
    assert client.calls == ["main"]
