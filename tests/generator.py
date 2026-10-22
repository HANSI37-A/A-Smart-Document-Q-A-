"""Tests for prompt building and the OpenRouter call.

The OpenRouter tests use a fake client, so no API key or internet is needed.
They still check exactly what our code would send to the API."""
from types import SimpleNamespace

import httpx
import openai
import pytest

from rag.generator import SYSTEM_PROMPT, OpenRouterGenerator, build_prompt, find_citations
from rag.models import Chunk, SearchResult
from rag.openrouter import OpenRouterBusyError, make_client


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


def completion(text):
    """The part of an OpenAI chat completion that our code reads."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeClient:
    """Mimics client.chat.completions.create and records every request."""

    def __init__(self, text="Answer [1].", fail_times=0, status=503):
        self.requests = []
        self._text, self._fail, self._status = text, fail_times, status
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) <= self._fail:
            raise status_error(self._status)
        return completion(self._text)

    @property
    def models_used(self):
        return [r["model"] for r in self.requests]


def status_error(code):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(code, request=request, json={"error": {"message": "x"}})
    return openai.APIStatusError("error", response=response, body=None)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("rag.openrouter.time.sleep", lambda s: None)


def test_generator_sends_chat_completions_payload():
    client = FakeClient("You get 14 days of annual leave [1].")
    gen = OpenRouterGenerator(client)
    answer = gen.generate("How much leave?", [result("Leave is 14 days.")])

    assert answer == "You get 14 days of annual leave [1]."
    sent = client.requests[0]
    assert sent["model"] == "openai/gpt-4o-mini"
    assert sent["max_tokens"] == gen.max_tokens
    assert sent["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert sent["messages"][1]["role"] == "user"
    assert "Leave is 14 days." in sent["messages"][1]["content"]


def test_empty_response_falls_back_to_not_found():
    answer = OpenRouterGenerator(FakeClient(None)).generate("Q?", [result("text")])
    assert answer == "I couldn't find that in the documents."


def test_make_client_points_at_openrouter_with_attribution_headers():
    client = make_client("sk-or-test", "http://localhost:8501", "Document Q&A")
    assert str(client.base_url).startswith("https://openrouter.ai/api/v1")
    assert client.api_key == "sk-or-test"
    assert client.default_headers["HTTP-Referer"] == "http://localhost:8501"
    assert client.default_headers["X-Title"] == "Document Q&A"


def test_make_client_headers_are_optional():
    client = make_client("sk-or-test")
    assert "HTTP-Referer" not in client.default_headers


# ---------------- overload handling ----------------

def test_temporary_503_is_retried():
    client = FakeClient(fail_times=2)
    assert OpenRouterGenerator(client, "main").generate("Q?", [result("t")]) == "Answer [1]."
    assert client.models_used == ["main", "main", "main"]


def test_falls_back_to_second_model_when_main_stays_busy():
    client = FakeClient(fail_times=3)
    gen = OpenRouterGenerator(client, "main", fallback_model="backup")
    assert gen.generate("Q?", [result("t")]) == "Answer [1]."
    assert client.models_used == ["main", "main", "main", "backup"]


def test_gives_a_friendly_error_when_everything_is_busy():
    gen = OpenRouterGenerator(FakeClient(fail_times=99), "main", fallback_model="backup")
    with pytest.raises(OpenRouterBusyError, match="try again"):
        gen.generate("Q?", [result("t")])


def test_bad_api_key_is_not_retried():
    client = FakeClient(fail_times=99, status=401)
    gen = OpenRouterGenerator(client, "main", fallback_model="backup")
    with pytest.raises(openai.APIStatusError):
        gen.generate("Q?", [result("t")])
    assert client.models_used == ["main"]
