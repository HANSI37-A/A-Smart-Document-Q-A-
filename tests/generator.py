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


# ---------------- images and conversation history ----------------

def figure_result(tmp_path, n=1):
    from PIL import Image
    path = tmp_path / f"fig{n}.png"
    Image.new("RGB", (60, 40), "green").save(path)
    chunk = Chunk(id=f"x-p2-f{n}", text="Figure on page 2: Bar chart of sales.", source="h.pdf",
                  page=2, doc_id="x", kind="figure", image_path=str(path))
    return SearchResult(chunk, 0.8)


def test_images_are_sent_as_image_parts_and_use_the_vision_model(tmp_path):
    fig = figure_result(tmp_path)
    client = FakeClient("Sales rise [2].")
    gen = OpenRouterGenerator(client, "text-model", vision_model="vision-model")
    gen.generate("What does the chart show?", [result("Sales text."), fig], images=[fig])

    sent = client.requests[0]
    assert sent["model"] == "vision-model"
    content = sent["messages"][-1]["content"]
    assert content[0]["type"] == "text"
    assert "<images>" in content[0]["text"] and "source [2]" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_without_images_the_request_is_plain_text_on_the_main_model():
    client = FakeClient()
    OpenRouterGenerator(client, "text-model", vision_model="vision-model").generate(
        "Q?", [result("t")])
    sent = client.requests[0]
    assert sent["model"] == "text-model"
    assert isinstance(sent["messages"][-1]["content"], str)
    assert "<images>" not in sent["messages"][-1]["content"]


def test_an_image_whose_file_is_gone_is_left_out(tmp_path):
    fig = figure_result(tmp_path)
    (tmp_path / "fig1.png").unlink()
    client = FakeClient()
    OpenRouterGenerator(client, "text-model", vision_model="vision-model").generate(
        "Q?", [fig], images=[fig])
    assert client.requests[0]["model"] == "text-model"
    assert isinstance(client.requests[0]["messages"][-1]["content"], str)


def test_an_image_that_is_not_among_the_sources_is_not_sent(tmp_path):
    fig = figure_result(tmp_path)
    client = FakeClient()
    OpenRouterGenerator(client).generate("Q?", [result("t")], images=[fig])
    assert isinstance(client.requests[0]["messages"][-1]["content"], str)


def test_history_comes_between_the_system_prompt_and_the_question():
    from rag.models import Turn
    client = FakeClient()
    history = [Turn("user", "How much leave?"), Turn("assistant", "14 days [1].")]
    OpenRouterGenerator(client).generate("And sick leave?", [result("t")], history=history)
    roles = [m["role"] for m in client.requests[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert client.requests[0]["messages"][1]["content"] == "How much leave?"
    assert "Question: And sick leave?" in client.requests[0]["messages"][-1]["content"]


def test_build_prompt_without_images_is_unchanged():
    assert "<images>" not in build_prompt("Q?", [result("t")])
