"""Google Search fallback through Gemini grounding, with a fake HTTP client (no network)."""

import httpx
import pytest

from rag.generator import NOT_FOUND, is_not_found
from rag.models import Turn
from rag.web_search import GeminiWebSearcher, WebSearchError, add_citations, parse_grounded_answer


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakePost:
    """Replaces httpx.post: answers from a list, one response (or exception) per call."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        reply = self.responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def grounded(text, chunks, supports=()):
    """The shape of a generateContent response that used Google Search."""
    return {"candidates": [{
        "content": {"role": "model", "parts": [{"text": text}]},
        "groundingMetadata": {
            "groundingChunks": [{"web": {"uri": u, "title": t}} for u, t in chunks],
            "groundingSupports": list(supports),
        },
    }]}


def support(end, *indices):
    return {"segment": {"startIndex": 0, "endIndex": end}, "groundingChunkIndices": list(indices)}


def searcher(post, **kwargs):
    return GeminiWebSearcher("secret-key", post=post, sleep=lambda s: None, **kwargs)


TEXT = "Spain won Euro 2024. The final was in Berlin."


def test_search_sends_a_grounded_request_and_returns_web_sources():
    payload = grounded(TEXT, [("https://a.example/1", "uefa.com"), ("https://b.example/2", "bbc.com")],
                       [support(20, 0), support(len(TEXT), 0, 1)])
    post = FakePost(FakeResponse(payload=payload))
    history = [Turn("user", "Hi"), Turn("assistant", "Hello!")]
    answer = searcher(post, model="gemini-test").search("Who won Euro 2024?", history)

    assert answer.text == "Spain won Euro 2024.[1] The final was in Berlin.[1][2]"
    assert [(r.chunk.source, r.chunk.url, r.chunk.kind) for r in answer.sources] == [
        ("uefa.com", "https://a.example/1", "web"), ("bbc.com", "https://b.example/2", "web")]

    url, kwargs = post.calls[0]
    assert url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
    assert kwargs["headers"] == {"x-goog-api-key": "secret-key"}
    body = kwargs["json"]
    assert body["tools"] == [{"google_search": {}}]
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][-1]["parts"] == [{"text": "Who won Euro 2024?"}]


def test_citations_use_utf8_byte_offsets():
    text = "Café prices rose. Tea fell."
    end_first = len("Café prices rose.".encode())      # 18 bytes, but 17 characters
    result = add_citations(text, [support(end_first, 0), support(len(text.encode()), 1)],
                           {0: 1, 1: 2})
    assert result == "Café prices rose.[1] Tea fell.[2]"


def test_duplicate_pages_share_one_number_and_sources_are_capped():
    chunks = [("https://same/x", "same.com"), ("https://same/x", "same.com"),
              ("https://two/y", "two.com"), ("https://three/z", "three.com")]
    answer = parse_grounded_answer(grounded(TEXT, chunks, [support(len(TEXT), 1, 2, 3)]),
                                   max_sources=2)
    assert [r.chunk.url for r in answer.sources] == ["https://same/x", "https://two/y"]
    assert answer.text.endswith("[1][2]")      # chunk 3 was dropped by the cap


def test_answer_without_grounding_is_an_error():
    with pytest.raises(WebSearchError, match="no sources"):
        parse_grounded_answer(grounded(TEXT, []))


def test_empty_answer_is_an_error():
    with pytest.raises(WebSearchError, match="no answer"):
        parse_grounded_answer(grounded("", [("https://a", "a")]))
    with pytest.raises(WebSearchError, match="no answer"):
        parse_grounded_answer({"candidates": []})


def test_retries_a_busy_server_then_succeeds():
    post = FakePost(FakeResponse(503, {"error": {"message": "overloaded"}}),
                    httpx.ConnectTimeout("timed out"),
                    FakeResponse(payload=grounded(TEXT, [("https://a", "a")])))
    assert searcher(post).search("q").sources[0].chunk.url == "https://a"
    assert len(post.calls) == 3


def test_does_not_retry_a_bad_request_or_leak_the_key():
    post = FakePost(FakeResponse(400, {"error": {"message": "API key not valid"}}))
    with pytest.raises(WebSearchError, match="HTTP 400: API key not valid") as caught:
        searcher(post).search("q")
    assert len(post.calls) == 1
    assert "secret-key" not in str(caught.value)


@pytest.mark.parametrize("reply", [
    NOT_FOUND,
    "  I couldn't find that in the documents  ",
    "**I couldn’t find that in the documents.**",
    "I COULDN'T FIND THAT IN THE DOCUMENTS.",
])
def test_not_found_replies_are_recognised(reply):
    assert is_not_found(reply)


def test_real_answers_are_not_mistaken_for_not_found():
    assert not is_not_found("Employees get 14 days of annual leave [1].")
    assert not is_not_found("")
