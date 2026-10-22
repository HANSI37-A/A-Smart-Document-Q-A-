"""Tests for prompt building and the Claude call.

The Claude test uses the real Anthropic SDK, but with a fake HTTP transport,
so no API key or internet is needed and no money is spent. It still checks
exactly what our code would send to the API."""
import json

import anthropic
import pytest

# The guide says the SDK version it targets uses httpx2. Older SDKs use httpx.
# Whichever your installed SDK uses, import it under the name `httpx2` here.
try:
    import httpx2
except ImportError:  # pragma: no cover
    import httpx as httpx2

from rag.generator import SYSTEM_PROMPT, ClaudeGenerator, build_prompt, find_citations
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


def mock_client(handler):
    return anthropic.Anthropic(api_key="test-key", max_retries=0,
                               http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


def test_claude_generator_sends_correct_request():
    sent = {}

    def handler(request):
        sent["path"] = request.url.path
        sent["body"] = json.loads(request.content)
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": "You get 14 days of annual leave [1]."}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 120, "output_tokens": 12},
        })

    gen = ClaudeGenerator(mock_client(handler))
    answer = gen.generate("How much leave?", [result("Leave is 14 days.")])

    assert answer == "You get 14 days of annual leave [1]."
    assert sent["path"] == "/v1/messages"
    assert sent["body"]["model"] == "claude-haiku-5-5"
    assert sent["body"]["system"] == SYSTEM_PROMPT
    assert "Leave is 14 days." in sent["body"]["messages"][0]["content"]


def test_invalid_api_key_raises_clear_error():
    def handler(request):
        return httpx2.Response(401, json={"type": "error", "error": {
            "type": "authentication_error", "message": "invalid x-api-key"}})

    gen = ClaudeGenerator(mock_client(handler))
    with pytest.raises(anthropic.AuthenticationError):
        gen.generate("Q?", [result("text")])