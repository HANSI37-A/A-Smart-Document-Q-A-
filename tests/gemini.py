"""Gemini as the provider that answers questions: client, provider choice and wiring."""
from types import SimpleNamespace

import openai
import pytest

from rag.config import build_llm_client, load_settings
from rag.gemini import GEMINI_BASE_URL, THINKING_HEADROOM, GeminiClient, make_gemini_client
from rag.memory import OpenRouterRewriter
from rag.models import Turn
from rag.openrouter import BASE_URL as OPENROUTER_URL


class InnerClient:
    """Stands in for openai.OpenAI: records what is sent and returns a fixed reply."""

    def __init__(self, reply="What about part-time staff leave?"):
        self.sent = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.reply = reply

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


# ---------------------------------------------------------------- the client

def test_client_gives_max_tokens_room_for_thinking():
    inner = InnerClient()
    GeminiClient("key", client=inner).chat.completions.create(
        model="m", max_tokens=256, messages=[{"role": "user", "content": "hi"}])
    assert inner.sent[0]["max_tokens"] == 256 + THINKING_HEADROOM
    assert inner.sent[0]["model"] == "m" and inner.sent[0]["messages"][0]["content"] == "hi"


def test_client_leaves_requests_without_max_tokens_alone():
    inner = InnerClient()
    GeminiClient("key", client=inner).chat.completions.create(model="m", messages=[])
    assert "max_tokens" not in inner.sent[0] and "reasoning_effort" not in inner.sent[0]


def test_reasoning_effort_is_only_sent_when_configured():
    inner = InnerClient()
    GeminiClient("key", reasoning_effort="low", client=inner).chat.completions.create(
        model="m", messages=[])
    GeminiClient("key", reasoning_effort="low", client=inner).chat.completions.create(
        model="m", messages=[], reasoning_effort="high")     # a caller's choice wins
    assert inner.sent[0]["reasoning_effort"] == "low"
    assert inner.sent[1]["reasoning_effort"] == "high"


def test_client_talks_to_gemini_without_openrouter_headers():
    real = make_gemini_client("secret-key")._client
    assert str(real.base_url) == GEMINI_BASE_URL
    assert real.api_key == "secret-key"
    assert "HTTP-Referer" not in real.default_headers and "X-Title" not in real.default_headers
    assert real.max_retries == 0          # call_with_retry does the retrying


def test_existing_components_work_unchanged_with_the_gemini_client():
    inner = InnerClient("What about part-time staff leave?")
    rewriter = OpenRouterRewriter(GeminiClient("key", client=inner), "gemini-3.8-flash")
    out = rewriter.rewrite([Turn("user", "How many leave days?"), Turn("assistant", "14 [1].")],
                           "and part-time?")
    assert out == "What about part-time staff leave?"
    assert inner.sent[0]["model"] == "gemini-3.8-flash"
    assert inner.sent[0]["max_tokens"] == 256 + THINKING_HEADROOM


# ---------------------------------------------------------------- settings

@pytest.fixture
def llm_env(tmp_path, monkeypatch):
    """Settings that ignore the real .env file and any keys in the environment."""
    monkeypatch.setattr("rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for name in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "LLM_PROVIDER", "GEMINI_MODEL",
                 "GEMINI_FALLBACK_MODEL", "GEMINI_REASONING", "GEMINI_STT_MODEL",
                 "GEMINI_TTS_MODEL", "GEMINI_VOICE", "OPENROUTER_MODEL",
                 "OPENROUTER_FALLBACK_MODEL", "VOICE_STT", "VOICE_TTS", "VISION_PROVIDER",
                 "VISION_MODEL", "HELPER_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL",
                 "OPENAI_FALLBACK_MODEL", "OPENAI_REASONING"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_without_any_key_nothing_changes(llm_env):
    s = load_settings()
    assert (s.llm_provider, s.api_key, s.model) == ("openrouter", None, "openai/gpt-4o-mini")
    assert (s.voice_stt, s.voice_tts) == ("whisper", "piper")    # local voice, as before
    assert build_llm_client(s) is None


def test_openrouter_key_alone_behaves_as_before(llm_env):
    llm_env.setenv("OPENROUTER_API_KEY", "or-key")
    llm_env.setenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    s = load_settings()
    assert (s.llm_provider, s.api_key, s.model) == ("openrouter", "or-key", "google/gemini-2.5-flash")
    client = build_llm_client(s)
    assert isinstance(client, openai.OpenAI)
    assert str(client.base_url).rstrip("/") == OPENROUTER_URL    # the SDK adds a trailing slash


def test_a_gemini_key_selects_gemini_for_answers_and_voice(llm_env):
    llm_env.setenv("GEMINI_API_KEY", "g-key")
    s = load_settings()
    assert (s.llm_provider, s.api_key, s.model) == ("gemini", "g-key", "gemini-3.8-flash")
    assert (s.voice_stt, s.voice_tts) == ("gemini", "gemini")
    assert (s.gemini_stt_model, s.gemini_tts_model, s.gemini_voice) == \
        ("gemini-3.8-flash", "gemini-3.8-flash-tts", "Kore")
    client = build_llm_client(s)
    assert isinstance(client, GeminiClient) and str(client._client.base_url) == GEMINI_BASE_URL


def test_gemini_wins_when_both_keys_are_set_and_ignores_openrouter_models(llm_env):
    llm_env.setenv("GEMINI_API_KEY", "g-key")
    llm_env.setenv("OPENROUTER_API_KEY", "or-key")
    llm_env.setenv("OPENROUTER_MODEL", "openai/gpt-4o")          # not a Gemini model id
    llm_env.setenv("OPENROUTER_FALLBACK_MODEL", "openai/gpt-4o-mini")
    s = load_settings()
    assert s.llm_provider == "gemini" and s.model == "gemini-3.8-flash"
    assert s.fallback_model is None and s.helper_model == s.vision_model == s.model


def test_provider_can_be_forced_back_to_openrouter(llm_env):
    llm_env.setenv("GEMINI_API_KEY", "g-key")
    llm_env.setenv("OPENROUTER_API_KEY", "or-key")
    llm_env.setenv("LLM_PROVIDER", "OpenRouter")
    s = load_settings()
    assert (s.llm_provider, s.api_key, s.model) == ("openrouter", "or-key", "openai/gpt-4o-mini")
    assert (s.voice_stt, s.voice_tts) == ("gemini", "gemini")     # voice is chosen separately


def test_chosen_provider_without_its_key_means_search_only(llm_env):
    llm_env.setenv("OPENROUTER_API_KEY", "or-key")
    llm_env.setenv("LLM_PROVIDER", "gemini")                       # but no Gemini key
    s = load_settings()
    assert s.api_key is None and build_llm_client(s) is None


def test_unknown_provider_is_a_clear_error(llm_env):
    llm_env.setenv("LLM_PROVIDER", "gpt")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        load_settings()


def test_gemini_options_come_from_the_environment(llm_env):
    llm_env.setenv("GEMINI_API_KEY", "g-key")
    llm_env.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    llm_env.setenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    llm_env.setenv("GEMINI_REASONING", "low")
    llm_env.setenv("GEMINI_STT_MODEL", "gemini-3.5-flash-lite")
    llm_env.setenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")
    llm_env.setenv("GEMINI_VOICE", "Puck")
    llm_env.setenv("VOICE_TTS", "piper")                           # explicit choice beats default
    s = load_settings()
    assert (s.model, s.fallback_model, s.gemini_reasoning) == \
        ("gemini-3.6-flash", "gemini-3.5-flash-lite", "low")
    assert (s.gemini_stt_model, s.gemini_tts_model, s.gemini_voice) == \
        ("gemini-3.5-flash-lite", "gemini-3.8-flash-lite-tts", "Puck")
    assert (s.voice_stt, s.voice_tts) == ("gemini", "piper")
    assert build_llm_client(s).reasoning_effort == "low"


def test_vision_stays_on_unless_switched_off(llm_env):
    assert load_settings().vision == "auto"
    llm_env.setenv("VISION_PROVIDER", "none")
    assert load_settings().vision == "none"
