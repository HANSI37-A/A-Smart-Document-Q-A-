"""OpenAI as the provider that answers questions, hears and speaks: client, settings, voice."""
from types import SimpleNamespace

import httpx
import openai
import pytest

from rag.config import build_llm_client, build_voice, load_settings
from rag.memory import OpenRouterRewriter
from rag.models import Turn
from rag.openai_client import THINKING_HEADROOM, OpenAIClient, make_openai_client
from rag.voice import OpenAISpeaker, OpenAITranscriber, VoiceError


class InnerClient:
    """Stands in for openai.OpenAI: records what is sent and returns fixed replies."""

    def __init__(self, reply="What about part-time staff leave?", transcript="hello there",
                 audio_bytes=b"RIFFfakewav"):
        self.sent, self.transcribed, self.spoken = [], [], []
        self.reply, self.transcript, self.audio_bytes = reply, transcript, audio_bytes
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.audio = SimpleNamespace(                    # client.audio.transcriptions / .speech
            transcriptions=SimpleNamespace(create=self._transcribe),
            speech=SimpleNamespace(create=self._speak))

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])

    def _transcribe(self, **kwargs):
        self.transcribed.append(kwargs)
        return SimpleNamespace(text=self.transcript)

    def _speak(self, **kwargs):
        self.spoken.append(kwargs)
        return SimpleNamespace(content=self.audio_bytes)


def make_inner(**kw):
    return InnerClient(**kw)


def auth_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    return openai.AuthenticationError("bad key", response=httpx.Response(401, request=request),
                                      body=None)


# ---------------------------------------------------------------- the client

def test_client_renames_max_tokens_for_every_model():
    inner = make_inner()
    OpenAIClient("key", client=inner).chat.completions.create(
        model="gpt-4o-mini", max_tokens=256, messages=[{"role": "user", "content": "hi"}])
    sent = inner.sent[0]
    assert sent["max_completion_tokens"] == 256 and "max_tokens" not in sent
    assert sent["model"] == "gpt-4o-mini" and sent["messages"][0]["content"] == "hi"


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "o3", "o4-mini"])
def test_reasoning_models_get_room_for_thinking(model):
    inner = make_inner()
    OpenAIClient("key", client=inner).chat.completions.create(
        model=model, max_tokens=256, messages=[])
    assert inner.sent[0]["max_completion_tokens"] == 256 + THINKING_HEADROOM


def test_reasoning_effort_is_only_sent_to_reasoning_models_when_configured():
    inner = make_inner()
    client = OpenAIClient("key", reasoning_effort="low", client=inner)
    client.chat.completions.create(model="gpt-5", messages=[])
    client.chat.completions.create(model="gpt-4o-mini", messages=[])
    client.chat.completions.create(model="gpt-5", messages=[], reasoning_effort="high")
    assert inner.sent[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in inner.sent[1]
    assert inner.sent[2]["reasoning_effort"] == "high"       # a caller's choice wins


def test_requests_without_max_tokens_are_left_alone():
    inner = make_inner()
    OpenAIClient("key", client=inner).chat.completions.create(model="gpt-4o-mini", messages=[])
    assert "max_tokens" not in inner.sent[0] and "max_completion_tokens" not in inner.sent[0]


def test_client_talks_to_openai_directly():
    real = make_openai_client("secret-key")._client
    assert str(real.base_url).rstrip("/") == "https://api.openai.com/v1"
    assert real.api_key == "secret-key"
    assert "HTTP-Referer" not in real.default_headers and "X-Title" not in real.default_headers
    assert real.max_retries == 0          # call_with_retry does the retrying


def test_existing_components_work_unchanged_with_the_openai_client():
    inner = make_inner(reply="What about part-time staff leave?")
    rewriter = OpenRouterRewriter(OpenAIClient("key", client=inner), "gpt-4o-mini")
    out = rewriter.rewrite([Turn("user", "How many leave days?"), Turn("assistant", "14 [1].")],
                           "and part-time?")
    assert out == "What about part-time staff leave?"
    assert inner.sent[0]["model"] == "gpt-4o-mini" and inner.sent[0]["max_completion_tokens"] == 256


# ---------------------------------------------------------------- settings

@pytest.fixture
def openai_env(tmp_path, monkeypatch):
    """Settings that ignore the real .env file and any keys in the environment."""
    monkeypatch.setattr("rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_FALLBACK_MODEL", "OPENAI_REASONING",
                 "OPENAI_STT_MODEL", "OPENAI_STT_FALLBACK_MODEL", "OPENAI_TTS_MODEL",
                 "OPENAI_VOICE", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "LLM_PROVIDER",
                 "GEMINI_MODEL", "OPENROUTER_MODEL", "OPENROUTER_FALLBACK_MODEL", "VOICE_STT",
                 "VOICE_TTS", "VISION_MODEL", "HELPER_MODEL", "VOICE_LANGUAGE"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_an_openai_key_selects_openai_for_answers_and_voice(openai_env):
    openai_env.setenv("OPENAI_API_KEY", "sk-test")
    s = load_settings()
    assert (s.llm_provider, s.api_key, s.model) == ("openai", "sk-test", "gpt-4o-mini")
    assert (s.voice_stt, s.voice_tts) == ("openai", "openai")
    assert (s.openai_stt_model, s.openai_tts_model, s.openai_voice) == \
        ("gpt-4o-mini-transcribe", "gpt-4o-mini-tts", "alloy")
    assert s.helper_model == s.vision_model == "gpt-4o-mini"
    client = build_llm_client(s)
    assert isinstance(client, OpenAIClient)


def test_openai_wins_over_gemini_and_openrouter_but_can_be_overridden(openai_env):
    openai_env.setenv("OPENAI_API_KEY", "sk-test")
    openai_env.setenv("GEMINI_API_KEY", "g-key")
    openai_env.setenv("OPENROUTER_API_KEY", "or-key")
    openai_env.setenv("GEMINI_MODEL", "gemini-3.8-flash")        # not an OpenAI model id
    s = load_settings()
    assert (s.llm_provider, s.model) == ("openai", "gpt-4o-mini")
    openai_env.setenv("LLM_PROVIDER", "Gemini")
    s = load_settings()
    assert (s.llm_provider, s.api_key, s.model) == ("gemini", "g-key", "gemini-3.8-flash")
    assert (s.voice_stt, s.voice_tts) == ("openai", "openai")    # voice is chosen separately


def test_without_an_openai_key_gemini_and_openrouter_behave_as_before(openai_env):
    openai_env.setenv("GEMINI_API_KEY", "g-key")
    assert load_settings().llm_provider == "gemini"
    openai_env.delenv("GEMINI_API_KEY")
    openai_env.setenv("OPENROUTER_API_KEY", "or-key")
    s = load_settings()
    assert (s.llm_provider, s.model, s.voice_stt) == ("openrouter", "openai/gpt-4o-mini", "whisper")


def test_openai_chosen_without_its_key_means_search_only(openai_env):
    openai_env.setenv("OPENROUTER_API_KEY", "or-key")
    openai_env.setenv("LLM_PROVIDER", "openai")
    s = load_settings()
    assert s.api_key is None and build_llm_client(s) is None


def test_openai_options_come_from_the_environment(openai_env):
    openai_env.setenv("OPENAI_API_KEY", "sk-test")
    openai_env.setenv("OPENAI_MODEL", "gpt-5-mini")
    openai_env.setenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
    openai_env.setenv("OPENAI_REASONING", "low")
    openai_env.setenv("OPENAI_STT_MODEL", "gpt-4o-transcribe")
    openai_env.setenv("OPENAI_TTS_MODEL", "tts-1")
    openai_env.setenv("OPENAI_VOICE", "nova")
    openai_env.setenv("VOICE_TTS", "piper")                      # an explicit choice beats the default
    s = load_settings()
    assert (s.model, s.fallback_model, s.openai_reasoning) == ("gpt-5-mini", "gpt-4o-mini", "low")
    assert (s.openai_stt_model, s.openai_tts_model, s.openai_voice) == \
        ("gpt-4o-transcribe", "tts-1", "nova")
    assert (s.voice_stt, s.voice_tts) == ("openai", "piper")
    assert build_llm_client(s).reasoning_effort == "low"


def test_unknown_provider_error_names_all_choices(openai_env):
    openai_env.setenv("LLM_PROVIDER", "gpt")
    with pytest.raises(ValueError, match="openai"):
        load_settings()


# ---------------------------------------------------------------- out of credit

def quota_error(code="credit_balance_exhausted"):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"message": "You have no credits remaining.", "type": "insufficient_quota", "code": code}
    return openai.RateLimitError("quota", response=httpx.Response(429, request=request), body=body)


def rate_limit_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"message": "Slow down.", "type": "requests", "code": "rate_limit_exceeded"}
    return openai.RateLimitError("slow", response=httpx.Response(429, request=request), body=body)


def test_no_credit_is_reported_clearly_and_not_retried():
    from rag.openrouter import OpenRouterBusyError, call_with_retry
    calls, waits = [], []

    def call(model):
        calls.append(model)
        raise quota_error()

    with pytest.raises(OpenRouterBusyError, match="no credit left"):
        call_with_retry(call, ["gpt-4o-mini", "gpt-4.1-mini"], sleep=waits.append)
    assert calls == ["gpt-4o-mini"] and waits == []       # no retries, no fallback model


def test_a_plain_rate_limit_is_still_retried():
    from rag.openrouter import call_with_retry
    calls = []

    def call(model):
        calls.append(model)
        if len(calls) < 3:
            raise rate_limit_error()
        return "ok"

    assert call_with_retry(call, ["m"], sleep=lambda s: None) == "ok" and len(calls) == 3


# ---------------------------------------------------------------- hearing

def test_transcriber_sends_the_recording_and_returns_the_text():
    inner = make_inner(transcript="  What was the revenue?  ")
    ears = OpenAITranscriber(OpenAIClient("key", client=inner), "gpt-4o-mini-transcribe",
                             language="en")
    assert ears.transcribe(b"RIFFrecording") == "What was the revenue?"
    sent = inner.transcribed[0]
    assert sent["model"] == "gpt-4o-mini-transcribe" and sent["language"] == "en"
    assert sent["file"] == ("question.wav", b"RIFFrecording")
    assert ears.label == "OpenAI gpt-4o-mini-transcribe"


def test_transcriber_leaves_language_to_the_model_by_default_and_skips_empty_audio():
    inner = make_inner()
    ears = OpenAITranscriber(OpenAIClient("key", client=inner))
    ears.transcribe(b"RIFFx")
    assert "language" not in inner.transcribed[0]
    assert ears.transcribe(b"") == "" and len(inner.transcribed) == 1


def test_transcriber_turns_service_errors_into_voice_errors():
    inner = make_inner()

    def refuse(**kwargs):
        raise auth_error()

    inner.audio.transcriptions.create = refuse
    with pytest.raises(VoiceError, match="Could not transcribe"):
        OpenAITranscriber(OpenAIClient("key", client=inner)).transcribe(b"RIFFx")


# ---------------------------------------------------------------- speaking

def test_speaker_asks_for_wav_and_returns_the_audio():
    inner = make_inner(audio_bytes=b"RIFF....WAVE")
    mouth = OpenAISpeaker(OpenAIClient("key", client=inner), "gpt-4o-mini-tts", "nova")
    assert mouth.synthesize("Revenue was 12 million.") == b"RIFF....WAVE"
    sent = inner.spoken[0]
    assert (sent["model"], sent["voice"], sent["response_format"]) == \
        ("gpt-4o-mini-tts", "nova", "wav")
    assert sent["input"] == "Revenue was 12 million." and mouth.label == "OpenAI nova"


def test_speaker_skips_empty_text_and_rejects_empty_audio():
    inner = make_inner(audio_bytes=b"")
    mouth = OpenAISpeaker(OpenAIClient("key", client=inner))
    assert mouth.synthesize("   ") == b"" and inner.spoken == []
    with pytest.raises(VoiceError, match="no audio"):
        mouth.synthesize("Hello")


def test_speaker_turns_service_errors_into_voice_errors():
    inner = make_inner()

    def refuse(**kwargs):
        raise auth_error()

    inner.audio.speech.create = refuse
    with pytest.raises(VoiceError, match="could not speak"):
        OpenAISpeaker(OpenAIClient("key", client=inner)).synthesize("Hello")


# ---------------------------------------------------------------- build_voice

def test_build_voice_uses_openai_when_selected(openai_env):
    openai_env.setenv("OPENAI_API_KEY", "sk-test")
    transcriber, speaker = build_voice(load_settings())
    assert isinstance(transcriber, OpenAITranscriber) and isinstance(speaker, OpenAISpeaker)
    assert transcriber.models == ["gpt-4o-mini-transcribe", "whisper-1"]


def test_build_voice_without_the_key_falls_back_to_local_engines(openai_env):
    openai_env.setenv("VOICE_STT", "openai")
    openai_env.setenv("VOICE_TTS", "openai")
    transcriber, speaker = build_voice(load_settings())
    assert not isinstance(transcriber, OpenAITranscriber)
    assert not isinstance(speaker, OpenAISpeaker)
