"""Voice: text for speaking, speech engines (with fakes) and how the app builds them."""
import base64
import io
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from rag.config import build_voice, load_settings
from rag.openrouter import OpenRouterBusyError
from rag.voice import (MORE_ON_SCREEN, GeminiSpeaker, GeminiTranscriber, PiperSpeaker,
                       Pyttsx3Speaker, VoiceError, pcm_to_wav,
                       WhisperTranscriber, question_from_input, speech_text)

from conftest import FakeSpeaker, FakeTranscriber


# ---------------------------------------------------------------- speech_text

def test_speech_text_removes_citation_markers():
    assert speech_text("Employees get 14 days of leave [1]. Managers get 20 [2][3].") == \
        "Employees get 14 days of leave. Managers get 20."


def test_speech_text_removes_markdown_and_icons():
    text = ("## Leave\n\n- **Annual:** 14 days [1]\n- _Sick:_ 10 days\n\n"
            ":material/error: See [the handbook](http://x.y/z) and `HR`.")
    spoken = speech_text(text)
    assert spoken == "Leave Annual: 14 days Sick: 10 days See the handbook and HR."
    for junk in ("#", "*", "_", "`", "[", "]", "http", "material"):
        assert junk not in spoken


def test_speech_text_reads_a_table_as_plain_words():
    table = "| Year | Revenue |\n|---|---|\n| 2023 | $4M |\n| 2024 | $6M |"
    spoken = speech_text(table)
    assert "---" not in spoken and "|" not in spoken
    assert "2024" in spoken and "$6M" in spoken


def test_speech_text_drops_code_blocks_and_handles_empty():
    assert speech_text("Run this:\n```\nrm -rf /\n```\nDone.") == "Run this: Done."
    assert speech_text("") == ""
    assert speech_text("[1] [2]") == ""


def test_speech_text_cuts_long_answers_at_a_sentence():
    text = "First sentence here. Second sentence here. Third sentence here."
    spoken = speech_text(text, max_chars=45)
    assert spoken == f"First sentence here. Second sentence here. {MORE_ON_SCREEN}"
    assert len(speech_text("word " * 1000, max_chars=100)) < 140   # no sentence end at all


def test_speech_text_keeps_short_answers_whole():
    assert MORE_ON_SCREEN not in speech_text("Short answer.")


# ---------------------------------------------------------------- question_from_input

def audio_of(data=b"wav"):
    return SimpleNamespace(getvalue=lambda: data)


def test_typed_question_is_returned_as_is_and_not_transcribed():
    ears = FakeTranscriber()
    assert question_from_input("  hello  ", ears) == ("hello", False)
    assert ears.audio == []


def test_recording_is_transcribed_and_marked_as_spoken():
    ears = FakeTranscriber("what is the revenue")
    value = SimpleNamespace(text="", audio=audio_of(b"RIFF1"))
    assert question_from_input(value, ears) == ("what is the revenue", True)
    assert ears.audio == [b"RIFF1"]


def test_typed_text_wins_over_a_recording():
    ears = FakeTranscriber()
    value = SimpleNamespace(text="typed", audio=audio_of())
    assert question_from_input(value, ears) == ("typed", False)
    assert ears.audio == []


def test_silent_recording_gives_an_empty_spoken_question():
    value = SimpleNamespace(text="", audio=audio_of())
    assert question_from_input(value, FakeTranscriber("  ")) == ("", True)


def test_recording_without_a_transcriber_is_ignored():
    value = SimpleNamespace(text="", audio=audio_of())
    assert question_from_input(value, None) == ("", False)


def test_nothing_submitted():
    assert question_from_input(SimpleNamespace(text="", audio=None), FakeTranscriber()) == ("", False)


# ---------------------------------------------------------------- WhisperTranscriber

class FakeWhisperModel:
    def __init__(self, words=("  What was ", "the revenue? "), error=None):
        self.words, self.error, self.calls = words, error, []

    def transcribe(self, audio, **options):
        self.calls.append((audio.read(), options))
        if self.error:
            raise self.error
        return (SimpleNamespace(text=w) for w in self.words), SimpleNamespace()


def test_whisper_joins_segments_and_filters_silence():
    model = FakeWhisperModel()
    text = WhisperTranscriber(language="en", model=model).transcribe(b"RIFFdata")
    assert text == "What was the revenue?"
    data, options = model.calls[0]
    assert data == b"RIFFdata"
    assert options["language"] == "en" and options["vad_filter"] is True


def test_whisper_returns_empty_text_for_silence_or_no_audio():
    quiet = FakeWhisperModel(words=())
    assert WhisperTranscriber(model=quiet).transcribe(b"RIFFdata") == ""
    assert WhisperTranscriber(model=quiet).transcribe(b"") == ""
    assert len(quiet.calls) == 1   # empty bytes never reach the model


def test_whisper_failure_becomes_a_voice_error():
    model = FakeWhisperModel(error=RuntimeError("bad audio"))
    with pytest.raises(VoiceError, match="bad audio"):
        WhisperTranscriber(model=model).transcribe(b"RIFFdata")


def test_whisper_model_loads_lazily(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)   # not installed
    t = WhisperTranscriber()          # creating it must not need the package
    with pytest.raises(ImportError):
        t.transcribe(b"RIFFdata")     # only the first use does


# ---------------------------------------------------------------- speakers

class FakePiperVoice:
    def synthesize_wav(self, text, wav_file):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 100)


def test_piper_speaker_returns_a_wav_file():
    audio = PiperSpeaker("voice.onnx", voice=FakePiperVoice()).synthesize("Hello there.")
    assert audio[:4] == b"RIFF"
    with wave.open(io.BytesIO(audio)) as wav:
        assert wav.getframerate() == 22050 and wav.getnframes() == 100


def test_piper_speaker_says_nothing_for_empty_text():
    assert PiperSpeaker("voice.onnx", voice=FakePiperVoice()).synthesize("  ") == b""


def test_piper_speaker_failure_becomes_a_voice_error():
    class Broken:
        def synthesize_wav(self, text, wav_file):
            raise RuntimeError("no phonemes")

    with pytest.raises(VoiceError, match="no phonemes"):
        PiperSpeaker("voice.onnx", voice=Broken()).synthesize("Hello")


def test_piper_speaker_reports_a_missing_voice_file(tmp_path):
    pytest.importorskip("piper")
    with pytest.raises(FileNotFoundError, match="download_voice_models"):
        PiperSpeaker(tmp_path / "missing.onnx")


class FakeEngine:
    """Mimics the pyttsx3 calls used: save_to_file, runAndWait, stop."""

    def __init__(self, data=b"RIFFsystem"):
        self.data, self.saved_to, self.said = data, None, None

    def save_to_file(self, text, path):
        self.said, self.saved_to = text, Path(path)
        self.saved_to.write_bytes(self.data)

    def runAndWait(self):
        pass

    def stop(self):
        pass


def test_pyttsx3_speaker_returns_audio_and_removes_its_temp_file():
    engine = FakeEngine()
    audio = Pyttsx3Speaker(engine_factory=lambda: engine).synthesize("Hello")
    assert audio == b"RIFFsystem" and engine.said == "Hello"
    assert not engine.saved_to.exists()


def test_pyttsx3_speaker_complains_when_no_audio_was_made():
    engine = FakeEngine(data=b"")
    with pytest.raises(VoiceError, match="no audio"):
        Pyttsx3Speaker(engine_factory=lambda: engine).synthesize("Hello")
    assert not engine.saved_to.exists()


# ---------------------------------------------------------------- settings and build_voice

@pytest.fixture
def voice_env(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.config.load_dotenv", lambda *a, **k: None)   # ignore the real .env
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for name in ("VOICE_STT", "VOICE_TTS", "WHISPER_MODEL", "WHISPER_DEVICE", "WHISPER_COMPUTE",
                 "VOICE_LANGUAGE", "PIPER_VOICE", "VOICE_AUTOPLAY", "GEMINI_API_KEY",
                 "GEMINI_STT_MODEL", "GEMINI_TTS_MODEL", "GEMINI_VOICE", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_voice_settings_have_working_defaults(voice_env):
    s = load_settings()
    assert (s.voice_stt, s.whisper_model, s.whisper_device, s.whisper_compute) == \
        ("whisper", "base", "cpu", "int8")
    assert s.voice_language is None and s.voice_tts == "piper" and s.voice_autoplay is True
    assert s.piper_voice == voice_env / "models" / "piper" / "en_US-lessac-medium.onnx"


def test_voice_settings_come_from_the_environment(voice_env, monkeypatch):
    monkeypatch.setenv("VOICE_STT", "NONE")
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.setenv("VOICE_LANGUAGE", "de")
    monkeypatch.setenv("VOICE_TTS", "pyttsx3")
    monkeypatch.setenv("VOICE_AUTOPLAY", "false")
    s = load_settings()
    assert (s.voice_stt, s.whisper_model, s.voice_language, s.voice_tts, s.voice_autoplay) == \
        ("none", "small", "de", "pyttsx3", False)


def test_voice_can_be_switched_off(voice_env, monkeypatch):
    monkeypatch.setenv("VOICE_STT", "none")
    monkeypatch.setenv("VOICE_TTS", "none")
    assert build_voice() == (None, None)


def test_missing_packages_switch_voice_off_instead_of_crashing(voice_env, monkeypatch):
    for module in ("faster_whisper", "piper", "pyttsx3"):
        monkeypatch.setitem(sys.modules, module, None)   # makes `import module` fail
    assert build_voice() == (None, None)


def test_missing_piper_voice_falls_back_to_the_system_voice(voice_env, monkeypatch):
    class SystemVoice(FakeSpeaker):
        pass

    monkeypatch.setattr("rag.voice.Pyttsx3Speaker", SystemVoice)
    monkeypatch.setenv("VOICE_STT", "none")
    ears, mouth = build_voice()          # the Piper .onnx file does not exist in tmp_path
    assert ears is None and isinstance(mouth, SystemVoice)


def test_whisper_is_configured_from_settings(voice_env, monkeypatch):
    pytest.importorskip("faster_whisper")
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("VOICE_LANGUAGE", "fr")
    monkeypatch.setenv("VOICE_TTS", "none")
    ears, mouth = build_voice()
    assert isinstance(ears, WhisperTranscriber) and mouth is None
    assert (ears.model_size, ears.language, ears._model) == ("tiny", "fr", None)   # not loaded yet


# ---------------------------------------------------------------- Gemini speech to text

class FakeChatClient:
    """Stands in for the Gemini chat client: records requests, returns text or raises."""

    def __init__(self, reply="What was the revenue?", error=None):
        self.reply, self.error, self.sent = reply, error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def test_gemini_transcriber_sends_the_recording_as_audio():
    client = FakeChatClient("  What was the revenue?\n")
    text = GeminiTranscriber(client, "gemini-3.8-flash").transcribe(b"RIFFwavbytes")
    assert text == "What was the revenue?"
    sent = client.sent[0]
    assert sent["model"] == "gemini-3.8-flash" and sent["max_tokens"] == 512
    parts = sent["messages"][0]["content"]
    assert parts[0]["type"] == "text" and "transcript only" in parts[0]["text"]
    assert parts[1] == {"type": "input_audio", "input_audio": {
        "data": base64.b64encode(b"RIFFwavbytes").decode(), "format": "wav"}}


def test_gemini_transcriber_passes_a_language_hint():
    client = FakeChatClient()
    GeminiTranscriber(client, "m", language="de").transcribe(b"RIFFx")
    assert "'de'" in client.sent[0]["messages"][0]["content"][0]["text"]


@pytest.mark.parametrize("reply", ["NO_SPEECH", " no_speech. ", "", None])
def test_gemini_transcriber_treats_no_speech_as_silence(reply):
    assert GeminiTranscriber(FakeChatClient(reply), "m").transcribe(b"RIFFx") == ""


def test_gemini_transcriber_skips_empty_recordings():
    client = FakeChatClient()
    assert GeminiTranscriber(client, "m").transcribe(b"") == ""
    assert client.sent == []


def test_gemini_transcriber_errors_become_voice_errors():
    bad_key = openai.AuthenticationError(
        "API key not valid", body=None,
        response=httpx.Response(401, request=httpx.Request("POST", "http://x")))
    with pytest.raises(VoiceError, match="API key not valid"):
        GeminiTranscriber(FakeChatClient(error=bad_key), "m").transcribe(b"RIFFx")


def test_gemini_transcriber_reports_a_busy_service(monkeypatch):
    def always_busy(call, models, **kwargs):
        assert models == ["main", "backup"]      # the fallback model is offered too
        raise OpenRouterBusyError("The AI service is busy right now")

    monkeypatch.setattr("rag.voice.call_with_retry", always_busy)
    with pytest.raises(VoiceError, match="busy"):
        GeminiTranscriber(FakeChatClient(), "main", fallback_model="backup").transcribe(b"RIFFx")


# ---------------------------------------------------------------- Gemini text to speech

class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def audio_payload(audio: bytes) -> dict:
    """The shape of a Gemini interaction response that holds audio."""
    return {"steps": [
        {"type": "user_input", "content": [{"type": "text", "text": "hi"}]},
        {"type": "model_output", "content": [
            {"type": "audio", "data": base64.b64encode(audio).decode()}]},
    ]}


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


def gemini_speaker(post, **kwargs):
    return GeminiSpeaker("secret-key", post=post, sleep=lambda s: None, **kwargs)


def test_gemini_speaker_sends_the_documented_request():
    wav = pcm_to_wav(b"\x00\x00" * 50)
    post = FakePost(FakeResponse(payload=audio_payload(wav)))
    audio = gemini_speaker(post, model="gemini-3.8-flash-tts", voice="Puck").synthesize("Hello there.")
    assert audio == wav
    url, kwargs = post.calls[0]
    assert url == "https://generativelanguage.googleapis.com/v1beta/interactions"
    assert kwargs["headers"] == {"x-goog-api-key": "secret-key"}
    assert kwargs["json"] == {
        "model": "gemini-3.8-flash-tts",
        "input": [{"type": "user_input", "content": [{"type": "text", "text": "Hello there."}]}],
        "response_format": {"type": "audio", "mime_type": "audio/wav"},
        "generation_config": {"speech_config": [{"voice": "Puck"}]},
    }


def test_gemini_speaker_wraps_raw_pcm_in_a_wav_header():
    audio = gemini_speaker(FakePost(FakeResponse(payload=audio_payload(b"\x01\x00" * 240)))
                           ).synthesize("Hello")
    assert audio[:4] == b"RIFF"
    with wave.open(io.BytesIO(audio)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) ==             (24000, 1, 2, 240)


def test_gemini_speaker_says_nothing_for_empty_text():
    post = FakePost()
    assert gemini_speaker(post).synthesize("   ") == b""
    assert post.calls == []


def test_gemini_speaker_retries_a_busy_server_then_succeeds():
    wav = pcm_to_wav(b"\x00\x00" * 10)
    post = FakePost(FakeResponse(429, {"error": {"message": "slow down"}}),
                    httpx.ConnectTimeout("timed out"),
                    FakeResponse(payload=audio_payload(wav)))
    assert gemini_speaker(post).synthesize("Hello") == wav
    assert len(post.calls) == 3


def test_gemini_speaker_gives_up_after_the_last_attempt():
    post = FakePost(*[FakeResponse(503, {"error": {"message": "overloaded"}})] * 3)
    with pytest.raises(VoiceError, match="HTTP 503: overloaded"):
        gemini_speaker(post).synthesize("Hello")
    assert len(post.calls) == 3


def test_gemini_speaker_does_not_retry_a_bad_request_or_leak_the_key():
    post = FakePost(FakeResponse(400, {"error": {"message": "API key not valid"}}))
    with pytest.raises(VoiceError, match="HTTP 400: API key not valid") as caught:
        gemini_speaker(post).synthesize("Hello")
    assert len(post.calls) == 1
    assert "secret-key" not in str(caught.value)


def test_gemini_speaker_reports_a_plain_text_error_body():
    post = FakePost(FakeResponse(404, None, text="Not Found"))
    with pytest.raises(VoiceError, match="HTTP 404: Not Found"):
        gemini_speaker(post).synthesize("Hello")


@pytest.mark.parametrize("payload", [{}, {"steps": []}, {"steps": [{"type": "model_output"}]},
                                     {"steps": [{"content": [{"type": "text", "text": "x"}]}]}])
def test_gemini_speaker_complains_when_no_audio_comes_back(payload):
    with pytest.raises(VoiceError, match="no audio"):
        gemini_speaker(FakePost(FakeResponse(payload=payload))).synthesize("Hello")


def test_gemini_speaker_complains_about_a_reply_that_is_not_json():
    with pytest.raises(VoiceError, match="not audio"):
        gemini_speaker(FakePost(FakeResponse(200, None))).synthesize("Hello")


# ---------------------------------------------------------------- build_voice with Gemini

def test_with_a_gemini_key_voice_uses_gemini_and_loads_no_local_model(voice_env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("GEMINI_VOICE", "Puck")
    ears, mouth = build_voice()
    assert isinstance(ears, GeminiTranscriber) and isinstance(mouth, GeminiSpeaker)
    assert (ears.label, mouth.label) == ("Gemini gemini-3.8-flash", "Gemini Puck")
    assert mouth._api_key == "g-key"
    assert ears.client._client.api_key == "g-key"


def test_gemini_voice_without_a_key_falls_back_to_the_local_engines(voice_env, monkeypatch):
    pytest.importorskip("faster_whisper")
    monkeypatch.setenv("VOICE_STT", "gemini")
    monkeypatch.setenv("VOICE_TTS", "gemini")
    monkeypatch.setattr("rag.voice.Pyttsx3Speaker", FakeSpeaker)     # Piper file is missing here
    ears, mouth = build_voice()
    assert isinstance(ears, WhisperTranscriber) and isinstance(mouth, FakeSpeaker)


def test_voice_halves_can_use_different_providers(voice_env, monkeypatch):
    pytest.importorskip("faster_whisper")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("VOICE_STT", "whisper")           # hear locally, speak with Gemini
    ears, mouth = build_voice()
    assert isinstance(ears, WhisperTranscriber) and isinstance(mouth, GeminiSpeaker)


def test_gemini_voice_can_be_switched_off(voice_env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("VOICE_STT", "none")
    monkeypatch.setenv("VOICE_TTS", "none")
    assert build_voice() == (None, None)


# ---------------------------------------------------------------- the real engines

@pytest.mark.slow
def test_spoken_question_survives_a_round_trip_through_the_real_engines(voice_env):
    s = load_settings()
    piper_file = Path(__file__).resolve().parent.parent / "data" / "models" / "piper" \
        / "en_US-lessac-medium.onnx"
    pytest.importorskip("faster_whisper")
    pytest.importorskip("piper")
    if not piper_file.exists():
        pytest.skip("Run `python scripts/download_voice_models.py` first")
    s.piper_voice = piper_file
    ears, mouth = build_voice(s)
    assert ears is not None and mouth is not None

    heard = ears.transcribe(mouth.synthesize("What was the total revenue last year?"))
    assert "revenue" in heard.lower()
