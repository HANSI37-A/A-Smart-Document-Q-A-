"""Voice input and output: speech-to-text and text-to-speech, all running locally.

Like every other external dependency in this project, each engine sits behind a small
Protocol, so the app and the tests only depend on `transcribe()` and `synthesize()`.
The heavy packages (faster-whisper, piper, pyttsx3) are imported lazily, so the rest of
the app works without them (see requirements-voice.txt).
"""

import base64
import contextlib
import io
import logging
import re
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Protocol

import httpx
import openai

from rag.openrouter import RETRYABLE, OpenRouterBusyError, call_with_retry

log = logging.getLogger(__name__)

MORE_ON_SCREEN = "See the full answer on screen."


class VoiceError(Exception):
    """Speech could not be recognised or produced (the message is safe to show to the user)."""


class Transcriber(Protocol):
    def transcribe(self, audio: bytes) -> str:
        """Audio file bytes (WAV, or anything ffmpeg/PyAV can decode) -> text ('' if silent)."""


class Speaker(Protocol):
    def synthesize(self, text: str) -> bytes:
        """Text -> WAV file bytes."""


# ---------------------------------------------------------------- text for speaking

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_ICON = re.compile(r":material/[a-z0-9_]+:")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")           # [text](url) -> text
_CITATION = re.compile(r"\s*\[\d+\]")                       # same markers find_citations() reads
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_TABLE_RULE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|`|~~)")
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def speech_text(answer: str, max_chars: int = 1500) -> str:
    """Turn a Markdown answer with [n] citations into plain text that sounds natural aloud.

    Long answers are cut at a sentence boundary; the full text is still on screen.
    """
    text = _CODE_FENCE.sub(" ", answer)
    text = _ICON.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _CITATION.sub("", text)
    text = _TABLE_RULE.sub("", text)
    text = text.replace("|", ", ")
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(,\s*)+(?=[.!?]|$)", "", text)           # tidy dangling table commas
    text = re.sub(r"^(,\s*)+", "", text)

    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    ends = [m.end() for m in _SENTENCE_END.finditer(cut)]
    cut = cut[:ends[-1]] if ends else cut.rsplit(" ", 1)[0] + "..."
    return f"{cut} {MORE_ON_SCREEN}"


def question_from_input(value, transcriber) -> tuple[str, bool]:
    """(question, spoken) from what the chat bar returned.

    `value` is a plain string (typed), or an object with `.text` and `.audio` (the chat
    bar with a microphone). Typed text always wins; otherwise the recording is transcribed.
    Returns ('', True) when the recording held no speech; raises VoiceError if it fails.
    """
    if isinstance(value, str):
        return value.strip(), False
    text = (getattr(value, "text", "") or "").strip()
    audio = getattr(value, "audio", None)
    if text or audio is None or transcriber is None:
        return text, False
    return transcriber.transcribe(audio.getvalue()).strip(), True


# ---------------------------------------------------------------- speech to text

class WhisperTranscriber:
    """faster-whisper running on this machine. The model loads on first use."""

    def __init__(self, model_size: str = "base", device: str = "cpu",
                 compute_type: str = "int8", language: str | None = None, model=None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language or None
        self.label = f"Whisper {model_size}"
        self._model = model              # tests pass a fake; otherwise loaded lazily
        self._lock = threading.Lock()    # Streamlit runs sessions in different threads

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            log.info("Loading Whisper model %r (first use downloads it)...", self.model_size)
            self._model = WhisperModel(self.model_size, device=self.device,
                                       compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio: bytes) -> str:
        if not audio:
            return ""
        try:
            with self._lock:
                segments, _info = self._load().transcribe(
                    io.BytesIO(audio), language=self.language, beam_size=1, vad_filter=True)
                return " ".join(s.text.strip() for s in segments).strip()
        except ImportError:
            raise
        except Exception as e:
            raise VoiceError(f"Could not transcribe the recording: {e}") from e


# ---------------------------------------------------------------- text to speech

class PiperSpeaker:
    """Piper neural voice (a .onnx file plus its .json next to it)."""

    label = "Piper"

    def __init__(self, voice_path: Path | str, voice=None):
        self.voice_path = Path(voice_path)
        if voice is None:
            from piper import PiperVoice
            if not self.voice_path.is_file():
                raise FileNotFoundError(
                    f"Piper voice not found: {self.voice_path} "
                    f"(run: python scripts/download_voice_models.py)")
            voice = PiperVoice.load(self.voice_path)
        self._voice = voice
        self._lock = threading.Lock()

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        try:
            buffer = io.BytesIO()
            with self._lock:
                wav = wave.open(buffer, "wb")
                try:
                    self._voice.synthesize_wav(text, wav)
                except Exception:
                    # Closing a half-written WAV raises its own error and would hide the
                    # real one, so swallow that one.
                    with contextlib.suppress(wave.Error):
                        wav.close()
                    raise
                wav.close()
            return buffer.getvalue()
        except Exception as e:
            raise VoiceError(f"Could not speak the answer: {e}") from e


class Pyttsx3Speaker:
    """The operating system's own voice (SAPI5 on Windows). No download, lower quality."""

    label = "system voice"

    def __init__(self, engine_factory=None):
        if engine_factory is None:
            import pyttsx3
            engine_factory = pyttsx3.init
            engine_factory().stop()     # fail now, not on the first answer, if there is no voice
        self._engine_factory = engine_factory
        self._lock = threading.Lock()

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        path = None
        com = _com_session()
        try:
            with self._lock:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    path = Path(tmp.name)
                engine = self._engine_factory()      # a fresh engine per call: reusing one
                engine.save_to_file(text, str(path))  # breaks runAndWait() on some systems
                engine.runAndWait()
                engine.stop()
            data = path.read_bytes()
            if not data:
                raise VoiceError("The system voice produced no audio.")
            return data
        except VoiceError:
            raise
        except Exception as e:
            raise VoiceError(f"Could not speak the answer: {e}") from e
        finally:
            if path is not None:
                path.unlink(missing_ok=True)   # Windows: never leave temp files behind
            com.close()


# ---------------------------------------------------------------- Gemini (cloud)

NO_SPEECH = "NO_SPEECH"
TRANSCRIBE_PROMPT = (
    "Transcribe the speech in this audio recording exactly as it was spoken, in the language "
    "spoken. Reply with the transcript only: no quotes, labels, timestamps or commentary. "
    f"If there is no intelligible speech, reply exactly {NO_SPEECH}."
)
GEMINI_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


def pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    """Wrap raw 16-bit mono PCM (what some speech APIs return) in a WAV header."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


class GeminiTranscriber:
    """Speech to text with a Gemini model, sent as audio through the OpenAI-compatible endpoint.

    `client` is a GeminiClient (rag/gemini.py), so retry and fallback work as for answers.
    """

    def __init__(self, client, model: str, fallback_model: str | None = None,
                 language: str | None = None, max_tokens: int = 512):
        self.client = client
        self.models = [model] + ([fallback_model] if fallback_model else [])
        self.language = language or None
        self.max_tokens = max_tokens
        self.label = f"Gemini {model}"

    def transcribe(self, audio: bytes) -> str:
        if not audio:
            return ""
        prompt = TRANSCRIBE_PROMPT
        if self.language:
            prompt += f" The language spoken is '{self.language}'."
        encoded = base64.b64encode(audio).decode("ascii")
        fmt = "mp3" if audio[:3] == b"ID3" else "wav"      # the chat bar records WAV
        try:
            response = call_with_retry(
                lambda model: self.client.chat.completions.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": encoded, "format": fmt}},
                    ]}],
                ),
                self.models,
            )
        except (OpenRouterBusyError, openai.OpenAIError) as e:
            raise VoiceError(f"Could not transcribe the recording: {e}") from e
        text = (response.choices[0].message.content or "").strip()
        return "" if text.strip(" .").upper() == NO_SPEECH else text


class GeminiSpeaker:
    """Text to speech with a Gemini TTS model (its native speech endpoint, not the chat one)."""

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash-tts", voice: str = "Kore",
                 post=None, attempts: int = 3, delay: float = 1.0, sleep=time.sleep):
        self._api_key = api_key
        self.model = model
        self.voice = voice
        self._post = post or httpx.post     # tests pass a fake with the same signature
        self.attempts, self.delay, self._sleep = attempts, delay, sleep
        self.label = f"Gemini {voice}"

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        body = {
            "model": self.model,
            "input": [{"type": "user_input", "content": [{"type": "text", "text": text}]}],
            "response_format": {"type": "audio", "mime_type": "audio/wav"},
            "generation_config": {"speech_config": [{"voice": self.voice}]},
        }
        response = self._send(body)
        try:
            audio = _audio_from_interaction(response.json())
        except ValueError as e:
            raise VoiceError("Gemini sent back something that is not audio.") from e
        if not audio:
            raise VoiceError("Gemini returned no audio for this answer.")
        return audio if audio[:4] == b"RIFF" else pcm_to_wav(audio)

    def _send(self, body: dict):
        """POST with a few retries for busy servers; never puts the key in an error message."""
        error = "unknown error"
        for attempt in range(self.attempts):
            try:
                response = self._post(GEMINI_INTERACTIONS_URL, json=body, timeout=90,
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
        raise VoiceError(f"Gemini could not speak the answer ({error}).")


def _audio_from_interaction(payload: dict) -> bytes:
    """The first audio item of a Gemini interaction response (base64 in steps[].content[])."""
    for step in payload.get("steps") or []:
        for item in step.get("content") or []:
            if item.get("type") == "audio" and item.get("data"):
                return base64.b64decode(item["data"])
    return b""


def _error_message(response) -> str:
    try:
        message = response.json()["error"]["message"]
    except Exception:
        message = getattr(response, "text", "") or "no details"
    return str(message)[:200]


# ---------------------------------------------------------------- OpenAI (cloud)

class OpenAITranscriber:
    """Speech to text with OpenAI's transcription models (gpt-4o-mini-transcribe, whisper-1).

    `client` is an OpenAIClient (rag/openai_client.py), so retry and fallback work as for answers.
    """

    def __init__(self, client, model: str = "gpt-4o-mini-transcribe",
                 fallback_model: str | None = None, language: str | None = None):
        self.client = client
        self.models = [model] + ([fallback_model] if fallback_model else [])
        self.language = language or None
        self.label = f"OpenAI {model}"

    def transcribe(self, audio: bytes) -> str:
        if not audio:
            return ""
        extra = {"language": self.language} if self.language else {}
        name = "question.mp3" if audio[:3] == b"ID3" else "question.wav"   # the chat bar records WAV
        try:
            response = call_with_retry(
                lambda model: self.client.audio.transcriptions.create(
                    model=model, file=(name, audio), **extra),
                self.models,
            )
        except (OpenRouterBusyError, openai.OpenAIError) as e:
            raise VoiceError(f"Could not transcribe the recording: {e}") from e
        return (getattr(response, "text", None) or "").strip()


class OpenAISpeaker:
    """Text to speech with OpenAI's speech models (gpt-4o-mini-tts, tts-1)."""

    def __init__(self, client, model: str = "gpt-4o-mini-tts", voice: str = "alloy"):
        self.client = client
        self.model = model
        self.voice = voice
        self.label = f"OpenAI {voice}"

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        try:
            response = call_with_retry(
                lambda model: self.client.audio.speech.create(
                    model=model, voice=self.voice, input=text[:4000], response_format="wav"),
                [self.model],
            )
        except (OpenRouterBusyError, openai.OpenAIError) as e:
            raise VoiceError(f"OpenAI could not speak the answer: {e}") from e
        audio = getattr(response, "content", None)
        if audio is None:
            audio = response.read()
        if not audio:
            raise VoiceError("OpenAI returned no audio for this answer.")
        return audio


class _com_session:
    """Streamlit runs scripts in worker threads; SAPI5 needs COM initialised in each one."""

    def __init__(self):
        self._comtypes = None
        try:
            import comtypes
            comtypes.CoInitialize()
            self._comtypes = comtypes
        except Exception:      # not Windows, or comtypes missing: nothing to do
            pass

    def close(self):
        if self._comtypes is not None:
            try:
                self._comtypes.CoUninitialize()
            except Exception:
                pass
