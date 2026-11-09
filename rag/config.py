"""Settings from environment variables (or a .env file) and the app factory."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
log = logging.getLogger(__name__)


@dataclass
class Settings:
    openrouter_api_key: str | None
    model: str
    fallback_model: str | None   # used when the main model is overloaded
    vision: str                  # "openrouter" or "none"
    vision_model: str
    site_url: str | None         # optional OpenRouter attribution headers
    app_name: str | None
    data_dir: Path
    top_k: int
    min_score: float
    segmenter: str = "sam"       # "sam" or "none" (crop pdfplumber's boxes)
    sam_weights: Path | None = None
    helper_model: str = ""       # small calls: follow-up rewriting, image judging
    image_min_score: float = 0.45
    max_images: int = 2
    memory_messages: int = 6     # earlier messages sent along with a question
    voice_stt: str = "whisper"   # "whisper" or "none" (speech to text)
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute: str = "int8"
    voice_language: str | None = None   # None = detect the spoken language
    voice_tts: str = "piper"     # "piper", "pyttsx3" or "none" (text to speech)
    piper_voice: Path | None = None
    voice_autoplay: bool = True  # play spoken answers as soon as they are ready
    llm_provider: str = "gemini"   # "openrouter" or "openrouter": who writes the answers
    gemini_api_key: str | None = None
    gemini_reasoning: str | None = None   # e.g. "low"; None = leave the model's default
    gemini_stt_model: str = "gemini-3.8-flash"
    gemini_tts_model: str = "gemini-3.8-flash-tts"
    gemini_voice: str = "Kore"
    gemini_fallback_model: str | None = None
    web_search: bool = True      # answer from Google when the documents don't (needs a Gemini key)
    web_search_model: str = "gemini-3.8-flash"
    openai_api_key: str | None = None
    openai_reasoning: str | None = None   # e.g. "low" (gpt-5 / o-series only); None = default
    openai_stt_model: str = "gpt-4o-mini-transcribe"
    openai_stt_fallback_model: str | None = "whisper-1"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_voice: str = "alloy"

    @property
    def api_key(self) -> str | None:
        """The key of the provider that answers questions (None = search-only mode)."""
        return {"openai": self.openai_api_key,
                "gemini": self.gemini_api_key}.get(self.llm_provider, self.openrouter_api_key)


LLM_PROVIDERS = ("openai", "gemini", "openrouter")


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def load_settings() -> Settings:
    load_dotenv(PROJECT_DIR / ".env")
    gemini_key = os.getenv("GEMINI_API_KEY") or None
    openai_key = os.getenv("OPENAI_API_KEY") or None
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    if provider == "auto":     # an OpenAI key wins, then Gemini; otherwise OpenRouter as before
        provider = "openai" if openai_key else "gemini" if gemini_key else "openrouter"
    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER must be auto, openai, gemini or openrouter (got {provider!r}).")
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        fallback = os.getenv("OPENAI_FALLBACK_MODEL") or None
    elif provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        fallback = os.getenv("GEMINI_FALLBACK_MODEL") or None
    else:
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        fallback = os.getenv("OPENROUTER_FALLBACK_MODEL") or None
    data_dir = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data"))
    voice_default = "openai" if openai_key else "gemini" if gemini_key else "whisper"
    tts_default = "openai" if openai_key else "gemini" if gemini_key else "piper"
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
        llm_provider=provider,
        gemini_api_key=gemini_key,
        gemini_reasoning=os.getenv("GEMINI_REASONING") or None,
        gemini_stt_model=os.getenv("GEMINI_STT_MODEL") or "gemini-3.8-flash",
        gemini_tts_model=os.getenv("GEMINI_TTS_MODEL") or "gemini-3.8-flash-tts",
        gemini_voice=os.getenv("GEMINI_VOICE") or "Kore",
        gemini_fallback_model=os.getenv("GEMINI_FALLBACK_MODEL") or None,
        model=model,
        fallback_model=fallback,
        vision=os.getenv("VISION_PROVIDER", "auto").lower(),
        vision_model=os.getenv("VISION_MODEL") or model,   # must accept images
        site_url=os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501") or None,
        app_name=os.getenv("OPENROUTER_APP_NAME", "Document Q&A") or None,
        data_dir=data_dir,
        top_k=int(os.getenv("TOP_K", "5")),
        min_score=float(os.getenv("MIN_SCORE", "0.5")),
        segmenter=os.getenv("SEGMENTER", "sam").lower(),
        sam_weights=Path(os.getenv("SAM_WEIGHTS") or data_dir / "models" / "mobile_sam.pt"),
        helper_model=os.getenv("HELPER_MODEL") or model,
        image_min_score=float(os.getenv("IMAGE_MIN_SCORE", "0.45")),
        max_images=int(os.getenv("MAX_IMAGES", "2")),
        memory_messages=int(os.getenv("MEMORY_MESSAGES", "6")),
        # with a cloud key, voice defaults to the same company (OpenAI, else Gemini), otherwise
        # it runs locally; set VOICE_STT / VOICE_TTS to change it
        voice_stt=os.getenv("VOICE_STT", voice_default).lower(),
        whisper_model=os.getenv("WHISPER_MODEL", "base"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute=os.getenv("WHISPER_COMPUTE", "int8"),
        voice_language=os.getenv("VOICE_LANGUAGE") or None,
        voice_tts=os.getenv("VOICE_TTS", tts_default).lower(),
        piper_voice=Path(os.getenv("PIPER_VOICE")
                         or data_dir / "models" / "piper" / "en_US-lessac-medium.onnx"),
        voice_autoplay=_flag("VOICE_AUTOPLAY", True),
        web_search=_flag("WEB_SEARCH", True),
        web_search_model=(os.getenv("WEB_SEARCH_MODEL")
                          or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"),
        openai_api_key=openai_key,
        openai_reasoning=os.getenv("OPENAI_REASONING") or None,
        openai_stt_model=os.getenv("OPENAI_STT_MODEL") or "gpt-4o-mini-transcribe",
        openai_stt_fallback_model=os.getenv("OPENAI_STT_FALLBACK_MODEL", "whisper-1") or None,
        openai_tts_model=os.getenv("OPENAI_TTS_MODEL") or "gpt-4o-mini-tts",
        openai_voice=os.getenv("OPENAI_VOICE") or "alloy",
    )


def build_segmenter(s: Settings):
    """SAM if it is wanted and installed; otherwise crop the boxes pdfplumber finds."""
    from rag.segmenter import PassThroughSegmenter, SamSegmenter

    if s.segmenter != "sam":
        return PassThroughSegmenter()
    try:
        return SamSegmenter(s.sam_weights)
    except ImportError:
        log.warning("SEGMENTER=sam but the 'ultralytics' package is not installed; "
                    "figures are cropped without SAM.")
    except Exception as e:   # e.g. the weights could not be downloaded
        log.warning("Could not load SAM (%s); figures are cropped without SAM.", e)
    return PassThroughSegmenter()


def build_llm_client(s: Settings):
    """The chat client of the provider that answers questions, or None without a key.

    All of them return an OpenAI-SDK style client, so the classes named OpenRouter* (generator,
    rewriter, image judge, vision describer) work with any provider unchanged.
    """
    if not s.api_key:
        return None
    if s.llm_provider == "openai":
        from rag.openai_client import make_openai_client
        return make_openai_client(s.api_key, s.openai_reasoning)
    if s.llm_provider == "gemini":
        from rag.gemini import make_gemini_client
        return make_gemini_client(s.api_key, s.gemini_reasoning)
    from rag.openrouter import make_client
    return make_client(s.api_key, s.site_url, s.app_name)


def build_app(settings: Settings | None = None):
    """The one place where real objects are created and wired together."""
    from rag.embeddings import FastEmbedder
    from rag.generator import OpenRouterGenerator
    from rag.image_selector import ImageSelector, OpenRouterImageJudge
    from rag.memory import OpenRouterRewriter
    from rag.pipeline import DocQA
    from rag.vector_store import VectorStore
    from rag.vision import OpenRouterVisionDescriber

    s = settings or load_settings()
    store = VectorStore(str(s.data_dir / "chroma"))
    generator = describer = rewriter = judge = segmenter = None
    client = build_llm_client(s)
    if client is not None:
        generator = OpenRouterGenerator(client, s.model, fallback_model=s.fallback_model,
                                        vision_model=s.vision_model)
        rewriter = OpenRouterRewriter(client, s.helper_model, fallback_model=s.fallback_model)
        judge = OpenRouterImageJudge(client, s.helper_model, fallback_model=s.fallback_model)
        if s.vision != "none":   # "auto" (default) or a provider name: describe figures
            describer = OpenRouterVisionDescriber(client, s.vision_model,
                                                  fallback_model=s.fallback_model)
            segmenter = build_segmenter(s)   # only needed when figures are described
    web_searcher = None
    if s.web_search and s.gemini_api_key:   # Google Search always goes through Gemini
        from rag.web_search import GeminiWebSearcher
        web_searcher = GeminiWebSearcher(s.gemini_api_key, s.web_search_model)
    return DocQA(
        store=store,
        embedder=FastEmbedder(),
        generator=generator,
        top_k=s.top_k,
        min_score=s.min_score,
        describer=describer,
        segmenter=segmenter,
        rewriter=rewriter,
        image_selector=ImageSelector(store, judge, s.image_min_score, s.max_images),
        figures_dir=s.data_dir / "figures",
        web_searcher=web_searcher,
    )


def build_voice(settings: Settings | None = None):
    """(transcriber, speaker) for the voice assistant; either is None when not available.

    Each half is "openai" (cloud, needs OPENAI_API_KEY), "gemini" (cloud, needs
    GEMINI_API_KEY) or local (Whisper / Piper / system voice). Nothing is downloaded or loaded
    here except a local text-to-speech voice: the Whisper model loads on its first use. A
    missing key, package or voice file switches to the local engine, or that half off, with a
    warning (like build_segmenter() for SAM).
    """
    from rag.voice import (GeminiSpeaker, GeminiTranscriber, OpenAISpeaker, OpenAITranscriber,
                           PiperSpeaker, Pyttsx3Speaker, WhisperTranscriber)

    s = settings or load_settings()
    stt, tts = s.voice_stt, s.voice_tts
    keys = {"openai": s.openai_api_key, "gemini": s.gemini_api_key}
    for name, key in keys.items():
        if name in (stt, tts) and not key:
            log.warning("Voice is set to %s but %s_API_KEY is not set; using local engines.",
                        name.capitalize(), name.upper())
            stt = "whisper" if stt == name else stt
            tts = "piper" if tts == name else tts

    transcriber = None
    if stt == "openai":
        from rag.openai_client import make_openai_client
        transcriber = OpenAITranscriber(make_openai_client(s.openai_api_key), s.openai_stt_model,
                                        fallback_model=s.openai_stt_fallback_model,
                                        language=s.voice_language)
    elif stt == "gemini":
        from rag.gemini import make_gemini_client
        transcriber = GeminiTranscriber(make_gemini_client(s.gemini_api_key), s.gemini_stt_model,
                                        fallback_model=s.gemini_fallback_model,
                                        language=s.voice_language)
    elif stt == "whisper":
        try:
            import faster_whisper  # noqa: F401  (fail now if it is not installed)
            transcriber = WhisperTranscriber(s.whisper_model, s.whisper_device,
                                             s.whisper_compute, s.voice_language)
        except ImportError:
            log.warning("VOICE_STT=whisper but 'faster-whisper' is not installed "
                        "(pip install -r requirements-voice.txt); voice input is off.")

    speaker = None
    if tts == "openai":
        from rag.openai_client import make_openai_client
        speaker = OpenAISpeaker(make_openai_client(s.openai_api_key), s.openai_tts_model,
                                s.openai_voice)
    elif tts == "gemini":
        speaker = GeminiSpeaker(s.gemini_api_key, s.gemini_tts_model, s.gemini_voice)
    elif tts == "piper":
        try:
            speaker = PiperSpeaker(s.piper_voice)
        except ImportError:
            log.warning("VOICE_TTS=piper but 'piper-tts' is not installed; trying pyttsx3.")
        except Exception as e:   # e.g. the voice file has not been downloaded
            log.warning("Could not load the Piper voice (%s); trying pyttsx3.", e)
    if speaker is None and tts in ("piper", "pyttsx3"):
        try:
            speaker = Pyttsx3Speaker()
        except ImportError:
            log.warning("'pyttsx3' is not installed; spoken answers are off.")
        except Exception as e:
            log.warning("Could not start the system voice (%s); spoken answers are off.", e)
    return transcriber, speaker


def build_chat_store(settings: Settings | None = None):
    """Saved conversations, in the same data folder as the index."""
    from rag.memory import ChatStore

    s = settings or load_settings()
    return ChatStore(s.data_dir / "chats.sqlite3")
