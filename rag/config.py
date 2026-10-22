"""Settings from environment variables (or a .env file) and the app factory."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    gemini_api_key: str | None
    model: str
    fallback_model: str | None   # used when the main model is overloaded
    vision: str          # "gemini", "claude" or "none"
    anthropic_api_key: str | None
    data_dir: Path
    top_k: int
    min_score: float


def load_settings() -> Settings:
    load_dotenv(PROJECT_DIR / ".env")
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        fallback_model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash") or None,
        vision=os.getenv("VISION_PROVIDER", "gemini").lower(),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        data_dir=Path(os.getenv("DATA_DIR", PROJECT_DIR / "data")),
        top_k=int(os.getenv("TOP_K", "5")),
        min_score=float(os.getenv("MIN_SCORE", "0.5")),
    )


def build_app(settings: Settings | None = None):
    """The one place where real objects are created and wired together."""
    from google import genai
    from rag.embeddings import FastEmbedder
    from rag.generator import GeminiGenerator
    from rag.pipeline import DocQA
    from rag.vector_store import VectorStore

    s = settings or load_settings()
    generator = None
    if s.gemini_api_key:
        generator = GeminiGenerator(genai.Client(api_key=s.gemini_api_key), s.model,
                                    fallback_model=s.fallback_model)
    describer = _build_describer(s)
    return DocQA(
        store=VectorStore(str(s.data_dir / "chroma")),
        embedder=FastEmbedder(),
        generator=generator,
        top_k=s.top_k,
        min_score=s.min_score,
        describer=describer,
    )


def _build_describer(s: Settings):
    """Vision model for charts and images. None = figures are skipped (tables still work)."""
    if s.vision == "gemini" and s.gemini_api_key:
        from google import genai
        from rag.vision import GeminiVisionDescriber
        return GeminiVisionDescriber(genai.Client(api_key=s.gemini_api_key), s.model,
                                     fallback_model=s.fallback_model)
    if s.vision == "claude" and s.anthropic_api_key:
        import anthropic  # optional dependency: pip install anthropic
        from rag.vision import ClaudeVisionDescriber
        return ClaudeVisionDescriber(anthropic.Anthropic(api_key=s.anthropic_api_key))
    return None
