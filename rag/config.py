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
    data_dir: Path
    top_k: int
    min_score: float


def load_settings() -> Settings:
    load_dotenv(PROJECT_DIR / ".env")
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
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
        generator = GeminiGenerator(genai.Client(api_key=s.gemini_api_key), s.model)
    return DocQA(
        store=VectorStore(str(s.data_dir / "chroma")),
        embedder=FastEmbedder(),
        generator=generator,
        top_k=s.top_k,
        min_score=s.min_score,
    )