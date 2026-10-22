"""Settings from environment variables (or a .env file) and the app factory."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent


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


def load_settings() -> Settings:
    load_dotenv(PROJECT_DIR / ".env")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
        model=model,
        fallback_model=os.getenv("OPENROUTER_FALLBACK_MODEL") or None,
        vision=os.getenv("VISION_PROVIDER", "openrouter").lower(),
        vision_model=os.getenv("VISION_MODEL") or model,   # must accept images
        site_url=os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501") or None,
        app_name=os.getenv("OPENROUTER_APP_NAME", "Document Q&A") or None,
        data_dir=Path(os.getenv("DATA_DIR", PROJECT_DIR / "data")),
        top_k=int(os.getenv("TOP_K", "5")),
        min_score=float(os.getenv("MIN_SCORE", "0.5")),
    )


def build_app(settings: Settings | None = None):
    """The one place where real objects are created and wired together."""
    from rag.embeddings import FastEmbedder
    from rag.generator import OpenRouterGenerator
    from rag.openrouter import make_client
    from rag.pipeline import DocQA
    from rag.vector_store import VectorStore
    from rag.vision import OpenRouterVisionDescriber

    s = settings or load_settings()
    generator = describer = None
    if s.openrouter_api_key:
        client = make_client(s.openrouter_api_key, s.site_url, s.app_name)
        generator = OpenRouterGenerator(client, s.model, fallback_model=s.fallback_model)
        if s.vision == "openrouter":
            describer = OpenRouterVisionDescriber(client, s.vision_model,
                                                  fallback_model=s.fallback_model)
    return DocQA(
        store=VectorStore(str(s.data_dir / "chroma")),
        embedder=FastEmbedder(),
        generator=generator,
        top_k=s.top_k,
        min_score=s.min_score,
        describer=describer,
    )
