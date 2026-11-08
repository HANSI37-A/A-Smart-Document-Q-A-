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


def load_settings() -> Settings:
    load_dotenv(PROJECT_DIR / ".env")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    data_dir = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data"))
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
        model=model,
        fallback_model=os.getenv("OPENROUTER_FALLBACK_MODEL") or None,
        vision=os.getenv("VISION_PROVIDER", "openrouter").lower(),
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


def build_app(settings: Settings | None = None):
    """The one place where real objects are created and wired together."""
    from rag.embeddings import FastEmbedder
    from rag.generator import OpenRouterGenerator
    from rag.image_selector import ImageSelector, OpenRouterImageJudge
    from rag.memory import OpenRouterRewriter
    from rag.openrouter import make_client
    from rag.pipeline import DocQA
    from rag.vector_store import VectorStore
    from rag.vision import OpenRouterVisionDescriber

    s = settings or load_settings()
    store = VectorStore(str(s.data_dir / "chroma"))
    generator = describer = rewriter = judge = segmenter = None
    if s.openrouter_api_key:
        client = make_client(s.openrouter_api_key, s.site_url, s.app_name)
        generator = OpenRouterGenerator(client, s.model, fallback_model=s.fallback_model,
                                        vision_model=s.vision_model)
        rewriter = OpenRouterRewriter(client, s.helper_model, fallback_model=s.fallback_model)
        judge = OpenRouterImageJudge(client, s.helper_model, fallback_model=s.fallback_model)
        if s.vision == "openrouter":
            describer = OpenRouterVisionDescriber(client, s.vision_model,
                                                  fallback_model=s.fallback_model)
            segmenter = build_segmenter(s)   # only needed when figures are described
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
    )


def build_chat_store(settings: Settings | None = None):
    """Saved conversations, in the same data folder as the index."""
    from rag.memory import ChatStore

    s = settings or load_settings()
    return ChatStore(s.data_dir / "chats.sqlite3")
