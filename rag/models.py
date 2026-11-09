"""Plain data classes that flow through the pipeline."""

from dataclasses import dataclass, field


@dataclass
class Page:
    source: str
    page: int
    text: str


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    doc_id: str
    kind: str = "text"     # "text", "table", "figure" or "web"
    image_path: str = ""   # figures only: the saved crop ("" = no image)
    url: str = ""          # web results only: the page that was found


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


@dataclass
class Turn:
    """One message of an earlier conversation, used as chat memory."""
    role: str              # "user" or "assistant"
    content: str


@dataclass
class Answer:
    text: str
    sources: list[SearchResult]
    cited: list[int] = field(default_factory=list)
    images: list[SearchResult] = field(default_factory=list)   # figures shown to the model
    search_query: str = ""   # what was actually searched (the rewritten follow-up)
    web: bool = False        # True = answered from a Google search, not from the documents
