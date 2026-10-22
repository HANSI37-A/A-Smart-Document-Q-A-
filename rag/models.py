
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
    kind: str = "text"   # "text", "table" or "figure"


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    sources: list[SearchResult]
    cited: list[int] = field(default_factory=list)
