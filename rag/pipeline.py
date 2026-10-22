"""Connects everything: ingest PDFs and answer questions."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag.chunker import chunk_pages
from rag.embeddings import Embedder
from rag.generator import find_citations
from rag.models import Answer, SearchResult
from rag.pdf_loader import load_pdf
from rag.vector_store import VectorStore

NOT_FOUND = "I couldn't find that in the documents."


class Generator(Protocol):
    def generate(self, question: str, results: list[SearchResult]) -> str: ...


@dataclass
class IngestReport:
    source: str
    doc_id: str
    pages: int
    chunks: int


def file_fingerprint(path: Path) -> str:
    """Short id based only on the file's bytes, not its name."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


class DocQA:
    def __init__(self, store: VectorStore, embedder: Embedder, generator: Generator | None,
                 top_k: int = 5, min_score: float = 0.5, chunk_size: int = 500,
                 overlap: int = 100):
        self.store = store
        self.embedder = embedder
        self.generator = generator   # None = search-only mode
        self.top_k = top_k
        self.min_score = min_score
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest_pdf(self, path: str | Path, source_name: str | None = None) -> IngestReport:
        path = Path(path)
        doc_id = file_fingerprint(path)
        pages = load_pdf(path, source_name)
        chunks = chunk_pages(pages, doc_id, self.chunk_size, self.overlap)
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        return IngestReport(pages[0].source, doc_id, len(pages), len(chunks))

    def retrieve(self, question: str) -> list[SearchResult]:
        results = self.store.search(self.embedder.embed_query(question), self.top_k)
        return [r for r in results if r.score >= self.min_score]   # drop weak matches

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Please type a question.")
        results = self.retrieve(question)
        if not results:
            return Answer(text=NOT_FOUND, sources=[])    # no API call spent
        if self.generator is None:
            return Answer(text="Search-only mode: here are the most relevant passages.",
                          sources=results)
        text = self.generator.generate(question, results)
        return Answer(text=text, sources=results, cited=find_citations(text, len(results)))