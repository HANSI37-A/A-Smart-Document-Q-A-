"""Connects everything: ingest PDFs and answer questions."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pdfplumber

from rag.chunker import chunk_pages
from rag.embeddings import Embedder
from rag.generator import find_citations
from rag.models import Answer, SearchResult
from rag.multimodal import extract_figures, extract_tables, figure_chunks, figure_id, table_chunks
from rag.pdf_loader import NoTextError, load_pdf
from rag.vector_store import VectorStore
from rag.vision import VisionDescriber

NOT_FOUND = "I couldn't find that in the documents."


class Generator(Protocol):
    def generate(self, question: str, results: list[SearchResult]) -> str: ...


@dataclass
class IngestReport:
    source: str
    doc_id: str
    pages: int
    chunks: int
    tables: int = 0
    figures: int = 0


def file_fingerprint(path: Path) -> str:
    """Short id based only on the file's bytes, not its name."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


class DocQA:
    def __init__(self, store: VectorStore, embedder: Embedder, generator: Generator | None,
                 top_k: int = 5, min_score: float = 0.5, chunk_size: int = 500,
                 overlap: int = 100, describer: VisionDescriber | None = None):
        self.store = store
        self.embedder = embedder
        self.generator = generator   # None = search-only mode
        self.top_k = top_k
        self.min_score = min_score
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.describer = describer   # None = tables are indexed, figures are not

    def ingest_pdf(self, path: str | Path, source_name: str | None = None) -> IngestReport:
        path = Path(path)
        name = source_name or path.name
        doc_id = file_fingerprint(path)
        try:
            pages = load_pdf(path, name)
        except NoTextError:
            pages = []                       # may still hold tables or figures
        chunks = chunk_pages(pages, doc_id, self.chunk_size, self.overlap)
        tables, figures = self._visual_chunks(path, doc_id, name)
        if not (chunks or tables or figures):
            raise NoTextError(
                f"No text found in {name}. If it is a scanned document, run OCR on it first."
            )
        everything = chunks + tables + figures
        embeddings = self.embedder.embed_documents([c.text for c in everything])
        self.store.add(everything, embeddings)
        return IngestReport(name, doc_id, len(pages), len(chunks), len(tables), len(figures))

    def _visual_chunks(self, path: Path, doc_id: str, name: str):
        """Tables (always) and figure descriptions (if a describer is set)."""
        try:
            with pdfplumber.open(path) as pdf:
                tables = table_chunks(extract_tables(pdf), doc_id, name)
                found = extract_figures(path, pdf) if self.describer else []
        except Exception:   # an odd PDF must never block plain-text indexing
            return [], []
        if not found:
            return tables, []
        known = self.store.existing_ids([figure_id(doc_id, f) for f in found])
        return tables, figure_chunks(found, self.describer, doc_id, name, known)

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