"""Connects everything: ingest PDFs and answer questions."""

import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pdfplumber

from rag.chunker import chunk_pages
from rag.embeddings import Embedder
from rag.generator import NOT_FOUND, find_citations, is_not_found
from rag.image_selector import ImageSelector
from rag.memory import QuestionRewriter
from rag.models import Answer, SearchResult, Turn
from rag.multimodal import extract_figures, extract_tables, figure_chunks, figure_id, table_chunks
from rag.pdf_loader import NoTextError, load_pdf
from rag.segmenter import RegionSegmenter
from rag.vector_store import VectorStore
from rag.vision import VisionDescriber
from rag.web_search import WebAnswer, WebSearchError

log = logging.getLogger(__name__)

# shown above every answer that came from the web (also spoken, and kept in chat history)
WEB_PREFIX = "*Not found in your documents — here is what a Google search found:*\n\n"


class Generator(Protocol):
    def generate(self, question: str, results: list[SearchResult],
                 history: list[Turn] = (), images: list[SearchResult] = ()) -> str: ...


class WebSearcher(Protocol):
    def search(self, question: str, history: list[Turn] = ()) -> WebAnswer: ...


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
                 overlap: int = 100, describer: VisionDescriber | None = None,
                 segmenter: RegionSegmenter | None = None,
                 rewriter: QuestionRewriter | None = None,
                 image_selector: ImageSelector | None = None,
                 figures_dir: str | Path | None = None,
                 web_searcher: WebSearcher | None = None):
        self.store = store
        self.embedder = embedder
        self.generator = generator   # None = search-only mode
        self.top_k = top_k
        self.min_score = min_score
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.describer = describer   # None = tables are indexed, figures are not
        self.segmenter = segmenter   # None = crop the boxes pdfplumber finds, no SAM
        self.rewriter = rewriter     # None = follow-ups are searched as typed
        self.image_selector = image_selector   # None = never attach images
        self.figures_dir = Path(figures_dir) if figures_dir else None   # None = don't keep crops
        self.web_searcher = web_searcher   # None = answer only from the documents

    # ---------------- documents ----------------

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
                found = extract_figures(path, pdf, self.segmenter) if self.describer else []
        except Exception:   # an odd PDF must never block plain-text indexing
            return [], []
        if not found:
            return tables, []
        known = self.store.existing_ids([figure_id(doc_id, f) for f in found])
        return tables, figure_chunks(found, self.describer, doc_id, name, known,
                                     figures_dir=self.figures_dir)

    def delete_document(self, doc_id: str) -> None:
        """Remove a document's chunks and its saved figure images."""
        self.store.delete_document(doc_id)
        if self.figures_dir:
            shutil.rmtree(self.figures_dir / doc_id, ignore_errors=True)

    # ---------------- questions ----------------

    def retrieve(self, question: str, query_vec: list[float] | None = None) -> list[SearchResult]:
        if query_vec is None:
            query_vec = self.embedder.embed_query(question)
        results = self.store.search(query_vec, self.top_k)
        return [r for r in results if r.score >= self.min_score]   # drop weak matches

    def standalone_question(self, question: str, history: list[Turn]) -> str:
        """Turn a follow-up into a question that can be searched on its own."""
        if not history or self.rewriter is None:
            return question
        try:
            return self.rewriter.rewrite(history, question).strip() or question
        except Exception:   # memory is a bonus: a failed rewrite must not fail the answer
            return question

    def ask(self, question: str, history: list[Turn] = ()) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Please type a question.")
        history = list(history)
        search_query = self.standalone_question(question, history)
        query_vec = self.embedder.embed_query(search_query)
        results = self.retrieve(search_query, query_vec)
        if not results:   # nothing relevant in the documents: no model call
            not_found = Answer(text=NOT_FOUND, sources=[], search_query=search_query)
            return self._web_answer(search_query, history, not_found)

        images = []
        if self.image_selector is not None:
            images = self.image_selector.select(search_query, query_vec, results,
                                                use_judge=self.generator is not None)
        # an image found through a nearby paragraph also becomes a numbered source
        known = {r.chunk.id for r in results}
        sources = results + [img for img in images if img.chunk.id not in known]

        if self.generator is None:
            return Answer(text="Search-only mode: here are the most relevant passages.",
                          sources=sources, images=images, search_query=search_query)
        text = self.generator.generate(question, sources, history=history, images=images)
        answer = Answer(text=text, sources=sources, cited=find_citations(text, len(sources)),
                        images=images, search_query=search_query)
        if is_not_found(text):   # the passages were related but did not hold the answer
            return self._web_answer(search_query, history, answer)
        return answer

    def _web_answer(self, search_query: str, history: list[Turn], fallback: Answer) -> Answer:
        """Answer from a Google search; `fallback` (the "not found" answer) if that fails."""
        if self.web_searcher is None:
            return fallback
        try:
            found = self.web_searcher.search(search_query, history=history)
        except WebSearchError as e:
            fallback.text += f"\n\n_Web search was not available: {e}_"
            return fallback
        except Exception:   # the web is a bonus: it must never break the answer
            log.exception("Web search failed unexpectedly")
            return fallback
        return Answer(text=WEB_PREFIX + found.text, sources=found.sources,
                      cited=find_citations(found.text, len(found.sources)),
                      search_query=search_query, web=True)
