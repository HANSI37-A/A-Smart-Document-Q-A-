"""Find tables and figures inside a PDF and turn them into searchable chunks.

Tables become Markdown text (so they embed and read well). Figures (images and
charts) are rendered to PNG; a vision model (rag/vision.py) then describes them
in words, and that description is what gets indexed."""

import io
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import pypdfium2

from rag.models import Chunk
from rag.vision import DECORATIVE, VisionDescriber

RENDER_SCALE = 2          # 2x the PDF's 72 dpi: sharp enough to read chart labels
MIN_IMAGE_SIZE = 80       # points; smaller images are icons or logos
CHART_MIN_CURVES = 8      # a page with this many curved paths probably has a chart
CHART_MIN_SHAPES = 10     # ...or this many rects/lines/curves (bars, axes), if no table
MAX_TABLE_CHARS = 1500    # longer tables are split by rows so each part embeds well


@dataclass
class TableBlock:
    page: int
    index: int
    rows: list[list[str]]


@dataclass
class Figure:
    page: int
    index: int
    png: bytes


# ---------------- tables ----------------

def _cell(value) -> str:
    return " ".join(str(value or "").split()).replace("|", "\\|")


def table_to_markdown(rows: list[list[str]]) -> str:
    """First row is the header. Empty rows are dropped; rows are padded to equal width."""
    rows = [[_cell(c) for c in row] for row in rows if any(_cell(c) for c in row)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def split_markdown_table(markdown: str, max_chars: int = MAX_TABLE_CHARS) -> list[str]:
    """Split a long table by rows, repeating the header in every part."""
    lines = markdown.splitlines()
    header, body = lines[:2], lines[2:]
    parts, current = [], []
    for line in body:
        if current and len("\n".join(header + current + [line])) > max_chars:
            parts.append("\n".join(header + current))
            current = []
        current.append(line)
    parts.append("\n".join(header + current))
    return parts


def extract_tables(pdf: pdfplumber.PDF) -> list[TableBlock]:
    blocks = []
    for page in pdf.pages:
        for i, rows in enumerate(page.extract_tables()):
            if len(rows) >= 2 and table_to_markdown(rows):
                blocks.append(TableBlock(page=page.page_number, index=i, rows=rows))
    return blocks


def table_chunks(tables: list[TableBlock], doc_id: str, source: str) -> list[Chunk]:
    chunks = []
    for t in tables:
        parts = split_markdown_table(table_to_markdown(t.rows))
        for j, part in enumerate(parts):
            suffix = f".{j}" if len(parts) > 1 else ""
            chunks.append(Chunk(
                id=f"{doc_id}-p{t.page}-t{t.index}{suffix}",
                text=f"Table on page {t.page}:\n{part}",
                source=source, page=t.page, doc_id=doc_id, kind="table",
            ))
    return chunks


# ---------------- figures ----------------

def _to_png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _looks_like_chart(page) -> bool:
    if len(page.curves) >= CHART_MIN_CURVES:
        return True
    shapes = len(page.rects) + len(page.lines) + len(page.curves)
    return shapes >= CHART_MIN_SHAPES and not page.find_tables()


def extract_figures(path: Path, pdf: pdfplumber.PDF) -> list[Figure]:
    """Embedded images are cropped out; pages drawn as charts are rendered whole."""
    doc = pypdfium2.PdfDocument(str(path))
    figures = []
    try:
        for page in pdf.pages:
            candidates = [i for i in page.images
                          if i["x1"] - i["x0"] >= MIN_IMAGE_SIZE
                          and i["bottom"] - i["top"] >= MIN_IMAGE_SIZE]
            chart = not page.images and _looks_like_chart(page)
            if not candidates and not chart:
                continue

            rendered = doc[page.page_number - 1].render(scale=RENDER_SCALE).to_pil()
            if chart:
                figures.append(Figure(page.page_number, 0, _to_png(rendered)))
            s = RENDER_SCALE
            for index, img in enumerate(candidates):
                box = (max(0, img["x0"] * s), max(0, img["top"] * s),
                       min(rendered.width, img["x1"] * s), min(rendered.height, img["bottom"] * s))
                figures.append(Figure(page.page_number, index, _to_png(rendered.crop(box))))
    finally:
        doc.close()
    return figures


def figure_id(doc_id: str, figure: Figure) -> str:
    return f"{doc_id}-p{figure.page}-f{figure.index}"


def figure_chunks(figures: list[Figure], describer: VisionDescriber, doc_id: str, source: str,
                  known_ids: set[str] = frozenset()) -> list[Chunk]:
    """Describe each figure. Ids already in the index are skipped, so re-uploading a
    PDF costs no vision calls. A failed or decorative figure never stops the rest."""
    chunks = []
    for f in figures:
        id_ = figure_id(doc_id, f)
        if id_ in known_ids:
            continue
        try:
            description = describer.describe(f.png)
        except Exception:
            continue
        if not description or description.upper().startswith(DECORATIVE):
            continue
        chunks.append(Chunk(id=id_, text=f"Figure on page {f.page}: {description}",
                            source=source, page=f.page, doc_id=doc_id, kind="figure"))
    return chunks
