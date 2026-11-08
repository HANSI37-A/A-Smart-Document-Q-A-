"""Find tables and figures inside a PDF and turn them into searchable chunks.

Tables become Markdown text (so they embed and read well). Figures (images and
charts) are cut out of the rendered page (rag/segmenter.py, SAM), obvious
decoration such as logos is dropped, a vision model (rag/vision.py) describes the
rest in words, and that description is indexed. The cropped picture is saved, so
it can be shown to the answer model and the user when a question needs it."""

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import pypdfium2
from PIL import Image, ImageStat

from rag.models import Chunk
from rag.segmenter import Box, PassThroughSegmenter, RegionSegmenter
from rag.vision import DECORATIVE, MAX_CONTEXT_CHARS, VisionDescriber

RENDER_SCALE = 2          # 2x the PDF's 72 dpi: sharp enough to read chart labels
MIN_IMAGE_SIZE = 80       # points; smaller images are icons or logos
CHART_MIN_CURVES = 8      # a page with this many curved paths probably has a chart
CHART_MIN_SHAPES = 10     # ...or this many rects/lines/curves (bars, axes), if no table
MAX_TABLE_CHARS = 1500    # longer tables are split by rows so each part embeds well
CHART_LABEL_REACH = 30    # points around a chart's drawing where axis labels still belong to it
CROP_MARGIN = 0.015      # margin around a cropped figure, as a share of the page size
EDGE_BAND = 0.1           # top/bottom 10% of the page: where headers, footers and logos live
SMALL_FIGURE = 0.08       # figures under 8% of the page can be logos
MIN_STDDEV = 6.0          # an almost single-colour crop carries no information


@dataclass
class TableBlock:
    page: int
    index: int
    rows: list[list[str]]


@dataclass
class Figure:
    page: int
    index: int
    image: Image.Image                     # the cropped figure
    bbox: Box | None = None                # where it was on the rendered page (pixels)
    page_size: tuple[int, int] = (0, 0)    # rendered page width, height
    page_text: str = ""                    # page text, context for the vision model


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

def _looks_like_chart(page) -> bool:
    if len(page.curves) >= CHART_MIN_CURVES:
        return True
    shapes = len(page.rects) + len(page.lines) + len(page.curves)
    return shapes >= CHART_MIN_SHAPES and not page.find_tables()


def _is_vector_chart(page) -> bool:
    """A chart drawn with shapes (not an embedded picture): its labels are page text."""
    big_image = any(i["x1"] - i["x0"] >= MIN_IMAGE_SIZE and i["bottom"] - i["top"] >= MIN_IMAGE_SIZE
                    for i in page.images)
    return not big_image and _looks_like_chart(page)


def _obj_box(obj) -> Box:
    return Box(float(obj["x0"]), float(obj["top"]), float(obj["x1"]), float(obj["bottom"]))


def page_hint_boxes(page, scale: float = RENDER_SCALE) -> list[Box]:
    """Where pdfplumber thinks the figures are, in rendered-page pixels.

    Embedded images big enough to matter, plus (on a page without such an image that
    looks like a chart) the area covered by its drawn shapes. A small logo in the header
    must not stop a chart on the same page from being found. No hints = no figure here."""
    hints = [_obj_box(i).scaled(scale) for i in page.images
             if i["x1"] - i["x0"] >= MIN_IMAGE_SIZE and i["bottom"] - i["top"] >= MIN_IMAGE_SIZE]
    if _is_vector_chart(page):
        drawn = [_obj_box(o) for o in page.curves + page.rects + page.lines]
        area = drawn[0]
        for b in drawn[1:]:
            area = area.union(b)
        hints.append(_with_nearby_labels(area, page).scaled(scale))
    return hints


def _with_nearby_labels(area: Box, page) -> Box:
    """Grow a chart box to include the axis labels, ticks and legend just outside it.

    Those are text, not drawn shapes, so the shapes alone would cut them off."""
    reach = CHART_LABEL_REACH
    zone = Box(area.x0 - reach, area.y0 - reach, area.x1 + reach, area.y1 + reach)
    for w in page.extract_words():
        word = Box(float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"]))
        if zone.contains(word):
            area = area.union(word)
    return area


def extract_figures(path: Path, pdf: pdfplumber.PDF,
                    segmenter: RegionSegmenter | None = None) -> list[Figure]:
    """Cut the figures out of every page that holds a photo or a chart.

    The page is rendered once; the segmenter (SAM) finds each figure's outline and the
    figure is cropped with a small margin. The page text travels with each crop, so the
    vision model still knows the slide title and captions."""
    segmenter = segmenter or PassThroughSegmenter()
    doc = pypdfium2.PdfDocument(str(path))
    figures = []
    try:
        for page in pdf.pages:
            hints = page_hint_boxes(page)
            if not hints:
                continue
            rendered = doc[page.page_number - 1].render(scale=RENDER_SCALE).to_pil()
            try:
                boxes = segmenter.segment(rendered, hints)
            except Exception:   # a SAM failure must not lose the figure
                boxes = PassThroughSegmenter().segment(rendered, hints)
            if _is_vector_chart(page):
                # SAM outlines the lines and bars, but the axis labels are text, so give
                # them back to the box; a chart without its labels cannot be read.
                boxes = [_with_nearby_labels(b.scaled(1 / RENDER_SCALE), page).scaled(RENDER_SCALE)
                         for b in boxes]
            text = (page.extract_text() or "")[:MAX_CONTEXT_CHARS]
            pad = CROP_MARGIN * max(rendered.size)
            for i, box in enumerate(boxes):
                box = box.padded(pad, rendered.width, rendered.height)
                crop = rendered.crop((int(box.x0), int(box.y0),
                                      math.ceil(box.x1), math.ceil(box.y1)))
                figures.append(Figure(page.page_number, i, crop, box, rendered.size, text))
    finally:
        doc.close()
    return figures


def figure_id(doc_id: str, figure: Figure) -> str:
    return f"{doc_id}-p{figure.page}-f{figure.index}"


# ---------------- decorative filter (free, before any vision call) ----------------

def average_hash(image: Image.Image) -> int:
    """64-bit fingerprint of how an image looks; the same logo gives the same hash."""
    pixels = list(image.convert("L").resize((8, 8)).getdata())
    mean = sum(pixels) / len(pixels)
    return sum(1 << i for i, p in enumerate(pixels) if p > mean)


def is_probably_decorative(figure: Figure, repeated_on_pages: int = 1) -> bool:
    """Logos, letterheads and blank boxes, recognised without paying for a vision call."""
    page_w, page_h = figure.page_size
    share = figure.bbox.area / (page_w * page_h) if figure.bbox and page_w and page_h else 0.0
    small = 0 < share < SMALL_FIGURE
    if small and figure.bbox:
        middle = (figure.bbox.y0 + figure.bbox.y1) / 2
        if middle < EDGE_BAND * page_h or middle > (1 - EDGE_BAND) * page_h:
            return True                       # small thing in the header or footer
    if small and repeated_on_pages >= 2:
        return True                           # the same small image on several pages: a logo
    return ImageStat.Stat(figure.image.convert("L")).stddev[0] < MIN_STDDEV   # nearly blank


def save_figure(figure: Figure, figures_dir: Path, doc_id: str) -> str:
    folder = Path(figures_dir) / doc_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"p{figure.page}-f{figure.index}.png"
    figure.image.convert("RGB").save(path, format="PNG")
    return str(path)


def figure_chunks(figures: list[Figure], describer: VisionDescriber, doc_id: str, source: str,
                  known_ids: set[str] = frozenset(),
                  figures_dir: Path | None = None) -> list[Chunk]:
    """Describe each useful figure and keep its picture.

    Ids already in the index are skipped, so re-uploading a PDF costs no vision calls.
    Decorative figures are dropped before (cheap checks) or by (DECORATIVE) the vision
    model. A failed figure never stops the rest."""
    hashes = [average_hash(f.image) for f in figures]
    pages_per_hash = defaultdict(set)
    for f, h in zip(figures, hashes):
        pages_per_hash[h].add(f.page)

    chunks = []
    for f, h in zip(figures, hashes):
        id_ = figure_id(doc_id, f)
        if id_ in known_ids or is_probably_decorative(f, len(pages_per_hash[h])):
            continue
        try:
            description = describer.describe(f.image, f.page_text)
        except Exception:
            continue
        if not description or description.upper().startswith(DECORATIVE):
            continue
        image_path = save_figure(f, figures_dir, doc_id) if figures_dir else ""
        chunks.append(Chunk(id=id_, text=f"Figure on page {f.page}: {description}",
                            source=source, page=f.page, doc_id=doc_id, kind="figure",
                            image_path=image_path))
    return chunks
