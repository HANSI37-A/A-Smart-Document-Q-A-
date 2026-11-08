"""Tests for tables and figures: extraction, description and indexing.

PDFs are built on the fly with reportlab, and the vision model is a fake,
so no API key, internet or embedding model is needed."""
import pdfplumber
import pytest
from PIL import Image, ImageDraw
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

from rag.multimodal import (
    extract_figures, extract_tables, figure_chunks, split_markdown_table, table_chunks,
    table_to_markdown,
)
from rag.pipeline import DocQA
from rag.vector_store import VectorStore
from rag.vision import OpenRouterVisionDescriber
from tests.conftest import FakeGenerator

REVENUE = [["Region", "Revenue", "Growth"], ["North", "120", "5%"], ["South", "80", "-2%"]]


class FakeDescriber:
    def __init__(self, reply="Bar chart of quarterly sales, rising from Q1 to Q4."):
        self.reply = reply
        self.calls = 0
        self.contexts = []

    def describe(self, image, context=""):
        self.calls += 1
        self.contexts.append(context)
        assert image.width > 100 and image.height > 100   # a real cropped figure
        return self.reply


@pytest.fixture
def rich_pdf(tmp_path):
    """Page 1: text. Page 2: a table. Page 3: an embedded image. Page 4: a bar chart."""
    path = tmp_path / "report.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.drawString(50, 800, "Annual report introduction text.")
    c.showPage()

    c.drawString(50, 800, "Revenue by region")
    table = Table(REVENUE)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    table.wrapOn(c, 400, 200)
    table.drawOn(c, 50, 650)
    c.showPage()

    img = Image.new("RGB", (300, 200), "white")
    ImageDraw.Draw(img).rectangle((20, 20, 280, 180), fill="blue")
    img_path = tmp_path / "photo.png"
    img.save(img_path)
    c.drawString(50, 800, "Photo page")
    c.drawImage(str(img_path), 50, 500, width=300, height=200)
    c.showPage()

    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 30, 30, 300, 200
    chart.data = [[10, 20, 35, 50]]
    drawing = Drawing(400, 300)
    drawing.add(chart)
    c.drawString(50, 800, "Sales chart page")
    renderPDF.draw(drawing, c, 50, 450)
    c.showPage()
    c.save()
    return path


# ---------------- tables ----------------

def test_table_to_markdown():
    md = table_to_markdown(REVENUE)
    assert md.splitlines() == [
        "| Region | Revenue | Growth |",
        "| --- | --- | --- |",
        "| North | 120 | 5% |",
        "| South | 80 | -2% |",
    ]


def test_table_to_markdown_cleans_cells():
    md = table_to_markdown([["A", "B"], [None, "x|y\nz"], [None, None], ["only"]])
    assert md.splitlines()[2:] == ["|  | x\\|y z |", "| only |  |"]   # blank row dropped, padded


def test_long_tables_are_split_with_header_repeated():
    rows = [["Name", "Value"]] + [[f"item{i}", str(i)] for i in range(50)]
    parts = split_markdown_table(table_to_markdown(rows), max_chars=200)
    assert len(parts) > 1
    for part in parts:
        assert part.startswith("| Name | Value |\n| --- | --- |")
        assert len(part) <= 200


def test_extract_tables_from_pdf(rich_pdf):
    with pdfplumber.open(rich_pdf) as pdf:
        tables = extract_tables(pdf)
    assert [t.page for t in tables] == [2]
    assert tables[0].rows[1][:2] == ["North", "120"]


def test_table_chunks_have_stable_ids_and_metadata(rich_pdf):
    with pdfplumber.open(rich_pdf) as pdf:
        tables = extract_tables(pdf)
    chunks = table_chunks(tables, "doc1", "report.pdf")
    assert [c.id for c in chunks] == ["doc1-p2-t0"]
    assert chunks[0].kind == "table" and chunks[0].page == 2 and chunks[0].source == "report.pdf"
    assert "| North | 120 | 5% |" in chunks[0].text


# ---------------- figures ----------------

def test_extract_figures_finds_image_and_chart_only(rich_pdf):
    with pdfplumber.open(rich_pdf) as pdf:
        figures = extract_figures(rich_pdf, pdf)
    assert sorted(f.page for f in figures) == [3, 4]     # not the text or table pages
    assert all(f.image.width > 100 for f in figures)


def test_figure_chunks_use_the_describer(rich_pdf):
    with pdfplumber.open(rich_pdf) as pdf:
        figures = extract_figures(rich_pdf, pdf)
    chunks = figure_chunks(figures, FakeDescriber(), "doc1", "report.pdf")
    assert {c.id for c in chunks} == {"doc1-p3-f0", "doc1-p4-f0"}
    assert all(c.kind == "figure" for c in chunks)
    assert chunks[0].text.startswith("Figure on page")
    assert "Bar chart of quarterly sales" in chunks[0].text


def test_decorative_empty_and_failing_figures_are_skipped(rich_pdf):
    with pdfplumber.open(rich_pdf) as pdf:
        figures = extract_figures(rich_pdf, pdf)

    class Failing:
        def describe(self, image, context=""):
            raise RuntimeError("quota exceeded")

    assert figure_chunks(figures, FakeDescriber("DECORATIVE"), "d", "r.pdf") == []
    assert figure_chunks(figures, FakeDescriber(""), "d", "r.pdf") == []
    assert figure_chunks(figures, Failing(), "d", "r.pdf") == []


def test_known_figures_are_not_described_again(rich_pdf):
    with pdfplumber.open(rich_pdf) as pdf:
        figures = extract_figures(rich_pdf, pdf)
    describer = FakeDescriber()
    figure_chunks(figures, describer, "d", "r.pdf", known_ids={"d-p3-f0", "d-p4-f0"})
    assert describer.calls == 0


# ---------------- vision describer (real class, fake client) ----------------

def test_openrouter_describer_sends_image_as_data_url():
    from types import SimpleNamespace
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        message = SimpleNamespace(content="  A line chart.  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert OpenRouterVisionDescriber(client).describe(Image.new("RGB", (4, 4))) == "A line chart."
    parts = sent["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[1]["type"] == "text"


# ---------------- pipeline ----------------

@pytest.fixture
def qa(tmp_path, fake_embedder):
    return DocQA(VectorStore(str(tmp_path / "db")), fake_embedder, FakeGenerator(),
                 min_score=0.1, describer=FakeDescriber())


def test_ingest_indexes_text_tables_and_figures(qa, rich_pdf):
    report = qa.ingest_pdf(rich_pdf)
    assert (report.pages, report.tables, report.figures) == (4, 1, 2)
    assert qa.store.count() == report.chunks + 1 + 2


def test_table_question_retrieves_the_table_page(qa, rich_pdf):
    qa.ingest_pdf(rich_pdf)
    hit = next(r for r in qa.retrieve("revenue north south region growth")
               if r.chunk.kind == "table")
    assert hit.chunk.page == 2


def test_chart_question_retrieves_the_figure(qa, rich_pdf):
    qa.ingest_pdf(rich_pdf)
    hit = next(r for r in qa.retrieve("bar chart of quarterly sales rising")
               if r.chunk.kind == "figure")
    assert hit.chunk.page in (3, 4)
    answer = qa.ask("bar chart of quarterly sales rising")
    prompt_sources = qa.generator.calls[0][1]
    assert any(r.chunk.kind == "figure" for r in prompt_sources)
    assert answer.sources == prompt_sources


def test_reingesting_does_not_call_vision_again(qa, rich_pdf):
    qa.ingest_pdf(rich_pdf)
    first_calls = qa.describer.calls
    count = qa.store.count()
    qa.ingest_pdf(rich_pdf)
    assert qa.describer.calls == first_calls
    assert qa.store.count() == count


def test_without_describer_tables_are_indexed_but_figures_are_not(tmp_path, fake_embedder,
                                                                  rich_pdf):
    qa = DocQA(VectorStore(str(tmp_path / "db")), fake_embedder, None, describer=None)
    report = qa.ingest_pdf(rich_pdf)
    assert (report.tables, report.figures) == (1, 0)


def test_plain_text_pdf_is_unchanged(qa, make_pdf):
    report = qa.ingest_pdf(make_pdf(["Just some plain text on one page."]))
    assert (report.tables, report.figures) == (0, 0)
    assert qa.describer.calls == 0


def test_prompt_labels_tables_and_figures():
    from rag.generator import build_prompt
    from rag.models import Chunk, SearchResult
    text = build_prompt("Q?", [
        SearchResult(Chunk("a", "t", "h.pdf", 1, "d"), 0.9),
        SearchResult(Chunk("b", "| x |", "h.pdf", 2, "d", kind="table"), 0.8),
    ])
    assert '<source id="1" file="h.pdf" page="1">' in text      # plain text unchanged
    assert '<source id="2" file="h.pdf" page="2" type="table">' in text


# ---------------- SAM-style cropping, decorative filter, saved images ----------------

from rag.multimodal import (  # noqa: E402
    Figure, average_hash, is_probably_decorative, page_hint_boxes, save_figure,
)
from rag.segmenter import Box  # noqa: E402


class FixedSegmenter:
    """Pretends to be SAM and returns the boxes it was told to."""
    def __init__(self, boxes):
        self.boxes, self.seen = boxes, []

    def segment(self, image, hints):
        self.seen.append((image.size, hints))
        return self.boxes


def test_hint_boxes_cover_the_image_and_the_chart(rich_pdf):
    with pdfplumber.open(rich_pdf) as pdf:
        assert page_hint_boxes(pdf.pages[0]) == []          # text page: no figure
        assert page_hint_boxes(pdf.pages[1]) == []          # table page: no figure
        image_hint, = page_hint_boxes(pdf.pages[2])
        chart_hint, = page_hint_boxes(pdf.pages[3])
    assert image_hint.width == pytest.approx(300 * 2, abs=2)    # 300 pt image at 2x
    assert chart_hint.width > 200 and chart_hint.height > 200


def test_figures_are_cropped_to_what_the_segmenter_returns(rich_pdf):
    seg = FixedSegmenter([Box(200, 300, 700, 700)])
    with pdfplumber.open(rich_pdf) as pdf:
        figures = extract_figures(rich_pdf, pdf, seg)
    assert sorted(f.page for f in figures) == [3, 4]
    assert all(hints for _, hints in seg.seen)               # SAM got the pdfplumber hints
    for f in figures:
        assert 500 < f.image.width < 560 and 400 < f.image.height < 460   # box + margin
        assert f.page_size[0] > f.image.width
    assert "Photo page" in figures[0].page_text


def test_a_crashing_segmenter_falls_back_to_the_hint_crop(rich_pdf):
    class Crashing:
        def segment(self, image, hints):
            raise RuntimeError("out of memory")

    with pdfplumber.open(rich_pdf) as pdf:
        figures = extract_figures(rich_pdf, pdf, Crashing())
    assert sorted(f.page for f in figures) == [3, 4]


def test_figure_crops_are_saved_and_recorded_on_the_chunk(rich_pdf, tmp_path):
    with pdfplumber.open(rich_pdf) as pdf:
        figures = extract_figures(rich_pdf, pdf)
    chunks = figure_chunks(figures, FakeDescriber(), "doc1", "r.pdf",
                           figures_dir=tmp_path / "figs")
    assert len(chunks) == 2
    for c in chunks:
        assert (tmp_path / "figs" / "doc1" / f"p{c.page}-f0.png").is_file()
        assert c.image_path.endswith(f"p{c.page}-f0.png")


def test_the_describer_gets_the_page_text_as_context(rich_pdf):
    with pdfplumber.open(rich_pdf) as pdf:
        figures = extract_figures(rich_pdf, pdf)
    describer = FakeDescriber()
    figure_chunks(figures, describer, "d", "r.pdf")
    assert any("Sales chart page" in c for c in describer.contexts)


def test_image_path_survives_the_vector_store(rich_pdf, tmp_path, fake_embedder):
    qa = DocQA(VectorStore(str(tmp_path / "db")), fake_embedder, FakeGenerator(),
               min_score=0.1, describer=FakeDescriber(), figures_dir=tmp_path / "figs")
    qa.ingest_pdf(rich_pdf)
    hit = next(r for r in qa.retrieve("bar chart of quarterly sales rising")
               if r.chunk.kind == "figure")
    assert hit.chunk.image_path.startswith(str(tmp_path / "figs"))
    text_hit = next(r for r in qa.retrieve("annual report introduction") if r.chunk.kind == "text")
    assert text_hit.chunk.image_path == ""


def _figure(page=1, bbox=Box(100, 400, 700, 900), color="white", draw=True, size=(600, 500)):
    img = Image.new("RGB", size, color)
    if draw:
        d = ImageDraw.Draw(img)
        top = size[1] // 5
        for x in range(0, size[0], 40):
            d.rectangle((x, top, x + 20, size[1] - 1), fill="black")
    return Figure(page, 0, img, bbox, (1000, 1400), "")


def test_logo_in_the_header_is_decorative_but_a_header_sized_chart_is_not():
    logo = _figure(bbox=Box(50, 20, 250, 100), size=(200, 80))
    assert is_probably_decorative(logo)
    big_chart_at_top = _figure(bbox=Box(50, 20, 950, 600))
    assert not is_probably_decorative(big_chart_at_top)


def test_same_small_image_on_several_pages_is_a_logo():
    small = _figure(bbox=Box(400, 600, 600, 700), size=(200, 100))
    assert not is_probably_decorative(small, repeated_on_pages=1)
    assert is_probably_decorative(small, repeated_on_pages=2)
    assert not is_probably_decorative(_figure(), repeated_on_pages=3)   # big, so not a logo


def test_blank_crop_is_decorative():
    assert is_probably_decorative(_figure(draw=False))


def test_average_hash_matches_identical_images_only():
    assert average_hash(_figure().image) == average_hash(_figure().image)
    assert average_hash(_figure().image) != average_hash(_figure(draw=False, color="black").image)


def test_logos_are_dropped_before_any_vision_call(tmp_path):
    def logo(page):
        return _figure(page=page, bbox=Box(50, 20, 250, 100), size=(200, 80))

    describer = FakeDescriber()
    chunks = figure_chunks([logo(1), _figure(page=2), logo(3)], describer, "d", "r.pdf",
                           figures_dir=tmp_path)
    assert describer.calls == 1                            # only the chart is paid for
    assert [c.page for c in chunks] == [2]


def test_save_figure_writes_a_png(tmp_path):
    path = save_figure(_figure(page=7), tmp_path / "figs", "docx")
    assert path.endswith("p7-f0.png")
    assert Image.open(path).size == (600, 500)


def test_a_small_header_logo_does_not_hide_a_chart_on_the_same_page(tmp_path):
    logo = Image.new("RGB", (120, 50), "navy")
    logo_path = tmp_path / "logo.png"
    logo.save(logo_path)
    path = tmp_path / "logo_and_chart.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.drawImage(str(logo_path), 50, 770, width=120, height=50)     # under 80 pt tall
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 30, 30, 300, 200
    chart.data = [[10, 20, 35, 50]]
    drawing = Drawing(400, 300)
    drawing.add(chart)
    renderPDF.draw(drawing, c, 50, 400)
    c.showPage()
    c.save()
    with pdfplumber.open(path) as pdf:
        assert pdf.pages[0].images                                  # the logo is really there
        hints = page_hint_boxes(pdf.pages[0])
    assert len(hints) == 1 and hints[0].width > 200 and hints[0].height > 200


def test_chart_box_includes_its_axis_labels(tmp_path):
    path = tmp_path / "labelled_chart.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 40, 30, 300, 200
    chart.data = [[10, 20, 35, 50]]
    chart.categoryAxis.categoryNames = ["Q1", "Q2", "Q3", "Q4"]
    drawing = Drawing(400, 300)
    drawing.add(chart)
    c.drawString(50, 800, "Far away title that is not part of the chart")
    renderPDF.draw(drawing, c, 50, 400)
    c.showPage()
    c.save()
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        hint, = page_hint_boxes(page, scale=1)
        words = {w["text"]: w for w in page.extract_words()}
    for label in ("Q1", "Q4", "50"):                    # axis labels sit outside the shapes
        w = words[label]
        assert hint.x0 <= w["x0"] and hint.x1 >= w["x1"], label
        assert hint.y0 <= w["top"] and hint.y1 >= w["bottom"], label
    assert hint.y0 > words["Far"]["bottom"]             # unrelated text stays outside


def test_axis_labels_stay_in_the_crop_even_if_the_segmenter_outlines_only_the_drawing(tmp_path):
    path = tmp_path / "labelled_chart.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 40, 30, 300, 200
    chart.data = [[10, 20, 35, 50]]
    chart.categoryAxis.categoryNames = ["Q1", "Q2", "Q3", "Q4"]
    drawing = Drawing(400, 300)
    drawing.add(chart)
    renderPDF.draw(drawing, c, 50, 400)
    c.showPage()
    c.save()
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        drawn = [(o["x0"], o["top"], o["x1"], o["bottom"])
                 for o in page.curves + page.rects + page.lines]
        tight = Box(min(d[0] for d in drawn) * 2, min(d[1] for d in drawn) * 2,
                    max(d[2] for d in drawn) * 2, max(d[3] for d in drawn) * 2)
        words = {w["text"]: w for w in page.extract_words()}
        figures = extract_figures(path, pdf, FixedSegmenter([tight]))
    crop_box = figures[0].bbox            # pixels at 2x
    for label in ("Q1", "Q4", "50"):
        w = words[label]
        assert crop_box.x0 <= w["x0"] * 2 and crop_box.x1 >= w["x1"] * 2, label
        assert crop_box.y0 <= w["top"] * 2 and crop_box.y1 >= w["bottom"] * 2, label
