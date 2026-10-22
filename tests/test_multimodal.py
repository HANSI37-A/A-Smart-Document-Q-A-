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
from rag.vision import ClaudeVisionDescriber, GeminiVisionDescriber
from tests.conftest import FakeGenerator

REVENUE = [["Region", "Revenue", "Growth"], ["North", "120", "5%"], ["South", "80", "-2%"]]


class FakeDescriber:
    def __init__(self, reply="Bar chart of quarterly sales, rising from Q1 to Q4."):
        self.reply = reply
        self.calls = 0

    def describe(self, png):
        self.calls += 1
        assert png.startswith(b"\x89PNG")   # really is a PNG
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
    assert all(f.png.startswith(b"\x89PNG") for f in figures)


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
        def describe(self, png):
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


# ---------------- vision describers (real classes, fake clients) ----------------

def test_gemini_describer_sends_image_and_prompt():
    from types import SimpleNamespace
    sent = {}

    def generate_content(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(text="  A line chart.  ")

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    assert GeminiVisionDescriber(client).describe(b"\x89PNGdata") == "A line chart."
    image_part = sent["contents"][0]
    assert image_part.inline_data.mime_type == "image/png"
    assert image_part.inline_data.data == b"\x89PNGdata"


def test_claude_describer_sends_base64_image():
    from types import SimpleNamespace
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="A pie chart.")])

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    assert ClaudeVisionDescriber(client).describe(b"\x89PNGdata") == "A pie chart."
    image_block = sent["messages"][0]["content"][0]
    assert image_block["source"]["media_type"] == "image/png"


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
