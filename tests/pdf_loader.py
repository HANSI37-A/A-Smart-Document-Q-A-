import pytest
from pypdf import PdfWriter

from rag.pdf_loader import NoTextError, clean_text, load_pdf


def test_clean_text_collapses_whitespace():
    assert clean_text("Hello   world\n\n again  ") == "Hello world again"


def test_clean_text_rejoins_hyphenated_words():
    assert clean_text("annual em-\nployees leave") == "annual employees leave"


def test_load_pdf_returns_one_page_per_page(make_pdf):
    path = make_pdf(["First page text", "Second page text"])
    pages = load_pdf(path)
    assert [p.page for p in pages] == [1, 2]
    assert pages[1].text == "Second page text"
    assert pages[0].source == "test.pdf"


def test_load_pdf_uses_custom_source_name(make_pdf):
    pages = load_pdf(make_pdf(["Hello"]), source_name="Policy.pdf")
    assert pages[0].source == "Policy.pdf"


def test_load_pdf_skips_empty_pages(make_pdf):
    pages = load_pdf(make_pdf(["Has text", "", "More text"]))
    assert [p.page for p in pages] == [1, 3]


def test_pdf_without_text_raises(tmp_path):
    path = tmp_path / "scanned.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with open(path, "wb") as f:
        writer.write(f)
    with pytest.raises(NoTextError, match="OCR"):
        load_pdf(path)