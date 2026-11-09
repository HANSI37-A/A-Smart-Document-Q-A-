"""Tests for finding figure regions: the box filter, the no-SAM fallback, and real SAM."""
from pathlib import Path

import pdfplumber
import pytest
from PIL import Image

from rag.segmenter import Box, PassThroughSegmenter, SamSegmenter, filter_boxes

ROOT_DIR = Path(__file__).resolve().parent.parent
W, H = 1000, 1400   # a rendered page, in pixels


def test_box_geometry():
    a, b = Box(0, 0, 10, 10), Box(5, 5, 15, 15)
    assert a.area == 100
    assert a.intersection(b) == 25
    assert a.iou(b) == pytest.approx(25 / 175)
    assert Box(0, 0, 100, 100).contains(Box(10, 10, 20, 20))
    assert a.union(b) == Box(0, 0, 15, 15)
    assert Box(1, 2, 3, 4).scaled(2) == Box(2, 4, 6, 8)
    assert Box(5, 5, 95, 95).padded(10, 100, 100) == Box(0, 0, 100, 100)


def test_filter_drops_tiny_whole_page_and_thin_boxes():
    chart = Box(100, 300, 700, 800)
    boxes = [chart,
             Box(10, 10, 40, 40),        # icon: far below 1% of the page
             Box(0, 0, W, H),            # the whole page
             Box(50, 900, 950, 920)]     # a horizontal rule (45:1)
    assert filter_boxes(boxes, W, H) == [chart]


def test_filter_merges_overlapping_and_nested_boxes():
    outer = Box(100, 100, 600, 500)
    boxes = [outer, Box(150, 150, 400, 400),          # inside the chart (a legend)
             Box(110, 110, 610, 510)]                 # the same chart, slightly shifted
    merged = filter_boxes(boxes, W, H)
    assert len(merged) == 1
    assert merged[0].contains(outer)


def test_filter_caps_and_orders_top_to_bottom():
    boxes = [Box(100, y, 400, y + 150) for y in (1100, 100, 500, 300, 900, 700)]
    kept = filter_boxes(boxes, W, H, max_figures=4)
    assert len(kept) == 4
    assert [b.y0 for b in kept] == sorted(b.y0 for b in kept)


def test_pass_through_uses_hints_or_the_whole_page():
    page = Image.new("RGB", (W, H), "white")
    hint = Box(100, 300, 700, 800)
    assert PassThroughSegmenter().segment(page, [hint]) == [hint]
    assert PassThroughSegmenter().segment(page, []) == [Box(0, 0, W, H)]


def test_sam_keeps_the_hint_when_its_outline_is_much_smaller(monkeypatch):
    """SAM sometimes outlines only the bars of a chart; the crop must keep the labels."""
    sam = SamSegmenter.__new__(SamSegmenter)            # skip loading the real model
    hint_chart, hint_photo = Box(100, 100, 600, 500), Box(100, 700, 600, 1100)
    monkeypatch.setattr(sam, "_predict", lambda image, hints: [
        Box(300, 300, 400, 400),                        # tiny: keep the chart hint
        Box(110, 710, 590, 1090),                       # good: use SAM's tighter box
    ])
    boxes = sam.segment(Image.new("RGB", (W, H)), [hint_chart, hint_photo])
    assert boxes == [hint_chart, Box(110, 710, 590, 1090)]


@pytest.mark.slow
def test_real_mobile_sam_finds_the_embedded_image(tmp_path):
    pytest.importorskip("ultralytics")
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    from rag.multimodal import RENDER_SCALE, page_hint_boxes

    img = Image.new("RGB", (300, 200), "white")
    for x in range(0, 300, 20):                         # some structure for SAM to outline
        img.paste((30, 90, 200), (x, 40, x + 12, 180))
    img_path = tmp_path / "chart.png"
    img.save(img_path)
    pdf_path = tmp_path / "p.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    c.drawString(50, 800, "A page with a picture")
    c.drawImage(str(img_path), 100, 400, width=300, height=200)
    c.showPage()
    c.save()

    import pypdfium2
    with pdfplumber.open(pdf_path) as pdf:
        hints = page_hint_boxes(pdf.pages[0])
    doc = pypdfium2.PdfDocument(str(pdf_path))
    page = doc[0].render(scale=RENDER_SCALE).to_pil()
    doc.close()

    weights = ROOT_DIR / "data" / "models" / "mobile_sam.pt"   # downloaded once, then cached
    boxes = SamSegmenter(weights).segment(page, hints)
    assert boxes
    assert boxes[0].iou(hints[0]) > 0.5
