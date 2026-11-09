"""The sample annual report must keep the layout the demo questions rely on."""
import sys

import pdfplumber
import pytest

from rag.multimodal import extract_tables, page_hint_boxes
from tests.conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from make_sample_report_pdf import DEPARTMENTS, HEADCOUNT, REGIONS, REVENUE, build  # noqa: E402


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    return build(tmp_path_factory.mktemp("report") / "report.pdf")


def test_figures_are_on_pages_2_4_and_5_only(report):
    with pdfplumber.open(report) as pdf:
        assert len(pdf.pages) == 6
        pages_with_figures = [p.page_number for p in pdf.pages if page_hint_boxes(p)]
    assert pages_with_figures == [2, 4, 5]     # bar chart, line chart, diagram image


def test_every_page_has_a_small_logo_that_is_not_a_figure(report):
    with pdfplumber.open(report) as pdf:
        for page in pdf.pages:
            small = [i for i in page.images if i["x1"] - i["x0"] < 100]
            assert small, f"page {page.page_number} has no header logo"


def test_the_table_is_found_and_is_not_mistaken_for_a_chart(report):
    with pdfplumber.open(report) as pdf:
        tables = extract_tables(pdf)
    assert [t.page for t in tables] == [3]
    assert tables[0].rows[0][0] == "Region"


def test_the_numbers_in_the_report_add_up():
    assert round(sum(REVENUE), 1) == 23.6
    assert round(sum(r for _, r, _ in REGIONS), 1) == 23.6
    assert sum(n for _, n in DEPARTMENTS) == HEADCOUNT[-1] == 172


def test_text_carries_the_facts_the_demo_questions_ask_for(report):
    with pdfplumber.open(report) as pdf:
        text = {p.page_number: p.extract_text() for p in pdf.pages}
    assert "23.6 million USD" in text[1]
    assert "4.2 million USD in Q1" in text[2] and "Q4 was the strongest" in text[2]
    assert "North America is the largest market" in text[3]
    assert "120 people at the end of Q1 to 172" in text[4]
    assert "Engineering is the largest, with 78" in text[5]
    assert "Singapore" in text[6]
