"""Shared test helpers (pytest loads this file automatically)."""

import hashlib
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.models import SearchResult  # noqa: E402

SAMPLE_PDF = ROOT / "sample_docs" / "employee_handbook.pdf"


class FakeEmbedder:
    """A tiny, fast, offline stand-in for the real model.

    It hashes each word into one of 256 slots ("bag of words"), so texts that
    share words get similar vectors. Good enough to test the plumbing."""

    DIM = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
            v[slot] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeGenerator:
    """Records what it was asked and returns a fixed answer."""

    def __init__(self, reply: str = "Employees get 14 days of annual leave [1]."):
        self.reply = reply
        self.calls: list[tuple[str, list[SearchResult]]] = []

    def generate(self, question, results):
        self.calls.append((question, results))
        return self.reply


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def sample_pdf():
    if not SAMPLE_PDF.exists():
        pytest.skip("Run `python scripts/make_sample_pdf.py` first")
    return SAMPLE_PDF


@pytest.fixture
def make_pdf(tmp_path):
    """Factory: make_pdf(["page one text", "page two text"]) -> path to a new PDF."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    def _make(pages: list[str], name: str = "test.pdf") -> Path:
        path = tmp_path / name
        c = canvas.Canvas(str(path), pagesize=A4)
        for text in pages:
            y = 800
            for line in text.split("\n"):
                c.drawString(50, y, line)
                y -= 15
            c.showPage()
        c.save()
        return path

    return _make