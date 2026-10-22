"""Get clean text out of a PDF, page by page."""

import re
from pathlib import Path

from pypdf import PdfReader

from rag.models import Page


class NoTextError(ValueError):
    """The PDF has no extractable text (usually a scan)."""


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")             # non-breaking spaces
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # re-join hyphenated line breaks
    text = re.sub(r"\s+", " ", text)               # collapse whitespace
    return text.strip()


def load_pdf(path: str | Path, source_name: str | None = None) -> list[Page]:
    path = Path(path)
    reader = PdfReader(path)
    name = source_name or path.name
    pages = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        text = clean_text(pdf_page.extract_text() or "")
        if text:
            pages.append(Page(source=name, page=number, text=text))
    if not pages:
        raise NoTextError(
            f"No text found in {name}. If it is a scanned document, run OCR on it first."
        )
    return pages