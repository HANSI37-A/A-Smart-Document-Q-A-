"""Run the Streamlit app headlessly and click through it like a user would."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

pytestmark = pytest.mark.slow  # loads the real embedding model


@pytest.fixture
def app_env(tmp_path, monkeypatch, sample_pdf):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))   # use a throwaway index
    monkeypatch.setenv("GEMINI_API_KEY", "")     # force search-only mode
    from rag.config import build_app
    build_app().ingest_pdf(sample_pdf)              # pre-load the sample document


def test_app_shows_documents_and_answers(app_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert "employee_handbook.pdf" in " ".join(c.value for c in at.sidebar.caption)
    assert "search-only" in at.sidebar.warning[0].value

    at.chat_input[0].set_value("How long is the probation period?").run()
    assert not at.exception
    assistant_text = " ".join(m.value for m in at.markdown)
    assert "Search-only mode" in assistant_text
    assert "employee_handbook.pdf, page 4" in assistant_text