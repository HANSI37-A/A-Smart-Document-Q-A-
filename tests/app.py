"""Run the Streamlit app headlessly and click through it like a user would."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

pytestmark = pytest.mark.slow  # loads the real embedding model


@pytest.fixture
def app_env(tmp_path, monkeypatch, sample_pdf):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))   # use a throwaway index
    monkeypatch.setenv("OPENROUTER_API_KEY", "")     # force search-only mode
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

def test_chat_is_saved_listed_and_survives_a_new_session(app_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value("How long is the probation period?").run()
    assert not at.exception

    # the chat is named after the first question and shown in the sidebar
    labels = [b.label for b in at.sidebar.button]
    assert any(label.startswith("How long is the probation") for label in labels)

    # a brand-new browser session finds the saved chat and can reopen it
    again = AppTest.from_file(APP, default_timeout=120).run()
    reopen = next(b for b in again.sidebar.button
                  if b.label.startswith("How long is the probation"))
    reopen.click().run()
    assert not again.exception
    text = " ".join(m.value for m in again.markdown)
    assert "How long is the probation period?" in text
    assert "employee_handbook.pdf, page 4" in text


def test_new_chat_button_starts_an_empty_conversation(app_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value("How long is the probation period?").run()
    next(b for b in at.sidebar.button if b.label == "New chat").click().run()
    assert not at.exception
    assert len(at.chat_message) == 0
