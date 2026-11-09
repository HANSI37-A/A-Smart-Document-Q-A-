"""Run the Streamlit app headlessly and click through it like a user would."""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import FakeSpeaker, FakeTranscriber

APP = str(Path(__file__).resolve().parent.parent / "app.py")

pytestmark = pytest.mark.slow  # loads the real embedding model


@pytest.fixture
def app_env(tmp_path, monkeypatch, sample_pdf):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))   # use a throwaway index
    monkeypatch.setenv("OPENROUTER_API_KEY", "")     # force search-only mode
    monkeypatch.setenv("GEMINI_API_KEY", "")         # (a real key in .env must not leak in)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("VOICE_STT", "none")          # no real speech models in these tests
    monkeypatch.setenv("VOICE_TTS", "none")
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


# ---------------------------------------------------------------- voice

QUESTION = "How long is the probation period?"


@pytest.fixture
def voice_app(app_env, monkeypatch):
    """The app with fake ears and a fake voice, so no speech model is needed."""
    ears, mouth = FakeTranscriber(QUESTION), FakeSpeaker()
    monkeypatch.setattr("rag.config.build_voice", lambda settings=None: (ears, mouth))
    st.cache_resource.clear()   # get_voice() caches for the whole server
    st.cache_data.clear()       # speak() caches audio by text
    yield ears, mouth
    st.cache_resource.clear()
    st.cache_data.clear()


def spoken_input(monkeypatch):
    """AppTest can only type text, so pretend the chat bar returned a recording."""
    monkeypatch.setattr("rag.voice.question_from_input", lambda value, ears: (QUESTION, True))


def test_voice_is_off_when_not_configured(app_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert not any("Voice" in c.value for c in at.sidebar.caption)
    assert at.chat_input[0].placeholder == "Ask a question about your documents"


def test_sidebar_and_chat_bar_show_that_voice_is_ready(voice_app):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any("heard by Fake ears, spoken by Fake voice" in c.value for c in at.sidebar.caption)
    assert "microphone" in at.chat_input[0].placeholder


def test_spoken_question_is_answered_and_read_aloud(voice_app, monkeypatch):
    ears, mouth = voice_app
    spoken_input(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value("(a recording)").run()
    assert not at.exception

    text = " ".join(m.value for m in at.markdown)
    assert QUESTION in text and "employee_handbook.pdf, page 4" in text
    assert any("Spoken question" in c.value for c in at.caption)   # the user bubble says so
    assert len(at.get("audio")) == 1                               # the answer is played
    assert len(mouth.spoken) == 1
    assert "[" not in mouth.spoken[0] and "**" not in mouth.spoken[0]   # no markers read out


def test_typed_question_is_not_read_aloud_but_can_be_listened_to(voice_app):
    ears, mouth = voice_app
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value(QUESTION).run()
    assert not at.exception
    assert mouth.spoken == [] and ears.audio == [] and len(at.get("audio")) == 0
    assert not any("Spoken question" in c.value for c in at.caption)

    next(b for b in at.button if b.label == "Listen").click().run()   # on demand
    assert not at.exception
    assert len(mouth.spoken) == 1 and len(at.get("audio")) == 1


def test_speech_failure_keeps_the_written_answer(voice_app, monkeypatch):
    from rag.voice import VoiceError

    ears, mouth = voice_app

    def broken(text):
        raise VoiceError("The voice crashed.")

    mouth.synthesize = broken
    spoken_input(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value("(a recording)").run()
    assert not at.exception
    assert "employee_handbook.pdf, page 4" in " ".join(m.value for m in at.markdown)
    assert any("The voice crashed." in w.value for w in at.warning)
    assert len(at.get("audio")) == 0
