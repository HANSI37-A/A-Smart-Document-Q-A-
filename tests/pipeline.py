import pytest

from rag.models import Turn
from rag.pipeline import NOT_FOUND, WEB_PREFIX, DocQA
from rag.vector_store import VectorStore
from rag.web_search import WebSearchError
from tests.conftest import FakeGenerator, FakeWebSearcher


@pytest.fixture
def qa(tmp_path, fake_embedder):
    return DocQA(VectorStore(str(tmp_path)), fake_embedder, FakeGenerator(), min_score=0.2)


def test_ingest_reports_pages_and_chunks(qa, sample_pdf):
    report = qa.ingest_pdf(sample_pdf)
    assert report.source == "employee_handbook.pdf"
    assert report.pages == 6
    assert report.chunks >= 6
    assert qa.store.count() == report.chunks


def test_ingesting_same_file_twice_is_harmless(qa, sample_pdf):
    first = qa.ingest_pdf(sample_pdf)
    qa.ingest_pdf(sample_pdf)
    assert qa.store.count() == first.chunks
    assert len(qa.store.documents()) == 1


def test_ask_passes_relevant_chunks_to_generator(qa, sample_pdf):
    qa.ingest_pdf(sample_pdf)
    answer = qa.ask("How many days of paid annual leave?")

    question, results = qa.generator.calls[0]
    assert question == "How many days of paid annual leave?"
    assert results[0].chunk.page == 2          # the leave policy page
    assert answer.cited == [1]
    assert answer.sources == results


def test_unrelated_question_skips_the_model(tmp_path, fake_embedder, sample_pdf):
    qa = DocQA(VectorStore(str(tmp_path)), fake_embedder, FakeGenerator(), min_score=0.5)
    qa.ingest_pdf(sample_pdf)
    answer = qa.ask("zebra quantum volcano")
    assert answer.text == NOT_FOUND
    assert answer.sources == []
    assert qa.generator.calls == []            # no API call was made


def test_search_only_mode_without_generator(tmp_path, fake_embedder, sample_pdf):
    qa = DocQA(VectorStore(str(tmp_path)), fake_embedder, generator=None, min_score=0.2)
    qa.ingest_pdf(sample_pdf)
    answer = qa.ask("laptop replaced years")
    assert "Search-only" in answer.text
    assert answer.sources[0].chunk.page == 5   # the IT security page


def test_empty_question_is_rejected(qa):
    with pytest.raises(ValueError):
        qa.ask("   ")

def test_deleting_a_document_removes_its_chunks_and_saved_figures(tmp_path, fake_embedder,
                                                                   sample_pdf):
    figs = tmp_path / "figs"
    qa = DocQA(VectorStore(str(tmp_path / "db")), fake_embedder, FakeGenerator(),
               figures_dir=figs)
    report = qa.ingest_pdf(sample_pdf)
    (figs / report.doc_id).mkdir(parents=True)
    (figs / report.doc_id / "p1-f0.png").write_bytes(b"png")
    (figs / "other-doc").mkdir()

    qa.delete_document(report.doc_id)
    assert qa.store.count() == 0
    assert not (figs / report.doc_id).exists()
    assert (figs / "other-doc").exists()          # other documents are untouched


# ---------------------------------------------------------------- web fallback

def test_unrelated_question_is_answered_from_the_web(tmp_path, fake_embedder, sample_pdf):
    web = FakeWebSearcher()
    qa = DocQA(VectorStore(str(tmp_path)), fake_embedder, FakeGenerator(), min_score=0.5,
               web_searcher=web)
    qa.ingest_pdf(sample_pdf)
    answer = qa.ask("zebra quantum volcano")
    assert answer.web is True
    assert answer.text == WEB_PREFIX + web.text
    assert [r.chunk.url for r in answer.sources] == ["https://example.com/france"]
    assert answer.cited == [1]
    assert web.calls == [("zebra quantum volcano", [])]
    assert qa.generator.calls == []            # the document model was not asked


def test_model_refusal_falls_back_to_the_web(tmp_path, fake_embedder, sample_pdf):
    web = FakeWebSearcher()
    qa = DocQA(VectorStore(str(tmp_path)), fake_embedder, FakeGenerator(reply=NOT_FOUND),
               min_score=0.2, web_searcher=web)
    qa.ingest_pdf(sample_pdf)
    answer = qa.ask("How many days of paid annual leave?")
    assert len(qa.generator.calls) == 1        # the documents were tried first
    assert len(web.calls) == 1
    assert answer.web is True
    assert all(r.chunk.kind == "web" for r in answer.sources)   # document passages replaced


def test_answer_found_in_documents_does_not_search_the_web(tmp_path, fake_embedder, sample_pdf):
    web = FakeWebSearcher()
    qa = DocQA(VectorStore(str(tmp_path)), fake_embedder, FakeGenerator(), min_score=0.2,
               web_searcher=web)
    qa.ingest_pdf(sample_pdf)
    answer = qa.ask("How many days of paid annual leave?")
    assert web.calls == []
    assert answer.web is False


def test_failed_web_search_still_returns_not_found(tmp_path, fake_embedder, sample_pdf):
    qa = DocQA(VectorStore(str(tmp_path)), fake_embedder, FakeGenerator(), min_score=0.5,
               web_searcher=FakeWebSearcher(error=WebSearchError("Google Search failed (HTTP 503).")))
    qa.ingest_pdf(sample_pdf)
    answer = qa.ask("zebra quantum volcano")
    assert answer.text.startswith(NOT_FOUND)
    assert "HTTP 503" in answer.text
    assert answer.web is False and answer.sources == []


def test_unexpected_web_error_never_breaks_the_answer(tmp_path, fake_embedder, sample_pdf):
    qa = DocQA(VectorStore(str(tmp_path)), fake_embedder, FakeGenerator(), min_score=0.5,
               web_searcher=FakeWebSearcher(error=RuntimeError("boom")))
    qa.ingest_pdf(sample_pdf)
    assert qa.ask("zebra quantum volcano").text == NOT_FOUND


def test_follow_up_web_search_uses_the_rewritten_question_and_history(tmp_path, fake_embedder,
                                                                     sample_pdf):
    class Rewriter:
        def rewrite(self, history, question):
            return "zebra quantum volcano population"

    web = FakeWebSearcher()
    qa = DocQA(VectorStore(str(tmp_path)), fake_embedder, FakeGenerator(), min_score=0.5,
               rewriter=Rewriter(), web_searcher=web)
    qa.ingest_pdf(sample_pdf)
    history = [Turn("user", "Tell me about zebras"), Turn("assistant", "They are striped.")]
    qa.ask("how many are there?", history=history)
    assert web.calls == [("zebra quantum volcano population", history)]
