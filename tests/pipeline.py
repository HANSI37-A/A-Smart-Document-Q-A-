import pytest

from rag.pipeline import NOT_FOUND, DocQA
from rag.vector_store import VectorStore
from tests.conftest import FakeGenerator


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