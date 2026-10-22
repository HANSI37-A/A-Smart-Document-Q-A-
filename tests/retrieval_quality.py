"""Slow tests that use the REAL embedding model (downloaded on first run).

Skip them with: pytest -m "not slow"
"""
import json
from pathlib import Path

import pytest

from eval.evaluate import evaluate

QUESTIONS = json.loads((Path(__file__).parent.parent / "eval" / "questions.json").read_text())

pytestmark = pytest.mark.slow


def test_retrieval_finds_the_right_page(sample_pdf):
    m = evaluate(sample_pdf, QUESTIONS, chunk_size=500, overlap=100, k=3, verbose=False)
    # A quality gate: if a change makes search worse, this test fails.
    assert m["hit@3"] == 1.0
    assert m["hit@1"] >= 0.8   # tighten this once you've seen your own baseline


def test_off_topic_question_scores_below_threshold(tmp_path, sample_pdf):
    from rag.embeddings import FastEmbedder
    from rag.pipeline import NOT_FOUND, DocQA
    from rag.vector_store import VectorStore

    qa = DocQA(VectorStore(str(tmp_path)), FastEmbedder(), generator=None, min_score=0.5)
    qa.ingest_pdf(sample_pdf)
    assert qa.ask("What is the capital of France?").text == NOT_FOUND
    assert qa.ask("How long is the probation period?").sources[0].chunk.page == 4