import pytest

from rag.chunker import chunk_pages, split_text
from rag.models import Page

LONG_TEXT = " ".join(f"word{i}" for i in range(500))  # about 3,900 characters


def test_short_text_is_one_chunk():
    assert split_text("A short sentence.", chunk_size=100, overlap=20) == ["A short sentence."]


def test_chunks_respect_max_size():
    chunks = split_text(LONG_TEXT, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = split_text(LONG_TEXT, chunk_size=200, overlap=40)
    for first, second in zip(chunks, chunks[1:]):
        last_word = first.split()[-1]
        assert last_word in second.split()  # the end of one chunk starts the next


def test_no_words_are_lost():
    chunks = split_text(LONG_TEXT, chunk_size=200, overlap=40)
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == set(LONG_TEXT.split())


def test_word_longer_than_chunk_size_does_not_loop_forever():
    text = "short " + "x" * 300 + " end"
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert "x" * 300 in chunks  # kept whole, in its own chunk
    assert chunks[-1].endswith("end")


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_text("text", chunk_size=100, overlap=100)


def test_chunk_pages_keeps_page_numbers_and_ids():
    pages = [Page("a.pdf", 1, "Page one."), Page("a.pdf", 2, LONG_TEXT)]
    chunks = chunk_pages(pages, doc_id="abc", chunk_size=300, overlap=50)
    assert chunks[0].id == "abc-p1-c0" and chunks[0].page == 1
    assert chunks[1].id == "abc-p2-c0" and chunks[1].page == 2
    assert all(c.source == "a.pdf" and c.doc_id == "abc" for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)  # ids are unique