from rag.models import Chunk
from rag.vector_store import VectorStore


def make_chunks(doc_id, source, texts):
    return [Chunk(id=f"{doc_id}-p{i + 1}-c0", text=t, source=source, page=i + 1, doc_id=doc_id)
            for i, t in enumerate(texts)]


def test_search_ranks_most_similar_first(tmp_path, fake_embedder):
    store = VectorStore(str(tmp_path))
    chunks = make_chunks("d1", "h.pdf", [
        "Annual leave is 14 days per year.",
        "Laptops are replaced every 3 years.",
        "The cafeteria opens at 8 am.",
    ])
    store.add(chunks, fake_embedder.embed_documents([c.text for c in chunks]))

    results = store.search(fake_embedder.embed_query("how often are laptops replaced"), k=3)
    assert results[0].chunk.text == "Laptops are replaced every 3 years."
    assert results[0].chunk.page == 2
    assert results[0].score > results[1].score  # sorted best first


def test_empty_store_returns_no_results(tmp_path, fake_embedder):
    store = VectorStore(str(tmp_path))
    assert store.search(fake_embedder.embed_query("anything")) == []


def test_adding_same_chunks_twice_does_not_duplicate(tmp_path, fake_embedder):
    store = VectorStore(str(tmp_path))
    chunks = make_chunks("d1", "h.pdf", ["one", "two"])
    vectors = fake_embedder.embed_documents(["one", "two"])
    store.add(chunks, vectors)
    store.add(chunks, vectors)
    assert store.count() == 2


def test_list_and_delete_documents(tmp_path, fake_embedder):
    store = VectorStore(str(tmp_path))
    for doc_id, name in [("d1", "a.pdf"), ("d2", "b.pdf")]:
        chunks = make_chunks(doc_id, name, ["alpha text", "beta text"])
        store.add(chunks, fake_embedder.embed_documents([c.text for c in chunks]))
    assert store.documents() == {"d1": "a.pdf", "d2": "b.pdf"}

    store.delete_document("d1")
    assert store.documents() == {"d2": "b.pdf"}
    assert store.count() == 2


def test_data_survives_reopening(tmp_path, fake_embedder):
    chunks = make_chunks("d1", "a.pdf", ["persistent text"])
    VectorStore(str(tmp_path)).add(chunks, fake_embedder.embed_documents(["persistent text"]))
    assert VectorStore(str(tmp_path)).count() == 1  # a new object, same folder