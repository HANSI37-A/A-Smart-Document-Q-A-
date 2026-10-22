"""Store chunk embeddings in Chroma and search them."""

import chromadb

from rag.models import Chunk, SearchResult


class VectorStore:
    def __init__(self, path: str, collection: str = "documents"):
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            collection,
            embedding_function=None,                       # we pass our own vectors
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        # upsert + content-based ids = uploading the same PDF twice is harmless
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page, "doc_id": c.doc_id}
                       for c in chunks],
        )

    def search(self, query_embedding: list[float], k: int = 5) -> list[SearchResult]:
        if self.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for id_, text, meta, dist in zip(res["ids"][0], res["documents"][0],
                                         res["metadatas"][0], res["distances"][0]):
            chunk = Chunk(id=id_, text=text, source=meta["source"],
                          page=meta["page"], doc_id=meta["doc_id"])
            results.append(SearchResult(chunk=chunk, score=round(1 - dist, 4)))
        return results

    def documents(self) -> dict[str, str]:
        metas = self._col.get(include=["metadatas"])["metadatas"]
        return {m["doc_id"]: m["source"] for m in metas}

    def delete_document(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self._col.count()

    def close(self) -> None:
        """Release Chroma's SQLite file handles (Windows can't delete a locked directory)."""
        self._client.close()
        self._client.clear_system_cache()  # drop the cached client that keeps the DB open