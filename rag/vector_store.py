"""Store chunk embeddings in Chroma and search them."""

import chromadb

from rag.models import Chunk, SearchResult


def _metadata(c: Chunk) -> dict:
    meta = {"source": c.source, "page": c.page, "doc_id": c.doc_id, "kind": c.kind}
    if c.image_path:                      # Chroma does not accept None values
        meta["image_path"] = c.image_path
    return meta


def _chunk(id_: str, text: str, meta: dict) -> Chunk:
    # .get() with defaults: indexes built before these fields existed still load
    return Chunk(id=id_, text=text, source=meta["source"], page=meta["page"],
                 doc_id=meta["doc_id"], kind=meta.get("kind", "text"),
                 image_path=meta.get("image_path", ""))


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
            metadatas=[_metadata(c) for c in chunks],
        )

    def search(self, query_embedding: list[float], k: int = 5) -> list[SearchResult]:
        if self.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [SearchResult(chunk=_chunk(id_, text, meta), score=round(1 - dist, 4))
                for id_, text, meta, dist in zip(res["ids"][0], res["documents"][0],
                                                 res["metadatas"][0], res["distances"][0])]

    def figures_on_pages(self, pages: list[tuple[str, int]]
                         ) -> list[tuple[Chunk, list[float]]]:
        """Figure chunks (with their stored embeddings) on the given (doc_id, page) pairs.

        Used to find the chart that sits next to a retrieved paragraph."""
        found = []
        by_doc: dict[str, set[int]] = {}
        for doc_id, page in pages:
            by_doc.setdefault(doc_id, set()).add(page)
        for doc_id, doc_pages in by_doc.items():
            res = self._col.get(
                where={"$and": [{"kind": "figure"}, {"doc_id": doc_id},
                                {"page": {"$in": sorted(doc_pages)}}]},
                include=["documents", "metadatas", "embeddings"],
            )
            for id_, text, meta, emb in zip(res["ids"], res["documents"], res["metadatas"],
                                            res["embeddings"]):
                found.append((_chunk(id_, text, meta), [float(x) for x in emb]))
        return found

    def embeddings_for(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        res = self._col.get(ids=ids, include=["embeddings"])
        return {id_: [float(x) for x in emb] for id_, emb in zip(res["ids"], res["embeddings"])}

    def documents(self) -> dict[str, str]:
        metas = self._col.get(include=["metadatas"])["metadatas"]
        return {m["doc_id"]: m["source"] for m in metas}

    def delete_document(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def existing_ids(self, ids: list[str]) -> set[str]:
        """Which of these chunk ids are already indexed (used to skip paid vision calls)."""
        if not ids:
            return set()
        return set(self._col.get(ids=ids, include=[])["ids"])

    def count(self) -> int:
        return self._col.count()

    def close(self) -> None:
        """Release Chroma's SQLite file handles (Windows can't delete a locked directory)."""
        self._client.close()
        self._client.clear_system_cache()  # drop the cached client that keeps the DB open
