"""Split pages into small, overlapping chunks."""

from rag.models import Chunk, Page


def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        # grow the chunk word by word until the next word would overflow
        end = start
        length = 0
        while end < len(words) and (length + len(words[end]) + 1 <= chunk_size or end == start):
            length += len(words[end]) + 1
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        # step back to create the overlap
        back = end
        kept = 0
        while back > start + 1 and kept + len(words[back - 1]) + 1 <= overlap:
            back -= 1
            kept += len(words[back]) + 1
        start = back
    return chunks


def chunk_pages(pages: list[Page], doc_id: str, chunk_size: int = 500,
                overlap: int = 100) -> list[Chunk]:
    """Chunks never cross a page boundary, so every citation has one exact page."""
    chunks = []
    for page in pages:
        for i, text in enumerate(split_text(page.text, chunk_size, overlap)):
            chunks.append(Chunk(
                id=f"{doc_id}-p{page.page}-c{i}",
                text=text,
                source=page.source,
                page=page.page,
                doc_id=doc_id,
            ))
    return chunks