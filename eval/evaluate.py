"""Measure retrieval quality: for each test question, is the right page found?

    python eval/evaluate.py
    python eval/evaluate.py --chunk-size 400 --overlap 80    # try other settings

Metrics:
    hit@1  share of questions where the best result is from the right page
    hit@k  share where the right page is anywhere in the top k results
    MRR    mean reciprocal rank: 1.0 if always first, 0.5 if always second, ...
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.embeddings import FastEmbedder  # noqa: E402
from rag.pipeline import DocQA  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402


def evaluate(pdf: Path, questions: list[dict], chunk_size: int, overlap: int, k: int,
             embedder=None, verbose: bool = True) -> dict:
    # fresh index, never touches your real data; ignore_cleanup_errors as a Windows safety net
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = VectorStore(tmp)
        try:
            qa = DocQA(store, embedder or FastEmbedder(), generator=None,
                       top_k=k, min_score=0.0, chunk_size=chunk_size, overlap=overlap)
            qa.ingest_pdf(pdf)

            hits1 = hitsk = 0
            rr_total = 0.0
            for q in questions:
                pages = [r.chunk.page for r in qa.retrieve(q["question"])]
                rank = pages.index(q["page"]) + 1 if q["page"] in pages else None
                hits1 += rank == 1
                hitsk += rank is not None
                rr_total += 1 / rank if rank else 0
                if verbose:
                    mark = "OK  " if rank == 1 else ("~   " if rank else "MISS")
                    print(f"{mark} rank={rank} pages={pages} {q['question']}")
        finally:
            store.close()  # release chroma.sqlite3 before the directory is deleted

    n = len(questions)
    return {"hit@1": hits1 / n, f"hit@{k}": hitsk / n, "mrr": rr_total / n, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=ROOT / "sample_docs" / "employee_handbook.pdf", type=Path)
    ap.add_argument("--questions", default=ROOT / "eval" / "questions.json", type=Path)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    questions = json.loads(args.questions.read_text())
    m = evaluate(args.pdf, questions, args.chunk_size, args.overlap, args.k)
    print(f"\n{m['n']} questions | hit@1 {m['hit@1']:.0%} | hit@{args.k} {m[f'hit@{args.k}']:.0%}"
          f" | MRR {m['mrr']:.2f}")


if __name__ == "__main__":
    main()