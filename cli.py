"""Command-line interface: useful for testing without the web UI.

    python cli.py ingest sample_docs/employee_handbook.pdf
    python cli.py ask "How many days of annual leave do I get?"
    python cli.py docs
"""
import argparse
import sys

from rag.config import build_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask questions about your PDFs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="add PDF files to the index")
    p_ingest.add_argument("files", nargs="+")

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")

    sub.add_parser("docs", help="list indexed documents")

    args = parser.parse_args(argv)
    app = build_app()

    if args.command == "ingest":
        for f in args.files:
            r = app.ingest_pdf(f)
            print(f"Indexed {r.source}: {r.pages} pages, {r.chunks} chunks (id {r.doc_id})")
    elif args.command == "docs":
        for doc_id, name in app.store.documents().items():
            print(f"{doc_id}  {name}")
    elif args.command == "ask":
        answer = app.ask(args.question)
        print(answer.text, "\n")
        for n, r in enumerate(answer.sources, start=1):
            mark = "*" if n in answer.cited else " "   # * = cited by the model
            print(f"{mark}[{n}] {r.chunk.source} p.{r.chunk.page} (score {r.score:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())