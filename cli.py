"""Command-line interface: useful for testing without the web UI.

    python cli.py ingest sample_docs/employee_handbook.pdf
    python cli.py ask "How many days of annual leave do I get?"
    python cli.py chat          # a conversation with memory (type "exit" to stop)
    python cli.py docs
"""
import argparse
import sys

from rag.config import build_app, build_chat_store, load_settings
from rag.memory import recent_history


def print_answer(answer) -> None:
    print(answer.text, "\n")
    for n, r in enumerate(answer.sources, start=1):
        mark = "*" if n in answer.cited else " "   # * = cited by the model
        print(f"{mark}[{n}] {r.chunk.source} p.{r.chunk.page} (score {r.score:.2f})")
    for r in answer.images:
        print(f"  image: {r.chunk.image_path}")


def chat(app, chat_id: str | None = None) -> None:
    chats = build_chat_store()
    settings = load_settings()
    chat_id = chat_id or chats.create_chat()
    print(f"Chat {chat_id}. Ask questions; type 'exit' to stop.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        history = recent_history(chats.messages(chat_id), settings.memory_messages)
        chats.add_message(chat_id, "user", question)
        try:
            answer = app.ask(question, history=history)
        except Exception as e:
            print(f"Something went wrong: {e}\n")
            chats.add_message(chat_id, "assistant", f"Something went wrong: {e}", error=True)
            continue
        if answer.search_query and answer.search_query != question:
            print(f"(searched for: {answer.search_query})")
        print_answer(answer)
        print()
        chats.add_message(chat_id, "assistant", answer.text, cited=answer.cited,
                          search_query=answer.search_query,
                          images=[{"path": r.chunk.image_path, "source": r.chunk.source,
                                   "page": r.chunk.page} for r in answer.images])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask questions about your PDFs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="add PDF files to the index")
    p_ingest.add_argument("files", nargs="+")

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")

    p_chat = sub.add_parser("chat", help="have a conversation with memory")
    p_chat.add_argument("--id", help="continue a saved chat")

    sub.add_parser("docs", help="list indexed documents")

    args = parser.parse_args(argv)
    app = build_app()

    if args.command == "ingest":
        for f in args.files:
            r = app.ingest_pdf(f)
            print(f"Indexed {r.source}: {r.pages} pages, {r.chunks} chunks, "
                  f"{r.tables} tables, {r.figures} figures (id {r.doc_id})")
    elif args.command == "docs":
        for doc_id, name in app.store.documents().items():
            print(f"{doc_id}  {name}")
    elif args.command == "ask":
        print_answer(app.ask(args.question))
    elif args.command == "chat":
        chat(app, args.id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
