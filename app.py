"""Streamlit web app: upload PDFs and chat with them.

Run: streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st

from rag.config import build_app, build_chat_store, load_settings
from rag.memory import recent_history
from rag.pdf_loader import NoTextError

st.set_page_config(page_title="Document Q&A", page_icon=":material/description:", layout="wide")


@st.cache_resource  # build once per server, not on every click (loading the model is slow)
def get_app():
    return build_app()


@st.cache_resource
def get_chats():
    return build_chat_store()


app = get_app()
chats = get_chats()
settings = load_settings()

if "chat_id" not in st.session_state or (
        st.session_state.chat_id and not chats.exists(st.session_state.chat_id)):
    st.session_state.chat_id = None   # a new chat is created when the first question is asked

# ---------------- sidebar: documents and chats ----------------
with st.sidebar:
    st.header("Your documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploads and st.button("Add to index", type="primary"):
        for up in uploads:
            with st.spinner(f"Reading {up.name}..."):
                # pypdf needs a real file, so save the upload to a temporary one.
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(up.getbuffer())
                try:
                    r = app.ingest_pdf(tmp.name, source_name=up.name)
                    extra = f", {r.tables} tables" if r.tables else ""
                    extra += f", {r.figures} figures" if r.figures else ""
                    st.success(f"{r.source}: {r.pages} pages, {r.chunks} chunks{extra}")
                except NoTextError as e:
                    st.error(str(e))
                finally:
                    Path(tmp.name).unlink(missing_ok=True)

    docs = app.store.documents()
    if not docs:
        st.info("No documents yet. Upload a PDF to start.")
    for doc_id, name in docs.items():
        col1, col2 = st.columns([5, 1], vertical_alignment="center")
        short = name if len(name) <= 28 else name[:25] + "..."
        col1.caption(f":material/description: {short}")
        if col2.button(":material/delete:", key=f"del-{doc_id}", help=f"Remove {name}",
                       type="tertiary"):
            app.delete_document(doc_id)
            st.rerun()

    st.divider()
    st.header("Chats")
    if st.button("New chat", icon=":material/add:", width="stretch"):
        st.session_state.chat_id = None
        st.rerun()
    for chat in chats.list_chats():
        col1, col2 = st.columns([5, 1], vertical_alignment="center")
        active = chat["id"] == st.session_state.chat_id
        label = chat["title"] if len(chat["title"]) <= 30 else chat["title"][:27] + "..."
        if col1.button(label, key=f"chat-{chat['id']}", type="secondary" if active else "tertiary",
                       icon=":material/chat:", disabled=active):
            st.session_state.chat_id = chat["id"]
            st.rerun()
        if col2.button(":material/delete:", key=f"delchat-{chat['id']}",
                       help="Delete this chat", type="tertiary"):
            chats.delete_chat(chat["id"])
            if active:
                st.session_state.chat_id = None
            st.rerun()

    st.divider()
    if app.generator is None:
        st.warning("No OPENROUTER_API_KEY set: search-only mode (shows passages, no answers).")
    else:
        st.caption(f"Model: {settings.model}")
        st.caption("Charts and images: " + ("read by a vision model" if app.describer
                                              else "not indexed (VISION_PROVIDER=none)"))

# ---------------- main area: chat ----------------
st.title(":material/description: Ask your documents")


def show_images(images):
    shown = [img for img in images if Path(img["path"]).is_file()]
    if not shown:
        return
    for col, img in zip(st.columns(len(shown)), shown):
        col.image(img["path"], caption=f"Figure · {img['source']}, page {img['page']}",
                  width="stretch")


def show_sources(sources, cited):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for n, s in enumerate(sources, start=1):
            star = " · :material/star: cited" if n in cited else ""
            kind = f" · {s['kind']}" if s.get("kind", "text") != "text" else ""
            st.markdown(f"**[{n}] {s['source']}, page {s['page']}**{kind} · "
                        f"score {s['score']:.2f}{star}")
            st.caption(s["text"])


def show_message(msg):
    st.markdown(msg["content"])
    query = msg.get("search_query", "")
    if query and msg.get("question") and query.strip().lower() != msg["question"].strip().lower():
        st.caption(f":material/search: Searched for: {query}")
    show_images(msg.get("images", []))
    show_sources(msg.get("sources", []), msg.get("cited", []))


messages = chats.messages(st.session_state.chat_id) if st.session_state.chat_id else []

# Streamlit re-runs this whole script on every interaction, so redraw the history.
for msg in messages:
    with st.chat_message(msg["role"]):
        show_message(msg)

if question := st.chat_input("Ask a question about your documents"):
    if st.session_state.chat_id is None:
        st.session_state.chat_id = chats.create_chat()
    chat_id = st.session_state.chat_id
    history = recent_history(messages, settings.memory_messages)   # memory for follow-ups
    chats.add_message(chat_id, "user", question)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                answer = app.ask(question, history=history)
                reply = {
                    "content": answer.text,
                    "question": question,
                    "search_query": answer.search_query,
                    "sources": [{"source": r.chunk.source, "page": r.chunk.page,
                                 "score": r.score, "text": r.chunk.text, "kind": r.chunk.kind}
                                for r in answer.sources],
                    "cited": answer.cited,
                    "images": [{"path": r.chunk.image_path, "source": r.chunk.source,
                                "page": r.chunk.page} for r in answer.images],
                }
            except Exception as e:  # e.g. invalid API key or no internet
                reply = {"content": f":material/error: Something went wrong: {e}", "error": True}
        show_message(reply)

    content = reply.pop("content")
    chats.add_message(chat_id, "assistant", content, **reply)
    st.rerun()   # redraw so the sidebar shows the new or renamed chat
