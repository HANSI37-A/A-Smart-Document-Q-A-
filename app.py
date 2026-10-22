"""Streamlit web app: upload PDFs and chat with them.

Run: streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st

from rag.config import build_app, load_settings
from rag.pdf_loader import NoTextError

st.set_page_config(page_title="Document Q&A", page_icon=":material/description:", layout="wide")


@st.cache_resource  # build once per server, not on every click (loading the model is slow)
def get_app():
    return build_app()


app = get_app()
settings = load_settings()

# ---------------- sidebar: documents ----------------
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
                    st.success(f"{r.source}: {r.pages} pages, {r.chunks} chunks")
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
            app.store.delete_document(doc_id)
            st.rerun()

    st.divider()
    if app.generator is None:
        st.warning("No ANTHROPIC_API_KEY set: search-only mode (shows passages, no answers).")
    else:
        st.caption(f"Model: {settings.model}")

# ---------------- main area: chat ----------------
st.title(":material/description: Ask your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role", "content", "sources", "cited"}


def show_sources(sources, cited):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for n, s in enumerate(sources, start=1):
            star = " · :material/star: cited" if n in cited else ""
            st.markdown(f"**[{n}] {s['source']}, page {s['page']}** · score {s['score']:.2f}{star}")
            st.caption(s["text"])


# Streamlit re-runs this whole script on every interaction, so redraw the history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources", []), msg.get("cited", []))

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                answer = app.ask(question)
                sources = [{"source": r.chunk.source, "page": r.chunk.page, "score": r.score,
                            "text": r.chunk.text} for r in answer.sources]
                text, cited = answer.text, answer.cited
            except Exception as e:  # e.g. invalid API key or no internet
                text, sources, cited = f":material/error: Something went wrong: {e}", [], []
        st.markdown(text)
        show_sources(sources, cited)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources, "cited": cited})