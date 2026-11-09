"""Streamlit web app: upload PDFs and chat with them.

Run: streamlit run app.py
"""
import contextlib
import tempfile
from pathlib import Path

import streamlit as st

from rag.config import build_app, build_chat_store, build_voice, load_settings
from rag.memory import recent_history
from rag.pdf_loader import NoTextError
from rag.voice import VoiceError, question_from_input, speech_text

st.set_page_config(page_title="Document Q&A", page_icon=":material/description:", layout="wide")


@st.cache_resource  # build once per server, not on every click (loading the model is slow)
def get_app():
    return build_app()


@st.cache_resource
def get_chats():
    return build_chat_store()


settings = load_settings()


@st.cache_resource   # loads the speech voice once; Whisper itself loads on first use
def get_voice():
    return build_voice(settings)


app = get_app()
chats = get_chats()
transcriber, speaker = get_voice()   # either is None when voice is off or not installed


@st.cache_data(max_entries=50, show_spinner="Speaking...")
def speak(text: str) -> bytes:
    """WAV for a text. Cached, so replaying an answer costs nothing."""
    return speaker.synthesize(text)


def speak_answer(content: str) -> bytes | None:
    """Audio for an answer (no citation markers or Markdown); None if there is nothing to say."""
    text = speech_text(content)
    return speak(text) if text else None


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
        st.warning("No OPENAI_API_KEY, GEMINI_API_KEY or OPENROUTER_API_KEY set: search-only "
                   "mode (shows passages, no answers).")
    else:
        st.caption(f"Model: {settings.model} ({settings.llm_provider})")
        st.caption("Charts and images: " + ("read by a vision model" if app.describer
                                              else "not indexed (VISION_PROVIDER=none)"))
    if transcriber or speaker:
        st.caption(f":material/mic: Voice: heard by {transcriber.label if transcriber else 'off'}"
                   f", spoken by {speaker.label if speaker else 'off'}")
    elif settings.voice_stt != "none" or settings.voice_tts != "none":
        st.caption(":material/mic_off: Voice is off. Install it with "
                   "`pip install -r requirements-voice.txt`.")

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
    web = all(s.get("kind") == "web" for s in sources)
    with st.expander(f"{'Web sources' if web else 'Sources'} ({len(sources)})"):
        for n, s in enumerate(sources, start=1):
            star = " · :material/star: cited" if n in cited else ""
            if s.get("kind") == "web":   # a page found by Google: a link, no page or score
                st.markdown(f"**[{n}] [{s['source']}]({s.get('url', '')})** · web{star}")
                continue
            kind = f" · {s['kind']}" if s.get("kind", "text") != "text" else ""
            st.markdown(f"**[{n}] {s['source']}, page {s['page']}**{kind} · "
                        f"score {s['score']:.2f}{star}")
            st.caption(s["text"])


def show_message(msg, key=None):
    """Draw one message. `key` (saved messages only) enables the Listen button."""
    st.markdown(msg["content"])
    if msg.get("role") == "user" and msg.get("voice"):
        st.caption(":material/mic: Spoken question")
    query = msg.get("search_query", "")
    if query and msg.get("question") and query.strip().lower() != msg["question"].strip().lower():
        st.caption(f":material/search: Searched for: {query}")
    if msg.get("web"):
        st.caption(":material/public: Answer from Google Search — not from your documents")
    show_images(msg.get("images", []))
    show_sources(msg.get("sources", []), msg.get("cited", []))
    if key and speaker and msg.get("role") == "assistant" and not msg.get("error"):
        if st.button("Listen", key=key, icon=":material/volume_up:", type="tertiary"):
            try:
                audio = speak_answer(msg["content"])
                if audio:
                    st.audio(audio, format="audio/wav", autoplay=True)
            except VoiceError as e:
                st.warning(str(e))


messages = chats.messages(st.session_state.chat_id) if st.session_state.chat_id else []

# A spoken answer is made while handling the question, then played after the redraw below.
pending_audio = st.session_state.pop("pending_audio", None)
voice_notice = st.session_state.pop("voice_notice", None)

# Streamlit re-runs this whole script on every interaction, so redraw the history.
for i, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        show_message(msg, key=f"listen-{st.session_state.chat_id}-{i}")
        if i == len(messages) - 1 and msg["role"] == "assistant":
            if pending_audio:
                st.audio(pending_audio, format="audio/wav", autoplay=settings.voice_autoplay)
            if voice_notice:
                st.warning(voice_notice)


def handle_question(question: str, *, spoken: bool = False):
    """Answer a question, save both messages and redraw. Typed and spoken questions share this."""
    if st.session_state.chat_id is None:
        st.session_state.chat_id = chats.create_chat()
    chat_id = st.session_state.chat_id
    history = recent_history(messages, settings.memory_messages)   # memory for follow-ups
    voice_meta = {"voice": True} if spoken else {}
    chats.add_message(chat_id, "user", question, **voice_meta)
    with st.chat_message("user"):
        show_message({"role": "user", "content": question, **voice_meta})

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                answer = app.ask(question, history=history)
                reply = {
                    "content": answer.text,
                    "question": question,
                    "search_query": answer.search_query,
                    "sources": [{"source": r.chunk.source, "page": r.chunk.page,
                                 "score": r.score, "text": r.chunk.text, "kind": r.chunk.kind,
                                 "url": r.chunk.url}
                                for r in answer.sources],
                    "cited": answer.cited,
                    "web": answer.web,
                    "images": [{"path": r.chunk.image_path, "source": r.chunk.source,
                                "page": r.chunk.page} for r in answer.images],
                }
            except Exception as e:  # e.g. invalid API key or no internet
                reply = {"content": f":material/error: Something went wrong: {e}", "error": True}
        show_message(reply)

        if spoken and speaker and not reply.get("error"):
            with st.spinner("Preparing the spoken answer..."):
                try:
                    st.session_state.pending_audio = speak_answer(reply["content"])
                except VoiceError as e:   # the written answer is still shown
                    st.session_state.voice_notice = str(e)

    content = reply.pop("content")
    chats.add_message(chat_id, "assistant", content, **reply)
    st.rerun()   # redraw so the sidebar shows the new or renamed chat


def read_question(value):
    """(question, spoken) from the chat bar; shows what went wrong if speech was not understood."""
    recording = (not isinstance(value, str) and value.audio is not None
                 and not (value.text or "").strip())
    try:
        with st.spinner("Transcribing...") if recording else contextlib.nullcontext():
            question, spoken = question_from_input(value, transcriber)
    except VoiceError as e:
        st.error(str(e))
        return "", False
    if spoken and not question:
        st.warning("I didn't catch that. Please try again.")
    return question, spoken


placeholder = "Ask a question about your documents"
value = st.chat_input(placeholder + (" or tap the microphone" if transcriber else ""),
                      accept_audio=transcriber is not None)
if value is not None:
    question, spoken = read_question(value)
    if question:
        handle_question(question, spoken=spoken)
