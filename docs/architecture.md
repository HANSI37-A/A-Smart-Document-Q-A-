# Architecture & Engineering Reference

Technical specification of the **Smart Document Q&A** application: a multimodal Retrieval-Augmented Generation (RAG) system that answers questions about PDFs, using their text, tables, and images or charts, with page-level citations.

This document describes the code as implemented. File and symbol names refer to the repository root.

## Contents

1. [System architecture overview](#1-system-architecture-overview)
2. [Multimodal ingestion pipeline](#2-multimodal-ingestion-pipeline)
3. [Vector storage and embedding strategy](#3-vector-storage-and-embedding-strategy)
4. [Retrieval and question answering](#4-retrieval-and-question-answering)
5. [Configuration and environment management](#5-configuration-and-environment-management)
6. [Testing and quality assurance](#6-testing-and-quality-assurance)
7. [Known limitations and extension points](#7-known-limitations-and-extension-points)
8. [Image-aware answers and conversation memory](#8-image-aware-answers-and-conversation-memory)
9. [Voice assistant](#9-voice-assistant)

---

## 1. System architecture overview

### 1.1 Design goalsA. Basic and concept questions
Q1. What is RAG? Why did you use it?
RAG means Retrieval-Augmented Generation. First we retrieve the relevant parts of the documents. Then we give those parts to an AI model to generate the answer. I used it because a normal AI model doesn't know my private PDFs and can make things up. With RAG the answer comes from my documents.

Q2. What is an embedding?
An embedding is a list of numbers (384 numbers here) that represents the meaning of a text. Texts with similar meaning get similar numbers. This lets us search by meaning, not just by exact words.

Q3. What is a vector database? Why ChromaDB?
It is a database that finds the closest vectors to a query vector. I used ChromaDB because it runs inside my app with no separate server, it is free, and it saves data on disk, so it survives restarts.

Q4. What is chunking? Why not send the whole PDF?
Chunking means cutting the text into small pieces. A whole PDF is too big for the model's limit, costs more, and has too much noise. Small chunks make the search more exact.

Q5. Why chunk size 500 and overlap 100?
500 characters is big enough to hold one full idea and small enough to stay focused. The 100-character overlap stops a fact from being lost when it sits on the boundary between two chunks. These values can be changed, and I can test them with eval/evaluate.py.

Q6. What is cosine similarity?
It measures how close two vectors point in the same direction. 1 means very similar and 0 means unrelated. Chroma gives a distance, so I compute score = 1 - distance.

| Goal | How it is met |
|---|---|
| Every answer is traceable | Chunks carry `source`, `page` and `kind`. Chunks never cross page boundaries, so each citation maps to exactly one page. |
| No invented answers | Weak matches are filtered out, an empty retrieval skips the model entirely, and the prompt requires a fixed "not found" reply. |
| Runs without paid services | Embeddings and the vector database are local. With no API key the app still works in search-only mode. |
| Safe to re-run | Content-based document fingerprints and `upsert` make ingestion idempotent. |
| Testable offline | Every external dependency sits behind a small `Protocol` and is injected, so tests use fakes. |

### 1.2 Components

| Component | Module | Responsibility |
|---|---|---|
| Web UI | `app.py` | Streamlit chat interface: upload, document list, chat, sources panel. |
| CLI | `cli.py` | `ingest`, `ask` and `docs` commands over the same pipeline. |
| Composition root | `rag/config.py` | Reads settings and builds the object graph in `build_app()`. |
| Pipeline | `rag/pipeline.py` | `DocQA`: orchestrates ingestion and question answering. |
| PDF text loader | `rag/pdf_loader.py` | `pypdf` text extraction and cleaning, page by page. |
| Chunker | `rag/chunker.py` | Overlapping, page-bounded text chunks. |
| Multimodal extraction | `rag/multimodal.py` | Tables to Markdown; figure crops; decorative filter; figure chunks. |
| Segmenter | `rag/segmenter.py` | SAM (MobileSAM) figure outlines; box filtering; no-SAM fallback. |
| Image selector | `rag/image_selector.py` | Decides which figures, if any, a question needs. |
| Chat memory | `rag/memory.py` | SQLite chat store, history trimming, follow-up rewriting. |
| Vision | `rag/vision.py` | `VisionDescriber` protocol and the OpenRouter implementation. |
| Embedder | `rag/embeddings.py` | `Embedder` protocol and local `FastEmbedder`. |
| Vector store | `rag/vector_store.py` | Chroma wrapper: add, search, list, delete, close. |
| Generator | `rag/generator.py` | Prompt building, model call, citation parsing. |
| LLM client | `rag/openrouter.py`, `rag/gemini.py` | OpenAI-SDK clients for OpenRouter and for Gemini (its OpenAI-compatible endpoint), plus the shared retry and fallback. |
| Data classes | `rag/models.py` | `Page`, `Chunk`, `SearchResult`, `Answer`. |

### 1.3 Component interaction

```
                       ┌──────────────────────────────────────────────┐
                       │  app.py (Streamlit)        cli.py            │
                       └───────────────┬──────────────────────────────┘
                                       │ build_app()  (rag/config.py)
                                       ▼
                       ┌──────────────────────────────────────────────┐
                       │                DocQA  (pipeline.py)          │
                       │  ingest_pdf()  retrieve()  ask()             │
                       └──┬─────────┬──────────┬──────────┬───────────┘
        ingestion         │         │          │          │   question answering
   ┌──────────────────────┘         │          │          └────────────────────┐
   ▼                                ▼          ▼                               ▼
pdf_loader (pypdf)          multimodal.py    Embedder                    Generator
chunker                     pdfplumber       (FastEmbedder,              (OpenRouterGenerator)
                            pypdfium2         local CPU)                       │
                                │                 │                            ▼
                                ▼                 ▼                       openrouter.py
                          VisionDescriber    VectorStore (ChromaDB)       (retry, fallback)
                          (OpenRouter)       data/chroma/ on disk
```

`DocQA` depends only on the abstractions `Embedder`, `Generator` and `VisionDescriber` (all `typing.Protocol`), plus the concrete `VectorStore`. It never creates its own dependencies.

### 1.4 End-to-end data flow

**Ingestion**

```
PDF file
 ├─ file_fingerprint()  ─────────────► doc_id (SHA-256, first 12 hex chars)
 ├─ pypdf  ─► clean_text ─► Page[] ─► chunk_pages ─► text Chunk[]      kind="text"
 ├─ pdfplumber.extract_tables ─► Markdown ─► table Chunk[]             kind="table"
 └─ pdfplumber (detect) + pypdfium2 (render) ─► PIL image
        └─ VisionDescriber.describe ─► caption ─► figure Chunk[]        kind="figure"
                          │
         all chunks ─► Embedder.embed_documents ─► VectorStore.add (upsert)
```

**Question answering**

```
question ─► Embedder.embed_query ─► Chroma top-k (cosine) ─► drop score < min_score
        ├─ nothing left ──────────────► "I couldn't find that in the documents." (no model call)
        ├─ no generator (no API key) ─► search-only answer + passages
        └─ otherwise ─► build_prompt ─► OpenRouter chat completion ─► answer
                                         └─► find_citations ─► cited source numbers
```

### 1.5 The UI layer

`app.py` is a thin presentation layer. Key behaviours:

- `get_app()` is decorated with `@st.cache_resource`, so the object graph (including the embedding model, which is slow to load) is built **once per server process**, not on every rerun. Changing `.env` or code therefore needs a Streamlit restart.
- Uploaded files are written to a `NamedTemporaryFile(delete=False)` because `pypdf`, `pdfplumber` and `pypdfium2` all need a real path. The file is removed in a `finally` block.
- The sidebar lists indexed documents (`store.documents()`), offers per-document deletion (`store.delete_document(doc_id)`), and reports the active model and whether charts and images are being read.
- Chats are saved in `ChatStore` (SQLite) and the current one is tracked by `st.session_state.chat_id`. Each assistant message stores its `sources` and `cited` lists so the history can be redrawn on every rerun.
- `handle_question()` is the single path for answering: typed and spoken questions both go through it. Voice adds only an input step and an output step around it (section 9).
- The Sources panel shows `[n] file, page p`, a `table` or `figure` tag for non-text chunks, the similarity score, a star for sources the model actually cited, and the passage text.
- Exceptions from `app.ask()` (for example an invalid key, or all models busy) are caught and shown as a message in the chat instead of crashing the page.

---

## 2. Multimodal ingestion pipeline

`DocQA.ingest_pdf()` produces three kinds of chunk from one PDF. All share the `Chunk` dataclass:

```python
@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    doc_id: str
    kind: str = "text"      # "text", "table" or "figure"
```

`kind` defaults to `"text"`, so code and indexes created before multimodal support keep working.

### 2.1 Text extraction (`pypdf`)

`load_pdf(path, source_name)` in `rag/pdf_loader.py`:

1. Opens the file with `pypdf.PdfReader` and iterates the pages, numbering them from 1.
2. Calls `extract_text()` on each page and passes the result through `clean_text()`, which:
   - replaces non-breaking spaces with normal spaces,
   - re-joins words hyphenated across line breaks (`inter-\nnational` becomes `international`),
   - collapses all runs of whitespace to a single space.
3. Skips pages with no text and returns `Page(source, page, text)` objects.
4. Raises `NoTextError` if no page has any text. This usually means a scanned document that needs OCR.

In `ingest_pdf`, `NoTextError` is caught and treated as "zero text pages", because the file may still contain tables or figures. The error is re-raised only if text, tables and figures are **all** empty.

### 2.2 Table processing (`pdfplumber`)

`extract_tables(pdf)` in `rag/multimodal.py` runs `page.extract_tables()` on every page. A table is kept only if it has at least two rows and produces non-empty Markdown.

**Conversion to Markdown (`table_to_markdown`)**

- The first row is the header; a separator row of `---` is generated.
- Each cell is normalised: `None` becomes an empty string, all whitespace (including newlines) collapses to single spaces, and literal `|` is escaped as `\|`.
- Rows that are entirely empty are dropped; short rows are padded to the widest row.

```markdown
| Region | Revenue | Growth |
| --- | --- | --- |
| North | 120 | 5% |
| South | 80 | -2% |
```

**Why Markdown?** An embedding model reads text, not grid geometry. A Markdown table keeps headers next to values, so a question like "what was South's revenue?" lands near the right chunk, and a language model can read the table directly when it is placed in the prompt.

**Splitting long tables (`split_markdown_table`)**: tables longer than `MAX_TABLE_CHARS` (1500 characters) are split by rows, and the header and separator are repeated in every part so each part stays self-describing.

**Chunk construction (`table_chunks`)**

| Property | Value |
|---|---|
| `id` | `{doc_id}-p{page}-t{table_index}`, with a `.{part}` suffix when a table was split |
| `text` | `"Table on page {page}:\n"` followed by the Markdown |
| `kind` | `"table"` |

Table content normally also appears in the page's plain text chunks. That duplication is deliberate: the plain-text version helps keyword-like matches, and the Markdown version preserves structure.

### 2.3 Visual and figure captioning (`pypdfium2` and a vision model)

Figures are handled in three steps: **detect**, **render**, **describe**.

**Detect (`extract_figures`)**: for each page, a page is selected if either condition holds:

1. It contains an embedded image at least `MIN_IMAGE_SIZE` (80 points) wide **and** tall. This filters out logos and icons.
2. It contains **no** images but looks like a chart (`_looks_like_chart`): it has at least `CHART_MIN_CURVES` (8) curved paths, **or** at least `CHART_MIN_SHAPES` (10) rectangles, lines and curves combined, **and** `pdfplumber` finds no table on the page. The table exclusion stops table borders from being mistaken for chart axes.

**Render**: selected pages are rendered with `pypdfium2` at `RENDER_SCALE = 2` (twice the PDF's native 72 dpi) and converted to a `PIL.Image`. The **whole page** is rendered, not just the picture, so the vision model also sees the slide title, axis labels and captions that give the figure its meaning. One `Figure(page, index, image)` is produced per selected page (index `0`).

**Describe (`figure_chunks`)**: each image is passed to a `VisionDescriber`:

```python
class VisionDescriber(Protocol):
    def describe(self, image: Image.Image) -> str: ...
```

`OpenRouterVisionDescriber` (`rag/vision.py`):

1. Converts the image to RGB and encodes it as PNG, then as a base64 `data:image/png;base64,...` URL.
2. Sends one chat completion whose user message has two parts: an `image_url` part and the instruction text.
3. Uses the same retry and fallback logic as the generator (section 5.3).

The instruction asks for the type of visual, title, axis labels and units, legend entries, key numbers, main trends, and a transcription of visible text, without guessing beyond what is shown. If the image is only decoration, the model is told to reply with exactly `DECORATIVE`.

Rules applied by `figure_chunks`:

| Case | Behaviour |
|---|---|
| Caption is empty or starts with `DECORATIVE` | Skipped, not indexed |
| `describe()` raises any exception | That figure is skipped and the others continue |
| Figure id already exists in the store (`known_ids`) | Skipped with no model call |

Resulting chunk: `id = {doc_id}-p{page}-f{index}`, `text = "Figure on page {page}: {caption}"`, `kind = "figure"`.

Because the caption is ordinary text, it is embedded and searched like any other chunk. The image itself is not stored.

**Cost control.** Vision calls are the only per-document paid step at ingestion. They are avoided when no describer is configured (`VISION_PROVIDER=none`, or no API key) and for figures already in the index. Decorative figures are not stored, so they are re-described if the same PDF is uploaded again.

### 2.4 Failure isolation

`DocQA._visual_chunks` wraps table and figure extraction in a `try/except Exception`. If an unusual PDF breaks `pdfplumber` or `pypdfium2`, the document is still indexed from its text. Failures inside individual vision calls are isolated per figure as described above.

---

## 3. Vector storage and embedding strategy

### 3.1 Chunking (`rag/chunker.py`)

`split_text(text, chunk_size=500, overlap=100)`:

- Sizes are measured in **characters**, but chunks are built from whole **words**, so words are never cut in half.
- A chunk grows word by word until the next word would exceed `chunk_size`. A single word longer than `chunk_size` still becomes its own chunk, so the loop always advances.
- The next chunk starts by stepping back over whole words totalling up to `overlap` characters, so consecutive chunks share context at their boundaries.
- `overlap` must be smaller than `chunk_size`, otherwise a `ValueError` is raised.

`chunk_pages()` applies this per page, so **a chunk never spans two pages**. Chunk ids are `{doc_id}-p{page}-c{index}`.

**Why 500 and 100?** The chunk is large enough to hold a full policy statement and small enough that the embedding stays focused on one topic. The 100-character overlap prevents a fact that straddles a boundary from being lost. Both values are constructor arguments of `DocQA` and options of `eval/evaluate.py`, so they can be tuned with measurements (section 6.4).

### 3.2 Document fingerprinting (`file_fingerprint`)

```python
def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
```

The `doc_id` depends **only on the file's bytes**, not its name or location. Consequences:

| Situation | Result |
|---|---|
| Same PDF uploaded twice | Same `doc_id`, same chunk ids, `upsert` overwrites. No duplicates. |
| Same bytes under a different name | Treated as the same document. |
| PDF edited in any way | New `doc_id`, indexed as a new document alongside the old one. |
| Delete a document | `delete_document(doc_id)` removes every chunk (text, table, figure) with that `doc_id`. |

Chunk ids embed `doc_id`, page and position, which makes the whole ingestion **idempotent**: running it again produces exactly the same ids.

Twelve hex characters (48 bits) is ample to avoid collisions in a personal document collection.

### 3.3 Embeddings (`rag/embeddings.py`)

```python
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

`FastEmbedder` wraps `fastembed.TextEmbedding` with `BAAI/bge-small-en-v1.5`:

- Runs locally on CPU through ONNX, with no GPU and no API key. The model (about 65 MB) downloads on first use.
- Produces **384-dimensional** dense vectors.
- Uses `passage_embed()` for stored chunks and `query_embed()` for questions. BGE models are trained asymmetrically: queries get an instruction prefix, passages do not. Using the matching method for each improves retrieval, which is why the protocol has two methods rather than one.
- Text, table Markdown and figure captions are all embedded the same way, so one query searches all three kinds.

Any object with the two methods can replace it, which is how tests inject a fast fake embedder.

### 3.4 Persistent storage (`rag/vector_store.py`)

`VectorStore` wraps `chromadb.PersistentClient(path=...)`:

- **Embedded, serverless.** Chroma runs inside the application process. The app's data directory is `DATA_DIR/chroma` (default `./data/chroma`), containing a SQLite database (`chroma.sqlite3`) and the vector index files. It survives restarts.
- **One collection**, `documents`, created with `embedding_function=None` (the app supplies its own vectors) and `configuration={"hnsw": {"space": "cosine"}}`.
- **Stored per chunk**

| Chroma field | Content |
|---|---|
| `ids` | chunk id |
| `embeddings` | 384-dim vector |
| `documents` | chunk text, Markdown table, or figure caption |
| `metadatas` | `{"source", "page", "doc_id", "kind"}` |

- **Operations**

| Method | Behaviour |
|---|---|
| `add(chunks, embeddings)` | `upsert` by id; no-op for an empty list |
| `search(query_embedding, k)` | Top-k nearest chunks; returns `[]` for an empty store |
| `documents()` | `{doc_id: source}` for the sidebar |
| `delete_document(doc_id)` | Delete by metadata filter `doc_id` |
| `existing_ids(ids)` | Which ids are already indexed (used to skip vision calls) |
| `count()` | Number of chunks |
| `close()` | Release SQLite handles (see section 5.4) |

`meta.get("kind", "text")` is used when reading results, so indexes built before the `kind` field existed still load.

---

## 4. Retrieval and question answering

### 4.1 Similarity search and scoring

`DocQA.retrieve(question)`:

1. Embeds the question with `embed_query`.
2. Asks Chroma for the `top_k` nearest chunks (default 5). Because text, table and figure chunks share one collection and one embedding space, a single query ranks all three together.
3. Converts Chroma's **cosine distance** into a **similarity score**: `score = round(1 - distance, 4)`. For normalised vectors, 1.0 means identical direction and values near 0 mean unrelated.
4. Drops every result with `score < min_score` (default 0.5).

The threshold is the first line of defence against hallucination: if nothing clears it, the question is treated as unanswerable.

### 4.2 Answer routing (`DocQA.ask`)

| Condition | Result |
|---|---|
| Empty or whitespace question | `ValueError("Please type a question.")` |
| No result above `min_score` | `Answer(text="I couldn't find that in the documents.", sources=[])`. **No model call is made.** |
| Results exist but no generator (no API key) | `"Search-only mode: here are the most relevant passages."` plus the passages as sources |
| Results and generator | `generator.generate(question, results)`; `cited` is computed from the reply |

### 4.3 Prompt construction (`rag/generator.py`)

`build_prompt()` wraps each retrieved chunk in a numbered tag. Plain text chunks keep a minimal tag; table and figure chunks add `type`:

```xml
<sources>
<source id="1" file="handbook.pdf" page="2">Full-time employees receive 14 days ...</source>
<source id="2" file="report.pdf" page="3" type="table">Table on page 3: | Region | ... </source>
<source id="3" file="report.pdf" page="4" type="figure">Figure on page 4: Bar chart ... </source>
</sources>

Question: How many days of annual leave do employees get?
```

The `id` is the 1-based rank of the result. It is what the model cites and what the UI uses to label sources. If there are no sources the block contains `(no sources found)`.

### 4.4 Hallucination-prevention rules

The system prompt (`SYSTEM_PROMPT`) enforces five rules:

1. Answer **only** from the provided sources.
2. Be concise.
3. If the sources do not contain the answer, reply **exactly** `I couldn't find that in the documents.` A fixed phrase makes the refusal predictable and easy to test.
4. Cite every factual claim with its source number, such as `[1]`.
5. Do not invent information or citations. The prompt also explains that sources may be text, Markdown tables or figure descriptions, and that all are valid evidence.

These prompt rules work together with the code-level guards:

| Layer | Mechanism |
|---|---|
| Retrieval | `min_score` removes weak matches |
| Routing | No retrieved evidence means no model call and a fixed "not found" answer |
| Prompt | Closed-book instruction plus a fixed refusal sentence |
| Output validation | `find_citations()` extracts `[n]` markers and keeps only numbers between 1 and the number of sources, so invented citation numbers are discarded |
| Empty model reply | `OpenRouterGenerator` falls back to the "not found" sentence if the model returns no content |

Prompt rules reduce hallucination but cannot guarantee that it never happens. The Sources panel lets a user check every claim against the original page.

### 4.5 Citations in the UI and CLI

`find_citations(answer, n_sources)` returns the sorted set of valid cited numbers. The UI marks those sources with a star, and the CLI prefixes them with `*`. Every source shows its file, page, kind (for tables and figures) and score.

---

## 5. Configuration and environment management

### 5.1 Settings (`rag/config.py`)

`load_settings()` calls `load_dotenv(PROJECT_DIR / ".env")` and reads the environment. Real environment variables take precedence over `.env` values (the `python-dotenv` default), which lets tests override settings with `monkeypatch.setenv`.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto`, `gemini` or `openrouter`. `auto` picks Gemini when `GEMINI_API_KEY` is set, otherwise OpenRouter. |
| `GEMINI_API_KEY` | none | Enables answers, vision and (by default) voice through Gemini. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Model used for answers when Gemini is the provider. |
| `GEMINI_FALLBACK_MODEL` | none | Optional second Gemini model tried if the main one stays unavailable. |
| `GEMINI_REASONING` | none | Optional `reasoning_effort` (for example `low`) sent with every Gemini request. |
| `OPENROUTER_API_KEY` | none | Enables answers and vision through OpenRouter. With neither key set, the app is in search-only mode. |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Model used for answers when OpenRouter is the provider. Any OpenRouter model id. |
| `OPENROUTER_FALLBACK_MODEL` | none | Optional second model tried if the main model stays unavailable. |
| `VISION_PROVIDER` | `auto` | `auto` (describe figures with the answering provider) or `none` (skip figure captioning). |
| `VISION_MODEL` | same as the answer model | Model used for images; it must accept image input. |
| `OPENROUTER_SITE_URL` | `http://localhost:8501` | Sent as the optional `HTTP-Referer` attribution header. |
| `OPENROUTER_APP_NAME` | `Document Q&A` | Sent as the optional `X-Title` attribution header. |
| `DATA_DIR` | `./data` | Parent folder of the Chroma database. |
| `TOP_K` | `5` | Passages sent to the model. |
| `MIN_SCORE` | `0.5` | Minimum similarity for a passage to count. |

`.env` is listed in `.gitignore`; `.env.example` documents the variables with placeholders.

### 5.2 Composition root (`build_app`)

`build_app(settings=None)` is the one place where real objects are created and wired:

1. Always creates the `VectorStore` and `FastEmbedder`.
2. `build_llm_client()` returns the chat client of the chosen provider (`make_gemini_client()` or `make_client()`), or `None` without a key. With a client, it builds the `OpenRouterGenerator`, rewriter, image judge and, unless `VISION_PROVIDER=none`, the `OpenRouterVisionDescriber`, which all share that client. Despite their names these classes only need an OpenAI-style `chat.completions.create`, so they run on either provider unchanged.
3. Returns `DocQA(store, embedder, generator, top_k, min_score, describer)`.

Heavy imports happen inside the function, so importing `rag.config` stays cheap. Because `DocQA` receives its dependencies instead of creating them, tests construct it directly with fakes and never touch the network.

### 5.3 OpenRouter client, retry and fallback (`rag/openrouter.py`)

OpenRouter exposes an OpenAI-compatible API. The app uses the standard `openai.OpenAI` client with:

- `base_url = "https://openrouter.ai/api/v1"`
- `api_key` from `OPENROUTER_API_KEY`
- optional `HTTP-Referer` and `X-Title` default headers
- `max_retries=0` and `timeout=60`; the SDK's own retries are disabled so that the application controls them

`call_with_retry(call, models, attempts=3, delay=2.0)`:

- For each model in order, tries up to 3 times, waiting 2 s and then 4 s between attempts.
- Retries only **temporary** conditions: HTTP 408, 429, 500, 502, 503, 504, plus connection errors and timeouts.
- Raises any other error **immediately** (for example 401 invalid key or 400 bad request), since retrying cannot fix it.
- After exhausting all models it raises `OpenRouterBusyError` with a readable message, which the UI shows in the chat.

The generator sends the standard payload:

```python
client.chat.completions.create(
    model=model,
    max_tokens=2048,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(question, results)},
    ],
)
```

`max_tokens` is 2048 rather than a small number because some models spend part of the output budget on internal reasoning, which could otherwise leave no room for the answer.

### 5.4 Windows-specific considerations

**SQLite file locks.** Chroma keeps `chroma.sqlite3` open for the life of its client. On Windows an open file cannot be deleted, so removing a temporary index directory fails with `PermissionError: [WinError 32]`. The project handles this in two ways:

- `VectorStore.close()` calls `client.close()` and `clear_system_cache()`, releasing the handle and dropping Chroma's cached client for that path.
- `eval/evaluate.py` creates the store inside `tempfile.TemporaryDirectory(ignore_cleanup_errors=True)` and calls `store.close()` in a `finally` block **before** the directory is removed. The `ignore_cleanup_errors` flag is a safety net so a stray lock cannot crash a run that has already produced its results.

**Temporary upload files.** The Streamlit upload handler writes to `NamedTemporaryFile(delete=False)`, closes it, processes it, and deletes it in `finally` with `Path.unlink(missing_ok=True)`. With the default `delete=True`, Windows would refuse to let a second reader open the still-open file.

**Event loops and `asyncio`.** The application code contains no `asyncio` code and creates no event loops. Streamlit manages its own runtime, and the OpenAI client is used synchronously. The only visible trace is a `DeprecationWarning` about `asyncio.iscoroutinefunction` raised inside the `chromadb` package under Python 3.14; it comes from the library, is harmless, and appears in test output. If the project later moves to async clients, calls into the synchronous pipeline from an async context should run in a worker thread so they do not block the loop.

**Paths and shells.** Paths are built with `pathlib`, so Windows and POSIX separators both work. `README.md` gives PowerShell activation (`.venv\Scripts\Activate.ps1`) and `copy` commands alongside the POSIX equivalents.

### 5.5 Gemini (`rag/gemini.py`)

Gemini is reached through its OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`) with the same `openai` SDK, so the generator, rewriter, image judge and vision describer are unchanged. `GeminiClient` wraps `openai.OpenAI` and makes two adjustments:

- **Thinking headroom.** Gemini's newer models think before answering, and thinking tokens count against `max_tokens`. Every request gets `THINKING_HEADROOM` (1,024) extra tokens, so a small limit such as the rewriter's 256 cannot be used up before any answer text appears.
- **Optional `reasoning_effort`.** Sent only when `GEMINI_REASONING` is set; otherwise the model's own default is used.

It sends no OpenRouter attribution headers, keeps `max_retries=0` and `timeout=60`, and relies on `call_with_retry` exactly as before. `OpenRouterBusyError` keeps its name but its message is provider-neutral.

---

## 6. Testing and quality assurance

### 6.1 Strategy

The suite runs with `pytest` and needs **no network, API key or paid calls**. External systems are replaced by fakes defined in `tests/conftest.py` and `tests/test_multimodal.py`.

| Fake | Replaces | Behaviour |
|---|---|---|
| `FakeEmbedder` | `FastEmbedder` | Hashes words into a 256-slot bag-of-words vector, so texts that share words are close. Fast and deterministic. |
| `FakeGenerator` | `OpenRouterGenerator` | Records `(question, results)` and returns a fixed answer with a citation. |
| `FakeDescriber` | `OpenRouterVisionDescriber` | Returns a fixed caption and counts calls. It also asserts it received a real rendered page image. |
| `FakeTranscriber` | `WhisperTranscriber` | Returns a fixed sentence for any recording and keeps the recordings it received. |
| `FakeSpeaker` | `PiperSpeaker`, `Pyttsx3Speaker` | Returns a tiny fake WAV and records the text it was asked to say. |
| Fake OpenAI clients | `openai.OpenAI` | Expose `chat.completions.create`, record the payload, and can raise `APIStatusError` with a chosen status code. |

PDFs for multimodal tests are generated on the fly with `reportlab`: a text page, a page with a real grid table, a page with an embedded image, and a page with a bar chart.

### 6.2 Test modules

194 tests in total:

| Module | Tests | Focus |
|---|---|---|
| `tests/test_multimodal.py` | 32 | Table to Markdown conversion and cell cleaning; splitting long tables with repeated headers; table detection in a PDF; table chunk ids and metadata; figure detection (image and chart pages only); figure chunks via the describer; decorative, empty and failing captions skipped; known figures not re-described; the OpenRouter vision payload (`image_url` data URL); pipeline ingestion counts; table and chart questions retrieving the right page and kind; re-ingestion making no vision calls; SAM-style crops and fallback; logos, repeats and blank crops dropped before any vision call; chart boxes keep their axis labels; a header logo does not hide a chart; saved crops and `image_path`; no-describer mode; unchanged behaviour for plain-text PDFs; prompt labelling of `type="table"` and `type="figure"`. |
| `tests/generator.py` | 16 | Prompt numbering; citation extraction ignoring invalid numbers; chat-completions payload (model, `max_tokens`, system and user messages); empty reply fallback; client base URL and attribution headers; retry on 503; fallback model; friendly error when everything is busy; no retry for a bad key; images sent as `image_url` parts on the vision model; history order; plain-text request when there are no images. |
| `tests/pipeline.py` | 7 | Ingestion report; idempotent re-ingestion; relevant chunks reach the generator; unrelated question skips the model; search-only mode; empty question rejected; deleting a document removes its figure files. |
| `tests/chunker.py` | 7 | Split sizes, overlap, page-bounded chunks, invalid parameters. |
| `tests/pdf_loader.py` | 6 | Text cleaning, page numbering, empty-text error. |
| `tests/vector_store.py` | 5 | Ranking, empty store, duplicate-free upserts, document listing and deletion. |
| `tests/segmenter.py` | 7 | Box geometry; filtering tiny, whole-page and thin boxes; merging; cap and ordering; no-SAM fallback; SAM keeps the hint when its outline is much smaller; the **real MobileSAM** finds an embedded image (`slow`). |
| `tests/image_selection.py` | 14 | Same-page figures as candidates; unrelated question gets no image and no judge call; judge choices, none, cap, bad indexes; fallback when the judge fails; judge reply parsing; pipeline wiring. |
| `tests/memory.py` | 17 | Saved chats, titles, reopening, deletion, no locked file; history trimming; follow-up rewriting and its fallbacks; history reaches the answer model. |
| `tests/sample_report.py` | 5 | The sample annual report keeps its layout: figures only on pages 2, 4 and 5, a logo on every page that is not a figure, the table found and not mistaken for a chart, and numbers that add up. |
| `tests/app.py` | 8 | Headless Streamlit run (`streamlit.testing.v1.AppTest`): sidebar, search-only warning, a question, and the cited page in the sources; chats are saved, listed in the sidebar, reopen in a new session; New chat starts empty. Voice (with fakes): off by default in tests, sidebar status and microphone prompt, a spoken question is answered and played, typed questions are not read aloud but the Listen button works, a speech failure keeps the written answer. |
| `tests/voice.py` | 54 | `speech_text` (citations, Markdown, icons, tables, code, truncation); `question_from_input` (typed, spoken, both, silent); Whisper wrapper (segments, silence, errors, lazy loading); Piper and pyttsx3 wrappers (WAV output, empty text, errors, temp files removed); Gemini speech to text (audio payload, language hint, no-speech reply, errors, busy service); Gemini text to speech (request shape, WAV wrapping, retries, no retry for bad requests, key never in errors, empty or non-audio replies); voice settings; `build_voice` choosing Gemini or local engines, mixing them, and falling back without a key; a real Piper to Whisper round trip (`slow`). |
| `tests/gemini.py` | 14 | `GeminiClient` (thinking headroom, optional reasoning effort, endpoint and headers, works with the existing rewriter); provider selection (`auto`, forced, both keys, missing key, unknown value); per-provider models; voice defaults following the key; `build_llm_client`. |
| `tests/retrieval_quality.py` | 2 | Retrieval quality on the sample handbook with the real embedding model. |

Tests that load a real model (embeddings, SAM, Whisper and Piper) are marked `slow` (declared in `pytest.ini`).

### 6.3 Running the tests

The test files are named without the `test_` prefix (apart from `test_multimodal.py`), so pytest needs to be told to collect them all:

```bash
python -m pytest tests -o python_files="*.py"                  # everything
python -m pytest tests -o python_files="*.py" -m "not slow"    # skip real-model tests
python -m pytest tests/test_multimodal.py                      # multimodal tests only
```

### 6.4 Retrieval evaluation (`eval/evaluate.py`)

Beyond unit tests, `eval/evaluate.py` measures retrieval quality on the sample handbook using 18 questions in `eval/questions.json`, each mapped to the page that holds its answer. It builds a throwaway index (never touching real data) with `generator=None` and reports:

- **hit@1**: share of questions whose best result is on the right page,
- **hit@k**: share where the right page is anywhere in the top k,
- **MRR**: mean reciprocal rank (1.0 if the right page is always first).

Chunk size, overlap and `k` are command-line options, so settings can be compared with numbers. On the sample handbook the default settings score 100% hit@1, 100% hit@3 and MRR 1.00.

### 6.4.1 What the tests do not cover

- Real network calls to OpenRouter and Gemini (answers, speech to text, text to speech) and the quality of real vision captions. Payloads and error handling are tested with fakes; live quality depends on the chosen model. The Gemini text-to-speech request follows Google's documented Interactions API and has not been run against the live service in this repository's tests.
- OCR for scanned PDFs (such files are reported, not read).
- The heuristics for chart detection on arbitrary real-world PDFs; they are tested on generated PDFs only.
- The browser microphone itself. `AppTest` can only type text, so the recording step is covered by `question_from_input` and by tests that stand in for it; recognition quality with real voices and accents depends on the Whisper model chosen.

---

## 7. Known limitations and extension points

**Limitations**

- Scanned PDFs without a text layer are not OCR'd. A page of one big picture is treated as a single figure and captioned.
- Chart detection is heuristic (counts of drawing primitives and images). A chart drawn with few shapes may be missed, and a decorative page may occasionally be flagged. Thresholds are constants at the top of `rag/multimodal.py`.
- Figure crops are saved under `data/figures/`, but the vision caption is what is searched. A figure with a poor caption is hard to find.
- Embeddings are English-oriented (`bge-small-en-v1.5`).
- Editing a PDF produces a new `doc_id`; the old version stays indexed until deleted.
- Figure captioning is one vision call per figure page, which costs money and time on image-heavy documents.

**Extension points** (each is isolated behind an existing interface)

| Change | Where |
|---|---|
| Different embedding model | Implement `Embedder` or pass another `model_name` to `FastEmbedder`. |
| Different LLM or provider | Implement `Generator`; pass it to `DocQA`. |
| Different vision model | Set `VISION_MODEL`, or implement `VisionDescriber`. |
| Another store | Replace `VectorStore`, keeping `add`, `search`, `documents`, `delete_document`, `existing_ids`. |
| Tune retrieval | `TOP_K`, `MIN_SCORE`, `chunk_size`, `overlap`; measure with `eval/evaluate.py`. |
| Add a content type | Produce `Chunk` objects with a new `kind` in `rag/multimodal.py` and add them in `DocQA.ingest_pdf`. |

---

## 8. Image-aware answers and conversation memory

### 8.1 Cutting figures out with SAM

Earlier versions rendered the **whole page** of any page with a chart and captioned it. Now each figure is cut out and kept.

1. `page_hint_boxes()` asks `pdfplumber` where embedded images and chart drawings are (rough boxes, in rendered pixels).
2. The page is rendered once with `pypdfium2` at `RENDER_SCALE = 2`.
3. `SamSegmenter` (`rag/segmenter.py`) gives those boxes to **MobileSAM** (through `ultralytics`) as prompts and gets a tighter outline for each. If SAM's outline is less than half or more than twice the hint's area, the hint is kept, because SAM sometimes outlines only the bars of a chart.
4. `filter_boxes()` drops boxes under 1% or over 95% of the page and very thin strips (aspect over 8:1), merges overlapping or nested boxes, keeps at most four per page, and orders them top to bottom.
5. Each box is cropped with a small margin. The page text is attached to the crop as context for the vision model, so titles and captions are not lost.

SAM finds **outlines**. It cannot tell a logo from a chart. If SAM or PyTorch is missing, fails, or `SEGMENTER=none`, `PassThroughSegmenter` crops the hint boxes instead, and a SAM error on one page never loses that page's figure.

### 8.2 Dropping logos and decoration

Checked **before** any paid vision call (`is_probably_decorative`):

| Rule | Catches |
|---|---|
| Small (under 8% of the page) and in the top or bottom 10% | Headers, footers, logos |
| Small and the same 8x8 average hash on 2 or more pages | A logo repeated on every page |
| Nearly one colour (pixel standard deviation under 6) | Blank boxes, backgrounds |

Then the vision model gets the crop and may answer exactly `DECORATIVE`, which also drops it. Kept figures are saved as `data/figures/{doc_id}/p{page}-f{i}.png`, and the path is stored in the chunk metadata as `image_path`. Chunk ids are unchanged (`{doc_id}-p{page}-f{i}`), so re-ingestion still skips figures that are already indexed. `DocQA.delete_document()` removes both the chunks and the figure folder.

### 8.3 Deciding whether a question needs an image

`ImageSelector.select()` runs after normal retrieval:

1. **Candidates**: figure chunks among the results, plus figures on the **same document and page** as any retrieved chunk (a paragraph often refers to the chart next to it). Only figures whose image file still exists count.
2. **Stage 1, similarity (free)**: cosine similarity between the question and each figure's stored caption embedding. Below `IMAGE_MIN_SCORE` (0.45), the figure is dropped. At most four go on.
3. **Stage 2, judge (one small text-only call)**: `OpenRouterImageJudge` lists the captions as `F1..Fn` and returns JSON such as `{"use": ["F2"]}`. "None" is a valid answer. At most `MAX_IMAGES` (2) are kept.

No candidates after stage 1 means no judge call and no image. If the judge call fails, only a very close match (score 0.6 or more) is shown, and in search-only mode the same rule is used without a judge. A figure found through a nearby paragraph is added to the sources so it gets a number, which keeps citations as `[n]`.

The generator then sends the images to the model (`image_url` parts, shrunk to 1024 px) using `VISION_MODEL`, and the system prompt asks it to explain what the image shows and cite its source number. With no images the request is the same plain-text request as before.

### 8.4 Conversation memory

Chats are stored with `sqlite3` in `DATA_DIR/chats.sqlite3` (`ChatStore`). A connection is opened and closed on every call, so no file lock is left behind on Windows. Each message stores its sources, cited numbers, images and the rewritten query as JSON, so a reopened chat looks exactly as it did.

For a follow-up question:

1. `recent_history()` takes the last `MEMORY_MESSAGES` (6) messages within a 4,000-character budget, skips failed answers, and starts at a user message.
2. `OpenRouterRewriter` turns the question into a standalone one ("What about part-time staff?" becomes "How many days of annual leave do part-time employees get?"). With no history there is **no call**. On any error the original question is used.
3. The rewritten question is embedded and searched, and the image selector uses it too.
4. The answer model receives the history as earlier chat messages, but the prompt says facts must come from the sources.

The UI shows "Searched for: ..." when the rewrite differs from what the user typed.

### 8.5 New settings

| Variable | Default | Purpose |
|---|---|---|
| `SEGMENTER` | `sam` | `sam` or `none` |
| `SAM_WEIGHTS` | `DATA_DIR/models/mobile_sam.pt` | Weights file; downloaded on first use |
| `IMAGE_MIN_SCORE` | `0.45` | Caption similarity needed before a figure is considered |
| `MAX_IMAGES` | `2` | Most figures sent with one question |
| `HELPER_MODEL` | same as `OPENROUTER_MODEL` | Small model for rewriting and image judging |
| `MEMORY_MESSAGES` | `6` | Earlier messages sent with a question |

### 8.6 Limits of this design

- The judge decides from the **caption**, not the picture, so a wrong caption can hide a useful figure.
- Only the last few messages are used; very long chats are not summarised.
- Text-only helper calls (rewriting, judging) add a small cost to a question, but only when there is history or a candidate figure.
- SAM runs on CPU and adds about a second or more per figure page at ingestion.


---

## 9. Voice assistant

Voice is an input and output layer around the existing pipeline. `DocQA`, retrieval, memory and citations are unchanged.

```
microphone (browser)
   |  WAV, 16 kHz  (st.chat_input(accept_audio=True))
   v
question_from_input()  -->  Transcriber.transcribe()  (Gemini or Whisper)      rag/voice.py
   |  text + "spoken" flag
   v
handle_question()  -->  DocQA.ask()  -->  answer + sources        app.py, rag/pipeline.py
   |  answer text (Markdown, [n] citations)
   v
speech_text()  -->  Speaker.synthesize()  (Gemini or Piper)  -->  st.audio(autoplay)
```

### 9.1 Components (`rag/voice.py`)

- `Transcriber` and `Speaker` are `Protocol`s, like every other external dependency. `build_voice()` in `rag/config.py` is the only place the real engines are created.
- `WhisperTranscriber` wraps `faster-whisper`. The model loads on the first recording, not at start-up. The audio is passed as an in-memory file, so no temporary file is needed. `vad_filter=True` removes silence, and a recording with no speech gives an empty string.
- `PiperSpeaker` wraps `piper-tts` and returns WAV bytes. `Pyttsx3Speaker` uses the operating system's voice (SAPI5 on Windows) and is the fallback when no Piper voice file is present. It makes a fresh engine per call and initialises COM in the calling thread, because Streamlit runs each session in a worker thread.
- `GeminiTranscriber` sends the recording to a Gemini model as an `input_audio` part of an ordinary chat request (through the same `GeminiClient`, so `call_with_retry` and the fallback model apply). The prompt asks for the transcript only, and a reply of `NO_SPEECH` becomes an empty string, which the UI reports as "I didn't catch that."
- `GeminiSpeaker` calls Gemini's native speech endpoint (`POST /v1beta/interactions`, key in the `x-goog-api-key` header) with a TTS model and a prebuilt voice, and decodes the base64 audio from `steps[].content[]`. If the audio has no WAV header, `pcm_to_wav()` adds one (24 kHz, 16-bit, mono). Busy responses (408, 429, 5xx) and network errors are retried with growing waits; bad requests are not, and the key never appears in an error message. Gemini TTS is not part of the OpenAI-compatible API, so it uses `httpx` directly.
- `speech_text()` prepares an answer for the ear: it removes `[n]` markers (the same pattern `find_citations` reads), Markdown, `:material/...:` icons, links and code blocks, reads tables as plain words, and cuts long answers at a sentence boundary with "See the full answer on screen."
- `question_from_input()` decides what the chat bar returned: typed text wins, otherwise a recording is transcribed. It is a pure function so it can be tested without a browser.
- All engine failures are raised as `VoiceError`, whose message is safe to show.

### 9.2 Graceful degradation

`build_voice()` follows the same pattern as `build_segmenter()`. If `faster-whisper` is missing, `VOICE_STT=none`, or a voice file is absent, that half of the feature is switched off with a log warning and the app keeps working:

| Situation | Result |
|---|---|
| No voice packages installed | No microphone in the chat bar; the sidebar says how to install voice |
| Gemini voice chosen but no `GEMINI_API_KEY` | The local engines are used instead, with a warning |
| Gemini is busy or unreachable for one request | Retried a few times, then the error is shown (speech to text) or a warning appears under the written answer (text to speech); local engines are not tried automatically |
| Piper voice file missing | Falls back to the system voice (`pyttsx3`) |
| Neither speech output works | The microphone still works; answers are shown but not spoken |
| Speech output fails on one answer | The written answer is saved and shown, with a warning under it |
| Recording has no speech | "I didn't catch that." and nothing is sent |

### 9.3 UI behaviour (`app.py`)

- The microphone is the built-in one in `st.chat_input(accept_audio=True)`, shown only when a transcriber exists. One submit returns text or a recording, so the same recording cannot be processed twice on a rerun.
- A spoken question is saved with `voice=True` in its message metadata (`ChatStore.add_message(**meta)` already stores extra fields) and is shown with a "Spoken question" caption.
- The answer to a spoken question is turned into audio while the question is handled and kept in `st.session_state.pending_audio`. The page then reruns to redraw the sidebar, and the audio is played under the newest assistant message on that rerun, then discarded.
- Typed questions are not read aloud. Every saved assistant answer has a **Listen** button that speaks it on demand. `speak()` is wrapped in `st.cache_data(max_entries=50)` keyed on the text, so replaying costs nothing and no audio is stored in SQLite.
- `get_voice()` is `@st.cache_resource`, so the voice loads once per server process (changing voice settings needs a restart, like the rest of `.env`).

### 9.4 Settings

| Variable | Default | Purpose |
|---|---|---|
| `VOICE_STT` | `gemini` with a Gemini key, else `whisper` | `gemini`, `whisper` or `none` |
| `WHISPER_MODEL` | `base` | faster-whisper model size; the multilingual models detect the language |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE` | `int8` | Quantisation (`int8` is fast on CPU) |
| `VOICE_LANGUAGE` | empty | Force a spoken language; empty detects it |
| `VOICE_TTS` | `gemini` with a Gemini key, else `piper` | `gemini`, `piper`, `pyttsx3` or `none` |
| `PIPER_VOICE` | `DATA_DIR/models/piper/en_US-lessac-medium.onnx` | Piper voice (`.onnx` with its `.json` beside it) |
| `VOICE_AUTOPLAY` | `true` | Start playing a spoken answer right away |
| `GEMINI_STT_MODEL` | `gemini-3.8-flash` | Gemini model that transcribes recordings |
| `GEMINI_TTS_MODEL` | `gemini-3.8-flash-tts` | Gemini model that speaks answers |
| `GEMINI_VOICE` | `Kore` | Gemini prebuilt voice name |

`python scripts/download_voice_models.py` downloads the Whisper model and the Piper voice once. Voice packages are in `requirements-voice.txt`, separate from `requirements.txt`, so the core install stays small.

### 9.5 Limits of this design

- It is push to talk: one recording per question. There is no hands-free listening, interruption of a spoken answer, or streaming of the answer as it is generated.
- Everything runs on CPU. Expect about 1 to 2 seconds to transcribe a short question with `base`, plus the time the answer model already takes. The first recording also loads the model.
- The default Piper voice is English. Answers in another language need a matching Piper voice in `PIPER_VOICE`; Whisper itself understands many languages.
- Browsers allow the microphone only on `localhost` or HTTPS, and may block autoplay until the page has been clicked. The Listen button always works.
- `requirements-voice.txt` pins `av<17`: `faster-whisper` 1.2 calls `av.open(metadata_errors=...)`, which newer PyAV releases removed.
- Gemini voice sends the recording, and the answer text, to Google, and needs internet. It is subject to the account's rate limits; the local engines are the offline option.
- Gemini model ids change over time (older ones are retired). If a request fails with "model not found", set `GEMINI_MODEL`, `GEMINI_STT_MODEL` or `GEMINI_TTS_MODEL` to a current id from Google's model list.
