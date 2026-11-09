"""Tests for conversation memory: saved chats, history trimming, follow-up rewriting."""
from types import SimpleNamespace

from rag.memory import ChatStore, OpenRouterRewriter, recent_history
from rag.models import Turn
from rag.pipeline import NOT_FOUND, DocQA
from rag.vector_store import VectorStore
from tests.conftest import FakeGenerator


# ---------------- saved chats ----------------

def test_chat_is_saved_and_named_after_the_first_question(tmp_path):
    store = ChatStore(tmp_path / "chats.sqlite3")
    chat = store.create_chat()
    store.add_message(chat, "user", "How many days of annual leave do I get?")
    store.add_message(chat, "assistant", "14 days [1].", cited=[1], sources=[{"page": 2}])
    assert store.list_chats()[0]["title"] == "How many days of annual leave do I get?"
    msgs = store.messages(chat)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["cited"] == [1] and msgs[1]["sources"] == [{"page": 2}]


def test_long_titles_are_shortened_and_later_questions_keep_the_title(tmp_path):
    store = ChatStore(tmp_path / "c.sqlite3")
    chat = store.create_chat()
    store.add_message(chat, "user", "word " * 40)
    title = store.list_chats()[0]["title"]
    assert len(title) <= 60 and title.endswith("...")
    store.add_message(chat, "user", "a second question")
    assert store.list_chats()[0]["title"] == title


def test_chats_survive_reopening_the_file(tmp_path):
    path = tmp_path / "c.sqlite3"
    chat = ChatStore(path).create_chat()
    ChatStore(path).add_message(chat, "user", "hello")
    assert ChatStore(path).messages(chat)[0]["content"] == "hello"


def test_most_recent_chat_is_listed_first_and_delete_removes_messages(tmp_path):
    store = ChatStore(tmp_path / "c.sqlite3")
    first, second = store.create_chat(), store.create_chat()
    store.add_message(first, "user", "older chat, newer message")
    assert [c["id"] for c in store.list_chats()] == [first, second]
    store.delete_chat(first)
    assert store.exists(first) is False and store.messages(first) == []
    assert [c["id"] for c in store.list_chats()] == [second]


def test_database_file_can_be_deleted_after_use(tmp_path):
    """No connection is kept open (a locked file cannot be deleted on Windows)."""
    path = tmp_path / "c.sqlite3"
    store = ChatStore(path)
    store.add_message(store.create_chat(), "user", "hi")
    path.unlink()


# ---------------- history ----------------

def msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


def test_recent_history_keeps_only_the_last_messages():
    history = recent_history(msgs(*[(("user", "assistant")[i % 2], f"m{i}") for i in range(10)]),
                             max_messages=4)
    assert [t.content for t in history] == ["m6", "m7", "m8", "m9"]   # m6 is a user message


def test_recent_history_starts_with_a_question_and_skips_errors():
    messages = msgs(("user", "q1"), ("assistant", "a1")) + [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "Something went wrong", "error": True},
        {"role": "user", "content": "q3"},
    ]
    history = recent_history(messages)
    assert [t.content for t in history] == ["q1", "a1", "q2", "q3"]


def test_recent_history_respects_the_character_budget():
    long = "x" * 1000
    history = recent_history(msgs(("user", long), ("assistant", long), ("user", long),
                                  ("assistant", long)), max_chars=2500)
    assert sum(len(t.content) for t in history) <= 2500
    assert history[0].role == "user"


def test_recent_history_of_nothing_is_empty():
    assert recent_history([]) == []


# ---------------- rewriting follow-ups ----------------

def fake_client(reply):
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), sent


HISTORY = [Turn("user", "How many days of annual leave do full-time employees get?"),
           Turn("assistant", "Full-time employees get 14 days [1].")]


def test_no_history_means_no_rewrite_call():
    client, sent = fake_client("should not be used")
    assert OpenRouterRewriter(client).rewrite([], "What is the notice period?") == \
        "What is the notice period?"
    assert sent == []


def test_rewriter_sends_the_conversation_and_returns_one_clean_line():
    client, sent = fake_client('"How many days of annual leave do part-time employees get?"\nextra')
    out = OpenRouterRewriter(client).rewrite(HISTORY, "What about part-time staff?")
    assert out == "How many days of annual leave do part-time employees get?"
    user = sent[0]["messages"][1]["content"]
    assert "annual leave" in user and "Latest message: What about part-time staff?" in user


def test_an_empty_or_huge_rewrite_falls_back_to_the_original():
    client, _ = fake_client("")
    assert OpenRouterRewriter(client).rewrite(HISTORY, "and then?") == "and then?"
    client, _ = fake_client("y" * 900)
    assert OpenRouterRewriter(client).rewrite(HISTORY, "and then?") == "and then?"


class FakeRewriter:
    def __init__(self, mapping=None, fail=False):
        self.mapping, self.fail, self.calls = mapping or {}, fail, []

    def rewrite(self, history, question):
        self.calls.append((list(history), question))
        if self.fail:
            raise RuntimeError("model busy")
        return self.mapping.get(question, question)


def make_qa(tmp_path, fake_embedder, rewriter, generator=None):
    return DocQA(VectorStore(str(tmp_path / "db")), fake_embedder,
                 generator or FakeGenerator(), min_score=0.2, rewriter=rewriter)


def test_follow_up_is_searched_with_the_rewritten_question(tmp_path, fake_embedder, sample_pdf):
    rewriter = FakeRewriter({"What about carrying over unused days?":
                             "Up to how many days of unused annual leave can be carried over"})
    qa = make_qa(tmp_path, fake_embedder, rewriter)
    qa.ingest_pdf(sample_pdf)
    answer = qa.ask("What about carrying over unused days?", history=HISTORY)
    assert answer.search_query.startswith("Up to how many days")
    assert answer.sources[0].chunk.page == 2           # the leave policy page
    assert rewriter.calls[0][0] == HISTORY


def test_the_answer_model_receives_the_history(tmp_path, fake_embedder, sample_pdf):
    gen = FakeGenerator()
    qa = make_qa(tmp_path, fake_embedder, FakeRewriter(), gen)
    qa.ingest_pdf(sample_pdf)
    qa.ask("How many days of paid annual leave?", history=HISTORY)
    assert gen.histories[0] == HISTORY
    assert gen.calls[0][0] == "How many days of paid annual leave?"   # the user's own words


def test_first_question_is_not_rewritten(tmp_path, fake_embedder, sample_pdf):
    rewriter = FakeRewriter()
    qa = make_qa(tmp_path, fake_embedder, rewriter)
    qa.ingest_pdf(sample_pdf)
    qa.ask("How many days of paid annual leave?")
    assert rewriter.calls == []


def test_a_failing_rewriter_does_not_break_the_answer(tmp_path, fake_embedder, sample_pdf):
    qa = make_qa(tmp_path, fake_embedder, FakeRewriter(fail=True))
    qa.ingest_pdf(sample_pdf)
    answer = qa.ask("How many days of paid annual leave?", history=HISTORY)
    assert answer.search_query == "How many days of paid annual leave?"
    assert answer.sources


def test_unanswerable_follow_up_still_skips_the_model(tmp_path, fake_embedder, sample_pdf):
    gen = FakeGenerator()
    qa = DocQA(VectorStore(str(tmp_path / "db")), fake_embedder, gen, min_score=0.9,
               rewriter=FakeRewriter())
    qa.ingest_pdf(sample_pdf)
    answer = qa.ask("zebra quantum helicopter", history=HISTORY)
    assert answer.text == NOT_FOUND and gen.calls == []
