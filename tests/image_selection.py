"""Tests for deciding which figures to show: similarity stage, judge stage, fallbacks."""
import pytest
from PIL import Image

from rag.image_selector import ImageSelector, OpenRouterImageJudge, parse_judge_reply
from rag.models import Chunk
from rag.pipeline import DocQA
from rag.vector_store import VectorStore
from tests.conftest import FakeGenerator

SALES = "Figure on page 2: Bar chart of quarterly sales revenue rising from Q1 to Q4."
ORG = "Figure on page 2: Diagram of the company organisation structure with departments."


class FakeJudge:
    def __init__(self, picks=(), fail=False):
        self.picks, self.fail, self.calls = list(picks), fail, []

    def pick(self, question, candidates):
        self.calls.append((question, [c.chunk.id for c in candidates]))
        if self.fail:
            raise RuntimeError("model busy")
        return self.picks


@pytest.fixture
def store(tmp_path, fake_embedder):
    """Page 2 holds a paragraph, a sales chart and an org chart. Page 3 holds a paragraph."""
    store = VectorStore(str(tmp_path / "db"))
    (tmp_path / "img").mkdir()
    chunks = []
    for name, text in [("sales", SALES), ("org", ORG)]:
        path = tmp_path / "img" / f"{name}.png"
        Image.new("RGB", (40, 40), "blue").save(path)
        chunks.append(Chunk(f"d-p2-{name}", text, "r.pdf", 2, "d", "figure", str(path)))
    chunks.append(Chunk("d-p2-c0", "Quarterly sales revenue grew strongly this year.",
                        "r.pdf", 2, "d"))
    chunks.append(Chunk("d-p3-c0", "The leave policy gives 14 days of annual leave.",
                        "r.pdf", 3, "d"))
    store.add(chunks, fake_embedder.embed_documents([c.text for c in chunks]))
    yield store
    store.close()


def run(store, fake_embedder, question, judge, **kw):
    selector = ImageSelector(store, judge, min_score=0.3, **kw)
    vec = fake_embedder.embed_query(question)
    results = [r for r in store.search(vec, 5) if r.chunk.kind == "text"][:1]
    return selector, selector.select(question, vec, results), results


def test_figure_on_the_same_page_as_the_paragraph_is_a_candidate(store, fake_embedder):
    question = "quarterly sales revenue"
    selector = ImageSelector(store, None, min_score=0.3)
    vec = fake_embedder.embed_query(question)
    paragraph = [r for r in store.search(vec, 5) if r.chunk.id == "d-p2-c0"]
    ids = [c.chunk.id for c in selector.candidates(vec, paragraph)]
    assert ids[0] == "d-p2-sales"            # best match first
    assert set(ids) == {"d-p2-sales", "d-p2-org"}


def test_figure_on_a_different_page_is_not_a_candidate(store, fake_embedder):
    selector = ImageSelector(store, None, min_score=0.0)
    vec = fake_embedder.embed_query("leave policy annual leave")
    page3 = [r for r in store.search(vec, 5) if r.chunk.id == "d-p3-c0"]
    assert selector.candidates(vec, page3) == []


def test_unrelated_question_gets_no_image_and_no_judge_call(store, fake_embedder):
    judge = FakeJudge(picks=[0])
    _, images, _ = run(store, fake_embedder, "annual leave carry over days", judge)
    assert images == []
    assert judge.calls == []


def test_judge_picks_only_the_useful_figure(store, fake_embedder):
    judge = FakeJudge(picks=[0])
    _, images, _ = run(store, fake_embedder, "quarterly sales revenue chart", judge)
    assert [i.chunk.id for i in images] == ["d-p2-sales"]
    assert judge.calls[0][1][0] == "d-p2-sales"


def test_judge_saying_none_means_no_image(store, fake_embedder):
    judge = FakeJudge(picks=[])
    _, images, _ = run(store, fake_embedder, "quarterly sales revenue chart", judge)
    assert images == [] and len(judge.calls) == 1


def test_max_images_caps_the_result(store, fake_embedder):
    judge = FakeJudge(picks=[0, 1])
    _, images, _ = run(store, fake_embedder, "sales revenue organisation chart diagram",
                       judge, max_images=1)
    assert len(images) == 1


def test_bad_judge_indexes_are_ignored(store, fake_embedder):
    judge = FakeJudge(picks=[7, -1, 0, 0])
    _, images, _ = run(store, fake_embedder, "quarterly sales revenue chart", judge)
    assert [i.chunk.id for i in images] == ["d-p2-sales"]


def test_judge_failure_falls_back_to_a_confident_match_only(store, fake_embedder):
    judge = FakeJudge(fail=True)
    _, images, _ = run(store, fake_embedder,
                       "Bar chart of quarterly sales revenue rising from Q1 to Q4", judge)
    assert [i.chunk.id for i in images] == ["d-p2-sales"]      # very close to the caption
    _, images, _ = run(store, fake_embedder, "sales chart", FakeJudge(fail=True))
    assert images == []                                        # only loosely related


def test_figure_whose_file_is_missing_is_skipped(store, fake_embedder, tmp_path):
    (tmp_path / "img" / "sales.png").unlink()
    judge = FakeJudge(picks=[0])
    _, images, _ = run(store, fake_embedder, "quarterly sales revenue chart", judge)
    offered = [cid for _, ids in judge.calls for cid in ids]
    assert "d-p2-sales" not in offered                    # never offered to the judge
    assert images == []                                   # the org chart is unrelated


# ---------------- judge reply parsing and the real judge class ----------------

def test_parse_judge_reply():
    assert parse_judge_reply('{"use": ["F2", "F1"]}', 3) == [1, 0]
    assert parse_judge_reply('Sure! ```json\n{"use": ["F1"]}\n```', 3) == [0]
    assert parse_judge_reply('{"use": []}', 3) == []
    assert parse_judge_reply('{"use": ["F9", "F0"]}', 3) == []
    assert parse_judge_reply("I would use F3.", 3) == [2]
    assert parse_judge_reply("not json at all", 3) == []
    assert parse_judge_reply("", 3) == []


def test_openrouter_judge_sends_captions_and_reads_the_reply():
    from types import SimpleNamespace

    from rag.models import SearchResult
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        message = SimpleNamespace(content='{"use": ["F2"]}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    cands = [SearchResult(Chunk("a", SALES, "r.pdf", 2, "d", "figure"), 0.9),
             SearchResult(Chunk("b", ORG, "r.pdf", 2, "d", "figure"), 0.8)]
    assert OpenRouterImageJudge(client).pick("Which chart?", cands) == [1]
    user = sent["messages"][1]["content"]
    assert "Question: Which chart?" in user and "F1." in user and "F2." in user
    assert all(isinstance(m["content"], str) for m in sent["messages"])   # text only


# ---------------- in the pipeline ----------------

def test_pipeline_attaches_the_chosen_figure_as_a_numbered_source(store, fake_embedder):
    gen = FakeGenerator("Sales rise [2].")
    qa = DocQA(store, fake_embedder, gen, min_score=0.2,
               image_selector=ImageSelector(store, FakeJudge(picks=[0]), min_score=0.3))
    answer = qa.ask("quarterly sales revenue chart")
    assert [i.chunk.id for i in answer.images] == ["d-p2-sales"]
    assert "d-p2-sales" in [s.chunk.id for s in answer.sources]    # got a source number
    assert [i.chunk.id for i in gen.images[0]] == ["d-p2-sales"]


def test_pipeline_attaches_nothing_when_no_selector(store, fake_embedder):
    gen = FakeGenerator()
    qa = DocQA(store, fake_embedder, gen, min_score=0.2)
    assert qa.ask("quarterly sales revenue chart").images == []
    assert gen.images == [[]]


def test_search_only_mode_shows_confident_images_without_a_judge(store, fake_embedder):
    judge = FakeJudge(picks=[0])
    qa = DocQA(store, fake_embedder, None, min_score=0.2,
               image_selector=ImageSelector(store, judge, min_score=0.3))
    answer = qa.ask("Bar chart of quarterly sales revenue rising from Q1 to Q4")
    assert judge.calls == []
    assert [i.chunk.id for i in answer.images] == ["d-p2-sales"]
