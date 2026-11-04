import pytest

from rag_lab.evaluate import ndcg_at, recall_at, reciprocal_rank
from rag_lab.retrieval import BM25, Chunk, Hybrid, load_chunks, tokenize


def test_chunks_have_unique_ids():
    chunks = load_chunks()
    assert len(chunks) == len({c.id for c in chunks}) >= 25


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The Approved payments") == ["approv", "payment"]


def test_bm25_prefers_rare_term():
    chunks = [
        Chunk("a", "a", "loan loan loan payment"),
        Chunk("b", "b", "loan collateral"),
        Chunk("c", "c", "loan payment"),
    ]
    assert BM25(chunks).search("collateral", 1)[0][0].id == "b"


def test_rrf_rewards_agreement():
    class Fixed:
        def __init__(self, ids):
            self.ids = ids

        def search(self, query, k=5):
            return [(Chunk(i, i, ""), 1.0) for i in self.ids[:k]]

    hybrid = Hybrid([Fixed(["a", "b"]), Fixed(["c", "b"])])
    assert hybrid.search("q", 1)[0][0].id == "b"


def test_metrics_by_hand():
    assert recall_at(["a", "b"], {"b", "c"}, 2) == 0.5
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert ndcg_at(["a"], {"a"}, 5) == pytest.approx(1.0)
