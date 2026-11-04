"""Retrieval benchmark. Run: python -m rag_lab.evaluate"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .retrieval import BM25, LSA, ROOT, Chunk, Embedding, Hybrid, Retriever, load_chunks

QUESTIONS = ROOT / "data" / "eval" / "rag_questions.json"


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    return next((1 / (i + 1) for i, c in enumerate(ranked) if c in relevant), 0.0)


def ndcg_at(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, c in enumerate(ranked[:k]) if c in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal


def evaluate(retriever: Retriever, questions: list[dict]) -> dict:
    rows = {"recall@1": [], "recall@3": [], "recall@5": [], "mrr": [], "ndcg@5": []}
    misses = []
    for item in questions:
        ranked = [c.id for c, _ in retriever.search(item["q"], 10)]
        relevant = set(item["relevant"])
        rows["recall@1"].append(recall_at(ranked, relevant, 1))
        rows["recall@3"].append(recall_at(ranked, relevant, 3))
        rows["recall@5"].append(recall_at(ranked, relevant, 5))
        rows["mrr"].append(reciprocal_rank(ranked, relevant))
        rows["ndcg@5"].append(ndcg_at(ranked, relevant, 5))
        if not relevant & set(ranked[:3]):
            misses.append(item["q"])
    result = {name: round(sum(v) / len(v), 4) for name, v in rows.items()}
    return {**result, "misses_at_3": misses}


def main(out: Path = ROOT / "reports" / "rag_benchmark.json") -> dict:
    chunks: list[Chunk] = load_chunks()
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    bm25, lsa = BM25(chunks), LSA(chunks)
    systems: dict[str, Retriever] = {
        "bm25": bm25,
        "lsa": lsa,
        "hybrid_bm25_lsa": Hybrid([bm25, lsa]),
    }
    try:
        embedding = Embedding(chunks)
    except ImportError:
        print("sentence-transformers not installed: skipping the neural retrievers")
    else:
        systems["minilm"] = embedding
        systems["hybrid_bm25_minilm"] = Hybrid([bm25, embedding])
    report = {
        "n_chunks": len(chunks),
        "n_questions": len(questions),
        "results": {name: evaluate(r, questions) for name, r in systems.items()},
    }
    out.write_text(json.dumps(report, indent=2))
    for name, r in report["results"].items():
        print(name, {k: v for k, v in r.items() if k != "misses_at_3"})
    best = max(report["results"], key=lambda name: report["results"][name]["mrr"])
    print(f"best by MRR: {best}; missed at 3:", report["results"][best]["misses_at_3"])
    return report


if __name__ == "__main__":
    main()
