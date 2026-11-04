"""Three retrievers behind one interface, built on numpy and scikit-learn only.

BM25 is written out because the formula is the point. The dense retriever uses LSA
(TF-IDF + truncated SVD) so the benchmark runs offline and in CI. Replacing it with a
neural embedding only needs a new class with the same `search` method.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT / "data" / "policies"


@dataclass(frozen=True)
class Chunk:
    id: str  # "<file>#<section-slug>", stable enough to cite
    title: str
    text: str


def load_chunks(folder: Path = POLICY_DIR) -> list[Chunk]:
    """Split markdown on '## ' so one chunk is one citable policy section."""
    chunks = []
    for path in sorted(folder.glob("*.md")):
        for block in path.read_text(encoding="utf-8").split("\n## ")[1:]:
            title, _, body = block.partition("\n")
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            chunks.append(Chunk(f"{path.stem}#{slug}", title.strip(), body.strip()))
    return chunks


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(t) for t in tokens if t not in ENGLISH_STOP_WORDS]


def _stem(token: str) -> str:
    """Tiny suffix stripper: enough to match approve/approved, payment/payments."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]: ...


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(f"{c.title} {c.text}")) for c in chunks]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg_len = self.lengths.mean()
        df = Counter(term for doc in self.docs for term in doc)
        n = len(chunks)
        self.idf = {t: np.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.chunks))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_len)
        for term in tokenize(query):
            if term not in self.idf:
                continue
            tf = np.array([d[term] for d in self.docs], dtype=float)
            out += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        return _top(self.chunks, self.scores(query), k)


class LSA:
    def __init__(self, chunks: list[Chunk], dim: int = 40, seed: int = 0):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True
        )
        matrix = self.vectorizer.fit_transform(f"{c.title} {c.text}" for c in chunks)
        dim = min(dim, min(matrix.shape) - 1)
        self.svd = TruncatedSVD(dim, random_state=seed)
        self.vectors = normalize(self.svd.fit_transform(matrix))

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = normalize(self.svd.transform(self.vectorizer.transform([query])))
        return _top(self.chunks, (self.vectors @ q.T).ravel(), k)


class Embedding:
    """Neural retriever. Optional: needs `pip install .[embed]` and downloads the model."""

    def __init__(self, chunks: list[Chunk], model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer(model)
        self.vectors = self.model.encode(
            [f"{c.title}. {c.text}" for c in chunks], normalize_embeddings=True
        )

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = self.model.encode([query], normalize_embeddings=True)
        return _top(self.chunks, (self.vectors @ q.T).ravel(), k)


class Hybrid:
    """Reciprocal rank fusion: uses ranks, so the two score scales never need aligning."""

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, pool: int = 20):
        self.retrievers, self.rrf_k, self.pool = retrievers, rrf_k, pool

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for rank, (chunk, _) in enumerate(retriever.search(query, self.pool)):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1 / (self.rrf_k + rank + 1)
                by_id[chunk.id] = chunk
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(by_id[cid], score) for cid, score in ranked[:k]]


def _top(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [(chunks[i], float(scores[i])) for i in order]


def build_default(chunks: list[Chunk] | None = None) -> Hybrid:
    chunks = chunks or load_chunks()
    return Hybrid([BM25(chunks), LSA(chunks)])
