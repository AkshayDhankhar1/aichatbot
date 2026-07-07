"""Self-contained BM25 sparse encoder for Pinecone hybrid search.

Why this exists: the `pinecone-text` wrapper pins `mmh3 < 5`, which has no
prebuilt wheel for Python 3.13 on Windows and needs a C++ compiler to build.
This module reproduces exactly what we need from it — BM25-weighted sparse
vectors keyed by 32-bit `mmh3` token hashes — using `mmh3` 5.x directly, so it
installs from a wheel everywhere (Win/Linux, py3.11–3.13) with no compiler.

Scheme (standard BM25 dot-product decomposition, same as pinecone-text):
  - document term weight: tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))
  - query term weight:    idf(t) = ln(1 + (N - df + 0.5)/(df + 0.5))
The dot product of a document's sparse vector with a query's sparse vector
approximates the BM25 relevance score.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter

import mmh3

# Lightweight tokenizer (avoids an nltk dependency/downloads).
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = set(
    "a an and are as at be by for from has have had he her his i in is it its of on "
    "or that the their them they this to was were will with you your we our but not no "
    "if then else when where which who how what why can could should would may might "
    "do does did done so than too very just about above below up down out over under".split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def _hash(token: str) -> int:
    # Unsigned 32-bit index — valid, non-negative Pinecone sparse index.
    return mmh3.hash(token, signed=False)


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.avgdl = 0.0
        self.n_docs = 0
        self.idf: dict[str, float] = {}  # json keys are strings: str(index) -> idf

    # ---- fit ----
    def fit(self, corpus: list[str]) -> "BM25":
        df: Counter[int] = Counter()
        total_len = 0
        n = 0
        for text in corpus:
            toks = _tokenize(text)
            total_len += len(toks)
            n += 1
            for idx in {_hash(t) for t in toks}:
                df[idx] += 1
        self.n_docs = n
        self.avgdl = total_len / max(n, 1)
        self.idf = {
            str(idx): math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for idx, freq in df.items()
        }
        return self

    # ---- encode ----
    def _doc_vector(self, text: str) -> dict:
        toks = _tokenize(text)
        dl = len(toks)
        tf = Counter(_hash(t) for t in toks)
        norm = self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
        indices, values = [], []
        for idx, f in tf.items():
            indices.append(idx)
            values.append((f * (self.k1 + 1)) / (f + norm))
        return {"indices": indices, "values": values}

    def encode_documents(self, texts: list[str]) -> list[dict]:
        return [self._doc_vector(t) for t in texts]

    def encode_queries(self, text: str) -> dict:
        weights: dict[int, float] = {}
        for t in _tokenize(text):
            idx = _hash(t)
            idf = self.idf.get(str(idx))
            if idf is not None:  # ignore terms unseen in the corpus
                weights[idx] = idf
        return {"indices": list(weights.keys()), "values": list(weights.values())}

    # ---- persistence ----
    def dump(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"k1": self.k1, "b": self.b, "avgdl": self.avgdl,
                 "n_docs": self.n_docs, "idf": self.idf},
                f,
            )

    def load(self, path) -> "BM25":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.k1 = d["k1"]
        self.b = d["b"]
        self.avgdl = d["avgdl"]
        self.n_docs = d["n_docs"]
        self.idf = d["idf"]
        return self
