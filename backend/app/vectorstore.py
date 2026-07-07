"""Shared helpers for embeddings, the BM25 sparse encoder, and Pinecone access.

Used by both `ingest.py` (write path) and `rag_chain.py` (read path) so the two
stay perfectly in sync on model, dimension, and hybrid-scaling math.
"""
from __future__ import annotations

from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec

from .bm25 import BM25
from .config import BM25_PARAMS_PATH, DATA_DIR, get_settings


# --------------------------------------------------------------------------- #
# Dense embeddings (local sentence-transformers, 384 dims).                    #
# --------------------------------------------------------------------------- #
@lru_cache
def get_embedder():
    # Imported lazily so the module is cheap to import when only Pinecone is used.
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    print(f"Loading embedding model: {settings.embedding_model_name}")
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vecs = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


# --------------------------------------------------------------------------- #
# Sparse BM25 encoder. Fitted during ingest, persisted to bm25_params.json,    #
# reloaded at query time so dense+sparse use identical term statistics.        #
# --------------------------------------------------------------------------- #
def new_bm25() -> BM25:
    return BM25()


def save_bm25(encoder: BM25) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    encoder.dump(str(BM25_PARAMS_PATH))
    print(f"Saved BM25 params -> {BM25_PARAMS_PATH}")


@lru_cache
def load_bm25() -> BM25:
    """Load BM25 params fitted during ingest."""
    if not BM25_PARAMS_PATH.exists():
        raise RuntimeError(
            f"BM25 params not found at {BM25_PARAMS_PATH}. "
            "Run `python -m app.ingest` first (and commit bm25_params.json for deploy)."
        )
    encoder = BM25().load(str(BM25_PARAMS_PATH))
    print(f"Loaded fitted BM25 params from {BM25_PARAMS_PATH}")
    return encoder


# --------------------------------------------------------------------------- #
# Pinecone.                                                                     #
# --------------------------------------------------------------------------- #
@lru_cache
def get_pinecone() -> Pinecone:
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set. Fill in backend/.env.")
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index():
    """Create the hybrid index if it doesn't exist, then return a handle.

    Hybrid (sparse+dense) search in Pinecone REQUIRES the `dotproduct` metric.
    """
    settings = get_settings()
    pc = get_pinecone()
    existing = set(pc.list_indexes().names())
    if settings.pinecone_index_name not in existing:
        print(f"Creating Pinecone index '{settings.pinecone_index_name}' "
              f"(dim={settings.embedding_dim}, metric=dotproduct)...")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dim,
            metric="dotproduct",  # required for hybrid sparse-dense
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    return pc.Index(settings.pinecone_index_name)


def get_index():
    settings = get_settings()
    return get_pinecone().Index(settings.pinecone_index_name)


def hybrid_scale(dense: list[float], sparse: dict, alpha: float):
    """Weight dense/sparse vectors by alpha (1.0=dense only, 0.0=sparse only).

    Standard Pinecone convex scaling: dense*alpha, sparse*(1-alpha).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    scaled_sparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    scaled_dense = [v * alpha for v in dense]
    return scaled_dense, scaled_sparse
