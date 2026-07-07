"""Standalone ingestion script — run manually / on demand.

    cd backend
    python -m app.ingest

Pipeline: read docs/  ->  chunk  ->  fit BM25 (sparse)  ->  embed (dense 384)
->  upsert dense+sparse vectors to Pinecone.

This is the ONLY component that reads `docs/`, and it is READ-ONLY: it never
writes, edits, or deletes anything under `docs/`. The runtime chatbot only ever
queries Pinecone.
"""
from __future__ import annotations

import argparse
import itertools

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DOCS_DIR, get_settings
from .loaders import load_docs
from .vectorstore import embed_texts, ensure_index, new_bm25, save_bm25

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
UPSERT_BATCH = 100


def _chunk(docs):
    """Split loaded docs into overlapping chunks, keeping the source filename."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []  # list of dicts: {id, text, source}
    for doc in docs:
        for i, piece in enumerate(splitter.split_text(doc.text)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append({
                "id": f"{doc.source}::chunk-{i}",
                "text": piece,
                "source": doc.source,
            })
    return chunks


def _batched(iterable, n):
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


def run(recreate: bool = False):
    settings = get_settings()

    print(f"Reading documents from: {DOCS_DIR}")
    docs = load_docs(DOCS_DIR)
    if not docs:
        raise SystemExit(f"No supported documents found in {DOCS_DIR}")

    chunks = _chunk(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} documents.")

    texts = [c["text"] for c in chunks]

    # 1) Fit BM25 on THIS corpus and persist params for query-time reuse.
    print("Fitting BM25 sparse encoder on corpus...")
    bm25 = new_bm25()
    bm25.fit(texts)
    save_bm25(bm25)
    sparse_vectors = bm25.encode_documents(texts)

    # 2) Dense embeddings (384 dims).
    print("Computing dense embeddings...")
    dense_vectors = embed_texts(texts)

    # 3) Upsert to Pinecone (hybrid: values + sparse_values).
    index = ensure_index()
    if recreate:
        print("Clearing existing vectors (--recreate)...")
        try:
            index.delete(delete_all=True)
        except Exception as exc:
            print(f"  (nothing to clear or delete failed: {exc})")

    # We upsert the RAW dense + sparse vectors here; alpha-weighting is applied
    # at query time so alpha can be tuned without re-ingesting.
    vectors = []
    for c, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
        vec = {
            "id": c["id"],
            "values": dense,
            "metadata": {"text": c["text"], "source": c["source"]},
        }
        # Only attach sparse values when non-empty (a chunk of pure stopwords
        # would otherwise send an empty sparse vector, which Pinecone rejects).
        if sparse["indices"]:
            vec["sparse_values"] = {"indices": sparse["indices"], "values": sparse["values"]}
        vectors.append(vec)

    total = 0
    for batch in _batched(vectors, UPSERT_BATCH):
        index.upsert(vectors=batch)
        total += len(batch)
        print(f"  upserted {total}/{len(vectors)}")

    print(f"\nDone. Ingested {total} chunks into '{settings.pinecone_index_name}'.")
    print("Reminder: bm25_params.json was (re)generated — include it when you "
          "deploy the backend so query-time sparse encoding matches ingest.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest docs/ into Pinecone.")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete all existing vectors before upserting.")
    args = parser.parse_args()
    run(recreate=args.recreate)
