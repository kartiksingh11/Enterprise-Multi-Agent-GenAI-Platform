"""
app/vector_store.py  (Step 3b: embed chunks + store in ChromaDB)

WHAT AN EMBEDDING IS, CONCRETELY
------------------------------------------------------------------------
An embedding model maps text -> a fixed-length vector of floats (384
numbers, for the model we're using) such that semantically SIMILAR
text produces NEARBY vectors in that 384-dimensional space. "Nearby"
is measured by cosine similarity (the angle between two vectors) —
this is the core idea behind "semantic search": a question and an
answer can share almost no exact words and still be judged highly
similar, because they're talking about the same thing.

WHY A SEPARATE EMBEDDING MODEL FROM OUR CHAT LLM
------------------------------------------------------------------------
Embedding and text-generation are different tasks, usually served by
different, specialized models — this is the standard production
pattern (e.g. OpenAI's embedding models are entirely separate from
their chat models). We use `sentence-transformers` with
`all-MiniLM-L6-v2`: a small (~80MB), fast, CPU-friendly model that's
the most common baseline in RAG tutorials/papers for exactly that
quality-to-cost ratio.

WHY CHROMADB INSTEAD OF A PLAIN LIST OF VECTORS
------------------------------------------------------------------------
For 25 chunks, brute-force comparing a query vector against every
stored vector would be instant either way. We use a real vector DB
anyway because:
  1. It's what the resume bullet claims ("1,000+ chunks... ChromaDB
     vector search") — the code should actually do that, not fake it.
  2. The interface (store once, query(vector, k) many times) doesn't
     change at all as the data grows from 25 to 25,000 chunks. Chroma
     uses an approximate-nearest-neighbor index (HNSW) under the hood,
     which is what makes that scaling possible without rewriting code.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

from app.chunking import chunk_all_documents

# Persisted to disk so we don't have to re-embed every time we run
# the app — embeddings are deterministic for a given model + text, so
# recomputing them on every startup would be pure wasted work.
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "enterprise_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_embedding_model = None  # lazy-loaded singleton, see get_embedding_model()


def get_embedding_model() -> SentenceTransformer:
    """
    Lazily load the embedding model once and reuse it.

    WHY LAZY + SINGLETON: loading a sentence-transformers model reads
    weights from disk into memory, which takes a noticeable moment.
    We don't want to pay that cost every time embed_texts() is called —
    once per process is enough.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of strings into a list of embedding vectors.
    Batches all texts into a single model call for efficiency rather
    than embedding one string at a time in a loop.
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def get_chroma_collection():
    """
    Open (or create) a persistent Chroma collection on disk.

    PersistentClient writes to CHROMA_DB_PATH, so the index survives
    between runs of the program — same philosophy as why we persist
    the SQLite database to a file rather than rebuilding it every time.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def build_index(reset: bool = True) -> int:
    """
    Chunk all documents, embed every chunk, and store them in ChromaDB.

    Args:
        reset: if True, wipe any existing collection first. We default
               to True because re-running indexing with stale + new
               data mixed together is a common, confusing RAG bug —
               better to rebuild cleanly each time at this project's
               scale (25 chunks; cheap to redo).

    Returns the number of chunks indexed.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet — nothing to delete

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    chunks = chunk_all_documents()
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # Chroma needs a unique string ID per item — we build one from the
    # source filename + chunk index, which also makes IDs human-readable
    # when debugging (e.g. "leave_policy.txt::2").
    ids = [f"{c['source']}::{c['chunk_index']}" for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    return len(chunks)


if __name__ == "__main__":
    # Manual smoke test — run: python -m app.vector_store
    print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    count = build_index(reset=True)
    print(f"Indexed {count} chunks into ChromaDB at {CHROMA_DB_PATH}")

    # Quick sanity check: confirm the collection actually has that many items
    collection = get_chroma_collection()
    print(f"Collection count (via Chroma): {collection.count()}")
