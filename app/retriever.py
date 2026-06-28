"""
app/retriever.py  (Step 3c: query -> top-k relevant chunks)

THE CORE RAG TRICK, MADE CONCRETE
------------------------------------------------------------------------
We embed the user's QUESTION using the exact same model we used to
embed the document CHUNKS (all-MiniLM-L6-v2). Because both live in the
same 384-dimensional vector space, we can measure how close the
question's vector is to each chunk's vector. "Close" here means
small cosine distance — Chroma's default distance metric — which
corresponds to high semantic similarity.

THIS ONLY WORKS IF QUERY AND DOCS USE THE SAME EMBEDDING MODEL.
Mixing embedding models between indexing and querying is a classic,
silent RAG bug: nothing crashes, you just get meaningless/irrelevant
results, because the two vector spaces aren't aligned with each other.

CHOOSING k (how many chunks to retrieve)
------------------------------------------------------------------------
- Too small (k=1): risk missing a relevant chunk if the answer is
  split across two adjacent paragraphs that got chunked separately.
- Too large (k=10): floods the LLM's prompt with marginally-relevant
  text, which both wastes context budget and increases the chance the
  LLM gets distracted/hallucinates from irrelevant content.
We default to k=3 as a reasonable starting point and treat it as a
tunable parameter, not a fixed constant — see retrieve() below.
"""

from app.vector_store import get_chroma_collection, embed_texts

DEFAULT_K = 3


def retrieve(query: str, k: int = DEFAULT_K) -> list[dict]:
    """
    Embed `query` and return the top-k most similar chunks from ChromaDB.

    Returns a list of dicts, each:
        {
            "text": the chunk's original text,
            "source": which document it came from,
            "chunk_index": its position within that document,
            "distance": Chroma's reported distance (LOWER = more similar)
        }

    We return distance explicitly (not hidden) so that later steps —
    and you, while debugging — can actually see HOW confident a match
    is, rather than blindly trusting whatever comes back top-ranked.
    """
    collection = get_chroma_collection()

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    # Chroma returns results as parallel lists-of-lists (one outer list
    # per query embedding we sent; we only sent one, so we index [0]).
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []
    for doc_text, meta, dist in zip(documents, metadatas, distances):
        retrieved.append({
            "text": doc_text,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        })
    return retrieved


if __name__ == "__main__":
    # Manual smoke test — run: python -m app.retriever
    test_questions = [
        "How many vacation days do employees get?",
        "How do I authenticate API requests?",
        "What happens if I exceed the rate limit?",
        "What's the maximum hotel rate for international travel?",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        results = retrieve(q, k=3)
        for r in results:
            print(f"  [{r['source']} #{r['chunk_index']}] dist={r['distance']:.4f}")
            print(f"    {r['text'][:100]}...")
        print("-" * 60)
