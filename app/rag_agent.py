"""
app/rag_agent.py  (Step 3d: synthesize a grounded answer from retrieved chunks)

THE CENTRAL RISK IN RAG: UNGROUNDED ANSWERS
------------------------------------------------------------------------
The most common RAG failure isn't "retrieval found nothing" — it's
the LLM answering from its own general/training knowledge instead of
the retrieved text, sometimes blending both without any signal that
it did so. This is dangerous specifically because it can look
exactly as confident and fluent as a correctly-grounded answer.

Our defenses (same "prompt + code" layering philosophy as Step 2):
  1. PROMPT: explicit instruction to use ONLY the provided context,
     and to say so plainly if the context doesn't contain the answer.
  2. CODE: a distance threshold cutoff (DISTANCE_THRESHOLD) — if even
     the best-retrieved chunk is too dissimilar from the question, we
     refuse to even attempt an answer, rather than trusting the LLM to
     self-police "should I really be answering this?" the way we
     learned NOT to trust raw LLM self-assessment in the SQL agent.

WHY WE PASS ALL k RETRIEVED CHUNKS, NOT JUST THE TOP ONE
------------------------------------------------------------------------
We observed (Step 3c) that for "vacation days", the BEST-ranked chunk
was about sick leave, with the correct PTO chunk ranked #2 by a small
margin. Passing all k=3 chunks to the LLM means it still SEES the
correct chunk and can pick the right one using its own reasoning over
the full set — rather than us silently throwing away a relevant chunk
just because vector similarity ranked it imperfectly.
"""

from app.llm_client import call_llm
from app.retriever import retrieve, DEFAULT_K

# Chroma's default distance metric here is roughly squared L2 over
# normalized embeddings; in practice (see Step 3c output) distances
# above ~1.3 corresponded to genuinely unrelated chunks, while real
# matches landed under ~1.0. This threshold is an empirical choice
# tuned from OUR observed data, not a universal constant — a real
# system would tune this against a labeled eval set.
DISTANCE_THRESHOLD = 1.3

RAG_SYSTEM_PROMPT = """You answer questions using ONLY the provided context below.

Rules:
- Use ONLY information in the context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say
  exactly: "I don't have enough information to answer that."
- Be concise: 1-3 sentences.
- Do not mention "the context" or "the documents" explicitly — just answer naturally.
"""


def answer_question(question: str, k: int = DEFAULT_K) -> dict:
    """
    Full RAG pipeline: retrieve -> (threshold check) -> synthesize.

    Returns a dict with the answer AND the retrieval metadata used to
    produce it (sources, distances) — we deliberately return this
    "explainability" info alongside the answer, not hide it, since
    transparency about WHAT was retrieved is the whole point of the
    observability angle from the original project description.
    """
    results = retrieve(question, k=k)

    if not results or results[0]["distance"] > DISTANCE_THRESHOLD:
        return {
            "answer": "I don't have enough information to answer that.",
            "sources": [],
            "retrieved": results,
        }

    context_blocks = []
    for r in results:
        context_blocks.append(f"[{r['source']}]: {r['text']}")
    context_text = "\n\n".join(context_blocks)

    prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
    answer = call_llm(prompt=prompt, system=RAG_SYSTEM_PROMPT, temperature=0.2)

    sources = sorted(set(r["source"] for r in results))

    return {
        "answer": answer,
        "sources": sources,
        "retrieved": results,
    }


if __name__ == "__main__":
    # Manual smoke test — run: python -m app.rag_agent
    test_questions = [
        "How many vacation days do employees get?",   # the tricky one from Step 3c
        "How do I authenticate API requests?",
        "What happens if I exceed the rate limit?",
        "What's the maximum hotel rate for international travel?",
        "What's the company's stock price?",          # should trigger the "not enough info" path
    ]

    for q in test_questions:
        result = answer_question(q)
        print(f"Q: {q}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("Retrieved (for debugging):")
        for r in result["retrieved"]:
            print(f"  [{r['source']} #{r['chunk_index']}] dist={r['distance']:.4f}")
        print("-" * 60)
