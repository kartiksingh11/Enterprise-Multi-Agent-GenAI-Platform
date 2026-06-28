"""
app/web_agent.py  (Step 4b: synthesize an answer from search results)

SAME GROUNDING DISCIPLINE AS THE RAG AGENT
------------------------------------------------------------------------
Just like rag_agent.py, the risk here is the LLM answering from its
own (possibly stale — remember our knowledge cutoff) training
knowledge instead of the actual search results we just fetched. The
whole POINT of a web agent is to get CURRENT information beyond the
model's training cutoff, so if the LLM ignores the search results and
free-associates from memory, the agent has failed at its one job.

Same two-layer defense as before:
  1. PROMPT: explicit instruction to use only the search results given.
  2. CODE: if search returns zero results, we don't even call the LLM —
     we return a clear "couldn't find anything" response. We never let
     the LLM "wing it" when we know the tool itself came back empty.
"""

from app.llm_client import call_llm
from app.web_search import web_search, MAX_RESULTS

WEB_SYSTEM_PROMPT = """You answer questions using ONLY the web search results provided below.

Rules:
- Use ONLY the information in the search results. Do not use outside knowledge,
  since your own knowledge may be outdated and the user needs current information.
- Some search results are generic category/landing pages whose snippet is just a
  page description (e.g. "Explore the latest news on X..."), not an actual reported
  fact. IGNORE these snippets — they contain no real information to answer with.
- If, after ignoring generic/non-substantive snippets, no result actually answers
  the question, say exactly: "I couldn't find current information to answer that."
- Be concise: 2-4 sentences.
- Do not mention "search results" explicitly — just answer naturally, as if reporting current information.
"""


def answer_question(question: str, max_results: int = MAX_RESULTS) -> dict:
    """
    Full web agent pipeline: search -> (empty-result check) -> synthesize.

    Returns a dict with the answer and the source URLs used, mirroring
    the shape of rag_agent.answer_question() (answer + sources) so that
    later, when we build the LangGraph router, all 3 agents can return
    a consistent shape regardless of which one handled the question.
    """
    results = web_search(question, max_results=max_results)

    if not results:
        return {
            "answer": "I couldn't find current information to answer that.",
            "sources": [],
            "retrieved": results,
        }

    context_blocks = []
    for r in results:
        context_blocks.append(f"Title: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['url']}")
    context_text = "\n\n".join(context_blocks)

    prompt = f"Search results:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
    answer = call_llm(prompt=prompt, system=WEB_SYSTEM_PROMPT, temperature=0.2)

    sources = [r["url"] for r in results]

    return {
        "answer": answer,
        "sources": sources,
        "retrieved": results,
    }


if __name__ == "__main__":
    # Manual smoke test — run: python -m app.web_agent
    test_questions = [
        "What is the current weather in Delhi?",
        "What is the latest iPhone model?",
    ]

    for q in test_questions:
        result = answer_question(q)
        print(f"Q: {q}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 60)
