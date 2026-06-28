"""
app/web_search.py  (Step 4a: the search tool itself, no LLM involved)

WHY THIS IS ITS OWN FILE, SEPARATE FROM THE AGENT LOGIC
------------------------------------------------------------------------
Same philosophy as llm_client.py: isolate the "talk to an external
system" code behind a small function so that:
  1. We can test/debug search results independently of any LLM call
     (if results look wrong, we know immediately whether it's the
     search or the synthesis step at fault).
  2. Swapping providers later (DuckDuckGo -> Tavily -> Bing -> Google)
     means changing ONE function's internals, not every call site.
This is the same "tool use" pattern that real agent frameworks
(LangGraph included) formalize: an agent's "tools" are just functions
with a clear input/output contract that the LLM is told it can invoke.
"""

from ddgs import DDGS

MAX_RESULTS = 4


def web_search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """
    Run a web search and return a list of {"title", "snippet", "url"} dicts.

    NOTE: this uses the unofficial, no-API-key DuckDuckGo search route —
    fine for local prototyping, but not an officially supported product,
    so behavior/availability can change without notice. A production
    system would swap this for a paid, supported API (Tavily, Bing, etc.)
    — that swap only requires changing the inside of this one function.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
            })
    return results


if __name__ == "__main__":
    # Manual smoke test — run: python -m app.web_search
    test_queries = [
        "current weather in Delhi",
        "latest iPhone model 2026",
    ]

    for q in test_queries:
        print(f"Query: {q}")
        results = web_search(q)
        for r in results:
            print(f"  - {r['title']}")
            print(f"    {r['snippet'][:100]}...")
            print(f"    {r['url']}")
        print("-" * 60)
