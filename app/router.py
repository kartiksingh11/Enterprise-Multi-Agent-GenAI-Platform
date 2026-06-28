"""
app/router.py  (Step 5b: classify a question into sql / rag / web)

THIS IS THE HIGHEST-STAKES PROMPT IN THE SYSTEM
------------------------------------------------------------------------
If this gets it wrong, the right specialist agent never even gets a
chance to answer — the question goes to the wrong place entirely.
Same engineering discipline as every other classification step we've
built (SQL relevance check, RAG distance threshold):
  1. PROMPT: concrete examples per category, forced single-word output.
  2. CODE: validate the LLM's raw output against the known valid
     labels; never trust it blindly. If it returns garbage, fall back
     to a deterministic default rather than crashing.

WHY A DETERMINISTIC FALLBACK MATTERS
------------------------------------------------------------------------
We chose "rag" as the fallback (not "sql" or "web") because RAG is the
safest default failure mode: worst case, it correctly says "I don't
have enough information" (we built that exact safeguard in Step 3d),
rather than the SQL agent risking a malformed query or the web agent
returning ungrounded live content. A deliberate, named default is
better than silently picking whichever branch happens to be first in
an if/else chain.
"""

from app.llm_client import call_llm

VALID_LABELS = {"sql", "rag", "web"}
FALLBACK_LABEL = "rag"  # see docstring above for why this specific choice

ROUTER_SYSTEM_PROMPT = """You classify a user question into exactly one category: sql, rag, or web.

sql: questions about structured company data — employees, departments,
     salaries, customers, products, orders, revenue, counts, totals.
rag: questions about company policy or technical documentation — leave
     policy, expense reimbursement, API authentication, troubleshooting
     errors, rate limits.
web: questions needing current/live information not in our company
     data — news, weather, prices of public products, anything about
     the outside world right now.

Reply with EXACTLY one word: sql, rag, or web.

Examples:
Question: How many employees are in Engineering?
Category: sql

Question: What's our policy on parental leave?
Category: rag

Question: What's the current weather in Mumbai?
Category: web

Question: How do I fix a 401 Unauthorized error?
Category: rag

Question: What's the total revenue from all orders?
Category: sql

Question: Who is the current CEO of OpenAI?
Category: web
"""


def route_question(question: str) -> str:
    """
    Classify a question into "sql", "rag", or "web".

    Always returns one of VALID_LABELS — if the LLM's raw output
    doesn't cleanly match one of them, we fall back to FALLBACK_LABEL
    rather than propagating garbage downstream.
    """
    raw = call_llm(prompt=question, system=ROUTER_SYSTEM_PROMPT, temperature=0.0)
    label = _extract_label(raw)

    if label not in VALID_LABELS:
        return FALLBACK_LABEL

    return label


def _extract_label(raw: str) -> str:
    """
    Defensive parsing: lowercase, strip whitespace/punctuation, and
    take just the first word — covers cases where the model adds
    stray punctuation or a trailing period despite instructions.
    """
    cleaned = raw.strip().lower().strip(".:!")
    first_word = cleaned.split()[0] if cleaned else ""
    return first_word


if __name__ == "__main__":
    # Manual smoke test — run: python -m app.router
    test_questions = [
        "How many employees work in Engineering?",
        "What's our PTO carryover limit?",
        "What's the latest news on AI regulation?",
        "How do I authenticate API requests?",
        "What's the total revenue from all orders?",
        "What's the weather like in Tokyo right now?",
        "Tell me about quantum computing.",  # genuinely ambiguous / out of all 3 domains
    ]

    for q in test_questions:
        label = route_question(q)
        print(f"Q: {q}\n  -> routed to: {label}")
