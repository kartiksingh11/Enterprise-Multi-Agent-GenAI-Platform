"""
app/graph_main.py  (Step 5c: the real orchestrator graph)

GRAPH SHAPE
------------------------------------------------------------------------
                    +-------+
         +--------->|  sql  |---------+
         |          +-------+         |
START -> router                       +--> END
         |          +-------+         |
         +--------->|  rag  |---------+
         |          +-------+         |
         |          +-------+         |
         +--------->|  web  |---------+
                    +-------+

HOW THE CONDITIONAL EDGE WORKS
------------------------------------------------------------------------
add_conditional_edges(source, routing_fn, mapping):
  - After `source` node runs, LangGraph calls routing_fn(state).
  - routing_fn must return a string key.
  - That key is looked up in `mapping` to decide which node runs next.

We keep the DECISION (what label was chosen) and the ROUTING
(picking a node based on that label) as two separate concerns:
  - The `router` node's job is just to classify and write `route` into
    state.
  - `decide_next_node()` below is a tiny, separate function whose only
    job is "read state['route'], return it" — it exists as its own
    function (rather than inlining route_question directly into the
    conditional edge call) so the routing LOGIC and the routing
    DECISION-MAKING stay independently testable and readable.

WHY EACH AGENT NODE IS A THIN ADAPTER
------------------------------------------------------------------------
sql_node / rag_node / web_node below don't contain any real logic —
they just call the agent functions we already built (Steps 2-4) and
shape the result into this graph's State. This is intentional: the
actual agent logic stays in its own well-tested module; the node
function here is purely "adapt that module's output to fit the graph."
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

from app.router import route_question
from app.sql_agent import generate_sql
from app.sql_executor import execute_sql, format_answer as sql_format_answer, UnsafeQueryError
from app.rag_agent import answer_question as rag_answer_question
from app.web_agent import answer_question as web_answer_question
from app.tracing import log_trace, timed

import sqlite3


class GraphState(TypedDict):
    question: str
    route: str          # "sql" | "rag" | "web" — written by the router node
    answer: str
    sources: list[str]  # which doc(s)/URL(s)/"database" the answer came from
    details: dict        # agent-specific debug info, for tracing/observability


def router_node(state: GraphState) -> dict:
    """Classify the question and record which agent should handle it."""
    label = route_question(state["question"])
    print(f"[router_node] question routed to: {label}")
    return {"route": label}


def decide_next_node(state: GraphState) -> Literal["sql", "rag", "web"]:
    """
    The routing function passed to add_conditional_edges.
    Deliberately just reads what router_node already decided — kept
    separate from router_node itself so "deciding the label" and
    "acting on the label" remain two distinct, individually-clear steps.
    """
    return state["route"]


def sql_node(state: GraphState) -> dict:
    """
    Thin adapter around the SQL agent pipeline built in Step 2.
    Handles its own errors so a bad query never crashes the whole graph.
    """
    question = state["question"]
    sql = generate_sql(question)

    if sql == "NO_QUERY":
        return {
            "answer": "I can't answer that from our company database.",
            "sources": [],
            "details": {"generated_sql": "NO_QUERY"},
        }

    try:
        rows = execute_sql(sql)
        answer = sql_format_answer(question, rows)
        return {
            "answer": answer,
            "sources": ["company database"],
            "details": {"generated_sql": sql, "row_count": len(rows)},
        }
    except (UnsafeQueryError, sqlite3.Error) as e:
        # Same philosophy as sql_executor.py's own __main__ block:
        # never let a bad generated query crash the system — degrade
        # gracefully and say so.
        print(f"[sql_node] query failed: {e}")
        return {
            "answer": "I had trouble answering that from our database.",
            "sources": [],
            "details": {"generated_sql": sql, "error": str(e)},
        }


def rag_node(state: GraphState) -> dict:
    """Thin adapter around the RAG agent pipeline built in Step 3."""
    result = rag_answer_question(state["question"])
    retrieved_summary = [
        {"source": r["source"], "distance": round(r["distance"], 4)}
        for r in result["retrieved"]
    ]
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "details": {"retrieved": retrieved_summary},
    }


def web_node(state: GraphState) -> dict:
    """Thin adapter around the web agent pipeline built in Step 4."""
    result = web_answer_question(state["question"])
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "details": {"num_results": len(result["retrieved"])},
    }


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("sql", sql_node)
    graph.add_node("rag", rag_node)
    graph.add_node("web", web_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        decide_next_node,
        {"sql": "sql", "rag": "rag", "web": "web"},
    )

    # All 3 agent nodes are terminal — once one runs, we're done.
    graph.add_edge("sql", END)
    graph.add_edge("rag", END)
    graph.add_edge("web", END)

    return graph.compile()


# Compiled once at import time and reused — recompiling the graph on
# every request would be wasted work, same "load once, reuse" idea as
# the embedding model singleton in vector_store.py.
_compiled_graph = build_graph()


def ask(question: str) -> dict:
    """
    The single entry point the rest of the platform (FastAPI in Step 7,
    or this file's own __main__ block) should call. Times the full
    pipeline and logs a trace — every call through here is automatically
    observable, callers don't need to remember to log anything themselves.
    """
    with timed() as elapsed:
        final_state = _compiled_graph.invoke(
            {"question": question, "route": "", "answer": "", "sources": [], "details": {}}
        )
    latency_ms = elapsed()

    log_trace(
        question=question,
        route=final_state["route"],
        answer=final_state["answer"],
        sources=final_state["sources"],
        details=final_state["details"],
        latency_ms=latency_ms,
    )

    return {**final_state, "latency_ms": latency_ms}


if __name__ == "__main__":
    # Manual smoke test — run: python -m app.graph_main
    test_questions = [
        "How many employees work in Engineering?",
        "What's our policy on parental leave?",
        "What's the latest news on AI regulation?",
        "What is the total revenue from all orders?",
        "How do I fix a 401 Unauthorized error?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = ask(q)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Latency: {result['latency_ms']}ms")
