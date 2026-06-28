"""
streamlit_app.py  (Step 9: local demo UI for screenshots/portfolio)

WHY THIS FILE LIVES AT THE PROJECT ROOT, NOT IN app/
------------------------------------------------------------------------
Streamlit's convention is to run `streamlit run <file>.py` from the
project root. Keeping it at the root (rather than inside app/) also
visually signals what it is: a thin UI layer on top of the real
platform, not part of the platform's core logic — same "thin adapter"
philosophy as app/api.py, just for a different frontend.

WHAT THIS UI CALLS
------------------------------------------------------------------------
Directly imports app.graph_main.ask() — the exact same function the
FastAPI layer (app/api.py) calls. No new logic, no duplicated agent
code. This file is PURELY presentation.

Run with:  streamlit run streamlit_app.py
"""

import streamlit as st
from app.graph_main import ask

# ---------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Multi-Agent GenAI Platform",
    page_icon="🧭",
    layout="wide",
)

ROUTE_COLORS = {
    "sql": "#D98E48",   # amber
    "rag": "#4FA89B",   # teal
    "web": "#9B8AC4",   # violet
}
ROUTE_LABELS = {
    "sql": "SQL Agent",
    "rag": "RAG Agent",
    "web": "Web Agent",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
.mono {
    font-family: 'JetBrains Mono', monospace;
}

.stApp {
    background-color: #15171C;
    color: #EDEAE3;
}

.agent-node {
    border: 2px solid #333740;
    border-radius: 10px;
    padding: 14px 10px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.25s ease;
    opacity: 0.55;
}
.agent-node.active {
    opacity: 1;
    box-shadow: 0 0 18px var(--glow-color);
    transform: scale(1.04);
}

.finding-card {
    background-color: #1C1F26;
    border-left: 3px solid #4FA89B;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.92rem;
}
.limitation-card {
    background-color: #1C1F26;
    border-left: 3px solid #D98E48;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.92rem;
}

.source-tag {
    display: inline-block;
    background-color: #262A33;
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #B8B4AA;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown("## 🧭 Enterprise Multi-Agent GenAI Platform")
st.markdown(
    "<span style='color:#9A968C;'>LangGraph orchestration across SQL, RAG, and Web agents — "
    "running on a local open-source LLM (Ollama + qwen2.5:3b).</span>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------
# Architecture / routing visual setup. We DEFINE the render function
# here (so the section title stays near the top of the file for
# readability) but don't CALL it until after the Ask button's logic
# runs further down — see the st.empty() placeholder note below for why.
# ---------------------------------------------------------------------
if "last_route" not in st.session_state:
    st.session_state.last_route = None


def render_architecture():
    st.markdown("#### Architecture")
    col_router, col_sql, col_rag, col_web = st.columns([1, 1, 1, 1])

    with col_router:
        st.markdown(
            "<div class='agent-node' style='opacity:1;'>ROUTER<br>"
            "<span style='font-size:0.75rem; opacity:0.7;'>question classifier</span></div>",
            unsafe_allow_html=True,
        )

    for col, key in [(col_sql, "sql"), (col_rag, "rag"), (col_web, "web")]:
        active = st.session_state.last_route == key
        active_class = "active" if active else ""
        glow = ROUTE_COLORS[key]
        with col:
            st.markdown(
                f"<div class='agent-node {active_class}' style='--glow-color:{glow}; "
                f"border-color:{glow if active else '#333740'};'>"
                f"{ROUTE_LABELS[key].upper()}</div>",
                unsafe_allow_html=True,
            )

    st.caption("The active node highlights after you ask a question below.")
    st.divider()


# st.empty() reserves this section's POSITION on the page now, but we
# don't draw into it until after the Ask button logic (below) has had
# a chance to update st.session_state.last_route for THIS run. This is
# the standard Streamlit pattern for "draw something near the top that
# depends on a computation that happens later in the script."
architecture_slot = st.empty()

# ---------------------------------------------------------------------
# Live demo panel
# ---------------------------------------------------------------------
st.markdown("#### Ask the platform")

example_questions = [
    "How many employees work in Engineering?",
    "What's our policy on parental leave?",
    "What's the latest news on AI regulation?",
    "What is the total revenue from all orders?",
    "How do I fix a 401 Unauthorized error?",
]

col_input, col_examples = st.columns([2, 1])
with col_examples:
    chosen_example = st.selectbox("Or pick an example:", ["—"] + example_questions)

with col_input:
    default_text = chosen_example if chosen_example != "—" else ""
    question = st.text_input("Your question:", value=default_text, key="question_input")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Routing and generating answer... (local CPU inference can take 30-150s)"):
        try:
            result = ask(question)
            st.session_state.last_route = result["route"]
            st.session_state.last_result = result
            st.session_state.last_question = question
        except Exception as e:
            st.error(
                f"Couldn't reach the local backend: {e}\n\n"
                "Make sure Ollama is running (`ollama serve`) and the model is pulled "
                "(`ollama pull qwen2.5:3b`), and that the RAG index has been built "
                "(`python -m app.vector_store`)."
            )
            st.session_state.last_result = None

# Now that session_state.last_route reflects THIS run's outcome (if the
# button was clicked above), draw the architecture section into the slot
# we reserved earlier — this is what makes the highlight appear on the
# SAME click rather than lagging one interaction behind.
with architecture_slot.container():
    render_architecture()

if st.session_state.get("last_result"):
    result = st.session_state.last_result
    route = result["route"]
    color = ROUTE_COLORS.get(route, "#888")

    st.markdown(
        f"<span class='mono' style='color:{color}; font-weight:700;'>"
        f"→ routed to {ROUTE_LABELS.get(route, route)}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{result['answer']}**")

    if result["sources"]:
        tags = "".join(f"<span class='source-tag'>{s}</span>" for s in result["sources"])
        st.markdown(f"Sources: {tags}", unsafe_allow_html=True)

    st.caption(f"Latency: {result['latency_ms']}ms · Route: {route}")

    with st.expander("Raw trace details (observability)"):
        st.json(result.get("details", {}))

st.divider()

# ---------------------------------------------------------------------
# Findings: what works vs. limitations identified
# ---------------------------------------------------------------------
st.markdown("#### What I found while building this")
col_works, col_limits = st.columns(2)

with col_works:
    st.markdown("**✅ Working well**")
    findings = [
        "Router correctly classifies questions across all 3 domains using few-shot examples.",
        "SQL agent generates correct JOINs/aggregations and is blocked from anything but read-only SELECTs at the database connection level, not just in the prompt.",
        "RAG agent recovers from imperfect retrieval ranking by passing all top-k chunks to the LLM, not just the top-1 match.",
        "Every request is traced end-to-end (route, generated SQL / retrieved chunks / search results, latency) for explainability.",
    ]
    for f in findings:
        st.markdown(f"<div class='finding-card'>{f}</div>", unsafe_allow_html=True)

with col_limits:
    st.markdown("**⚠️ Limitations identified**")
    limitations = [
        "CPU-only local inference is slow (60-150s/question with multiple sequential LLM calls) — production would need GPU or hosted inference.",
        "Web agent grounding is only as good as the search results returned — generic landing-page snippets can lack real substantive content.",
        "Date filtering in generated SQL sometimes uses pattern matching rather than proper range comparisons.",
        "Vector retrieval ranking isn't always precise for closely related concepts (e.g. 'vacation days' vs 'sick days') — mitigated, not eliminated, by passing multiple chunks to generation.",
    ]
    for l in limitations:
        st.markdown(f"<div class='limitation-card'>{l}</div>", unsafe_allow_html=True)

st.divider()
st.caption(
    "Built with LangGraph, ChromaDB, sentence-transformers, FastAPI, and Ollama (qwen2.5:3b). "
    "Source: see README.md and INTERVIEW_PREP.md in this repository."
)
