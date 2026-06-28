"""
app/api.py  (Step 7: FastAPI layer)

WHY THIS FILE IS TINY
------------------------------------------------------------------------
All real logic already lives in graph_main.ask(). This file's only
job is to be an HTTP adapter: validate the request shape (Pydantic),
call ask(), shape the response. This mirrors the "thin adapter" idea
we used for sql_node/rag_node/web_node — the API layer doesn't know or
care HOW an answer was produced, only that ask() gives it one.

Run with:  uvicorn app.api:app --reload
Then visit http://127.0.0.1:8000/docs for interactive Swagger UI —
FastAPI generates this automatically from the Pydantic models below,
which is a big part of why FastAPI is the standard choice here.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from app.graph_main import ask
from app.tracing import get_recent_traces, get_route_stats

app = FastAPI(
    title="Enterprise Multi-Agent GenAI Platform",
    description="Routes questions across SQL, RAG, and Web agents via LangGraph.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    route: str
    answer: str
    sources: list[str]
    latency_ms: int


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    """Main entry point: ask a question, get a routed, grounded answer."""
    result = ask(request.question)
    return AskResponse(
        question=request.question,
        route=result["route"],
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
    )


@app.get("/traces")
def traces_endpoint(limit: int = 20):
    """Observability endpoint: recent decision traces (Step 6)."""
    return get_recent_traces(limit=limit)


@app.get("/stats")
def stats_endpoint():
    """Observability endpoint: aggregate stats per route (Step 6)."""
    return get_route_stats()


@app.get("/health")
def health():
    return {"status": "ok"}
