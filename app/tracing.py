"""
app/tracing.py  (Step 6: observability layer)

WHY SQLITE FOR TRACES TOO
------------------------------------------------------------------------
We already have SQLite wired up (Step 2) — reusing it for traces means
no new infra, and traces are exactly the kind of structured, queryable
data a relational table suits well (filter by route, by date, compute
average latency, etc. — all just SQL).

WHAT "EXPLAINABLE" CONCRETELY MEANS HERE
------------------------------------------------------------------------
For each question we log: which route was chosen, what the agent
actually did internally (generated SQL / retrieved chunk sources+
distances / web URLs used), the final answer, and latency. This is
exactly what lets a human later answer "why did the system say X?"
without re-running anything — the decision trail is persisted.
"""

import sqlite3
import os
import json
import time
from contextlib import contextmanager

TRACE_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "traces.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    question TEXT NOT NULL,
    route TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT NOT NULL,       -- JSON-encoded list
    details TEXT NOT NULL,       -- JSON-encoded agent-specific debug info
    latency_ms INTEGER NOT NULL
);
"""


def _init_db():
    conn = sqlite3.connect(TRACE_DB_PATH)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()


def log_trace(question: str, route: str, answer: str, sources: list, details: dict, latency_ms: int):
    """Persist one full agent decision to the traces table."""
    _init_db()
    conn = sqlite3.connect(TRACE_DB_PATH)
    conn.execute(
        "INSERT INTO traces (timestamp, question, route, answer, sources, details, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            time.strftime("%Y-%m-%d %H:%M:%S"),
            question,
            route,
            answer,
            json.dumps(sources),
            json.dumps(details),
            latency_ms,
        ),
    )
    conn.commit()
    conn.close()


@contextmanager
def timed():
    """Small helper: `with timed() as t: ...` then t() gives elapsed ms."""
    start = time.time()
    yield lambda: int((time.time() - start) * 1000)


def get_recent_traces(limit: int = 20) -> list[dict]:
    """Fetch the most recent traces — this is our 'dashboard' query."""
    _init_db()
    conn = sqlite3.connect(TRACE_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM traces ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_route_stats() -> list[dict]:
    """Aggregate stats per route — count + average latency. A real
    'observability dashboard' query: shows routing distribution and
    performance at a glance, no need to scroll raw logs."""
    _init_db()
    conn = sqlite3.connect(TRACE_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT route, COUNT(*) as count, AVG(latency_ms) as avg_latency_ms "
        "FROM traces GROUP BY route"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    # Manual smoke test / mini "dashboard" — run: python -m app.tracing
    print("Recent traces:")
    for t in get_recent_traces(5):
        print(f"  [{t['timestamp']}] ({t['route']}, {t['latency_ms']}ms) Q: {t['question'][:50]}")
    print("\nStats by route:")
    for s in get_route_stats():
        print(f"  {s['route']}: {s['count']} questions, avg {s['avg_latency_ms']:.0f}ms")
