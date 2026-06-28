"""
app/sql_executor.py  (Step 2c: safe execution + answer formatting)

THREAT MODEL
------------
The SQL we're about to run was WRITTEN BY AN LLM. Even with a good
prompt and a relevance pre-check (sql_agent.py), we should never fully
trust generated SQL the way we'd trust SQL we wrote ourselves. This
file is "defense in depth" — multiple independent layers, so that if
one fails, another still protects the database.

LAYER 1 — statement-type check (in Python, before touching the DB):
    Reject anything that isn't a SELECT. Cheap, fast, catches the
    obvious case.

LAYER 2 — read-only connection (in SQLite itself):
    SQLite lets us open a connection in TRUE read-only mode via a URI
    (`file:...?mode=ro`). Even if Layer 1 had a bug or was bypassed,
    the database engine itself will refuse to execute a write — this
    is what "defense in depth" really means: don't rely on one layer.

LAYER 3 — LLM only EXPLAINS the result, never recalculates it:
    When phrasing the natural-language answer, we give the LLM the
    actual query result rows and ask it to describe them — not to
    redo any math. LLMs (especially smaller/local ones) are unreliable
    at arithmetic; SQLite already did the correct computation, so we
    let the LLM's job be "communicate this clearly," not "verify this."
"""

import sqlite3
import os

from app.llm_client import call_llm

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "enterprise.db")


class UnsafeQueryError(Exception):
    """Raised when a generated query fails our safety checks."""
    pass


def _ensure_select_only(sql: str) -> None:
    """
    LAYER 1: reject anything that isn't a single SELECT statement.

    We check the FIRST keyword only intentionally — this is simple and
    fast. It is NOT a full SQL parser, and that's a deliberate scope
    limit (see Layer 2 for the real safety guarantee).
    """
    normalized = sql.strip().lstrip("(").strip().upper()
    if not normalized.startswith("SELECT"):
        raise UnsafeQueryError(
            f"Rejected non-SELECT statement: {sql[:50]}..."
        )

    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "PRAGMA"]
    for kw in forbidden:
        if kw in normalized:
            raise UnsafeQueryError(
                f"Rejected query containing forbidden keyword '{kw}': {sql[:50]}..."
            )


def execute_sql(sql: str) -> list[tuple]:
    """
    Execute a SELECT query against the database in a TRUE read-only
    connection (LAYER 2). Raises UnsafeQueryError if Layer 1 rejects
    the statement first.

    Returns the raw rows (list of tuples). We deliberately return raw
    rows here rather than a formatted string — execution and
    presentation are separate concerns (same philosophy as keeping
    SQL generation and execution in separate functions/files).
    """
    _ensure_select_only(sql)

    # mode=ro is what actually makes this read-only at the SQLite engine
    # level, not just "we promise not to write." uri=True is required
    # for SQLite to parse the connection string as a URI with options.
    db_uri = f"file:{os.path.abspath(DB_PATH)}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return rows
    finally:
        conn.close()


ANSWER_SYSTEM_PROMPT = """You are explaining a database query result to a business user
in plain English.

Rules:
- Use ONLY the numbers/values given to you. Do NOT recalculate or guess any number.
- Be concise: 1-3 sentences.
- Do not mention SQL, queries, or databases — just state the answer naturally.
"""


def format_answer(question: str, rows: list[tuple]) -> str:
    """
    Turn raw SQL result rows into a natural-language answer.

    IMPORTANT: we pass the rows to the LLM as already-computed facts
    and explicitly instruct it not to recalculate anything (Layer 3
    in the threat model above). The LLM's job is purely linguistic.
    """
    if not rows:
        return "I couldn't find any matching data for that question."

    rows_str = str(rows)
    prompt = f"Question: {question}\nQuery result rows: {rows_str}\n\nAnswer:"
    return call_llm(prompt=prompt, system=ANSWER_SYSTEM_PROMPT, temperature=0.2)


if __name__ == "__main__":
    # Manual smoke test — run: python -m app.sql_executor
    from app.sql_agent import generate_sql

    test_questions = [
        "How many employees work in the Engineering department?",
        "What is the total revenue from all orders?",
        "List all customers who signed up in 2023.",
    ]

    for q in test_questions:
        sql = generate_sql(q)
        print(f"Q: {q}")
        print(f"SQL: {sql}")

        if sql == "NO_QUERY":
            print("Answer: I can't answer that from this database.")
        else:
            try:
                rows = execute_sql(sql)
                print(f"Raw rows: {rows}")
                answer = format_answer(q, rows)
                print(f"Answer: {answer}")
            except UnsafeQueryError as e:
                print(f"BLOCKED: {e}")
            except sqlite3.Error as e:
                print(f"SQL ERROR (model likely hallucinated a column/table): {e}")

        print("-" * 60)
