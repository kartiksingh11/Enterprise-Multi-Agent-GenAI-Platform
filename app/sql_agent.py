"""
app/sql_agent.py  (Step 2b: generation only — no execution yet)

WHY WE HAND-CRAFT THE SCHEMA STRING
------------------------------------
The LLM has zero awareness of our actual database. The ONLY way it
knows table/column names is if we put them in the prompt as plain
text. This is called "schema grounding" and it's the #1 lever for
reducing hallucinated column/table names in text-to-SQL systems.

We could auto-generate this string by introspecting SQLite
(PRAGMA table_info), and we WILL do that in a later refinement —
but writing it by hand first means you understand exactly what the
model is seeing, with nothing hidden by tooling magic.

WHY A STRICT, EXAMPLE-DRIVEN PROMPT
------------------------------------
Even with a stronger model (qwen2.5:3b, upgraded from llama3.2:1b after
hitting real failures — see app/llm_client.py for that history), small
local models can still:
- add explanations before/after the SQL unless explicitly told not to
- sometimes wrap the SQL in markdown fences (```sql ... ```)
So our prompt stays explicit and our parser stays defensive (we strip
markdown fences if present). This is a real engineering pattern:
"prompt for the ideal case, code defensively for the realistic case" —
worth keeping even after upgrading models, since no model is 100% reliable.
"""

from app.llm_client import call_llm

SCHEMA_DESCRIPTION = """
Tables:

departments(id INTEGER, name TEXT, budget REAL)
employees(id INTEGER, name TEXT, department_id INTEGER, role TEXT, salary REAL, hire_date TEXT)
customers(id INTEGER, name TEXT, email TEXT, signup_date TEXT)
products(id INTEGER, name TEXT, category TEXT, price REAL)
orders(id INTEGER, customer_id INTEGER, product_id INTEGER, quantity INTEGER, order_date TEXT)

Relationships:
- employees.department_id -> departments.id
- orders.customer_id -> customers.id
- orders.product_id -> products.id

Known values (do not invent other values for these columns):
- departments.name: 'Engineering', 'Sales', 'Marketing', 'HR'
- employees.role: 'Senior Engineer', 'Engineering Manager', 'Junior Engineer', 'Sales Lead', 'Account Executive', 'Marketing Manager', 'HR Generalist'
- products.category: 'Software', 'Services'

IMPORTANT: To find employees "in" or "working in" a department, filter ONLY by
departments.name. Do NOT also filter by employees.role unless the question
explicitly names a role.
""".strip()

SQL_SYSTEM_PROMPT = f"""You are a SQL generator for a SQLite database.

{SCHEMA_DESCRIPTION}

Rules:
- Output ONLY a single SQL SELECT statement. No explanation. No markdown. No comments.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, or any statement that modifies data.
- Use only the tables and columns listed above.
- If the question cannot be answered with these tables, output exactly: NO_QUERY

Examples:

Question: How many products are in the Software category?
SQL: SELECT COUNT(*) FROM products WHERE category = 'Software'

Question: What is today's weather forecast?
SQL: NO_QUERY

Question: Tell me a joke.
SQL: NO_QUERY
"""

# Known tables — used to validate the model's output in code.
# We never trust the LLM's own judgement about whether a query is
# in-scope; we check it ourselves against ground truth.
KNOWN_TABLES = {"departments", "employees", "customers", "products", "orders"}


RELEVANCE_SYSTEM_PROMPT = """You decide if a question can be answered using a company database
about: departments, employees, customers, products, and orders.

The database includes dates (employees.hire_date, customers.signup_date,
orders.order_date), so questions that filter by year or date ARE answerable.

Reply with EXACTLY one word: YES or NO.

Examples:
Question: How many employees are in Sales?
Answer: YES

Question: What's the weather today?
Answer: NO

Question: Tell me a joke.
Answer: NO

Question: What is the most expensive product?
Answer: YES

Question: List all customers who signed up in 2023.
Answer: YES

Question: Which employees were hired before 2022?
Answer: YES
"""


def is_relevant_question(question: str) -> bool:
    """
    Cheap pre-check run BEFORE SQL generation: is this question even
    about our data domain at all?

    WHY THIS IS A SEPARATE STEP (not folded into generate_sql):
    Asking one small model to do two jobs at once ("decide relevance"
    AND "write correct SQL") in a single prompt overloads it — we saw
    this fail in practice (Step 2b: an out-of-scope question caused the
    model to just repeat its last successful SQL output instead of
    refusing). Splitting into two focused, single-purpose prompts is
    a real prompting pattern: smaller models do better with one job
    per call than one call doing many jobs.

    This is also a miniature preview of what the LangGraph ROUTER will
    do later for all 3 agents — we're learning that pattern here first,
    scoped to just this one agent.
    """
    # NOTE: temperature 0.1 (not 0.0) — fully greedy decoding occasionally
    # over-anchored on the closest few-shot example's surface pattern rather
    # than reasoning about the actual schema (we saw this cause a false
    # negative on a valid date-filtering question). A touch of temperature
    # reduces that brittleness without sacrificing much consistency.
    raw = call_llm(prompt=question, system=RELEVANCE_SYSTEM_PROMPT, temperature=0.1)
    return raw.strip().upper().startswith("YES")


def generate_sql(question: str) -> str:
    """
    Ask the LLM to convert a natural-language question into a SQL SELECT query.
    Does NOT execute it — that's a separate, deliberately isolated step (2c).

    Pipeline:
      1. Check relevance first (is_relevant_question) — cheap, focused call.
      2. Only if relevant, ask the model to generate SQL.
      3. Validate the generated SQL touches a real table (catches gibberish/
         malformed output — NOT a semantic-correctness check; see note below).

    Returns "NO_QUERY" if the question is out of scope, or if generation
    produces something that doesn't look like valid SQL against our schema.
    """
    if not is_relevant_question(question):
        return "NO_QUERY"

    raw = call_llm(prompt=question, system=SQL_SYSTEM_PROMPT, temperature=0.1)
    sql = _clean_sql_output(raw)

    if sql == "NO_QUERY":
        return sql

    if not _references_known_table(sql):
        return "NO_QUERY"

    return sql


def _references_known_table(sql: str) -> bool:
    """
    Lightweight syntactic safety net: does the SQL mention at least one
    table we actually have? This catches gibberish or malformed output
    from the SQL-generation call itself.

    IMPORTANT LIMITATION (confirmed by testing): this does NOT catch a
    stale/wrong query that happens to reference a real table — e.g. the
    model echoing a previous question's correct-looking SQL for an
    unrelated new question. That class of error can only be caught by
    checking relevance BEFORE generation, which is why is_relevant_question()
    exists as a separate, earlier step in the pipeline.
    """
    sql_lower = sql.lower()
    return any(table in sql_lower for table in KNOWN_TABLES)


def _clean_sql_output(raw: str) -> str:
    """
    Defensive cleanup for small-model quirks:
    - strips markdown code fences (```sql ... ``` or ``` ... ```)
    - strips leading/trailing whitespace
    - takes only the first statement if the model rambles extra text after it
    """
    text = raw.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    return text


if __name__ == "__main__":
    # Manual smoke test — run: python -m app.sql_agent
    test_questions = [
        "How many employees work in the Engineering department?",
        "What is the total revenue from all orders?",
        "List all customers who signed up in 2023.",
        "What's the weather like today?",  # should trigger NO_QUERY
    ]

    for q in test_questions:
        sql = generate_sql(q)
        print(f"Q: {q}\nSQL: {sql}\n{'-'*60}")
