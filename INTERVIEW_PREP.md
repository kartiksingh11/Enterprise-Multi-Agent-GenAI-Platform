# Interview Cheat-Sheet

## 30-second pitch
"I built a multi-agent platform that routes questions to a SQL agent, a RAG
agent, or a web-search agent using LangGraph. Each agent has its own
grounding and safety layer — schema-grounded SQL with read-only execution,
vector retrieval with a distance-threshold cutoff, and live search with
explicit anti-hallucination instructions. Every request is traced end-to-end
for explainability, and it's exposed via FastAPI."

## Why LangGraph (not just if/else)?
State management + conditional routing as first-class concepts; scales
cleanly to retries/loops/multi-step chains even though 3 fixed agents alone
don't strictly require a graph. Be honest about this if pressed — it's the
production-standard pattern, not a hard requirement at this scale.

## Real bugs you found and fixed (your strongest material)
1. **SQL agent over-filtering**: model invented a non-existent role value
   ('Engineer') when asked about department headcount. Fixed by injecting
   known categorical column values into the schema prompt, not just names.
2. **SQL agent refusal bypass**: an out-of-scope question made the model
   echo the previous question's SQL instead of refusing. Root cause: one
   prompt doing two jobs (relevance + generation). Fixed by splitting into
   a separate relevance-check call before generation.
3. **Relevance check false negative**: after fixing #2, a valid question
   (date-filtered customer list) got wrongly rejected — classic precision/
   recall tradeoff. Fixed with a directly-relevant few-shot example.
4. **RAG retrieval ranking miss**: "vacation days" query ranked a sick-leave
   chunk above the correct PTO chunk by vector similarity. Mitigated by
   passing all k retrieved chunks to generation (not just top-1), letting
   the LLM reason past an imperfect ranking — validated by testing, not
   assumed.
5. **Web agent grounding on landing pages**: search results from category/
   landing pages had generic page-description snippets, not real content;
   the LLM faithfully echoed them as if they were news. This was *correct
   grounding behavior undermined by low-quality input* — not a hallucination
   — and the fix was an explicit prompt instruction to ignore non-substantive
   snippets.

## Defense-in-depth pattern (mention this explicitly — it's a strong signal)
Never rely on a single layer for safety/correctness:
- SQL: prompt rules + code-level statement validation + read-only DB
  connection (3 independent layers).
- RAG: prompt grounding instruction + code-level distance threshold.
- Router: prompt classification + code-level label validation with a
  named fallback.

## Likely follow-up questions and your answers
- **"What would you change for production?"** GPU/hosted inference for
  latency; a real search API (Tavily/Bing) instead of unofficial DuckDuckGo;
  auto-sampling distinct column values for SQL grounding instead of
  hand-written; LangSmith or OpenTelemetry instead of a custom SQLite trace
  table.
- **"How did you choose chunk size?"** Started from typical paragraph
  length in the source docs, then iterated after finding boundary effects
  (a paragraph marginally over the limit produced a near-duplicate trailing
  chunk) — added tolerance and a minimum-length filter based on actually
  inspecting output, not guessing upfront.
- **"How do you prevent SQL injection from the LLM itself?"** Statement-type
  allowlist (SELECT only) + forbidden-keyword check + a SQLite connection
  opened in true read-only mode at the engine level, so even a validation
  bug can't cause a write.
- **"How do you know retrieval is working well?"** Distance scores
  correlate with relevance in practice — correct matches scored under ~1.0,
  irrelevant ones scored 1.7+, which I confirmed by deliberately testing an
  out-of-domain question and inspecting the actual numbers.
- **"What's the latency/cost tradeoff in your design?"** The relevance-check
  split (SQL agent) and multi-chunk passing (RAG agent) both trade extra
  LLM calls/tokens for reliability — explicit, named tradeoffs, not
  accidents.
