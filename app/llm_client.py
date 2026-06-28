"""
app/llm_client.py

A tiny wrapper around the Ollama Python client.

WHY A WRAPPER AT ALL (instead of calling `ollama.chat(...)` everywhere)?
------------------------------------------------------------------------
1. Single point of change: if we later swap models, add retries, add
   logging/tracing (we will, in the observability step), or even swap
   providers (Ollama -> OpenAI), we change ONE function, not every
   call site across 3 agents + the router.
2. Interview talking point: "I abstracted the LLM provider behind a
   thin client so agents don't depend on a specific vendor's SDK."
   This is a real architectural pattern (dependency inversion).

MODEL CHOICE HISTORY (worth knowing for interviews)
------------------------------------------------------------------------
We started with llama3.2:1b (1B params) to prototype cheaply. It exposed
two real failure modes during SQL-agent testing:
  1. Over-filtering: it sometimes invented plausible-but-wrong WHERE
     clauses (e.g. guessing a role name that didn't exist in the data).
  2. Refusal bypass: on out-of-scope questions it sometimes echoed the
     previous answer instead of correctly refusing.
Both are capacity-related, not purely prompt-design issues, so we
upgraded to qwen2.5:3b — still small enough for an 8GB-RAM laptop, but
meaningfully stronger at instruction-following and structured output
(SQL, JSON), which is exactly what an agentic system leans on most.
"""

import ollama

MODEL_NAME = "qwen2.5:3b"


def call_llm(prompt: str, system: str | None = None, temperature: float = 0.1) -> str:
    """
    Send a single-turn prompt to the local Ollama model and return the text reply.

    Args:
        prompt: the user-facing instruction/question.
        system: optional system prompt to set behavior/role.
        temperature: lower = more deterministic. We default low because
                     agent tasks (routing, SQL gen) need reliability over creativity.

    Returns:
        The model's text response, stripped of leading/trailing whitespace.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


if __name__ == "__main__":
    # Quick manual smoke test — run this from the project root:
    #   python -m app.llm_client
    reply = call_llm("Say 'pong' and nothing else.")
    print("Model replied:", reply)
