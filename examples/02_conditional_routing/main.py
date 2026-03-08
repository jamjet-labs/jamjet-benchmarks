"""
Example 2: Conditional routing
================================
A workflow that classifies the user's question and routes it to a
specialist handler — factual questions get a structured answer,
opinion questions get a balanced perspective, unclear questions
get a clarification request.

What makes this JamJet-native:
  - Routing is a plain Python predicate on the Pydantic state — readable,
    testable, no separate graph wiring or edge function API to learn
  - The classifier output is structured state, not an opaque router function
  - You can unit-test the routing logic without running the LLM

Run:
    export OPENAI_API_KEY="ollama"
    export OPENAI_BASE_URL="http://localhost:11434/v1"
    export MODEL_NAME="llama3.2"
    python examples/02_conditional_routing.py

    # Try different question types:
    QUESTION="What year was Python created?" python examples/02_conditional_routing.py
    QUESTION="Is Python better than Go?" python examples/02_conditional_routing.py
    QUESTION="blorp florp zorp?" python examples/02_conditional_routing.py
"""

from __future__ import annotations

import os

from openai import OpenAI
from pydantic import BaseModel

from jamjet import Workflow

# ── Config ────────────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
)
MODEL = os.getenv("MODEL_NAME", "llama3.2")
QUESTION = os.getenv("QUESTION", "Is Python a good language for building AI agents?")


def llm(system: str, user: str, max_tokens: int = 250) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


# ── State ──────────────────────────────────────────────────────────────────────


class State(BaseModel):
    question: str
    question_type: str = ""   # "factual" | "opinion" | "unclear"
    answer: str = ""
    route_taken: str = ""


# ── Workflow ───────────────────────────────────────────────────────────────────

wf = Workflow("conditional-router")


@wf.state
class RouterState(State):
    pass


@wf.step(
    next={
        # Routing predicates: plain Python lambdas on state.
        # These are unit-testable without touching the LLM.
        "answer_factual":  lambda s: s.question_type == "factual",
        "answer_opinion":  lambda s: s.question_type == "opinion",
        "ask_to_clarify":  lambda s: s.question_type == "unclear",
    }
)
async def classify(state: RouterState) -> RouterState:
    """Classify the question, then route."""
    raw = llm(
        'Classify the question as exactly one of: "factual", "opinion", or "unclear". '
        "Respond with the single word only.",
        state.question,
        max_tokens=5,
    )
    q_type = raw.lower().strip().strip('"').strip("'")
    if q_type not in ("factual", "opinion", "unclear"):
        q_type = "unclear"
    return state.model_copy(update={"question_type": q_type})


@wf.step
async def answer_factual(state: RouterState) -> RouterState:
    """Factual path: concise, direct answer with the key fact up front."""
    answer = llm(
        "Answer factually and concisely. Lead with the key fact. 2 sentences max.",
        state.question,
    )
    return state.model_copy(update={"answer": answer, "route_taken": "factual"})


@wf.step
async def answer_opinion(state: RouterState) -> RouterState:
    """Opinion path: balanced perspective with pros and cons."""
    answer = llm(
        "Give a balanced perspective. Present both sides briefly. 3 sentences max.",
        state.question,
    )
    return state.model_copy(update={"answer": answer, "route_taken": "opinion"})


@wf.step
async def ask_to_clarify(state: RouterState) -> RouterState:
    """Unclear path: ask a targeted clarifying question."""
    answer = llm(
        "The question is unclear. Ask one targeted clarifying question to understand what the user needs.",
        state.question,
        max_tokens=80,
    )
    return state.model_copy(update={"answer": answer, "route_taken": "clarify"})


# ── Run ────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"\nModel   : {MODEL}")
    print(f"Question: {QUESTION}\n")

    result = wf.run_sync(RouterState(question=QUESTION))
    s = result.state

    print("─" * 60)
    print(f"  Classified as : {s.question_type}")
    print(f"  Route taken   : {s.route_taken}")
    print(f"  Steps executed: {result.steps_executed}")
    print(f"  Total time    : {result.total_duration_us / 1000:.1f}ms")
    print("─" * 60)

    print("\nTimeline:")
    for evt in result.events:
        arrow = " →" if evt.step == "classify" else "   "
        active = "●" if evt.status == "completed" else "○"
        print(f"  {active}{arrow} {evt.step:<20} {evt.duration_us / 1000:>7.1f}ms")

    print(f"\nAnswer:\n  {s.answer}\n")

    # ── Show that routing logic is independently testable ─────────────────────
    print("─" * 60)
    print("  Routing logic test (no LLM needed):")
    from jamjet.workflow.types import StepDef

    factual_state = RouterState(question="", question_type="factual")
    opinion_state = RouterState(question="", question_type="opinion")
    unclear_state = RouterState(question="", question_type="unclear")

    classify_step = wf._steps[0]
    for state, expected in [(factual_state, "answer_factual"), (opinion_state, "answer_opinion"), (unclear_state, "ask_to_clarify")]:
        for target, predicate in classify_step.next.items():
            if predicate(state):
                status = "✓" if target == expected else "✗"
                print(f"  {status} {state.question_type!r:10} → {target}")
                break


if __name__ == "__main__":
    main()
