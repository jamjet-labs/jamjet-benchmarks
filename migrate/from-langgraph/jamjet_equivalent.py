"""
JamJet equivalent of the LangGraph multi-step agent with routing.

Key differences:
  - State is a typed Pydantic model (not TypedDict) — validated at every step
  - Routing uses plain Python predicates on state (no separate edge functions)
  - `workflow.run_sync()` for local execution; `workflow.compile()` + runtime for production
  - Steps are async by default — sync steps work too
  - For production: swap run_sync() for compile() + submit to `jamjet dev` runtime
    → gets you durability, retries, event sourcing, and crash recovery for free
"""

from __future__ import annotations

from pydantic import BaseModel

from jamjet import Workflow


# ── State ──────────────────────────────────────────────────────────────────────


class State(BaseModel):
    question: str
    needs_search: bool = False
    search_results: list[str] = []
    answer: str = ""


# ── Workflow ───────────────────────────────────────────────────────────────────

wf = Workflow("research-agent")


@wf.state
class AgentState(State):
    pass


@wf.step
async def route(state: AgentState) -> AgentState:
    """Decide whether the question needs a web search."""
    q = state.question.lower()
    needs = any(w in q for w in ["latest", "current", "today", "news", "2024", "2025"])
    return state.model_copy(update={"needs_search": needs})


@wf.step(
    next={
        # If needs_search is True, jump to `search`; otherwise fall through to `answer`
        "search": lambda s: s.needs_search,
    }
)
async def check_route(state: AgentState) -> AgentState:
    return state  # pure routing step — no mutation


@wf.step
async def search(state: AgentState) -> AgentState:
    """Simulate a web search (replace with a real MCP tool call)."""
    results = [f"[result for: {state.question}]"]
    return state.model_copy(update={"search_results": results})


@wf.step
async def answer(state: AgentState) -> AgentState:
    """Synthesize a final answer from state."""
    context = "\n".join(state.search_results)
    final = f"Answer to '{state.question}'"
    if context:
        final += f"\nContext used: {context}"
    return state.model_copy(update={"answer": final})


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = wf.run_sync(AgentState(question="What is durable workflow orchestration?"))
    print(result.state.answer)

    # ── For production: submit to the durable runtime ──────────────────────────
    # ir = wf.compile()                        # compile to IR
    # # then: jamjet dev  (in another terminal)
    # # then: jamjet run --workflow research-agent --input '{"question": "..."}'
