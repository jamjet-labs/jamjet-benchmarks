"""
JamJet equivalent of the raw OpenAI agentic loop.

What you get for free vs the raw loop:
  - Automatic retry with configurable backoff (no hand-rolled retry logic)
  - Tool calls via MCP — no manual dispatch table
  - Every step is an event: queryable, replayable, diffable
  - Durable execution: crash at step N, resume from step N (not from scratch)
  - `jamjet inspect <exec-id>` to see the full timeline
  - Zero changes to go from local dev to production runtime
"""

from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel

from jamjet import Workflow

client = OpenAI()


# ── State ──────────────────────────────────────────────────────────────────────


class State(BaseModel):
    question: str
    search_results: str = ""
    answer: str = ""


# ── Simulated tool (swap for MCP in production) ────────────────────────────────


def web_search(query: str) -> str:
    return f"[search results for: {query}]"


# ── Workflow ───────────────────────────────────────────────────────────────────

wf = Workflow("research-agent")


@wf.state
class AgentState(State):
    pass


@wf.step
async def search(state: AgentState) -> AgentState:
    """
    Tool step: search the web.

    In production with YAML:
      type: tool
      server: brave-search
      tool: web_search
      arguments:
        query: "{{ state.question }}"
      output_key: search_results
      retry:
        max_attempts: 3
        backoff: exponential

    No manual retry logic, no dispatch table.
    """
    results = web_search(state.question)
    return state.model_copy(update={"search_results": results})


@wf.step
async def synthesize(state: AgentState) -> AgentState:
    """
    Model step: synthesize results into a final answer.

    In production with YAML:
      type: model
      model: gpt-4.1-mini
      system: "You are a helpful research assistant."
      prompt: |
        Question: {{ state.question }}
        Search results: {{ state.search_results }}
        Provide a comprehensive answer.
      output_key: answer
    """
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant."},
            {
                "role": "user",
                "content": (
                    f"Question: {state.question}\n"
                    f"Search results: {state.search_results}\n"
                    "Provide a comprehensive answer."
                ),
            },
        ],
    )
    answer = resp.choices[0].message.content or ""
    return state.model_copy(update={"answer": answer})


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = wf.run_sync(AgentState(question="What are the latest developments in AI agent frameworks?"))
    print(result.state.answer)
    print(f"\nExecuted {result.steps_executed} steps in {result.total_duration_us / 1000:.1f}ms")
