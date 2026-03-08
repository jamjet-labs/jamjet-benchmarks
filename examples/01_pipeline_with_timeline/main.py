"""
Example 1: Multi-step pipeline with execution timeline
=======================================================
A 3-step research pipeline that shows JamJet's per-step event log.

After each run you get a full breakdown of what each step did, how long
it took, and what the state looked like at every transition — without
adding a single line of logging code.

Other frameworks: add print() everywhere, hope nothing crashes mid-run.
JamJet: result.events gives you the full timeline automatically.

Run:
    export OPENAI_API_KEY="ollama"
    export OPENAI_BASE_URL="http://localhost:11434/v1"
    export MODEL_NAME="llama3.2"          # or qwen3:8b, etc.
    python examples/01_pipeline_with_timeline.py
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
QUESTION = os.getenv("QUESTION", "What are the key benefits of event sourcing in distributed systems?")


def llm(system: str, user: str, max_tokens: int = 300) -> str:
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
    keywords: list[str] = []
    outline: str = ""
    answer: str = ""


# ── Workflow ───────────────────────────────────────────────────────────────────

wf = Workflow("research-pipeline")


@wf.state
class PipelineState(State):
    pass


@wf.step
async def extract_keywords(state: PipelineState) -> PipelineState:
    """Step 1: Pull key concepts from the question."""
    raw = llm(
        "You are a keyword extractor. Extract 3-5 key technical concepts from the question. "
        "Respond with a comma-separated list only. No explanation.",
        state.question,
        max_tokens=60,
    )
    keywords = [k.strip() for k in raw.split(",") if k.strip()]
    return state.model_copy(update={"keywords": keywords})


@wf.step
async def build_outline(state: PipelineState) -> PipelineState:
    """Step 2: Build an answer outline from the keywords."""
    raw = llm(
        "You are a technical writer. Create a brief outline (3 bullet points) for answering the question. "
        "Use the provided keywords. Respond with bullet points only.",
        f"Question: {state.question}\nKeywords: {', '.join(state.keywords)}",
        max_tokens=120,
    )
    return state.model_copy(update={"outline": raw})


@wf.step
async def write_answer(state: PipelineState) -> PipelineState:
    """Step 3: Write the final answer using the outline."""
    raw = llm(
        "You are a concise technical writer. Write a clear, structured answer following the outline. "
        "3-4 sentences max.",
        f"Question: {state.question}\nOutline:\n{state.outline}",
        max_tokens=200,
    )
    return state.model_copy(update={"answer": raw})


# ── Run + timeline ─────────────────────────────────────────────────────────────


def main() -> None:
    print(f"\nModel  : {MODEL}")
    print(f"Question: {QUESTION}\n")

    result = wf.run_sync(PipelineState(question=QUESTION))

    # ── What you get for free: per-step event log ──────────────────────────────
    print("─" * 60)
    print("  Execution timeline")
    print("─" * 60)
    for evt in result.events:
        bar = "█" * max(1, int(evt.duration_us / 50_000))
        status = "✓" if evt.status == "completed" else "✗"
        print(f"  {status} {evt.step:<22} {evt.duration_us / 1000:>7.1f}ms  {bar}")
    print("─" * 60)
    print(f"  Total: {result.total_duration_us / 1000:.1f}ms  ({result.steps_executed} steps)\n")

    # ── State at each step ────────────────────────────────────────────────────
    print("Keywords extracted:")
    for kw in result.state.keywords:
        print(f"  · {kw}")

    print("\nOutline:")
    for line in result.state.outline.splitlines():
        print(f"  {line}")

    print("\nFinal answer:")
    for line in result.state.answer.splitlines():
        print(f"  {line}")


if __name__ == "__main__":
    main()
