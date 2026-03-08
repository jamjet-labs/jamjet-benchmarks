"""
Example 4: Self-evaluating workflow with retry
===============================================
A workflow that generates an answer, evaluates its own output using
LLM-as-judge, and retries with feedback if quality is below threshold.

This is the "eval as a workflow node" pattern — the same capability
that JamJet exposes as `type: eval` in YAML workflows, shown here
as pure Python.

The loop:
  draft → judge → (score ≥ 4) → done
                → (score < 4) → refine with feedback → judge again
                                (up to MAX_RETRIES attempts)

Run:
    export OPENAI_API_KEY="ollama"
    export OPENAI_BASE_URL="http://localhost:11434/v1"
    export MODEL_NAME="llama3.2"
    python examples/04_self_evaluating_workflow.py

    # Try a harder question to see retries trigger:
    QUESTION="Explain the Byzantine Generals Problem" python examples/04_self_evaluating_workflow.py
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
QUESTION = os.getenv("QUESTION", "Explain what durable AI workflow execution means and why it matters.")
MIN_SCORE = int(os.getenv("MIN_SCORE", "4"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


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
    draft: str = ""
    judge_score: int = 0
    judge_reason: str = ""
    feedback: str = ""
    attempts: int = 0
    final: str = ""


# ── Workflow ───────────────────────────────────────────────────────────────────

wf = Workflow("self-eval-agent")


@wf.state
class SelfEvalState(State):
    pass


@wf.step(
    next={
        # If we already have a good-enough draft, skip straight to refine with its feedback.
        # Fresh start goes to judge.
        "judge": lambda s: s.draft != "" and s.judge_score == 0,
    }
)
async def draft(state: SelfEvalState) -> SelfEvalState:
    """Generate or refine the answer draft."""
    if state.feedback:
        system = (
            "You are a technical writer. Revise the draft based on the feedback provided. "
            "Be specific, accurate, and clear. 3-4 sentences."
        )
        user = f"Question: {state.question}\n\nPrevious draft:\n{state.draft}\n\nFeedback:\n{state.feedback}"
    else:
        system = "You are a concise technical writer. Answer clearly and accurately in 3-4 sentences."
        user = state.question

    text = llm(system, user)
    return state.model_copy(update={
        "draft": text,
        "judge_score": 0,
        "judge_reason": "",
        "attempts": state.attempts + 1,
    })


@wf.step(
    next={
        # Score meets bar → accept
        "accept": lambda s: s.judge_score >= MIN_SCORE,
        # Score below bar and retries remain → go back and revise
        "draft":  lambda s: s.judge_score < MIN_SCORE and s.attempts < MAX_RETRIES,
        # Retries exhausted → accept whatever we have
        "accept": lambda s: s.attempts >= MAX_RETRIES,
    }
)
async def judge(state: SelfEvalState) -> SelfEvalState:
    """LLM-as-judge: score the draft and generate feedback."""
    import json

    prompt = (
        f"Evaluate this answer to the question:\n\n"
        f"Question: {state.question}\n\n"
        f"Answer: {state.draft}\n\n"
        f"Score 1-5 on: accuracy, clarity, completeness.\n"
        f'Respond with ONLY valid JSON: {{"score": <1-5>, "reason": "<one sentence>", "feedback": "<specific improvement>"}}'
    )
    raw = llm("You are an impartial technical evaluator.", prompt, max_tokens=150)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        score = int(parsed.get("score", 0))
        reason = parsed.get("reason", "")
        feedback = parsed.get("feedback", "")
    except Exception:
        score = 3
        reason = "could not parse judge response"
        feedback = "Make the answer more specific and complete."

    return state.model_copy(update={
        "judge_score": score,
        "judge_reason": reason,
        "feedback": feedback,
    })


@wf.step
async def accept(state: SelfEvalState) -> SelfEvalState:
    """Accept the draft as the final answer."""
    return state.model_copy(update={"final": state.draft})


# ── Run ────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"\nModel      : {MODEL}")
    print(f"Question   : {QUESTION}")
    print(f"Min score  : {MIN_SCORE}/5")
    print(f"Max retries: {MAX_RETRIES}\n")

    result = wf.run_sync(SelfEvalState(question=QUESTION))
    s = result.state

    print("─" * 64)
    print("  Execution timeline")
    print("─" * 64)
    for evt in result.events:
        status = "✓" if evt.status == "completed" else "✗"
        bar = "█" * max(1, int(evt.duration_us / 100_000))
        print(f"  {status} {evt.step:<18} {evt.duration_us / 1000:>7.1f}ms  {bar}")
    print("─" * 64)
    print(f"  Total: {result.total_duration_us / 1000:.1f}ms  |  {s.attempts} draft(s)  |  {result.steps_executed} steps")
    print()

    if s.attempts > 1:
        print(f"  Revised {s.attempts} time(s) to reach score {s.judge_score}/5")
    else:
        print(f"  Accepted on first attempt (score {s.judge_score}/5)")

    print(f"\nJudge score : {s.judge_score}/5")
    print(f"Judge reason: {s.judge_reason}")
    print(f"\nFinal answer:\n")
    for line in s.final.splitlines():
        print(f"  {line}")
    print()


if __name__ == "__main__":
    main()
