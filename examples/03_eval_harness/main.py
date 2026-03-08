"""
Example 3: Built-in eval harness
==================================
Run a JSONL dataset through a JamJet workflow and score every output
with multiple scorers — assertions, latency budgets, and LLM-as-judge.
Print a Rich summary table with pass rates and per-row details.

This is completely unique to JamJet. LangGraph, CrewAI, AutoGen — none
of them ship an eval harness. You either write one yourself or pay for
a separate platform. JamJet includes it in the SDK.

Run:
    export OPENAI_API_KEY="ollama"
    export OPENAI_BASE_URL="http://localhost:11434/v1"
    export MODEL_NAME="llama3.2"
    python examples/03_eval_harness.py
"""

from __future__ import annotations

import asyncio
import os
import time

from openai import OpenAI
from pydantic import BaseModel

from jamjet import Workflow
from jamjet.eval.dataset import EvalDataset, EvalRow
from jamjet.eval.runner import EvalResult, EvalRunner
from jamjet.eval.scorers import AssertionScorer, BaseScorer, LatencyScorer, LlmJudgeScorer, ScorerResult

# ── Config ────────────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
)
MODEL = os.getenv("MODEL_NAME", "llama3.2")

# ── The workflow under test ───────────────────────────────────────────────────


class State(BaseModel):
    question: str
    answer: str = ""


wf = Workflow("qa-agent")


@wf.state
class QAState(State):
    pass


@wf.step
async def answer(state: QAState) -> QAState:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=200,
        messages=[
            {"role": "system", "content": "Answer clearly and concisely in 2-3 sentences."},
            {"role": "user", "content": state.question},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    return state.model_copy(update={"answer": text})


# ── Inline runner (bypasses HTTP runtime — runs the workflow in-process) ───────


class InlineRunner:
    """
    Runs the JamJet workflow in-process and applies scorers.
    Identical scoring logic to EvalRunner — but no runtime server needed.
    """

    def __init__(self, scorers: list[BaseScorer]) -> None:
        self.scorers = scorers

    async def run(self, dataset: EvalDataset) -> list[EvalResult]:
        results = []
        for row in dataset:
            t0 = time.perf_counter()
            error = None
            output = None
            try:
                exec_result = await wf.run(QAState(question=row.input["question"]))
                output = {"answer": exec_result.state.answer}
            except Exception as e:
                error = str(e)
            duration_ms = (time.perf_counter() - t0) * 1000.0

            scorer_results: list[ScorerResult] = []
            if output is not None:
                for scorer in self.scorers:
                    try:
                        sr = await scorer.score(
                            output,
                            expected=row.expected,
                            duration_ms=duration_ms,
                            input_data=row.input,
                        )
                        scorer_results.append(sr)
                    except Exception as e:
                        scorer_results.append(ScorerResult(scorer.name, False, None, f"scorer error: {e}"))

            results.append(
                EvalResult(
                    row_id=row.id,
                    input=row.input,
                    expected=row.expected,
                    output=output,
                    scorers=scorer_results,
                    duration_ms=duration_ms,
                    cost_usd=None,
                    error=error,
                )
            )
        return results


# ── Dataset ───────────────────────────────────────────────────────────────────

DATASET = EvalDataset([
    EvalRow(id="q1", input={"question": "What is event sourcing?"}, expected={"min_words": 20}),
    EvalRow(id="q2", input={"question": "Name the three laws of robotics."}, expected={"min_words": 15}),
    EvalRow(id="q3", input={"question": "What does REST stand for?"}, expected={"min_words": 5}),
    EvalRow(id="q4", input={"question": "Explain CAP theorem in one sentence."}, expected={"min_words": 10}),
    EvalRow(id="q5", input={"question": "What is a Pydantic model?"}, expected={"min_words": 15}),
])


# ── LLM judge using the same local Ollama model ───────────────────────────────


async def local_judge(prompt: str) -> str:
    """Use the same local model as judge — free, no API key needed."""
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


# ── Scorers ───────────────────────────────────────────────────────────────────

SCORERS: list[BaseScorer] = [
    AssertionScorer(checks=[
        "len(output.get('answer', '').split()) >= 10",   # at least 10 words
        "output.get('answer', '') != ''",                 # non-empty
    ]),
    LatencyScorer(threshold_ms=15_000),   # 15s — generous for local Ollama
    LlmJudgeScorer(
        rubric="Is this answer accurate, clear, and directly responsive to the question? Score 1-5.",
        min_score=3,
        model_fn=local_judge,
    ),
]


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    print(f"\nModel   : {MODEL}")
    print(f"Dataset : {len(DATASET)} rows")
    print(f"Scorers : {', '.join(s.name for s in SCORERS)}\n")

    runner = InlineRunner(scorers=SCORERS)
    results = await runner.run(DATASET)

    # ── Rich summary table ─────────────────────────────────────────────────────
    from rich.console import Console
    from rich.table import Table

    console = Console()
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_pct = passed / total * 100 if total else 0

    console.rule(f"[bold]Eval Results — {passed}/{total} passed ({pass_pct:.0f}%)[/bold]")

    table = Table(show_header=True, header_style="bold yellow")
    table.add_column("Row", style="dim", width=4)
    table.add_column("Question", width=38)
    table.add_column("Pass", justify="center", width=5)
    table.add_column("ms", justify="right", width=7)
    table.add_column("Assertion", justify="center", width=9)
    table.add_column("Latency", justify="center", width=8)
    table.add_column("LLM judge", justify="center", width=10)
    table.add_column("Answer (truncated)", width=40)

    for r in results:
        by_name = {s.scorer: s for s in r.scorers}
        a = by_name.get("assertion")
        l = by_name.get("latency")
        j = by_name.get("llm_judge")

        def icon(s: ScorerResult | None) -> str:
            if s is None: return "—"
            return "[green]✓[/green]" if s.passed else "[red]✗[/red]"

        def judge_str(s: ScorerResult | None) -> str:
            if s is None or s.score is None: return "—"
            return f"[green]{s.score:.0f}/5[/green]" if s.passed else f"[red]{s.score:.0f}/5[/red]"

        answer_preview = ""
        if r.output:
            answer_preview = r.output.get("answer", "")[:60].replace("\n", " ")
            if len(r.output.get("answer", "")) > 60:
                answer_preview += "…"
        if r.error:
            answer_preview = f"[red]ERROR: {r.error[:40]}[/red]"

        overall = "[green]✓[/green]" if r.passed else "[red]✗[/red]"

        table.add_row(
            r.row_id,
            r.input.get("question", "")[:38],
            overall,
            f"{r.duration_ms:.0f}",
            icon(a),
            icon(l),
            judge_str(j),
            answer_preview,
        )

    console.print(table)

    # ── Per-scorer summary ─────────────────────────────────────────────────────
    console.print()
    scorer_names = [s.name for s in SCORERS]
    for name in scorer_names:
        scores = [r.scorers for r in results]
        matching = [s for row in scores for s in row if s.scorer == name]
        n_pass = sum(1 for s in matching if s.passed)
        console.print(f"  [bold]{name}[/bold]: {n_pass}/{len(matching)} passed")

    console.print(f"\n[bold]Overall pass rate: {pass_pct:.0f}%[/bold]")
    console.print(f"[dim]Run `QUESTION=... python 01_pipeline_with_timeline.py` to inspect any row individually[/dim]\n")


if __name__ == "__main__":
    asyncio.run(main())
