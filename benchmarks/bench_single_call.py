"""
bench_single_call.py
====================
Head-to-head latency benchmark: Raw LLM call vs JamJet vs LangGraph.

All three runners make the *same* LLM call through the same OpenAI-compatible
client — what we are measuring is the **framework orchestration overhead**
on top of the raw network time.

Works with any OpenAI-compatible endpoint:
  - OpenAI:   default (no env overrides needed)
  - Ollama:   OPENAI_API_KEY=ollama  OPENAI_BASE_URL=http://localhost:11434/v1
  - Together, Groq, etc.: set OPENAI_BASE_URL + OPENAI_API_KEY accordingly

Usage
-----
  # Ollama (local, free)
  export OPENAI_API_KEY="ollama"
  export OPENAI_BASE_URL="http://localhost:11434/v1"
  export MODEL_NAME="qwen3:8b"
  python bench_single_call.py

  # OpenAI
  export OPENAI_API_KEY="sk-..."
  export MODEL_NAME="gpt-4.1-mini"
  python bench_single_call.py

  # Control runs / question
  RUNS=30 QUESTION="What is a durable workflow?" python bench_single_call.py

  # Save results to JSON
  python bench_single_call.py --json results/$(date +%Y-%m-%d).json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

RUNS = int(os.getenv("RUNS", "20"))
WARMUP = int(os.getenv("WARMUP", "3"))
QUESTION = os.getenv(
    "QUESTION",
    "Explain what durable AI workflows are in 3 short bullet points.",
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "180"))

SYSTEM_PROMPT = (
    "You are a concise technical assistant. "
    "Answer in exactly 3 short bullet points. "
    "Do not use markdown headings."
)

# ── Shared LLM client ─────────────────────────────────────────────────────────

_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def llm_call(question: str) -> str:
    resp = _client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    msg = resp.choices[0].message
    # Some thinking models (qwen3, deepseek-r1) put the final answer in `content`
    # but chain-of-thought in a `reasoning` field — fall back to reasoning if
    # content is empty (happens when max_tokens is hit during the thinking phase).
    content = msg.content or ""
    if not content.strip():
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
        if reasoning:
            content = f"[thinking] {reasoning[:200]}..."
    return content


# ── Runner factories ──────────────────────────────────────────────────────────


def make_raw_runner() -> Callable[[str], str]:
    """Baseline: bare OpenAI client call, no framework."""

    def run_once(question: str) -> str:
        return llm_call(question)

    return run_once


def make_jamjet_runner() -> Callable[[str], str]:
    """
    JamJet in-process runner using Workflow.run_sync().

    JamJet's Python SDK has a built-in local executor — no runtime server
    needed for single-process workloads. For production, compile() the
    workflow and submit to the Rust runtime for durable execution.
    """
    from pydantic import BaseModel

    from jamjet import Workflow

    wf = Workflow("bench_single_call")

    @wf.state
    class State(BaseModel):
        question: str
        answer: str | None = None

    @wf.step
    async def ask_model(state: State) -> State:
        return state.model_copy(update={"answer": llm_call(state.question)})

    def run_once(question: str) -> str:
        result = wf.run_sync(State(question=question))
        return result.state.answer or ""

    return run_once


def make_langgraph_runner() -> Callable[[str], str]:
    """LangGraph runner using StateGraph with a single node."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        question: str
        answer: str

    def ask_model(state: State) -> State:
        return {"question": state["question"], "answer": llm_call(state["question"])}

    g = StateGraph(State)
    g.add_node("ask_model", ask_model)
    g.add_edge(START, "ask_model")
    g.add_edge("ask_model", END)
    app = g.compile()

    def run_once(question: str) -> str:
        out = app.invoke({"question": question, "answer": ""})
        return out["answer"]

    return run_once


# ── Benchmark harness ─────────────────────────────────────────────────────────


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.array(values, dtype=float), p))


def run_benchmark(name: str, fn: Callable[[str], str]) -> dict[str, Any]:
    sample_output: str | None = None

    # Warmup (not measured)
    for _ in range(WARMUP):
        fn(QUESTION)

    times_ms: list[float] = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = fn(QUESTION)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        if sample_output is None:
            sample_output = out

    result = {
        "name": name,
        "runs": RUNS,
        "warmup": WARMUP,
        "model": MODEL_NAME,
        "base_url": OPENAI_BASE_URL,
        "mean_ms": round(statistics.mean(times_ms), 2),
        "median_ms": round(statistics.median(times_ms), 2),
        "p95_ms": round(percentile(times_ms, 95), 2),
        "p99_ms": round(percentile(times_ms, 99), 2),
        "min_ms": round(min(times_ms), 2),
        "max_ms": round(max(times_ms), 2),
        "stdev_ms": round(statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0, 2),
        "sample_output": sample_output,
        "raw_ms": times_ms,
    }
    return result


def print_results(results: list[dict[str, Any]], baseline_ms: float | None = None) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  Benchmark: single LLM call orchestration overhead")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  Endpoint:  {OPENAI_BASE_URL}")
    print(f"  Runs:      {RUNS}  (+ {WARMUP} warmup)")
    print(sep)
    header = f"  {'Framework':<18} {'mean':>8} {'median':>8} {'p95':>8} {'p99':>8} {'min':>8} {'max':>8} {'stdev':>8}"
    if baseline_ms:
        header += f"  {'overhead':>10}"
    print(header)
    print(f"  {'':─<18} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8}")
    for r in results:
        row = (
            f"  {r['name']:<18} "
            f"{r['mean_ms']:>8.1f} "
            f"{r['median_ms']:>8.1f} "
            f"{r['p95_ms']:>8.1f} "
            f"{r['p99_ms']:>8.1f} "
            f"{r['min_ms']:>8.1f} "
            f"{r['max_ms']:>8.1f} "
            f"{r['stdev_ms']:>8.1f}"
        )
        if baseline_ms and r["name"] != "Raw (baseline)":
            overhead = r["mean_ms"] - baseline_ms
            sign = "+" if overhead >= 0 else ""
            row += f"  {sign}{overhead:>8.1f}ms"
        print(row)
    print(sep)
    print()
    for r in results:
        print(f"  [{r['name']}] sample output:")
        for line in (r["sample_output"] or "").strip().splitlines():
            print(f"    {line}")
        print()


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="JamJet vs LangGraph latency benchmark")
    parser.add_argument("--json", metavar="FILE", help="Write full results to a JSON file")
    parser.add_argument("--skip-langgraph", action="store_true", help="Skip LangGraph benchmark")
    parser.add_argument("--skip-jamjet", action="store_true", help="Skip JamJet benchmark")
    args = parser.parse_args()

    print(f"\nModel: {MODEL_NAME}  |  Endpoint: {OPENAI_BASE_URL}")
    print(f"Question: {QUESTION}\n")

    runners: list[tuple[str, Callable[[str], str] | None]] = [("Raw (baseline)", None)]

    if not args.skip_jamjet:
        try:
            runners.append(("JamJet 0.1.1", make_jamjet_runner()))
        except Exception as e:
            print(f"[JamJet setup failed] {e}", file=sys.stderr)

    if not args.skip_langgraph:
        try:
            runners.append(("LangGraph", make_langgraph_runner()))
        except Exception as e:
            print(f"[LangGraph setup failed] {e}", file=sys.stderr)

    # Always build raw runner last so we don't double-count its setup time
    runners[0] = ("Raw (baseline)", make_raw_runner())

    all_results: list[dict[str, Any]] = []
    for name, fn in runners:
        if fn is None:
            continue
        print(f"Running {name} ({WARMUP} warmup + {RUNS} timed)...", flush=True)
        r = run_benchmark(name, fn)
        all_results.append(r)

    baseline_ms = next((r["mean_ms"] for r in all_results if r["name"] == "Raw (baseline)"), None)
    print_results(all_results, baseline_ms=baseline_ms)

    if args.json:
        import datetime

        payload = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "question": QUESTION,
            "results": all_results,
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Results written to {args.json}")


if __name__ == "__main__":
    main()
