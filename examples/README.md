# JamJet Examples

Four self-contained examples that demonstrate what makes JamJet different — each with its own virtual environment, verified to run locally with Ollama.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally with at least one model pulled

```bash
# Pull a model (one-time)
ollama pull llama3.2      # fast, 2GB — recommended for these examples
ollama pull qwen3:8b      # more capable, 5GB, slower (thinking model)
```

## Setup (one-time, per example)

Each example is self-contained. Pick the one you want:

```bash
cd examples/01_pipeline_with_timeline   # or 02_, 03_, 04_

python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# From inside the example directory, with the venv active:

export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export MODEL_NAME="llama3.2"

python main.py
```

You can override any env var:

```bash
MODEL_NAME="qwen3:8b" QUESTION="What is the CAP theorem?" python main.py
```

---

## The examples

### [01 — Pipeline with execution timeline](./01_pipeline_with_timeline/)

3-step research pipeline: extract keywords → build outline → write answer.

**What it shows:** JamJet's per-step event log (`result.events`) gives you a full execution timeline with per-step timings — without any logging code. Other frameworks: add `print()` everywhere. JamJet: it's built in.

```
✓ extract_keywords        1935ms  ██████████████████████████
✓ build_outline           1031ms  █████████████
✓ write_answer            1491ms  ██████████████████
─────────────────────────────────────────
Total: 4458ms  (3 steps)
```

---

### [02 — Conditional routing](./02_conditional_routing/)

Classifies question type (factual / opinion / unclear) and routes to a specialist handler.

**What it shows:** Routing is a plain Python predicate on the Pydantic state — readable, testable without running the LLM, no separate edge function API to learn.

```
● → classify               167ms    (classified: "opinion")
●   answer_opinion        1934ms
─────────────────────────────────
Routing logic test (no LLM needed):
  ✓ 'factual'  → answer_factual
  ✓ 'opinion'  → answer_opinion
  ✓ 'unclear'  → ask_to_clarify
```

---

### [03 — Eval harness](./03_eval_harness/)

Runs a 5-row JSONL dataset through a Q&A workflow and scores every output with three scorers.

**What it shows:** JamJet ships a built-in eval harness — assertion scoring, latency budgets, and LLM-as-judge. No separate platform needed. Run it in CI with `--fail-under 0.9`.

```
Eval Results — 5/5 passed (100%)
assertion : 5/5 passed
latency   : 5/5 passed
llm_judge : 5/5 passed  (local Ollama used as judge — free)
```

---

### [04 — Self-evaluating workflow](./04_self_evaluating_workflow/)

Generates an answer, judges its own output, retries with specific feedback if score is below threshold.

**What it shows:** Eval-as-a-workflow-node — the same pattern as `type: eval` in JamJet YAML. The routing predicate `lambda s: s.judge_score < MIN_SCORE and s.attempts < MAX_RETRIES` controls the retry loop.

```
✓ draft               1200ms
✓ judge                810ms
✓ accept                 0ms
─────────────────────
Judge score : 5/5  — accepted on first attempt
```

---

## Works with any OpenAI-compatible endpoint

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4.1-mini"

# Together AI
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.together.xyz/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct-Turbo"

# Groq
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
```
