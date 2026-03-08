# jamjet-benchmarks

Benchmarks, migration guides, and feature comparisons for [JamJet](https://github.com/jamjet-labs/jamjet) vs other AI agent frameworks.

## Contents

| Directory | What's inside |
|---|---|
| [`benchmarks/`](./benchmarks/) | Latency benchmarks: JamJet vs LangGraph vs raw LLM call |
| [`migrate/from-langgraph/`](./migrate/from-langgraph/) | Side-by-side code + concept mapping |
| [`migrate/from-crewai/`](./migrate/from-crewai/) | Side-by-side code + concept mapping |
| [`migrate/from-openai-direct/`](./migrate/from-openai-direct/) | Side-by-side code + concept mapping |
| [`feature-matrix.md`](./feature-matrix.md) | Full feature comparison across 4 frameworks |

## Quick benchmark

Works with any OpenAI-compatible endpoint. Tested with Ollama locally.

```bash
cd benchmarks
pip install -r requirements.txt

# Local (Ollama)
export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export MODEL_NAME="qwen3:8b"
python bench_single_call.py --json results/$(date +%Y-%m-%d)-ollama-qwen3-8b.json

# OpenAI
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4.1-mini"
python bench_single_call.py --json results/$(date +%Y-%m-%d)-openai-gpt41mini.json
```

## What we measure

The benchmark measures **framework orchestration overhead** on top of a raw LLM call. All three runners (Raw, JamJet, LangGraph) make the identical LLM call through the same OpenAI-compatible client. The difference in latency is purely the framework tax.

Metrics per framework: mean, median, p95, p99, min, max, stdev (all in ms), overhead vs raw baseline.

## Latest results

See [`benchmarks/results/`](./benchmarks/results/) for recorded runs.

## Migration guides

Each guide has:
- The original framework code (unchanged, idiomatic)
- The JamJet equivalent
- A concept-mapping table
- Key differences explained

## Contributing

Results from different hardware, models, and endpoints are welcome. Open a PR with your `results/YYYY-MM-DD-<model>-<notes>.json`.

Corrections to the feature matrix are welcome too — if we marked something wrong, open an issue or PR.
