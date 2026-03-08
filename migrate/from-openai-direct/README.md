# Migrating from Raw OpenAI SDK to JamJet

## Why migrate

The raw OpenAI SDK agentic loop works great for demos and prototypes. In production you eventually build:

- Manual retry logic with exponential backoff
- State threading between tool calls and model calls
- Logging and tracing (what did step 7 actually receive?)
- Restart logic when your process crashes mid-run
- Tool dispatch tables that grow as you add tools

JamJet handles all of this as infrastructure, not application code.

## Concept mapping

| Raw OpenAI | JamJet |
|---|---|
| `messages` list | `State` (Pydantic model, typed, validated) |
| `while True:` agentic loop | Workflow graph — explicit, inspectable |
| Manual `tool_calls` dispatch | MCP tool nodes (`type: tool`) |
| `client.chat.completions.create(...)` | `type: model` node (or `@wf.step` calling the client directly) |
| Hand-rolled retry | `retry: max_attempts: 3, backoff: exponential` in YAML |
| `print()` debugging | `jamjet inspect <exec-id>` — full structured event timeline |
| Process restart on crash | Durable runtime — resume from last completed step |
| Nothing | `jamjet eval run` — regression test your agent on every commit |

## Migration path

**Step 1 — lift your state into a Pydantic model.**

```python
# Before
messages = [{"role": "user", "content": question}]
search_results = None
final_answer = None

# After
class State(BaseModel):
    question: str
    search_results: str = ""
    answer: str = ""
```

**Step 2 — split your loop into named steps.**

Each logical "phase" of your loop becomes a `@wf.step`. Tool dispatch becomes a tool node.

**Step 3 — keep your LLM calls as-is.**

You don't have to change how you call the model. Use the OpenAI client inside your step function exactly as before. Swap to a YAML `type: model` node when you want the runtime to handle it.

**Step 4 — run locally first.**

`wf.run_sync(State(...))` works without any server. Exact same behavior as your loop.

**Step 5 — go durable when you need it.**

`wf.compile()` + `jamjet dev` → your workflow is now crash-safe, observable, and testable with `jamjet eval run`.

See [`openai_example.py`](./openai_example.py) and [`jamjet_equivalent.py`](./jamjet_equivalent.py) for a side-by-side comparison.
