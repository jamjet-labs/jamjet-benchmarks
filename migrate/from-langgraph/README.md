# Migrating from LangGraph to JamJet

## Concept mapping

| LangGraph | JamJet |
|---|---|
| `TypedDict` state | `pydantic.BaseModel` state |
| `StateGraph` | `Workflow` |
| `graph.add_node("name", fn)` | `@workflow.step` decorator |
| `graph.add_conditional_edges(node, router_fn)` | `@workflow.step(next={"target": predicate})` |
| `graph.add_edge(A, B)` | Sequential by default; explicit `next=` for branches |
| `graph.compile()` | `workflow.compile()` → IR for runtime |
| `app.invoke(state)` | `workflow.run_sync(state)` (local) |
| `app.astream(state)` | `workflow.run(state)` (async, local) |
| Checkpointing via `MemorySaver` | Built into the Rust runtime — automatic |
| Human-in-the-loop via `interrupt_before` | `type: wait` node in YAML or `human_approval=True` in step |

## Key differences

**State validation.** LangGraph uses `TypedDict` — dict access with no validation. JamJet uses Pydantic — fields are validated at every step transition. If a step returns the wrong shape, you get an error immediately, not silent data corruption.

**Routing syntax.** LangGraph requires a separate routing function passed to `add_conditional_edges`. JamJet routing is inline on the step: `@workflow.step(next={"branch": lambda s: s.condition})`. For simple linear workflows, you write nothing — steps execute in declaration order.

**Durability.** LangGraph's checkpointing is opt-in and in-process (SQLite, Redis, Postgres adapters). JamJet's Rust runtime is durable by default — every step transition is an event-sourced write. Crash at step 7 of 12? Resume from step 7 without replaying 1–6.

**Production path.** With LangGraph you stay in Python. With JamJet you have two modes:
- **Local (in-process):** `workflow.run_sync(state)` — same as LangGraph, great for dev/test
- **Runtime (durable):** `workflow.compile()` → submit to `jamjet dev` → full durability, retries, event log, observability

## Quick-start migration

```bash
pip install jamjet
```

1. Replace `TypedDict` state with a `pydantic.BaseModel`
2. Replace `StateGraph` with `Workflow`
3. Replace `graph.add_node("name", fn)` + `graph.add_edge` with `@wf.step`
4. For conditional routing: `@wf.step(next={"target": lambda s: s.flag})`
5. Replace `app.invoke(state)` with `wf.run_sync(State(...))`
6. When ready for production: `wf.compile()` and run with `jamjet dev`

See [`langgraph_example.py`](./langgraph_example.py) and [`jamjet_equivalent.py`](./jamjet_equivalent.py) for a side-by-side comparison.
