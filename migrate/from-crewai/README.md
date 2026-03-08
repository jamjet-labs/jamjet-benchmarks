# Migrating from CrewAI to JamJet

## Concept mapping

| CrewAI | JamJet |
|---|---|
| `Agent(role, goal, backstory, tools)` | `@wf.step` with system prompt + MCP tools |
| `Task(description, agent, context)` | Workflow step — context is shared state |
| `Crew(agents, tasks, process)` | `Workflow` (sequential = steps in order; parallel = `type: parallel` node) |
| `crew.kickoff(inputs)` | `wf.run_sync(State(...))` or `jamjet run workflow.yaml` |
| `Process.sequential` | Default step order |
| `Process.hierarchical` | Orchestrator workflow calling sub-workflows via A2A |
| `crewai_tools.SerperDevTool` | MCP tool node (`type: tool`, `server: brave-search`) |
| Memory (short/long term) | State object (short term); Postgres-backed runtime (long term) |
| `agent.verbose = True` | Event log — every step emits structured events, queryable |

## Key differences

**Agents vs steps.** CrewAI treats agents as first-class objects with roles, goals, backstories, and tool access. JamJet treats these as what they actually are: a prompt (system message) and a set of tools. Your "agent" is a step with a well-crafted system prompt. The role/goal/backstory go in the system prompt — where they belong — and you have full control over them.

**Tools.** CrewAI has its own tool ecosystem (`crewai_tools`). JamJet uses MCP — the open standard. Any MCP server works: Brave Search, GitHub, Postgres, your own custom server. No lock-in to a proprietary tool registry.

**Multi-agent.** CrewAI's `Process.hierarchical` uses an internal manager agent. JamJet uses A2A (Agent-to-Agent protocol) — an open standard where each agent is a separate workflow with its own endpoint. An orchestrator workflow delegates to specialist workflows via `type: a2a_task` nodes. This scales across machines and organizations.

**Observability.** CrewAI's verbose logging is human-readable text. JamJet emits structured events for every step — queryable, diffable, replayable. `jamjet inspect <exec-id>` shows you exactly what happened.

**Durability.** CrewAI has no built-in persistence — if the process dies mid-run, you start over. JamJet's Rust runtime persists every step transition. Crash recovery is automatic.

## Quick-start migration

```bash
pip install jamjet
```

1. Convert each `Agent` into a `@wf.step` function
   - `role` + `goal` + `backstory` → `system` message in the LLM call
   - `tools` → MCP tool nodes (`type: tool` in YAML)
2. Convert `Task` ordering into step declaration order
   - `context=[other_task]` → shared `State` fields (results from prior steps)
3. Replace `crew.kickoff(inputs)` with `wf.run_sync(State(...))`
4. For multi-agent (`Process.hierarchical`): use `type: a2a_task` nodes to delegate

See [`crewai_example.py`](./crewai_example.py) and [`jamjet_equivalent.py`](./jamjet_equivalent.py) for a side-by-side comparison.
