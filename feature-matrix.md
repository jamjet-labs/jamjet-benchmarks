# Feature Matrix: JamJet vs LangGraph vs CrewAI vs AutoGen

> Last updated: 2026-03-08 | JamJet v0.1.1

Legend: ✅ Built-in | 🔧 Via plugin/extension | ⚠️ Partial | ❌ Not supported | 🚧 In progress

## Core execution

| Feature | JamJet | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| Graph-based workflow | ✅ | ✅ | ⚠️ Sequential/hierarchical | ✅ |
| Async execution | ✅ | ✅ | ✅ | ✅ |
| Local in-process runner | ✅ | ✅ | ✅ | ✅ |
| Typed state | ✅ Pydantic | ⚠️ TypedDict | ❌ Dict | ⚠️ Dict |
| State validation | ✅ Every step | ❌ | ❌ | ❌ |
| Conditional routing | ✅ Inline predicates | ✅ Edge functions | ⚠️ Process type | ✅ |
| Parallel branches | ✅ `type: parallel` | ✅ | ❌ | ✅ |
| Cycle / loop support | ✅ | ✅ | ⚠️ | ✅ |

## Durability & reliability

| Feature | JamJet | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| Durable execution (crash recovery) | ✅ Rust runtime | 🔧 Checkpointers | ❌ | ❌ |
| Event sourcing | ✅ Native | ❌ | ❌ | ❌ |
| Automatic retry with backoff | ✅ YAML config | 🔧 Manual | 🔧 Manual | 🔧 Manual |
| Human-in-the-loop / pause | ✅ `type: wait` | ✅ `interrupt_before` | ❌ | ⚠️ |
| Resume from checkpoint | ✅ Any step | 🔧 Requires saver | ❌ | ❌ |
| Timeout per step | ✅ | ⚠️ | ❌ | ⚠️ |

## Observability

| Feature | JamJet | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| Structured event log | ✅ Per-step events | ⚠️ Callbacks | ⚠️ verbose logs | ⚠️ |
| Execution inspection CLI | ✅ `jamjet inspect` | ❌ | ❌ | ❌ |
| Event timeline | ✅ | ❌ | ❌ | ❌ |
| OpenTelemetry tracing | 🚧 | 🔧 LangSmith | 🔧 | ❌ |
| Time-travel debugging | 🚧 | ❌ | ❌ | ❌ |

## Tool & protocol integration

| Feature | JamJet | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| MCP (Model Context Protocol) | ✅ Native client | 🔧 Via adapter | 🔧 Via adapter | 🔧 Via adapter |
| MCP server (expose tools) | 🚧 | ❌ | ❌ | ❌ |
| A2A (Agent-to-Agent) | ✅ Client + server | ❌ | ❌ | ❌ |
| OpenAI function calling | ✅ | ✅ | ✅ | ✅ |
| Custom Python tools | ✅ `@tool` decorator | ✅ | ✅ | ✅ |
| Tool retry on error | ✅ Node-level config | 🔧 Manual | 🔧 Manual | 🔧 Manual |

## Eval & testing

| Feature | JamJet | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| Built-in eval harness | ✅ | ❌ | ❌ | ❌ |
| LLM-as-judge scoring | ✅ `LlmJudgeScorer` | ❌ | ❌ | ❌ |
| Assertion scoring | ✅ `AssertionScorer` | ❌ | ❌ | ❌ |
| Latency budgets | ✅ `LatencyScorer` | ❌ | ❌ | ❌ |
| Cost budgets | ✅ `CostScorer` | ❌ | ❌ | ❌ |
| JSONL dataset format | ✅ | ❌ | ❌ | ❌ |
| CI regression via exit code | ✅ `--fail-under` | ❌ | ❌ | ❌ |
| Eval as workflow node | ✅ `type: eval` | ❌ | ❌ | ❌ |

## Developer experience

| Feature | JamJet | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| YAML workflow authoring | ✅ | ❌ | ❌ | ❌ |
| Python decorator API | ✅ `@wf.step` | ✅ `@graph.node` | ✅ `@agent` | ✅ |
| Project templates | ✅ `jamjet init --template` | ❌ | ❌ | ❌ |
| Local dev server | ✅ `jamjet dev` | ❌ | ❌ | ❌ |
| Workflow validation | ✅ `jamjet validate` | ❌ | ❌ | ❌ |
| Multi-model support | ✅ Any OpenAI-compat | ✅ | ✅ | ✅ |
| Local model (Ollama) | ✅ | ✅ | ✅ | ✅ |

## Production & scale

| Feature | JamJet | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| Runtime language | Rust | Python | Python | Python |
| Polyglot SDK support | Python (TS 🚧) | Python, JS | Python | Python, .NET |
| Multi-tenant | 🚧 | ❌ | ❌ | ❌ |
| Kubernetes-ready | ✅ Stateless binary | 🔧 | 🔧 | 🔧 |
| Managed cloud offering | 🚧 | ✅ LangGraph Cloud | ❌ | ❌ |
| Open source | ✅ Apache-2.0 | ✅ MIT | ✅ MIT | ✅ CC-BY-4 |

---

*Corrections welcome — open a PR.*
