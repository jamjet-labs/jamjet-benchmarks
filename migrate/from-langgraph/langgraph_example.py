"""
LangGraph: multi-step agent with routing.

A research agent that:
  1. Decides whether to search or answer directly (router)
  2. Searches the web if needed
  3. Synthesizes a final answer
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


# ── State ──────────────────────────────────────────────────────────────────────


class State(TypedDict):
    question: str
    needs_search: bool
    search_results: list[str]
    answer: str


# ── Nodes ──────────────────────────────────────────────────────────────────────


def route(state: State) -> State:
    """Decide whether the question needs a web search."""
    q = state["question"].lower()
    needs_search = any(w in q for w in ["latest", "current", "today", "news", "2024", "2025"])
    return {**state, "needs_search": needs_search}


def search(state: State) -> State:
    """Simulate a web search (replace with real MCP tool call)."""
    return {**state, "search_results": [f"[result for: {state['question']}]"]}


def answer(state: State) -> State:
    """Synthesize a final answer from state."""
    context = "\n".join(state.get("search_results", []))
    # In practice: call your LLM here with context
    final = f"Answer to '{state['question']}'"
    if context:
        final += f"\nContext used: {context}"
    return {**state, "answer": final}


def should_search(state: State) -> Literal["search", "answer"]:
    return "search" if state["needs_search"] else "answer"


# ── Graph ──────────────────────────────────────────────────────────────────────


graph = StateGraph(State)
graph.add_node("route", route)
graph.add_node("search", search)
graph.add_node("answer", answer)

graph.add_edge(START, "route")
graph.add_conditional_edges("route", should_search)
graph.add_edge("search", "answer")
graph.add_edge("answer", END)

app = graph.compile()

# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = app.invoke({"question": "What is durable workflow orchestration?", "needs_search": False, "search_results": [], "answer": ""})
    print(result["answer"])
