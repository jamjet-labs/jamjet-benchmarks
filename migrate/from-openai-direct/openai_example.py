"""
Raw OpenAI SDK: multi-step agent with tool use (function calling).

A common pattern before adopting a framework — direct API calls,
manual state threading, manual tool dispatch.
"""

import json

from openai import OpenAI

client = OpenAI()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    }
]


def web_search(query: str) -> str:
    """Simulated search — replace with real API call."""
    return f"[search results for: {query}]"


def dispatch_tool(name: str, args: dict) -> str:
    if name == "web_search":
        return web_search(args["query"])
    raise ValueError(f"Unknown tool: {name}")


def run_agent(question: str) -> str:
    """
    A basic agentic loop:
    1. Send question to model
    2. If model calls a tool, dispatch it and continue
    3. Repeat until model returns a final text response
    """
    messages = [
        {"role": "system", "content": "You are a helpful research assistant. Use tools when needed."},
        {"role": "user", "content": question},
    ]

    while True:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = dispatch_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            return msg.content or ""


if __name__ == "__main__":
    answer = run_agent("What are the latest developments in AI agent frameworks?")
    print(answer)
