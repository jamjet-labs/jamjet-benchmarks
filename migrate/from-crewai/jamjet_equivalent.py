"""
JamJet equivalent of the CrewAI two-agent research + write crew.

Key differences:
  - No "agents" as opaque objects — workflows are explicit graphs you can read
  - Tool calls are MCP-native: any MCP server works (Brave, GitHub, Postgres, etc.)
  - Roles/goals live in the system prompt (where they belong — you control the prompt)
  - For multi-agent delegation, use A2A protocol: one workflow calls another via a2a_task node
  - Full observability: every step is an event you can inspect, replay, and diff
"""

from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel

from jamjet import Workflow

client = OpenAI()  # or configure for any OpenAI-compatible endpoint


# ── State ──────────────────────────────────────────────────────────────────────


class State(BaseModel):
    topic: str
    research: str = ""
    report: str = ""


# ── Workflow ───────────────────────────────────────────────────────────────────

wf = Workflow("research-crew")


@wf.state
class CrewState(State):
    pass


@wf.step
async def research(state: CrewState) -> CrewState:
    """
    Research agent: finds information on the topic.

    In production: replace the direct LLM call with an MCP tool node
    (e.g. Brave Search, Exa, Tavily) — just change type to 'tool' in workflow.yaml.
    """
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert research analyst. "
                    "Find key facts, recent developments, and expert opinions. "
                    "Return a structured summary with bullet points."
                ),
            },
            {"role": "user", "content": f"Research this topic: {state.topic}"},
        ],
    )
    findings = resp.choices[0].message.content or ""
    return state.model_copy(update={"research": findings})


@wf.step
async def write_report(state: CrewState) -> CrewState:
    """
    Writer agent: synthesizes research into a structured report.
    """
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a skilled technical writer. "
                    "Transform research findings into a clear, readable report "
                    "with introduction, key findings, and conclusion."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {state.topic}\n\n"
                    f"Research findings:\n{state.research}\n\n"
                    "Write a comprehensive report."
                ),
            },
        ],
    )
    report = resp.choices[0].message.content or ""
    return state.model_copy(update={"report": report})


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = wf.run_sync(CrewState(topic="durable AI workflow orchestration"))
    print(result.state.report)

    # ── YAML equivalent (no Python needed) ────────────────────────────────────
    # workflow.yaml:
    #
    #   nodes:
    #     research:
    #       type: model
    #       model: gpt-4.1-mini
    #       system: "You are an expert research analyst..."
    #       prompt: "Research this topic: {{ state.topic }}"
    #       output_key: research
    #       next: write_report
    #
    #     write_report:
    #       type: model
    #       model: gpt-4.1-mini
    #       system: "You are a skilled technical writer..."
    #       prompt: |
    #         Topic: {{ state.topic }}
    #         Research: {{ state.research }}
    #         Write a comprehensive report.
    #       output_key: report
    #       next: end
    #
    #     end:
    #       type: end
