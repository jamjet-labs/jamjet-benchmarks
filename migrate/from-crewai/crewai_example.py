"""
CrewAI: multi-agent crew with roles, goals, and task delegation.

A two-agent crew: one researcher and one writer that collaborate
to produce a research report.
"""

from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool

search_tool = SerperDevTool()

# ── Agents ─────────────────────────────────────────────────────────────────────

researcher = Agent(
    role="Research Analyst",
    goal="Find comprehensive, accurate information on the given topic",
    backstory=(
        "You are an expert research analyst with deep knowledge across many domains. "
        "You excel at finding and synthesizing information from multiple sources."
    ),
    tools=[search_tool],
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear, concise reports from research findings",
    backstory=(
        "You are a skilled technical writer who can transform complex research "
        "into clear, readable reports for a technical audience."
    ),
    verbose=True,
)

# ── Tasks ──────────────────────────────────────────────────────────────────────

research_task = Task(
    description="Research the topic: {topic}. Find key facts, recent developments, and expert opinions.",
    expected_output="A structured summary of findings with bullet points and sources.",
    agent=researcher,
)

write_task = Task(
    description="Using the research findings, write a comprehensive report on: {topic}",
    expected_output="A well-structured report with introduction, key findings, and conclusion.",
    agent=writer,
    context=[research_task],
)

# ── Crew ───────────────────────────────────────────────────────────────────────

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
)

# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = crew.kickoff(inputs={"topic": "durable AI workflow orchestration"})
    print(result)
