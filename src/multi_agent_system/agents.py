from __future__ import annotations

from .llm import AgentProvider
from .models import AgentReport, TaskMessage


class ResearcherAgent:
    """Specialist for collecting structured facts, approaches, and trade-offs."""

    name = "researcher"

    def __init__(self, provider: AgentProvider) -> None:
        self.provider = provider

    async def execute(self, task: TaskMessage) -> AgentReport:
        raw = await self.provider.run_agent(
            agent_name=self.name,
            user_query=task.user_query,
            instruction=task.instruction,
            context=task.context,
        )
        report = AgentReport.model_validate(raw)
        if report.agent_name != self.name:
            raise ValueError(f"Researcher returned mismatched agent_name: {report.agent_name}")
        return report


class SummarizerAgent:
    """Specialist for themes, constraints, risks, and concise outlines."""

    name = "summarizer"

    def __init__(self, provider: AgentProvider) -> None:
        self.provider = provider

    async def execute(self, task: TaskMessage) -> AgentReport:
        raw = await self.provider.run_agent(
            agent_name=self.name,
            user_query=task.user_query,
            instruction=task.instruction,
            context=task.context,
        )
        report = AgentReport.model_validate(raw)
        if report.agent_name != self.name:
            raise ValueError(f"Summarizer returned mismatched agent_name: {report.agent_name}")
        return report
