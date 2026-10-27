import asyncio

import pytest

from multi_agent_system.config import Settings
from multi_agent_system.engine import MultiAgentEngine
from multi_agent_system.llm import DeterministicProvider
from multi_agent_system.state import InMemoryStateStore


class MalformedOnceProvider(DeterministicProvider):
    def __init__(self):
        super().__init__()
        self.bad_once = True

    async def run_agent(self, agent_name, user_query, instruction, context):
        if agent_name == "researcher" and self.bad_once:
            self.bad_once = False
            self.agent_calls[agent_name] += 1
            return {"agent_name": "researcher", "summary": "bad output"}
        return await super().run_agent(agent_name, user_query, instruction, context)


class InvalidPlanOnceProvider(DeterministicProvider):
    def __init__(self):
        super().__init__()
        self.first = True

    async def plan(self, user_query):
        if self.first:
            self.first = False
            return {"goal": "missing task assignments"}
        return await super().plan(user_query)


class ResearcherTimeoutProvider(DeterministicProvider):
    async def run_agent(self, agent_name, user_query, instruction, context):
        if agent_name == "researcher":
            await asyncio.sleep(0.08)
        return await super().run_agent(agent_name, user_query, instruction, context)


@pytest.mark.asyncio
async def test_malformed_agent_output_is_retried_and_recovered():
    provider = MalformedOnceProvider()
    engine = MultiAgentEngine(
        provider=provider,
        settings=Settings(max_agent_retries=1, retry_base_delay_seconds=0, workflow_timeout_seconds=2),
    )
    try:
        result = await engine.run("Review the reliability of async workers.")
        state = await engine.get_state(result.workflow_id)
        assert result.status == "complete"
        assert provider.agent_calls["researcher"] == 2
        assert any(event.event_type == "agent_attempt_failed" for event in state.events)
        assert state.tasks[next(k for k, v in state.tasks.items() if v.agent_name == "researcher")].attempts == 2
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_invalid_plan_is_retried_before_dispatch():
    provider = InvalidPlanOnceProvider()
    engine = MultiAgentEngine(
        provider=provider,
        settings=Settings(structured_output_retries=1, retry_base_delay_seconds=0),
    )
    try:
        result = await engine.run("Analyze a queue-based workflow.")
        state = await engine.get_state(result.workflow_id)
        assert result.status == "complete"
        assert state.plan is not None
        assert any(event.event_type == "plan_validation_failed" for event in state.events)
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_agent_timeout_is_recorded_and_other_agent_can_complete():
    provider = ResearcherTimeoutProvider()
    engine = MultiAgentEngine(
        provider=provider,
        settings=Settings(
            max_agent_retries=0,
            agent_timeout_seconds=0.015,
            workflow_timeout_seconds=0.5,
            retry_base_delay_seconds=0,
        ),
    )
    try:
        result = await engine.run("Outline a system design for async workflow coordination.")
        assert result.status == "partial"
        assert "summarizer" in result.agent_outputs
        assert any(error["agent_name"] == "researcher" for error in result.errors)
    finally:
        await engine.close()
