import asyncio
from collections import Counter

import pytest
from pydantic import ValidationError

from multi_agent_system.bus import AsyncMessageBus
from multi_agent_system.config import Settings
from multi_agent_system.engine import MultiAgentEngine
from multi_agent_system.llm import DeterministicProvider
from multi_agent_system.models import AgentReport, TaskMessage, WorkflowState
from multi_agent_system.state import InMemoryStateStore


def test_agent_report_rejects_malformed_output():
    with pytest.raises(ValidationError):
        AgentReport.model_validate({"agent_name": "researcher", "summary": "missing required fields"})


@pytest.mark.asyncio
async def test_happy_path_uses_both_agents_and_shared_state():
    provider = DeterministicProvider(delay_seconds=0.01)
    store = InMemoryStateStore()
    engine = MultiAgentEngine(
        provider=provider,
        state_store=store,
        settings=Settings(max_agent_retries=1, retry_base_delay_seconds=0, workflow_timeout_seconds=2),
    )
    try:
        result = await engine.run("Compare EV market trends and regulatory considerations.")
        state = await engine.get_state(result.workflow_id)
        assert result.status == "complete"
        assert set(result.agent_outputs) == {"researcher", "summarizer"}
        assert state.status == "complete"
        assert all(task.status == "succeeded" for task in state.tasks.values())
        assert state.plan is not None
        assert state.final_answer is not None
        assert any(event.event_type == "plan_created" for event in state.events)
        assert any(event.event_type == "agent_completed" for event in state.events)
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_agents_execute_concurrently():
    provider = DeterministicProvider(delay_seconds=0.06)
    engine = MultiAgentEngine(provider=provider, settings=Settings(retry_base_delay_seconds=0))
    try:
        await engine.run("Summarize asynchronous architecture patterns.")
        assert provider.max_concurrent_agents >= 2
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_queue_retains_message_until_worker_consumes_it():
    bus = AsyncMessageBus(maxsize=1)
    task = TaskMessage(
        workflow_id="wf-test",
        agent_name="researcher",
        user_query="Explain queues",
        instruction="Identify queue semantics",
    )
    await bus.publish(task)
    assert bus.queued_count("researcher") == 1
    received = await bus.receive("researcher")
    assert received is not None
    assert received.task_id == task.task_id
    bus.queues["researcher"].task_done()
    assert bus.queued_count("researcher") == 0


@pytest.mark.asyncio
async def test_bounded_queue_applies_backpressure_without_dropping_messages():
    bus = AsyncMessageBus(maxsize=1)
    first = TaskMessage(workflow_id="wf", agent_name="researcher", user_query="q", instruction="first")
    second = TaskMessage(workflow_id="wf", agent_name="researcher", user_query="q", instruction="second")
    await bus.publish(first)
    blocked_publish = asyncio.create_task(bus.publish(second))
    await asyncio.sleep(0)
    assert not blocked_publish.done()
    taken_first = await bus.receive("researcher")
    bus.queues["researcher"].task_done()
    await asyncio.wait_for(blocked_publish, timeout=0.2)
    taken_second = await bus.receive("researcher")
    bus.queues["researcher"].task_done()
    assert taken_first and taken_first.task_id == first.task_id
    assert taken_second and taken_second.task_id == second.task_id


@pytest.mark.asyncio
async def test_concurrent_state_mutations_are_not_lost():
    store = InMemoryStateStore()
    state = WorkflowState(workflow_id="wf-atomic", user_query="test state mutations")
    await store.create(state)

    async def write(index: int) -> None:
        await store.mutate(
            "wf-atomic",
            lambda current: current.errors.append({"index": index}),
        )

    await asyncio.gather(*(write(i) for i in range(30)))
    final = await store.get("wf-atomic")
    assert len(final.errors) == 30
    assert final.version == 30
    assert {item["index"] for item in final.errors} == set(range(30))


@pytest.mark.asyncio
async def test_multiple_workflows_share_workers_without_crossing_results():
    provider = DeterministicProvider(delay_seconds=0.01)
    engine = MultiAgentEngine(provider=provider, settings=Settings(retry_base_delay_seconds=0, workflow_timeout_seconds=3))
    try:
        first, second = await asyncio.gather(
            engine.run("Explain backpressure in an asynchronous queue."),
            engine.run("Outline a safe retry strategy for worker failures."),
        )
        assert first.workflow_id != second.workflow_id
        assert "backpressure" in first.final_answer.lower()
        assert "retry strategy" in second.final_answer.lower()
        first_state = await engine.get_state(first.workflow_id)
        second_state = await engine.get_state(second.workflow_id)
        assert first_state.user_query != second_state.user_query
        assert set(first_state.outputs) == {"researcher", "summarizer"}
        assert set(second_state.outputs) == {"researcher", "summarizer"}
    finally:
        await engine.close()
