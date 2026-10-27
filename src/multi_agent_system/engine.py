from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from pydantic import ValidationError

from .agents import ResearcherAgent, SummarizerAgent
from .bus import AGENT_NAMES, AsyncMessageBus
from .config import Settings
from .llm import AgentProvider, DeterministicProvider, GroqProvider
from .models import (
    AgentReport,
    FinalAnswer,
    TaskMessage,
    TaskProgress,
    TaskResult,
    WorkflowPlan,
    WorkflowResult,
    WorkflowState,
    utc_now,
)
from .state import StateStore, append_event, build_state_store


class WorkflowExecutionError(RuntimeError):
    pass


class MultiAgentEngine:
    """Coordinates an orchestrator, two worker agents, an async bus, and shared state."""

    def __init__(
        self,
        provider: AgentProvider | None = None,
        state_store: StateStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.settings.validate()
        self.provider = provider or GroqProvider(
            api_key=self.settings.groq_api_key,
            model=self.settings.model,
        )
        self.state_store = state_store or build_state_store(
            self.settings.state_backend, self.settings.redis_url
        )
        self.bus = AsyncMessageBus(maxsize=self.settings.queue_maxsize)
        self.agents = {
            "researcher": ResearcherAgent(self.provider),
            "summarizer": SummarizerAgent(self.provider),
        }
        self._workers: list[asyncio.Task[None]] = []
        self._result_router: asyncio.Task[None] | None = None
        self._started = False
        self._closed = False
        self._waiters: dict[str, asyncio.Queue[TaskResult]] = {}
        self._lifecycle_lock = asyncio.Lock()

    async def _start_workers(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("Engine is closed.")
            self._workers = [asyncio.create_task(self._worker_loop(name), name=f"worker-{name}") for name in AGENT_NAMES]
            self._result_router = asyncio.create_task(self._route_results(), name="result-router")
            self._started = True

    async def _route_results(self) -> None:
        while True:
            result = await self.bus.next_result()
            target = self._waiters.get(result.workflow_id)
            if target is not None:
                await target.put(result)

    async def _record_event(
        self,
        workflow_id: str,
        event_type: str,
        message: str,
        *,
        task_id: str | None = None,
        agent_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        def update(state: WorkflowState) -> None:
            append_event(
                state,
                event_type,
                message,
                task_id=task_id,
                agent_name=agent_name,
                details=details or {},
            )
        await self.state_store.mutate(workflow_id, update)

    async def _mark_task(
        self,
        workflow_id: str,
        task_id: str,
        *,
        status: str,
        attempt: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        output: AgentReport | None = None,
    ) -> None:
        def update(state: WorkflowState) -> None:
            task = state.tasks.get(task_id)
            if task is None:
                raise KeyError(f"Unknown task: {task_id}")
            task.status = status  # Pydantic validates the state transition label.
            if attempt is not None:
                task.attempts = attempt
            if error_type is not None:
                task.error_type = error_type
            if error_message is not None:
                task.error_message = error_message
            if started_at is not None and task.started_at is None:
                task.started_at = started_at
            if completed_at is not None:
                task.completed_at = completed_at
            if output is not None:
                state.outputs[task.agent_name] = output
                task.error_type = None
                task.error_message = None
        await self.state_store.mutate(workflow_id, update)

    async def _worker_loop(self, agent_name: str) -> None:
        agent = self.agents[agent_name]
        while True:
            message = await self.bus.receive(agent_name)
            if message is None:
                self.bus.queues[agent_name].task_done()
                return
            started = utc_now()
            last_error: Exception | None = None
            output: AgentReport | None = None
            attempts_used = 0
            for attempt in range(1, self.settings.max_agent_retries + 2):
                attempts_used = attempt
                try:
                    await self._mark_task(
                        message.workflow_id,
                        message.task_id,
                        status="running" if attempt == 1 else "retrying",
                        attempt=attempt,
                        started_at=started,
                    )
                    await self._record_event(
                        message.workflow_id,
                        "agent_started" if attempt == 1 else "agent_retry",
                        f"{agent_name} started task attempt {attempt}.",
                        task_id=message.task_id,
                        agent_name=agent_name,
                        details={"attempt": attempt},
                    )
                    output = await asyncio.wait_for(
                        agent.execute(message),
                        timeout=self.settings.agent_timeout_seconds,
                    )
                    await self._mark_task(
                        message.workflow_id,
                        message.task_id,
                        status="succeeded",
                        attempt=attempt,
                        completed_at=utc_now(),
                        output=output,
                    )
                    await self._record_event(
                        message.workflow_id,
                        "agent_completed",
                        f"{agent_name} completed successfully.",
                        task_id=message.task_id,
                        agent_name=agent_name,
                        details={"attempt": attempt, "confidence": output.confidence},
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    await self._mark_task(
                        message.workflow_id,
                        message.task_id,
                        status="retrying" if attempt <= self.settings.max_agent_retries else "failed",
                        attempt=attempt,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:500],
                        completed_at=utc_now() if attempt > self.settings.max_agent_retries else None,
                    )
                    await self._record_event(
                        message.workflow_id,
                        "agent_attempt_failed",
                        f"{agent_name} attempt {attempt} failed.",
                        task_id=message.task_id,
                        agent_name=agent_name,
                        details={"error_type": type(exc).__name__, "error": str(exc)[:300]},
                    )
                    if attempt <= self.settings.max_agent_retries:
                        await asyncio.sleep(self.settings.retry_base_delay_seconds * (2 ** (attempt - 1)))

            if last_error is not None:
                await self.state_store.mutate(
                    message.workflow_id,
                    lambda state: state.errors.append({
                        "task_id": message.task_id,
                        "agent_name": agent_name,
                        "error_type": type(last_error).__name__,
                        "message": str(last_error)[:500],
                        "attempts": attempts_used,
                    }),
                )
                result = TaskResult(
                    task_id=message.task_id,
                    workflow_id=message.workflow_id,
                    agent_name=agent_name,
                    success=False,
                    error_type=type(last_error).__name__,
                    error_message=str(last_error)[:500],
                    attempts=attempts_used,
                    started_at=started,
                    completed_at=utc_now(),
                )
            else:
                result = TaskResult(
                    task_id=message.task_id,
                    workflow_id=message.workflow_id,
                    agent_name=agent_name,
                    success=True,
                    output=output,
                    attempts=attempts_used,
                    started_at=started,
                    completed_at=utc_now(),
                )
            await self.bus.publish_result(result)
            self.bus.queues[agent_name].task_done()

    async def _validated_plan(self, query: str, workflow_id: str) -> WorkflowPlan:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.structured_output_retries + 2):
            try:
                raw = await self.provider.plan(query)
                plan = WorkflowPlan.model_validate(raw)
                if not plan.research_task or not plan.summarizer_task:
                    raise ValueError("The plan must assign work to both agents.")
                return plan
            except Exception as exc:
                last_error = exc
                await self._record_event(
                    workflow_id,
                    "plan_validation_failed",
                    f"Orchestrator plan attempt {attempt} failed validation or execution.",
                    details={"error_type": type(exc).__name__, "error": str(exc)[:300]},
                )
                if attempt <= self.settings.structured_output_retries:
                    await asyncio.sleep(self.settings.retry_base_delay_seconds * (2 ** (attempt - 1)))
        raise WorkflowExecutionError(
            f"Orchestrator could not produce a valid plan after retries: {type(last_error).__name__}: {last_error}"
        ) from last_error

    async def run(self, user_query: str) -> WorkflowResult:
        query = user_query.strip()
        if not query:
            raise ValueError("user_query cannot be empty")
        if self._closed:
            raise RuntimeError("Engine is closed.")

        started_wall = time.perf_counter()
        workflow_id = str(uuid4())
        initial_state = WorkflowState(workflow_id=workflow_id, user_query=query)
        await self.state_store.create(initial_state)
        await self._record_event(workflow_id, "workflow_started", "Workflow accepted by orchestrator.")
        await self._start_workers()
        result_waiter: asyncio.Queue[TaskResult] = asyncio.Queue()
        self._waiters[workflow_id] = result_waiter

        try:
            plan = await asyncio.wait_for(
                self._validated_plan(query, workflow_id),
                timeout=self.settings.workflow_timeout_seconds,
            )
            def add_plan(state: WorkflowState) -> None:
                state.plan = plan
                state.status = "running"
                append_event(state, "plan_created", "Orchestrator created a validated two-agent plan.", details={"goal": plan.goal})
            await self.state_store.mutate(workflow_id, add_plan)

            task_specs = [
                ("researcher", plan.research_task),
                ("summarizer", plan.summarizer_task),
            ]
            messages: list[TaskMessage] = []
            for agent_name, instruction in task_specs:
                message = TaskMessage(
                    workflow_id=workflow_id,
                    agent_name=agent_name,
                    user_query=query,
                    instruction=instruction,
                    # Each worker receives only assigned context, not another agent's private prompt/history.
                    context={"goal": plan.goal, "scope_notes": plan.scope_notes},
                )
                messages.append(message)
                def add_task(state: WorkflowState, m: TaskMessage = message) -> None:
                    state.tasks[m.task_id] = TaskProgress(
                        task_id=m.task_id,
                        agent_name=m.agent_name,
                        instruction=m.instruction,
                        status="queued",
                    )
                    append_event(state, "task_queued", f"Task queued for {m.agent_name}.", task_id=m.task_id, agent_name=m.agent_name)
                await self.state_store.mutate(workflow_id, add_task)
                await self.bus.publish(message)

            results: dict[str, TaskResult] = {}
            deadline = time.monotonic() + self.settings.workflow_timeout_seconds
            while len(results) < len(messages):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(result_waiter.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                results[item.agent_name] = item

            # A timed-out workflow can still synthesize from any completed worker output.
            expected_agents = {name for name, _ in task_specs}
            missing = expected_agents - set(results)
            if missing:
                await self.state_store.mutate(
                    workflow_id,
                    lambda state: [state.errors.append({
                        "agent_name": name,
                        "error_type": "WorkflowTimeout",
                        "message": "No result received before the workflow deadline.",
                    }) for name in sorted(missing)],
                )
                await self._record_event(
                    workflow_id,
                    "workflow_timeout",
                    "Workflow deadline expired while waiting for agent results.",
                    details={"missing_agents": sorted(missing)},
                )

            snapshot = await self.state_store.get(workflow_id)
            reports = dict(snapshot.outputs)
            errors = list(snapshot.errors)
            final_raw: dict[str, Any] | None = None
            final_error: Exception | None = None
            for attempt in range(1, self.settings.structured_output_retries + 2):
                try:
                    final_raw = await self.provider.synthesize(query, reports, errors)
                    final_answer = FinalAnswer.model_validate(final_raw)
                    break
                except Exception as exc:
                    final_error = exc
                    await self._record_event(
                        workflow_id,
                        "synthesis_attempt_failed",
                        f"Final synthesis attempt {attempt} failed.",
                        details={"error_type": type(exc).__name__, "error": str(exc)[:300]},
                    )
                    if attempt <= self.settings.structured_output_retries:
                        await asyncio.sleep(self.settings.retry_base_delay_seconds * (2 ** (attempt - 1)))
            else:
                final_answer = FinalAnswer(
                    answer=("The workflow finished without a valid synthesized answer. "
                            f"Synthesis error: {type(final_error).__name__}: {final_error}"),
                    key_takeaways=[f"Successful agents: {', '.join(sorted(reports)) or 'none'}"],
                    limitations=["Final answer synthesis failed schema validation or provider execution."],
                    completion_status="partial" if reports else "failed",
                )

            if final_answer.completion_status == "complete" and (errors or len(reports) < 2):
                final_answer.completion_status = "partial"

            def finish(state: WorkflowState) -> None:
                state.final_answer = final_answer
                state.status = final_answer.completion_status
                append_event(
                    state,
                    "workflow_finished",
                    f"Workflow finished with status {final_answer.completion_status}.",
                    details={"successful_agents": sorted(reports), "error_count": len(errors)},
                )
            finished = await self.state_store.mutate(workflow_id, finish)

            return WorkflowResult(
                workflow_id=workflow_id,
                status=final_answer.completion_status,
                final_answer=final_answer.answer,
                key_takeaways=final_answer.key_takeaways,
                agent_outputs=reports,
                errors=errors,
                duration_ms=round((time.perf_counter() - started_wall) * 1000, 2),
                state_version=finished.version,
            )
        except Exception as exc:
            await self.state_store.mutate(
                workflow_id,
                lambda state: (
                    setattr(state, "status", "failed"),
                    state.errors.append({"error_type": type(exc).__name__, "message": str(exc)[:500]}),
                    append_event(state, "workflow_failed", "Workflow failed before normal synthesis.", details={"error_type": type(exc).__name__}),
                ),
            )
            if isinstance(exc, WorkflowExecutionError):
                raise
            raise WorkflowExecutionError(f"Workflow failed: {type(exc).__name__}: {exc}") from exc
        finally:
            self._waiters.pop(workflow_id, None)

    async def get_state(self, workflow_id: str) -> WorkflowState:
        return await self.state_store.get(workflow_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            # Sentinel messages gracefully stop each worker after pending queue items.
            await self.bus.stop_workers()
            await asyncio.gather(*self._workers, return_exceptions=True)
            if self._result_router is not None:
                self._result_router.cancel()
                await asyncio.gather(self._result_router, return_exceptions=True)
        await self.provider.close()
        await self.state_store.close()

    async def __aenter__(self) -> "MultiAgentEngine":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()
