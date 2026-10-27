from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Protocol

from pydantic import ValidationError

from .models import WorkflowState, WorkflowEvent, utc_now


Updater = Callable[[WorkflowState], None]


class StateStore(Protocol):
    async def create(self, state: WorkflowState) -> None: ...
    async def mutate(self, workflow_id: str, updater: Updater) -> WorkflowState: ...
    async def get(self, workflow_id: str) -> WorkflowState: ...
    async def close(self) -> None: ...


class InMemoryStateStore:
    """Atomic in-process shared state for a single application process."""

    def __init__(self) -> None:
        self._records: dict[str, WorkflowState] = {}
        self._lock = asyncio.Lock()

    async def create(self, state: WorkflowState) -> None:
        async with self._lock:
            if state.workflow_id in self._records:
                raise ValueError(f"Workflow already exists: {state.workflow_id}")
            self._records[state.workflow_id] = state.model_copy(deep=True)

    async def mutate(self, workflow_id: str, updater: Updater) -> WorkflowState:
        async with self._lock:
            current = self._records.get(workflow_id)
            if current is None:
                raise KeyError(f"Unknown workflow: {workflow_id}")
            updated = current.model_copy(deep=True)
            updater(updated)
            updated.version = current.version + 1
            updated.updated_at = utc_now()
            self._records[workflow_id] = updated
            return updated.model_copy(deep=True)

    async def get(self, workflow_id: str) -> WorkflowState:
        async with self._lock:
            current = self._records.get(workflow_id)
            if current is None:
                raise KeyError(f"Unknown workflow: {workflow_id}")
            return current.model_copy(deep=True)

    async def close(self) -> None:
        return None


class RedisStateStore:
    """Redis-backed JSON state with optimistic concurrency using WATCH/MULTI/EXEC."""

    def __init__(self, redis_url: str, key_prefix: str = "multi-agent:workflow:") -> None:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - covered when Redis is selected
            raise RuntimeError("Install the redis package to use STATE_BACKEND=redis") from exc
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        self._prefix = key_prefix

    def _key(self, workflow_id: str) -> str:
        return f"{self._prefix}{workflow_id}"

    async def create(self, state: WorkflowState) -> None:
        ok = await self._redis.set(self._key(state.workflow_id), state.model_dump_json(), nx=True)
        if not ok:
            raise ValueError(f"Workflow already exists: {state.workflow_id}")

    async def mutate(self, workflow_id: str, updater: Updater) -> WorkflowState:
        from redis.exceptions import WatchError

        key = self._key(workflow_id)
        for _ in range(8):
            async with self._redis.pipeline() as pipe:
                try:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    if raw is None:
                        raise KeyError(f"Unknown workflow: {workflow_id}")
                    current = WorkflowState.model_validate_json(raw)
                    updated = current.model_copy(deep=True)
                    updater(updated)
                    updated.version = current.version + 1
                    updated.updated_at = utc_now()
                    await pipe.multi()
                    pipe.set(key, updated.model_dump_json())
                    await pipe.execute()
                    return updated
                except WatchError:
                    continue
        raise RuntimeError(f"Concurrent update retry limit exceeded for workflow {workflow_id}")

    async def get(self, workflow_id: str) -> WorkflowState:
        raw = await self._redis.get(self._key(workflow_id))
        if raw is None:
            raise KeyError(f"Unknown workflow: {workflow_id}")
        return WorkflowState.model_validate_json(raw)

    async def close(self) -> None:
        await self._redis.aclose()


def build_state_store(backend: str, redis_url: str) -> StateStore:
    if backend == "redis":
        return RedisStateStore(redis_url)
    return InMemoryStateStore()


def append_event(state: WorkflowState, event_type: str, message: str, **kwargs: object) -> None:
    task_id = kwargs.pop("task_id", None)
    agent_name = kwargs.pop("agent_name", None)
    details = kwargs.pop("details", {})
    state.events.append(
        WorkflowEvent(
            event_type=event_type,
            message=message,
            task_id=str(task_id) if task_id is not None else None,
            agent_name=str(agent_name) if agent_name is not None else None,
            details=details if isinstance(details, dict) else {"value": details},
        )
    )
