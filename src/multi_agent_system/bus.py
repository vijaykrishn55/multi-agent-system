from __future__ import annotations

import asyncio
from collections.abc import Iterable

from .models import TaskMessage, TaskResult

AGENT_NAMES = ("researcher", "summarizer")


class AsyncMessageBus:
    """Bounded, per-agent asyncio queues. Items remain queued until a worker consumes them."""

    def __init__(self, maxsize: int = 4) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self.queues: dict[str, asyncio.Queue[TaskMessage | None]] = {
            name: asyncio.Queue(maxsize=maxsize) for name in AGENT_NAMES
        }
        self.results: asyncio.Queue[TaskResult] = asyncio.Queue()

    async def publish(self, message: TaskMessage) -> None:
        if message.agent_name not in self.queues:
            raise ValueError(f"Unknown agent queue: {message.agent_name}")
        await self.queues[message.agent_name].put(message)

    async def receive(self, agent_name: str) -> TaskMessage | None:
        if agent_name not in self.queues:
            raise ValueError(f"Unknown agent queue: {agent_name}")
        return await self.queues[agent_name].get()

    async def publish_result(self, result: TaskResult) -> None:
        await self.results.put(result)

    async def next_result(self) -> TaskResult:
        return await self.results.get()

    def queued_count(self, agent_name: str) -> int:
        return self.queues[agent_name].qsize()

    async def stop_workers(self, agent_names: Iterable[str] = AGENT_NAMES) -> None:
        for name in agent_names:
            await self.queues[name].put(None)
