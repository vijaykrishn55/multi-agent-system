from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multi_agent_system.bus import AsyncMessageBus
from multi_agent_system.config import Settings
from multi_agent_system.engine import MultiAgentEngine
from multi_agent_system.llm import DeterministicProvider
from multi_agent_system.models import TaskMessage


class InvalidOnceProvider(DeterministicProvider):
    def __init__(self) -> None:
        super().__init__(delay_seconds=0.01)
        self.invalid_once = True

    async def run_agent(self, agent_name, user_query, instruction, context):
        if agent_name == "researcher" and self.invalid_once:
            self.invalid_once = False
            self.agent_calls[agent_name] += 1
            return {"agent_name": "researcher", "summary": "intentionally malformed test response"}
        return await super().run_agent(agent_name, user_query, instruction, context)


async def run_checks() -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    provider = DeterministicProvider(delay_seconds=0.02)
    engine = MultiAgentEngine(
        provider=provider,
        settings=Settings(max_agent_retries=1, retry_base_delay_seconds=0, workflow_timeout_seconds=3),
    )
    try:
        result = await engine.run("Analyze EV market themes and summarize policy considerations.")
        state = await engine.get_state(result.workflow_id)
        checks.append({
            "name": "happy_path_orchestration",
            "passed": result.status == "complete" and set(result.agent_outputs) == {"researcher", "summarizer"},
            "details": {"status": result.status, "agents": sorted(result.agent_outputs)},
        })
        checks.append({
            "name": "shared_state_updates",
            "passed": state.status == "complete" and len(state.tasks) == 2 and all(t.status == "succeeded" for t in state.tasks.values()),
            "details": {"state_version": state.version, "tasks": {k: v.status for k, v in state.tasks.items()}},
        })
        checks.append({
            "name": "concurrent_agent_execution",
            "passed": provider.max_concurrent_agents >= 2,
            "details": {"max_concurrent_agents": provider.max_concurrent_agents},
        })
    finally:
        await engine.close()

    retry_provider = InvalidOnceProvider()
    retry_engine = MultiAgentEngine(
        provider=retry_provider,
        settings=Settings(max_agent_retries=1, retry_base_delay_seconds=0, workflow_timeout_seconds=3),
    )
    try:
        retry_result = await retry_engine.run("Explain reliable asynchronous message delivery.")
        retry_state = await retry_engine.get_state(retry_result.workflow_id)
        checks.append({
            "name": "malformed_output_retry",
            "passed": retry_result.status == "complete" and retry_provider.agent_calls["researcher"] == 2 and any(e.event_type == "agent_attempt_failed" for e in retry_state.events),
            "details": {"researcher_attempts": retry_provider.agent_calls["researcher"], "status": retry_result.status},
        })
    finally:
        await retry_engine.close()

    bus = AsyncMessageBus(maxsize=1)
    pending = TaskMessage(
        workflow_id="verification-queue",
        agent_name="researcher",
        user_query="queue verification",
        instruction="Confirm retained message delivery",
    )
    await bus.publish(pending)
    retained_count = bus.queued_count("researcher")
    received = await bus.receive("researcher")
    bus.queues["researcher"].task_done()
    checks.append({
        "name": "queued_message_retained_until_worker_consumes",
        "passed": retained_count == 1 and received is not None and received.task_id == pending.task_id,
        "details": {"queued_before_consume": retained_count, "received_after_consume": received is not None},
    })

    return checks


def main() -> int:
    print("Running automated tests...")
    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    test_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        env=child_env,
        text=True,
        capture_output=True,
    )
    print(test_run.stdout)
    if test_run.stderr:
        print(test_run.stderr[-2000:])

    checks = asyncio.run(run_checks())
    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'}: {check['name']}")

    all_passed = test_run.returncode == 0 and all(bool(check["passed"]) for check in checks)
    report = {
        "project": "Multi-Agent System",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "automated_test_command": "python -m pytest -q",
        "automated_tests_passed": test_run.returncode == 0,
        "pytest_output": test_run.stdout.strip(),
        "checks": checks,
        "passed": all_passed,
    }
    report_path = ROOT / "reports" / "verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nVerification {'PASSED' if all_passed else 'FAILED'}")
    print(f"Report: {report_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
