from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from .config import Settings
from .engine import MultiAgentEngine
from .llm import DeterministicProvider, GroqProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the asynchronous multi-agent workflow.")
    parser.add_argument("query", help="The request to delegate to the multi-agent system.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic responses for a quick reproducible demo.",
    )
    parser.add_argument(
        "--report",
        default="reports/last_run.json",
        help="Path for the workflow and shared-state JSON report.",
    )
    parser.add_argument(
        "--print-state",
        action="store_true",
        help="Print the full shared workflow state after execution.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.validate()
    provider = DeterministicProvider() if args.mock else GroqProvider(
        api_key=settings.groq_api_key,
        model=settings.model,
    )
    async with MultiAgentEngine(provider=provider, settings=settings) as engine:
        result = await engine.run(args.query)
        state = await engine.get_state(result.workflow_id)
        output = {
            "result": result.model_dump(mode="json"),
            "shared_state": state.model_dump(mode="json"),
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n=== FINAL RESPONSE ===\n")
        print(result.final_answer)
        print(f"\nWorkflow status: {result.status}")
        print(f"Duration: {result.duration_ms:.2f} ms")
        print(f"Agent outputs: {', '.join(sorted(result.agent_outputs)) or 'none'}")
        if result.errors:
            print(f"Recorded errors: {len(result.errors)}")
        print(f"Report: {report_path.resolve()}")
        if args.print_state:
            print("\n=== SHARED STATE ===\n")
            print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return 0 if result.status != "failed" else 2


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise SystemExit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
