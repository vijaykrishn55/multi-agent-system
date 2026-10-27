from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Protocol, Type

from pydantic import BaseModel, ValidationError

from .models import AgentReport, FinalAnswer, WorkflowPlan


class ProviderError(RuntimeError):
    """LLM provider or structured-output failure."""


class _InvalidOutput(Exception):
    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


class AgentProvider(Protocol):
    async def plan(self, user_query: str) -> dict[str, Any]: ...
    async def run_agent(
        self, agent_name: str, user_query: str, instruction: str,
        context: dict[str, Any],
    ) -> dict[str, Any]: ...
    async def synthesize(
        self, user_query: str, reports: dict[str, AgentReport],
        errors: list[dict[str, Any]],
    ) -> dict[str, Any]: ...
    async def close(self) -> None: ...


def _schema_response_format(model: Type[BaseModel], name: str) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(schema.get("properties", {}))
    return {"type": "json_schema", "json_schema": {
        "name": name, "strict": True, "schema": schema,
    }}


def _schema_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(term in text for term in (
        "json_validate_failed", "generated json does not match",
        "failed to validate json", "missing properties", "schema validation",
    ))


class GroqProvider:
    """Async Groq provider with schema validation and one repair fallback."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        try:
            from groq import AsyncGroq
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt to use the Groq provider.") from exc

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.client = AsyncGroq(api_key=key, timeout=50.0, max_retries=0)

    async def _structured_completion(
        self, *, model_type: Type[BaseModel], schema_name: str,
        system_prompt: str, user_prompt: str, max_tokens: int = 1800,
    ) -> dict[str, Any]:
        schema = model_type.model_json_schema()
        fields = list(model_type.model_fields)
        contract = (
            "Return one JSON object only. Include every required field exactly once: "
            + ", ".join(fields)
            + ". Use [] for empty lists, an empty string for empty text fields, and valid "
            "numbers for numeric fields. Do not add keys, markdown, or commentary.\nSchema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )

        async def request(mode: dict[str, Any], prompt: str) -> dict[str, Any]:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt + "\n\n" + contract},
                    {"role": "user", "content": prompt},
                ],
                response_format=mode,
                max_completion_tokens=max_tokens,
            )
            raw = response.choices[0].message.content or ""
            try:
                parsed = json.loads(raw)
                return model_type.model_validate(parsed).model_dump(mode="json")
            except (json.JSONDecodeError, ValidationError) as exc:
                raise _InvalidOutput(str(exc), raw) from exc

        try:
            return await request(_schema_response_format(model_type, schema_name), user_prompt)
        except Exception as first:
            if not isinstance(first, _InvalidOutput) and not _schema_error(first):
                raise ProviderError(f"Structured completion failed: {first}") from first

        # JSON mode is a compatibility fallback; Pydantic still validates locally.
        try:
            return await request({"type": "json_object"}, user_prompt)
        except _InvalidOutput as invalid:
            repair = json.dumps({
                "task": user_prompt,
                "invalid_output": invalid.raw[:6000],
                "validation_error": str(invalid),
                "required_fields": fields,
                "schema": schema,
                "instruction": (
                    "Repair the supplied output to match the schema exactly. "
                    "Preserve supported facts, include all fields, and return JSON only."
                ),
            }, ensure_ascii=False)
            try:
                return await request({"type": "json_object"}, repair)
            except Exception as exc:
                raise ProviderError(f"JSON fallback and one repair attempt failed: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"JSON fallback failed: {exc}") from exc

    async def plan(self, user_query: str) -> dict[str, Any]:
        return await self._structured_completion(
            model_type=WorkflowPlan, schema_name="workflow_plan", max_tokens=1000,
            system_prompt=(
                "You are a multi-agent orchestrator. Create two complementary tasks: "
                "one for the researcher and one for the summarizer. Avoid duplicate work. "
                "Do not expose hidden reasoning or claim external browsing."
            ),
            user_prompt=f"User request:\n{user_query}\nCreate the workflow plan.",
        )

    async def run_agent(
        self, agent_name: str, user_query: str, instruction: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if agent_name == "summarizer":
            try:
                delay = max(0.0, float(os.getenv("SUMMARIZER_DELAY_SECONDS", "2")))
            except ValueError:
                delay = 2.0
            await asyncio.sleep(delay)

        prompts = {
            "researcher": (
                "You are the Researcher. Extract relevant facts, approaches, trade-offs, "
                "and technical details. State uncertainty and never invent citations."
            ),
            "summarizer": (
                "You are the Summarizer. Organize the request into themes, decisions, "
                "constraints, risks, and actionable points. Do not claim external verification. "
                "Include all AgentReport fields: agent_name, summary, key_points, assumptions, "
                "risks, evidence_notes, confidence. Use [] for empty lists and a number from 0 to 1."
            ),
        }
        if agent_name not in prompts:
            raise ValueError(f"Unsupported agent: {agent_name}")

        payload = {
            "user_request": user_query,
            "assigned_task": instruction,
            "shared_context": context,
        }
        return await self._structured_completion(
            model_type=AgentReport, schema_name="agent_report",
            system_prompt=prompts[agent_name],
            user_prompt=json.dumps(payload, ensure_ascii=False), max_tokens=1800,
        )

    async def synthesize(
        self, user_query: str, reports: dict[str, AgentReport],
        errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "user_request": user_query,
            "agent_reports": {name: report.model_dump(mode="json") for name, report in reports.items()},
            "errors": errors,
        }
        return await self._structured_completion(
            model_type=FinalAnswer, schema_name="final_answer", max_tokens=2200,
            system_prompt=(
                "You synthesize specialist reports into one clear answer. Do not invent facts. "
                "If an agent failed, explain the limitation and return a partial answer. "
                "Include answer, key_takeaways, limitations, and completion_status."
            ),
            user_prompt=json.dumps(payload, ensure_ascii=False),
        )

    async def close(self) -> None:
        await self.client.close()


class DeterministicProvider:
    """Predictable provider used by tests and the demo mode."""

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds
        self.active_agents = self.max_concurrent_agents = 0
        self.agent_calls = {"researcher": 0, "summarizer": 0}

    async def plan(self, user_query: str) -> dict[str, Any]:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return {
            "goal": "Produce a useful response to the user's request.",
            "research_task": "Identify key facts, approaches, and trade-offs.",
            "summarizer_task": "Extract main themes, constraints, risks, and actions.",
            "scope_notes": ["Use supplied information as the source of truth."],
        }

    async def run_agent(
        self, agent_name: str, user_query: str, instruction: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self.agent_calls[agent_name] = self.agent_calls.get(agent_name, 0) + 1
        self.active_agents += 1
        self.max_concurrent_agents = max(self.max_concurrent_agents, self.active_agents)
        try:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            summary = f"{agent_name.title()} pass for: {user_query.strip()}"
            points = [instruction, "Use explicit assumptions and identify uncertainty."]
            return {
                "agent_name": agent_name,
                "summary": summary,
                "key_points": points,
                "assumptions": ["The user wants a practical answer."],
                "risks": ["Current facts may require external verification."],
                "evidence_notes": ["Generated from the prompt; no external sources queried."],
                "confidence": 0.82,
            }
        finally:
            self.active_agents -= 1

    async def synthesize(
        self, user_query: str, reports: dict[str, AgentReport],
        errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        answer = "\n\n".join(
            f"{name.title()}: {report.summary}" for name, report in reports.items()
        ) or "No agent produced a usable result."
        points = list(dict.fromkeys(
            point for report in reports.values() for point in report.key_points
        ))[:8]
        status = "complete" if len(reports) == 2 and not errors else ("partial" if reports else "failed")
        limits = ["Deterministic demo output; not domain research."]
        if errors:
            limits.append(f"{len(errors)} task failure(s) recorded.")
        return {
            "answer": f"Request: {user_query.strip()}\n\n{answer}",
            "key_takeaways": points,
            "limitations": limits,
            "completion_status": status,
        }

    async def close(self) -> None:
        return None
