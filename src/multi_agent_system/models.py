from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WorkflowPlan(StrictModel):
    goal: str = Field(description="A short description of the user's intended outcome.")
    research_task: str = Field(description="Focused assignment for the research specialist.")
    summarizer_task: str = Field(description="Independent assignment for the summarization specialist.")
    scope_notes: list[str] = Field(description="Important boundaries, assumptions, or missing information.")


class AgentReport(StrictModel):
    agent_name: Literal["researcher", "summarizer"]
    summary: str = Field(description="Concise result from this specialized agent.")
    key_points: list[str] = Field(description="Important points relevant to the assigned task.")
    assumptions: list[str] = Field(description="Assumptions made; empty if none.")
    risks: list[str] = Field(description="Limitations, uncertainties, or risks; empty if none.")
    evidence_notes: list[str] = Field(description="Evidence basis or verification notes; do not invent citations.")
    confidence: float = Field(ge=0.0, le=1.0)


class FinalAnswer(StrictModel):
    answer: str = Field(description="A coherent final response synthesizing the available agent outputs.")
    key_takeaways: list[str]
    limitations: list[str]
    completion_status: Literal["complete", "partial", "failed"]


class TaskMessage(StrictModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_id: str
    agent_name: Literal["researcher", "summarizer"]
    user_query: str
    instruction: str
    context: dict[str, Any] = Field(default_factory=dict)
    attempt: int = 1
    created_at: datetime = Field(default_factory=utc_now)


class TaskResult(StrictModel):
    task_id: str
    workflow_id: str
    agent_name: Literal["researcher", "summarizer"]
    success: bool
    output: AgentReport | None = None
    error_type: str | None = None
    error_message: str | None = None
    attempts: int = 1
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime = Field(default_factory=utc_now)


class TaskProgress(StrictModel):
    task_id: str
    agent_name: Literal["researcher", "summarizer"]
    instruction: str
    status: Literal["queued", "running", "retrying", "succeeded", "failed"] = "queued"
    attempts: int = 0
    error_type: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class WorkflowEvent(StrictModel):
    event_type: str
    message: str
    task_id: str | None = None
    agent_name: str | None = None
    at: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(StrictModel):
    workflow_id: str
    user_query: str
    status: Literal["planning", "running", "complete", "partial", "failed"] = "planning"
    plan: WorkflowPlan | None = None
    tasks: dict[str, TaskProgress] = Field(default_factory=dict)
    outputs: dict[str, AgentReport] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: FinalAnswer | None = None
    events: list[WorkflowEvent] = Field(default_factory=list)
    version: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("user_query")
    @classmethod
    def query_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("user_query cannot be empty")
        return value.strip()


class WorkflowResult(StrictModel):
    workflow_id: str
    status: Literal["complete", "partial", "failed"]
    final_answer: str
    key_takeaways: list[str]
    agent_outputs: dict[str, AgentReport]
    errors: list[dict[str, Any]]
    duration_ms: float
    state_version: int
