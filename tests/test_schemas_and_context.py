import pytest
from pydantic import ValidationError

from multi_agent_system.models import AgentReport, WorkflowPlan, WorkflowState


def test_workflow_plan_requires_two_specialized_assignments():
    plan = WorkflowPlan(
        goal="Create a multi-agent response",
        research_task="Extract important facts",
        summarizer_task="Organize key themes and risks",
        scope_notes=[],
    )
    assert plan.research_task
    assert plan.summarizer_task
    with pytest.raises(ValidationError):
        WorkflowPlan.model_validate({"goal": "missing tasks"})


def test_agent_name_and_confidence_are_validated():
    with pytest.raises(ValidationError):
        AgentReport.model_validate({
            "agent_name": "writer",
            "summary": "summary",
            "key_points": [],
            "assumptions": [],
            "risks": [],
            "evidence_notes": [],
            "confidence": 1.3,
        })


def test_empty_query_is_rejected():
    with pytest.raises(ValidationError):
        WorkflowState(workflow_id="wf", user_query="  ")
