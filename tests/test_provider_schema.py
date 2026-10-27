from multi_agent_system.llm import _schema_response_format
from multi_agent_system.models import AgentReport, FinalAnswer, WorkflowPlan


def test_structured_output_schemas_are_strict_and_closed():
    for model, name in [
        (WorkflowPlan, "workflow_plan"),
        (AgentReport, "agent_report"),
        (FinalAnswer, "final_answer"),
    ]:
        response_format = _schema_response_format(model, name)
        assert response_format["type"] == "json_schema"
        schema_info = response_format["json_schema"]
        schema = schema_info["schema"]
        assert schema_info["strict"] is True
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
