"""Agent Harness 的预算、重复调用和参数约束测试。"""

from medical_rag.harness.runtime import (
    AgentHarness,
    HarnessConfig,
    ToolCallRequest,
)
from medical_rag.retrieval.tools import ToolResult


def _executor(name, arguments):
    return ToolResult(name=name, content=f"ok:{arguments['query']}")


def test_harness_clamps_top_k_and_blocks_duplicate_calls():
    harness = AgentHarness(
        HarnessConfig(max_tool_calls=3, max_same_tool_calls=1, max_top_k=6)
    )
    request = ToolCallRequest(
        name="medical_vector_search",
        arguments={"query": "皮炎治疗", "top_k": 100},
    )

    result, status = harness.execute_tool(request, _executor)
    assert status == "completed"
    assert result.content == "ok:皮炎治疗"
    assert harness.trace[-1]["arguments"]["top_k"] == 6

    result, status = harness.execute_tool(request, _executor)
    assert result is None
    assert status == "duplicate_tool_call"


def test_harness_rejects_unknown_tool_empty_query_and_budget():
    harness = AgentHarness(HarnessConfig(max_tool_calls=1))

    _, status = harness.execute_tool(
        ToolCallRequest("unknown", {"query": "皮炎"}),
        _executor,
    )
    assert status == "tool_not_allowed"

    _, status = harness.execute_tool(
        ToolCallRequest("medical_graph_search", {"query": ""}),
        _executor,
    )
    assert status == "empty_query"

    _, status = harness.execute_tool(
        ToolCallRequest("medical_graph_search", {"query": "皮炎"}),
        _executor,
    )
    assert status == "completed"

    _, status = harness.execute_tool(
        ToolCallRequest("medical_vector_search", {"query": "皮炎"}),
        _executor,
    )
    assert status == "tool_budget_exceeded"


def test_harness_recovers_from_invalid_top_k():
    harness = AgentHarness(HarnessConfig(max_top_k=6))
    result, status = harness.execute_tool(
        ToolCallRequest(
            "medical_vector_search",
            {"query": "皮炎", "top_k": "not-a-number"},
        ),
        _executor,
    )

    assert status == "completed"
    assert result is not None
