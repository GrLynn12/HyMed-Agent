"""Agent UI 展示适配层测试。"""

from medical_rag.ui.agent_view import build_agent_view, route_from_tools


def test_route_from_actual_tool_calls() -> None:
    assert route_from_tools(["medical_vector_search"]) == "vector"
    assert route_from_tools(["medical_graph_search"]) == "graph"
    assert route_from_tools(
        ["medical_graph_search", "medical_vector_search"]
    ) == "hybrid"
    assert route_from_tools([]) == "none"


def test_build_agent_view_uses_react_result_fields() -> None:
    result = {
        "selected_skill_names": ["medication_advice"],
        "evidence_assessment": {
            "status": "partial",
            "reason": "文本证据不足",
            "answer_mode": "partial",
            "risk_level": "high",
        },
        "stop_reason": "evidence_insufficient",
        "harness_summary": {"tool_call_count": 2},
        "output_review": {"passed": True},
        "agent_trace": [
            {
                "node": "tool_observation",
                "tool_name": "medical_graph_search",
                "arguments": {"query": "药物联用"},
                "result_preview": "图谱结果",
            },
            {
                "node": "tool_observation",
                "tool_name": "medical_vector_search",
                "arguments": {"query": "药物联用", "top_k": 5},
                "result_preview": "向量结果",
                "debug": {"results": [{"rerank_score": 0.71}]},
            },
        ],
    }

    view = build_agent_view(result)

    assert view["route"] == "hybrid"
    assert view["skills"] == ["medication_advice"]
    assert view["evidence_status"] == "partial"
    assert view["risk_level"] == "high"
    assert view["harness"]["tool_call_count"] == 2
    assert view["tool_events"][1]["debug"]["results"][0][
        "rerank_score"
    ] == 0.71
