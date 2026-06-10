"""将 ReAct Agent 运行结果整理为 UI 可直接展示的数据。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def route_from_tools(tool_names: Iterable[str]) -> str:
    """根据实际工具调用计算本轮检索路线。"""
    tools = set(tool_names)
    has_graph = "medical_graph_search" in tools
    has_vector = "medical_vector_search" in tools
    if has_graph and has_vector:
        return "hybrid"
    if has_graph:
        return "graph"
    if has_vector:
        return "vector"
    return "none"


def build_agent_view(result: Dict[str, Any]) -> Dict[str, Any]:
    """从工作流结果提取路线、证据、工具观察和 Harness 信息。"""
    trace = list(result.get("agent_trace") or [])
    tool_events = [
        item for item in trace if item.get("node") == "tool_observation"
    ]
    tool_names = _unique(
        str(item.get("tool_name", "")) for item in tool_events
    )
    evidence = dict(result.get("evidence_assessment") or {})
    if not evidence:
        evidence_nodes = [
            item for item in trace if item.get("node") == "evidence_check"
        ]
        if evidence_nodes:
            evidence = dict(evidence_nodes[-1])

    return {
        "route": route_from_tools(tool_names),
        "skills": list(result.get("selected_skill_names") or []),
        "tools": tool_names,
        "tool_events": tool_events,
        "evidence_status": str(evidence.get("status", "")),
        "evidence_reason": str(evidence.get("reason", "")),
        "answer_mode": str(evidence.get("answer_mode", "")),
        "risk_level": str(evidence.get("risk_level", "")),
        "stop_reason": str(result.get("stop_reason", "")),
        "harness": dict(result.get("harness_summary") or {}),
        "output_review": dict(result.get("output_review") or {}),
        "trace": trace,
    }
