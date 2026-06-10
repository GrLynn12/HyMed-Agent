"""真实 Agent 评测器的纯离线回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import evaluate_real_agent


def _result(
    case_id: str,
    *,
    passed: bool,
    checks: dict[str, bool],
    tools: list[str],
) -> evaluate_real_agent.RealCaseResult:
    return evaluate_real_agent.RealCaseResult(
        case_id=case_id,
        query="测试问题",
        passed=passed,
        checks=checks,
        actual={
            "route": "vector",
            "skills": [],
            "tools": tools,
            "evidence_status": "sufficient",
        },
        expected={"tools": ["medical_vector_search"]},
        answer="测试回答",
        trace=[],
        failures=[],
        latency_ms=100.0,
        api_calls=2,
    )


def test_load_real_cases_rejects_missing_query(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"id": "missing_query"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        evaluate_real_agent.load_cases(path)
    except ValueError as exc:
        assert "缺少 id 或 query" in str(exc)
    else:
        raise AssertionError("缺少 query 的样例应被拒绝")


def test_real_summary_calculates_requested_metrics() -> None:
    results = [
        _result(
            "ok",
            passed=True,
            checks={
                "route": True,
                "skills": True,
                "retrieval_hit_at_k": True,
                "evidence_status": True,
                "unsupported_claims": True,
                "safety": True,
                "memory": True,
            },
            tools=["medical_vector_search"],
        ),
        _result(
            "bad",
            passed=False,
            checks={
                "route": False,
                "skills": True,
                "retrieval_hit_at_k": False,
                "evidence_status": False,
                "unsupported_claims": False,
                "safety": False,
                "memory": False,
            },
            tools=["medical_graph_search"],
        ),
    ]

    summary = evaluate_real_agent.summarize(results)
    metrics = summary["metrics"]

    assert metrics["Route / Skill Accuracy"]["route_accuracy"] == 0.5
    assert metrics["Route / Skill Accuracy"]["skill_accuracy"] == 1.0
    assert metrics["Tool Precision"] == 0.5
    assert metrics["Tool Recall"] == 0.5
    assert metrics["Retrieval Hit@K"] == 0.5
    assert metrics["Evidence Sufficiency Accuracy"] == 0.5
    assert metrics["Unsupported Claim Rate"] == 0.5
    assert metrics["Safety Error Rate"] == 0.5
    assert metrics["Memory Accuracy"] == 0.5
    assert metrics["Task Success Rate"] == 0.5
