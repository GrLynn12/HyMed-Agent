"""使用真实千问、Neo4j、FAISS 和 NER 评测医疗 Agent。

测试集只包含问题与人工标注，不提供预生成证据或回答。运行前需要确保：

* 千问 API Key 已配置；
* Neo4j 已启动且连接配置正确；
* FAISS 索引已经构建；
* NER 模型与 checkpoint 文件可用。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = (
    PROJECT_ROOT / "evaluation" / "cases" / "real_medical_agent.jsonl"
)
DEFAULT_REPORT_DIR = PROJECT_ROOT / "evaluation_reports" / "real"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class RealCaseResult:
    case_id: str
    query: str
    passed: bool
    checks: Dict[str, bool]
    actual: Dict[str, Any]
    expected: Dict[str, Any]
    answer: str
    trace: List[Dict[str, Any]]
    failures: List[str]
    latency_ms: float
    api_calls: int


class CountingLLM:
    """为真实 QwenClient 增加 API 调用计数，不改变请求内容。"""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.api_calls = 0

    def complete(self, prompt: str) -> str:
        self.api_calls += 1
        return self.client.complete(prompt)

    def chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        self.api_calls += 1
        return self.client.chat(messages, **kwargs)

    def stream_chat(self, messages: List[Dict[str, Any]]) -> Iterable[str]:
        self.api_calls += 1
        return self.client.stream_chat(messages)


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 格式错误: {exc}") from exc
            if not case.get("id") or not case.get("query"):
                raise ValueError(f"{path}:{line_no} 缺少 id 或 query")
            cases.append(case)
    return cases


def build_real_runtime() -> Dict[str, Any]:
    """加载与 WebUI 相同的真实运行资源。"""
    import py2neo
    import torch
    from transformers import BertTokenizer

    from medical_rag import ner
    from medical_rag.clients.neo4j import KGClient
    from medical_rag.clients.qwen import get_llm_client
    from medical_rag.core.config import settings
    from medical_rag.core.devices import resolve_torch_device
    from medical_rag.retrieval.tools import MedicalRAGTools
    from medical_rag.retrieval.vector_store import MedicalVectorStore

    device = resolve_torch_device(settings.COMPUTE_DEVICE)
    tag_path = os.path.join(settings.TMP_DIR, "tag2idx.npy")
    checkpoint_path = os.path.join(
        settings.MODEL_DIR,
        f"{settings.NER_CHECKPOINT}.pt",
    )
    with open(tag_path, "rb") as handle:
        tag2idx = pickle.load(handle)

    idx2tag = list(tag2idx)
    tokenizer = BertTokenizer.from_pretrained(settings.NER_MODEL_NAME)
    model = ner.Bert_Model(
        settings.NER_MODEL_NAME,
        hidden_size=128,
        tag_num=len(tag2idx),
        bi=True,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    vector_store = MedicalVectorStore()
    vector_store.load()

    graph = py2neo.Graph(
        settings.NEO4J_URL,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        name=settings.NEO4J_DBNAME,
    )
    graph.run("RETURN 1 AS ok").data()

    llm = CountingLLM(get_llm_client())
    tools = MedicalRAGTools(
        llm=llm,
        kg=KGClient(graph),
        bert_model=model,
        bert_tokenizer=tokenizer,
        rule=ner.rule_find(),
        tfidf_r=ner.tfidf_alignment(),
        device=device,
        idx2tag=idx2tag,
        vector_store=vector_store,
    )
    return {"llm": llm, "tools": tools}


def _trace_tools(trace: Sequence[Dict[str, Any]]) -> List[str]:
    return [
        str(item["tool_name"])
        for item in trace
        if item.get("node") == "tool_observation" and item.get("tool_name")
    ]


def _trace_skills(trace: Sequence[Dict[str, Any]]) -> List[str]:
    for item in trace:
        if item.get("node") == "select_skills":
            return [
                str(skill.get("name", ""))
                for skill in item.get("skills", [])
                if skill.get("name")
            ]
    return []


def _route(tools: Sequence[str]) -> str:
    names = set(tools)
    if {"medical_graph_search", "medical_vector_search"} <= names:
        return "hybrid"
    if "medical_graph_search" in names:
        return "graph"
    if "medical_vector_search" in names:
        return "vector"
    return "none"


def _vector_results(trace: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for item in trace:
        if (
            item.get("node") == "tool_observation"
            and item.get("tool_name") == "medical_vector_search"
        ):
            results.extend((item.get("debug") or {}).get("results") or [])
    return results


def _memory_types(trace: Sequence[Dict[str, Any]], key: str) -> List[str]:
    for item in trace:
        if item.get("node") == "prepare_context":
            return [
                str(memory.get("memory_type", ""))
                for memory in item.get(key, [])
                if memory.get("memory_type")
            ]
    return []


def _contains_all(text: str, markers: Sequence[str]) -> bool:
    return all(marker in text for marker in markers)


def _contains_none(text: str, markers: Sequence[str]) -> bool:
    return all(marker not in text for marker in markers)


def _set_precision(actual: Sequence[str], expected: Sequence[str]) -> float:
    if not actual:
        return 1.0 if not expected else 0.0
    return len(set(actual) & set(expected)) / len(set(actual))


def _set_recall(actual: Sequence[str], expected: Sequence[str]) -> float:
    if not expected:
        return 1.0
    return len(set(actual) & set(expected)) / len(set(expected))


def _retrieval_hit(
    results: Sequence[Dict[str, Any]],
    keywords: Sequence[str],
    top_k: int,
) -> bool:
    if not keywords:
        return True
    text = "\n".join(
        " ".join(
            [
                str(item.get("content", "")),
                str(item.get("source", "")),
                str(item.get("section", "")),
            ]
        )
        for item in results[:top_k]
    )
    return any(keyword in text for keyword in keywords)


def evaluate_case(
    case: Dict[str, Any],
    runtime: Dict[str, Any],
) -> RealCaseResult:
    from medical_rag.memory.store import SQLiteMemoryStore
    from medical_rag.workflow.react_rag import run_react_rag

    expected = case.get("expect") or {}
    llm = runtime["llm"]
    before_calls = llm.api_calls
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="real_agent_eval_") as tmp_dir:
        result = run_react_rag(
            case["query"],
            llm,
            runtime["tools"],
            history=case.get("history", []),
            username=f"real_eval_{case['id']}",
            memory_store=SQLiteMemoryStore(Path(tmp_dir) / "memory.db"),
        )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    api_calls = llm.api_calls - before_calls

    trace = result.get("agent_trace", [])
    answer = str(result.get("final_answer", ""))
    tools = _trace_tools(trace)
    skills = _trace_skills(trace)
    retrieval_results = _vector_results(trace)
    saved_memory = _memory_types(trace, "saved_memories")
    injected_memory = _memory_types(trace, "relevant_memories")
    actual = {
        "route": _route(tools),
        "skills": skills,
        "tools": tools,
        "evidence_status": str(
            (result.get("evidence_assessment") or {}).get("status", "")
        ),
        "answer_mode": str(
            (result.get("evidence_assessment") or {}).get("answer_mode", "")
        ),
        "risk_level": str(
            (result.get("evidence_assessment") or {}).get("risk_level", "")
        ),
        "retrieval_results": retrieval_results,
        "saved_memory_types": saved_memory,
        "injected_memory_types": injected_memory,
        "output_review": result.get("output_review") or {},
    }

    checks: Dict[str, bool] = {}
    failures: List[str] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks[name] = passed
        if not passed:
            failures.append(detail)

    if "route" in expected:
        check(
            "route",
            actual["route"] == expected["route"],
            f"route expected={expected['route']}, actual={actual['route']}",
        )

    expected_skills = list(expected.get("skills") or [])
    if "skills" in expected:
        check(
            "skills",
            set(skills) == set(expected_skills),
            f"skills expected={expected_skills}, actual={skills}",
        )

    expected_tools = list(expected.get("tools") or [])
    if "tools" in expected:
        check(
            "tools",
            set(tools) == set(expected_tools),
            f"tools expected={expected_tools}, actual={tools}",
        )

    expected_evidence = expected.get("evidence_status")
    if expected_evidence:
        check(
            "evidence_status",
            actual["evidence_status"] == expected_evidence,
            "evidence_status "
            f"expected={expected_evidence}, actual={actual['evidence_status']}",
        )

    retrieval_keywords = list(expected.get("retrieval_hit_keywords") or [])
    retrieval_k = int(expected.get("retrieval_k", 5))
    if retrieval_keywords:
        check(
            "retrieval_hit_at_k",
            _retrieval_hit(retrieval_results, retrieval_keywords, retrieval_k),
            f"top@{retrieval_k} 未命中 {retrieval_keywords}",
        )

    required_claims = list(expected.get("required_claims") or [])
    if required_claims:
        check(
            "required_claims",
            _contains_all(answer, required_claims),
            f"回答未覆盖必要事实 {required_claims}",
        )

    unsupported_markers = list(expected.get("unsupported_claim_markers") or [])
    check(
        "unsupported_claims",
        _contains_none(answer, unsupported_markers),
        f"回答出现无依据陈述 {unsupported_markers}",
    )

    safety_markers = list(expected.get("critical_safety_markers") or [])
    check(
        "safety",
        _contains_none(answer, safety_markers),
        f"回答出现关键安全错误 {safety_markers}",
    )

    expected_saved = list(expected.get("saved_memory_types") or [])
    expected_injected = list(expected.get("injected_memory_types") or [])
    if "saved_memory_types" in expected or "injected_memory_types" in expected:
        check(
            "memory",
            set(saved_memory) == set(expected_saved)
            and set(injected_memory) == set(expected_injected),
            "memory "
            f"expected write={expected_saved}, inject={expected_injected}; "
            f"actual write={saved_memory}, inject={injected_memory}",
        )

    return RealCaseResult(
        case_id=case["id"],
        query=case["query"],
        passed=all(checks.values()),
        checks=checks,
        actual=actual,
        expected=expected,
        answer=answer,
        trace=trace,
        failures=failures,
        latency_ms=latency_ms,
        api_calls=api_calls,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _accuracy(results: Sequence[RealCaseResult], check_name: str) -> float | None:
    values = [
        result.checks[check_name]
        for result in results
        if check_name in result.checks
    ]
    return round(sum(values) / len(values), 4) if values else None


def _percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * ratio
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 2)


def summarize(results: Sequence[RealCaseResult]) -> Dict[str, Any]:
    expected_tool_sets = [list(item.expected.get("tools") or []) for item in results]
    actual_tool_sets = [list(item.actual.get("tools") or []) for item in results]
    tool_precision = [
        _set_precision(actual, expected)
        for actual, expected in zip(actual_tool_sets, expected_tool_sets)
    ]
    tool_recall = [
        _set_recall(actual, expected)
        for actual, expected in zip(actual_tool_sets, expected_tool_sets)
    ]
    unsupported = [
        not item.checks["unsupported_claims"]
        for item in results
        if "unsupported_claims" in item.checks
    ]
    safety_errors = [
        not item.checks["safety"]
        for item in results
        if "safety" in item.checks
    ]
    latencies = [item.latency_ms for item in results]
    passed = sum(item.passed for item in results)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "metrics": {
            "Route / Skill Accuracy": {
                "route_accuracy": _accuracy(results, "route"),
                "skill_accuracy": _accuracy(results, "skills"),
            },
            "Tool Precision": round(_mean(tool_precision), 4),
            "Tool Recall": round(_mean(tool_recall), 4),
            "Retrieval Hit@K": _accuracy(results, "retrieval_hit_at_k"),
            "Evidence Sufficiency Accuracy": _accuracy(
                results,
                "evidence_status",
            ),
            "Unsupported Claim Rate": (
                round(sum(unsupported) / len(unsupported), 4)
                if unsupported
                else None
            ),
            "Safety Error Rate": (
                round(sum(safety_errors) / len(safety_errors), 4)
                if safety_errors
                else None
            ),
            "Memory Accuracy": _accuracy(results, "memory"),
            "Task Success Rate": (
                round(passed / len(results), 4) if results else 0.0
            ),
            "P50 / P95 Latency": {
                "p50_ms": round(median(latencies), 2) if latencies else 0.0,
                "p95_ms": _percentile(latencies, 0.95),
            },
            "Average API Calls": round(
                _mean([item.api_calls for item in results]),
                2,
            ),
            "Average Tool Calls": round(
                _mean([len(item.actual["tools"]) for item in results]),
                2,
            ),
        },
    }


def write_reports(
    results: Sequence[RealCaseResult],
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    payload = {
        "summary": summary,
        "cases": [asdict(result) for result in results],
    }
    (output_dir / "latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Real Medical Agent Evaluation",
        "",
        f"- total: {summary['total']}",
        f"- passed: {summary['passed']}",
        f"- failed: {summary['failed']}",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(
        f"- {name}: {value}" for name, value in summary["metrics"].items()
    )
    lines.extend(["", "## Cases", ""])
    for result in results:
        lines.extend(
            [
                f"### {result.case_id} - {'PASS' if result.passed else 'FAIL'}",
                "",
                f"- query: {result.query}",
                f"- route: {result.actual['route']}",
                f"- skills: {result.actual['skills']}",
                f"- tools: {result.actual['tools']}",
                f"- evidence_status: {result.actual['evidence_status']}",
                f"- answer_mode: {result.actual.get('answer_mode', '')}",
                f"- risk_level: {result.actual.get('risk_level', '')}",
                f"- latency_ms: {result.latency_ms}",
                f"- api_calls: {result.api_calls}",
                f"- answer: {' '.join(result.answer.split())}",
                "",
            ]
        )
        if result.failures:
            lines.append("Failures:")
            lines.extend(f"- {failure}" for failure in result.failures)
            lines.append("")
    (output_dir / "latest.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="真实医疗 Agent 离线评测")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.limit > 0:
        cases = cases[: args.limit]
    runtime = build_real_runtime()
    results = [evaluate_case(case, runtime) for case in cases]
    summary = write_reports(results, args.output_dir)
    print(
        "Real evaluation complete: "
        f"{summary['passed']}/{summary['total']} passed, "
        f"report={args.output_dir / 'latest.md'}"
    )
    return int(args.fail_on_error and summary["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
