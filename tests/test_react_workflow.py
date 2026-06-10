"""受控 ReAct 工作流的端到端假模型测试。"""

import json
from types import SimpleNamespace

from medical_rag.harness.runtime import HarnessConfig
from medical_rag.memory.store import SQLiteMemoryStore
from medical_rag.retrieval.tools import ToolResult
from medical_rag.workflow.react_rag import run_react_rag


class ReActFakeLLM:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if "长期记忆提取器" in prompt:
            return json.dumps(
                [{"memory_type": "allergy", "content": "用户对青霉素过敏"}],
                ensure_ascii=False,
            )
        if "受控医疗检索 Agent" in prompt:
            if "medical_graph_search:" not in prompt:
                return json.dumps(
                    {
                        "decision": "tool",
                        "tool_name": "medical_graph_search",
                        "arguments": {"query": "皮炎外用药治疗"},
                        "reason": "先查结构化治疗信息",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "decision": "tool",
                    "tool_name": "medical_vector_search",
                    "arguments": {"query": "皮炎常用外用药治疗", "top_k": 5},
                    "reason": "图谱结果过于泛化",
                },
                ensure_ascii=False,
            )
        if "没有通过安全检查" in prompt:
            return "证据只说明治疗需根据皮炎类型选择，且应告知医生青霉素过敏史。"
        return "皮炎外用药需要根据具体类型选择。"

    def chat(self, messages, tools=None, tool_choice=None):
        message = SimpleNamespace(
            content="皮炎外用药需要根据具体类型选择，并告知医生青霉素过敏史。"
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ReActFakeTools:
    def call_tool(self, name, arguments):
        if name == "medical_graph_search":
            return ToolResult(
                name=name,
                content="药物治疗",
                entities={"疾病": "皮炎"},
                intents="查询疾病的治疗方法",
                debug={"query": arguments["query"]},
            )
        return ToolResult(
            name=name,
            content="皮炎外用治疗需根据皮炎类型和皮损情况选择。",
            sources="test-corpus",
            debug={
                "query": arguments["query"],
                "results": [{"score": 0.82, "content": "皮炎外用治疗"}],
            },
        )


class MalformedDecisionLLM(ReActFakeLLM):
    def complete(self, prompt):
        if "受控医疗检索 Agent" in prompt:
            self.prompts.append(prompt)
            return "我觉得应该先检索一下"
        return super().complete(prompt)


class UnsafeAnswerLLM(ReActFakeLLM):
    def chat(self, messages, tools=None, tool_choice=None):
        message = SimpleNamespace(content="你患有皮炎，建议每日使用10mg。")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DuplicateGraphLLM(ReActFakeLLM):
    def complete(self, prompt):
        if "受控医疗检索 Agent" in prompt:
            self.prompts.append(prompt)
            return json.dumps(
                {
                    "decision": "tool",
                    "tool_name": "medical_graph_search",
                    "arguments": {"query": "皮炎治疗"},
                    "reason": "重复请求图谱",
                },
                ensure_ascii=False,
            )
        return super().complete(prompt)


class NoEvidenceTools:
    def call_tool(self, name, arguments):
        if name == "medical_vector_search":
            return ToolResult(
                name=name,
                content="未找到可用医疗文本片段。",
                debug={"query": arguments["query"], "results": []},
            )
        return ToolResult(
            name=name,
            content="图谱检索未找到可用结构化知识。",
            debug={"query": arguments["query"]},
        )


class SafeFallbackLLM(ReActFakeLLM):
    def chat(self, messages, tools=None, tool_choice=None):
        message = SimpleNamespace(
            content=(
                "知识库未检索到足够证据。规律睡眠通常有助于维持日间精力，"
                "如长期睡眠不佳可咨询医生。"
            )
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_react_workflow_uses_graph_then_vector_and_memory_same_turn(tmp_path):
    result = run_react_rag(
        "我对青霉素过敏，皮炎可以擦哪些药",
        ReActFakeLLM(),
        ReActFakeTools(),
        username="alice",
        memory_store=SQLiteMemoryStore(str(tmp_path / "memory.db")),
        harness_config=HarnessConfig(max_tool_calls=3),
    )

    tool_nodes = [
        item for item in result["agent_trace"] if item["node"] == "tool_observation"
    ]
    assert [item["tool_name"] for item in tool_nodes] == [
        "medical_graph_search",
        "medical_vector_search",
    ]
    assert result["selected_skill_names"] == [
        "medication_advice",
        "disease_treatment",
    ]
    assert result["evidence_assessment"]["status"] == "sufficient"
    assert result["harness_summary"]["tool_call_count"] == 2
    assert result["output_review"]["passed"] is True
    assert result["saved_memories"][0]["memory_type"] == "allergy"


def test_react_workflow_stops_at_tool_budget(tmp_path):
    result = run_react_rag(
        "皮炎应该怎么治疗",
        ReActFakeLLM(),
        ReActFakeTools(),
        username="alice",
        memory_store=SQLiteMemoryStore(str(tmp_path / "memory.db")),
        harness_config=HarnessConfig(max_tool_calls=1),
    )

    assert result["harness_summary"]["tool_call_count"] == 1
    assert result["evidence_assessment"]["status"] == "partial"
    assert result["stop_reason"] == "evidence_insufficient"


def test_malformed_react_decision_falls_back_to_skill_tool(tmp_path):
    result = run_react_rag(
        "皮炎不治疗会怎样",
        MalformedDecisionLLM(),
        ReActFakeTools(),
        username="alice",
        memory_store=SQLiteMemoryStore(str(tmp_path / "memory.db")),
        harness_config=HarnessConfig(max_tool_calls=1),
    )

    decision = next(
        item for item in result["agent_trace"] if item["node"] == "react_decision"
    )
    assert decision["source"] == "fallback"
    assert decision["tool_name"] == "medical_vector_search"


def test_unsafe_answer_is_repaired_once(tmp_path):
    result = run_react_rag(
        "皮炎不治疗会怎样",
        UnsafeAnswerLLM(),
        ReActFakeTools(),
        username="alice",
        memory_store=SQLiteMemoryStore(str(tmp_path / "memory.db")),
    )

    guardrail = next(
        item for item in result["agent_trace"] if item["node"] == "output_guardrail"
    )
    assert guardrail["repaired"] is True
    assert guardrail["passed"] is True
    assert "10mg" not in result["final_answer"]


def test_harness_recovers_duplicate_tool_with_evidence_recommendation(tmp_path):
    result = run_react_rag(
        "皮炎应该怎么治疗",
        DuplicateGraphLLM(),
        ReActFakeTools(),
        username="alice",
        memory_store=SQLiteMemoryStore(str(tmp_path / "memory.db")),
    )

    tool_names = [
        item["tool_name"]
        for item in result["agent_trace"]
        if item["node"] == "tool_observation"
    ]
    assert tool_names == ["medical_graph_search", "medical_vector_search"]
    assert any(
        item["node"] == "harness_recovery"
        for item in result["agent_trace"]
    )


def test_low_risk_no_evidence_uses_restricted_llm_fallback(tmp_path):
    result = run_react_rag(
        "为什么规律睡眠很重要",
        SafeFallbackLLM(),
        NoEvidenceTools(),
        username="alice",
        memory_store=SQLiteMemoryStore(str(tmp_path / "memory.db")),
    )

    assert (
        result["evidence_assessment"]["answer_mode"]
        == "insufficient_safe_fallback"
    )
    assert "知识库未检索到足够证据" in result["final_answer"]


def test_high_risk_no_evidence_does_not_use_freeform_fallback(tmp_path):
    result = run_react_rag(
        "两种药能一起吃吗",
        SafeFallbackLLM(),
        NoEvidenceTools(),
        username="alice",
        memory_store=SQLiteMemoryStore(str(tmp_path / "memory.db")),
    )

    assert result["evidence_assessment"]["answer_mode"] == "insufficient_refuse"
    assert "无法安全提供具体诊疗或用药建议" in result["final_answer"]
