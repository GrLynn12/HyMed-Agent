"""Harness 约束下的 Skill-Augmented ReAct Hybrid RAG 工作流。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict

from medical_rag.agent.decision import decide_next_action
from medical_rag.clients.qwen import QwenClient
from medical_rag.harness.evidence import assess_evidence
from medical_rag.harness.guardrails import review_output
from medical_rag.harness.runtime import AgentHarness, HarnessConfig, ToolCallRequest
from medical_rag.memory.service import (
    format_memories_for_prompt,
    load_relevant_memories,
    save_explicit_memories,
)
from medical_rag.memory.store import SQLiteMemoryStore
from medical_rag.retrieval.tools import MedicalRAGTools, ToolResult
from medical_rag.skills.registry import Skill, SkillRegistry
from medical_rag.workflow.hybrid_rag import (
    REWRITE_PROMPT,
    REWRITE_REPAIR_PROMPT,
    SYSTEM_PROMPT,
    _extract_topic_hint,
    _format_history,
    _inherits_previous_topic,
    _last_user_query,
    _message_to_dict,
    _needs_rewrite,
    _trim_history,
)


FINAL_PROMPT = """
请根据审核后的检索证据回答医疗问题。

要求：
- 只使用证据中能够支持的医学事实。
- 不做确诊，不把用户记忆当作诊断证据。
- 具体药物、剂量、频次和疗程必须在证据中明确出现。
- 用户记忆只用于过敏、用药和表达偏好的个性化提醒。
- 证据不完整时明确说明限制，不要用常识补齐。
- 回答自然、具体、简洁。

用户问题：{query}

已加载 Skills：
{skills}

相关用户长期记忆：
{memories}

检索证据：
{evidence}

证据状态：{evidence_status}
"""

REPAIR_PROMPT = """
下面的医疗回答没有通过安全检查。请只依据给定证据修复，不得增加新的医学事实。

违规项：{violations}
用户问题：{query}
用户记忆：{memories}
检索证据：{evidence}
原回答：{answer}

只输出修复后的中文回答。
"""

SAFE_FALLBACK_PROMPT = """
知识库没有找到足够证据。请仅用一般性医学常识回答这个低风险问题。

必须遵守：
1. 开头明确说明“知识库未检索到足够证据”。
2. 只能提供一般性、非个体化的健康教育信息。
3. 不得诊断疾病，不得推荐具体药物。
4. 不得提供剂量、频次、疗程或药物联用判断。
5. 不得把用户记忆当作医学证据。
6. 信息不确定时建议咨询医生。

用户问题：{query}

只输出简洁中文回答。
"""


class ReActState(TypedDict, total=False):
    query: str
    username: str
    history: List[Dict[str, str]]
    working_query: str
    memory_used: bool
    relevant_memories: List[Dict[str, Any]]
    memory_context: str
    saved_memories: List[Dict[str, Any]]
    selected_skills: List[Skill]
    selected_skill_names: List[str]
    skill_prompt: str
    observations: List[Dict[str, Any]]
    evidence_assessment: Dict[str, Any]
    next_action: str
    decision: Dict[str, Any]
    final_answer: str
    output_review: Dict[str, Any]
    stop_reason: str
    harness: AgentHarness
    agent_trace: List[Dict[str, Any]]
    graph_entities: str
    graph_intents: str
    knowledge_context: str
    harness_summary: Dict[str, Any]


def _result_to_observation(result: ToolResult) -> Dict[str, Any]:
    return {
        "name": result.name,
        "content": result.content,
        "entities": result.entities or {},
        "intents": result.intents,
        "sources": result.sources,
        "debug": result.debug or {},
    }


def _format_evidence(observations: List[dict]) -> str:
    if not observations:
        return "没有检索证据。"
    blocks = []
    for index, item in enumerate(observations, start=1):
        blocks.append(
            f"[E{index}] 工具={item.get('name')}\n"
            f"来源={item.get('sources') or '未标注'}\n"
            f"{item.get('content', '')}"
        )
    return "\n\n".join(blocks)


def _collect_compatibility_fields(observations: List[dict]) -> Dict[str, str]:
    entities = ""
    intents = ""
    for item in observations:
        if item.get("entities"):
            entities = json.dumps(item["entities"], ensure_ascii=False)
        if item.get("intents"):
            intents = str(item["intents"])
    return {
        "graph_entities": entities,
        "graph_intents": intents,
        "knowledge_context": _format_evidence(observations),
    }


def build_react_graph(
    llm: QwenClient,
    tools: MedicalRAGTools,
    memory_store: SQLiteMemoryStore,
    skill_registry: SkillRegistry,
):
    """构建 prepare → skills → ReAct/tool/evidence loop → guardrail workflow。"""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "未安装 langgraph，请先执行 `pip install -r requirements.txt`。"
        ) from exc

    def prepare_context_node(state: ReActState) -> ReActState:
        harness = state["harness"]
        username = state.get("username", "").strip()
        saved = (
            save_explicit_memories(username, state["query"], llm, memory_store)
            if username
            else []
        )
        memories = (
            load_relevant_memories(username, state["query"], memory_store)
            if username
            else []
        )
        history = state.get("history", [])
        query = state["query"]
        working_query = query
        correction = "not_needed"
        should_rewrite = _needs_rewrite(query, history)
        if should_rewrite and harness.consume_rewrite():
            previous_query = _last_user_query(history)
            candidate = llm.complete(
                REWRITE_PROMPT.format(
                    history=_format_history(history),
                    query=query,
                )
            ).strip() or query
            working_query = candidate
            correction = "accepted"
            if not _inherits_previous_topic(candidate, previous_query):
                repaired = llm.complete(
                    REWRITE_REPAIR_PROMPT.format(
                        previous_query=previous_query,
                        query=query,
                        candidate=candidate,
                    )
                ).strip()
                if repaired and _inherits_previous_topic(repaired, previous_query):
                    working_query = repaired
                    correction = "llm_repair"
                else:
                    topic = _extract_topic_hint(previous_query)
                    if topic:
                        working_query = f"关于{topic}，{query}"
                        correction = "rule_fallback"
        trace = list(state.get("agent_trace", []))
        trace.extend(harness.trace)
        harness.trace.clear()
        trace.append(
            {
                "node": "prepare_context",
                "saved_memories": [item.to_dict() for item in saved],
                "relevant_memories": [item.to_dict() for item in memories],
                "memory_used": should_rewrite,
                "original_query": query,
                "working_query": working_query,
                "rewrite_correction": correction,
            }
        )
        return {
            "working_query": working_query,
            "memory_used": should_rewrite,
            "saved_memories": [item.to_dict() for item in saved],
            "relevant_memories": [item.to_dict() for item in memories],
            "memory_context": format_memories_for_prompt(memories),
            "agent_trace": trace,
        }

    def select_skills_node(state: ReActState) -> ReActState:
        skills = skill_registry.select(state["working_query"])
        present_memory_types = {
            item.get("memory_type") for item in state.get("relevant_memories", [])
        }
        required_memory_types = {
            memory_type
            for skill in skills
            for memory_type in skill.required_memory_types
        }
        trace = list(state.get("agent_trace", []))
        trace.append(
            {
                "node": "select_skills",
                "skills": [skill.to_dict() for skill in skills],
                "memory_check": {
                    "required_types": sorted(required_memory_types),
                    "present_types": sorted(
                        item for item in present_memory_types if item
                    ),
                    "checked": True,
                },
            }
        )
        return {
            "selected_skills": skills,
            "selected_skill_names": [skill.name for skill in skills],
            "skill_prompt": skill_registry.format_for_prompt(skills),
            "evidence_assessment": assess_evidence(
                state["working_query"],
                [],
                skills,
            ).to_dict(),
            "agent_trace": trace,
        }

    def react_decision_node(state: ReActState) -> ReActState:
        harness = state["harness"]
        remaining = harness.config.max_tool_calls - harness.tool_call_count
        decision = decide_next_action(
            llm=llm,
            query=state["working_query"],
            memory_context=state.get("memory_context", ""),
            skills=state.get("selected_skills", []),
            skill_prompt=state.get("skill_prompt", ""),
            observations=state.get("observations", []),
            assessment=state.get("evidence_assessment", {}),
            remaining_calls=remaining,
        )
        assessment_status = state.get("evidence_assessment", {}).get("status")
        if (
            decision.decision == "final"
            and assessment_status != "sufficient"
            and remaining > 0
        ):
            recommended = state["evidence_assessment"].get(
                "recommended_next_tool",
                "medical_vector_search",
            )
            decision = decision.__class__(
                decision="tool",
                tool_name=recommended,
                arguments={"query": state["working_query"]},
                reason="证据尚不充分，Harness 阻止提前结束",
                source="harness_override",
                raw_response=decision.raw_response,
            )
        trace = list(state.get("agent_trace", []))
        trace.append({"node": "react_decision", **decision.to_dict()})
        return {
            "decision": decision.to_dict(),
            "next_action": decision.decision,
            "agent_trace": trace,
        }

    def execute_tool_node(state: ReActState) -> ReActState:
        harness = state["harness"]
        decision = state["decision"]
        before = len(harness.trace)
        result, status = harness.execute_tool(
            ToolCallRequest(
                name=decision.get("tool_name", ""),
                arguments=decision.get("arguments", {}),
                reason=decision.get("reason", ""),
            ),
            tools.call_tool,
        )
        if result is None and status in {
            "duplicate_tool_call",
            "tool_not_allowed",
            "empty_query",
        }:
            recommended = state.get("evidence_assessment", {}).get(
                "recommended_next_tool",
                "",
            )
            if recommended and recommended != decision.get("tool_name"):
                harness.add_trace(
                    "harness_recovery",
                    failed_reason=status,
                    replacement_tool=recommended,
                )
                result, status = harness.execute_tool(
                    ToolCallRequest(
                        name=recommended,
                        arguments={"query": state["working_query"]},
                        reason="Harness 根据 Evidence Checker 自动修复工具调用",
                    ),
                    tools.call_tool,
                )
        trace = list(state.get("agent_trace", []))
        trace.extend(harness.trace[before:])
        observations = list(state.get("observations", []))
        if result is not None:
            observations.append(_result_to_observation(result))
        return {
            "observations": observations,
            "stop_reason": "" if result is not None else status,
            "next_action": "evidence" if result is not None else "final",
            "agent_trace": trace,
        }

    def evidence_check_node(state: ReActState) -> ReActState:
        assessment = assess_evidence(
            state["working_query"],
            state.get("observations", []),
            state.get("selected_skills", []),
        )
        harness = state["harness"]
        can_continue = (
            assessment.status != "sufficient"
            and harness.tool_call_count < harness.config.max_tool_calls
            and not harness.timed_out()
            and bool(assessment.recommended_next_tool)
        )
        trace = list(state.get("agent_trace", []))
        trace.append(
            {
                "node": "evidence_check",
                **assessment.to_dict(),
                "continue_react": can_continue,
            }
        )
        stop_reason = state.get("stop_reason", "")
        if assessment.status == "sufficient":
            stop_reason = "evidence_sufficient"
        elif not can_continue:
            stop_reason = stop_reason or "evidence_insufficient"
        return {
            "evidence_assessment": assessment.to_dict(),
            "next_action": "react" if can_continue else "final",
            "stop_reason": stop_reason,
            "agent_trace": trace,
        }

    def final_answer_node(state: ReActState) -> ReActState:
        observations = state.get("observations", [])
        evidence = _format_evidence(observations)
        assessment = state.get("evidence_assessment", {})
        evidence_status = str(assessment.get("status", "insufficient"))
        answer_mode = str(assessment.get("answer_mode", "retrieving"))
        risk_level = str(assessment.get("risk_level", "low"))
        if evidence_status == "insufficient" and answer_mode == "insufficient_safe_fallback":
            completion = llm.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": SAFE_FALLBACK_PROMPT.format(
                            query=state["query"],
                        ),
                    },
                ]
            )
            answer = str(
                _message_to_dict(completion.choices[0].message).get("content")
                or ""
            )
        elif evidence_status == "insufficient":
            if risk_level == "critical":
                answer = (
                    "知识库未找到足够证据，但该问题可能涉及紧急情况。"
                    "请立即联系急救或尽快前往急诊，不要自行用药或等待。"
                )
            else:
                answer = (
                    "知识库未找到足够证据，无法安全提供具体诊疗或用药建议。"
                    "请咨询医生或药师。"
                )
        else:
            completion = llm.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": FINAL_PROMPT.format(
                            query=state["query"],
                            skills=state.get("skill_prompt", ""),
                            memories=state.get("memory_context", ""),
                            evidence=evidence,
                            evidence_status=evidence_status,
                        ),
                    },
                ]
            )
            answer = str(
                _message_to_dict(completion.choices[0].message).get("content")
                or ""
            )

        review = review_output(
            answer=answer,
            evidence_text=evidence,
            memories=state.get("relevant_memories", []),
            evidence_status=evidence_status,
            answer_mode=answer_mode,
        )
        repaired = False
        if not review.passed and observations:
            repaired = True
            answer = llm.complete(
                REPAIR_PROMPT.format(
                    violations=", ".join(review.violations),
                    query=state["query"],
                    memories=state.get("memory_context", ""),
                    evidence=evidence,
                    answer=answer,
                )
            ).strip()
            review = review_output(
                answer=answer,
                evidence_text=evidence,
                memories=state.get("relevant_memories", []),
                evidence_status=evidence_status,
                answer_mode=answer_mode,
            )
        if not review.passed:
            answer = "根据已知信息无法安全回答该问题，建议咨询医生或药师。"

        trace = list(state.get("agent_trace", []))
        trace.append(
            {
                "node": "output_guardrail",
                **review.to_dict(),
                "repaired": repaired,
            }
        )
        trace.append(
            {
                "node": "final_answer",
                "content": answer,
                "stop_reason": state.get("stop_reason", ""),
                "evidence_status": evidence_status,
                "answer_mode": answer_mode,
                "risk_level": risk_level,
            }
        )
        return {
            "final_answer": answer,
            "output_review": review.to_dict(),
            "agent_trace": trace,
            "harness_summary": state["harness"].summary(),
            **_collect_compatibility_fields(observations),
        }

    graph = StateGraph(ReActState)
    graph.add_node("prepare_context", prepare_context_node)
    graph.add_node("select_skills", select_skills_node)
    graph.add_node("react_decision", react_decision_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("evidence_check", evidence_check_node)
    graph.add_node("final_answer", final_answer_node)

    graph.set_entry_point("prepare_context")
    graph.add_edge("prepare_context", "select_skills")
    graph.add_edge("select_skills", "react_decision")
    graph.add_conditional_edges(
        "react_decision",
        lambda state: state["next_action"],
        {"tool": "execute_tool", "final": "final_answer"},
    )
    graph.add_conditional_edges(
        "execute_tool",
        lambda state: state["next_action"],
        {"evidence": "evidence_check", "final": "final_answer"},
    )
    graph.add_conditional_edges(
        "evidence_check",
        lambda state: state["next_action"],
        {"react": "react_decision", "final": "final_answer"},
    )
    graph.add_edge("final_answer", END)
    return graph.compile()


def run_react_rag(
    query: str,
    llm: QwenClient,
    tools: MedicalRAGTools,
    history: List[Dict[str, Any]] | None = None,
    username: str = "",
    memory_store: SQLiteMemoryStore | None = None,
    skill_registry: SkillRegistry | None = None,
    harness_config: HarnessConfig | None = None,
) -> ReActState:
    """执行受控 ReAct + Harness + Skills 医疗问答。"""
    store = memory_store or SQLiteMemoryStore()
    registry = skill_registry or SkillRegistry()
    harness = AgentHarness(config=harness_config or HarnessConfig())
    graph = build_react_graph(
        llm,
        tools,
        store,
        registry,
    )
    initial_state: ReActState = {
        "query": query,
        "username": username,
        "history": _trim_history(history),
        "working_query": query,
        "memory_used": False,
        "relevant_memories": [],
        "memory_context": "无相关用户长期记忆。",
        "saved_memories": [],
        "selected_skills": [],
        "selected_skill_names": [],
        "skill_prompt": "",
        "observations": [],
        "evidence_assessment": {},
        "next_action": "",
        "decision": {},
        "final_answer": "",
        "output_review": {},
        "stop_reason": "",
        "harness": harness,
        "agent_trace": [],
    }
    return graph.invoke(initial_state)
