"""ReAct 决策 Prompt、结构化解析和保守回退。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from medical_rag.clients.qwen import QwenClient
from medical_rag.harness.runtime import ToolCallRequest
from medical_rag.skills.registry import Skill


REACT_PROMPT = """
你是受控医疗检索 Agent。你只能决定下一步调用哪个检索工具，或在证据已经充分时结束检索。

可用工具：
- medical_graph_search：查询疾病、症状、药品、检查、科室、治疗方法等结构化事实
- medical_vector_search：查询开放解释、风险、预后、相似医疗问答和具体文本证据

约束：
- 每轮最多调用一个工具。
- 不得调用列表外工具。
- 不得输出医学答案；最终答案由独立节点生成。
- 图谱只返回“药物治疗”“手术治疗”等泛化词时，必须继续向量检索。
- 不得重复完全相同的工具和参数。
- 优先遵守已加载 Skill 的执行顺序。

当前问题：{query}

相关长期记忆：
{memories}

已加载 Skills：
{skills}

已有 Observation：
{observations}

证据检查：
{assessment}

剩余工具调用预算：{remaining_calls}

只输出 JSON：
调用工具：
{{"decision":"tool","tool_name":"medical_graph_search|medical_vector_search","arguments":{{"query":"检索问题","top_k":5}},"reason":"简短原因"}}

结束检索：
{{"decision":"final","reason":"证据已充分"}}
"""


@dataclass(frozen=True)
class ReActDecision:
    """模型单轮 ReAct 决策。"""

    decision: str
    reason: str
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    source: str = "llm"
    raw_response: str = ""

    def to_tool_request(self) -> ToolCallRequest:
        return ToolCallRequest(
            name=self.tool_name,
            arguments=self.arguments,
            reason=self.reason,
        )

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "source": self.source,
            "raw_response": self.raw_response,
        }


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _format_observations(observations: Iterable[dict]) -> str:
    items = []
    for observation in observations:
        items.append(
            f"- {observation.get('name')}: "
            f"{str(observation.get('content', ''))[:1600]}"
        )
    return "\n".join(items) or "暂无。"


def _fallback_decision(
    query: str,
    skills: List[Skill],
    observations: List[dict],
    assessment: dict,
    raw_response: str,
) -> ReActDecision:
    recommended = str(assessment.get("recommended_next_tool", ""))
    called_tools = [item.get("name") for item in observations]
    if recommended and recommended not in called_tools:
        tool_name = recommended
    else:
        preferred = [
            tool
            for skill in skills
            for tool in skill.preferred_tools
            if tool not in called_tools
        ]
        tool_name = preferred[0] if preferred else "medical_vector_search"
    return ReActDecision(
        decision="tool",
        tool_name=tool_name,
        arguments={"query": query},
        reason="ReAct 输出无法解析，使用 Skill 与证据检查的保守回退",
        source="fallback",
        raw_response=raw_response,
    )


def decide_next_action(
    *,
    llm: QwenClient,
    query: str,
    memory_context: str,
    skills: List[Skill],
    skill_prompt: str,
    observations: List[dict],
    assessment: dict,
    remaining_calls: int,
) -> ReActDecision:
    """调用 LLM 生成下一步，并解析为严格决策对象。"""
    if assessment.get("status") == "sufficient":
        return ReActDecision(
            decision="final",
            reason="Evidence Checker 判定证据充分",
            source="harness",
        )
    if remaining_calls <= 0:
        return ReActDecision(
            decision="final",
            reason="工具调用预算已耗尽",
            source="harness",
        )

    prompt = REACT_PROMPT.format(
        query=query,
        memories=memory_context,
        skills=skill_prompt,
        observations=_format_observations(observations),
        assessment=json.dumps(assessment, ensure_ascii=False),
        remaining_calls=remaining_calls,
    )
    raw = llm.complete(prompt)
    try:
        payload = json.loads(_strip_code_fence(raw))
    except (json.JSONDecodeError, TypeError):
        return _fallback_decision(
            query,
            skills,
            observations,
            assessment,
            raw,
        )

    decision = str(payload.get("decision", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if decision == "final":
        if observations:
            return ReActDecision(
                decision="final",
                reason=reason or "模型结束检索",
                raw_response=raw,
            )
        return _fallback_decision(
            query,
            skills,
            observations,
            assessment,
            raw,
        )

    tool_name = str(payload.get("tool_name", "")).strip()
    arguments = payload.get("arguments") or {}
    if decision != "tool" or not isinstance(arguments, dict):
        return _fallback_decision(
            query,
            skills,
            observations,
            assessment,
            raw,
        )
    arguments.setdefault("query", query)
    return ReActDecision(
        decision="tool",
        tool_name=tool_name,
        arguments=arguments,
        reason=reason or "模型请求补充检索",
        raw_response=raw,
    )
