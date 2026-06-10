"""医疗问题检索路由。

规则优先处理高置信度问题，避免每次都调用 LLM 路由：

* graph: 明确的结构化属性/关系查询
* vector: 后果、风险、自愈、预后等开放解释型问题
* hybrid: 同时包含结构化查询和开放解释需求
* none: 非医疗或无需检索

规则无法判断时，再调用 LLM 返回结构化路由结果。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, List

from medical_rag.clients.qwen import QwenClient


VECTOR_PATTERNS: Dict[str, tuple[str, ...]] = {
    "不治疗后果": (
        "不治疗",
        "不想治疗",
        "不治",
        "不管",
        "拖着",
        "放着不管",
        "会怎么样",
        "会怎样",
        "有什么后果",
        "后果",
    ),
    "严重程度与风险": (
        "严重吗",
        "严不严重",
        "会不会严重",
        "会恶化",
        "恶化",
        "危险吗",
        "有危险",
        "影响寿命",
        "会致命",
    ),
    "自愈与预后": (
        "能自愈",
        "会自愈",
        "自己会好",
        "能自己好",
        "多久能好",
        "预后",
        "后遗症",
        "会留疤",
        "留下疤痕",
        "反复发作",
        "会复发",
    ),
}


GRAPH_PATTERNS: Dict[str, tuple[str, ...]] = {
    "症状": ("有哪些症状", "有什么症状", "症状是什么", "临床表现", "表现有哪些"),
    "药品": (
        "吃什么药",
        "吃哪些药",
        "用什么药",
        "用哪些药",
        "擦什么药",
        "擦哪些药",
        "需要什么药",
        "常用药",
        "推荐药品",
    ),
    "检查": ("做什么检查", "需要检查什么", "检查项目", "怎么检查"),
    "科目": ("挂什么科", "属于什么科", "看什么科", "所属科目"),
    "治疗方法": ("怎么治疗", "如何治疗", "治疗方法", "治疗方案", "怎么治"),
    "饮食": ("宜吃什么", "适合吃什么", "不能吃什么", "忌吃什么", "饮食禁忌"),
    "病因": ("什么原因", "为什么会得", "发病原因", "病因是什么", "怎么引起"),
    "预防": ("怎么预防", "如何预防", "预防措施", "避免患上"),
    "治愈概率": ("治愈率", "能治好吗", "能不能治好", "治愈概率"),
    "治疗周期": ("治疗多久", "多久治好", "治疗周期"),
    "并发疾病": ("并发症", "并发疾病", "会引发什么病"),
    "生产商": ("生产商", "厂家", "谁生产的"),
}


ROUTER_PROMPT = """
你是医疗检索路由器。请判断用户问题应该使用哪种检索方式。

路由定义：
- graph：明确查询疾病症状、病因、药品、检查、科室、治疗方法、饮食、预防、治愈率、治疗周期、并发疾病或药品生产商。
- vector：询问不治疗的后果、是否严重、能否自愈、预后、风险、后遗症、开放式医学解释。
- hybrid：用户同时包含 graph 和 vector 两类明确需求。
- none：非医疗问题，或不需要查询医疗知识库。

重要规则：
- 不要因为问题里出现“治疗”二字就选择 graph。
- “不治疗会怎么样”“不想治有什么后果”属于 vector。
- “怎么治疗”“治疗方法是什么”才属于 graph。
- 只输出 JSON，不要解释或 Markdown。

输出格式：
{{"route":"graph|vector|hybrid|none","reason":"简短原因"}}

用户问题：{query}
"""


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    source: str
    vector_matches: List[str]
    graph_matches: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _match_patterns(query: str, patterns: Dict[str, tuple[str, ...]]) -> List[str]:
    matches: List[str] = []
    for category, keywords in patterns.items():
        if any(keyword in query for keyword in keywords):
            matches.append(category)
    return matches


def route_by_rules(query: str) -> RouteDecision | None:
    """用高置信度规则路由；无法判断时返回 None。"""
    normalized = re.sub(r"\s+", "", query)
    vector_matches = _match_patterns(normalized, VECTOR_PATTERNS)
    graph_matches = _match_patterns(normalized, GRAPH_PATTERNS)

    if vector_matches and graph_matches:
        return RouteDecision(
            route="hybrid",
            reason=f"同时命中开放解释规则 {vector_matches} 和结构化查询规则 {graph_matches}",
            source="rule",
            vector_matches=vector_matches,
            graph_matches=graph_matches,
        )
    if vector_matches:
        return RouteDecision(
            route="vector",
            reason=f"命中开放解释/风险规则 {vector_matches}",
            source="rule",
            vector_matches=vector_matches,
            graph_matches=[],
        )
    if graph_matches:
        return RouteDecision(
            route="graph",
            reason=f"命中结构化图谱查询规则 {graph_matches}",
            source="rule",
            vector_matches=[],
            graph_matches=graph_matches,
        )
    return None


def route_query(query: str, llm: QwenClient) -> RouteDecision:
    """规则优先，无法判断时调用 LLM 兜底。"""
    rule_decision = route_by_rules(query)
    if rule_decision is not None:
        return rule_decision

    raw = llm.complete(ROUTER_PROMPT.format(query=query))
    try:
        payload = json.loads(raw)
        route = str(payload.get("route", "vector")).lower()
        if route not in {"graph", "vector", "hybrid", "none"}:
            route = "vector"
        reason = str(payload.get("reason", "LLM 路由兜底"))
    except (json.JSONDecodeError, TypeError):
        route = "vector"
        reason = f"LLM 路由输出无法解析，保守使用 vector；原始输出：{raw[:200]}"
    return RouteDecision(
        route=route,
        reason=reason,
        source="llm",
        vector_matches=[],
        graph_matches=[],
    )
