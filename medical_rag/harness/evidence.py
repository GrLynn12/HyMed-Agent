"""检索证据充分性评估。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

from medical_rag.core.config import settings
from medical_rag.harness.risk import classify_medical_risk
from medical_rag.skills.registry import Skill


EMPTY_MARKERS = (
    "未找到可用",
    "缺少 query",
    "未知工具",
    "知识库异常",
)

GENERIC_GRAPH_MARKERS = {
    "药物治疗",
    "手术治疗",
    "支持治疗",
    "对症治疗",
    "一般治疗",
}


@dataclass(frozen=True)
class EvidenceAssessment:
    """对当前全部 Observation 的判断。"""

    status: str
    reason: str
    recommended_next_tool: str = ""
    recommended_query: str = ""
    answer_mode: str = "retrieving"
    risk_level: str = "low"

    def to_dict(self) -> dict:
        return asdict(self)


def _graph_is_generic(content: str) -> bool:
    cleaned = content.replace("<提示>", "").replace("</提示>", "")
    return any(marker in cleaned for marker in GENERIC_GRAPH_MARKERS) and len(cleaned) < 180


def _vector_has_relevant_result(observation: dict) -> bool:
    results = (observation.get("debug") or {}).get("results") or []
    if not results:
        return bool(observation.get("content")) and not any(
            marker in observation.get("content", "") for marker in EMPTY_MARKERS
        )
    best = results[0]
    rerank_score = best.get("rerank_score")
    if rerank_score is not None:
        return float(rerank_score) >= settings.RERANKER_SCORE_THRESHOLD
    return float(best.get("score", 0)) >= settings.AGENT_VECTOR_SCORE_THRESHOLD


def assess_evidence(
    query: str,
    observations: List[dict],
    skills: List[Skill],
) -> EvidenceAssessment:
    """用透明规则判断是否需要继续检索。"""
    risk = classify_medical_risk(query, skills)
    if not observations:
        preferred = [
            tool for skill in skills for tool in skill.preferred_tools
        ]
        next_tool = preferred[0] if preferred else "medical_vector_search"
        return EvidenceAssessment(
            status="insufficient",
            reason="尚未执行检索工具",
            recommended_next_tool=next_tool,
            recommended_query=query,
            risk_level=risk.level,
        )

    graph_items = [
        item for item in observations if item.get("name") == "medical_graph_search"
    ]
    vector_items = [
        item for item in observations if item.get("name") == "medical_vector_search"
    ]
    graph_useful = any(
        item.get("content")
        and not any(marker in item["content"] for marker in EMPTY_MARKERS)
        and not _graph_is_generic(item["content"])
        for item in graph_items
    )
    graph_generic = any(
        _graph_is_generic(item.get("content", "")) for item in graph_items
    )
    vector_useful = any(_vector_has_relevant_result(item) for item in vector_items)

    skill_names = {skill.name for skill in skills}
    needs_specific_treatment = bool(
        skill_names & {"disease_treatment", "medication_advice"}
    )
    if vector_useful:
        return EvidenceAssessment(
            status="sufficient",
            reason="向量检索返回达到相关性阈值的文本证据",
            answer_mode="grounded",
            risk_level=risk.level,
        )
    if graph_useful and not needs_specific_treatment:
        return EvidenceAssessment(
            status="sufficient",
            reason="图谱返回了可直接回答问题的结构化事实",
            answer_mode="grounded",
            risk_level=risk.level,
        )
    if graph_useful and needs_specific_treatment and not vector_items:
        return EvidenceAssessment(
            status="partial",
            reason="已有图谱事实，但治疗或用药问题需要文本证据补充",
            recommended_next_tool="medical_vector_search",
            recommended_query=query,
            answer_mode="partial",
            risk_level=risk.level,
        )
    if graph_generic and not vector_items:
        return EvidenceAssessment(
            status="partial",
            reason="图谱结果过于泛化，无法支持具体回答",
            recommended_next_tool="medical_vector_search",
            recommended_query=query,
            answer_mode="partial",
            risk_level=risk.level,
        )
    if graph_useful:
        return EvidenceAssessment(
            status="partial",
            reason="图谱有部分事实，但向量证据未达到阈值",
            recommended_next_tool="medical_vector_search",
            recommended_query=query,
            answer_mode="partial",
            risk_level=risk.level,
        )
    if graph_items and not graph_useful and not vector_items:
        return EvidenceAssessment(
            status="insufficient",
            reason="图谱未命中，继续使用向量检索",
            recommended_next_tool="medical_vector_search",
            recommended_query=query,
            risk_level=risk.level,
        )
    if not graph_items:
        return EvidenceAssessment(
            status="insufficient",
            reason="当前检索未获得足够证据",
            recommended_next_tool="medical_graph_search",
            recommended_query=query,
            risk_level=risk.level,
        )
    return EvidenceAssessment(
        status="insufficient",
        reason="图谱和向量检索均未获得可靠证据",
        answer_mode=(
            "insufficient_safe_fallback"
            if risk.level == "low"
            else "insufficient_refuse"
        ),
        risk_level=risk.level,
    )
