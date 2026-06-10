"""医疗问题风险分类，用于控制无证据时的回答边界。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from medical_rag.skills.registry import Skill


CRITICAL_SIGNALS = (
    "胸痛",
    "胸口压榨",
    "呼吸困难",
    "喘不上气",
    "说话不清",
    "一侧手脚没力",
    "偏瘫",
    "昏迷",
    "抽搐",
    "大出血",
    "呕血",
    "便血",
    "自杀",
    "轻生",
    "过量服药",
)

HIGH_RISK_SIGNALS = (
    "药",
    "剂量",
    "停药",
    "一起吃",
    "同时吃",
    "能不能吃",
    "孕",
    "儿童",
    "婴儿",
    "过敏",
    "手术",
    "治疗",
    "怎么办",
    "严重吗",
)


@dataclass(frozen=True)
class RiskAssessment:
    """问题风险等级和触发依据。"""

    level: str
    reason: str
    matched_signals: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def classify_medical_risk(
    query: str,
    skills: Iterable[Skill] = (),
) -> RiskAssessment:
    """按透明规则将问题分为 critical/high/low。"""
    critical = tuple(signal for signal in CRITICAL_SIGNALS if signal in query)
    if critical:
        return RiskAssessment(
            level="critical",
            reason="命中急症或人身安全信号",
            matched_signals=critical,
        )

    high = tuple(signal for signal in HIGH_RISK_SIGNALS if signal in query)
    skill_names = {skill.name for skill in skills}
    if high or skill_names & {"medication_advice", "disease_treatment"}:
        return RiskAssessment(
            level="high",
            reason="涉及个体化治疗、用药或特殊人群",
            matched_signals=high,
        )

    return RiskAssessment(
        level="low",
        reason="未命中急症、具体治疗或用药风险信号",
    )
