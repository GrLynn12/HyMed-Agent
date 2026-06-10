"""医疗回答输出 Guardrail。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, List


DIAGNOSIS_PATTERNS = (
    "你患有",
    "你得了",
    "可以确诊为",
    "确诊为",
)

DOSAGE_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg|g|ml|mL|片|粒|毫克|克|毫升|次/日|次每天)",
    re.IGNORECASE,
)

UNSAFE_FALLBACK_PATTERNS = (
    "建议服用",
    "可以服用",
    "建议使用",
    "可以使用",
    "自行停药",
    "自行加量",
)


@dataclass(frozen=True)
class OutputReview:
    """最终回答的安全检查结果。"""

    passed: bool
    violations: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_allergens(memories: Iterable[dict]) -> List[str]:
    allergens = []
    for memory in memories:
        if memory.get("memory_type") != "allergy":
            continue
        content = str(memory.get("content", ""))
        match = re.search(r"用户对(.+?)过敏", content)
        if match:
            allergens.append(match.group(1).strip())
    return allergens


def review_output(
    *,
    answer: str,
    evidence_text: str,
    memories: List[dict],
    evidence_status: str,
    answer_mode: str = "grounded",
) -> OutputReview:
    """检查直接诊断、无证据剂量和明确过敏冲突。"""
    violations: List[str] = []
    if any(pattern in answer for pattern in DIAGNOSIS_PATTERNS):
        violations.append("direct_diagnosis")

    answer_doses = set(DOSAGE_PATTERN.findall(answer))
    evidence_doses = set(DOSAGE_PATTERN.findall(evidence_text))
    if answer_doses - evidence_doses:
        violations.append("unsupported_dosage")

    for allergen in _extract_allergens(memories):
        unsafe_patterns = (
            f"建议使用{allergen}",
            f"可以使用{allergen}",
            f"建议服用{allergen}",
            f"可以服用{allergen}",
            f"推荐{allergen}",
        )
        if any(pattern in answer for pattern in unsafe_patterns):
            violations.append(f"allergy_conflict:{allergen}")

    if (
        answer_mode == "insufficient_safe_fallback"
        and any(pattern in answer for pattern in UNSAFE_FALLBACK_PATTERNS)
    ):
        violations.append("unsafe_fallback_advice")

    if evidence_status == "insufficient" and answer_mode not in {
        "insufficient_safe_fallback",
        "insufficient_refuse",
    } and answer.strip() not in {
        "根据已知信息无法回答该问题。",
        "根据已知信息无法回答该问题",
    }:
        violations.append("answer_without_sufficient_evidence")

    return OutputReview(passed=not violations, violations=violations)
