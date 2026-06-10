"""医疗输出 Guardrail 测试。"""

from medical_rag.harness.guardrails import review_output


def test_guardrail_blocks_unsupported_dosage_and_direct_diagnosis():
    review = review_output(
        answer="你患有皮炎，可以每日使用10mg。",
        evidence_text="证据仅说明皮炎需要根据类型治疗。",
        memories=[],
        evidence_status="sufficient",
    )

    assert review.passed is False
    assert "direct_diagnosis" in review.violations
    assert "unsupported_dosage" in review.violations


def test_guardrail_blocks_recommending_known_allergen():
    review = review_output(
        answer="建议使用青霉素。",
        evidence_text="青霉素属于抗菌药物。",
        memories=[
            {"memory_type": "allergy", "content": "用户对青霉素过敏"}
        ],
        evidence_status="sufficient",
    )

    assert review.passed is False
    assert "allergy_conflict:青霉素" in review.violations


def test_guardrail_allows_allergy_warning():
    review = review_output(
        answer="你有青霉素过敏史，应避免自行使用青霉素。",
        evidence_text="青霉素属于抗菌药物。",
        memories=[
            {"memory_type": "allergy", "content": "用户对青霉素过敏"}
        ],
        evidence_status="sufficient",
    )

    assert review.passed is True


def test_guardrail_blocks_specific_advice_in_safe_fallback():
    review = review_output(
        answer="知识库未检索到足够证据，建议服用某种药物。",
        evidence_text="没有检索证据。",
        memories=[],
        evidence_status="insufficient",
        answer_mode="insufficient_safe_fallback",
    )

    assert review.passed is False
    assert "unsafe_fallback_advice" in review.violations
