"""Evidence Checker 对泛化图谱结果和向量分数的测试。"""

from medical_rag.harness.evidence import assess_evidence
from medical_rag.skills.registry import SkillRegistry


def test_generic_graph_treatment_requires_vector_follow_up():
    skills = SkillRegistry().select("皮炎应该怎么治疗")
    assessment = assess_evidence(
        "皮炎应该怎么治疗",
        [
            {
                "name": "medical_graph_search",
                "content": "药物治疗",
                "debug": {},
            }
        ],
        skills,
    )

    assert assessment.status == "partial"
    assert assessment.recommended_next_tool == "medical_vector_search"


def test_relevant_vector_result_is_sufficient():
    skills = SkillRegistry().select("皮炎不治疗会怎样")
    assessment = assess_evidence(
        "皮炎不治疗会怎样",
        [
            {
                "name": "medical_vector_search",
                "content": "不治疗可能反复发作。",
                "debug": {"results": [{"score": 0.8}]},
            }
        ],
        skills,
    )

    assert assessment.status == "sufficient"
    assert assessment.answer_mode == "grounded"


def test_empty_graph_result_falls_back_to_vector_search():
    skills = SkillRegistry().select("阿司匹林和华法林这两种药能一起吃吗")
    assessment = assess_evidence(
        "阿司匹林和华法林这两种药能一起吃吗",
        [
            {
                "name": "medical_graph_search",
                "content": "图谱检索未找到可用结构化知识。",
                "debug": {},
            }
        ],
        skills,
    )

    assert assessment.status == "insufficient"
    assert assessment.reason == "图谱未命中，继续使用向量检索"
    assert assessment.recommended_next_tool == "medical_vector_search"


def test_low_risk_insufficient_evidence_allows_safe_fallback():
    assessment = assess_evidence(
        "为什么规律睡眠很重要",
        [
            {
                "name": "medical_vector_search",
                "content": "未找到可用医疗文本片段。",
                "debug": {"results": []},
            },
            {
                "name": "medical_graph_search",
                "content": "图谱检索未找到可用结构化知识。",
                "debug": {},
            },
        ],
        [],
    )

    assert assessment.status == "insufficient"
    assert assessment.answer_mode == "insufficient_safe_fallback"
    assert assessment.risk_level == "low"


def test_high_risk_insufficient_evidence_requires_refusal():
    skills = SkillRegistry().select("两种药能一起吃吗")
    assessment = assess_evidence(
        "两种药能一起吃吗",
        [
            {
                "name": "medical_vector_search",
                "content": "未找到可用医疗文本片段。",
                "debug": {"results": []},
            },
            {
                "name": "medical_graph_search",
                "content": "图谱检索未找到可用结构化知识。",
                "debug": {},
            },
        ],
        skills,
    )

    assert assessment.answer_mode == "insufficient_refuse"
    assert assessment.risk_level == "high"
