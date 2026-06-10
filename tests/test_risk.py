"""医疗问题风险分类测试。"""

from medical_rag.harness.risk import classify_medical_risk
from medical_rag.skills.registry import SkillRegistry


def test_emergency_symptoms_are_critical():
    assessment = classify_medical_risk("突然一侧手脚没力，说话不清楚")

    assert assessment.level == "critical"


def test_medication_question_is_high_risk():
    skills = SkillRegistry().select("阿司匹林和华法林这两种药能一起吃吗")
    assessment = classify_medical_risk(
        "阿司匹林和华法林这两种药能一起吃吗",
        skills,
    )

    assert assessment.level == "high"


def test_general_health_education_is_low_risk():
    assessment = classify_medical_risk("为什么规律睡眠很重要")

    assert assessment.level == "low"
