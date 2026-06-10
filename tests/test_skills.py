"""Skills 加载、选择和 Prompt 格式化测试。"""

from medical_rag.skills.registry import SkillRegistry


def test_skill_registry_loads_markdown_and_selects_relevant_skills():
    registry = SkillRegistry()

    selected = registry.select("我对青霉素过敏，皮炎可以擦哪些药")
    names = [skill.name for skill in selected]

    assert names[0] == "medication_advice"
    assert "disease_treatment" in names
    assert "allergy" in registry.get("medication_advice").required_memory_types
    assert "不得自行补药名" in registry.format_for_prompt(selected)


def test_prognosis_skill_prefers_vector_search():
    registry = SkillRegistry()
    selected = registry.select("皮炎不治疗会有什么后果")

    assert [skill.name for skill in selected] == ["prognosis_and_risk"]
    assert selected[0].preferred_tools == ("medical_vector_search",)
