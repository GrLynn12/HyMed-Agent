"""从 SKILL.md 加载医疗任务策略，并为问题选择最相关的 Skills。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


SKILL_RULES: Dict[str, tuple[str, ...]] = {
    "disease_treatment": (
        "治疗",
        "怎么治",
        "怎么办",
        "治疗方案",
        "治疗方法",
        "擦什么",
        "擦哪些",
        "用什么药",
        "用哪些药",
    ),
    "medication_advice": (
        "药",
        "药膏",
        "用药",
        "服用",
        "剂量",
        "副作用",
        "禁忌",
        "过敏",
    ),
    "prognosis_and_risk": (
        "不治疗",
        "不治",
        "后果",
        "严重",
        "自愈",
        "预后",
        "复发",
        "后遗症",
        "风险",
    ),
}


@dataclass(frozen=True)
class Skill:
    """一个从磁盘加载的医疗任务策略。"""

    name: str
    description: str
    instructions: str
    preferred_tools: tuple[str, ...]
    required_memory_types: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "preferred_tools": list(self.preferred_tools),
            "required_memory_types": list(self.required_memory_types),
        }


class SkillRegistry:
    """管理项目内置 Skills，并根据透明规则选择。"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent
        self._skills = self._load_all()

    def _load_all(self) -> Dict[str, Skill]:
        skills = {}
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            skill = self._parse_skill(skill_file)
            skills[skill.name] = skill
        return skills

    @staticmethod
    def _parse_list(value: str) -> tuple[str, ...]:
        value = value.strip().strip("[]")
        return tuple(
            item.strip().strip("\"'")
            for item in value.split(",")
            if item.strip()
        )

    def _parse_skill(self, path: Path) -> Skill:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            raise ValueError(f"Skill 缺少 front matter: {path}")
        _empty, metadata_text, instructions = text.split("---", 2)
        metadata = {}
        for line in metadata_text.strip().splitlines():
            key, separator, value = line.partition(":")
            if separator:
                metadata[key.strip()] = value.strip()
        name = metadata.get("name", path.parent.name)
        return Skill(
            name=name,
            description=metadata.get("description", ""),
            instructions=instructions.strip(),
            preferred_tools=self._parse_list(
                metadata.get("preferred_tools", "")
            ),
            required_memory_types=self._parse_list(
                metadata.get("required_memory", "")
            ),
        )

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def select(self, query: str, limit: int = 2) -> List[Skill]:
        """按规则选择 Skills；无命中时返回空列表。"""
        scored = []
        for name, keywords in SKILL_RULES.items():
            matched = [keyword for keyword in keywords if keyword in query]
            if (
                name == "disease_treatment"
                and matched == ["治疗"]
                and any(term in query for term in ("不治疗", "不想治疗"))
            ):
                matched = []
            score = len(matched)
            if score and name in self._skills:
                scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self._skills[name] for _score, name in scored[:limit]]

    @staticmethod
    def format_for_prompt(skills: Iterable[Skill]) -> str:
        blocks = []
        for skill in skills:
            blocks.append(
                f"## Skill: {skill.name}\n"
                f"用途：{skill.description}\n"
                f"建议工具：{', '.join(skill.preferred_tools)}\n"
                f"{skill.instructions}"
            )
        return "\n\n".join(blocks) or "未加载专用 Skill，按通用医疗检索规则执行。"
