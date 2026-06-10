"""长期记忆提取、过滤与 prompt 格式化。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from medical_rag.clients.qwen import QwenClient
from medical_rag.core.config import settings
from medical_rag.memory.store import (
    ALLOWED_MEMORY_TYPES,
    MemoryRecord,
    SQLiteMemoryStore,
)


MEMORY_EXTRACTION_PROMPT = """
你是用户长期记忆提取器。只从“用户原话”中提取用户明确陈述、适合跨会话保留的稳定事实。

允许的类型只有：
- medical_history：用户明确陈述的既往病史、确诊疾病、长期健康状况
- allergy：用户明确陈述的药物、食物或其他过敏
- medication：用户明确陈述正在服用、长期服用或医生开具的药物
- preference：用户明确陈述的回答偏好，如希望简洁、详细、使用中文

禁止：
- 不得保存模型推测、可能诊断、检索结果或助手回答
- 不得把“我是不是得了某病”“可能是某病”保存为病史
- 不保存一次性症状、普通问题、年龄住址等无关隐私
- 不得扩写用户没有说过的信息

只输出 JSON 数组，无可保存事实时输出 []：
[{{"memory_type":"allergy","content":"用户对青霉素过敏"}}]

用户原话：{query}
"""


EXPLICIT_MEMORY_SIGNALS = (
    "我有",
    "我患有",
    "我得过",
    "我确诊",
    "我的病史",
    "既往有",
    "过敏",
    "我在吃",
    "我正在吃",
    "我服用",
    "我正在服用",
    "医生给我开",
    "长期服用",
    "请简洁",
    "回答简洁",
    "详细一点",
    "回答详细",
    "用中文",
    "不要太长",
)

MEDICATION_QUERY_SIGNALS = (
    "药",
    "服用",
    "用药",
    "治疗",
    "剂量",
    "副作用",
    "禁忌",
)

TYPE_LABELS = {
    "medical_history": "病史",
    "allergy": "过敏",
    "medication": "用药",
    "preference": "偏好",
}


def should_extract_memory(query: str) -> bool:
    normalized = re.sub(r"\s+", "", query)
    return any(signal in normalized for signal in EXPLICIT_MEMORY_SIGNALS)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_memories(query: str, llm: QwenClient) -> List[Dict[str, str]]:
    """从用户原话提取长期事实；无显式信号时不调用 LLM。"""
    if not should_extract_memory(query):
        return []
    raw = llm.complete(MEMORY_EXTRACTION_PROMPT.format(query=query))
    try:
        payload = json.loads(_strip_code_fence(raw))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, list):
        return []
    memories: List[Dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        memory_type = str(item.get("memory_type", "")).strip()
        content = str(item.get("content", "")).strip()
        if memory_type in ALLOWED_MEMORY_TYPES and content:
            memories.append({"memory_type": memory_type, "content": content})
    return memories


def _query_terms(text: str) -> set[str]:
    normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text.lower())
    terms = set()
    for size in (2, 3, 4):
        terms.update(
            normalized[idx : idx + size]
            for idx in range(max(len(normalized) - size + 1, 0))
        )
    return terms


def filter_relevant_memories(
    query: str,
    memories: List[MemoryRecord],
    limit: int | None = None,
) -> List[MemoryRecord]:
    """用透明规则筛选当前问题相关的长期记忆。"""
    max_items = settings.MEMORY_MAX_RELEVANT if limit is None else max(limit, 0)
    if max_items == 0:
        return []
    query_terms = _query_terms(query)
    medication_related = any(signal in query for signal in MEDICATION_QUERY_SIGNALS)
    scored = []
    for record in memories:
        memory_terms = _query_terms(record.content)
        overlap = len(query_terms & memory_terms)
        score = float(overlap)
        if record.memory_type == "preference":
            score += 100
        elif medication_related and record.memory_type in {"allergy", "medication"}:
            score += 20
        elif record.memory_type == "medical_history" and overlap:
            score += 10
        elif record.memory_type == "allergy" and overlap:
            score += 10
        if score > 0:
            scored.append((score, record))
    scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
    return [record for _score, record in scored[:max_items]]


def load_relevant_memories(
    username: str,
    query: str,
    store: SQLiteMemoryStore,
) -> List[MemoryRecord]:
    return filter_relevant_memories(query, store.list_memories(username))


def format_memories_for_prompt(memories: List[MemoryRecord]) -> str:
    if not memories:
        return "无相关用户长期记忆。"
    return "\n".join(
        f"- [{TYPE_LABELS.get(record.memory_type, record.memory_type)}] {record.content}"
        for record in memories
    )


def save_explicit_memories(
    username: str,
    query: str,
    llm: QwenClient,
    store: SQLiteMemoryStore,
) -> List[MemoryRecord]:
    extracted = extract_memories(query, llm)
    if not extracted:
        return []
    return store.add_memories(username, extracted)
