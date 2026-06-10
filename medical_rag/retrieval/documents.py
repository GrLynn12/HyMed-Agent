"""外部医疗语料加载与切分。

传统 FAISS RAG 使用独立外部语料，不默认复用 Neo4j 图谱同源的
``medical_new_2.json``。这样 Hybrid RAG 才能形成互补：

* Neo4j：结构化、精确知识
* FAISS：开放问答/文档语料的语义补充

支持格式：

* ``jsonl_qa``: 每行 ``{"question": "...", "answer": "..."}``，适合 Huatuo-26M 类 QA 数据
* ``jsonl_text``: 每行 ``{"text": "..."}``
* ``txt``: 普通文本文件
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable, List

from medical_rag.core.config import settings


@dataclass(frozen=True)
class MedicalDocument:
    """可写入向量库的一段外部医疗文本。"""

    content: str
    source: str
    disease: str
    section: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MedicalDocument":
        return cls(
            content=str(data.get("content", "")),
            source=str(data.get("source", "")),
            disease=str(data.get("disease", "")),
            section=str(data.get("section", "")),
        )


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """轻量中文友好的字符切分，避免为切块额外引入 LangChain。"""
    text = " ".join(str(text).split())
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _first_non_empty(record: dict, names: tuple[str, ...]) -> str:
    for name in names:
        value = record.get(name)
        if value:
            return _normalize_text_value(value)
    return ""


def _normalize_text_value(value) -> str:
    """兼容 Huatuo 等数据集中 list / 嵌套 list 形式的 QA 字段。"""
    if isinstance(value, list):
        flattened = []
        for item in value:
            if isinstance(item, list):
                flattened.extend(str(x).strip() for x in item if str(x).strip())
            elif str(item).strip():
                flattened.append(str(item).strip())
        return "；".join(flattened)
    return str(value).strip()


def load_external_corpus(
    corpus_path: str | None = None,
    corpus_format: str | None = None,
    chunk_size: int = 500,
    overlap: int = 80,
) -> List[MedicalDocument]:
    """加载外部医疗语料并返回文档片段。"""
    path = corpus_path or settings.TRADITIONAL_RAG_CORPUS_PATH
    fmt = corpus_format or settings.TRADITIONAL_RAG_CORPUS_FORMAT
    if not os.path.exists(path):
        raise FileNotFoundError(f"外部 RAG 语料不存在: {path}")
    if fmt == "jsonl_qa":
        return _load_jsonl_qa(path, chunk_size, overlap)
    if fmt == "jsonl_text":
        return _load_jsonl_text(path, chunk_size, overlap)
    if fmt == "txt":
        return _load_txt(path, chunk_size, overlap)
    raise ValueError(f"不支持的外部语料格式: {fmt}")


def _load_jsonl_qa(path: str, chunk_size: int, overlap: int) -> List[MedicalDocument]:
    documents: List[MedicalDocument] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = _first_non_empty(
                record,
                ("question", "questions", "query", "instruction", "input"),
            )
            answer = _first_non_empty(
                record,
                ("answer", "answers", "response", "output", "target"),
            )
            if not question and not answer:
                continue
            disease = _first_non_empty(record, ("disease", "disease_name", "name")) or "外部医疗语料"
            text = f"问题：{question}\n回答：{answer}"
            for idx, chunk in enumerate(split_text(text, chunk_size, overlap), start=1):
                documents.append(
                    MedicalDocument(
                        content=chunk,
                        source=os.path.basename(path),
                        disease=disease,
                        section=f"QA#{line_no}.{idx}",
                    )
                )
    return documents


def _load_jsonl_text(path: str, chunk_size: int, overlap: int) -> List[MedicalDocument]:
    documents: List[MedicalDocument] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = _first_non_empty(record, ("text", "content", "document"))
            if not text:
                continue
            disease = _first_non_empty(record, ("disease", "disease_name", "name")) or "外部医疗语料"
            for idx, chunk in enumerate(split_text(text, chunk_size, overlap), start=1):
                documents.append(
                    MedicalDocument(
                        content=chunk,
                        source=os.path.basename(path),
                        disease=disease,
                        section=f"TEXT#{line_no}.{idx}",
                    )
                )
    return documents


def _load_txt(path: str, chunk_size: int, overlap: int) -> List[MedicalDocument]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [
        MedicalDocument(
            content=chunk,
            source=os.path.basename(path),
            disease="外部医疗语料",
            section=f"TXT#{idx}",
        )
        for idx, chunk in enumerate(split_text(text, chunk_size, overlap), start=1)
    ]


def format_documents_for_prompt(documents: Iterable[MedicalDocument]) -> str:
    """把检索结果整理成 prompt 友好的上下文。"""
    parts = []
    for idx, doc in enumerate(documents, start=1):
        parts.append(
            f"文本片段{idx}（来源:{doc.source}；主题:{doc.disease}；栏目:{doc.section}）：\n"
            f"{doc.content}"
        )
    return "\n\n".join(parts)
