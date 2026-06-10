"""Hybrid RAG 工具封装。

这里的工具既可以被 LangGraph agent 节点通过 OpenAI tool calling 调用，也可以在测试中
直接调用。工具内部仍然复用现有 NER、意图路由和 KGClient。
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from medical_rag import ner
from medical_rag.clients.neo4j import KGClient
from medical_rag.clients.qwen import QwenClient
from medical_rag.core.config import settings
from medical_rag.retrieval.documents import format_documents_for_prompt
from medical_rag.retrieval.graph_intents import execute_intents
from medical_rag.retrieval.vector_store import MedicalVectorStore

logger = logging.getLogger(__name__)


GRAPH_INTENT_PROMPT = """
你是医疗知识图谱的结构化意图识别器。只有用户明确查询下面固定类别时才选择。

固定类别：
["查询疾病简介","查询疾病病因","查询疾病预防措施","查询疾病治疗周期","查询治愈概率","查询疾病易感人群","查询疾病所需药品","查询疾病宜吃食物","查询疾病忌吃食物","查询疾病所需检查项目","查询疾病所属科目","查询疾病的症状","查询疾病的治疗方法","查询疾病的并发疾病","查询药品的生产商"]

规则：
- 如果问题无法准确映射到固定类别，输出 []，不要强行选择近似类别。
- “不治疗会怎么样”“有什么后果”“严重吗”“能自愈吗”“预后如何”不属于固定类别，应输出 []。
- 只有“怎么治疗”“治疗方法是什么”才选择“查询疾病的治疗方法”。
- 只有明确询问“症状/表现”才选择“查询疾病的症状”。
- 只有明确询问“并发症/会引发什么疾病”才选择“查询疾病的并发疾病”。

只输出 Python 列表格式，不要输出额外解释，最多选择 5 个类别。
用户问题：{query}
"""


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "medical_graph_search",
            "description": "查询医疗知识图谱中的结构化知识，适合明确疾病、症状、药品、检查、治疗、饮食等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户原始医疗问题。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "medical_vector_search",
            "description": "从医疗文本向量库检索相关片段，适合模糊问题、补充解释、图谱没有命中或需要长文本依据的问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户原始医疗问题。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的文本片段数量，默认 5。",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


@dataclass
class ToolResult:
    """工具执行结果。"""

    name: str
    content: str
    entities: Dict[str, str] | None = None
    intents: str = ""
    sources: str = ""
    debug: Dict[str, Any] | None = None


class MedicalRAGTools:
    """当前会话可用的医疗检索工具集合。"""

    def __init__(
        self,
        llm: QwenClient,
        kg: KGClient,
        bert_model,
        bert_tokenizer,
        rule,
        tfidf_r,
        device,
        idx2tag,
        vector_store: MedicalVectorStore | None = None,
    ) -> None:
        self.llm = llm
        self.kg = kg
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.rule = rule
        self.tfidf_r = tfidf_r
        self.device = device
        self.idx2tag = idx2tag
        self.vector_store = vector_store or MedicalVectorStore()

    def medical_graph_search(self, query: str) -> ToolResult:
        """NER + 意图识别 + Neo4j 图谱检索。"""
        entities = ner.get_ner_result(
            self.bert_model,
            self.bert_tokenizer,
            query,
            self.rule,
            self.tfidf_r,
            self.device,
            self.idx2tag,
        )
        intent_response = self.llm.complete(GRAPH_INTENT_PROMPT.format(query=query))
        prompt_parts: List[str] = []
        if "疾病症状" in entities and "疾病" not in entities:
            diseases = self.kg.get_diseases_by_symptom(entities["疾病症状"])
            if diseases:
                entities["疾病"] = random.choice(diseases)
                prompt_parts.append(
                    f"用户有{entities['疾病症状']}的情况，知识库推测其可能相关疾病包括："
                    f"{'、'.join(diseases)}。这只是知识库推测，不可作为诊断。"
                )

        intent_prompt, intent_names = execute_intents(intent_response, entities, self.kg)
        prompt_parts.append(intent_prompt)
        content = "\n".join(part for part in prompt_parts if part).strip()
        if not content:
            content = "图谱检索未找到可用结构化知识。"
        return ToolResult(
            name="medical_graph_search",
            content=content,
            entities=entities,
            intents="、".join(intent_names),
            sources="Neo4j 医疗知识图谱",
            debug={
                "query": query,
                "entities": entities,
                "intent_raw_response": intent_response,
                "intent_names": intent_names,
                "content": content,
            },
        )

    def medical_vector_search(self, query: str, top_k: int | None = None) -> ToolResult:
        """FAISS 文本向量检索。"""
        k = top_k or settings.VECTOR_TOP_K
        detailed_results = self.vector_store.search_with_details(query, top_k=k)
        documents = [item.document for item in detailed_results]
        raw_results = [
            item.to_debug_dict(rank)
            for rank, item in enumerate(detailed_results, start=1)
        ]
        if not detailed_results:
            content = (
                "向量检索未找到可用文本片段。若尚未构建索引，请先运行 "
                "`python -m scripts.build_vector_index`。"
            )
            sources = ""
        else:
            content = format_documents_for_prompt(documents)
            sources = "；".join(
                f"{doc.source}/{doc.disease}/{doc.section}" for doc in documents
            )
        return ToolResult(
            name="medical_vector_search",
            content=content,
            sources=sources,
            debug={
                "query": query,
                "top_k": k,
                "reranker_enabled": self.vector_store.reranker_enabled,
                "reranker_model": self.vector_store.reranker_model_name,
                "results": raw_results,
            },
        )

    def call_tool(self, name: str, arguments: dict) -> ToolResult:
        """按 OpenAI tool call 名称执行工具。"""
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolResult(name=name, content="工具调用缺少 query 参数。")
        if name == "medical_graph_search":
            return self.medical_graph_search(query)
        if name == "medical_vector_search":
            top_k = arguments.get("top_k")
            return self.medical_vector_search(query, top_k=int(top_k) if top_k else None)
        logger.warning("未知工具调用: %s", name)
        return ToolResult(name=name, content=f"未知工具：{name}")
