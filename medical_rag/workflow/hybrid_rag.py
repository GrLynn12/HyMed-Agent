"""LangGraph 规则优先 Hybrid RAG pipeline。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, TypedDict

from medical_rag.clients.qwen import QwenClient
from medical_rag.core.config import settings
from medical_rag.memory.service import (
    format_memories_for_prompt,
    load_relevant_memories,
    save_explicit_memories,
)
from medical_rag.memory.store import SQLiteMemoryStore
from medical_rag.retrieval.query_router import route_query
from medical_rag.retrieval.tools import MedicalRAGTools, ToolResult


SYSTEM_PROMPT = """
你是医疗知识问答助手。请根据检索工具返回的知识回答用户问题。

规则：
- 优先使用与用户问题最接近的检索证据，不要用泛泛常识替代具体证据。
- vector 结果来自独立医疗问答语料，适合后果、风险、自愈、预后和开放解释。
- graph 结果来自 Neo4j 医疗知识图谱，适合症状、药品、检查、科室、治疗方法等结构化事实。
- 如果 vector 中存在与用户问题高度相似的原始问答，应优先保留其中的关键结论。
- 不要编造检索证据之外的医学事实。
- 不要做确诊，不要替代医生诊疗。
- 用户长期记忆只代表用户过去明确陈述的信息，可能不完整或已发生变化；仅在与当前问题相关时用于个性化回答。
- 不要把用户长期记忆当作医学诊断或检索证据，不要据此擅自改变药物和剂量。
- 如果没有可靠证据，回答“根据已知信息无法回答该问题”。
"""

REWRITE_PROMPT = """
你是医疗对话中的问题改写器。请结合最近对话，把当前问题改写成无需历史也能理解的独立问题。

要求：
- 只补全“它、这个、那、这种病”等指代和省略的信息。
- 当前问题省略疾病或药品名称时，必须从最近一条用户问题中补回明确主题。
- 改写结果必须包含具体疾病、症状或药品主题，不能只把“药”改成“药膏”。
- 不添加用户没有表达的疾病、症状或诊断。
- 保留当前问题的真实意图。
- 只输出改写后的问题，不要解释，不要加引号。

最近对话：
{history}

当前问题：{query}
"""

REWRITE_REPAIR_PROMPT = """
上一次改写仍然缺少上一轮对话的明确主题，不能独立理解。请重新改写。

上一轮用户问题：{previous_query}
当前问题：{query}
不合格的改写：{candidate}

要求：
- 必须把上一轮用户问题中的具体疾病、症状或药品名称补入当前问题。
- 只保留当前问题的查询意图，不要带入上一轮已经问过的旧意图。
- 不添加对话中没有出现的信息。
- 只输出一个可独立理解的问题。
"""

CONTEXT_DEPENDENT_PATTERNS = (
    "它",
    "这个",
    "这种",
    "那个",
    "那",
    "上述",
    "刚才",
    "前面",
    "还有呢",
    "然后呢",
    "怎么办呢",
)

GENERIC_FOLLOW_UPS = {
    "怎么治疗",
    "怎么治",
    "怎么办",
    "严重吗",
    "能治好吗",
    "吃什么药",
    "要做什么检查",
    "需要检查吗",
    "会好吗",
    "多久能好",
    "为什么",
}

FOLLOW_UP_PATTERNS = (
    "应该擦哪些药",
    "该擦哪些药",
    "擦哪些药",
    "擦什么药",
    "应该用哪些药",
    "该用哪些药",
    "用哪些药",
    "用什么药",
    "应该吃哪些药",
    "该吃哪些药",
    "吃哪些药",
    "吃什么药",
    "该怎么治疗",
    "应该怎么治疗",
    "该怎么治",
    "应该怎么治",
)

REWRITE_STOP_TERMS = {
    "可以",
    "应该",
    "哪些",
    "什么",
    "怎么",
    "如何",
    "治疗",
    "不治",
    "药膏",
    "用药",
    "需要",
    "是否",
}

TOPIC_BOUNDARY_PATTERNS = (
    "可以不治疗",
    "能不能不治疗",
    "不治疗",
    "应该怎么",
    "该怎么",
    "怎么",
    "如何",
    "会不会",
    "有哪些",
    "有什么",
    "是什么",
    "严重吗",
    "能治好",
    "需要",
)


class QAState(TypedDict, total=False):
    query: str
    username: str
    rewritten_query: str
    history: List[Dict[str, str]]
    memory_used: bool
    relevant_memories: List[Dict[str, Any]]
    long_term_memory_context: str
    saved_memories: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    route_decision: Dict[str, Any]
    tool_results: List[Dict[str, Any]]
    final_answer: str
    graph_entities: str
    graph_intents: str
    knowledge_context: str
    agent_trace: List[Dict[str, Any]]


def _trim_history(history: List[Dict[str, Any]] | None) -> List[Dict[str, str]]:
    """保留最近 N 轮对话，只传递 role/content。"""
    if not history:
        return []
    limit = max(settings.MEMORY_RECENT_TURNS, 0) * 2
    if limit == 0:
        return []
    cleaned = [
        {
            "role": str(item.get("role", "")),
            "content": str(item.get("content", "")),
        }
        for item in history
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    return cleaned[-limit:]


def _needs_rewrite(query: str, history: List[Dict[str, str]]) -> bool:
    if not history:
        return False
    normalized = re.sub(r"\s+", "", query)
    if any(pattern in normalized for pattern in CONTEXT_DEPENDENT_PATTERNS):
        return True
    if any(pattern in normalized for pattern in FOLLOW_UP_PATTERNS):
        return True
    return normalized.rstrip("？?。") in GENERIC_FOLLOW_UPS


def _format_history(history: List[Dict[str, str]]) -> str:
    role_names = {"user": "用户", "assistant": "助手"}
    return "\n".join(
        f"{role_names.get(item['role'], item['role'])}：{item['content']}"
        for item in history
    )


def _last_user_query(history: List[Dict[str, str]]) -> str:
    for item in reversed(history):
        if item["role"] == "user":
            return item["content"]
    return ""


def _meaningful_ngrams(text: str) -> set[str]:
    normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text.lower())
    terms = set()
    for size in (2, 3, 4):
        terms.update(
            normalized[index : index + size]
            for index in range(max(len(normalized) - size + 1, 0))
        )
    return {term for term in terms if term not in REWRITE_STOP_TERMS}


def _inherits_previous_topic(candidate: str, previous_query: str) -> bool:
    if not previous_query:
        return True
    return bool(
        _meaningful_ngrams(candidate) & _meaningful_ngrams(previous_query)
    )


def _extract_topic_hint(previous_query: str) -> str:
    normalized = re.sub(r"[，。！？、,.!?；;：:\s]", "", previous_query)
    boundaries = [
        normalized.find(pattern)
        for pattern in TOPIC_BOUNDARY_PATTERNS
        if pattern in normalized
    ]
    if boundaries:
        normalized = normalized[: min(boundaries)]
    normalized = re.sub(r"^(请问|我想问|想问一下|关于)", "", normalized)
    return normalized if 2 <= len(normalized) <= 30 else ""


def _message_to_dict(message) -> Dict[str, Any]:
    if hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        data = dict(message)
    else:
        data = {"role": "assistant", "content": str(message)}
    data.setdefault("role", "assistant")
    data.setdefault("content", "")
    return data


def _result_to_dict(result: ToolResult) -> Dict[str, Any]:
    return {
        "name": result.name,
        "content": result.content,
        "entities": result.entities or {},
        "intents": result.intents,
        "sources": result.sources,
        "debug": result.debug or {},
    }


def _collect_debug(tool_results: List[Dict[str, Any]]) -> Dict[str, str]:
    contexts = []
    entities = ""
    intents = ""
    for item in tool_results:
        contexts.append(f"[{item.get('name')}]\n{item.get('content', '')}")
        if item.get("entities"):
            entities = json.dumps(item["entities"], ensure_ascii=False)
        if item.get("intents"):
            intents = str(item["intents"])
    return {
        "knowledge_context": "\n\n".join(contexts),
        "graph_entities": entities,
        "graph_intents": intents,
    }


def build_qa_graph(
    llm: QwenClient,
    tools: MedicalRAGTools,
    memory_store: SQLiteMemoryStore,
):
    """构建包含短期上下文与 SQLite 长期记忆的 Hybrid RAG workflow。"""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "未安装 langgraph，请先执行 `pip install -r requirements.txt`。"
        ) from exc

    def load_long_term_memory_node(state: QAState) -> QAState:
        username = state.get("username", "").strip()
        memories = (
            load_relevant_memories(username, state["query"], memory_store)
            if username
            else []
        )
        memory_dicts = [record.to_dict() for record in memories]
        trace = list(state.get("agent_trace", []))
        trace.append(
            {
                "node": "load_long_term_memory",
                "username": username,
                "relevant_count": len(memory_dicts),
                "memories": memory_dicts,
            }
        )
        return {
            "relevant_memories": memory_dicts,
            "long_term_memory_context": format_memories_for_prompt(memories),
            "agent_trace": trace,
        }

    def rewrite_query_node(state: QAState) -> QAState:
        query = state["query"]
        history = state.get("history", [])
        should_rewrite = _needs_rewrite(query, history)
        rewritten_query = query
        initial_candidate = query
        correction = "not_needed"
        if should_rewrite:
            previous_query = _last_user_query(history)
            initial_candidate = llm.complete(
                REWRITE_PROMPT.format(
                    history=_format_history(history),
                    query=query,
                )
            ).strip() or query
            rewritten_query = initial_candidate
            correction = "accepted"
            if not _inherits_previous_topic(rewritten_query, previous_query):
                repaired = llm.complete(
                    REWRITE_REPAIR_PROMPT.format(
                        previous_query=previous_query,
                        query=query,
                        candidate=rewritten_query,
                    )
                ).strip()
                if repaired and _inherits_previous_topic(repaired, previous_query):
                    rewritten_query = repaired
                    correction = "llm_repair"
                else:
                    topic_hint = _extract_topic_hint(previous_query)
                    if topic_hint:
                        rewritten_query = f"关于{topic_hint}，{query}"
                        correction = "rule_fallback"
                    else:
                        correction = "repair_failed"

        trace = list(state.get("agent_trace", []))
        trace.append(
            {
                "node": "rewrite_query",
                "memory_used": should_rewrite,
                "history_messages": history,
                "original_query": query,
                "initial_candidate": initial_candidate,
                "rewritten_query": rewritten_query,
                "correction": correction,
            }
        )
        return {
            "rewritten_query": rewritten_query,
            "memory_used": should_rewrite,
            "agent_trace": trace,
        }

    def router_node(state: QAState) -> QAState:
        decision = route_query(state["rewritten_query"], llm)
        trace = list(state.get("agent_trace", []))
        trace.append(
            {
                "node": "router",
                "route": decision.route,
                "reason": decision.reason,
                "source": decision.source,
                "vector_matches": decision.vector_matches,
                "graph_matches": decision.graph_matches,
            }
        )
        return {
            "route_decision": decision.to_dict(),
            "agent_trace": trace,
        }

    def retrieve_node(state: QAState) -> QAState:
        query = state["rewritten_query"]
        route = state["route_decision"]["route"]
        results: List[ToolResult] = []
        if route in {"graph", "hybrid"}:
            results.append(tools.medical_graph_search(query))
        if route in {"vector", "hybrid"}:
            results.append(tools.medical_vector_search(query))

        tool_results = [_result_to_dict(result) for result in results]
        trace = list(state.get("agent_trace", []))
        for result in results:
            trace.append(
                {
                    "node": "tools",
                    "tool_name": result.name,
                    "arguments": {"query": query},
                    "result_preview": result.content[:1000],
                    "debug": result.debug or {},
                }
            )
        return {
            "tool_results": tool_results,
            "agent_trace": trace,
            **_collect_debug(tool_results),
        }

    def final_agent_node(state: QAState) -> QAState:
        route = state["route_decision"]
        evidence = state.get("knowledge_context", "")
        long_term_memory = state.get(
            "long_term_memory_context",
            "无相关用户长期记忆。",
        )
        final_prompt = (
            f"路由结果：{route['route']}；原因：{route['reason']}\n\n"
            f"<检索证据>\n{evidence or '没有检索证据'}\n</检索证据>\n\n"
            f"<相关用户长期记忆>\n{long_term_memory}\n</相关用户长期记忆>\n\n"
            f"<原始用户问题>{state['query']}</原始用户问题>\n"
            f"<用于检索的独立问题>{state['rewritten_query']}</用于检索的独立问题>\n"
            "请严格基于检索证据生成自然、具体的中文回答。"
            "仅在确实相关时结合用户长期记忆进行提醒；若记忆与当前问题冲突或可能已过期，"
            "应提示用户核实，不得把记忆当作诊断。"
        )
        completion = llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": final_prompt},
            ]
        )
        message = _message_to_dict(completion.choices[0].message)
        answer = str(message.get("content") or "")
        trace = list(state.get("agent_trace", []))
        trace.append(
            {
                "node": "final_agent",
                "route": route["route"],
                "content": answer,
                "evidence_count": len(state.get("tool_results", [])),
                "memory_used": state.get("memory_used", False),
                "long_term_memory_count": len(
                    state.get("relevant_memories", [])
                ),
                "rewritten_query": state.get("rewritten_query", state["query"]),
            }
        )
        return {
            "final_answer": answer,
            "messages": state.get("messages", []) + [message],
            "agent_trace": trace,
        }

    def save_long_term_memory_node(state: QAState) -> QAState:
        username = state.get("username", "").strip()
        saved = (
            save_explicit_memories(
                username,
                state["query"],
                llm,
                memory_store,
            )
            if username
            else []
        )
        saved_dicts = [record.to_dict() for record in saved]
        trace = list(state.get("agent_trace", []))
        trace.append(
            {
                "node": "save_long_term_memory",
                "saved_count": len(saved_dicts),
                "memories": saved_dicts,
            }
        )
        return {
            "saved_memories": saved_dicts,
            "agent_trace": trace,
        }

    graph = StateGraph(QAState)
    graph.add_node("load_long_term_memory", load_long_term_memory_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("final_agent", final_agent_node)
    graph.add_node("save_long_term_memory", save_long_term_memory_node)
    graph.set_entry_point("save_long_term_memory")
    graph.add_edge("save_long_term_memory", "load_long_term_memory")
    graph.add_edge("load_long_term_memory", "rewrite_query")
    graph.add_edge("rewrite_query", "router")
    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "final_agent")
    graph.add_edge("final_agent", END)
    return graph.compile()


def run_hybrid_rag(
    query: str,
    llm: QwenClient,
    tools: MedicalRAGTools,
    history: List[Dict[str, Any]] | None = None,
    username: str = "",
    memory_store: SQLiteMemoryStore | None = None,
) -> QAState:
    """执行规则优先、LLM 兜底的 Hybrid RAG。"""
    store = memory_store or SQLiteMemoryStore()
    graph = build_qa_graph(llm, tools, store)
    initial_state: QAState = {
        "query": query,
        "username": username,
        "rewritten_query": query,
        "history": _trim_history(history),
        "memory_used": False,
        "relevant_memories": [],
        "long_term_memory_context": "无相关用户长期记忆。",
        "saved_memories": [],
        "messages": [{"role": "user", "content": query}],
        "tool_results": [],
        "agent_trace": [],
    }
    return graph.invoke(initial_state)
