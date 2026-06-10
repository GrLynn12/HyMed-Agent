"""长期记忆与 Hybrid RAG 工作流回归测试。"""

import json
from types import SimpleNamespace

from medical_rag.memory.service import (
    extract_memories,
    filter_relevant_memories,
    save_explicit_memories,
)
from medical_rag.memory.store import SQLiteMemoryStore
from medical_rag.retrieval.query_router import route_by_rules
from medical_rag.retrieval.tools import ToolResult
from medical_rag.workflow.hybrid_rag import run_hybrid_rag


class FakeLLM:
    def __init__(self):
        self.prompts = []
        self.chat_messages = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if "长期记忆提取器" in prompt:
            memories = []
            if "青霉素过敏" in prompt:
                memories.append(
                    {"memory_type": "allergy", "content": "用户对青霉素过敏"}
                )
            if "二甲双胍" in prompt:
                memories.append(
                    {"memory_type": "medication", "content": "用户正在服用二甲双胍"}
                )
            if "回答简洁" in prompt:
                memories.append(
                    {"memory_type": "preference", "content": "用户偏好简洁回答"}
                )
            return json.dumps(memories, ensure_ascii=False)
        if "问题改写器" in prompt:
            return "我对青霉素过敏，皮炎应该擦哪些药"
        return json.dumps(
            {"route": "vector", "reason": "测试路由"},
            ensure_ascii=False,
        )

    def chat(self, messages, tools=None, tool_choice=None):
        self.chat_messages.append(messages)
        message = SimpleNamespace(content="这是基于检索证据生成的测试回答。")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeTools:
    def medical_graph_search(self, query):
        return ToolResult(name="medical_graph_search", content=f"图谱证据：{query}")

    def medical_vector_search(self, query):
        return ToolResult(name="medical_vector_search", content=f"向量证据：{query}")


class BrokenRewriteLLM(FakeLLM):
    def complete(self, prompt):
        if "问题改写器" in prompt or "上一次改写仍然缺少" in prompt:
            self.prompts.append(prompt)
            return "可以擦哪些药膏"
        return super().complete(prompt)


def test_store_isolates_users_deduplicates_and_deletes(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "memory.db"))
    item = {"memory_type": "allergy", "content": "用户对青霉素过敏"}

    store.add_memories("alice", [item, item])
    store.add_memories("bob", [item])

    alice = store.list_memories("alice")
    assert len(alice) == 1
    assert len(store.list_memories("bob")) == 1
    assert store.delete_memory("bob", alice[0].id) is False
    assert store.delete_memory("alice", alice[0].id) is True
    assert store.list_memories("alice") == []
    assert store.delete_all("bob") == 1


def test_extraction_only_runs_for_explicit_user_statements(tmp_path):
    llm = FakeLLM()
    assert extract_memories("皮炎不治疗会怎样", llm) == []
    assert llm.prompts == []

    store = SQLiteMemoryStore(str(tmp_path / "memory.db"))
    saved = save_explicit_memories(
        "alice",
        "我对青霉素过敏，请回答简洁一点",
        llm,
        store,
    )
    assert {item.memory_type for item in saved} == {"allergy", "preference"}


def test_relevance_filter_keeps_preferences_and_medication_safety(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "memory.db"))
    store.add_memories(
        "alice",
        [
            {"memory_type": "allergy", "content": "用户对青霉素过敏"},
            {"memory_type": "medication", "content": "用户正在服用二甲双胍"},
            {"memory_type": "medical_history", "content": "用户有慢性胃炎病史"},
            {"memory_type": "preference", "content": "用户偏好简洁回答"},
        ],
    )

    relevant = filter_relevant_memories(
        "感冒应该吃什么药",
        store.list_memories("alice"),
    )
    memory_types = {item.memory_type for item in relevant}
    assert memory_types == {"allergy", "medication", "preference"}


def test_graph_loads_relevant_memory_and_saves_current_user_statement(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "memory.db"))
    store.add_memories(
        "alice",
        [
            {"memory_type": "allergy", "content": "用户对青霉素过敏"},
            {"memory_type": "preference", "content": "用户偏好简洁回答"},
        ],
    )
    llm = FakeLLM()

    result = run_hybrid_rag(
        "我正在服用二甲双胍，感冒应该吃什么药",
        llm,
        FakeTools(),
        username="alice",
        memory_store=store,
    )

    nodes = [item["node"] for item in result["agent_trace"]]
    assert nodes[0] == "save_long_term_memory"
    assert nodes[1] == "load_long_term_memory"
    assert nodes[-1] == "final_agent"
    assert result["agent_trace"][1]["relevant_count"] == 3
    assert result["saved_memories"][0]["memory_type"] == "medication"
    final_prompt = llm.chat_messages[-1][1]["content"]
    assert "用户对青霉素过敏" in final_prompt
    assert "用户偏好简洁回答" in final_prompt
    assert any(
        item.memory_type == "medication"
        for item in store.list_memories("alice")
    )


def test_follow_up_with_allergy_statement_is_rewritten_and_used_same_turn(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "memory.db"))
    llm = FakeLLM()
    history = [
        {"role": "user", "content": "皮炎应该怎么治疗？"},
        {"role": "assistant", "content": "皮炎通常需要根据类型选择治疗方法。"},
    ]

    result = run_hybrid_rag(
        "我对青霉素过敏，应该擦哪些药",
        llm,
        FakeTools(),
        history=history,
        username="alice",
        memory_store=store,
    )

    rewrite_trace = next(
        item for item in result["agent_trace"] if item["node"] == "rewrite_query"
    )
    memory_trace = next(
        item
        for item in result["agent_trace"]
        if item["node"] == "load_long_term_memory"
    )
    assert rewrite_trace["memory_used"] is True
    assert memory_trace["relevant_count"] == 1
    assert memory_trace["memories"][0]["memory_type"] == "allergy"
    final_prompt = llm.chat_messages[-1][1]["content"]
    assert "用户对青霉素过敏" in final_prompt
    assert route_by_rules("皮炎应该擦哪些药").route == "graph"


def test_incomplete_rewrite_is_corrected_with_previous_topic(tmp_path):
    result = run_hybrid_rag(
        "可以擦哪些药",
        BrokenRewriteLLM(),
        FakeTools(),
        history=[
            {"role": "user", "content": "皮炎可以不治疗吗"},
            {"role": "assistant", "content": "需要根据皮炎类型判断。"},
        ],
        username="alice",
        memory_store=SQLiteMemoryStore(str(tmp_path / "memory.db")),
    )

    rewrite_trace = next(
        item for item in result["agent_trace"] if item["node"] == "rewrite_query"
    )
    assert rewrite_trace["initial_candidate"] == "可以擦哪些药膏"
    assert rewrite_trace["rewritten_query"] == "关于皮炎，可以擦哪些药"
    assert rewrite_trace["correction"] == "rule_fallback"
