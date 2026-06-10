"""Streamlit 医疗问答工作台。

负责加载运行时资源、管理会话与长期记忆，并调用 ReAct RAG 工作流。
"""

from __future__ import annotations

import html
import logging
import os
import pickle

import py2neo
import streamlit as st
import torch
from transformers import BertTokenizer

from medical_rag import ner
from medical_rag.clients.neo4j import KGClient
from medical_rag.clients.qwen import QwenClient, get_llm_client
from medical_rag.core.config import settings
from medical_rag.core.devices import resolve_torch_device
from medical_rag.core.logging import setup_logging
from medical_rag.memory.store import SQLiteMemoryStore
from medical_rag.retrieval.tools import MedicalRAGTools
from medical_rag.retrieval.vector_store import MedicalVectorStore
from medical_rag.ui.agent_view import build_agent_view
from medical_rag.ui.styles import brand_html, inject_app_styles
from medical_rag.workflow.react_rag import run_react_rag

setup_logging()
logger = logging.getLogger(__name__)


@st.cache_resource
def load_ner_resources(checkpoint_name: str):
    """加载并缓存 NER tokenizer、模型和规则资源。"""
    device = resolve_torch_device(settings.COMPUTE_DEVICE)
    with open(os.path.join(settings.TMP_DIR, "tag2idx.npy"), "rb") as file:
        tag2idx = pickle.load(file)

    idx2tag = list(tag2idx)
    rule = ner.rule_find()
    tfidf_alignment = ner.tfidf_alignment()
    tokenizer = BertTokenizer.from_pretrained(settings.NER_MODEL_NAME)
    model = ner.Bert_Model(
        settings.NER_MODEL_NAME,
        hidden_size=128,
        tag_num=len(tag2idx),
        bi=True,
    )
    model.load_state_dict(
        torch.load(
            os.path.join(settings.MODEL_DIR, f"{checkpoint_name}.pt"),
            map_location=device,
        )
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model, idx2tag, rule, tfidf_alignment, device


@st.cache_resource
def load_llm_client() -> QwenClient:
    """加载并缓存千问 API 客户端。"""
    return get_llm_client()


@st.cache_resource
def load_vector_store() -> MedicalVectorStore:
    """加载 FAISS 向量库；索引异常时仍允许应用启动。"""
    store = MedicalVectorStore()
    try:
        store.load()
    except ImportError:
        logger.warning("FAISS 依赖未安装，向量检索暂不可用。")
    except Exception:
        logger.exception("FAISS 索引加载失败，向量检索暂不可用。")
    return store


@st.cache_resource
def load_memory_store() -> SQLiteMemoryStore:
    """加载按登录用户名隔离的 SQLite 长期记忆。"""
    return SQLiteMemoryStore()


MEMORY_TYPE_LABELS = {
    "medical_history": "病史",
    "allergy": "过敏",
    "medication": "用药",
    "preference": "偏好",
}


def render_memory_manager(username: str) -> None:
    """显示当前用户的长期记忆和删除操作。"""
    store = load_memory_store()
    with st.expander("长期记忆管理"):
        memories = store.list_memories(username)
        if not memories:
            st.caption("暂无长期记忆")
            return

        for memory in memories:
            label = MEMORY_TYPE_LABELS.get(memory.memory_type, memory.memory_type)
            text_col, action_col = st.columns([0.82, 0.18])
            with text_col:
                st.markdown(f"**{label}**：{memory.content}")
            with action_col:
                if st.button(
                    "删除",
                    key=f"delete_memory_{username}_{memory.id}",
                    help="删除这条长期记忆",
                ):
                    store.delete_memory(username, memory.id)
                    st.rerun()

        confirm_clear = st.checkbox(
            "确认清空全部长期记忆",
            key=f"confirm_clear_memory_{username}",
        )
        if st.button(
            "清空全部",
            key=f"clear_all_memory_{username}",
            disabled=not confirm_clear,
        ):
            store.delete_all(username)
            st.rerun()


def _render_assistant_debug(
    message: dict,
    *,
    show_overview: bool,
    show_tools: bool,
    show_trace: bool,
    show_graph_details: bool,
) -> None:
    """按真实 ReAct 运行结构展示单条回答的调试信息。"""
    view = message.get("agent_view") or build_agent_view(message)
    if show_overview:
        with st.expander("Agent 运行概览", expanded=True):
            st.json(
                {
                    "route": view.get("route"),
                    "skills": view.get("skills"),
                    "tools": view.get("tools"),
                    "evidence_status": view.get("evidence_status"),
                    "evidence_reason": view.get("evidence_reason"),
                    "answer_mode": view.get("answer_mode"),
                    "risk_level": view.get("risk_level"),
                    "stop_reason": view.get("stop_reason"),
                    "harness": view.get("harness"),
                    "output_review": view.get("output_review"),
                }
            )
    if show_tools:
        with st.expander("工具调用与检索证据"):
            tool_events = view.get("tool_events", [])
            if not tool_events:
                st.caption("本轮未调用检索工具")
            for index, event in enumerate(tool_events, start=1):
                st.markdown(
                    f"**{index}. {event.get('tool_name', 'unknown')}**"
                )
                st.json(
                    {
                        "arguments": event.get("arguments", {}),
                        "result_preview": event.get("result_preview", ""),
                        "debug": event.get("debug", {}),
                    }
                )
    if show_trace:
        with st.expander("完整 Agent Trace"):
            st.json(view.get("trace", []))
    if show_graph_details:
        with st.expander("图谱 NER 与意图解析"):
            st.json(
                {
                    "entities": message.get("ent", ""),
                    "intents": message.get("yitu", ""),
                }
            )


def main(is_admin: bool, username: str) -> None:
    """渲染登录后的聊天工作台。"""
    safe_username = html.escape(username)
    inject_app_styles()

    with st.sidebar:
        st.markdown(brand_html(compact=True), unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sidebar-user">
                <strong>{safe_username}</strong>
                <span>{'管理员' if is_admin else '普通用户'} · {settings.QWEN_MODEL}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "chat_windows" not in st.session_state:
            st.session_state.chat_windows = [[]]
            st.session_state.messages = [[]]

        if st.button("新建会话", type="primary", use_container_width=True):
            st.session_state.chat_windows.append([])
            st.session_state.messages.append([])

        window_options = [
            f"对话窗口 {index + 1}"
            for index in range(len(st.session_state.chat_windows))
        ]
        selected_window = st.selectbox("当前会话", window_options)
        active_window_index = int(selected_window.split()[1]) - 1

        render_memory_manager(username)

        show_overview = show_tools = show_trace = show_graph_details = False
        if is_admin:
            with st.expander("调试视图"):
                show_overview = st.checkbox("Agent 运行概览", value=True)
                show_tools = st.checkbox("工具调用与检索证据")
                show_trace = st.checkbox("完整 Agent Trace")
                show_graph_details = st.checkbox("图谱 NER 与意图解析")
                st.link_button(
                    "打开 Neo4j 管理页",
                    "http://127.0.0.1:7474/",
                    use_container_width=True,
                )

        st.divider()
        if st.button("退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.admin = False
            st.rerun()

    try:
        llm = load_llm_client()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    tokenizer, model, idx2tag, rule, tfidf_alignment, device = load_ner_resources(
        settings.NER_CHECKPOINT
    )
    graph = py2neo.Graph(
        settings.NEO4J_URL,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        name=settings.NEO4J_DBNAME,
    )
    kg_client = KGClient(graph)

    current_messages = st.session_state.messages[active_window_index]
    st.markdown(
        f"""
        <div class="page-heading">
            <h1>{window_options[active_window_index]}</h1>
            <p>结合医疗知识图谱、向量检索与个人记忆回答问题</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_assistant_debug(
                    message,
                    show_overview=show_overview,
                    show_tools=show_tools,
                    show_trace=show_trace,
                    show_graph_details=show_graph_details,
                )

    suggested_query = ""
    if not current_messages:
        st.markdown(
            """
            <div class="empty-state">
                <h2>今天想了解什么？</h2>
                <p>可以询问疾病表现、治疗方式、用药知识或检查项目。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        suggestions = (
            "皮炎有哪些常见症状？",
            "高血压患者日常要注意什么？",
            "感冒需要做哪些检查？",
        )
        for index, (column, suggestion) in enumerate(
            zip(st.columns(3), suggestions)
        ):
            with column:
                if st.button(
                    suggestion,
                    key=f"suggestion_{active_window_index}_{index}",
                    use_container_width=True,
                ):
                    suggested_query = suggestion

    typed_query = st.chat_input(
        "输入医疗问题",
        key=f"chat_input_{active_window_index}",
    )
    query = typed_query or suggested_query
    if query:
        conversation_history = list(current_messages)
        current_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        response_placeholder = st.empty()
        response_placeholder.text("ReAct Agent 正在检索并检查证据...")

        rag_tools = MedicalRAGTools(
            llm=llm,
            kg=kg_client,
            bert_model=model,
            bert_tokenizer=tokenizer,
            rule=rule,
            tfidf_r=tfidf_alignment,
            device=device,
            idx2tag=idx2tag,
            vector_store=load_vector_store(),
        )
        try:
            result = run_react_rag(
                query,
                llm,
                rag_tools,
                history=conversation_history,
                username=username,
                memory_store=load_memory_store(),
            )
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        assistant_message = {
            "role": "assistant",
            "content": result.get("final_answer", ""),
            "yitu": result.get("graph_intents", ""),
            "prompt": result.get("knowledge_context", ""),
            "ent": str(result.get("graph_entities", "")),
            "agent_trace": result.get("agent_trace", []),
            "selected_skills": result.get("selected_skill_names", []),
            "evidence_status": result.get("evidence_assessment", {}).get(
                "status",
                "",
            ),
            "harness_summary": result.get("harness_summary", {}),
            "stop_reason": result.get("stop_reason", ""),
            "output_review": result.get("output_review", {}),
        }
        assistant_message["agent_view"] = build_agent_view(result)
        response_placeholder.empty()
        with st.chat_message("assistant"):
            st.markdown(assistant_message["content"])
            _render_assistant_debug(
                assistant_message,
                show_overview=show_overview,
                show_tools=show_tools,
                show_trace=show_trace,
                show_graph_details=show_graph_details,
            )
        current_messages.append(assistant_message)

    st.session_state.messages[active_window_index] = current_messages
