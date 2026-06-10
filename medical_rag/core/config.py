"""项目集中配置。

所有外部依赖参数（Neo4j 连接、千问 API、本地模型/数据路径）都通过环境变量
覆盖，未设置时使用与原代码完全一致的默认值，保证零回归。

使用方式::

    from medical_rag.core.config import settings
    print(settings.NEO4J_URL)

环境变量列表见下方常量定义；亦可参考 README 的「环境变量」章节。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import local_config
except ImportError:
    local_config = None


def _env(name: str, default: str) -> str:
    """读取环境变量，未设置时返回 default。"""
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _env_first(names: tuple[str, ...], default: str) -> str:
    """按顺序读取多个环境变量，返回第一个非空值。"""
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default


def _local(name: str, default: str = "") -> str:
    """读取 local_config.py 中的本机私密配置。"""
    if local_config is None:
        return default
    value = getattr(local_config, name, default)
    return value if value is not None and value != "" else default


@dataclass(frozen=True)
class Settings:
    """运行时配置容器（不可变）。"""

    # --- Neo4j ---
    NEO4J_URL: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str
    NEO4J_DBNAME: str

    # --- 千问 / DashScope LLM ---
    QWEN_API_KEY: str
    QWEN_BASE_URL: str
    QWEN_MODEL: str
    QWEN_TEMPERATURE: float

    # --- 本地模型与数据路径 ---
    NER_MODEL_NAME: str         # HuggingFace BERT 路径（chinese-roberta-wwm-ext）
    NER_CHECKPOINT: str         # 训练好的 NER 权重文件名（不含 .pt 后缀）
    DATA_DIR: str
    TMP_DIR: str
    MODEL_DIR: str
    FAISS_INDEX_DIR: str
    EMBEDDING_MODEL_NAME: str
    EMBEDDING_BATCH_SIZE: int
    EMBEDDING_MAX_SEQ_LENGTH: int
    EMBEDDING_SHOW_PROGRESS: bool
    VECTOR_TOP_K: int
    RERANKER_ENABLED: bool
    RERANKER_MODEL_NAME: str
    RERANKER_DEVICE: str
    RERANKER_CANDIDATE_K: int
    RERANKER_BATCH_SIZE: int
    RERANKER_MAX_LENGTH: int
    RERANKER_SCORE_THRESHOLD: float
    TRADITIONAL_RAG_CORPUS_PATH: str
    TRADITIONAL_RAG_CORPUS_FORMAT: str
    COMPUTE_DEVICE: str
    EMBEDDING_DEVICE: str
    MEMORY_RECENT_TURNS: int
    MEMORY_DB_PATH: str
    MEMORY_MAX_RELEVANT: int

    # --- ReAct Agent Harness ---
    AGENT_MAX_TOOL_CALLS: int
    AGENT_MAX_REWRITES: int
    AGENT_TIMEOUT_SECONDS: float
    AGENT_MAX_SAME_TOOL_CALLS: int
    AGENT_VECTOR_SCORE_THRESHOLD: float
    AGENT_MAX_TOP_K: int

    # --- 日志 ---
    LOG_LEVEL: str


def _build_settings() -> Settings:
    return Settings(
        NEO4J_URL=_env("NEO4J_URL", "bolt://localhost:7687"),
        NEO4J_USER=_env("NEO4J_USER", "neo4j"),
        NEO4J_PASSWORD=_local("NEO4J_PASSWORD"),
        NEO4J_DBNAME=_env("NEO4J_DBNAME", "neo4j"),
        QWEN_API_KEY=_env_first(
            ("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
            _local("QWEN_API_KEY"),
        ),
        QWEN_BASE_URL=_env(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        QWEN_MODEL=_env("QWEN_MODEL", "qwen-turbo"),
        QWEN_TEMPERATURE=float(_env("QWEN_TEMPERATURE", "0.2")),
        NER_MODEL_NAME=_env("NER_MODEL_NAME", "model/chinese-roberta-wwm-ext"),
        NER_CHECKPOINT=_env("NER_CHECKPOINT", "best_roberta_rnn_model_ent_aug"),
        DATA_DIR=_env("DATA_DIR", "data"),
        TMP_DIR=_env("TMP_DIR", "tmp_data"),
        MODEL_DIR=_env("MODEL_DIR", "model"),
        FAISS_INDEX_DIR=_env("FAISS_INDEX_DIR", "vector_index"),
        EMBEDDING_MODEL_NAME=_local("EMBEDDING_MODEL_NAME"),
        EMBEDDING_BATCH_SIZE=int(_env("EMBEDDING_BATCH_SIZE", "16")),
        EMBEDDING_MAX_SEQ_LENGTH=int(_env("EMBEDDING_MAX_SEQ_LENGTH", "1024")),
        EMBEDDING_SHOW_PROGRESS=_env("EMBEDDING_SHOW_PROGRESS", "1").lower() in {"1", "true", "yes", "y"},
        VECTOR_TOP_K=int(_env("VECTOR_TOP_K", "5")),
        RERANKER_ENABLED=_env("RERANKER_ENABLED", "1").lower()
        in {"1", "true", "yes", "y"},
        RERANKER_MODEL_NAME=_local("RERANKER_MODEL_NAME"),
        # RERANKER_DEVICE=_env("RERANKER_DEVICE", "auto"),
        RERANKER_DEVICE = _local("DEVICE"),
        RERANKER_CANDIDATE_K=int(_env("RERANKER_CANDIDATE_K", "20")),
        RERANKER_BATCH_SIZE=int(_env("RERANKER_BATCH_SIZE", "8")),
        RERANKER_MAX_LENGTH=int(_env("RERANKER_MAX_LENGTH", "1024")),
        RERANKER_SCORE_THRESHOLD=float(
            _env("RERANKER_SCORE_THRESHOLD", "0.5")
        ),
        TRADITIONAL_RAG_CORPUS_PATH=_env(
            "TRADITIONAL_RAG_CORPUS_PATH",
            os.path.join("data", "external_medical_qa_sample.jsonl"),
        ),
        TRADITIONAL_RAG_CORPUS_FORMAT=_env("TRADITIONAL_RAG_CORPUS_FORMAT", "jsonl_qa"),
        # COMPUTE_DEVICE=_env("COMPUTE_DEVICE", "cuda:5"),
        COMPUTE_DEVICE = _local("DEVICE"),
        # EMBEDDING_DEVICE=_env("EMBEDDING_DEVICE", "cuda:5"),
        EMBEDDING_DEVICE = _local("DEVICE"),
        MEMORY_RECENT_TURNS=int(_env("MEMORY_RECENT_TURNS", "5")),
        MEMORY_DB_PATH=_env("MEMORY_DB_PATH", os.path.join("tmp_data", "user_memory.db")),
        MEMORY_MAX_RELEVANT=int(_env("MEMORY_MAX_RELEVANT", "6")),
        AGENT_MAX_TOOL_CALLS=int(_env("AGENT_MAX_TOOL_CALLS", "3")),
        AGENT_MAX_REWRITES=int(_env("AGENT_MAX_REWRITES", "1")),
        AGENT_TIMEOUT_SECONDS=float(_env("AGENT_TIMEOUT_SECONDS", "40")),
        AGENT_MAX_SAME_TOOL_CALLS=int(_env("AGENT_MAX_SAME_TOOL_CALLS", "1")),
        AGENT_VECTOR_SCORE_THRESHOLD=float(
            _env("AGENT_VECTOR_SCORE_THRESHOLD", "0.55")
        ),
        AGENT_MAX_TOP_K=int(_env("AGENT_MAX_TOP_K", "8")),
        LOG_LEVEL=_env("LOG_LEVEL", "INFO"),
    )


# 全局单例：模块导入时即冻结一份当前环境变量快照
settings: Settings = _build_settings()
