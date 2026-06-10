"""构建医疗文本 FAISS 索引。

用法::

    python -m scripts.build_vector_index

默认从独立外部医疗 QA 样例语料读取文档，索引保存到 ``vector_index/``。
如需使用 Huatuo-26M 类数据，请传入对应 JSONL 路径和 ``jsonl_qa`` 格式。
"""

from __future__ import annotations

import argparse
import logging

from medical_rag.core.config import settings
from medical_rag.core.logging import setup_logging
from medical_rag.retrieval.documents import load_external_corpus
from medical_rag.retrieval.vector_store import MedicalVectorStore

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="构建医疗文本 FAISS 向量索引")
    parser.add_argument("--corpus-path", default="/data0/grl_data/llm/rag/huatuo_5000.jsonl", help="外部医疗语料路径")
    parser.add_argument(
        "--corpus-format",
        default="jsonl_qa",
        choices=["jsonl_qa", "jsonl_text", "txt"],
        help="外部语料格式；Huatuo-26M 类 QA 数据通常使用 jsonl_qa",
    )
    parser.add_argument("--index-dir", default=settings.FAISS_INDEX_DIR, help="索引输出目录")
    parser.add_argument(
        "--embedding-model",
        default=settings.EMBEDDING_MODEL_NAME,
        help="SentenceTransformer embedding 模型名或本地路径",
    )
    parser.add_argument(
        "--embedding-device",
        default=settings.EMBEDDING_DEVICE,
        help="embedding 设备：auto/cpu/cuda/cuda:0 等",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=settings.EMBEDDING_BATCH_SIZE,
        help="embedding 批大小；显存占满或利用率低时先调小，如 4/8/16",
    )
    parser.add_argument(
        "--embedding-max-seq-length",
        type=int,
        default=settings.EMBEDDING_MAX_SEQ_LENGTH,
        help="embedding 最大 token 长度；bge-m3 默认较长，QA 检索一般 512 足够",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭 SentenceTransformer 进度条",
    )
    args = parser.parse_args()

    documents = load_external_corpus(args.corpus_path, args.corpus_format)
    logger.info("加载外部医疗文本片段 %d 条", len(documents))
    store = MedicalVectorStore(
        index_dir=args.index_dir,
        embedding_model_name=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_seq_length=args.embedding_max_seq_length,
        show_progress=not args.no_progress,
    )
    store.build(documents)
    store.save()
    logger.info("FAISS 索引已保存到 %s", args.index_dir)


if __name__ == "__main__":
    main()
