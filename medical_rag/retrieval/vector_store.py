"""FAISS 向量检索封装。"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np

from medical_rag.core.config import settings
from medical_rag.core.devices import resolve_sentence_transformer_device
from medical_rag.retrieval.documents import MedicalDocument

logger = logging.getLogger(__name__)


INDEX_FILE = "medical.faiss"
DOCS_FILE = "documents.json"


@dataclass(frozen=True)
class VectorSearchResult:
    """两阶段检索结果及其分数。"""

    document: MedicalDocument
    final_score: float
    faiss_score: float
    rerank_score: float | None = None

    def to_debug_dict(self, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "score": self.final_score,
            "faiss_score": self.faiss_score,
            "rerank_score": self.rerank_score,
            "source": self.document.source,
            "disease": self.document.disease,
            "section": self.document.section,
            "content": self.document.content,
        }


class MedicalVectorStore:
    """基于 SentenceTransformer + FAISS 的本地向量库。"""

    def __init__(
        self,
        index_dir: str | None = None,
        embedding_model_name: str | None = None,
        embedding_device: str | None = None,
        embedding_batch_size: int | None = None,
        embedding_max_seq_length: int | None = None,
        show_progress: bool | None = None,
        reranker_enabled: bool | None = None,
        reranker_model_name: str | None = None,
        reranker_device: str | None = None,
        reranker_candidate_k: int | None = None,
        reranker_batch_size: int | None = None,
        reranker_max_length: int | None = None,
    ) -> None:
        self.index_dir = index_dir or settings.FAISS_INDEX_DIR
        self.embedding_model_name = embedding_model_name or settings.EMBEDDING_MODEL_NAME
        self.embedding_device = embedding_device or settings.EMBEDDING_DEVICE
        self.embedding_batch_size = embedding_batch_size or settings.EMBEDDING_BATCH_SIZE
        self.embedding_max_seq_length = embedding_max_seq_length or settings.EMBEDDING_MAX_SEQ_LENGTH
        self.show_progress = settings.EMBEDDING_SHOW_PROGRESS if show_progress is None else show_progress
        self.reranker_enabled = (
            settings.RERANKER_ENABLED
            if reranker_enabled is None
            else reranker_enabled
        )
        self.reranker_model_name = (
            reranker_model_name or settings.RERANKER_MODEL_NAME
        )
        self.reranker_device = reranker_device or settings.RERANKER_DEVICE
        self.reranker_candidate_k = (
            reranker_candidate_k or settings.RERANKER_CANDIDATE_K
        )
        self.reranker_batch_size = (
            reranker_batch_size or settings.RERANKER_BATCH_SIZE
        )
        self.reranker_max_length = (
            reranker_max_length or settings.RERANKER_MAX_LENGTH
        )
        self.model = None
        self.reranker = None
        self.reranker_load_failed = False
        self.index = None
        self.documents: List[MedicalDocument] = []

    @property
    def index_path(self) -> str:
        return os.path.join(self.index_dir, INDEX_FILE)

    @property
    def docs_path(self) -> str:
        return os.path.join(self.index_dir, DOCS_FILE)

    def build(self, documents: Sequence[MedicalDocument]) -> None:
        """从文档集合构建 FAISS 索引。"""
        if not documents:
            raise ValueError("没有可构建向量索引的医疗文档。")
        self.documents = list(documents)
        texts = [doc.content for doc in self.documents]
        logger.info(
            "开始生成向量: docs=%d, model=%s, device=%s, batch_size=%d, max_seq_length=%d",
            len(texts),
            self.embedding_model_name,
            self.embedding_device,
            self.embedding_batch_size,
            self.embedding_max_seq_length,
        )
        embeddings = self._embed(texts)
        dim = embeddings.shape[1]
        import faiss

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.index = index

    def save(self) -> None:
        """保存索引和文档元数据。"""
        if self.index is None:
            raise RuntimeError("向量索引尚未构建，无法保存。")
        import faiss

        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.docs_path, "w", encoding="utf-8") as f:
            json.dump([doc.to_dict() for doc in self.documents], f, ensure_ascii=False)

    def load(self) -> bool:
        """加载本地索引；成功返回 True，索引不存在返回 False。"""
        if not os.path.exists(self.index_path) or not os.path.exists(self.docs_path):
            return False
        import faiss

        self.index = faiss.read_index(self.index_path)
        with open(self.docs_path, "r", encoding="utf-8") as f:
            self.documents = [MedicalDocument.from_dict(item) for item in json.load(f)]
        return True

    def search(self, query: str, top_k: int = 5) -> List[MedicalDocument]:
        """检索最相近的 top-k 文档。"""
        return [doc for doc, _score in self.search_with_scores(query, top_k)]

    def search_with_scores(self, query: str, top_k: int = 5) -> List[Tuple[MedicalDocument, float]]:
        """检索并返回最终排序分数；启用 Reranker 时该分数为 rerank 分数。"""
        return [
            (result.document, result.final_score)
            for result in self.search_with_details(query, top_k)
        ]

    def search_with_details(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[VectorSearchResult]:
        """FAISS 召回候选后，可选用 CrossEncoder 进行精排。"""
        if self.index is None:
            loaded = self.load()
            if not loaded:
                logger.warning("FAISS 索引不存在: %s", self.index_dir)
                return []
        if self.index is None or not self.documents:
            return []
        query_embedding = self._embed([query])
        candidate_k = (
            max(top_k, self.reranker_candidate_k)
            if self.reranker_enabled
            else top_k
        )
        limit = min(candidate_k, len(self.documents))
        scores, indices = self.index.search(query_embedding, limit)
        candidates: List[VectorSearchResult] = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.documents):
                candidates.append(
                    VectorSearchResult(
                        document=self.documents[int(idx)],
                        final_score=float(score),
                        faiss_score=float(score),
                    )
                )
        if not candidates or not self.reranker_enabled:
            return candidates[:top_k]

        rerank_scores = self._rerank(
            query,
            [item.document.content for item in candidates],
        )
        if rerank_scores is None:
            return candidates[:top_k]
        reranked = [
            VectorSearchResult(
                document=item.document,
                final_score=score,
                faiss_score=item.faiss_score,
                rerank_score=score,
            )
            for item, score in zip(candidates, rerank_scores)
        ]
        reranked.sort(key=lambda item: item.final_score, reverse=True)
        return reranked[:top_k]

    def _rerank(
        self,
        query: str,
        documents: Sequence[str],
    ) -> List[float] | None:
        if self.reranker_load_failed:
            return None
        try:
            if self.reranker is None:
                from sentence_transformers import CrossEncoder

                kwargs: dict[str, Any] = {
                    "max_length": self.reranker_max_length,
                }
                if os.path.exists(self.reranker_model_name):
                    kwargs["local_files_only"] = True
                device = resolve_sentence_transformer_device(
                    self.reranker_device
                )
                if device:
                    kwargs["device"] = device
                logger.info(
                    "加载 Reranker: model=%s, device=%s",
                    self.reranker_model_name,
                    device or "auto",
                )
                self.reranker = CrossEncoder(
                    self.reranker_model_name,
                    **kwargs,
                )
            raw_scores = self.reranker.predict(
                [(query, document) for document in documents],
                batch_size=self.reranker_batch_size,
                show_progress_bar=False,
            )
        except Exception:
            self.reranker_load_failed = True
            logger.exception("Reranker 加载或推理失败，本次回退到 FAISS 排序。")
            return None

        scores = np.asarray(raw_scores, dtype="float32").reshape(-1)
        return [
            float(score)
            if 0.0 <= float(score) <= 1.0
            else 1.0 / (1.0 + math.exp(-float(score)))
            for score in scores
        ]

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            kwargs = {}
            if os.path.exists(self.embedding_model_name):
                kwargs["local_files_only"] = True
            device = resolve_sentence_transformer_device(self.embedding_device)
            if device:
                kwargs["device"] = device
            self.model = SentenceTransformer(self.embedding_model_name, **kwargs)
            if self.embedding_max_seq_length > 0:
                self.model.max_seq_length = self.embedding_max_seq_length
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype="float32")
