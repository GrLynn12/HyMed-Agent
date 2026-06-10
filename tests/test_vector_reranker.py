"""FAISS 候选经过 Reranker 后的排序测试。"""

from __future__ import annotations

import numpy as np
import pytest

from medical_rag.retrieval.documents import MedicalDocument
from medical_rag.retrieval.vector_store import MedicalVectorStore


class _FakeIndex:
    def search(self, _embedding, _limit):
        return np.asarray([[0.9, 0.8]], dtype="float32"), np.asarray(
            [[0, 1]],
            dtype="int64",
        )


class _FakeReranker:
    def predict(self, pairs, **_kwargs):
        assert len(pairs) == 2
        return np.asarray([0.1, 0.95], dtype="float32")


def test_reranker_can_change_faiss_order():
    store = MedicalVectorStore(
        reranker_enabled=True,
        reranker_candidate_k=2,
    )
    store.index = _FakeIndex()
    store.documents = [
        MedicalDocument(
            content="与问题无关的肺部文本",
            source="test",
            disease="肺部疾病",
            section="A",
        ),
        MedicalDocument(
            content="阿司匹林与华法林合用会增加出血风险",
            source="test",
            disease="药物相互作用",
            section="B",
        ),
    ]
    store.model = object()
    store.reranker = _FakeReranker()
    store._embed = lambda _texts: np.asarray([[1.0]], dtype="float32")

    results = store.search_with_details("阿司匹林和华法林能一起吃吗", top_k=2)

    assert results[0].document.section == "B"
    assert results[0].rerank_score == pytest.approx(0.95)
    assert results[0].faiss_score < results[1].faiss_score
