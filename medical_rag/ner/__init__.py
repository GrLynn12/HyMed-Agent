"""NER 子包：BERT+BiRNN 命名实体识别。

公开 API：

* :class:`Bert_Model` —— 模型定义
* :class:`rule_find` —— Aho-Corasick 规则匹配器
* :class:`tfidf_alignment` —— TF-IDF 实体对齐
* :func:`get_ner_result` —— 推理入口（融合模型 + 规则 + 对齐）
* :func:`find_entities` / :func:`merge` —— 推理辅助
* :class:`Nerdataset` / :class:`Entity_Extend` —— 训练数据集与增强
* :func:`get_data` / :func:`build_tag2idx` —— 训练数据加载

子模块拆分原则：

* :mod:`ner.dataset` —— 训练阶段使用：数据集、增强、tag 字典
* :mod:`ner.model` —— 模型定义（推理 / 训练共用）
* :mod:`ner.inference` —— 推理阶段使用：规则匹配、TF-IDF 对齐、最终融合
* :mod:`ner.train` —— 训练入口（``python -m ner.train``）
"""

from .dataset import (
    Nerdataset,
    Entity_Extend,
    find_entities,
    get_data,
    build_tag2idx,
)
from .model import Bert_Model
from .inference import (
    rule_find,
    tfidf_alignment,
    merge,
    get_ner_result,
)

__all__ = [
    "Nerdataset",
    "Entity_Extend",
    "find_entities",
    "get_data",
    "build_tag2idx",
    "Bert_Model",
    "rule_find",
    "tfidf_alignment",
    "merge",
    "get_ner_result",
]
