"""Neo4j 知识图谱查询封装。

把历史界面代码中散落的 Cypher 查询集中到 :class:`KGClient`，统一处理：

* 空结果：返回空列表而非抛 ``IndexError``
* 异常：通过 logger 记录但不中断 UI 流
* 提示拼接：把 ``<提示>...</提示>`` 模板放在一处，便于后续维护

设计原则：**查询方法只返回数据**，提示文本拼接由调用方完成，
保持职责单一，便于单元测试。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


class KGClient:
    """对 :class:`py2neo.Graph` 的薄封装。

    :param graph: 已建立连接的 py2neo Graph 实例；接受任意支持 ``run(cypher).data()`` 接口的对象，
                  方便测试时用 mock 替换。
    """

    def __init__(self, graph) -> None:
        self.graph = graph

    # ------------------------------------------------------------------ helpers

    def _run_first(self, cypher: str) -> Optional[dict]:
        """运行 cypher，返回首行 dict；无结果或异常时返回 None。"""
        try:
            rows = self.graph.run(cypher).data()
        except Exception:
            logger.exception("Cypher 查询失败: %s", cypher)
            return None
        return rows[0] if rows else None

    def _run_all(self, cypher: str) -> List[dict]:
        """运行 cypher，返回所有行；异常时返回空 list。"""
        try:
            return self.graph.run(cypher).data() or []
        except Exception:
            logger.exception("Cypher 查询失败: %s", cypher)
            return []

    # ------------------------------------------------------------------ queries

    def get_disease_attribute(self, disease: str, attribute: str) -> Optional[str]:
        """查询疾病的某个属性值（如疾病简介、疾病病因），无结果返回 None。"""
        cypher = f"match (a:疾病{{名称:'{disease}'}}) return a.{attribute}"
        row = self._run_first(cypher)
        if not row:
            return None
        # row 形如 {"a.疾病简介": "..."}，只取值
        values = [v for v in row.values() if v]
        if not values:
            return None
        return "".join(str(v) for v in values)

    def get_related_entities(
        self, disease: str, relation: str, target_label: str
    ) -> List[str]:
        """查询疾病通过某关系连接的目标节点名称列表。"""
        cypher = (
            f"match (a:疾病{{名称:'{disease}'}})-[r:{relation}]->(b:{target_label}) "
            f"return b.名称"
        )
        rows = self._run_all(cypher)
        return [list(row.values())[0] for row in rows if row]

    def get_diseases_by_symptom(self, symptom: str) -> List[str]:
        """通过症状反查可能的疾病名称列表。"""
        cypher = (
            f"match (a:疾病)-[r:疾病的症状]->(b:疾病症状 {{名称:'{symptom}'}}) "
            f"return a.名称"
        )
        rows = self._run_all(cypher)
        return [list(row.values())[0] for row in rows if row]

    def get_drug_producers(self, drug: str) -> List[str]:
        """查询某药品对应的药品商列表。"""
        cypher = (
            f"match (a:药品商)-[r:生产]->(b:药品{{名称:'{drug}'}}) "
            f"return a.名称"
        )
        rows = self._run_all(cypher)
        return [list(row.values())[0] for row in rows if row]


# ---------------------------------------------------------------------- prompts

_PROMPT_TEMPLATE = "<提示>用户对{entity}可能有查询{action}需求，知识库内容如下：{body}</提示>"
_NO_INFO_TEXT = "图谱中无信息，查找失败。"


def build_attribute_prompt(entity: str, attribute: str, value: Optional[str]) -> str:
    """根据属性查询结果生成 ``<提示>...</提示>`` 文本。

    :param entity: 实体名（疾病名）
    :param attribute: 属性名（如 ``疾病简介``）
    :param value: 查询结果，None 或空字符串表示无信息
    """
    body = value if value else _NO_INFO_TEXT
    return _PROMPT_TEMPLATE.format(entity=entity, action=attribute, body=body)


def build_relation_prompt(entity: str, relation: str, items: Sequence[str]) -> str:
    """根据关系查询结果生成 ``<提示>...</提示>`` 文本。"""
    body = "、".join(items) if items else _NO_INFO_TEXT
    return _PROMPT_TEMPLATE.format(entity=entity, action=relation, body=body)
