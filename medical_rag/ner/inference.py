"""NER 推理流水线：规则匹配 + 模型预测 + 结果融合 + TF-IDF 实体对齐。

整体流程（:func:`get_ner_result`）：

1. BERT+RNN 模型预测 BIO 序列 → 得到模型实体集合
2. Aho-Corasick 自动机基于词典做规则匹配 → 得到规则实体集合
3. :func:`merge` 按实体长度优先级融合两组结果，去重叠
4. :class:`tfidf_alignment` 把抽取出的实体名对齐到知识图谱标准实体名
"""

from __future__ import annotations

import os
from typing import List, Tuple

import ahocorasick
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .dataset import find_entities


# --------------------------------------------------------------------- rule


class rule_find:
    """基于 Aho-Corasick 自动机的多模式实体匹配器。

    词典文件位于 ``data/ent_aug/{type}.txt``，按行存储实体名。
    """

    def __init__(self) -> None:
        self.idx2type = idx2type = ["食物", "药品商", "治疗方法", "药品", "检查项目", "疾病", "疾病症状", "科目"]
        self.type2idx = type2idx = {t: i for i, t in enumerate(idx2type)}
        self.ahos = [ahocorasick.Automaton() for _ in range(len(self.type2idx))]

        for type in idx2type:
            with open(os.path.join('data', 'ent_aug', f'{type}.txt'), encoding='utf-8') as f:
                all_en = f.read().split('\n')
            for en in all_en:
                en = en.split(' ')[0]
                if len(en) >= 2:
                    self.ahos[type2idx[type]].add_word(en, en)
        for i in range(len(self.ahos)):
            self.ahos[i].make_automaton()

    def find(self, sen: str) -> List[Tuple[int, int, str, str]]:
        """对句子做规则匹配，返回 ``[(begin, end, type, word), ...]``。"""
        rule_result: List[Tuple[int, int, str, str]] = []
        mp: dict = {}
        all_res: list = []
        all_ty: list = []
        for i in range(len(self.ahos)):
            now = list(self.ahos[i].iter(sen))
            all_res.extend(now)
            for _ in range(len(now)):
                all_ty.append(self.idx2type[i])
        if len(all_res) != 0:
            all_res = sorted(all_res, key=lambda x: len(x[1]), reverse=True)
            for i, res in enumerate(all_res):
                be = res[0] - len(res[1]) + 1
                ed = res[0]
                if be in mp or ed in mp:
                    continue
                rule_result.append((be, ed, all_ty[i], res[1]))
                for t in range(be, ed + 1):
                    mp[t] = 1
        return rule_result


# --------------------------------------------------------------------- alignment


class tfidf_alignment:
    """把 NER 抽出的实体名通过 TF-IDF 余弦相似度对齐到 KG 标准实体名。"""

    def __init__(self) -> None:
        eneities_path = os.path.join('data', 'ent_aug')
        files = os.listdir(eneities_path)
        files = [docu for docu in files if '.py' not in docu]

        self.tag_2_embs: dict = {}
        self.tag_2_tfidf_model: dict = {}
        self.tag_2_entity: dict = {}
        for ty in files:
            with open(os.path.join(eneities_path, ty), 'r', encoding='utf-8') as f:
                entities = f.read().split('\n')
                entities = [ent for ent in entities if 1 <= len(ent.split(' ')[0]) <= 15]
                en_name = [ent.split(' ')[0] for ent in entities]
                ty = ty.strip('.txt')
                self.tag_2_entity[ty] = en_name
                tfidf_model = TfidfVectorizer(analyzer="char")
                embs = tfidf_model.fit_transform(en_name).toarray()
                self.tag_2_embs[ty] = embs
                self.tag_2_tfidf_model[ty] = tfidf_model

    def align(self, ent_list):
        """对齐实体；相似度阈值 ≥ 0.5 的才会被保留。返回 ``{type: 标准实体名}``。"""
        new_result: dict = {}
        for s, e, cls, ent in ent_list:
            ent_emb = self.tag_2_tfidf_model[cls].transform([ent])
            sim_score = cosine_similarity(ent_emb, self.tag_2_embs[cls])
            max_idx = sim_score[0].argmax()
            max_score = sim_score[0][max_idx]
            if max_score >= 0.5:
                new_result[cls] = self.tag_2_entity[cls][max_idx]
        return new_result


# --------------------------------------------------------------------- merge


def merge(model_result_word, rule_result):
    """按 `word` 长度降序合并模型与规则两组结果，去除位置重叠。"""
    result = model_result_word + rule_result
    result = sorted(result, key=lambda x: len(x[-1]), reverse=True)
    check_result = []
    mp: dict = {}
    for res in result:
        if res[0] in mp or res[1] in mp:
            continue
        check_result.append(res)
        for i in range(res[0], res[1] + 1):
            mp[i] = 1
    return check_result


# --------------------------------------------------------------------- entry


def get_ner_result(model, tokenizer, sen, rule, tfidf_r, device, idx2tag):
    """NER 推理入口；返回 ``{type: 标准实体名}`` 字典。"""
    sen_to = tokenizer.encode(sen, add_special_tokens=True, return_tensors='pt').to(device)
    pre = model(sen_to).tolist()

    pre_tag = [idx2tag[i] for i in pre[1:-1]]
    model_result = find_entities(pre_tag)
    model_result_word = []
    for res in model_result:
        word = sen[res[0]:res[1] + 1]
        model_result_word.append((res[0], res[1], res[2], word))
    rule_result = rule.find(sen)

    merge_result = merge(model_result_word, rule_result)
    tfidf_result = tfidf_r.align(merge_result)
    return tfidf_result
