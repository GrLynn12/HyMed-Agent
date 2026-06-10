"""NER 训练数据集与实体增强。

本模块负责训练阶段的数据准备：

* :func:`get_data` 从 BIO 标注文件加载 ``(text, tag)`` 列表
* :func:`build_tag2idx` 构建 tag → idx 字典
* :func:`find_entities` 把 BIO 序列转为 ``(start, end, type)`` 实体三元组
* :class:`Entity_Extend` 三种数据增强策略：实体替换 / 掩码 / 拼接
* :class:`Nerdataset` PyTorch Dataset，支持数据增强
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------- io


def get_data(path: str, max_len: Optional[int] = None) -> Tuple[List[List[str]], List[List[str]]]:
    """读取 BIO 标注的训练数据文件。

    文件格式：每行 ``字 标签``，句子之间空行分隔。

    :param path: BIO 文件路径
    :param max_len: 仅返回前 ``max_len`` 条样本，None 表示全部
    """
    all_text: List[List[str]] = []
    all_tag: List[List[str]] = []
    with open(path, 'r', encoding='utf8') as f:
        all_data = f.read().split('\n')

    sen: List[str] = []
    tag: List[str] = []
    for data in all_data:
        parts = data.split(' ')
        if len(parts) != 2:
            if len(sen) > 2:
                all_text.append(sen)
                all_tag.append(tag)
            sen, tag = [], []
            continue
        te, ta = parts
        sen.append(te)
        tag.append(ta)
    if max_len is not None:
        return all_text[:max_len], all_tag[:max_len]
    return all_text, all_tag


def find_entities(tag: List[str]) -> List[Tuple[int, int, str]]:
    """把 BIO 标签序列转为 ``(start_idx, end_idx_inclusive, entity_type)`` 列表。

    示例::

        >>> find_entities(['B-药品', 'I-药品', 'O', 'B-疾病'])
        [(0, 1, '药品'), (3, 3, '疾病')]
    """
    result: List[Tuple[int, int, str]] = []
    label_len = len(tag)
    i = 0
    while i < label_len:
        if tag[i][0] == 'B':
            ent_type = tag[i].strip('B-')
            j = i + 1
            while j < label_len and tag[j][0] == 'I':
                j += 1
            result.append((i, j - 1, ent_type))
            i = j
        else:
            i = i + 1
    return result


def build_tag2idx(all_tag: List[List[str]]) -> dict:
    """根据训练集所有标签序列，构建 tag → idx 字典。``<PAD>`` 固定为 0。"""
    tag2idx = {'<PAD>': 0}
    for sen in all_tag:
        for tag in sen:
            tag2idx[tag] = tag2idx.get(tag, len(tag2idx))
    return tag2idx


# --------------------------------------------------------------------- augmentation


class Entity_Extend:
    """实体级数据增强：替换 / 掩码 / 拼接。"""

    def __init__(self) -> None:
        eneities_path = os.path.join('data', 'ent')
        if not os.path.isdir(eneities_path):
            alt_path = os.path.join('data', 'ent_aug')
            if os.path.isdir(alt_path):
                eneities_path = alt_path
            else:
                raise FileNotFoundError(
                    "Entity data directory not found: expected data/ent or data/ent_aug"
                )
        files = os.listdir(eneities_path)
        files = [docu for docu in files if '.py' not in docu]

        self.type2entity: dict = {}
        self.type2weight: dict = {}
        for type in files:
            with open(os.path.join(eneities_path, type), 'r', encoding='utf-8') as f:
                entities = f.read().split('\n')
                en_name = [ent for ent in entities if 1 <= len(ent.split(' ')[0]) <= 15]
                en_weight = [1] * len(en_name)
                type = type.strip('.txt')
                self.type2entity[type] = en_name
                self.type2weight[type] = en_weight

    def no_work(self, te, tag, type):
        return te, tag

    # 1. 实体替换
    def entity_replace(self, te, ta, type):
        choice_ent = random.choices(self.type2entity[type], weights=self.type2weight[type], k=1)[0]
        ta = ["B-" + type] + ["I-" + type] * (len(choice_ent) - 1)
        return list(choice_ent), ta

    # 2. 实体掩盖
    def entity_mask(self, te, ta, type):
        if len(te) <= 3:
            return te, ta
        elif len(te) <= 5:
            te.pop(random.randint(0, len(te) - 1))
        else:
            te.pop(random.randint(0, len(te) - 1))
            te.pop(random.randint(0, len(te) - 1))
        ta = ["B-" + type] + ["I-" + type] * (len(te) - 1)
        return te, ta

    # 3. 实体拼接
    def entity_union(self, te, ta, type):
        words = ['和', '与', '以及']
        wor = random.choice(words)
        choice_ent = random.choices(self.type2entity[type], weights=self.type2weight[type], k=1)[0]
        te = te + list(wor) + list(choice_ent)
        ta = ta + ['O'] * len(wor) + ["B-" + type] + ["I-" + type] * (len(choice_ent) - 1)
        return te, ta

    def entities_extend(self, text, tag, ents):
        cho = [self.no_work, self.entity_union, self.entity_mask, self.entity_replace, self.no_work]
        new_text = text.copy()
        new_tag = tag.copy()
        sign = 0
        for ent in ents:
            p = random.choice(cho)
            te, ta = p(text[ent[0]:ent[1] + 1], tag[ent[0]:ent[1] + 1], ent[2])
            new_text[ent[0] + sign:ent[1] + 1 + sign], new_tag[ent[0] + sign:ent[1] + 1 + sign] = te, ta
            sign += len(te) - (ent[1] - ent[0] + 1)
        return new_text, new_tag


# --------------------------------------------------------------------- dataset


class Nerdataset(Dataset):
    """PyTorch Dataset，封装 BIO 标注 + tokenizer 编码 + 实体增强。

    .. note:: 训练循环每 epoch 开头需调用 :meth:`set_epoch`，否则增强策略不会触发。
    """

    def __init__(self, all_text, all_label, tokenizer, max_len, tag2idx,
                 is_dev: bool = False, enhance_data: bool = False) -> None:
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2idx = tag2idx
        self.is_dev = is_dev
        self.entity_extend = Entity_Extend()
        self.enhance_data = enhance_data
        # 当前 epoch，由训练循环在每轮开头调用 set_epoch() 注入；
        # 用于控制实体增强策略只在 epoch >= 7 的奇数轮触发
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """训练循环每个 epoch 开头调用，用于控制数据增强触发条件。"""
        self.epoch = epoch

    def __getitem__(self, x):
        text, label = self.all_text[x], self.all_label[x]
        if self.is_dev:
            max_len = min(len(self.all_text[x]) + 2, 500)
        else:
            # 仅在训练阶段、启用增强、且 epoch>=7 的奇数轮做实体增强
            if self.enhance_data and self.epoch >= 7 and self.epoch % 2 == 1:
                ents = find_entities(label)
                text, label = self.entity_extend.entities_extend(text, label, ents)
            max_len = self.max_len
        text, label = text[:max_len - 2], label[:max_len - 2]

        x_len = len(text)
        assert len(text) == len(label)
        # 修复：原代码拼写为 add_special_token（缺 s），HuggingFace 会静默忽略
        # 导致 [CLS]/[SEP] 未加入，token 与 label 序列错位。
        text_idx = self.tokenizer.encode(text, add_special_tokens=True)
        label_idx = [self.tag2idx['<PAD>']] + [self.tag2idx[i] for i in label] + [self.tag2idx['<PAD>']]

        text_idx += [0] * (max_len - len(text_idx))
        label_idx += [self.tag2idx['<PAD>']] * (max_len - len(label_idx))
        return torch.tensor(text_idx), torch.tensor(label_idx), x_len

    def __len__(self) -> int:
        return len(self.all_text)
