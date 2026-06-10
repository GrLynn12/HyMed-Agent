"""NER 模型定义：BERT 特征抽取 + 双向 RNN 序列建模 + 线性分类。

.. note::
   实际网络结构使用 ``nn.RNN`` 而非 README 中提及的 ``nn.LSTM``——属性名 ``self.gru``
   也是历史遗留命名，并非真的 GRU。后续若要换成 LSTM/GRU，建议同时修正命名。
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import BertModel


class Bert_Model(nn.Module):
    """BERT + 双向 RNN + Linear 分类头。

    :param model_name: HuggingFace BERT 路径（如 ``model/chinese-roberta-wwm-ext``）
    :param hidden_size: RNN 隐藏维度
    :param tag_num: 标签数（含 ``<PAD>``）
    :param bi: 是否双向
    """

    def __init__(self, model_name: str, hidden_size: int, tag_num: int, bi: bool) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # 历史命名为 self.gru，但实际是 nn.RNN；保持原属性名以兼容已训练的权重 (.pt)
        self.gru = nn.RNN(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=bi,
        )
        if bi:
            self.classifier = nn.Linear(hidden_size * 2, tag_num)
        else:
            self.classifier = nn.Linear(hidden_size, tag_num)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, label=None):
        bert_0, _ = self.bert(x, attention_mask=(x > 0), return_dict=False)
        gru_0, _ = self.gru(bert_0)
        pre = self.classifier(gru_0)
        if label is not None:
            loss = self.loss_fn(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))
            return loss
        return torch.argmax(pre, dim=-1).squeeze(0)
