"""NER 模型训练入口。

用法::

    python -m medical_rag.ner.train

依赖：

* ``data/ner_data_aug.txt`` —— 由 ``scripts.build_ner_training_data`` 生成
* ``model/chinese-roberta-wwm-ext`` —— HuggingFace 预训练 BERT
* ``tmp_data/`` —— 缓存 tag2idx
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from seqeval.metrics import f1_score

from medical_rag.core.config import settings
from medical_rag.core.devices import resolve_torch_device
from medical_rag.core.logging import setup_logging

from .dataset import Nerdataset, get_data, build_tag2idx
from .inference import rule_find, tfidf_alignment, get_ner_result
from .model import Bert_Model

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练医疗实体识别模型")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=60, help="训练批大小")
    parser.add_argument("--max-length", type=int, default=50, help="最大序列长度")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="训练完成后进入交互式 NER 测试",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging()
    cache_model = settings.NER_CHECKPOINT

    all_text, all_label = get_data(os.path.join(settings.DATA_DIR, 'ner_data_aug.txt'))
    train_text, dev_text, train_label, dev_label = train_test_split(
        all_text, all_label, test_size=0.02, random_state=42
    )

    # 加载太慢了，预处理一下
    tag_path = os.path.join(settings.TMP_DIR, 'tag2idx.npy')
    if os.path.exists(tag_path):
        with open(tag_path, 'rb') as f:
            tag2idx = pickle.load(f)
    else:
        tag2idx = build_tag2idx(all_label)
        os.makedirs(settings.TMP_DIR, exist_ok=True)
        with open(tag_path, 'wb') as f:
            pickle.dump(tag2idx, f)

    idx2tag = list(tag2idx)

    max_len = args.max_length
    epoch = args.epochs
    batch_size = args.batch_size
    hidden_size = 128
    bi = True
    model_name = settings.NER_MODEL_NAME
    tokenizer = BertTokenizer.from_pretrained(model_name)
    lr = 1e-5
    is_train = True

    device = resolve_torch_device(settings.COMPUTE_DEVICE)

    train_dataset = Nerdataset(train_text, train_label, tokenizer, max_len, tag2idx, enhance_data=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = Nerdataset(dev_text, dev_label, tokenizer, max_len, tag2idx, is_dev=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    model = Bert_Model(model_name, hidden_size, len(tag2idx), bi)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bestf1 = -1

    if is_train:
        for e in range(epoch):
            # 把当前 epoch 注入数据集，控制实体增强触发时机
            train_dataset.set_epoch(e)
            loss_sum = 0
            ba = 0
            for x, y, batch_len in tqdm(train_dataloader):
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                loss = model(x, y)
                loss.backward()
                opt.step()
                loss_sum += loss
                ba += 1

            all_pre, all_label_eval = [], []
            for x, y, batch_len in tqdm(dev_dataloader):
                assert len(x) == len(y)
                x = x.to(device)
                pre = model(x)
                pre = [idx2tag[i] for i in pre[1:batch_len + 1]]
                all_pre.append(pre)
                label = [idx2tag[i] for i in y[0][1:batch_len + 1]]
                all_label_eval.append(label)
            f1 = f1_score(all_pre, all_label_eval)
            if f1 > bestf1:
                bestf1 = f1
                logger.info('e=%d, loss=%.5f, f1=%.5f ----> best', e, loss_sum / ba, f1)
                os.makedirs(settings.MODEL_DIR, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(settings.MODEL_DIR, f'{cache_model}.pt'))
            else:
                logger.info('e=%d, loss=%.5f, f1=%.5f', e, loss_sum / ba, f1)

    if args.interactive:
        rule = rule_find()
        tfidf_r = tfidf_alignment()
        while True:
            sentence = input("请输入（Ctrl+C 退出）:")
            print(
                get_ner_result(
                    model,
                    tokenizer,
                    sentence,
                    rule,
                    tfidf_r,
                    device,
                    idx2tag,
                )
            )


if __name__ == "__main__":
    main()
