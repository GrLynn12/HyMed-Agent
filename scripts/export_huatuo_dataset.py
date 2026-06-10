"""把 Huatuo 类 DatasetDict 导出为 FAISS 构建可用的 JSONL。

用法示例::

    python -m scripts.export_huatuo_dataset \
      --dataset-name huatuo_encyclopedia_qa \
      --split train \
      --output data/huatuo_encyclopedia_qa_train.jsonl

也支持本地 ``datasets.save_to_disk`` 目录::

    python -m scripts.export_huatuo_dataset \
      --dataset-path /path/to/huatuo_encyclopedia_qa \
      --split train \
      --output data/huatuo_encyclopedia_qa_train.jsonl
"""

from __future__ import annotations

import argparse
import json

from datasets import load_dataset, load_from_disk
from tqdm import tqdm


def _normalize(value) -> str:
    if isinstance(value, list):
        items = []
        for item in value:
            if isinstance(item, list):
                items.extend(str(x).strip() for x in item if str(x).strip())
            elif str(item).strip():
                items.append(str(item).strip())
        return "；".join(items)
    return str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Huatuo QA 数据为 JSONL")
    parser.add_argument("--dataset-name", default="huatuo_encyclopedia_qa", help="HuggingFace dataset 名称")
    parser.add_argument("--dataset-path", default=None, help="本地 datasets save_to_disk 目录")
    parser.add_argument("--split", default="train", help="导出的 split，如 train/validation/test")
    parser.add_argument("--output", default="/data0/grl_data/llm/rag/huatuo_5000.jsonl", help="输出 JSONL 路径")
    parser.add_argument("--limit", type=int, default=5000, help="最多导出多少条；0 表示全部")
    args = parser.parse_args()

    if args.dataset_path:
        dataset_dict = load_from_disk(args.dataset_path)
    elif args.dataset_name:
        dataset_dict = load_dataset(
            "FreedomIntelligence/huatuo_encyclopedia_qa",
            cache_dir="/data0/grl_data/llm/rag"
        )
    else:
        raise ValueError("请提供 --dataset-name 或 --dataset-path")

    dataset = dataset_dict[args.split]
    limit = args.limit or len(dataset)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in tqdm(dataset.select(range(min(limit, len(dataset))))):
            question = _normalize(item.get("question") or item.get("questions") or "")
            answer = _normalize(item.get("answer") or item.get("answers") or "")
            if not question and not answer:
                continue
            f.write(
                json.dumps(
                    {
                        "question": question,
                        "answer": answer,
                        "source": "huatuo_encyclopedia_qa",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
