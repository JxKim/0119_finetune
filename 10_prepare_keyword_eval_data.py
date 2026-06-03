"""
按照evalscope要求，准备数据集，用以进行评估测试
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DEFAULT_PROMPT_PREFIX = """/no_think
请从下面文本中抽取关键词。
要求：
1. 必须输出中文关键词，禁止翻译成英文。
2. 关键词应优先来自原文中的中文术语或规范中文概念。
3. 只输出关键词，不要输出解释、编号、JSON 或 Markdown。
4. 多个关键词用英文分号 ; 分隔。
5. 第一个关键词前不要加分号，最后一个关键词后不要加分号。
6. 不要输出整句、摘要或推理过程。
7. 如果原文包含标题，优先抽取标题和正文中反复出现的核心学术概念。

"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/keywords_data_test.jsonl")
    parser.add_argument(
        "--output",
        default="benchmark/custom_eval/text/qa/keyword_extraction.jsonl",
    )
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def convert_example(example: dict) -> dict:
    conversation = example["conversation"][0]
    query = f"{DEFAULT_PROMPT_PREFIX}{conversation['human']}"
    response = conversation["assistant"]

    item = {
        "query": query,
        "response": response,
    }
    if "conversation_id" in example:
        item["id"] = str(example["conversation_id"])
    if "dataset" in example:
        item["category"] = example["dataset"]
    return item


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "json",
        data_files=args.input,
        split="train",
    )
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    converted = dataset.map(
        convert_example,
        remove_columns=dataset.column_names,
    )

    converted.to_json(args.output,force_ascii=False)


if __name__ == "__main__":
    main()
