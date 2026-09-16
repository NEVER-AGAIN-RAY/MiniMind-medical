#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
情感二分类专用数据集类 (sentiment_dataset.py)

核心原则（沿用 lora_medical_mcq_pilot 的约定）：
1. 严禁截断评论正文。超长样本一律在 prepare_data.py 阶段依据真实 token 长度筛除；
   若运行时仍出现超长样本，直接抛 ValueError，绝不在前端截掉输入。
2. 训练、验证、评测 100% 对齐同一个对话模板：
       tokenizer.apply_chat_template(
           [{"role": "user", "content": prompt_text}],
           tokenize=False, add_generation_prompt=True, open_thinking=False,
       )
3. 监督机制：prompt 前缀与 padding 区全部置 -100，仅对标签词与结束符计算交叉熵。

与医学实验的关键差异：标签是封闭的两个词（正面/负面），因此除生成式评测外
还可以做候选打分（forced choice）——直接比较两个标签的对数似然，永远不会
出现"格式不合规导致无法判分"的情况。
"""

import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_INSTRUCTION_PROMPT = "判断下面这条酒店评论的情感倾向，只输出「正面」或「负面」，不要输出其他内容。"

POSITIVE_LABEL = "正面"
NEGATIVE_LABEL = "负面"

# 候选标签的固定顺序：候选打分与混淆矩阵均以此为准，不要随意调换。
LABELS = (NEGATIVE_LABEL, POSITIVE_LABEL)

# ChnSentiCorp 原始 label 列：1 = 正面, 0 = 负面
RAW_LABEL_TO_TEXT = {1: POSITIVE_LABEL, 0: NEGATIVE_LABEL}


def normalize_raw_label(raw) -> str:
    """把原始 CSV 的 label 列（可能是 str 或 int）映射为标签文字。"""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法解析的 label 值: {raw!r}") from exc
    if value not in RAW_LABEL_TO_TEXT:
        raise ValueError(f"label 只允许 0 或 1，实际收到: {value}")
    return RAW_LABEL_TO_TEXT[value]


def format_sentiment_prompt(review: str, instruction: str = DEFAULT_INSTRUCTION_PROMPT) -> str:
    """构建统一规范的情感分类输入 Prompt。"""
    review = (review or "").strip()
    if not review:
        raise ValueError("评论正文为空，应在数据准备阶段筛除")
    return f"{instruction}\n评论：{review}"


def build_prompt_str(
    review: str,
    tokenizer: PreTrainedTokenizerFast,
    instruction: str = DEFAULT_INSTRUCTION_PROMPT,
) -> str:
    """套用对话模板，得到送入模型的完整 prompt 字符串。"""
    conversation = [{"role": "user", "content": format_sentiment_prompt(review, instruction)}]
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )


def build_target_str(label_text: str, tokenizer: PreTrainedTokenizerFast) -> str:
    """构建监督目标字符串。标签是封闭集合，这里顺带做一次合法性校验。"""
    if label_text not in LABELS:
        raise ValueError(f"标签必须是 {LABELS} 之一，实际收到: {label_text!r}")
    return f"{label_text}{tokenizer.eos_token}\n"


def build_sentiment_tokens(
    review: str,
    label_text: str,
    tokenizer: PreTrainedTokenizerFast,
    instruction: str = DEFAULT_INSTRUCTION_PROMPT,
) -> tuple[list[int], list[int], str, str]:
    """构建单个样本的 prompt_ids 与 target_ids，同时返回原始字符串便于审计。"""
    prompt_str = build_prompt_str(review, tokenizer, instruction)
    target_str = build_target_str(label_text, tokenizer)
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
    target_ids = tokenizer(target_str, add_special_tokens=False).input_ids
    return prompt_ids, target_ids, prompt_str, target_str


def measure_total_length(
    review: str,
    label_text: str,
    tokenizer: PreTrainedTokenizerFast,
    instruction: str = DEFAULT_INSTRUCTION_PROMPT,
) -> int:
    """数据准备阶段用：返回 prompt + target 的真实 token 总长度。"""
    prompt_ids, target_ids, _, _ = build_sentiment_tokens(review, label_text, tokenizer, instruction)
    return len(prompt_ids) + len(target_ids)


class SentimentDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 448,
        instruction: str = DEFAULT_INSTRUCTION_PROMPT,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction
        self.jsonl_path = Path(jsonl_path)

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.jsonl_path}")

        self.samples = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        if not self.samples:
            raise ValueError(f"数据集为空: {self.jsonl_path}")

        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self):
        return len(self.samples)

    def label_distribution(self) -> dict[str, int]:
        """返回各标签样本数，供训练脚本核对切分是否平衡。"""
        counts = {label: 0 for label in LABELS}
        for sample in self.samples:
            counts[sample["label_text"]] = counts.get(sample["label_text"], 0) + 1
        return counts

    def __getitem__(self, index):
        sample = self.samples[index]

        prompt_ids, target_ids, _, _ = build_sentiment_tokens(
            review=sample["review"],
            label_text=sample["label_text"],
            tokenizer=self.tokenizer,
            instruction=self.instruction,
        )

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)

        # 严禁截断评论正文：超长必须在数据准备阶段就被筛除。
        if len(input_ids) > self.max_length:
            sid = sample.get("id", f"index_{index}")
            raise ValueError(
                f"样本超长拦截 [ID: {sid}]: 实际序列长度 {len(input_ids)} tokens 超过限制 "
                f"{self.max_length} tokens！为保护评论完整性，系统禁止截断输入，"
                f"请在 prepare_data.py 中提前筛除。"
            )

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        supervised_tokens = [tok for tok, lbl in zip(input_ids, labels) if lbl != -100]
        assert supervised_tokens == list(target_ids), (
            f"目标监督 Token 不匹配: 实际 {supervised_tokens} vs 预期 {list(target_ids)}"
        )

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
