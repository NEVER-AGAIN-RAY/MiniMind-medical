#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
科室分诊六分类专用数据集类 (triage_dataset.py)

沿用 lora_sentiment_20260914 的三条约定：
1. 严禁截断患者自述。超长样本一律在 prepare_data.py 阶段依据真实 token 长度筛除；
   若运行时仍出现超长样本，直接抛 ValueError，绝不在前端截掉输入。
2. 训练、验证、评测 100% 对齐同一个对话模板。
3. 监督机制：prompt 前缀与 padding 区全部置 -100，仅对科室名与结束符计算交叉熵。

与情感实验的关键差异——**标签 token 长度不等**：
    内科/外科/儿科/男科 = 4 tokens，妇产科 = 6 tokens，肿瘤科 = 7 tokens。
候选打分若直接比较各标签的 log-prob 之和，长标签会被系统性地压低分数（多乘几个
小于 1 的概率）。因此本实验的候选打分同时给出两个口径：
    · sum  —— 各 token log-prob 之和，与情感实验同口径，但在此处有长度偏置；
    · mean —— 每 token 平均 log-prob，消除长度偏置，本实验以此为主指标。
两者的差异本身就是一个诊断信号，evaluation 会把它一并记录下来。
"""

import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 候选标签的固定顺序：候选打分与混淆矩阵均以此为准，不要随意调换。
LABELS = ("内科", "外科", "妇产科", "儿科", "肿瘤科", "男科")

DEFAULT_INSTRUCTION_PROMPT = (
    "根据下面这段患者自述，判断应当挂哪个科室，只输出科室名称，不要输出其他内容。"
    "可选科室：内科、外科、妇产科、儿科、肿瘤科、男科。"
)


def normalize_raw_label(raw) -> str:
    """校验并归一化科室标签。标签来自目录名，是封闭集合。"""
    value = (str(raw) if raw is not None else "").strip()
    if value not in LABELS:
        raise ValueError(f"科室标签必须是 {LABELS} 之一，实际收到: {raw!r}")
    return value


def format_triage_prompt(ask: str, instruction: str = DEFAULT_INSTRUCTION_PROMPT) -> str:
    """构建统一规范的分诊输入 Prompt。"""
    ask = (ask or "").strip()
    if not ask:
        raise ValueError("患者自述为空，应在数据准备阶段筛除")
    return f"{instruction}\n患者自述：{ask}"


def build_prompt_str(
    ask: str,
    tokenizer: PreTrainedTokenizerFast,
    instruction: str = DEFAULT_INSTRUCTION_PROMPT,
) -> str:
    """套用对话模板，得到送入模型的完整 prompt 字符串。"""
    conversation = [{"role": "user", "content": format_triage_prompt(ask, instruction)}]
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


def build_triage_tokens(
    ask: str,
    label_text: str,
    tokenizer: PreTrainedTokenizerFast,
    instruction: str = DEFAULT_INSTRUCTION_PROMPT,
) -> tuple[list[int], list[int], str, str]:
    """构建单个样本的 prompt_ids 与 target_ids，同时返回原始字符串便于审计。"""
    prompt_str = build_prompt_str(ask, tokenizer, instruction)
    target_str = build_target_str(label_text, tokenizer)
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
    target_ids = tokenizer(target_str, add_special_tokens=False).input_ids
    return prompt_ids, target_ids, prompt_str, target_str


def measure_total_length(
    ask: str,
    label_text: str,
    tokenizer: PreTrainedTokenizerFast,
    instruction: str = DEFAULT_INSTRUCTION_PROMPT,
) -> int:
    """数据准备阶段用：返回 prompt + target 的真实 token 总长度。

    标签长度不等，因此长度预算按**最长标签**（肿瘤科）核算，
    否则同一条自述会出现「挂内科能进、挂肿瘤科超长」的不一致。
    """
    prompt_ids, _, _, _ = build_triage_tokens(ask, label_text, tokenizer, instruction)
    longest_target = max(
        len(tokenizer(build_target_str(label, tokenizer), add_special_tokens=False).input_ids)
        for label in LABELS
    )
    return len(prompt_ids) + longest_target


class TriageDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 320,
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

        prompt_ids, target_ids, _, _ = build_triage_tokens(
            ask=sample["ask"],
            label_text=sample["label_text"],
            tokenizer=self.tokenizer,
            instruction=self.instruction,
        )

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)

        # 严禁截断患者自述：超长必须在数据准备阶段就被筛除。
        if len(input_ids) > self.max_length:
            sid = sample.get("id", f"index_{index}")
            raise ValueError(
                f"样本超长拦截 [ID: {sid}]: 实际序列长度 {len(input_ids)} tokens 超过限制 "
                f"{self.max_length} tokens！为保护自述完整性，系统禁止截断输入，"
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
