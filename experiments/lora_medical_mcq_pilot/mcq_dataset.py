#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单选题专用数据集类 (mcq_dataset.py)

核心原则：
1. 严禁截断题干！任何超长题目均在数据准备阶段依据真实 Token 长度筛除。
   若运行时出现超出 max_length 的样本，直接抛出 ValueError，绝不在前端截掉输入。
2. 训练、验证、评测 100% 对齐同一个对话模板：
   prompt_str = tokenizer.apply_chat_template(
       [{"role": "user", "content": prompt_text}],
       tokenize=False,
       add_generation_prompt=True,
       open_thinking=False,
   )
3. 支持四种目标输出：
   - letter: "正确字母 + 结束符"（例如 "A<|im_end|>\n"）
   - answer_text: "正确答案文字（字母）+ 结束符"（例如 "正确答案：甲状腺C细胞（C）。"）
4. 监督机制:
   - prompt 前缀全部置为 -100 (不计算 loss)
   - 仅对正确选项字母和结束符计算交叉熵 loss
   - padding 区域全部置为 -100
"""

import json
import os
import re
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_INSTRUCTION_PROMPT = "请直接输出正确选项的字母（如 A、B、C、D、E），不要输出多余解释。"
ANSWER_TEXT_INSTRUCTION_PROMPT = "请按“正确答案：选项内容（字母）。”的格式回答，不要输出其他解释。"
ANSWER_RATIONALE_INSTRUCTION_PROMPT = "请先给出正确答案的内容和字母，再依据医学知识进行解析。"
CONCISE_KNOWLEDGE_INSTRUCTION_PROMPT = "请直接回答正确的医学内容，并用一句话说明依据。"


def select_concise_knowledge(explanation: str, answer_text: str) -> str:
    """从原解析中取第一句包含正确答案文字的知识句，避免监督整段错误选项分析。"""
    explanation = (explanation or "").strip()
    answer_text = answer_text.strip()
    if not explanation:
        raise ValueError("concise_knowledge 模式要求样本包含非空 explanation")

    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?])", explanation) if s.strip()]
    normalized_answer = re.sub(r"\s+", "", answer_text)
    normalized_sentences = [(s, re.sub(r"\s+", "", s)) for s in sentences]
    selected = next((s for s, ns in normalized_sentences if normalized_answer in ns), "")
    if not selected:
        # 部分正确选项本身由多句话组成；此时取与答案最长分句匹配的解析句。
        answer_clauses = [
            re.sub(r"\s+", "", part)
            for part in re.split(r"[。！？!?；;]", answer_text)
            if len(re.sub(r"\s+", "", part)) >= 4
        ]
        matches = [
            (len(clause), sentence)
            for clause in answer_clauses
            for sentence, normalized_sentence in normalized_sentences
            if clause in normalized_sentence
        ]
        if matches:
            selected = max(matches, key=lambda item: item[0])[1]
        else:
            raise ValueError(f"解析中找不到正确答案文字，无法构造精简知识目标: {answer_text}")

    # 清除数据集中的“X对/X错”等判题标记，只保留医学陈述。
    selected = re.sub(r"[（(][A-EＡ-Ｅ][对错](?:，[^）)]*)?[）)]", "", selected).strip()
    return selected


def format_mcq_prompt(question: str, options: dict[str, str], instruction: str = DEFAULT_INSTRUCTION_PROMPT) -> str:
    """构建统一规范的单选题输入 Prompt"""
    lines = [question.strip()]
    for k in sorted(options.keys()):
        lines.append(f"{k}. {options[k]}")
    lines.append(instruction)
    return "\n".join(lines)


def build_mcq_tokens(
    question: str,
    options: dict[str, str],
    answer: str,
    tokenizer: PreTrainedTokenizerFast,
    instruction: str = DEFAULT_INSTRUCTION_PROMPT,
    target_mode: str = "letter",
    explanation: str = "",
) -> tuple[list[int], list[int], str, str]:
    """构建单个样本的完整 prompt_ids 和 target_ids，返回结构化 token 列表及原始字符串"""
    prompt_text = format_mcq_prompt(question, options, instruction)
    conversation = [{"role": "user", "content": prompt_text}]
    prompt_str = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    answer = answer.strip().upper()
    if target_mode == "letter":
        target_str = f"{answer}{tokenizer.eos_token}\n"
    elif target_mode == "answer_text":
        target_str = f"正确答案：{options[answer]}（{answer}）。{tokenizer.eos_token}\n"
    elif target_mode == "answer_rationale":
        explanation = explanation.strip() if explanation else ""
        if not explanation:
            raise ValueError("answer_rationale 模式要求样本包含非空 explanation")
        target_str = (
            f"正确答案：{options[answer]}（{answer}）。"
            f"解析：{explanation}{tokenizer.eos_token}\n"
        )
    elif target_mode == "concise_knowledge":
        answer_text = options[answer].strip()
        knowledge = select_concise_knowledge(explanation, answer_text)
        separator = "" if knowledge.startswith(answer_text) else "知识点："
        target_str = f"{answer_text}。{separator}{knowledge}{tokenizer.eos_token}\n"
    else:
        raise ValueError(f"不支持的 target_mode: {target_mode}")

    prompt_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
    target_ids = tokenizer(target_str, add_special_tokens=False).input_ids
    return prompt_ids, target_ids, prompt_str, target_str


class MedicalMCQDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 340,
        instruction: str = DEFAULT_INSTRUCTION_PROMPT,
        target_mode: str = "letter",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction
        self.target_mode = target_mode
        self.jsonl_path = Path(jsonl_path)

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.jsonl_path}")

        self.samples = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    item = json.loads(line)
                    self.samples.append(item)

        if len(self.samples) == 0:
            raise ValueError(f"数据集为空: {self.jsonl_path}")

        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        question = sample["question"]
        options = sample["options"]
        answer = sample["answer"]
        explanation = sample.get("explanation", "")

        # 构建 Prompt 和 Target
        prompt_ids, target_ids, _, _ = build_mcq_tokens(
            question=question,
            options=options,
            answer=answer,
            tokenizer=self.tokenizer,
            instruction=self.instruction,
            target_mode=self.target_mode,
            explanation=explanation,
        )

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)

        # 严格禁止截断题干！超出长度必须抛出异常
        if len(input_ids) > self.max_length:
            qid = sample.get("id", f"index_{index}")
            raise ValueError(
                f"样本超长拦截 [ID: {qid}]: 实际序列长度 {len(input_ids)} tokens 超过限制 {self.max_length} tokens！"
                f"为保护题干和选项完整性，系统禁止截断题干。请在 prepare_data.py 中提前筛除。"
            )

        # 填充到统一 max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        # 校验监督 token 完整性
        supervised_tokens = [tok for tok, lbl in zip(input_ids, labels) if lbl != -100]
        assert supervised_tokens == list(target_ids), (
            f"目标监督 Token 不匹配: 实际 {supervised_tokens} vs 预期 {list(target_ids)}"
        )

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
