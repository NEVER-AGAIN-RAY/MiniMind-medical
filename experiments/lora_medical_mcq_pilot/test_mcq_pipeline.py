#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单选题实验离线单元测试集 (test_mcq_pipeline.py) - 升级隔离版

注意：严格遵循本地执行边界，本地不执行该测试！
仅供在云端执行时，在 Step 0 运行，验证 7 项问题的修复正确性。

修复项：
1. [Issue 2 修复]：使用隔离的临时目录与 Mock 数据测试 prepare_data_pipeline，
   绝不触碰真实磁盘数据目录，绝不依赖真实 CSV 文件是否存在。
2. [Issue 2 & 5 修复]：真实验证 train_mcq_lora 的检查点初值保证、改善判断逻辑及防覆盖拦截。
3. 真实覆盖 Issue 1 (截断拦截)、Issue 3 (对比完整性拦截)、Issue 4 (近重复与文件缺失)、Issue 6 (配置生效)。
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from experiments.lora_medical_mcq_pilot.mcq_dataset import (
    ANSWER_RATIONALE_INSTRUCTION_PROMPT,
    ANSWER_TEXT_INSTRUCTION_PROMPT,
    DEFAULT_INSTRUCTION_PROMPT,
    MedicalMCQDataset,
    build_mcq_tokens,
    format_mcq_prompt,
)
from experiments.lora_medical_mcq_pilot.prepare_data import (
    jaccard_similarity,
    norm_q,
    parse_options,
)


class MockTokenizer:
    """轻量模拟 Tokenizer，用于无模型依赖的离线逻辑测试"""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.eos_token = "<|im_end|>"
        self.bos_token = "<|im_start|>"

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=True, open_thinking=False):
        user_msg = conv[0]["content"]
        res = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if add_generation_prompt:
            res += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return res

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(c) % 1000 + 10 for c in text]
        class TokenizerOutput:
            def __init__(self, ids):
                self.input_ids = ids
        return TokenizerOutput(ids)


class MockSimpleModel(nn.Module):
    """轻量模拟模型，用于测试 LoRA 保存和检查点管理"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)
        self.device = torch.device("cpu")


class TestMCQPipelineIssues(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()

    # 1. 针对 Issue 1 的测试：禁止截掉题干
    def test_issue1_prohibit_stem_truncation(self):
        long_q = "这是一个非常详细且包含大量临床主诉描述的极长医学单选题干" * 10
        opts = {"A": "选项A", "B": "选项B"}
        sample = {"id": "test_001", "question": long_q, "options": opts, "answer": "A"}

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            tmp_path = f.name

        try:
            ds = MedicalMCQDataset(tmp_path, self.tokenizer, max_length=50)
            with self.assertRaises(ValueError) as ctx:
                _ = ds[0]
            self.assertIn("系统禁止截断题干", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_answer_text_target_contains_medical_answer_before_letter(self):
        prompt_ids, target_ids, _, target = build_mcq_tokens(
            question="血液中的降钙素主要由哪种细胞产生？",
            options={"A": "甲状腺滤泡细胞", "C": "甲状腺C细胞"},
            answer="C",
            tokenizer=self.tokenizer,
            instruction=ANSWER_TEXT_INSTRUCTION_PROMPT,
            target_mode="answer_text",
        )
        self.assertTrue(prompt_ids)
        self.assertTrue(target_ids)
        self.assertTrue(target.startswith("正确答案：甲状腺C细胞（C）。"))

    def test_answer_rationale_target_contains_source_explanation(self):
        _, _, _, target = build_mcq_tokens(
            question="血液中的降钙素主要由哪种细胞产生？",
            options={"A": "甲状腺滤泡细胞", "C": "甲状腺C细胞"},
            answer="C",
            tokenizer=self.tokenizer,
            instruction=ANSWER_RATIONALE_INSTRUCTION_PROMPT,
            target_mode="answer_rationale",
            explanation="降钙素由甲状腺C细胞分泌。",
        )
        self.assertTrue(target.startswith("正确答案：甲状腺C细胞（C）。解析："))
        self.assertIn("降钙素由甲状腺C细胞分泌。", target)

    # 2. 针对 Issue 2 的测试：冒烟模式安全拦截 (冒烟禁止测正式测试集)
    def test_issue2_smoke_test_guard(self):
        from experiments.lora_medical_mcq_pilot.eval_mcq import run_evaluation
        mock_args = SimpleNamespace(smoke=True, split="test", run_dir=None, overwrite=False)

        with self.assertRaises(ValueError) as ctx:
            run_evaluation(mock_args, mode="base")
        self.assertIn("冒烟模式 (--smoke) 严禁在正式测试集 (--split test) 上运行", str(ctx.exception))

    # 3. 针对 Issue 3 的测试：前后对比的完整性、清单核验、LoRA 选项核验与溯源校验
    def test_issue3_strict_comparison_check(self):
        from experiments.lora_medical_mcq_pilot import eval_mcq
        from experiments.lora_medical_mcq_pilot.eval_mcq import run_comparison

        # 3.1 验证混用冒烟与正式时立即报错
        b_res = {
            "is_smoke": False,
            "is_limited": False,
            "split": "test",
            "dataset": {"md5": "111"},
            "weights": {"base_weight_md5": "aaa", "lora_weight_path": None},
            "generation_config": {},
            "runtime_env": {},
            "details": [],
        }
        l_res = {
            "is_smoke": True,
            "is_limited": False,
            "split": "test",
            "dataset": {"md5": "111"},
            "weights": {"base_weight_md5": "aaa", "lora_weight_path": "best.pth"},
            "generation_config": {},
            "runtime_env": {},
            "details": [],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            eval_dir = Path(tmp_dir) / "eval_results"
            eval_dir.mkdir(parents=True)
            with open(eval_dir / "base_test_results.json", "w") as f:
                json.dump(b_res, f)
            with open(eval_dir / "lora_test_results.json", "w") as f:
                json.dump(l_res, f)

            mock_args = SimpleNamespace(smoke=False, split="test", run_dir=tmp_dir, overwrite=False)

            with self.assertRaises(ValueError) as ctx:
                run_comparison(mock_args)
            self.assertIn("禁止将冒烟评测与正式评测进行对比", str(ctx.exception))

        # 3.2 验证 manifest.json 缺失与哈希不符校验
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            exp_root = tmp_root / "experiments" / "lora_medical_mcq_pilot"
            data_dir = exp_root / "data" / "smoke"
            data_dir.mkdir(parents=True)
            split_file = data_dir / "val.jsonl"
            item_data = {"id": "q1", "question": "测试题干", "options": {"A": "甲", "B": "乙"}, "answer": "A"}
            split_file.write_text(json.dumps(item_data, ensure_ascii=False) + "\n", encoding="utf-8")
            split_md5 = eval_mcq.md5_file(split_file)

            run_dir = exp_root / "runs" / "smoke_run"
            eval_dir = run_dir / "eval_results"
            ckpt_dir = run_dir / "checkpoints"
            eval_dir.mkdir(parents=True)
            ckpt_dir.mkdir(parents=True)

            b_smoke = {
                "is_smoke": True,
                "is_limited": False,
                "split": "val",
                "dataset": {"md5": split_md5},
                "weights": {"base_weight_md5": "base_md5_123", "lora_weight_path": None},
                "generation_config": {"do_sample": False, "open_thinking": False, "max_new_tokens": 16, "instruction": "prompt"},
                "runtime_env": {"device": "cpu", "dtype": "float32"},
                "details": [{
                    "id": "q1", "question": "测试题干", "options": {"A": "甲", "B": "乙"}, "official_answer": "A",
                    "is_correct": True, "predicted_letter": "A", "raw_model_output": "A", "is_format_compliant": True,
                    "natural_termination": True, "truncated": False
                }],
            }
            l_smoke = {
                "is_smoke": True,
                "is_limited": False,
                "split": "val",
                "dataset": {"md5": split_md5},
                "weights": {"base_weight_md5": "base_md5_123", "lora_weight_path": "best.pth", "lora_weight_md5": "lora_md5_123"},
                "generation_config": {"do_sample": False, "open_thinking": False, "max_new_tokens": 16, "instruction": "prompt"},
                "runtime_env": {"device": "cpu", "dtype": "float32"},
                "details": [{
                    "id": "q1", "question": "测试题干", "options": {"A": "甲", "B": "乙"}, "official_answer": "A",
                    "is_correct": True, "predicted_letter": "A", "raw_model_output": "A", "is_format_compliant": True,
                    "natural_termination": True, "truncated": False
                }],
            }
            with open(eval_dir / "base_val_results.json", "w") as f:
                json.dump(b_smoke, f)
            with open(eval_dir / "lora_val_results.json", "w") as f:
                json.dump(l_smoke, f)

            orig_eval_root = eval_mcq.ROOT
            try:
                eval_mcq.ROOT = tmp_root
                mock_smoke_args = SimpleNamespace(smoke=True, split="val", run_dir=str(run_dir), overwrite=False)

                # (1) 清单文件不存在时拦截
                with self.assertRaises(FileNotFoundError) as ctx:
                    run_comparison(mock_smoke_args)
                self.assertIn("数据清单文件不存在", str(ctx.exception))

                # (2) 清单哈希不符时拦截
                manifest_p = data_dir / "manifest.json"
                manifest_p.write_text(json.dumps({
                    "output_hashes": {"smoke": {"val_jsonl_md5": "tampered_md5"}}
                }))
                with self.assertRaises(ValueError) as ctx:
                    run_comparison(mock_smoke_args)
                self.assertIn("数据清单校验失败", str(ctx.exception))

                # (3) 修复清单哈希，但篡改 LoRA 产物中的 options 字段，验证双向比对拦截
                manifest_p.write_text(json.dumps({
                    "output_hashes": {"smoke": {"val_jsonl_md5": split_md5, "train_jsonl_md5": "train_md5"}}
                }))
                l_corrupted_opts = dict(l_smoke)
                l_corrupted_opts["details"] = [{
                    "id": "q1", "question": "测试题干", "options": {"A": "篡改选项甲", "B": "乙"}, "official_answer": "A",
                    "is_correct": True, "predicted_letter": "A", "raw_model_output": "A", "is_format_compliant": True,
                    "natural_termination": True, "truncated": False
                }]
                with open(eval_dir / "lora_val_results.json", "w") as f:
                    json.dump(l_corrupted_opts, f)

                with self.assertRaises(ValueError) as ctx:
                    run_comparison(mock_smoke_args)
                self.assertIn("LoRA 题目 [q1] 选项字典与锁定数据不符", str(ctx.exception))
            finally:
                eval_mcq.ROOT = orig_eval_root

    # 4. 针对 Issue 4 的测试：完全在临时目录中隔离运行数据流，验证去重与缺失排除文件报错
    def test_issue4_deduplication_and_missing_exclusion_file_isolated(self):
        from experiments.lora_medical_mcq_pilot import prepare_data

        # 4.1 近重复相似度检验
        q1 = "下列关于急性心肌梗死的临床表现，错误的是"
        q2 = "下列关于急性心肌梗死的临床表现中，错误的是"
        from experiments.lora_medical_mcq_pilot.prepare_data import get_char_bigrams
        sim = jaccard_similarity(get_char_bigrams(q1), get_char_bigrams(q2))
        self.assertGreaterEqual(sim, 0.85)

        # 4.2 全隔离测试原评测排除文件缺失检测与数据管线
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            tmp_exp = tmp_root / "experiments" / "lora_medical_mcq_pilot"
            tmp_raw = tmp_exp / "data" / "raw"
            tmp_raw.mkdir(parents=True)

            # 写入虚拟的 train.csv, val.csv, test.csv（互不重复题，充足供给，避免去重后题目不足）
            def write_mock_csv(p: Path, count: int, prefix: str):
                with open(p, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Question", "Options", "Answer", "Explanation"])
                    for i in range(count):
                        opts = f"A 诊断A{i}\nB 诊断B{i}\nC 诊断C{i}\nD 诊断D{i}"
                        writer.writerow([f"{prefix}病症{i}详细主诉描述", opts, "A", f"解释{i}"])

            write_mock_csv(tmp_raw / "train.csv", 15, "训练集特有")
            write_mock_csv(tmp_raw / "val.csv", 10, "验证集特有")
            write_mock_csv(tmp_raw / "test.csv", 10, "测试集特有")

            # 模拟缺失排除文件的配置
            mock_cfg = {
                "experiment_name": "lora_medical_mcq_pilot",
                "base_model": {"model_dir": "model"},
                "sequence_config": {"max_seq_len": 340},
                "data_config": {
                    "source": "CMExam",
                    "official_urls": {"train": "", "val": "", "test": ""},
                    "local_fallback_val_raw": "non_existent_val.csv",
                    "original_eval_exclusion_path": "missing_eval_dir/missing_questions.jsonl",
                    "original_qa_exclusion_path": "missing_qa.jsonl",
                    "target_counts": {
                        "formal": {"train": 5, "val": 2, "test": 2},
                        "smoke": {"train": 2, "val": 1},
                    },
                    "deduplication": {"near_duplicate_jaccard_threshold": 0.85},
                    "instruction_prompt": DEFAULT_INSTRUCTION_PROMPT,
                    "seed": 42,
                },
                "run_paths": {
                    "experiments_root": "experiments/lora_medical_mcq_pilot",
                    "formal_dir": "experiments/lora_medical_mcq_pilot/runs/formal",
                    "smoke_dir": "experiments/lora_medical_mcq_pilot/runs/smoke",
                },
            }

            orig_root = prepare_data.ROOT
            orig_load_cfg = prepare_data.load_config
            try:
                prepare_data.ROOT = tmp_root
                prepare_data.load_config = lambda: mock_cfg

                mock_prep_args = SimpleNamespace(
                    train_count=5,
                    val_count=2,
                    test_count=2,
                    max_seq_len=340,
                    seed=42,
                    download=False,
                    overwrite=True,
                )

                # 验证排除文件缺失时直接报错阻断
                with self.assertRaises(FileNotFoundError) as ctx:
                    prepare_data.prepare_data_pipeline(mock_prep_args)
                self.assertIn("原评测排除文件不存在", str(ctx.exception))

                # 补齐排除文件，验证管线在隔离目录下能顺利构建产物且不污染真实目录
                eval_p = tmp_root / "missing_eval_dir" / "missing_questions.jsonl"
                eval_p.parent.mkdir(parents=True)
                eval_p.write_text(json.dumps({"question": "患者主诉病症999"}) + "\n")

                # Mock AutoTokenizer.from_pretrained
                from unittest.mock import patch
                with patch("experiments.lora_medical_mcq_pilot.prepare_data.AutoTokenizer.from_pretrained", return_value=self.tokenizer):
                    prepare_data.prepare_data_pipeline(mock_prep_args)

                # 检查隔离产物是否生成
                self.assertTrue((tmp_exp / "data" / "formal" / "train.jsonl").exists())
                self.assertTrue((tmp_exp / "data" / "formal" / "manifest.json").exists())
                self.assertTrue((tmp_exp / "data" / "smoke" / "train.jsonl").exists())
                # 确认 smoke 目录绝无 test.jsonl
                self.assertFalse((tmp_exp / "data" / "smoke" / "test.jsonl").exists())
            finally:
                prepare_data.ROOT = orig_root
                prepare_data.load_config = orig_load_cfg

    # 4.3 独立测试重复拦截与可用题目不足拦截用例
    def test_issue4_duplicate_rejection_and_insufficient_pool(self):
        from experiments.lora_medical_mcq_pilot import prepare_data

        # (1) 集合内部精确重复与近重复剔除单元测试
        duplicate_rows = [
            {"Question": "急性心肌梗死典型临床表现", "Options": "A 诊断A\nB 诊断B", "Answer": "A", "Explanation": ""},
            {"Question": "急性心肌梗死典型临床表现", "Options": "A 诊断A\nB 诊断B", "Answer": "A", "Explanation": ""},  # 精确重复
            {"Question": "急性心肌梗死典型临床表现等", "Options": "A 诊断A\nB 诊断B", "Answer": "A", "Explanation": ""}, # 近重复 (Jaccard >= 0.85)
            {"Question": "完全不同的其他临床疾病主诉症状描述", "Options": "A 诊断A\nB 诊断B", "Answer": "A", "Explanation": ""},
        ]
        cands, stats = prepare_data.filter_and_dedup_split(
            rows=duplicate_rows,
            split_name="test_dedup",
            tokenizer=self.tokenizer,
            max_seq_len=340,
            instruction=DEFAULT_INSTRUCTION_PROMPT,
            jaccard_threshold=0.85,
            excluded_norm_hashes=set(),
            excluded_bigrams_list=[],
            global_seen_hashes=set(),
            global_seen_bigrams=[],
        )
        self.assertEqual(stats["internal_exact_duplicate"], 1)
        self.assertEqual(stats["internal_near_duplicate"], 1)
        self.assertEqual(stats["passed"], 2)
        self.assertEqual(len(cands), 2)

        # (2) 跨集合题目重复导致可用题目不足时的管线拦截测试
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            tmp_exp = tmp_root / "experiments" / "lora_medical_mcq_pilot"
            tmp_raw = tmp_exp / "data" / "raw"
            tmp_raw.mkdir(parents=True)

            def write_mock_csv(p: Path, questions: list[str]):
                with open(p, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Question", "Options", "Answer", "Explanation"])
                    for i, q in enumerate(questions):
                        opts = f"A 诊断A{i}\nB 诊断B{i}"
                        writer.writerow([q, opts, "A", f"解释{i}"])

            train_qs = [f"重叠题干_{i}" for i in range(5)]
            write_mock_csv(tmp_raw / "train.csv", train_qs)
            # 验证集题目全部与训练集题目相同（跨集完全重复）
            write_mock_csv(tmp_raw / "val.csv", train_qs[:3])
            write_mock_csv(tmp_raw / "test.csv", [f"测试独有_{i}" for i in range(5)])

            # 排除文件
            eval_p = tmp_root / "missing_eval_dir" / "missing_questions.jsonl"
            eval_p.parent.mkdir(parents=True)
            eval_p.write_text(json.dumps({"question": "患者主诉病症999"}) + "\n")

            mock_cfg = {
                "experiment_name": "lora_medical_mcq_pilot",
                "base_model": {"model_dir": "model"},
                "sequence_config": {"max_seq_len": 340},
                "data_config": {
                    "source": "CMExam",
                    "official_urls": {"train": "", "val": "", "test": ""},
                    "local_fallback_val_raw": "non_existent_val.csv",
                    "original_eval_exclusion_path": "missing_eval_dir/missing_questions.jsonl",
                    "original_qa_exclusion_path": "missing_qa.jsonl",
                    "target_counts": {
                        "formal": {"train": 5, "val": 2, "test": 2},
                        "smoke": {"train": 2, "val": 1},
                    },
                    "deduplication": {"near_duplicate_jaccard_threshold": 0.85},
                    "instruction_prompt": DEFAULT_INSTRUCTION_PROMPT,
                    "seed": 42,
                },
                "run_paths": {
                    "experiments_root": "experiments/lora_medical_mcq_pilot",
                    "formal_dir": "experiments/lora_medical_mcq_pilot/runs/formal",
                    "smoke_dir": "experiments/lora_medical_mcq_pilot/runs/smoke",
                },
            }

            orig_root = prepare_data.ROOT
            orig_load_cfg = prepare_data.load_config
            try:
                prepare_data.ROOT = tmp_root
                prepare_data.load_config = lambda: mock_cfg

                mock_args = SimpleNamespace(
                    train_count=5,
                    val_count=2,
                    test_count=2,
                    max_seq_len=340,
                    seed=42,
                    download=False,
                    overwrite=True,
                )

                from unittest.mock import patch
                with patch("experiments.lora_medical_mcq_pilot.prepare_data.AutoTokenizer.from_pretrained", return_value=self.tokenizer):
                    with self.assertRaises(ValueError) as ctx:
                        prepare_data.prepare_data_pipeline(mock_args)
                    self.assertIn("验证集可用题目不足", str(ctx.exception))
            finally:
                prepare_data.ROOT = orig_root
                prepare_data.load_config = orig_load_cfg

    # 5. 针对 Issue 3 & Issue 5 的测试：真实验证 CheckpointTracker 选优逻辑与重跑防覆盖拦截
    def test_issue5_train_checkpoint_logic_and_rerun_protection(self):
        from experiments.lora_medical_mcq_pilot import train_mcq_lora
        from experiments.lora_medical_mcq_pilot.train_mcq_lora import CheckpointTracker
        from model.model_lora import apply_lora

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "runs" / "test_run"
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)

            model = MockSimpleModel()
            apply_lora(model, rank=4)

            # 5.1 验证 Step 0 初始化：初值恒存，step_0_lora 与 best_lora 同步存在且内容一致
            tracker = CheckpointTracker(ckpt_dir=ckpt_dir, model=model, init_val_loss=2.50)
            self.assertTrue(tracker.step_0_ckpt.exists())
            self.assertTrue(tracker.best_ckpt.exists())
            self.assertEqual(tracker.best_val_loss, 2.50)
            self.assertEqual(tracker.best_step, 0)
            self.assertFalse(tracker.improved)
            # 加载两个参数字典，逐项比较名称、形状和数值（参数完全一致）
            step0_sd = torch.load(tracker.step_0_ckpt, map_location="cpu")
            best_sd = torch.load(tracker.best_ckpt, map_location="cpu")
            self.assertEqual(set(step0_sd.keys()), set(best_sd.keys()))
            for k in step0_sd:
                self.assertEqual(step0_sd[k].shape, best_sd[k].shape)
                self.assertTrue(torch.equal(step0_sd[k], best_sd[k]))

            # 文件 MD5 用于追踪同一个文件 (best_lora.pth) 在后续训练中是否发生改变
            init_best_md5 = train_mcq_lora.md5_file(tracker.best_ckpt)

            # 5.2 验证未改善分支 (Loss 上升/持平)：不触发更新，improved 保持 False，选定保留 Step 0
            is_better_1 = tracker.step(global_step=1, val_loss=2.65)
            self.assertFalse(is_better_1)
            self.assertFalse(tracker.improved)
            self.assertEqual(tracker.best_step, 0)
            self.assertEqual(tracker.best_val_loss, 2.50)
            # best_lora.pth 权重未被污染，MD5 仍与初始化时一致
            self.assertEqual(train_mcq_lora.md5_file(tracker.best_ckpt), init_best_md5)

            # 5.3 验证改善分支 (Loss 下降)：触发更新，improved 变为 True，更新 best_lora.pth
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.add_(0.05)
            is_better_2 = tracker.step(global_step=2, val_loss=2.10)
            self.assertTrue(is_better_2)
            self.assertTrue(tracker.improved)
            self.assertEqual(tracker.best_step, 2)
            self.assertEqual(tracker.best_val_loss, 2.10)
            new_best_md5 = train_mcq_lora.md5_file(tracker.best_ckpt)
            self.assertNotEqual(new_best_md5, init_best_md5)

            # 验证新 best_lora.pth 的权重参数值确实已更新且与 step_0 不相同
            updated_best_sd = torch.load(tracker.best_ckpt, map_location="cpu")
            has_diff = any(not torch.equal(step0_sd[k], updated_best_sd[k]) for k in step0_sd)
            self.assertTrue(has_diff)

            # 5.4 验证 finalize 归档元数据完整性
            summary_info = tracker.finalize(model)
            self.assertTrue(summary_info["improved"])
            self.assertEqual(summary_info["best_step"], 2)
            self.assertEqual(summary_info["selected_checkpoint"]["source_step"], 2)
            self.assertEqual(summary_info["selected_checkpoint"]["md5"], new_best_md5)
            self.assertTrue(tracker.final_ckpt.exists())

            # 5.5 [Issue 3 修复验证] 验证当已有产物时，重跑在未加 --overwrite 时被拦截
            # 使用 SimpleNamespace 彻底解决类作用域 NameError 问题
            mock_train_args = SimpleNamespace(
                smoke=True,
                run_dir=str(run_dir),
                train_path=None,
                val_path=None,
                base_weight=None,
                epochs=1,
                batch_size=2,
                learning_rate=1e-4,
                weight_decay=0.01,
                accumulation_steps=1,
                grad_clip=1.0,
                max_seq_len=340,
                lora_rank=4,
                log_interval=1,
                val_interval=1,
                dtype="float32",
                device="cpu",
                seed=42,
                num_workers=0,
                overwrite=False,
            )

            with self.assertRaises(FileExistsError) as ctx:
                train_mcq_lora.train(mock_train_args)
            self.assertIn("运行目录已存在先前执行留存的产物", str(ctx.exception))

    # 6. 针对 Issue 6 的测试：统一配置有效性及排除无用参数
    def test_issue6_config_effective(self):
        with open(HERE / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertNotIn("lora_alpha", cfg.get("lora_config", {}))
        self.assertIn("training_hyperparameters", cfg)
        self.assertIn("smoke_hyperparameters", cfg)


if __name__ == "__main__":
    unittest.main()
