#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地测试脚本：在不加载真实模型的情况下验证：
1. MCQ 答案抽取规则与边界情况
2. evaluate_val_loss 数学加权与 mask 机制
3. run_comparison 对比报告生成、一致性校验与统计指标
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from run_eval import extract_mcq_choice, run_comparison, format_mcq_prompt, generate_single_response
from model.model_minimind import get_banned_ngram_tokens
from trainer.trainer_utils import evaluate_val_loss


class TestMCQExtraction(unittest.TestCase):
    def test_single_letter(self):
        opts = ["A", "B", "C", "D", "E"]
        self.assertEqual(extract_mcq_choice("A", opts), "A")
        self.assertEqual(extract_mcq_choice(" B ", opts), "B")
        self.assertEqual(extract_mcq_choice("C.", opts), "C")
        self.assertEqual(extract_mcq_choice("D、", opts), "D")
        self.assertEqual(extract_mcq_choice("(A)", opts), "A")
        self.assertEqual(extract_mcq_choice("[B]", opts), "B")
        self.assertEqual(extract_mcq_choice("【C】", opts), "C")
        self.assertEqual(extract_mcq_choice("**D**", opts), "D")
        self.assertEqual(extract_mcq_choice("E\n", opts), "E")
        self.assertEqual(extract_mcq_choice("a", opts), "A")
        self.assertEqual(extract_mcq_choice("b.", opts), "B")

    def test_prefixes(self):
        opts = ["A", "B", "C", "D", "E"]
        self.assertEqual(extract_mcq_choice("答案：B", opts), "B")
        self.assertEqual(extract_mcq_choice("答案: B", opts), "B")
        self.assertEqual(extract_mcq_choice("答案是 C", opts), "C")
        self.assertEqual(extract_mcq_choice("答案为 A", opts), "A")
        self.assertEqual(extract_mcq_choice("正确答案是 D", opts), "D")
        self.assertEqual(extract_mcq_choice("正确答案为 E", opts), "E")
        self.assertEqual(extract_mcq_choice("选A", opts), "A")
        self.assertEqual(extract_mcq_choice("选择 B", opts), "B")
        self.assertEqual(extract_mcq_choice("故选 C", opts), "C")
        self.assertEqual(extract_mcq_choice("所以选 D", opts), "D")
        self.assertEqual(extract_mcq_choice("本题选 E", opts), "E")
        self.assertEqual(extract_mcq_choice("应当选 A", opts), "A")

    def test_letter_with_content(self):
        opts = ["A", "B", "C", "D", "E"]
        self.assertEqual(extract_mcq_choice("A. 遗传因素", opts), "A")
        self.assertEqual(extract_mcq_choice("B: 微生物", opts), "B")
        self.assertEqual(extract_mcq_choice("C 饮食习惯", opts), "C")

    def test_invalid_and_conflicts(self):
        opts = ["A", "B", "C", "D", "E"]
        # 冲突选择
        self.assertIsNone(extract_mcq_choice("选A或者B", opts))
        self.assertIsNone(extract_mcq_choice("AB", opts))
        self.assertIsNone(extract_mcq_choice("A/B", opts))
        self.assertIsNone(extract_mcq_choice("答案可能是A也可能是C", opts))
        # 选项超出有效集合
        self.assertIsNone(extract_mcq_choice("F", opts))
        self.assertIsNone(extract_mcq_choice("选F", opts))
        # 纯文本没有明确选项开头
        self.assertIsNone(extract_mcq_choice("龋齿是一种常见的口腔感染性疾病...", opts))
        self.assertIsNone(extract_mcq_choice("", opts))
        self.assertIsNone(extract_mcq_choice("   ", opts))
        self.assertIsNone(extract_mcq_choice("After careful review...", opts))


class TestGenerationConfig(unittest.TestCase):
    def test_bans_token_that_would_repeat_ngram(self):
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 2, 3],
                [1, 2, 3, 4, 5, 6],
            ]
        )
        self.assertEqual(get_banned_ngram_tokens(input_ids, 3), [[4], []])

    def test_disabled_ngram_blocking_bans_nothing(self):
        input_ids = torch.tensor([[1, 2, 1, 2]])
        self.assertEqual(get_banned_ngram_tokens(input_ids, 0), [[]])

    def test_repetition_controls_are_forwarded_to_generate(self):
        class TokenBatch(dict):
            pass

        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def apply_chat_template(self, *args, **kwargs):
                return "prompt"

            def __call__(self, *args, **kwargs):
                return TokenBatch(
                    input_ids=torch.tensor([[1, 3]]),
                    attention_mask=torch.tensor([[1, 1]]),
                )

            def decode(self, token_ids, skip_special_tokens=True):
                return "answer"

        class DummyModel:
            config = type("Config", (), {"max_position_embeddings": 1024})()

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.tensor([[1, 3, 4, 2]])

        model = DummyModel()
        answer = generate_single_response(
            model,
            DummyTokenizer(),
            "question",
            max_new_tokens=32,
            device="cpu",
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
        )
        self.assertEqual(answer, "answer")
        self.assertEqual(model.kwargs["repetition_penalty"], 1.1)
        self.assertEqual(model.kwargs["no_repeat_ngram_size"], 4)


class TestValLossCalculation(unittest.TestCase):
    def test_weighted_cross_entropy(self):
        class MockOutput:
            def __init__(self, logits):
                self.logits = logits

        class MockModel(nn.Module):
            def __init__(self, vocab_size=10):
                super().__init__()
                self.vocab_size = vocab_size

            def forward(self, input_ids):
                # 返回固定的 logits: batch_size, seq_len, vocab_size
                b, s = input_ids.shape
                # 让 token 0-9 均有确定得分
                logits = torch.zeros(b, s, self.vocab_size)
                return MockOutput(logits)

        model = MockModel(vocab_size=10)
        # 构建两个 batch，有效 token 数量不同
        # Batch 1: 3 有效 token
        b1_input = torch.zeros(1, 6, dtype=torch.long)
        b1_label = torch.tensor([[ -100, -100, 1, 2, 3, -100 ]], dtype=torch.long)

        # Batch 2: 1 有效 token
        b2_input = torch.zeros(1, 6, dtype=torch.long)
        b2_label = torch.tensor([[ -100, -100, -100, -100, 4, -100 ]], dtype=torch.long)

        dataset = [
            (b1_input, b1_label),
            (b2_input, b2_label),
        ]

        class SimpleLoader:
            def __init__(self, data):
                self.data = data
            def __iter__(self):
                return iter(self.data)

        val_loss = evaluate_val_loss(model, SimpleLoader(dataset), device="cpu")
        # 每个有效 token 在 logits 全 0、vocab=10 时的 cross_entropy 为 ln(10) ≈ 2.302585
        expected_ce = torch.log(torch.tensor(10.0)).item()
        self.assertAlmostEqual(val_loss, expected_ce, places=4)


class TestCompareReport(unittest.TestCase):
    def setUp(self):
        from run_eval import md5_file
        from types import SimpleNamespace
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.eval_dir = self.root / "eval"
        self.eval_dir.mkdir()
        self.base_dir, self.lora_dir = self.root / "base", self.root / "lora"
        sources = {
            "mcq": [{"id": "m1", "question": "Q", "options": {"A": "1", "B": "2"}, "answer": "A"},
                    {"id": "m2", "question": "Q2", "options": {"A": "1", "B": "2"}, "answer": "B"}],
            "medqa": [{"id": "q1", "question": "QA", "reference_answer": "REF", "q_md5": "qhash"}],
            "general": [{"id": "g1", "question": "G"}],
        }
        for kind, rows in sources.items():
            self.write_rows(self.eval_dir / f"questions_{kind}.jsonl", rows)
        manifest = {"files": {p.name: md5_file(p) for p in self.eval_dir.glob("*.jsonl")}}
        (self.eval_dir / "manifest.json").write_text(json.dumps(manifest))
        for mode, directory in (("base", self.base_dir), ("lora", self.lora_dir)):
            directory.mkdir()
            for kind, rows in sources.items():
                results = []
                for row in rows:
                    result = dict(row)
                    if kind == "mcq":
                        answer = "A" if mode == "base" else "B"
                        result.update(official_answer=result.pop("answer"), model_raw_answer=answer,
                                      extracted_answer=answer, is_correct=answer == row["answer"], parse_status="parsed")
                    else:
                        result["model_answer"] = "第一行|内容\n第二行 **原文**"
                    results.append(result)
                self.write_rows(directory / f"{kind}_answers.jsonl", results)
            summary = {
                "mode": mode, "is_smoke": False,
                "completed_questions": {"mcq": 2, "medqa": 1, "general": 1, "total": 4},
                "mcq_metrics": {"total": 2, "correct": 1, "accuracy": 0.5, "invalid_count": 0},
                "generation_config": {"do_sample": False},
                "runtime_env": {"device": "cuda:0", "dtype": "bfloat16"},
                "identity": {"base_weight_path": "base.pth", "base_weight_md5": "base-hash",
                             "lora_weight_path": "lora.pth" if mode == "lora" else None,
                             "lora_weight_md5": "lora-hash" if mode == "lora" else None,
                             "eval_set_manifest": str(self.eval_dir / "manifest.json"),
                             "eval_set_manifest_md5": md5_file(self.eval_dir / "manifest.json")},
            }
            (directory / "summary.json").write_text(json.dumps(summary))
        self.args = SimpleNamespace(base_dir=str(self.base_dir), lora_dir=str(self.lora_dir),
                                    eval_dir=str(self.eval_dir), save_report=str(self.root / "report.md"), force=False)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    def mutate_summary(self, change):
        p = self.lora_dir / "summary.json"
        data = json.loads(p.read_text())
        change(data)
        p.write_text(json.dumps(data))

    def test_comparison_and_multiline_report(self):
        run_comparison(self.args)
        report = Path(self.args.save_report).read_text()
        self.assertIn("错→对 (翻盘修复)", report)
        self.assertIn("对→错 (退化倒退)", report)
        self.assertIn("第一行&#124;内容<br>第二行", report)
        self.assertNotIn("\n第二行", report)

    def test_runtime_and_weight_mismatch(self):
        original = (self.lora_dir / "summary.json").read_text()
        for group, field, value in [("identity", "base_weight_md5", "other"),
                                    ("runtime_env", "dtype", "float16"),
                                    ("runtime_env", "device", "cpu"),
                                    ("generation_config", "do_sample", True)]:
            with self.subTest(field=field):
                (self.lora_dir / "summary.json").write_text(original)
                self.mutate_summary(lambda d: d[group].update({field: value}))
                with self.assertRaises(ValueError):
                    run_comparison(self.args)

    def test_same_missing_question_rejected(self):
        for directory in (self.base_dir, self.lora_dir):
            (directory / "medqa_answers.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "缺题"):
            run_comparison(self.args)

    def test_edited_source_rejected(self):
        with (self.eval_dir / "questions_mcq.jsonl").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "内容变更"):
            run_comparison(self.args)

    def test_smoke_rejected(self):
        self.mutate_summary(lambda d: d.update(is_smoke=True))
        with self.assertRaisesRegex(ValueError, "冒烟"):
            run_comparison(self.args)

    def test_stale_metrics_rejected(self):
        self.mutate_summary(lambda d: d["mcq_metrics"].update(correct=2))
        with self.assertRaisesRegex(ValueError, "汇总"):
            run_comparison(self.args)


class TestRegressions(unittest.TestCase):
    def test_conflicting_choices(self):
        for text in ("答案：A、B", "A。答案是B", "答案：A，B", "选A或者B", "A\nB", "选项A不正确，答案是B"):
            with self.subTest(text=text):
                self.assertIsNone(extract_mcq_choice(text, "ABCDE"))

    def test_training_asset_paths_from_root(self):
        # Execute the actual asset-loading statements with a recording loader;
        # no tokenizer or model is constructed.
        import ast
        from types import SimpleNamespace
        source = ROOT / "trainer/train_lora.py"
        tree = ast.parse(source.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.If) and ast.unparse(node.test).startswith("__name__"))
        statements = [node for node in main.body if isinstance(node, ast.Assign)
                      and any(name in ast.unparse(node.targets[0]) for name in ("repo_root", "model, tokenizer"))]
        calls = []
        def loader(*args, **kwargs):
            calls.append(kwargs)
            return None, None
        env = {"os": os, "__file__": str(source), "init_model": loader, "lm_config": None,
               "args": SimpleNamespace(from_weight="full_sft", device="cuda:0")}
        exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec"), env)
        self.assertEqual(calls[0]["tokenizer_path"], str(ROOT / "model"))
        self.assertEqual(calls[0]["save_dir"], str(ROOT / "out"))

    def test_cloud_interrupted_training_is_not_complete(self):
        # Run the actual shell flow with a fake Python process, never a real model.
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "experiments/lora_medical_20260909/run_cloud.sh"
            script.parent.mkdir(parents=True)
            shutil.copy(ROOT / "experiments/lora_medical_20260909/run_cloud.sh", script)
            for name in ["out/full_sft_768.pth", "trainer/train_lora.py", "trainer/trainer_utils.py",
                         "model/model_lora.py", "model/model_minimind.py"]:
                p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("fixture")
            for name in ("train_5000.jsonl", "val.jsonl"):
                p = script.parent / "data/v2" / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("fixture")
            python = root / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n" +
                'if [ "$1" = "--version" ]; then echo stub; exit 0; fi\n' +
                'if [ "$1" = "-c" ]; then echo bfloat16; exit 0; fi\n' +
                'if [ "$1" = "trainer/train_lora.py" ]; then\n' +
                ' echo attempt >> attempts\n echo partial > out/lora_medical_formal_768.pth\n' +
                ' if [ -f fail ]; then exit 1; fi\n' +
                ' echo final > out/lora_medical_formal_768.pth\nfi\n')
            python.chmod(0o755)
            (root / "fail").touch()
            command = ["bash", str(script), "--step", "5"]
            first = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(first.returncode, 0)
            marker = root / "out/lora_medical_formal_768.complete.sha256"
            self.assertFalse(marker.exists())
            (root / "fail").unlink()
            second = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertTrue(marker.exists())
            third = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(third.returncode, 0, third.stderr)
            self.assertEqual(len((root / "attempts").read_text().splitlines()), 2)
            (root / "out/lora_medical_formal_768.pth").write_text("changed")
            fourth = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(fourth.returncode, 0, fourth.stderr)
            self.assertEqual(len((root / "attempts").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
