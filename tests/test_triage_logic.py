#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
科室分诊实验的纯逻辑测试：在不加载任何模型权重的情况下验证
1. 标签归一化与 prompt / target 构建的边界条件
2. TriageDataset 的监督掩码、padding 与「超长必须抛错而非截断」约定
3. eval_triage 的生成结果解析规则（含六类互不为子串这一前提）
4. candidate_score 的双口径打分——本实验特有的长度偏置修正
5. 六类评测报告的基线阈值与「是否显著优于随机」判定
6. 数据准备的轮转平衡、近重复剪枝等纯函数
7. 已入库切分数据本身的不变量（平衡、互斥、长度预算、嵌套前缀）

只用到分词器与几个元素的张量，2 核 CPU 上秒级完成。
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from experiments.lora_triage_20260917.triage_dataset import (
    DEFAULT_INSTRUCTION_PROMPT,
    LABELS,
    TriageDataset,
    build_prompt_str,
    build_target_str,
    build_triage_tokens,
    format_triage_prompt,
    measure_total_length,
    normalize_raw_label,
)
from experiments.lora_triage_20260917.eval_triage import candidate_score, parse_generated
from experiments.lora_triage_20260917.prepare_data import (
    drop_near_duplicates,
    get_char_bigrams,
    jaccard_similarity,
    norm_text,
    round_robin_balanced,
)
from experiments.lora_triage_20260917.analyze_scale import mcnemar

EXPERIMENT = ROOT / "experiments" / "lora_triage_20260917"
DATA = EXPERIMENT / "data"

_TOKENIZER = None


def tokenizer():
    """分词器只是词表文件，不是模型权重，可以在测试里直接加载。"""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(str(ROOT / "model"))
    return _TOKENIZER


def write_jsonl(directory: Path, name: str, rows: list[dict]) -> Path:
    path = directory / name
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


class TestLabelNormalization(unittest.TestCase):
    def test_accepts_all_six_labels(self):
        for label in LABELS:
            self.assertEqual(normalize_raw_label(label), label)

    def test_strips_whitespace(self):
        self.assertEqual(normalize_raw_label("  内科 "), "内科")

    def test_rejects_unknown_department(self):
        # 子科室名（心血管科）不是合法顶层标签，必须拒绝而不是悄悄归类。
        with self.assertRaises(ValueError):
            normalize_raw_label("心血管科")
        with self.assertRaises(ValueError):
            normalize_raw_label(None)


class TestPromptConstruction(unittest.TestCase):
    def test_prompt_contains_instruction_and_ask(self):
        text = format_triage_prompt("头疼三天了", DEFAULT_INSTRUCTION_PROMPT)
        self.assertIn(DEFAULT_INSTRUCTION_PROMPT, text)
        self.assertIn("头疼三天了", text)

    def test_prompt_lists_every_candidate_label(self):
        # 生成式评测要求模型在封闭集合里选，指令必须把六个选项都列出来。
        for label in LABELS:
            self.assertIn(label, DEFAULT_INSTRUCTION_PROMPT)

    def test_config_instruction_matches_module_default(self):
        """两处指令文本必须逐字一致，否则长度预算与实际训练输入对不上。

        prepare_data.py 用 config.json 的 instruction_prompt 算 token 长度预算，
        而训练与评测构造 TriageDataset 时不传 instruction、走模块默认值。
        两者一旦分叉，被判定"长度合规"的样本在训练时可能超长，且不会有任何报错——
        只会表现为准确率莫名其妙地差。
        """
        config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["data_config"]["instruction_prompt"], DEFAULT_INSTRUCTION_PROMPT)

    def test_config_labels_match_module_labels(self):
        config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(tuple(config["data_config"]["labels"]), LABELS)

    def test_empty_ask_rejected(self):
        with self.assertRaises(ValueError):
            format_triage_prompt("   ")

    def test_target_requires_known_label(self):
        with self.assertRaises(ValueError):
            build_target_str("心血管科", tokenizer())

    def test_target_ends_with_eos(self):
        target = build_target_str("儿科", tokenizer())
        self.assertIn(tokenizer().eos_token, target)

    def test_generation_prompt_is_applied(self):
        prompt = build_prompt_str("孩子发烧", tokenizer())
        self.assertTrue(len(prompt) > len("孩子发烧"))
        self.assertIn("孩子发烧", prompt)


class TestLengthBudgetUsesLongestLabel(unittest.TestCase):
    """长度预算必须按最长标签核算，否则同一条自述会「挂内科能进、挂肿瘤科超长」。"""

    def test_label_token_lengths_actually_differ(self):
        lengths = {
            label: len(tokenizer()(build_target_str(label, tokenizer()),
                                   add_special_tokens=False).input_ids)
            for label in LABELS
        }
        self.assertGreater(len(set(lengths.values())), 1, f"标签长度居然一致: {lengths}")

    def test_measured_length_independent_of_label(self):
        ask = "最近一周咳嗽有痰，晚上睡不好"
        measured = {label: measure_total_length(ask, label, tokenizer()) for label in LABELS}
        self.assertEqual(len(set(measured.values())), 1, f"长度预算随标签变化了: {measured}")

    def test_measured_length_covers_longest_target(self):
        ask = "腹部隐痛两天"
        prompt_ids, _, _, _ = build_triage_tokens(ask, "内科", tokenizer())
        longest = max(
            len(tokenizer()(build_target_str(label, tokenizer()),
                            add_special_tokens=False).input_ids)
            for label in LABELS
        )
        self.assertEqual(measure_total_length(ask, "内科", tokenizer()), len(prompt_ids) + longest)


class TestTriageDatasetMasking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _dataset(self, rows, max_length=320):
        path = write_jsonl(self.dir, "data.jsonl", rows)
        return TriageDataset(path, tokenizer(), max_length=max_length)

    def test_only_target_tokens_are_supervised(self):
        dataset = self._dataset([{"id": "a", "ask": "咳嗽发烧", "label_text": "内科"}])
        input_ids, labels = dataset[0]
        supervised = [int(t) for t, l in zip(input_ids, labels) if int(l) != -100]
        expected = tokenizer()(build_target_str("内科", tokenizer()),
                               add_special_tokens=False).input_ids
        self.assertEqual(supervised, expected)

    def test_padding_region_is_masked_and_length_fixed(self):
        dataset = self._dataset([{"id": "a", "ask": "咳嗽", "label_text": "外科"}], max_length=256)
        input_ids, labels = dataset[0]
        self.assertEqual(len(input_ids), 256)
        self.assertEqual(len(labels), 256)
        self.assertEqual(int(labels[-1]), -100)

    def test_oversized_sample_raises_instead_of_truncating(self):
        dataset = self._dataset([{"id": "big", "ask": "腹痛" * 500, "label_text": "内科"}],
                                max_length=64)
        with self.assertRaises(ValueError) as ctx:
            _ = dataset[0]
        self.assertIn("超长拦截", str(ctx.exception))

    def test_missing_and_empty_files_rejected(self):
        with self.assertRaises(FileNotFoundError):
            TriageDataset(self.dir / "nope.jsonl", tokenizer())
        empty = write_jsonl(self.dir, "empty.jsonl", [])
        with self.assertRaises(ValueError):
            TriageDataset(empty, tokenizer())

    def test_label_distribution_counts_all_six(self):
        rows = [{"id": str(i), "ask": f"症状{i}", "label_text": label}
                for i, label in enumerate(LABELS)]
        dataset = self._dataset(rows)
        self.assertEqual(dataset.label_distribution(), {label: 1 for label in LABELS})


class TestGeneratedParsing(unittest.TestCase):
    def test_labels_are_mutually_non_substring(self):
        # parse_generated 直接用整词交替匹配，前提是没有哪个标签是另一个的子串。
        for a in LABELS:
            for b in LABELS:
                if a != b:
                    self.assertNotIn(a, b, f"「{a}」是「{b}」的子串，解析规则不再安全")

    def test_bare_labels(self):
        for label in LABELS:
            self.assertEqual(parse_generated(label), label)
            self.assertEqual(parse_generated(f"  {label}  "), label)

    def test_labels_with_surrounding_punctuation(self):
        self.assertEqual(parse_generated("答案：儿科。"), "儿科")
        self.assertEqual(parse_generated("「妇产科」"), "妇产科")
        self.assertEqual(parse_generated("应该挂 肿瘤科，建议尽快就诊"), "肿瘤科")

    def test_ambiguous_double_label_is_rejected(self):
        # 同时吐出两个科室属于犹豫而非作答，不能任选其一记成答对。
        self.assertIsNone(parse_generated("内科或者外科"))
        self.assertIsNone(parse_generated("可能是儿科，也可能是内科"))

    def test_repetition_counts_as_non_compliant(self):
        """复读（「内科内科」）判为不合规，与 lora_sentiment_20260914 同口径。

        小模型复读是已知失败模式；把它算作合规会抬高 format_compliance_rate，
        使本实验与情感实验的合规率不再可比。这是约定，不是解析漏洞。
        """
        self.assertIsNone(parse_generated("内科内科"))

    def test_unknown_department_returns_none(self):
        self.assertIsNone(parse_generated("心血管科"))
        self.assertIsNone(parse_generated("不知道"))
        self.assertIsNone(parse_generated(""))


class _ConstantLogitsModel(torch.nn.Module):
    """桩模型：对任意输入返回同一组 logits，用于验证打分的切片位置与归一化。"""

    class _Output:
        def __init__(self, logits):
            self.logits = logits

    def __init__(self, vocab_size: int, hot_token: int, hot_value: float = 5.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hot_token = hot_token
        self.hot_value = hot_value

    def forward(self, ids):
        batch, length = ids.shape
        logits = torch.zeros(batch, length, self.vocab_size)
        logits[:, :, self.hot_token] = self.hot_value
        return self._Output(logits)

    def __call__(self, ids):
        return self.forward(ids)


class TestCandidateScore(unittest.TestCase):
    def test_mean_is_sum_divided_by_target_length(self):
        model = _ConstantLogitsModel(vocab_size=16, hot_token=3)
        total, mean = candidate_score(model, [1, 2, 3], [4, 5, 6, 7], device="cpu")
        self.assertAlmostEqual(mean, total / 4, places=6)

    def test_sum_penalises_longer_target_but_mean_does_not(self):
        """这正是本实验改用平均口径的理由，用桩模型把它固定成一条断言。"""
        model = _ConstantLogitsModel(vocab_size=16, hot_token=3)
        short_sum, short_mean = candidate_score(model, [1, 2], [5, 6], device="cpu")
        long_sum, long_mean = candidate_score(model, [1, 2], [5, 6, 7, 8], device="cpu")
        self.assertLess(long_sum, short_sum)          # 长标签被求和口径压低
        self.assertAlmostEqual(long_mean, short_mean, places=6)  # 平均口径不受影响

    def test_prompt_length_does_not_change_target_score(self):
        model = _ConstantLogitsModel(vocab_size=16, hot_token=3)
        _, mean_short = candidate_score(model, [1, 2], [5, 6], device="cpu")
        _, mean_long = candidate_score(model, [1, 2, 3, 4, 5], [5, 6], device="cpu")
        self.assertAlmostEqual(mean_short, mean_long, places=6)


class TestBaselineThresholds(unittest.TestCase):
    """六类平衡下随机基线是 1/6，不是二分类的 0.5。"""

    @staticmethod
    def threshold(n: int, classes: int = 6) -> float:
        chance = 1.0 / classes
        return chance + 1.96 * math.sqrt(chance * (1 - chance) / n)

    def test_threshold_shrinks_as_n_grows(self):
        self.assertGreater(self.threshold(30), self.threshold(600))

    def test_six_class_threshold_is_far_below_binary(self):
        self.assertLess(self.threshold(600, 6), self.threshold(600, 2))

    def test_known_value(self):
        self.assertAlmostEqual(self.threshold(600), 0.19648, places=4)

    def test_manifest_records_thresholds(self):
        manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        per_split = manifest["baselines"]["per_split"]
        self.assertIn("formal_test", per_split)
        recorded = per_split["formal_test"]
        self.assertAlmostEqual(
            recorded["min_accuracy_to_beat_random"],
            round(self.threshold(recorded["n"]), 4), places=4,
        )


class TestPreparationHelpers(unittest.TestCase):
    def test_round_robin_keeps_every_multiple_of_six_balanced(self):
        per_label = {label: [{"label_text": label, "id": f"{label}{i}"} for i in range(4)]
                     for label in LABELS}
        merged = round_robin_balanced(per_label)
        self.assertEqual(len(merged), 24)
        for size in (6, 12, 18, 24):
            prefix = merged[:size]
            counts = {label: sum(1 for r in prefix if r["label_text"] == label) for label in LABELS}
            self.assertEqual(len(set(counts.values())), 1, f"前缀 {size} 不平衡: {counts}")

    def test_round_robin_rejects_unequal_inputs(self):
        per_label = {label: [{"label_text": label}] for label in LABELS}
        per_label["内科"] = []
        with self.assertRaises(ValueError):
            round_robin_balanced(per_label)

    def test_norm_text_strips_all_whitespace(self):
        self.assertEqual(norm_text(" 头 疼\n三天 "), "头疼三天")

    def test_jaccard_bounds(self):
        a = get_char_bigrams("咳嗽发烧三天")
        self.assertAlmostEqual(jaccard_similarity(a, a), 1.0)
        self.assertAlmostEqual(jaccard_similarity(a, set()), 0.0)

    def test_near_duplicate_dropped_and_distinct_kept(self):
        rows = [
            {"id": "a", "ask": "咳嗽发烧三天了很难受"},
            {"id": "b", "ask": "咳嗽发烧三天了很难受"},   # 完全相同
            {"id": "c", "ask": "膝盖摔伤需要拍片子吗"},   # 明显不同
        ]
        for row in rows:
            row["_bigrams"] = get_char_bigrams(row["ask"])
        kept, dropped = drop_near_duplicates(rows, 0.85)
        self.assertEqual(dropped, 1)
        self.assertEqual({r["id"] for r in kept}, {"a", "c"})


class TestMcNemar(unittest.TestCase):
    def test_no_disagreement_is_p_one(self):
        a = {"1": True, "2": False}
        self.assertEqual(mcnemar(a, dict(a))[2], 1.0)

    def test_lopsided_disagreement_is_significant(self):
        a = {str(i): False for i in range(20)}
        b = {str(i): True for i in range(20)}
        a_only, b_only, p = mcnemar(a, b)
        self.assertEqual((a_only, b_only), (0, 20))
        self.assertLess(p, 0.05)

    def test_symmetric_disagreement_is_not_significant(self):
        a = {str(i): i % 2 == 0 for i in range(20)}
        b = {str(i): i % 2 == 1 for i in range(20)}
        self.assertGreater(mcnemar(a, b)[2], 0.05)


class TestFrozenSplits(unittest.TestCase):
    """已入库切分本身的不变量——这些性质一旦被破坏，所有结果都不可比。"""

    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        cls.splits = {}
        for name, info in cls.manifest["splits"].items():
            path = ROOT / info["path"]
            cls.splits[name] = [json.loads(line) for line in
                                path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_expected_sizes(self):
        expected = {"formal_train": 12000, "formal_val": 600, "formal_test": 600,
                    "smoke_train": 60, "smoke_val": 30}
        self.assertEqual({k: len(v) for k, v in self.splits.items()}, expected)

    def test_every_split_is_balanced(self):
        for name, rows in self.splits.items():
            counts = {label: sum(1 for r in rows if r["label_text"] == label) for label in LABELS}
            self.assertEqual(len(set(counts.values())), 1, f"{name} 不平衡: {counts}")

    def test_splits_are_disjoint_by_id_and_text(self):
        names = sorted(self.splits)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ids_a = {r["id"] for r in self.splits[a]}
                ids_b = {r["id"] for r in self.splits[b]}
                self.assertFalse(ids_a & ids_b, f"{a} 与 {b} 的 id 重叠")
                text_a = {norm_text(r["ask"]) for r in self.splits[a]}
                text_b = {norm_text(r["ask"]) for r in self.splits[b]}
                self.assertFalse(text_a & text_b, f"{a} 与 {b} 的自述文本重叠")

    def test_recorded_token_length_within_budget(self):
        budget = json.loads((EXPERIMENT / "config.json").read_text(
            encoding="utf-8"))["sequence_config"]["max_seq_len"]
        for name, rows in self.splits.items():
            worst = max(r["token_length"] for r in rows)
            self.assertLessEqual(worst, budget, f"{name} 有样本超出预算: {worst} > {budget}")

    def test_labels_are_all_known(self):
        for name, rows in self.splits.items():
            for row in rows:
                self.assertIn(row["label_text"], LABELS, f"{name} 出现未知标签")

    def test_scale_subsets_are_nested_prefixes(self):
        train_ids = [r["id"] for r in self.splits["formal_train"]]
        for size, info in sorted(self.manifest["scale_curve"].items(), key=lambda kv: int(kv[0])):
            rows = [json.loads(line) for line in
                    (ROOT / info["path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([r["id"] for r in rows], train_ids[:int(size)],
                             f"scale/{size} 不是 formal_train 的前缀")


if __name__ == "__main__":
    unittest.main()
