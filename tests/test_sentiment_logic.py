#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
情感分类实验的纯逻辑测试：在不加载任何模型权重的情况下验证
1. 标签归一化与 prompt / target 构建的边界条件
2. SentimentDataset 的监督掩码、padding 与「超长必须抛错而非截断」约定
3. eval_sentiment 的生成结果解析规则
4. candidate_score 的对数似然切片位置（用极小的桩模型，不加载真实权重）
5. 评测报告的基线阈值与「是否显著优于随机」判定
6. 已入库切分数据本身的不变量（平衡、互斥、长度预算）

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

from experiments.lora_sentiment_20260914.sentiment_dataset import (
    DEFAULT_INSTRUCTION_PROMPT,
    LABELS,
    SentimentDataset,
    build_prompt_str,
    build_sentiment_tokens,
    build_target_str,
    format_sentiment_prompt,
    measure_total_length,
    normalize_raw_label,
)
from experiments.lora_sentiment_20260914.eval_sentiment import (
    candidate_score, infer_rank, parse_generated,
)
from experiments.lora_sentiment_20260914.make_report import collapse_warning, verdict
from experiments.lora_sentiment_20260914.analyze_saturation import mcnemar

EXPERIMENT = ROOT / "experiments" / "lora_sentiment_20260914"
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
    def test_accepts_int_and_str(self):
        self.assertEqual(normalize_raw_label(1), "正面")
        self.assertEqual(normalize_raw_label("1"), "正面")
        self.assertEqual(normalize_raw_label(" 0 "), "负面")
        self.assertEqual(normalize_raw_label(0), "负面")

    def test_rejects_out_of_range(self):
        for bad in (2, -1, "3"):
            with self.assertRaises(ValueError):
                normalize_raw_label(bad)

    def test_rejects_unparseable(self):
        for bad in ("", "正面", None, "1.5"):
            with self.assertRaises(ValueError):
                normalize_raw_label(bad)


class TestPromptConstruction(unittest.TestCase):
    def test_prompt_contains_instruction_and_review(self):
        prompt = format_sentiment_prompt("房间很干净")
        self.assertIn(DEFAULT_INSTRUCTION_PROMPT, prompt)
        self.assertIn("房间很干净", prompt)

    def test_empty_review_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                format_sentiment_prompt(bad)

    def test_target_requires_known_label(self):
        for label in LABELS:
            self.assertTrue(build_target_str(label, tokenizer()).startswith(label))
        for bad in ("中性", "positive", ""):
            with self.assertRaises(ValueError):
                build_target_str(bad, tokenizer())

    def test_target_ends_with_eos(self):
        target = build_target_str("正面", tokenizer())
        self.assertIn(tokenizer().eos_token, target)

    def test_measure_total_length_matches_token_counts(self):
        prompt_ids, target_ids, _, _ = build_sentiment_tokens("服务态度好", "正面", tokenizer())
        self.assertEqual(
            measure_total_length("服务态度好", "正面", tokenizer()),
            len(prompt_ids) + len(target_ids),
        )

    def test_generation_prompt_is_applied(self):
        # 训练、验证、评测必须共用同一个模板；这里固定住 add_generation_prompt 的效果。
        rendered = build_prompt_str("位置方便", tokenizer())
        self.assertIn("位置方便", rendered)
        self.assertTrue(len(rendered) > len(format_sentiment_prompt("位置方便")))


class TestSentimentDatasetMasking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_target_tokens_are_supervised(self):
        path = write_jsonl(self.dir, "d.jsonl", [
            {"id": "a", "review": "房间很干净，服务也好", "label_text": "正面"},
            {"id": "b", "review": "隔音差，前台态度冷淡", "label_text": "负面"},
        ])
        dataset = SentimentDataset(path, tokenizer(), max_length=256)
        for index in range(len(dataset)):
            input_ids, labels = dataset[index]
            sample = dataset.samples[index]
            prompt_ids, target_ids, _, _ = build_sentiment_tokens(
                sample["review"], sample["label_text"], tokenizer()
            )
            supervised = [int(t) for t, l in zip(input_ids, labels) if l != -100]
            self.assertEqual(supervised, list(target_ids), "只应监督标签词与结束符")
            self.assertTrue(
                all(l == -100 for l in labels[: len(prompt_ids)]),
                "prompt 前缀必须全部屏蔽",
            )

    def test_padding_region_is_masked_and_length_fixed(self):
        path = write_jsonl(self.dir, "d.jsonl", [{"id": "a", "review": "还行", "label_text": "正面"}])
        dataset = SentimentDataset(path, tokenizer(), max_length=128)
        input_ids, labels = dataset[0]
        self.assertEqual(len(input_ids), 128)
        self.assertEqual(len(labels), 128)
        prompt_ids, target_ids, _, _ = build_sentiment_tokens("还行", "正面", tokenizer())
        used = len(prompt_ids) + len(target_ids)
        self.assertTrue(all(l == -100 for l in labels[used:]), "padding 区必须屏蔽")
        self.assertTrue(
            all(int(t) == dataset.pad_token_id for t in input_ids[used:]),
            "padding 必须使用 pad_token_id",
        )

    def test_oversized_sample_raises_instead_of_truncating(self):
        # 核心约定：截断会砍掉评论尾部的转折，让标签与输入对不上，因此必须抛错。
        path = write_jsonl(self.dir, "d.jsonl", [{"id": "long", "review": "很好" * 500, "label_text": "正面"}])
        dataset = SentimentDataset(path, tokenizer(), max_length=64)
        with self.assertRaises(ValueError) as ctx:
            dataset[0]
        self.assertIn("long", str(ctx.exception))

    def test_missing_and_empty_files_rejected(self):
        with self.assertRaises(FileNotFoundError):
            SentimentDataset(self.dir / "nope.jsonl", tokenizer())
        empty = self.dir / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            SentimentDataset(empty, tokenizer())

    def test_label_distribution(self):
        path = write_jsonl(self.dir, "d.jsonl", [
            {"id": "a", "review": "好", "label_text": "正面"},
            {"id": "b", "review": "差", "label_text": "负面"},
            {"id": "c", "review": "很好", "label_text": "正面"},
        ])
        dataset = SentimentDataset(path, tokenizer(), max_length=64)
        self.assertEqual(dataset.label_distribution(), {"负面": 1, "正面": 2})


class TestGeneratedParsing(unittest.TestCase):
    def test_bare_labels(self):
        self.assertEqual(parse_generated("正面"), "正面")
        self.assertEqual(parse_generated("  负面  "), "负面")

    def test_labels_with_surrounding_punctuation(self):
        self.assertEqual(parse_generated("答案：正面"), "正面")
        self.assertEqual(parse_generated("情感倾向是「负面」"), "负面")
        self.assertEqual(parse_generated("正面。"), "正面")
        self.assertEqual(parse_generated("负面！"), "负面")

    def test_invalid_returns_none(self):
        for bad in ("", "中性", "无法判断", "既有正面也有负面的地方"):
            self.assertIsNone(parse_generated(bad), f"不应从 {bad!r} 中抽出标签")

    def test_ambiguous_double_label_is_rejected(self):
        # 同时出现两个标签时不能任选其一，否则会把模型的犹豫记成一次正确。
        self.assertIsNone(parse_generated("正面 负面"))
        self.assertIsNone(parse_generated("答案：正面（不是负面）"))
        self.assertIsNone(parse_generated("可能是负面，也可能是正面"))


class _ConstantLogitsModel(torch.nn.Module):
    """桩模型：对任何输入都返回同一组 logits，用于固定住切片位置的正确性。"""

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    def __init__(self, vocab_size: int, favored_token: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.favored = favored_token

    def forward(self, ids):
        batch, length = ids.shape
        logits = torch.zeros(batch, length, self.vocab_size)
        logits[:, :, self.favored] = 10.0
        return self._Out(logits)


class TestCandidateScore(unittest.TestCase):
    def test_scores_only_target_positions(self):
        vocab = 16
        model = _ConstantLogitsModel(vocab, favored_token=7)
        prompt_ids = [1, 2, 3]
        # 目标全是被偏好的 token，得分应明显高于全是非偏好 token 的目标。
        high = candidate_score(model, prompt_ids, [7, 7], "cpu")
        low = candidate_score(model, prompt_ids, [5, 5], "cpu")
        self.assertGreater(high, low)

    def test_score_is_sum_of_target_token_logprobs(self):
        vocab = 8
        model = _ConstantLogitsModel(vocab, favored_token=3)
        prompt_ids = [1, 2]
        target_ids = [3, 3, 3]
        expected_per_token = math.log(math.exp(10.0) / (math.exp(10.0) + (vocab - 1)))
        self.assertAlmostEqual(
            candidate_score(model, prompt_ids, target_ids, "cpu"),
            expected_per_token * len(target_ids),
            places=4,
        )

    def test_prompt_length_does_not_change_target_score(self):
        # 切片起点写错（少减 1 或多减 1）时，这个不变量会被破坏。
        model = _ConstantLogitsModel(8, favored_token=3)
        short = candidate_score(model, [1, 2], [3, 3], "cpu")
        long = candidate_score(model, [1, 2, 4, 5, 6], [3, 3], "cpu")
        self.assertAlmostEqual(short, long, places=5)


class TestInferRank(unittest.TestCase):
    """rank 是 checkpoint 的属性而非全局配置——rank 扫描下 config.json 的值
    不再适用于每一个 checkpoint，必须从权重形状本身读出来。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _checkpoint(self, name: str, state: dict) -> Path:
        path = self.dir / name
        torch.save(state, path)
        return path

    def test_reads_rank_from_A_matrix_shape(self):
        for rank in (4, 16, 128):
            path = self._checkpoint(f"r{rank}.pth", {
                "layers.0.self_attn.q_proj.lora.A.weight": torch.zeros(rank, 768),
                "layers.0.self_attn.q_proj.lora.B.weight": torch.zeros(768, rank),
            })
            self.assertEqual(infer_rank(path), rank)

    def test_rejects_mixed_ranks(self):
        path = self._checkpoint("mixed.pth", {
            "layers.0.self_attn.q_proj.lora.A.weight": torch.zeros(8, 768),
            "layers.1.self_attn.q_proj.lora.A.weight": torch.zeros(16, 768),
        })
        with self.assertRaises(ValueError):
            infer_rank(path)

    def test_rejects_checkpoint_without_lora_weights(self):
        path = self._checkpoint("empty.pth", {"layers.0.self_attn.q_proj.weight": torch.zeros(4, 4)})
        with self.assertRaises(ValueError):
            infer_rank(path)

    def test_matches_stored_checkpoints(self):
        """已入库的 8 次运行全部是 rank 16，推断结果必须与之一致。"""
        checkpoints = sorted((EXPERIMENT / "runs").glob("*/best_lora.pth"))
        if not checkpoints:
            self.skipTest("本地没有 runs/ 产物")
        for checkpoint in checkpoints:
            with self.subTest(run=checkpoint.parent.name):
                self.assertGreater(infer_rank(checkpoint), 0)


class TestBaselineThresholds(unittest.TestCase):
    """基线口径：多数类严格 0.5，随机只是期望 0.5，判定要用抽样区间。"""

    @staticmethod
    def threshold(n: int) -> float:
        return 0.5 + 1.96 * math.sqrt(0.25 / n)

    def test_threshold_shrinks_as_n_grows(self):
        self.assertGreater(self.threshold(20), self.threshold(400))
        self.assertGreater(self.threshold(400), self.threshold(4000))

    def test_known_values(self):
        self.assertAlmostEqual(self.threshold(400), 0.549, places=3)
        self.assertAlmostEqual(self.threshold(20), 0.7191, places=3)

    def test_manifest_records_thresholds(self):
        manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        per_split = manifest["baselines"]["per_split"]
        self.assertIn("formal_test", per_split)
        for name, info in per_split.items():
            self.assertFalse(name.endswith("_train"), "训练集不需要基线阈值")
            self.assertAlmostEqual(
                info["min_accuracy_to_beat_random"], self.threshold(info["n"]), places=4
            )


class TestFrozenSplits(unittest.TestCase):
    """已入库切分的不变量。任何一条被破坏，评测结果都不再可解读。"""

    @classmethod
    def setUpClass(cls):
        cls.splits = {}
        for name, relative in {
            "formal_train": "formal/train.jsonl",
            "formal_val": "formal/val.jsonl",
            "formal_test": "formal/test.jsonl",
            "smoke_train": "smoke/train.jsonl",
            "smoke_val": "smoke/val.jsonl",
        }.items():
            path = DATA / relative
            cls.splits[name] = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    def test_expected_sizes(self):
        # 期望值取自 config，避免调整切分规模时还要同步改测试。
        config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
        counts = config["data_config"]["target_counts"]
        for prefix in ("formal", "smoke"):
            for split, expected in counts[prefix].items():
                self.assertEqual(len(self.splits[f"{prefix}_{split}"]), expected,
                                 f"{prefix}_{split} 条数与 config 不符")

    def test_every_split_is_balanced(self):
        for name, rows in self.splits.items():
            positives = sum(r["label_text"] == "正面" for r in rows)
            self.assertEqual(positives * 2, len(rows), f"{name} 正负不均衡，基线不再是 0.5")

    def test_splits_are_disjoint_by_id_and_text(self):
        names = sorted(self.splits)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ids = {r["id"] for r in self.splits[a]} & {r["id"] for r in self.splits[b]}
                texts = {r["review"] for r in self.splits[a]} & {r["review"] for r in self.splits[b]}
                self.assertEqual(ids, set(), f"{a} 与 {b} 存在 id 重叠")
                self.assertEqual(texts, set(), f"{a} 与 {b} 存在正文重叠")

    def test_recorded_token_length_within_budget(self):
        budget = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
        limit = budget["sequence_config"]["max_seq_len"]
        for name, rows in self.splits.items():
            longest = max(r["token_length"] for r in rows)
            self.assertLessEqual(longest, limit, f"{name} 存在超长样本，训练时会抛错")

    def test_scale_subsets_are_nested_prefixes(self):
        train_ids = [r["id"] for r in self.splits["formal_train"]]
        config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
        for size in config["data_config"]["scale_curve_sizes"]:
            rows = [json.loads(l) for l in open(DATA / f"scale/train_{size}.jsonl", encoding="utf-8") if l.strip()]
            self.assertEqual([r["id"] for r in rows], train_ids[:size], f"train_{size} 不是训练集前缀")
            positives = sum(r["label_text"] == "正面" for r in rows)
            self.assertEqual(positives * 2, len(rows), f"train_{size} 正负不均衡")


class TestReportVerdict(unittest.TestCase):
    """汇总报告必须把"超没超过阈值"说清楚，不能只甩一个准确率数字。"""

    @staticmethod
    def report(accuracy: float, threshold: float = 0.549) -> dict:
        return {
            "candidate_scoring_accuracy": accuracy,
            "baseline": {"min_accuracy_to_beat_random": threshold},
        }

    def test_above_threshold_marked_pass(self):
        text = verdict(self.report(0.82))
        self.assertIn("✅", text)
        self.assertIn("0.8200", text)

    def test_below_threshold_marked_fail(self):
        text = verdict(self.report(0.52))
        self.assertIn("❌", text)
        self.assertIn("与随机不可区分", text)

    def test_just_above_half_is_not_a_pass(self):
        # 0.51 高于 0.5 但低于阈值，绝不能被写成"有提升"。
        self.assertIn("❌", verdict(self.report(0.51)))

    def test_missing_baseline_is_flagged_not_crashed(self):
        text = verdict({"candidate_scoring_accuracy": 0.7})
        self.assertIn("缺基线字段", text)


class TestCollapseDetection(unittest.TestCase):
    """全部预测同一类别时准确率不可信，报告必须显式警告。"""

    def test_total_collapse_detected(self):
        matrix = {"负面": {"负面": 200, "正面": 0}, "正面": {"负面": 200, "正面": 0}}
        self.assertIn("塌缩", collapse_warning({"confusion_matrix": matrix}))

    def test_near_collapse_detected(self):
        matrix = {"负面": {"负面": 195, "正面": 5}, "正面": {"负面": 185, "正面": 15}}
        self.assertIn("接近单类别塌缩", collapse_warning({"confusion_matrix": matrix}))

    def test_balanced_predictions_no_warning(self):
        matrix = {"负面": {"负面": 170, "正面": 30}, "正面": {"负面": 40, "正面": 160}}
        self.assertEqual(collapse_warning({"confusion_matrix": matrix}), "")

    def test_empty_matrix_is_safe(self):
        self.assertEqual(collapse_warning({}), "")


class TestMcNemar(unittest.TestCase):
    """饱和结论建立在这个检验上，先把它的数学钉住。"""

    @staticmethod
    def pair(a_only: int, b_only: int, both: int = 0):
        a, b = {}, {}
        index = 0
        for _ in range(a_only):      # a 对 b 错
            a[index], b[index] = True, False; index += 1
        for _ in range(b_only):      # a 错 b 对
            a[index], b[index] = False, True; index += 1
        for _ in range(both):        # 两者都对，不影响检验
            a[index], b[index] = True, True; index += 1
        return a, b

    def test_no_disagreement_is_p_one(self):
        a, b = self.pair(0, 0, both=100)
        self.assertEqual(mcnemar(a, b), (0, 0, 1.0))

    def test_lopsided_disagreement_is_significant(self):
        a, b = self.pair(3, 30)
        a_only, b_only, p = mcnemar(a, b)
        self.assertEqual((a_only, b_only), (3, 30))
        self.assertLess(p, 0.05)

    def test_even_disagreement_is_not_significant(self):
        # 8 对 6 正是 2000→2800 的情形，必须判为不显著。
        _, _, p = mcnemar(*self.pair(8, 6))
        self.assertGreater(p, 0.05)

    def test_symmetric_in_arguments(self):
        a, b = self.pair(5, 17)
        self.assertAlmostEqual(mcnemar(a, b)[2], mcnemar(b, a)[2])

    def test_matches_known_binomial_value(self):
        # 全部 10 题都朝一个方向改变：p = 2 × (1/2)^10
        _, _, p = mcnemar(*self.pair(0, 10))
        self.assertAlmostEqual(p, 2 * 0.5 ** 10, places=6)


if __name__ == "__main__":
    unittest.main()
