#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
情感分类数据准备 (prepare_data.py)

从冻结的 ChnSentiCorp CSV 生成可复现的训练/验证/测试切分。

流水线：
  1. 解析 CSV（必须用 csv 模块：评论正文本身含逗号与引号）
  2. 基础过滤：空评论、过短评论
  3. 精确去重：按归一化文本
  4. Token 长度过滤：用真实 tokenizer 测量，超出预算的直接筛除（绝不截断）
  5. 近重复去重：字符 bigram Jaccard >= 阈值
  6. 平衡切分：正负各半，train/val/test 三者互不重叠
  7. 规模曲线子集：250/500/1000/2000 的嵌套前缀，每个前缀都保持正负平衡

设计理由见 README.md。核心是保证 val/test 平衡：固定预测多数类的准确率
因此严格等于 0.5，随机预测的期望准确率也是 0.5（单次实测会在抽样误差内
波动）。manifest 的 baselines 字段记录了各评测集的 95% 抽样区间与
"优于随机"所需的最低准确率，读结果时以该阈值为准，不要直接比 0.5。

用法：
    python experiments/lora_sentiment_20260914/prepare_data.py
    python experiments/lora_sentiment_20260914/prepare_data.py --overwrite
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from experiments.lora_sentiment_20260914.sentiment_dataset import (
    LABELS,
    measure_total_length,
    normalize_raw_label,
)

CONFIG_FILE = HERE / "config.json"


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"配置文件缺失: {CONFIG_FILE}")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def md5_file(filepath: Path | str) -> str:
    p = Path(filepath)
    if not p.is_file():
        return ""
    hasher = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def norm_text(text: str) -> str:
    """归一化：去除所有空白，用于精确去重与近重复比较。"""
    return re.sub(r"\s+", "", (text or "")).strip()


def get_char_bigrams(text: str) -> set[str]:
    normalized = norm_text(text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i:i + 2] for i in range(len(normalized) - 1)}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a) + len(set_b) - intersection
    return intersection / union if union else 0.0


def drop_near_duplicates(records: list[dict], threshold: float) -> tuple[list[dict], int]:
    """
    贪心保留：逐条与已接受样本比较，Jaccard >= threshold 则丢弃。

    剪枝依据：Jaccard(A,B) >= t 蕴含 min(|A|,|B|)/max(|A|,|B|) >= t，
    因此只需与 bigram 集合大小落在 [|A|*t, |A|/t] 区间的已接受样本比较。
    这个剪枝不会漏判，但把 O(n^2) 的比较量降到可接受范围。
    """
    accepted: list[dict] = []
    by_size: dict[int, list[set[str]]] = defaultdict(list)
    dropped = 0

    for record in records:
        bigrams = record["_bigrams"]
        size = len(bigrams)
        if size == 0:
            dropped += 1
            continue

        lo = int(size * threshold)
        hi = int(size / threshold) + 1
        is_dup = False
        for candidate_size in range(lo, hi + 1):
            for other in by_size.get(candidate_size, ()):
                if jaccard_similarity(bigrams, other) >= threshold:
                    is_dup = True
                    break
            if is_dup:
                break

        if is_dup:
            dropped += 1
        else:
            accepted.append(record)
            by_size[size].append(bigrams)

    return accepted, dropped


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            payload = {k: v for k, v in record.items() if not k.startswith("_")}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def interleave_balanced(negatives: list[dict], positives: list[dict]) -> list[dict]:
    """交替排列，使任意偶数长度的前缀都保持正负平衡（规模曲线子集依赖这个性质）。"""
    if len(negatives) != len(positives):
        raise ValueError(f"交替排列要求两类等量: neg={len(negatives)} pos={len(positives)}")
    merged = []
    for neg, pos in zip(negatives, positives):
        merged.append(neg)
        merged.append(pos)
    return merged


def label_counts(records: list[dict]) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for record in records:
        counts[record["label_text"]] += 1
    return counts


def baseline_block(all_named: dict[str, list[dict]]) -> dict:
    """各评测集的基线与判定阈值。

    平衡切分让固定预测多数类的准确率严格等于 0.5；随机预测的期望也是 0.5，
    但单次实测服从 Binomial(n, 0.5)/n，n 越小波动越大。因此这里同时给出每个
    评测集的 95% 抽样区间与"显著优于随机"所需的最低准确率——smoke 集只有 20
    条，区间宽到准确率数字基本不可解读，它只用来验证流程能跑通。
    """
    per_split = {}
    for name, items in all_named.items():
        if name.endswith("_train"):
            continue
        n = len(items)
        half_width = 1.96 * math.sqrt(0.25 / n)
        per_split[name] = {
            "n": n,
            "random_expected_accuracy": 0.5,
            "random_95ci": [round(0.5 - half_width, 4), round(0.5 + half_width, 4)],
            "min_accuracy_to_beat_random": round(0.5 + half_width, 4),
        }
    return {
        "majority_class_accuracy": 0.5,
        "majority_class_note": "各评测集正负各半，固定预测任一类的准确率严格等于 0.5。",
        "random_expected_accuracy": 0.5,
        "random_note": "随机预测的期望准确率是 0.5，单次实测按 Binomial(n, 0.5)/n 波动；"
                       "判断是否真的学到东西，要看下方各集合的 min_accuracy_to_beat_random。",
        "per_split": per_split,
    }


def prepare(args) -> None:
    config = load_config()
    data_cfg = config["data_config"]
    seq_cfg = config["sequence_config"]

    seed = args.seed or data_cfg["seed"]
    max_seq_len = args.max_seq_len or seq_cfg["max_seq_len"]
    threshold = data_cfg["deduplication"]["near_duplicate_jaccard_threshold"]
    min_chars = data_cfg["min_review_chars"]
    instruction = data_cfg["instruction_prompt"]

    raw_csv = ROOT / data_cfg["raw_csv"]
    if not raw_csv.exists():
        raise FileNotFoundError(f"原始语料不存在: {raw_csv}")

    out_root = HERE / "data"
    formal_dir = out_root / "formal"
    smoke_dir = out_root / "smoke"
    scale_dir = out_root / "scale"

    existing = [p for p in (formal_dir, smoke_dir, scale_dir) if p.exists() and any(p.iterdir())]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"数据目录已存在产物 (检测到: {existing[0]})！\n"
            f"为避免覆盖已被训练引用的冻结切分，默认拒绝执行。\n"
            f"确认要重新生成请显式追加 --overwrite。"
        )

    print("=" * 60)
    print("【ChnSentiCorp 情感分类数据准备】")
    print(f"原始语料: {raw_csv}")
    print(f"MD5: {md5_file(raw_csv)}")
    print(f"最大序列长度: {max_seq_len} | 近重复阈值: {threshold} | 种子: {seed}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / config["base_model"]["model_dir"]))

    # ---- 1. 解析 CSV ----
    with open(raw_csv, "r", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    stats = {"raw_total": len(raw_rows)}

    # ---- 2~4. 逐条过滤 ----
    records: list[dict] = []
    seen_norm: set[str] = set()
    counters = defaultdict(int)

    for row_index, row in enumerate(raw_rows):
        review = (row.get("review") or "").strip()
        if not review:
            counters["empty_review"] += 1
            continue
        if len(norm_text(review)) < min_chars:
            counters["too_short"] += 1
            continue

        try:
            label_text = normalize_raw_label(row.get("label"))
        except ValueError:
            counters["invalid_label"] += 1
            continue

        normalized = norm_text(review)
        if normalized in seen_norm:
            counters["exact_duplicate"] += 1
            continue
        seen_norm.add(normalized)

        total_len = measure_total_length(review, label_text, tokenizer, instruction)
        if total_len > max_seq_len:
            counters["token_length_exceeded"] += 1
            continue

        records.append({
            "id": f"chnsenticorp_{row_index}",
            "review": review,
            "label_text": label_text,
            "raw_label": int(str(row.get("label")).strip()),
            "token_length": total_len,
            "_bigrams": get_char_bigrams(review),
        })

    stats.update(counters)
    stats["after_basic_filters"] = len(records)
    print(f"基础过滤后: {len(records)} 条 (明细: {dict(counters)})")

    # ---- 5. 近重复去重 ----
    # 打乱后再去重，避免保留顺序与原始 CSV 的类别聚集耦合。
    rng = random.Random(seed)
    rng.shuffle(records)
    records, near_dup_dropped = drop_near_duplicates(records, threshold)
    stats["near_duplicate_dropped"] = near_dup_dropped
    stats["after_dedup"] = len(records)
    print(f"近重复去重: 丢弃 {near_dup_dropped} 条，剩余 {len(records)} 条")

    pools = {label: [r for r in records if r["label_text"] == label] for label in LABELS}
    stats["pool_by_label"] = {label: len(items) for label, items in pools.items()}
    print(f"可用池: {stats['pool_by_label']}")

    # ---- 6. 平衡切分 ----
    formal_counts = data_cfg["target_counts"]["formal"]
    smoke_counts = data_cfg["target_counts"]["smoke"]

    for name, total in list(formal_counts.items()) + list(smoke_counts.items()):
        if total % 2 != 0:
            raise ValueError(f"平衡切分要求各集合样本数为偶数，{name}={total}")

    need_per_label = (
        formal_counts["train"] + formal_counts["val"] + formal_counts["test"]
        + smoke_counts["train"] + smoke_counts["val"]
    ) // 2

    for label, items in pools.items():
        if len(items) < need_per_label:
            raise ValueError(
                f"标签「{label}」可用样本 {len(items)} 条，不足以支撑平衡切分所需的 {need_per_label} 条。\n"
                f"请下调 config.json 中 data_config.target_counts，或放宽 sequence_config.max_seq_len。"
            )

    cursor = {label: 0 for label in LABELS}

    def take(label: str, n: int) -> list[dict]:
        start = cursor[label]
        cursor[label] = start + n
        return pools[label][start:start + n]

    splits: dict[str, list[dict]] = {}
    # 先取 test 与 val，保证锁定评测集优先获得样本，后续调整训练规模不会动到它们。
    for split_name in ("test", "val", "train"):
        half = formal_counts[split_name] // 2
        negatives = take(LABELS[0], half)
        positives = take(LABELS[1], half)
        splits[split_name] = interleave_balanced(negatives, positives)

    smoke_splits: dict[str, list[dict]] = {}
    for split_name in ("train", "val"):
        half = smoke_counts[split_name] // 2
        negatives = take(LABELS[0], half)
        positives = take(LABELS[1], half)
        smoke_splits[split_name] = interleave_balanced(negatives, positives)

    # ---- 交叉泄漏校验 ----
    all_named = {f"formal_{k}": v for k, v in splits.items()}
    all_named.update({f"smoke_{k}": v for k, v in smoke_splits.items()})
    id_sets = {name: {r["id"] for r in items} for name, items in all_named.items()}
    names = sorted(id_sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = id_sets[a] & id_sets[b]
            if overlap:
                raise AssertionError(f"切分间存在泄漏: {a} 与 {b} 共有 {len(overlap)} 条样本")
    print("✅ 切分互斥校验通过")

    # ---- 7. 落盘 ----
    split_paths: dict[str, Path] = {}
    for split_name, items in splits.items():
        path = formal_dir / f"{split_name}.jsonl"
        write_jsonl(path, items)
        split_paths[f"formal_{split_name}"] = path
    for split_name, items in smoke_splits.items():
        path = smoke_dir / f"{split_name}.jsonl"
        write_jsonl(path, items)
        split_paths[f"smoke_{split_name}"] = path

    scale_files = {}
    train_records = splits["train"]
    for size in data_cfg["scale_curve_sizes"]:
        if size > len(train_records):
            print(f"⚠️  跳过规模 {size}：超出正式训练集容量 {len(train_records)}")
            continue
        subset = train_records[:size]
        counts = label_counts(subset)
        if counts[LABELS[0]] != counts[LABELS[1]]:
            raise AssertionError(f"规模子集 {size} 不平衡: {counts}")
        path = scale_dir / f"train_{size}.jsonl"
        write_jsonl(path, subset)
        scale_files[str(size)] = {
            "path": str(path.relative_to(ROOT)),
            "count": len(subset),
            "label_counts": counts,
            "md5": md5_file(path),
        }

    # ---- 8. Manifest ----
    manifest = {
        "generator": "prepare_data.py",
        "version": config["version"],
        "created_at": datetime.now().isoformat(),
        "source": {
            "path": str(raw_csv.relative_to(ROOT)),
            "md5": md5_file(raw_csv),
            "manifest": data_cfg["raw_manifest"],
        },
        "effective_config": {
            "max_seq_len": max_seq_len,
            "instruction_prompt": instruction,
            "jaccard_threshold": threshold,
            "min_review_chars": min_chars,
            "seed": seed,
            "balanced_splits": True,
            "formal_counts": formal_counts,
            "smoke_counts": smoke_counts,
        },
        "filter_statistics": stats,
        "splits": {
            name: {
                "path": str(split_paths[name].relative_to(ROOT)),
                "count": len(items),
                "label_counts": label_counts(items),
                "md5": md5_file(split_paths[name]),
            }
            for name, items in all_named.items()
        },
        "scale_curve": scale_files,
        "baselines": baseline_block(all_named),
        "token_length_summary": {
            "max_observed": max((r["token_length"] for r in records), default=0),
            "budget": max_seq_len,
        },
    }

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("🎉 数据准备完成")
    for name, items in all_named.items():
        print(f"  - {name:<14} {len(items):>5} 条  {label_counts(items)}")
    for size, info in scale_files.items():
        print(f"  - scale/{size:<8} {info['count']:>5} 条  {info['label_counts']}")
    print(f"  - Manifest: {manifest_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ChnSentiCorp 情感分类数据准备")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (覆盖 config)")
    parser.add_argument("--max_seq_len", type=int, default=None, help="最大序列长度 (覆盖 config)")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有切分")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
