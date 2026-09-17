#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
科室分诊数据准备 (prepare_data.py)

从冻结的六个科室 CSV 生成可复现的训练/验证/测试切分。

流水线：
  1. 第一遍扫描：全语料按归一化文本取摘要，找出重复条目与**跨科室矛盾**条目
     （同一段自述同时挂在两个科室下，标签自相矛盾，两边都丢）
  2. 第二遍扫描：过滤空/过短，跳过重复与矛盾条目，按类做蓄水池抽样
     （79 万条全量载入会撑爆本地内存，抽样在固定种子下完全可复现）
  3. Token 长度过滤：用真实 tokenizer 测量，超出预算的直接筛除（绝不截断）
  4. 近重复去重：字符 bigram Jaccard >= 阈值，**类内**进行
  5. 平衡切分：六类等量，test/val 优先锁定，train 最后取
  6. 规模曲线子集：600/1200/2400/4800/9600 的嵌套前缀，每个前缀六类等量

与情感实验的两点不同：
  · 语料远超所需（最小类 7.5 万），所以规模曲线可以一路推到 12000 条而不触顶——
    情感实验里「饱和」与「语料耗尽」纠缠在一起，这里能把两者分开。
  · 近重复去重只在类内做。跨类近重复是标签噪声而非泄漏（它压低准确率而不是抬高），
    且全局两两比较在 3 万条量级上不可行；exact 去重与矛盾剔除仍是全局的。

用法：
    python experiments/lora_triage_20260917/prepare_data.py
    python experiments/lora_triage_20260917/prepare_data.py --overwrite
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

from experiments.lora_triage_20260917.triage_dataset import (
    LABELS,
    build_prompt_str,
    build_target_str,
    normalize_raw_label,
)

CONFIG_FILE = HERE / "config.json"

# 单条自述最长 200 万字符也不该触发 csv 的默认字段上限，但原始文件存在异常长行，
# 放宽上限以免解析中断（异常长的样本随后会被长度过滤掉）。
csv.field_size_limit(10 ** 7)


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


def digest_of(normalized: str) -> int:
    """8 字节摘要。79 万条全量存原文会占掉几百 MB，存摘要只占几十 MB。"""
    return int.from_bytes(hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest(), "big")


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


def round_robin_balanced(per_label: dict[str, list[dict]]) -> list[dict]:
    """六类轮转排列，使任意 6 的倍数长度的前缀都保持类别平衡（规模曲线依赖这个性质）。"""
    sizes = {label: len(items) for label, items in per_label.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"轮转排列要求各类等量: {sizes}")
    merged = []
    for row in zip(*(per_label[label] for label in LABELS)):
        merged.extend(row)
    return merged


def label_counts(records: list[dict]) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for record in records:
        counts[record["label_text"]] += 1
    return counts


def baseline_block(all_named: dict[str, list[dict]]) -> dict:
    """各评测集的基线与判定阈值。

    六类平衡切分让固定预测任一类的准确率严格等于 1/6；随机预测的期望也是 1/6，
    但单次实测服从 Binomial(n, 1/6)/n，n 越小波动越大。因此这里同时给出每个
    评测集的 95% 抽样区间与「显著优于随机」所需的最低准确率——smoke 集只有 30
    条，区间宽到准确率数字基本不可解读，它只用来验证流程能跑通。
    """
    chance = 1.0 / len(LABELS)
    per_split = {}
    for name, items in all_named.items():
        if name.endswith("_train"):
            continue
        n = len(items)
        half_width = 1.96 * math.sqrt(chance * (1 - chance) / n)
        per_split[name] = {
            "n": n,
            "random_expected_accuracy": round(chance, 4),
            "random_95ci": [round(chance - half_width, 4), round(chance + half_width, 4)],
            "min_accuracy_to_beat_random": round(chance + half_width, 4),
        }
    return {
        "num_classes": len(LABELS),
        "majority_class_accuracy": round(chance, 4),
        "majority_class_note": f"各评测集六类等量，固定预测任一类的准确率严格等于 1/{len(LABELS)}。",
        "random_expected_accuracy": round(chance, 4),
        "random_note": "随机预测的期望准确率是 1/6，单次实测按 Binomial(n, 1/6)/n 波动；"
                       "判断是否真的学到东西，要看下方各集合的 min_accuracy_to_beat_random。",
        "per_split": per_split,
    }


def resolve_raw_files(config: dict) -> dict[str, Path]:
    """从 raw MANIFEST 读取每个科室对应的 CSV 文件名。"""
    manifest_path = ROOT / config["data_config"]["raw_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"原始语料清单缺失: {manifest_path}\n"
            f"请先按 README 的下载命令取回六个 CSV 并生成 MANIFEST.json。"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_dir = ROOT / config["data_config"]["raw_dir"]
    files = {}
    for entry in manifest["files"]:
        path = raw_dir / entry["name"]
        if not path.exists():
            raise FileNotFoundError(
                f"原始语料缺失: {path}\n"
                f"下载地址见 {manifest_path}（raw CSV 共 370MB，不入库）。"
            )
        files[normalize_raw_label(entry["label"])] = path
    missing = set(LABELS) - set(files)
    if missing:
        raise ValueError(f"MANIFEST 缺少科室: {sorted(missing)}")
    return files


def scan_for_duplicates(raw_files: dict[str, Path], encoding: str) -> tuple[set[int], dict]:
    """第一遍扫描：找出跨科室矛盾的摘要（同一段自述挂在两个不同科室下）。"""
    owner: dict[int, int] = {}
    contradictory: set[int] = set()
    label_index = {label: i for i, label in enumerate(LABELS)}
    stats = {"scanned": 0}

    for label, path in raw_files.items():
        idx = label_index[label]
        with open(path, "r", encoding=encoding, errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                stats["scanned"] += 1
                normalized = norm_text(row.get("ask"))
                if not normalized:
                    continue
                key = digest_of(normalized)
                previous = owner.get(key)
                if previous is None:
                    owner[key] = idx
                elif previous != idx:
                    contradictory.add(key)

    stats["distinct_asks"] = len(owner)
    stats["cross_label_contradictions"] = len(contradictory)
    return contradictory, stats


def sample_records(
    raw_files: dict[str, Path],
    encoding: str,
    contradictory: set[int],
    min_chars: int,
    per_label: int,
    seed: int,
) -> tuple[dict[str, list[dict]], dict]:
    """第二遍扫描：过滤 + 精确去重 + 按类蓄水池抽样。

    蓄水池抽样让内存占用与语料规模无关，同时在固定种子下完全可复现——
    79 万条全量载入在 4GB 内存的机器上会被 OOM 杀掉。
    """
    pools: dict[str, list[dict]] = {label: [] for label in LABELS}
    counters = defaultdict(int)
    seen: set[int] = set()

    for label, path in raw_files.items():
        rng = random.Random(f"{seed}:{label}")
        considered = 0
        with open(path, "r", encoding=encoding, errors="replace", newline="") as f:
            for row_index, row in enumerate(csv.DictReader(f)):
                ask = (row.get("ask") or "").strip()
                normalized = norm_text(ask)
                if not normalized:
                    counters["empty_ask"] += 1
                    continue
                if len(normalized) < min_chars:
                    counters["too_short"] += 1
                    continue
                key = digest_of(normalized)
                if key in contradictory:
                    counters["cross_label_contradiction"] += 1
                    continue
                if key in seen:
                    counters["exact_duplicate"] += 1
                    continue
                seen.add(key)

                record = {
                    "id": f"{label}_{row_index}",
                    "ask": ask,
                    "label_text": label,
                    "sub_department": (row.get("department") or "").strip(),
                }
                reservoir = pools[label]
                if len(reservoir) < per_label:
                    reservoir.append(record)
                else:
                    j = rng.randrange(considered + 1)
                    if j < per_label:
                        reservoir[j] = record
                considered += 1
        counters[f"considered_{label}"] = considered

    return pools, dict(counters)


def prepare(args) -> None:
    config = load_config()
    data_cfg = config["data_config"]
    seq_cfg = config["sequence_config"]

    seed = args.seed or data_cfg["seed"]
    max_seq_len = args.max_seq_len or seq_cfg["max_seq_len"]
    threshold = data_cfg["deduplication"]["near_duplicate_jaccard_threshold"]
    min_chars = data_cfg["min_ask_chars"]
    instruction = data_cfg["instruction_prompt"]
    per_label = args.subsample or data_cfg["subsample_per_label"]
    encoding = data_cfg["raw_encoding"]

    raw_files = resolve_raw_files(config)

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

    print("=" * 68)
    print("【中文医疗科室分诊数据准备】")
    print(f"原始语料: {ROOT / data_cfg['raw_dir']}（{len(raw_files)} 个科室，编码 {encoding}）")
    print(f"最大序列长度: {max_seq_len} | 近重复阈值: {threshold} | 每类抽样: {per_label} | 种子: {seed}")
    print("=" * 68)

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / config["base_model"]["model_dir"]))

    # ---- 1. 全语料扫描：跨科室矛盾 ----
    print("第一遍扫描（找跨科室矛盾条目）...", flush=True)
    contradictory, scan_stats = scan_for_duplicates(raw_files, encoding)
    print(f"  全语料 {scan_stats['scanned']} 条，去重后 {scan_stats['distinct_asks']} 段不同自述，"
          f"其中 {scan_stats['cross_label_contradictions']} 段挂在多个科室下（全部剔除）")

    # ---- 2. 过滤 + 精确去重 + 抽样 ----
    print("第二遍扫描（过滤与抽样）...", flush=True)
    pools, counters = sample_records(
        raw_files, encoding, contradictory, min_chars, per_label, seed
    )
    stats = dict(scan_stats)
    stats.update(counters)
    print(f"  抽样后各类: { {k: len(v) for k, v in pools.items()} }")

    # ---- 3. Token 长度过滤 ----
    # 长度预算按最长标签核算，见 triage_dataset.measure_total_length 的说明。
    longest_target = max(
        len(tokenizer(build_target_str(label, tokenizer), add_special_tokens=False).input_ids)
        for label in LABELS
    )
    print(f"Token 长度过滤（预算 {max_seq_len}，最长标签占 {longest_target} tokens）...", flush=True)
    length_dropped = 0
    observed_max = 0
    for label, items in pools.items():
        kept = []
        for record in items:
            prompt_ids = tokenizer(
                build_prompt_str(record["ask"], tokenizer, instruction), add_special_tokens=False
            ).input_ids
            total = len(prompt_ids) + longest_target
            if total > max_seq_len:
                length_dropped += 1
                continue
            record["token_length"] = total
            observed_max = max(observed_max, total)
            kept.append(record)
        pools[label] = kept
    stats["token_length_exceeded"] = length_dropped
    print(f"  丢弃 {length_dropped} 条，剩余 { {k: len(v) for k, v in pools.items()} }")

    # ---- 4. 近重复去重（类内）----
    print(f"近重复去重（类内，Jaccard >= {threshold}）...", flush=True)
    near_dup_dropped = 0
    for label, items in pools.items():
        # 打乱后再去重，避免保留顺序与原始 CSV 的子科室聚集耦合。
        rng = random.Random(f"{seed}:dedup:{label}")
        rng.shuffle(items)
        for record in items:
            record["_bigrams"] = get_char_bigrams(record["ask"])
        kept, dropped = drop_near_duplicates(items, threshold)
        near_dup_dropped += dropped
        pools[label] = kept
    stats["near_duplicate_dropped"] = near_dup_dropped
    stats["pool_by_label"] = {label: len(items) for label, items in pools.items()}
    print(f"  丢弃 {near_dup_dropped} 条，可用池: {stats['pool_by_label']}")

    # ---- 5. 平衡切分 ----
    formal_counts = data_cfg["target_counts"]["formal"]
    smoke_counts = data_cfg["target_counts"]["smoke"]
    n_labels = len(LABELS)

    for name, total in list(formal_counts.items()) + list(smoke_counts.items()):
        if total % n_labels != 0:
            raise ValueError(f"平衡切分要求各集合样本数是 {n_labels} 的倍数，{name}={total}")

    need_per_label = (
        formal_counts["train"] + formal_counts["val"] + formal_counts["test"]
        + smoke_counts["train"] + smoke_counts["val"]
    ) // n_labels

    for label, items in pools.items():
        if len(items) < need_per_label:
            raise ValueError(
                f"科室「{label}」可用样本 {len(items)} 条，不足以支撑平衡切分所需的 {need_per_label} 条。\n"
                f"请上调 data_config.subsample_per_label，或下调 data_config.target_counts。"
            )

    cursor = {label: 0 for label in LABELS}

    def take(label: str, n: int) -> list[dict]:
        start = cursor[label]
        cursor[label] = start + n
        return pools[label][start:start + n]

    splits: dict[str, list[dict]] = {}
    # 先取 test 与 val，保证锁定评测集优先获得样本，后续调整训练规模不会动到它们。
    for split_name in ("test", "val", "train"):
        share = formal_counts[split_name] // n_labels
        splits[split_name] = round_robin_balanced({label: take(label, share) for label in LABELS})

    smoke_splits: dict[str, list[dict]] = {}
    for split_name in ("train", "val"):
        share = smoke_counts[split_name] // n_labels
        smoke_splits[split_name] = round_robin_balanced({label: take(label, share) for label in LABELS})

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

    # ---- 6. 落盘 ----
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
        if len(set(counts.values())) != 1:
            raise AssertionError(f"规模子集 {size} 不平衡: {counts}")
        path = scale_dir / f"train_{size}.jsonl"
        write_jsonl(path, subset)
        scale_files[str(size)] = {
            "path": str(path.relative_to(ROOT)),
            "count": len(subset),
            "label_counts": counts,
            "md5": md5_file(path),
        }

    # ---- 7. Manifest ----
    manifest = {
        "generator": "prepare_data.py",
        "version": config["version"],
        "created_at": datetime.now().isoformat(),
        "source": {
            "dir": data_cfg["raw_dir"],
            "manifest": data_cfg["raw_manifest"],
            "files": {label: path.name for label, path in sorted(raw_files.items())},
        },
        "effective_config": {
            "max_seq_len": max_seq_len,
            "instruction_prompt": instruction,
            "jaccard_threshold": threshold,
            "near_duplicate_scope": data_cfg["deduplication"]["near_duplicate_scope"],
            "min_ask_chars": min_chars,
            "subsample_per_label": per_label,
            "seed": seed,
            "balanced_splits": True,
            "labels": list(LABELS),
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
            "max_observed": observed_max,
            "budget": max_seq_len,
            "longest_target_tokens": longest_target,
        },
    }

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 68)
    print("🎉 数据准备完成")
    for name, items in all_named.items():
        print(f"  - {name:<14} {len(items):>6} 条  {label_counts(items)}")
    for size, info in scale_files.items():
        print(f"  - scale/{size:<9} {info['count']:>6} 条")
    print(f"  - Manifest: {manifest_path}")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="中文医疗科室分诊数据准备")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (覆盖 config)")
    parser.add_argument("--max_seq_len", type=int, default=None, help="最大序列长度 (覆盖 config)")
    parser.add_argument("--subsample", type=int, default=None, help="每类抽样条数 (覆盖 config)")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有切分")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
