#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据准备脚本 (prepare_data.py) - 升级版

满足要求：
1. [Issue 1: 禁止截掉题干]
   - 使用真实 Tokenizer 模拟对话模板构建真实 Prompt + Target。
   - 超过 max_seq_len (默认 340) 的题目直接筛除并记录统计原因，保留题干、选项、答案及结束符完整性。
2. [Issue 2: 冒烟与正式分离 & 防覆盖]
   - 区分 formal 目录 (train:1000, val:200, test:200) 与 smoke 目录 (train:20, val:10)。
   - 冒烟严禁包含 test.jsonl，不使用正式测试集排查问题。
   - 已有产物默认拒绝覆盖，除非显式传入 --overwrite。
3. [Issue 4: 完善去重与近重复检测]
   - 抽样前先做集合内部去重 (精确对撞 + 归一化对撞 + 字符 n-gram 近重复检测)。
   - 原评测排除文件 (original_eval_path) 缺失时必须报错，严禁静默忽略。
   - 跨集合互斥与排除原评测题。
   - 数量不足时明确报错，严禁复制补足。
4. [Issue 6: 统一配置生效]
   - 读取 config.json 作为唯一默认配置源，记录最终生效配置。
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

# 添加项目根目录到 sys.path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from experiments.lora_medical_mcq_pilot.mcq_dataset import build_mcq_tokens, format_mcq_prompt

CONFIG_FILE = HERE / "config.json"


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"配置文件缺失: {CONFIG_FILE}")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def md5_file(filepath: Path | str) -> str:
    p = Path(filepath)
    if not p.is_file():
        return ""
    hasher = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def norm_q(q: str) -> str:
    """文本严格归一化：剥除标点、空白及停用语助词，提取核心语义骨架"""
    return re.sub(r"[\s，。？！,.?!、的了吗呢吧啊：:；;\-—_（）\(\)【】\[\]\"'“”‘’于在中为是有的]", "", q)


def get_char_bigrams(text: str) -> set[str]:
    """提取字符级 2-gram 集合用于近重复相似度计算"""
    cleaned = norm_q(text)
    if len(cleaned) <= 2:
        return {cleaned}
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """计算 Jaccard 相似度"""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def parse_options(options_str: str) -> dict[str, str]:
    """解析选项文本为字典 {'A': '...', 'B': '...'}"""
    opts = {}
    for line in options_str.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-E])[\s.、:：]+(.*)$", line)
        if m:
            opts[m.group(1).upper()] = m.group(2).strip()
    return opts


def download_file(url: str, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在从官方源下载: {url} -> {target_path} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(target_path, "wb") as f:
        while chunk := resp.read(65536):
            f.write(chunk)
    print(f"下载完成: {target_path} ({target_path.stat().st_size / 1024 / 1024:.2f} MB)")


def load_csv_data(filepath: Path) -> list[dict]:
    if not filepath.exists():
        raise FileNotFoundError(f"CSV 数据文件不存在: {filepath}")
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def filter_and_dedup_split(
    rows: list[dict],
    split_name: str,
    tokenizer,
    max_seq_len: int,
    instruction: str,
    jaccard_threshold: float,
    excluded_norm_hashes: set[str],
    excluded_bigrams_list: list[set[str]],
    global_seen_hashes: set[str],
    global_seen_bigrams: list[set[str]],
    max_candidates: int | None = None,
) -> tuple[list[dict], dict]:
    """
    单切分池内部去重、排除原评测、跨切分去重与 Token 级超长筛除
    执行顺序：
    1. 结构与格式过滤 (单字母 A-E、无图、选项 2-5 个)
    2. Token 真实长度过滤 (超过 max_seq_len 排除，严禁截断)
    3. 切分内部自身去重 (精确对撞 + 归一化对撞 + 近重复检测)
    4. 排除原医学评测集与原 QA 数据
    5. 跨集合互斥去重 (排除已在前面 split 选中的题目)
    """
    candidates = []
    stats = {
        "raw_total": len(rows),
        "invalid_answer_format": 0,
        "invalid_options": 0,
        "answer_not_in_options": 0,
        "contains_image": 0,
        "token_length_exceeded": 0,
        "internal_exact_duplicate": 0,
        "internal_near_duplicate": 0,
        "excluded_original_eval_or_qa": 0,
        "cross_split_duplicate": 0,
        "passed": 0,
    }

    internal_seen_exact = set()
    internal_seen_norm = set()
    internal_seen_bigrams = []

    for r in rows:
        if max_candidates is not None and len(candidates) >= max_candidates:
            break
        ans = r.get("Answer", "").strip().upper()
        if len(ans) != 1 or ans not in "ABCDE":
            stats["invalid_answer_format"] += 1
            continue

        opts = parse_options(r.get("Options", ""))
        if not (2 <= len(opts) <= 5):
            stats["invalid_options"] += 1
            continue

        if ans not in opts:
            stats["answer_not_in_options"] += 1
            continue

        q = r.get("Question", "").strip()
        opts_str = r.get("Options", "")
        if "图" in q or "图" in opts_str:
            stats["contains_image"] += 1
            continue

        # 1. 真实 Token 长度检测 (Issue 1)
        prompt_ids, target_ids, prompt_str, target_str = build_mcq_tokens(
            question=q, options=opts, answer=ans, tokenizer=tokenizer, instruction=instruction
        )
        total_tokens = len(prompt_ids) + len(target_ids)
        if total_tokens > max_seq_len:
            stats["token_length_exceeded"] += 1
            continue

        # 2. 集合内部精确去重与归一化去重 (Issue 4)
        raw_md5 = md5_text(q)
        norm_val = norm_q(q)
        norm_hash = md5_text(norm_val)

        if raw_md5 in internal_seen_exact or norm_hash in internal_seen_norm:
            stats["internal_exact_duplicate"] += 1
            continue

        # 集合内部近重复相似度检测 (Jaccard >= threshold)
        q_bigrams = get_char_bigrams(q)
        is_internal_near = False
        for seen_bg in internal_seen_bigrams:
            if jaccard_similarity(q_bigrams, seen_bg) >= jaccard_threshold:
                is_internal_near = True
                break
        if is_internal_near:
            stats["internal_near_duplicate"] += 1
            continue

        # 3. 排除原医学选择题评测与历史微调 QA 数据 (Issue 4)
        if norm_hash in excluded_norm_hashes:
            stats["excluded_original_eval_or_qa"] += 1
            continue

        is_orig_near = False
        for ex_bg in excluded_bigrams_list:
            if jaccard_similarity(q_bigrams, ex_bg) >= jaccard_threshold:
                is_orig_near = True
                break
        if is_orig_near:
            stats["excluded_original_eval_or_qa"] += 1
            continue

        # 4. 跨集合互斥去重 (与前序已选集合比较) (Issue 4)
        if norm_hash in global_seen_hashes:
            stats["cross_split_duplicate"] += 1
            continue

        is_cross_near = False
        for gl_bg in global_seen_bigrams:
            if jaccard_similarity(q_bigrams, gl_bg) >= jaccard_threshold:
                is_cross_near = True
                break
        if is_cross_near:
            stats["cross_split_duplicate"] += 1
            continue

        # 通过所有检验，加入当前切分内部已见记录及候选池
        internal_seen_exact.add(raw_md5)
        internal_seen_norm.add(norm_hash)
        internal_seen_bigrams.append(q_bigrams)

        stats["passed"] += 1
        candidates.append({
            "question": q,
            "options": opts,
            "answer": ans,
            "explanation": r.get("Explanation", "").strip(),
            "raw_md5": raw_md5,
            "norm_hash": norm_hash,
            "bigrams": q_bigrams,
            "total_tokens": total_tokens,
        })

    return candidates, stats


def prepare_data_pipeline(args):
    config = load_config()

    # 读取与合并有效配置 (Issue 6)
    data_cfg = config["data_config"]
    seq_cfg = config["sequence_config"]
    run_paths = config["run_paths"]

    max_seq_len = args.max_seq_len or seq_cfg.get("max_seq_len", 340)
    instruction = data_cfg.get("instruction_prompt", "")
    jaccard_th = data_cfg.get("deduplication", {}).get("near_duplicate_jaccard_threshold", 0.85)
    seed = args.seed or data_cfg.get("seed", 42)

    # 路径配置
    exp_root = ROOT / run_paths["experiments_root"]
    raw_dir = exp_root / "data" / "raw"
    formal_data_dir = exp_root / "data" / "formal"
    smoke_data_dir = exp_root / "data" / "smoke"

    # 防覆盖检查 (Issue 2)
    if not args.overwrite:
        for p in [formal_data_dir / "train.jsonl", formal_data_dir / "manifest.json"]:
            if p.exists():
                raise FileExistsError(
                    f"已有正式数据产物存在: {p}。为防止混淆和意外覆盖，默认拒绝覆盖！"
                    f"如需重新生成，请显式追加 --overwrite 参数。"
                )

    raw_dir.mkdir(parents=True, exist_ok=True)
    formal_data_dir.mkdir(parents=True, exist_ok=True)
    smoke_data_dir.mkdir(parents=True, exist_ok=True)

    # 检查原始 CSV 文件
    train_csv = raw_dir / "train.csv"
    val_csv = raw_dir / "val.csv"
    test_csv = raw_dir / "test.csv"

    # 复用本地 val.csv fallback
    local_val_fb = ROOT / data_cfg["local_fallback_val_raw"]
    if not val_csv.exists() and local_val_fb.exists():
        print(f"复用本地已有 CMExam val.csv: {local_val_fb}")
        import shutil
        shutil.copyfile(local_val_fb, val_csv)

    missing = []
    if not train_csv.exists(): missing.append(("train", train_csv, data_cfg["official_urls"]["train"]))
    if not val_csv.exists(): missing.append(("val", val_csv, data_cfg["official_urls"]["val"]))
    if not test_csv.exists(): missing.append(("test", test_csv, data_cfg["official_urls"]["test"]))

    if missing:
        if args.download:
            print("正在下载缺失的 CMExam 官方数据...")
            for name, path, url in missing:
                download_file(url, path)
        else:
            print("=" * 60)
            print("【错误】本地缺少以下 CMExam 官方数据源：")
            for name, path, url in missing:
                print(f"  - [{name}] {path} -> {url}")
            print("\n请追加 --download 参数自动从官方源下载。")
            print("=" * 60)
            sys.exit(1)

    # 1. 严格加载原评测排除文件 (Issue 4: 缺失必须报错)
    orig_eval_path = ROOT / data_cfg["original_eval_exclusion_path"]
    if not orig_eval_path.exists():
        raise FileNotFoundError(
            f"原评测排除文件不存在: {orig_eval_path}！"
            f"为防止原医学选择题考卷题目发生数据泄漏，禁止忽略该文件！"
        )

    excluded_norm_hashes = set()
    excluded_bigrams_list = []
    with open(orig_eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                q_text = item.get("question", "")
                if q_text:
                    excluded_norm_hashes.add(md5_text(norm_q(q_text)))
                    excluded_bigrams_list.append(get_char_bigrams(q_text))
    orig_eval_excluded_count = len(excluded_norm_hashes)
    print(f"已严格载入原评测题 {orig_eval_excluded_count} 道，列入全局排除黑名单")

    # 加载原 QA 数据排除项 (如存在)
    orig_qa_path = ROOT / data_cfg["original_qa_exclusion_path"]
    orig_qa_count = 0
    if orig_qa_path.exists():
        with open(orig_qa_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    convs = item.get("conversations", [])
                    if convs and convs[0].get("role") == "user":
                        q_text = convs[0].get("content", "")
                        if q_text:
                            excluded_norm_hashes.add(md5_text(norm_q(q_text)))
                            excluded_bigrams_list.append(get_char_bigrams(q_text))
                            orig_qa_count += 1
        print(f"已载入原 QA 数据 {orig_qa_count} 条列入去重排除名单")

    # 2. 初始化 Tokenizer 用于真实 Token 长度计算 (Issue 1)
    model_dir = ROOT / config["base_model"]["model_dir"]
    print(f"初始化 Tokenizer: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # 3. 分别读取各切分源 CSV
    print("\n读取 CMExam 官方原始 CSV...")
    train_rows = load_csv_data(train_csv)
    val_rows = load_csv_data(val_csv)
    test_rows = load_csv_data(test_csv)

    global_seen_hashes = set()
    global_seen_bigrams = []

    # 抽样与去重流程 (严格顺序：Train -> Val -> Test)
    target_counts = data_cfg["target_counts"]["formal"]
    target_train = args.train_count or target_counts["train"]
    target_val = args.val_count or target_counts["val"]
    target_test = args.test_count or target_counts["test"]

    rng = random.Random(seed)

    print("\n[1/3] 处理训练集候选池 (严格来自 train.csv) ...")
    train_pool = list(train_rows)
    rng.shuffle(train_pool)
    train_cands, train_stats = filter_and_dedup_split(
        rows=train_pool,
        split_name="train",
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        instruction=instruction,
        jaccard_threshold=jaccard_th,
        excluded_norm_hashes=excluded_norm_hashes,
        excluded_bigrams_list=excluded_bigrams_list,
        global_seen_hashes=global_seen_hashes,
        global_seen_bigrams=global_seen_bigrams,
        max_candidates=target_train,
    )
    if len(train_cands) < target_train:
        raise ValueError(
            f"训练集可用题目不足！需要 {target_train} 道，实际仅有 {len(train_cands)} 道通过严格筛选。严禁复制补足！"
        )
    selected_train = train_cands
    for c in selected_train:
        global_seen_hashes.add(c["norm_hash"])
        global_seen_bigrams.append(c["bigrams"])

    print(f"训练集完成: 原始 {len(train_rows)} -> 通过 {len(train_cands)} -> 抽样 {len(selected_train)}")

    print("\n[2/3] 处理验证集候选池 (严格来自 val.csv，已排除原评测题与训练题) ...")
    val_pool = list(val_rows)
    rng.shuffle(val_pool)
    val_cands, val_stats = filter_and_dedup_split(
        rows=val_pool,
        split_name="val",
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        instruction=instruction,
        jaccard_threshold=jaccard_th,
        excluded_norm_hashes=excluded_norm_hashes,
        excluded_bigrams_list=excluded_bigrams_list,
        global_seen_hashes=global_seen_hashes,
        global_seen_bigrams=global_seen_bigrams,
        max_candidates=target_val,
    )
    if len(val_cands) < target_val:
        raise ValueError(
            f"验证集可用题目不足！需要 {target_val} 道，实际仅有 {len(val_cands)} 道通过严格筛选。严禁复制补足！"
        )
    selected_val = val_cands
    for c in selected_val:
        global_seen_hashes.add(c["norm_hash"])
        global_seen_bigrams.append(c["bigrams"])

    print(f"验证集完成: 原始 {len(val_rows)} -> 通过 {len(val_cands)} -> 抽样 {len(selected_val)}")

    print("\n[3/3] 处理测试集候选池 (严格来自 test.csv，已排除前序所有题目与原评测题) ...")
    test_pool = list(test_rows)
    rng.shuffle(test_pool)
    test_cands, test_stats = filter_and_dedup_split(
        rows=test_pool,
        split_name="test",
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        instruction=instruction,
        jaccard_threshold=jaccard_th,
        excluded_norm_hashes=excluded_norm_hashes,
        excluded_bigrams_list=excluded_bigrams_list,
        global_seen_hashes=global_seen_hashes,
        global_seen_bigrams=global_seen_bigrams,
        max_candidates=target_test,
    )
    if len(test_cands) < target_test:
        raise ValueError(
            f"测试集可用题目不足！需要 {target_test} 道，实际仅有 {len(test_cands)} 道通过严格筛选。严禁复制补足！"
        )
    selected_test = test_cands
    for c in selected_test:
        global_seen_hashes.add(c["norm_hash"])
        global_seen_bigrams.append(c["bigrams"])

    print(f"测试集完成: 原始 {len(test_rows)} -> 通过 {len(test_cands)} -> 抽样 {len(selected_test)}")

    # 4. 构建并落盘正式数据集 (Issue 2)
    def serialize_records(items: list[dict], prefix: str) -> list[dict]:
        records = []
        for idx, it in enumerate(items, 1):
            prompt = format_mcq_prompt(it["question"], it["options"], instruction)
            rec = {
                "id": f"{prefix}_{idx:04d}",
                "question": it["question"],
                "options": it["options"],
                "answer": it["answer"],
                "explanation": it["explanation"],
                "raw_md5": it["raw_md5"],
                "norm_hash": it["norm_hash"],
                "total_tokens": it["total_tokens"],
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": it["answer"]},
                ],
            }
            records.append(rec)
        return records

    formal_train = serialize_records(selected_train, "formal_train")
    formal_val = serialize_records(selected_val, "formal_val")
    formal_test = serialize_records(selected_test, "formal_test")

    def write_jsonl(path: Path, items: list[dict]):
        tmp_p = path.with_suffix(".tmp")
        with open(tmp_p, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        tmp_p.replace(path)

    write_jsonl(formal_data_dir / "train.jsonl", formal_train)
    write_jsonl(formal_data_dir / "val.jsonl", formal_val)
    write_jsonl(formal_data_dir / "test.jsonl", formal_test)

    # 5. 构建并落盘冒烟数据集 (Issue 2: 仅用少量 train/val，绝不碰测试集)
    smoke_counts = data_cfg["target_counts"]["smoke"]
    smoke_train = formal_train[:smoke_counts["train"]]
    smoke_val = formal_val[:smoke_counts["val"]]

    write_jsonl(smoke_data_dir / "train.jsonl", smoke_train)
    write_jsonl(smoke_data_dir / "val.jsonl", smoke_val)
    # 明确确保 smoke 目录下绝不存在 test.jsonl
    if (smoke_data_dir / "test.jsonl").exists():
        (smoke_data_dir / "test.jsonl").unlink()

    # 6. 生成并落盘详细 Manifest 审计报告 (JSON 记录)
    manifest = {
        "generator": "prepare_data.py",
        "version": "2.0.0",
        "created_at": "2026-09-12",
        "effective_config": {
            "max_seq_len": max_seq_len,
            "instruction_prompt": instruction,
            "jaccard_threshold": jaccard_th,
            "seed": seed,
            "formal_counts": {"train": len(formal_train), "val": len(formal_val), "test": len(formal_test)},
            "smoke_counts": {"train": len(smoke_train), "val": len(smoke_val), "test": 0},
        },
        "exclusion_audit": {
            "original_eval_path": str(orig_eval_path),
            "original_eval_file_md5": md5_file(orig_eval_path),
            "original_eval_loaded_count": orig_eval_excluded_count,
            "original_qa_path": str(orig_qa_path),
            "original_qa_loaded_count": orig_qa_count,
        },
        "split_filter_statistics": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "output_hashes": {
            "formal": {
                "train_jsonl_md5": md5_file(formal_data_dir / "train.jsonl"),
                "val_jsonl_md5": md5_file(formal_data_dir / "val.jsonl"),
                "test_jsonl_md5": md5_file(formal_data_dir / "test.jsonl"),
            },
            "smoke": {
                "train_jsonl_md5": md5_file(smoke_data_dir / "train.jsonl"),
                "val_jsonl_md5": md5_file(smoke_data_dir / "val.jsonl"),
            },
        },
        "answer_distributions": {
            "formal_train": {k: sum(1 for x in formal_train if x["answer"] == k) for k in "ABCDE"},
            "formal_val": {k: sum(1 for x in formal_val if x["answer"] == k) for k in "ABCDE"},
            "formal_test": {k: sum(1 for x in formal_test if x["answer"] == k) for k in "ABCDE"},
        },
        "token_length_summary": {
            "max_seq_len_limit": max_seq_len,
            "formal_train_max_tokens": max(x["total_tokens"] for x in formal_train),
            "formal_val_max_tokens": max(x["total_tokens"] for x in formal_val),
            "formal_test_max_tokens": max(x["total_tokens"] for x in formal_test),
        },
        "guarantees": {
            "no_stem_truncation": True,
            "pairwise_split_disjoint": True,
            "original_eval_overlap_excluded": True,
            "smoke_isolated_from_formal_test": True,
        }
    }

    manifest_file = formal_data_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    smoke_manifest_file = smoke_data_dir / "manifest.json"
    with open(smoke_manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"🎉 数据准备与审计完成！")
    print(f"  - 正式数据目录: {formal_data_dir}")
    print(f"    * 训练集: {len(formal_train)} 题 (最大 Token: {manifest['token_length_summary']['formal_train_max_tokens']})")
    print(f"    * 验证集: {len(formal_val)} 题 (最大 Token: {manifest['token_length_summary']['formal_val_max_tokens']})")
    print(f"    * 测试集: {len(formal_test)} 题 (最大 Token: {manifest['token_length_summary']['formal_test_max_tokens']})")
    print(f"  - 冒烟数据目录: {smoke_data_dir}")
    print(f"    * 训练集: {len(smoke_train)} 题, 验证集: {len(smoke_val)} 题, 测试集: 0 题 (安全隔离)")
    print(f"  - 审计清册:     {manifest_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CMExam 医学单选题数据准备与严格审计工具")
    parser.add_argument("--train_count", type=int, default=None, help="正式训练题数 (覆盖 config)")
    parser.add_argument("--val_count", type=int, default=None, help="正式验证题数 (覆盖 config)")
    parser.add_argument("--test_count", type=int, default=None, help="正式测试题数 (覆盖 config)")
    parser.add_argument("--max_seq_len", type=int, default=None, help="最大序列长度 (覆盖 config)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (覆盖 config)")
    parser.add_argument("--download", action="store_true", help="缺少 CSV 时从 GitHub 自动下载")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的数据产物")
    args = parser.parse_args()

    prepare_data_pipeline(args)


if __name__ == "__main__":
    main()
