#!/usr/bin/env python
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from experiments.lora_medical_mcq_pilot.mcq_dataset import (
    ANSWER_RATIONALE_INSTRUCTION_PROMPT,
    build_mcq_tokens,
)
from experiments.lora_medical_mcq_pilot.prepare_data import norm_q, parse_options

def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def load_excluded_hashes() -> set[str]:
    hashes = set()
    paths = [
        HERE / "data" / "formal" / "val.jsonl",
        HERE / "data" / "formal" / "test.jsonl",
        ROOT / "experiments" / "lora_medical_20260909" / "eval" / "v1" / "questions_mcq.jsonl",
        ROOT / "dataset" / "lora_medical.jsonl",
    ]
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                question = item.get("question", "")
                if not question:
                    conversations = item.get("conversations", [])
                    question = conversations[0].get("content", "") if conversations else ""
                if question:
                    hashes.add(hashlib.md5(norm_q(question).encode()).hexdigest())
    return hashes


def select_records(raw_path, count, prefix, tokenizer, excluded, seed, max_seq_len):
    with open(raw_path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    random.Random(seed).shuffle(rows)
    selected = []
    seen = set()
    stats = {"raw": len(rows), "missing_explanation": 0, "answer_absent_from_explanation": 0,
             "invalid": 0, "excluded_or_duplicate": 0, "too_long": 0}
    for row in rows:
        question = row.get("Question", "").strip()
        answer = row.get("Answer", "").strip().upper()
        options = parse_options(row.get("Options", ""))
        explanation = row.get("Explanation", "").strip()
        if not question or answer not in options or not 2 <= len(options) <= 5 or "图" in question:
            stats["invalid"] += 1
            continue
        if not explanation or explanation == "请等待更新":
            stats["missing_explanation"] += 1
            continue
        compact_answer = "".join(options[answer].split())
        compact_explanation = "".join(explanation.split())
        if compact_answer not in compact_explanation:
            stats["answer_absent_from_explanation"] += 1
            continue
        norm_hash = hashlib.md5(norm_q(question).encode()).hexdigest()
        if norm_hash in excluded or norm_hash in seen:
            stats["excluded_or_duplicate"] += 1
            continue
        prompt_ids, target_ids, _, _ = build_mcq_tokens(
            question=question,
            options=options,
            answer=answer,
            tokenizer=tokenizer,
            instruction=ANSWER_RATIONALE_INSTRUCTION_PROMPT,
            target_mode="answer_rationale",
            explanation=explanation,
        )
        total_tokens = len(prompt_ids) + len(target_ids)
        if total_tokens > max_seq_len:
            stats["too_long"] += 1
            continue
        seen.add(norm_hash)
        selected.append({
            "id": f"{prefix}_{len(selected) + 1:05d}",
            "question": question,
            "options": options,
            "answer": answer,
            "explanation": explanation,
            "norm_hash": norm_hash,
            "total_tokens": total_tokens,
        })
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"{prefix} 可用数据不足：需要 {count}，得到 {len(selected)}")
    return selected, stats


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_count", type=int, default=5000)
    parser.add_argument("--val_count", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_name", default="knowledge_5k_v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = HERE / "data" / args.output_name
    outputs = [out_dir / "train.jsonl", out_dir / "val.jsonl", out_dir / "manifest.json"]
    if any(p.exists() for p in outputs) and not args.overwrite:
        raise FileExistsError(f"目标目录已有产物：{out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "model"))
    excluded = load_excluded_hashes()
    train, train_stats = select_records(
        HERE / "data" / "raw" / "train.csv", args.train_count, "knowledge_train",
        tokenizer, excluded, args.seed, args.max_seq_len,
    )
    excluded.update(item["norm_hash"] for item in train)
    val, val_stats = select_records(
        HERE / "data" / "raw" / "val.csv", args.val_count, "knowledge_val",
        tokenizer, excluded, args.seed + 1, args.max_seq_len,
    )
    write_jsonl(outputs[0], train)
    write_jsonl(outputs[1], val)
    manifest = {
        "experiment": args.output_name,
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "target_mode": "answer_rationale",
        "counts": {"train": len(train), "val": len(val)},
        "max_tokens": {"train": max(x["total_tokens"] for x in train), "val": max(x["total_tokens"] for x in val)},
        "source_md5": {
            "train": md5_file(HERE / "data" / "raw" / "train.csv"),
            "val": md5_file(HERE / "data" / "raw" / "val.csv"),
        },
        "output_md5": {"train": md5_file(outputs[0]), "val": md5_file(outputs[1])},
        "selection_stats": {"train": train_stats, "val": val_stats},
    }
    with open(outputs[2], "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
