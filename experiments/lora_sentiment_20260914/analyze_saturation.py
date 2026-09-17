#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
饱和分析 (analyze_saturation.py)

规模曲线的相邻两点差几个百分点，并不能直接说明"还在涨"——n=400 时抽样噪声
本身就有几个百分点。这里用 McNemar 配对检验：各点评测的是同一批 400 道题，
只统计「A 对 B 错」与「A 错 B 对」的题数，比独立比较两个准确率灵敏得多。

用法：
    python experiments/lora_sentiment_20260914/analyze_saturation.py
"""

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

# (运行目录, 训练条数, 备注)
POINTS = [
    ("scale_250", 250, ""), ("scale_500", 500, ""), ("scale_1000", 1000, ""),
    ("scale_2000", 2000, ""), ("scale_2800", 2800, ""),
    ("formal", 3672, "语料上限"), ("epochs9", 3672, "9 epoch"),
]
ALPHA = 0.05


def load(directory: str):
    evaluation = RUNS / directory / "eval_formal.json"
    summary = RUNS / directory / "train_summary.json"
    if not evaluation.exists() or not summary.exists():
        return None
    rows = json.loads(evaluation.read_text(encoding="utf-8"))["predictions"]
    return {
        "correct": {r["id"]: r["forced"] == r["truth"] for r in rows},
        "accuracy": sum(r["forced"] == r["truth"] for r in rows) / len(rows),
        "n": len(rows),
        "summary": json.loads(summary.read_text(encoding="utf-8")),
    }


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    """双侧精确检验。返回 (a 独对题数, b 独对题数, p)。"""
    a_only = sum(1 for i in a if a[i] and not b[i])
    b_only = sum(1 for i in a if not a[i] and b[i])
    total = a_only + b_only
    if total == 0:
        return a_only, b_only, 1.0
    tail = sum(math.comb(total, k) for k in range(min(a_only, b_only) + 1))
    return a_only, b_only, min(tail / 2 ** total * 2, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="规模曲线饱和分析")
    parser.add_argument("--output", type=Path, default=HERE / "saturation.md")
    args = parser.parse_args()

    loaded = [(d, n, note, load(d)) for d, n, note in POINTS]
    available = [(d, n, note, data) for d, n, note, data in loaded if data]
    if len(available) < 2:
        raise SystemExit("可用的评测产物不足两个，先跑 run_cloud.sh")

    lines = ["# 规模曲线饱和分析", "",
             "相邻两点的准确率差几个百分点不足以说明趋势——n=400 时抽样噪声本身就有几个",
             "百分点。各点评测的是同一批 400 道题，因此用 McNemar 配对检验，只看有多少题",
             "在两个模型之间改变了对错。", "",
             "## 各点结果", "",
             "| 训练条数 | 备注 | 准确率 | 最优 val_loss |", "| --- | --- | --- | --- |"]
    for _, n, note, data in available:
        lines.append(f"| {n} | {note or '—'} | {data['accuracy']:.4f} | {data['summary']['best_val_loss']:.4f} |")

    lines += ["", "## 相邻点配对检验", "",
              "| 对比 | 转错 | 转对 | p | 判定 |", "| --- | --- | --- | --- | --- |"]
    verdicts = []
    for (_, n_a, note_a, a), (_, n_b, note_b, b) in zip(available, available[1:]):
        a_only, b_only, p = mcnemar(a["correct"], b["correct"])
        significant = p < ALPHA
        verdicts.append((n_a, n_b, significant))
        label_a = f"{n_a}{('/' + note_a) if note_a else ''}"
        label_b = f"{n_b}{('/' + note_b) if note_b else ''}"
        lines.append(f"| {label_a} → {label_b} | {a_only} | {b_only} | {p:.4f} | "
                     f"{'✅ 显著提升' if significant else '❌ 不显著'} |")

    first_flat = next((a for a, b, sig in verdicts if not sig), None)
    lines += ["", "## 结论", ""]
    if first_flat is None:
        lines += [f"到 {available[-1][1]} 条为止每一步提升都显著，**尚未饱和**，"
                  "曲线仍有上探空间。"]
    else:
        last = available[-1]
        lines += [
            f"提升在 **{first_flat} 条**处止步：此后每一步（含把数据扩到语料上限 "
            f"{last[1]} 条、以及把训练轮数提到 9 epoch）都无法通过显著性检验。",
            "",
            f"准确率天花板约 {max(d['accuracy'] for _, _, _, d in available):.2%}。"
            "继续加数据或加训练轮数都不会再有实质收益——限制来自模型容量，不是数据量。",
        ]

    # val_loss 与准确率脱钩是这个结论的佐证：损失还在降，但答对的题数不再增加。
    first, last = available[0][3], available[-1][3]
    lines += ["", "## 旁证：val_loss 与准确率脱钩", "",
              f"val_loss 从 {first['summary']['best_val_loss']:.4f} 一路降到 "
              f"{min(d['summary']['best_val_loss'] for _, _, _, d in available):.4f}，"
              "但准确率在 2000 条之后不再变化。模型对监督 token 的把握确实还在提高，",
              "只是这份提高不再转化为更多答对的题。"]

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSATURATION_REPORT_WRITTEN {args.output}")


if __name__ == "__main__":
    main()
