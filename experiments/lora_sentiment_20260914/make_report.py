#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总报告 (make_report.py)

把 runs/ 下各次训练的 train_summary.json 与评测产出的 eval_*.json 汇总成一份
Markdown。核心约定：准确率一律与 baseline.min_accuracy_to_beat_random 比较，
不与 0.5 直接比较——平衡集合上多数类基线才严格等于 0.5，随机预测只是期望 0.5。

用法：
    python experiments/lora_sentiment_20260914/make_report.py
    python experiments/lora_sentiment_20260914/make_report.py --output 其他路径.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "runs"

# 规模曲线的各个点：formal 训练用的就是 2000 条，直接复用，不重复训练。
SCALE_POINTS = (("scale_250", 250), ("scale_500", 500), ("scale_1000", 1000), ("formal", 2000))


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"产物损坏，无法解析: {path} ({exc})")


def verdict(report: dict) -> str:
    """把准确率与抽样阈值的关系翻译成一句结论。"""
    baseline = report.get("baseline")
    accuracy = report["candidate_scoring_accuracy"]
    if not baseline:
        return f"{accuracy:.4f}（该产物由旧版脚本生成，缺基线字段）"
    threshold = baseline["min_accuracy_to_beat_random"]
    if accuracy > threshold:
        return f"**{accuracy:.4f}** ✅ 超过阈值 {threshold}"
    return f"{accuracy:.4f} ❌ 未超过阈值 {threshold}，与随机不可区分"


def collapse_warning(report: dict) -> str:
    """检测模型是否塌缩到单一类别——医学实验中字母偏向 B 就是这种信号。"""
    matrix = report.get("confusion_matrix", {})
    predicted = {}
    for truths in matrix.values():
        for label, count in truths.items():
            predicted[label] = predicted.get(label, 0) + count
    total = sum(predicted.values())
    if not total:
        return ""
    for label, count in predicted.items():
        if count == total:
            return f"⚠️ **全部预测为「{label}」**，模型塌缩到单一类别，准确率不可信"
        if count / total >= 0.9:
            return f"⚠️ {count / total:.0%} 的预测集中在「{label}」，接近单类别塌缩"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总情感 LoRA 实验结果")
    parser.add_argument("--output", type=Path, default=HERE / "results.md")
    args = parser.parse_args()

    lines = [
        "# 情感分类 LoRA 实验结果",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ｜ 由 `make_report.py` 自动汇总",
        "",
        "准确率一律与 `min_accuracy_to_beat_random` 比较：平衡集合上固定预测多数类的准确率",
        "严格等于 0.5，而随机预测只是期望 0.5，单次实测会在抽样误差内波动。",
        "",
    ]

    formal_eval = load_json(RUNS / "formal" / "eval_formal.json")
    formal_train = load_json(RUNS / "formal" / "train_summary.json")

    lines += ["## 正式结果（test，400 题）", ""]
    if formal_eval is None:
        lines += ["_尚未产出。_", ""]
    else:
        report = formal_eval["report"]
        lines += [
            "| 指标 | 数值 |",
            "| --- | --- |",
            f"| 候选打分准确率 | {verdict(report)} |",
            f"| 生成式准确率 | {report['generation_accuracy']:.4f} |",
            f"| 格式合规率 | {report['format_compliance_rate']:.4f} |",
            f"| Macro F1 | {report['macro_f1']:.4f} |",
            f"| 评测条数 | {report['samples']} |",
            "",
        ]
        warning = collapse_warning(report)
        if warning:
            lines += [warning, ""]
        lines += ["各类别表现：", "", "| 类别 | Precision | Recall | F1 |", "| --- | --- | --- | --- |"]
        for label, metrics in report["per_class"].items():
            lines.append(
                f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |"
            )
        lines.append("")
        lines += [
            "生成式回答分布（`INVALID` 表示未能解析出标签，或同时给出两个标签）：",
            "",
            "```json",
            json.dumps(report["generation_predictions"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]

    if formal_train:
        lines += [
            "## 训练过程", "",
            f"- 初始 val_loss：{formal_train['initial_val_loss']:.6f}",
            f"- 最优 val_loss：{formal_train['best_val_loss']:.6f}（step {formal_train['best_step']}）",
            f"- 训练样本：{formal_train['train_samples']}　验证样本：{formal_train['val_samples']}",
            f"- 耗时：{formal_train['elapsed_seconds']:.1f} 秒",
            f"- 基座权重 md5：`{formal_train['base_weight']['md5']}`",
            "",
        ]
        history = formal_train.get("history", [])
        if len(history) > 1:
            lines += ["| step | val_loss |", "| --- | --- |"]
            lines += [f"| {h['step']} | {h['val_loss']:.6f} |" for h in history]
            lines.append("")

    lines += ["## 规模曲线", "", "| 训练条数 | 候选打分准确率 | 最优 val_loss |", "| --- | --- | --- |"]
    any_scale = False
    for directory, size in SCALE_POINTS:
        evaluation = load_json(RUNS / directory / "eval_formal.json")
        training = load_json(RUNS / directory / "train_summary.json")
        if evaluation is None and training is None:
            continue
        any_scale = True
        accuracy = f"{evaluation['report']['candidate_scoring_accuracy']:.4f}" if evaluation else "—"
        val_loss = f"{training['best_val_loss']:.6f}" if training else "—"
        lines.append(f"| {size} | {accuracy} | {val_loss} |")
    if not any_scale:
        lines = lines[:-2] + ["_尚未产出。_", ""]
    else:
        lines += ["", "各点共用同一个验证集与测试集，因此可以横向比较。", ""]

    smoke_eval = load_json(RUNS / "smoke" / "eval_smoke.json")
    if smoke_eval:
        report = smoke_eval["report"]
        lines += [
            "## 冒烟结果（仅验证流程，不作结论）", "",
            f"- 候选打分准确率 {report['candidate_scoring_accuracy']:.4f}，评测条数 {report['samples']}",
            "- 冒烟集只有 20 条，随机基线的 95% 区间宽达 [0.281, 0.719]，",
            "  这个准确率数字不具备可解读性，只用于确认流程跑通。",
            "",
        ]

    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"REPORT_WRITTEN {args.output}")


if __name__ == "__main__":
    main()
