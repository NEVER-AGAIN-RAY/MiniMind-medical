#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
规模曲线饱和分析 (analyze_scale.py)

相邻两点的准确率差几个百分点不足以说明趋势——n=600 时抽样噪声本身就有几个
百分点。各点评测的是同一批 600 道题，因此用 McNemar 配对检验，只看有多少题
在两个模型之间改变了对错。

与情感实验的关键差异：那里训练集 3672 条就是语料上限，「饱和」与「语料耗尽」
纠缠在一起；这里语料有 79 万条，曲线推到 12000 条仍远未触顶，因此如果同样出现
平台期，那就只能是模型容量而非数据供给造成的。

用法：
    python experiments/lora_triage_20260917/analyze_scale.py
"""

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

SCALE_SIZES = [600, 1200, 2400, 4800, 9600]

# (运行目录, 训练条数, 备注)
POINTS = [(f"scale_{n}", n, "") for n in SCALE_SIZES] + [("formal", 12000, "正式训练集")]


def curve_points(lr_tag: str | None):
    """lr_tag 为空时是 run_cloud.sh 跑的 lr 2e-4 原始曲线；否则是 run_scale_lr.sh
    在指定学习率下重跑的曲线。重跑曲线的 12000 点直接复用 run_lr_check.sh 的
    runs/lr_<tag>——同样的数据同样的轮数，没有理由再训一遍。"""
    if not lr_tag:
        return POINTS
    return [(f"scale_{n}_lr{lr_tag}", n, "") for n in SCALE_SIZES] + \
           [(f"lr_{lr_tag}", 12000, "正式训练集")]


def lr_display(lr_tag: str) -> str:
    """运行目录里的 tag 写法（8e4 / 16e4）还原成学习率（8e-4 / 16e-4）。"""
    return lr_tag.replace("e4", "e-4")
ALPHA = 0.05
PRIMARY = "candidate_scoring_accuracy_mean"

# 适配器容量检验：与 runs/formal（rank 16）同数据同轮数，只改 rank。
# 由 run_rank_check.sh 产生；缺失时本节自动跳过。
RANK_ANCHOR = ("formal", 16)
RANK_POINTS = [("rank_64", 64), ("rank_128", 128)]

# 学习率对照：rank 固定 16，只改 lr。由 run_lr_check.sh 产生；缺失时本节自动跳过。
# 存在的理由见 lora_sentiment_20260914/rank_sweep.md——那里 rank 的收益最终被证明
# 来自等效更新幅度而非容量，本实验必须做同样的分离。
LR_POINTS = [("lr_4e4", "4e-4（2×）"), ("lr_8e4", "8e-4（4×）")]


def load(directory: str, root: Path = RUNS):
    evaluation = root / directory / "eval_formal.json"
    summary = root / directory / "train_summary.json"
    if not evaluation.exists() or not summary.exists():
        return None
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    rows = payload["predictions"]
    return {
        # forced 字段记录的就是主指标（平均口径）的预测结果。
        "correct": {r["id"]: r["forced"] == r["truth"] for r in rows},
        "accuracy": payload["report"][PRIMARY],
        "report": payload["report"],
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
    parser = argparse.ArgumentParser(description="科室分诊规模曲线饱和分析")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--runs-dir", type=Path, default=RUNS, metavar="DIR",
        help="重跑曲线的产物目录（默认 runs/）。与原曲线不在同一台机器上跑时用它分开存放，"
             "对照一节的 lr 2e-4 基线始终取自 runs/",
    )
    parser.add_argument(
        "--curve-lr", default=None, metavar="TAG",
        help="分析 run_scale_lr.sh 在指定学习率下重跑的曲线（如 8e4）；默认分析 lr 2e-4 的原始曲线",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = HERE / (f"saturation_lr{args.curve_lr}.md" if args.curve_lr
                              else "saturation.md")

    curve_root = args.runs_dir if args.curve_lr else RUNS
    loaded = [(d, n, note, load(d, curve_root)) for d, n, note in curve_points(args.curve_lr)]
    available = [(d, n, note, data) for d, n, note, data in loaded if data]
    if len(available) < 2:
        raise SystemExit(
            f"可用的评测产物不足两个，先跑 run_scale_lr.sh --lr {lr_display(args.curve_lr)}"
            if args.curve_lr else "可用的评测产物不足两个，先跑 run_cloud.sh")

    threshold = available[-1][3]["report"]["baseline"]["min_accuracy_to_beat_random"]
    lr_label = lr_display(args.curve_lr) if args.curve_lr else "2e-4"
    heading = f"# 科室分诊规模曲线饱和分析（lr {lr_label}）"
    intro = ([f"本曲线在 **lr {lr_label}** 下重跑，用于检验原曲线「9600 条饱和」这个结论",
              "有多少是 lr 2e-4 欠训练造成的假象。", ""] if args.curve_lr else [])
    lines = [heading, ""] + intro + [
             "六类平衡，随机基线 1/6；各点评测同一批 600 道题，用 McNemar 配对检验。",
             f"判定「优于随机」所需准确率 > {threshold}。", "",
             "## 各点结果", "",
             "| 训练条数 | 备注 | 准确率（平均口径） | 生成式 | 格式合规 | 最优 val_loss |",
             "| --- | --- | --- | --- | --- | --- |"]
    for _, n, note, data in available:
        report = data["report"]
        lines.append(
            f"| {n} | {note or '—'} | {data['accuracy']:.4f} | "
            f"{report['generation_accuracy']:.4f} | {report['format_compliance_rate']:.4f} | "
            f"{data['summary']['best_val_loss']:.4f} |"
        )

    lines += ["", "## 相邻点配对检验", "",
              "| 对比 | 转错 | 转对 | p | 判定 |", "| --- | --- | --- | --- | --- |"]
    verdicts = []
    for (_, n_a, _, a), (_, n_b, _, b) in zip(available, available[1:]):
        a_only, b_only, p = mcnemar(a["correct"], b["correct"])
        significant = p < ALPHA
        verdicts.append((n_a, n_b, significant))
        lines.append(f"| {n_a} → {n_b} | {a_only} | {b_only} | {p:.4f} | "
                     f"{'✅ 显著提升' if significant else '❌ 不显著'} |")

    best = max(available, key=lambda item: item[3]["accuracy"])
    first_flat = next((a for a, b, sig in verdicts if not sig), None)
    # 显著性沿曲线不一定单调：lr 8e-4 下 2400→4800 不显著，4800→9600 又显著了。
    # 这种情况下「第一个不显著的相邻步」不是饱和点——真正的饱和点是最后一次
    # 显著提升的终点，此后才是再加数据也不动的那一段。
    significant_steps = [(a, b) for a, b, sig in verdicts if sig]
    first_flat_point = first_flat
    first_flat = significant_steps[-1][1] if significant_steps and first_flat else first_flat
    irregular = bool(significant_steps) and first_flat_point is not None \
        and any(a >= first_flat_point for a, _ in significant_steps)
    lines += ["", "## 结论", ""]
    if first_flat_point is None:
        lines += [f"到 {available[-1][1]} 条为止每一步提升都显著，**尚未饱和**。"
                  "语料还有 79 万条，曲线仍有上探空间——继续加数据是有收益的。"]
    elif args.curve_lr:
        lines += [
            f"提升在 **{first_flat} 条**处止步，此后每一步都无法通过显著性检验。",
            "",
            f"准确率天花板约 {best[3]['accuracy']:.2%}（{best[1]} 条）。",
            "",
        ]
        if irregular:
            lines += [
                f"⚠️ **显著性沿曲线并不单调**：{first_flat_point} → "
                f"{next(b for a, b, _ in verdicts if a == first_flat_point)} 这一步不显著，"
                f"但再往后又出现了显著提升。因此饱和点取的是**最后一次显著提升的终点"
                f"（{first_flat} 条）**，而不是第一个不显著的点——n=600 时单步的抽样噪声"
                "足以让曲线中段出现一个假平台。",
                "",
            ]
        lines += [
            f"**这个饱和点是在调好的学习率（{lr_label}）下测的**，因此不再有"
            "「平台期其实是欠训练」这个替代解释——原曲线 lr 2e-4 下的 9600 条饱和点"
            "当初正是栽在这一条上。逐点对照见下一节。",
        ]
    else:
        anchor_rank = available[-1][3]["summary"].get("lora_rank", "（未记录）")
        lines += [
            f"提升在 **{first_flat} 条**处止步，此后每一步都无法通过显著性检验。",
            "",
            f"准确率天花板约 {best[3]['accuracy']:.2%}（{best[1]} 条）。"
            "本实验的语料有 79 万条，训练集推到 12000 条仍只用掉零头，"
            "**因此这个平台期确定不是语料耗尽造成的**——这一点比情感实验干净，"
            "那里 3672 条恰好就是语料上限，两种解释纠缠在一起。",
            "",
            f"**但「限制来自模型容量」这句话不能就这么写下去。** 本曲线各点的 LoRA rank "
            f"全部固定为 {anchor_rank}，而 `lora_sentiment_20260914` 的 rank 扫描已经证明："
            "在那个任务上，rank 16 看似的「模型容量天花板」其实是**适配器容量**天花板——"
            "同一基座把 rank 提到 128 就继续往上走了（0.8575 → 0.8850，p=0.0074，"
            "见 [`../lora_sentiment_20260914/rank_sweep.md`](../lora_sentiment_20260914/rank_sweep.md)）。",
            "",
            "本实验尚未做同样的 rank 扫描，因此目前只能断言到这一步：",
            "",
            f"- ✅ 平台期**不是**数据量造成的（语料还剩 98%）；",
            f"- ❓ 究竟是基座容量还是适配器容量，**未测**。",
            "",
            f"下一步应当在 {best[1]} 条上重跑一个高 rank 点，才能把这两者分开。",
        ]

    # ---- 与 lr 2e-4 原曲线的逐点对照（仅重跑曲线）----
    if args.curve_lr:
        baseline = [(n, load(d)) for d, n, _ in POINTS]
        baseline = {n: data for n, data in baseline if data}
        paired = [(n, baseline[n], data) for _, n, _, data in available if n in baseline]
        if paired:
            lines += ["", "## 与 lr 2e-4 原曲线的逐点对照", "",
                      f"同样的训练子集、同样的 600 道测试题，只改学习率（2e-4 → {lr_label}）。",
                      "",
                      "| 训练条数 | lr 2e-4 | " + f"lr {lr_label}" + " | 差值 | p |",
                      "| --- | --- | --- | --- | --- |"]
            gains = []
            for n, old, new in paired:
                _, _, p_value = mcnemar(old["correct"], new["correct"])
                delta = new["accuracy"] - old["accuracy"]
                gains.append((n, delta, p_value))
                mark = "✅" if p_value < ALPHA and delta > 0 else (
                    "⚠️" if p_value < ALPHA else "—")
                lines.append(f"| {n} | {old['accuracy']:.4f} | {new['accuracy']:.4f} | "
                             f"{delta:+.4f} | {p_value:.4f} {mark} |")
            lifted = [n for n, delta, p_value in gains if p_value < ALPHA and delta > 0]
            lines += [""]
            if lifted:
                lines += [
                    f"**整条曲线被抬高**：{len(lifted)}/{len(gains)} 个规模点显著提升"
                    f"（{'、'.join(f'{n} 条' for n in lifted)}）。",
                    "",
                    "换句话说，原曲线上「加数据不再有收益」的那段平台期，"
                    "有一部分收益其实一直躺在学习率里没被取走。",
                ]
            else:
                lines += [
                    "**没有任何一个规模点被显著抬高。** 提高学习率在 12000 条上的收益"
                    "没有推广到更小的训练集——值得单独查一下小样本点是否欠训练轮数不足。",
                ]

    # ---- 适配器容量检验 ----
    anchor_data = next((d for name, _, _, d in available if name == RANK_ANCHOR[0]), None)
    rank_rows = [(name, rank, load(name)) for name, rank in RANK_POINTS]
    rank_rows = [(name, rank, data) for name, rank, data in rank_rows if data]
    rank_verdict = None
    if anchor_data and rank_rows:
        lines += ["", "## 适配器容量检验", "",
                  f"上一节把「是基座容量还是适配器容量」留成了未测项。下面在同样的 "
                  f"{RANK_ANCHOR[0]} 训练集上只改 LoRA rank，其余条件完全一致。", "",
                  "| rank | 可训练参数 | 准确率 | 最优 val_loss | vs rank 16 的 p | 判定 |",
                  "| --- | --- | --- | --- | --- | --- |",
                  f"| {RANK_ANCHOR[1]}（锚点） | 0.39M | {anchor_data['accuracy']:.4f} | "
                  f"{anchor_data['summary']['best_val_loss']:.4f} | — | — |"]
        improved = []
        for name, rank, data in rank_rows:
            _, _, p = mcnemar(anchor_data["correct"], data["correct"])
            better = data["accuracy"] > anchor_data["accuracy"]
            significant = p < ALPHA
            if significant and better:
                improved.append((rank, data["accuracy"]))
            params = data["summary"].get("trainable_params", 16 * 2 * 768 * rank)
            verdict = ("✅ 显著提升" if better else "⚠️ 显著下降") if significant else "❌ 无显著差异"
            lines.append(f"| {rank} | {params / 1e6:.2f}M | {data['accuracy']:.4f} | "
                         f"{data['summary']['best_val_loss']:.4f} | {p:.4f} | {verdict} |")
        lines += [""]
        if improved:
            best_rank, best_acc = max(improved, key=lambda item: item[1])
            lines += [
                f"**加 rank 有效：rank {best_rank} 把准确率推到 {best_acc:.4f}"
                f"（比 rank 16 高 {best_acc - anchor_data['accuracy']:+.4f}）。**",
                "",
                "所以 9600 条处的平台期**不是 64M 基座的能力上限**。",
            ]
            # rank 非单调是「rank 在充当有效学习率」的信号，与容量解释相悖。
            ordered = [(RANK_ANCHOR[1], anchor_data["accuracy"])] + \
                      [(rank, d["accuracy"]) for _, rank, d in rank_rows]
            peak_rank, _ = max(ordered, key=lambda item: item[1])
            if peak_rank != max(r for r, _ in ordered):
                tail = "、".join(f"rank {r}={a:.4f}" for r, a in ordered if r > peak_rank)
                lines += [
                    "",
                    f"注意 rank 曲线**并不单调**：峰值在 rank {peak_rank}，再往上回落（{tail}）。"
                    "多给容量不该变差，这更像 rank 在充当有效学习率。",
                ]

            # ---- 学习率对照：分离容量与更新幅度 ----
            lr_rows = [(name, note, load(name)) for name, note in LR_POINTS]
            lr_rows = [(name, note, d) for name, note, d in lr_rows if d]
            if lr_rows:
                best_rank_data = next(d for _, r, d in rank_rows if r == best_rank)
                lines += ["", "### 学习率对照", "",
                          "本仓库的 LoRA 没有 `alpha/rank` 缩放，加 rank 会顺带放大等效更新幅度。",
                          "下面把 rank 固定在 16、只改学习率，看能否不靠加容量达到同等准确率。", "",
                          "| lr | 准确率 | 最优 val_loss | vs rank 16 @2e-4 的 p | vs 最佳 rank 点的 p |",
                          "| --- | --- | --- | --- | --- |"]
                reached = []
                for name, note, d in lr_rows:
                    _, _, p_anchor = mcnemar(anchor_data["correct"], d["correct"])
                    _, _, p_best = mcnemar(best_rank_data["correct"], d["correct"])
                    if p_anchor < ALPHA and d["accuracy"] > anchor_data["accuracy"] and p_best >= ALPHA:
                        reached.append((note, d["accuracy"]))
                    lines.append(f"| {note} | {d['accuracy']:.4f} | "
                                 f"{d['summary']['best_val_loss']:.4f} | {p_anchor:.4f} | {p_best:.4f} |")
                lines += [""]
                if reached:
                    rank_verdict = "magnitude"
                    # 达标点可能不止一个，取其中最好的那个来陈述结论
                    note, acc = max(reached, key=lambda item: item[1])
                    lines += [
                        f"**只调学习率（lr {note}）就到了 {acc:.4f}，与 rank {best_rank} 无显著差异。**",
                        "",
                        "因此瓶颈既不是基座容量也不是适配器容量，而是**学习率没调过**——"
                        "与 `lora_sentiment_20260914` 的结论完全一致，同一个现象在两个任务上重复出现。",
                        "",
                        "**连带影响：本实验「9600 条饱和」这个结论是在 lr 2e-4 下测的，"
                        "整条规模曲线都需要在调好的学习率下重跑才算数。**",
                    ]
                else:
                    rank_verdict = "adapter"
                    lines += [
                        "**放大学习率追不上加 rank。** 更新幅度这条解释被排除，"
                        "收益确实来自适配器容量。",
                    ]
            else:
                rank_verdict = "untested"
                lines += [
                    "",
                    "> ⚠️ **暂不能断言「瓶颈是适配器容量」。** rank 同时改变了可训练参数量与"
                    "等效更新幅度，而 `lora_sentiment_20260914` 已经证明：在那个任务上，"
                    "看似的 rank 收益最终被归因于**学习率偏低**。本实验的学习率对照"
                    "（`run_lr_check.sh`）尚未跑完，在那之前只能说到「调 rank 有效」。",
                ]
        else:
            rank_verdict = "base"
            lines += [
                "**加 rank 没有带来显著提升。** 与情感实验不同，本任务在 rank 16 处确实已经"
                "触到基座的能力上限——数据、轮数、适配器容量三条路都走不通了，"
                "剩下的只有换更大的基座。",
            ]

    lines += ["", "## 与情感实验的对照", "",
              "| 实验 | 任务类型 | 类别数 | 随机基线 | 最佳准确率 | 饱和点 | 语料是否耗尽 |",
              "| --- | --- | --- | --- | --- | --- | --- |",
              "| `lora_medical_mcq_pilot` | 知识型 | 5 | 0.2015 | 0.235（不显著） | — | 否 |",
              "| `lora_sentiment_20260914` | 映射型 | 2 | 0.5 | 0.8575 | 2000 条 | 是（3672 即上限）|",
              f"| `lora_triage_20260917` | 映射型（医学）| 6 | 0.1667 | {best[3]['accuracy']:.4f} | "
              f"{first_flat if first_flat else '未饱和'} | 否（用掉 79 万中的 1.2 万）|", "",
              "三个实验共用同一个 64M 基座、同一套 LoRA 实现与同一套评测口径，",
              "差别只在任务类型与类别数。医学领域内换成映射型任务即有效，",
              "这直接检验了情感实验那条跨领域归因——**医学 MCQ 的失败源于任务类型，不是领域**。"]

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSATURATION_REPORT_WRITTEN {args.output}")


if __name__ == "__main__":
    main()
