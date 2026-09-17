#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rank 扫描分析 (analyze_rank.py)

REPORT.md 里「0.86 的天花板来自 64M 基座容量」这句话，原本只有两条间接证据
（val_loss 与准确率脱钩、9 epoch 过拟合）。rank 扫描提供直接证据：其余条件全部
固定，只把 LoRA 容量放大一到两个数量级，看准确率动不动。

所有点都与锚点 rank 16（runs/epochs9）比较——问题是「加容量有没有用」，而不是
相邻 rank 之间差多少，所以不做相邻配对，统一对锚点做 McNemar 检验。

用法：
    python experiments/lora_sentiment_20260914/analyze_rank.py
"""

import argparse
import json
from pathlib import Path

from analyze_saturation import RUNS, load, mcnemar, ALPHA

HERE = Path(__file__).resolve().parent

ANCHOR = ("epochs9", 16)
# (运行目录, rank)；锚点之外的点由 run_rank_sweep.sh 生成
POINTS = [("rank_4", 4), ("rank_8", 8), ANCHOR, ("rank_32", 32),
          ("rank_64", 64), ("rank_128", 128), ("rank_256", 256)]

# 容量 / 更新幅度分离对照：rank 固定为 16，只放大学习率。
# 由 run_capacity_control.sh 产生；缺失时本节自动跳过。
CONTROLS = [
    ("lrctl_4e4", 16, "lr 4e-4（2×）"),
    ("lrctl_5e4", 16, "lr 5.7e-4（≈2.83×，匹配 rank 128 的更新范数）"),
    ("lrctl_8e4", 16, "lr 8e-4（4×）"),
]


def trainable_params(data: dict, rank: int) -> int:
    """优先用 train_summary.json 里记录的值；锚点是加 --rank 之前跑的，没有这个字段。"""
    recorded = data["summary"].get("trainable_params")
    if recorded is not None:
        return recorded
    # MiniMind 768 dim / 8 层：每层 q_proj 与 o_proj 两个方阵挂 LoRA，共 16 个模块，
    # 每个模块 A(768×r) + B(r×768)。
    return 16 * 2 * 768 * rank


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA rank 扫描分析")
    parser.add_argument("--output", type=Path, default=HERE / "rank_sweep.md")
    args = parser.parse_args()

    loaded = [(d, r, load(d)) for d, r in POINTS]
    available = [(d, r, data) for d, r, data in loaded if data]
    missing = [d for d, _, data in loaded if not data]
    anchor = next((data for d, _, data in available if d == ANCHOR[0]), None)
    if anchor is None:
        raise SystemExit(f"缺少锚点 runs/{ANCHOR[0]}/eval_formal.json，先跑 run_rank_sweep.sh")
    if len(available) < 2:
        raise SystemExit("可用的 rank 点不足两个，先跑 run_rank_sweep.sh")

    anchor_acc = anchor["accuracy"]
    lines = [
        "# LoRA rank 扫描（容量对照实验）", "",
        "训练数据 3672 条、9 epoch、同一批 400 题测试集全部固定，只改 LoRA rank。",
        f"锚点是 `runs/{ANCHOR[0]}`（rank {ANCHOR[1]}，即 REPORT.md 的最佳结果 {anchor_acc:.4f}）。", "",
        "## 各点结果", "",
        "| rank | 可训练参数 | 占基座比例 | 准确率 | 生成式准确率 | 最优 val_loss |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    base_params = 63_900_000
    for directory, rank, data in available:
        params = trainable_params(data, rank)
        evaluation = json.loads((RUNS / directory / "eval_formal.json").read_text(encoding="utf-8"))
        generation_accuracy = evaluation["report"]["generation_accuracy"]
        mark = " ←锚点" if directory == ANCHOR[0] else ""
        lines.append(
            f"| {rank}{mark} | {params / 1e6:.2f}M | {params / base_params:.2%} | "
            f"{data['accuracy']:.4f} | {generation_accuracy:.4f} | "
            f"{data['summary']['best_val_loss']:.4f} |"
        )

    lines += ["", f"## 与锚点 rank {ANCHOR[1]} 的配对检验", "",
              "各点评测的是同一批 400 道题，用 McNemar 精确检验，只看有多少题改变了对错。", "",
              "| 对比 | 锚点独对 | 该点独对 | 准确率差 | p | 判定 |",
              "| --- | --- | --- | --- | --- | --- |"]
    improvements = []
    for directory, rank, data in available:
        if directory == ANCHOR[0]:
            continue
        anchor_only, point_only, p = mcnemar(anchor["correct"], data["correct"])
        significant = p < ALPHA
        better = data["accuracy"] > anchor_acc
        improvements.append((rank, significant and better))
        if not significant:
            verdict = "❌ 无显著差异"
        else:
            verdict = "✅ 显著提升" if better else "⚠️ 显著下降"
        lines.append(
            f"| rank {ANCHOR[1]} → rank {rank} | {anchor_only} | {point_only} | "
            f"{data['accuracy'] - anchor_acc:+.4f} | {p:.4f} | {verdict} |"
        )

    gained = [rank for rank, improved in improvements if improved]
    max_rank = max(rank for _, rank, _ in available)
    min_rank = min(rank for _, rank, _ in available)
    # 对照判定必须先算出来，结论段要引用它；章节文本单独攒着，最后按顺序拼。
    control_lines, control_verdict, control_best = build_controls(
        anchor, anchor_acc, available, gained
    )

    lines += ["", "## 结论", ""]
    if gained:
        best_gained = max(gained)
        best_point = next(d for _, r, d in available if r == best_gained)
        lines += [
            f"**rank {', '.join(str(r) for r in gained)} 显著优于锚点——"
            f"REPORT.md 中「天花板来自 64M 基座容量」这一断言被证伪。**",
            "",
            f"rank {best_gained} 把准确率推到 {best_point['accuracy']:.4f}，比锚点高 "
            f"{best_point['accuracy'] - anchor_acc:+.4f}。只要还在同一个基座上把 LoRA 调大就能往上走，"
            "说明 0.86 并不是 64M 基座的能力上限。",
        ]
        if control_verdict == "magnitude":
            lines += [
                "",
                f"**但真正的原因不是容量。** 学习率对照（下一节）显示：rank 保持 16 不变、"
                f"只把学习率提到 {control_best[0].replace('lr ', '')}，"
                f"准确率就到 {control_best[1]:.4f}——"
                f"与 rank {best_gained} 无显著差异，可训练参数却只有 1/"
                f"{best_gained // 16}。rank 之所以有效，是因为本仓库的 LoRA 没有 alpha/rank "
                "缩放，加 rank 顺带放大了等效更新幅度，等于变相提了学习率。",
                "",
                "**所以瓶颈既不在基座容量，也不在适配器容量，而在一个从未被扫过的超参数。**",
                "",
                "REPORT.md 里「要突破天花板该换模型」的建议应当修正为：",
                "**先扫学习率 → 再考虑 rank → 最后才谈换基座**。",
            ]
        elif control_verdict == "capacity":
            lines += [
                "",
                "学习率对照（下一节）排除了「只是有效学习率更大」这一解释，"
                "收益确实来自适配器容量。REPORT.md 的建议应修正为：先把 rank 调够，再谈换基座。",
            ]
        else:
            lines += [
                "",
                "**但这还不足以反过来断言「瓶颈是 LoRA 容量」**——见下一节的混淆因素。",
            ]

        # rank 曲线是否单调，本身就是一条判据：若非单调（先升后降），
        # 更符合「rank 在充当有效学习率」而非「容量在起作用」。
        ordered = [(r, d["accuracy"]) for _, r, d in available]
        peak_rank, peak_acc = max(ordered, key=lambda item: item[1])
        if peak_rank != max_rank:
            tail = [f"rank {r}={a:.4f}" for r, a in ordered if r > peak_rank]
            lines += [
                "",
                f"**旁证：rank 曲线并不单调。** 峰值在 rank {peak_rank}（{peak_acc:.4f}），"
                f"再往上反而回落（{'、'.join(tail)}）。若 rank 提供的是容量，"
                "多给容量不该变差；若 rank 实际充当的是有效学习率，那么冲过最优点后变差"
                "正是预期行为。这与学习率对照的结论方向一致。",
            ]
    else:
        lines += [
            f"**rank 从 {min_rank} 放大到 {max_rank}（可训练参数 "
            f"{trainable_params(available[0][2], min_rank) / 1e6:.2f}M → "
            f"{trainable_params(available[-1][2], max_rank) / 1e6:.2f}M）没有任何一点显著优于锚点。**",
            "",
            "瓶颈不在 LoRA 容量。既然给适配器几十倍参数都换不来更多答对的题，"
            "限制只能来自 64M 基座本身——REPORT.md 里那条原本靠间接证据支撑的断言，"
            "现在有了直接证据。",
            "",
            "**推论**：继续在 64M 基座上调 rank 是浪费时间，要往上走只能换更大的基座。",
        ]

    lines += control_lines

    # 这条实现细节对两种结论的含义相反，必须跟着结论走，不能写死一句话。
    lines += ["", "## 一处需要说明的实现细节", "",
              "`model/model_lora.py` 的 LoRA 没有 `alpha/rank` 缩放，前向就是 `B(A(x))`。",
              "因此固定学习率下，rank 越大不仅容量越大，等效更新幅度也越大。"]
    if gained and control_verdict is None:
        lines += [
            "",
            "**这是本结论的一个混淆因素。** 高 rank 点同时改变了两件事——可训练参数量与",
            "等效更新幅度——因此「rank 128 更好」未必等于「容量是瓶颈」，也可能只是它",
            "拿到了更大的有效学习率。要分离这两者，需要一个把学习率而非 rank 调上去的",
            "对照点（rank 16 + 更大 lr，见 `run_capacity_control.sh`）。",
            "**在该对照点跑出来之前，本节结论只能说到「调 rank 有效」，",
            "不能说到「瓶颈是 LoRA 容量」。**",
        ]
    elif gained and control_verdict == "magnitude":
        lines += [
            "",
            "上一节的学习率对照表明，这个混淆因素确实在起作用：不加容量、只放大学习率",
            "就能达到同等准确率。因此本实验的正确结论是「原配置欠训练」，而不是「容量不足」。",
        ]
    elif gained and control_verdict == "inconclusive":
        lines += [
            "",
            "上一节的学习率对照未能分辨这个混淆因素（检验功效不足）。因此本报告的结论",
            "仍然只停在「在这个基座上调 rank 有效、0.86 不是基座上限」，",
            "**不延伸到「瓶颈是 LoRA 容量」**。",
        ]
    elif gained and control_verdict == "capacity":
        lines += [
            "",
            "上一节的学习率对照已经排除了这个混淆因素：单纯放大学习率追不上高 rank，",
            "收益确实来自容量。",
        ]

    if missing:
        lines += ["", f"> 未纳入（缺少评测产物）：{', '.join(missing)}"]

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nRANK_REPORT_WRITTEN {args.output}")


def build_controls(anchor, anchor_acc, available, gained):
    """跑出学习率对照的章节文本与判定。返回 (lines, verdict, (最佳 lr 描述, 准确率))。"""
    lines: list[str] = []
    controls = [(d, r, note, load(d)) for d, r, note in CONTROLS]
    controls = [(d, r, note, data) for d, r, note, data in controls if data]
    control_verdict = None
    control_best = None
    if controls and gained:
        best_gained_rank = max(gained)
        best_rank_point = next(d for _, r, d in available if r == best_gained_rank)
        lines += ["", "## 容量 vs 更新幅度：学习率对照", "",
                  "rank 同时改变了可训练参数量与等效更新幅度。下面的对照把 rank 固定在 16，",
                  f"只放大学习率，看能否不靠加容量就达到 rank {best_gained_rank} 的 "
                  f"{best_rank_point['accuracy']:.4f}。", "",
                  "| 对照点 | 准确率 | 最优 val_loss | vs 锚点 p | vs 最佳 rank 点 p |",
                  "| --- | --- | --- | --- | --- |"]
        reached, below_best = [], []
        for directory, _, note, data in controls:
            _, _, p_anchor = mcnemar(anchor["correct"], data["correct"])
            _, _, p_best = mcnemar(best_rank_point["correct"], data["correct"])
            # 「达到」= 显著高于锚点，且与最佳 rank 点无显著差异
            if p_anchor < ALPHA and data["accuracy"] > anchor_acc and p_best >= ALPHA:
                reached.append((note, data["accuracy"]))
            # 「确实追不上」= 显著低于最佳 rank 点
            if p_best < ALPHA and data["accuracy"] < best_rank_point["accuracy"]:
                below_best.append(note)
            lines.append(f"| {note} | {data['accuracy']:.4f} | "
                         f"{data['summary']['best_val_loss']:.4f} | {p_anchor:.4f} | {p_best:.4f} |")
        lines += [""]
        if reached:
            control_verdict = "magnitude"
            best_control = max(reached, key=lambda item: item[1])
            lines += [
                f"**只调学习率就达到了同等水平（{best_control[0]}）。** 因此 rank "
                f"{best_gained_rank} 的收益主要来自更大的等效更新幅度，而不是更多的容量——"
                "换句话说，原来的 rank 16 配置只是**欠训练**，不是容量不够。",
                "",
                "实践含义：先把学习率调对，再考虑加 rank。",
            ]
        elif below_best and len(below_best) == len(controls):
            control_verdict = "capacity"
            lines += [
                f"**每一个学习率对照都显著低于 rank {best_gained_rank}。** "
                "更新幅度这条解释被排除，收益确实来自 LoRA 容量。",
                "",
                f"实践含义：在这个基座上，rank 16（0.39M，占基座 0.62%）确实不够用；"
                f"瓶颈是适配器容量而非 64M 基座本身。",
            ]
        else:
            # 常见且必须如实承认的第三种情况：对照点既没有显著高于锚点，
            # 也没有显著低于最佳 rank 点——n=400 分辨不了这个量级的差距。
            control_verdict = "inconclusive"
            spread = max(d["accuracy"] for _, _, _, d in controls) - anchor_acc
            lines += [
                "**本对照无法判定。** 各对照点既没有显著高于锚点，也没有显著低于 "
                f"rank {best_gained_rank}——它们落在两者之间的灰区里。",
                "",
                f"原因是**检验功效不足**：评测集只有 {anchor['n']} 题，而这里要分辨的差距只有 "
                f"{spread:.2%} 量级。McNemar 看的是改变对错的题数，这个量级下的差异只对应个位数的",
                "翻转题目，达不到显著。",
                "",
                "**能说的**：放大学习率确实带来了一部分提升，方向上与「原配置欠训练」一致；",
                "**不能说的**：它是否足以完全替代加 rank。要分开这两者，需要更大的评测集"
                "（本实验语料上限只够 400 题）或多个随机种子重复，而不是继续在现有设定下加点。",
            ]

    # 记录达到同等水平的那个对照点，供结论段引用
    if control_verdict == "magnitude":
        control_best = (best_control[0].split("（")[0], best_control[1])

    return lines, control_verdict, control_best


if __name__ == "__main__":
    main()
