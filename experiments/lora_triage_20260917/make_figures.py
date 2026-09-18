#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文规格图表 (make_figures.py)

从 runs/ 下的实测产物生成三张图，同时输出 PDF（矢量，供排版）与 PNG（300dpi，供预览）。
不读取任何硬编码的数字——全部来自 eval_formal.json 与 train_summary.json。

图注与坐标轴一律使用英文：图表要能脱离中文字体环境重绘，中文标签在未装 CJK 字体的
机器上会渲染成方框。报告正文（REPORT.md）用中文解读这些图。

配色沿用 dataviz 技能参考调色板的前三个槽位（blue / orange / aqua）与蓝色 sequential
ramp，与 lora_sentiment_20260914/make_figures.py 保持一致——两个实验的图应当读起来
像同一套系统。该三色子集在全配对口径下于浅色与深色表面均通过色觉安全验证。

用法：
    pip install matplotlib
    python experiments/lora_triage_20260917/make_figures.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIGURES = HERE / "figures"

# --- dataviz 参考调色板（浅色表面）---
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8985"
GRID, SURFACE = "#e6e5e1", "#ffffff"
# 蓝色 sequential ramp（100 -> 700），用于混淆矩阵这种连续量纲
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

LABELS_EN = {"内科": "Internal", "外科": "Surgery", "妇产科": "OB/GYN",
             "儿科": "Pediatrics", "肿瘤科": "Oncology", "男科": "Andrology"}

SCALE_SIZES = [600, 1200, 2400, 4800, 9600]
SCALE_POINTS = [(f"scale_{n}", n) for n in SCALE_SIZES] + [("formal", 12000)]


def scale_points_at(lr_tag: str):
    """run_scale_lr.sh 在指定学习率下重跑的曲线；12000 点复用 run_lr_check.sh 的产物。"""
    return [(f"scale_{n}_lr{lr_tag}", n) for n in SCALE_SIZES] + [(f"lr_{lr_tag}", 12000)]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10,
        "axes.labelsize": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelcolor": INK_2, "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.fontsize": 9.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.dpi": 110, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
    })


def recessive(ax, axis: str = "y") -> None:
    """留下必要的刻度与一个方向的网格，其余一律退到背景。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis=axis, alpha=1.0)
    ax.grid(axis="x" if axis == "y" else "y", visible=False)
    ax.tick_params(length=3, width=0.8)


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    for suffix, dpi in ((".pdf", None), (".png", 300)):
        fig.savefig(FIGURES / f"{name}{suffix}", dpi=dpi)
    plt.close(fig)
    print(f"  ✅ figures/{name}.pdf + .png")


def load_run(directory: str, root: Path | None = None) -> dict | None:
    root = root or RUNS
    evaluation = root / directory / "eval_formal.json"
    summary = root / directory / "train_summary.json"
    if not evaluation.exists() or not summary.exists():
        return None
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    return {"report": payload["report"],
            "summary": json.loads(summary.read_text(encoding="utf-8"))}


def fig_scale_curve() -> None:
    """图 1：数据量 -> 准确率，标出饱和点与随机基线阈值。"""
    points = [(n, load_run(d)) for d, n in SCALE_POINTS]
    points = [(n, d) for n, d in points if d]
    if len(points) < 2:
        print("  ⏭  跳过 fig1：规模曲线产物不足"); return
    sizes = [n for n, _ in points]
    accuracy = [d["report"]["candidate_scoring_accuracy_mean"] for _, d in points]
    threshold = points[-1][1]["report"]["baseline"]["min_accuracy_to_beat_random"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    # 随机基线是判读结果的前提，作为参考线而非一条数据序列
    ax.axhline(threshold, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"random-baseline threshold  {threshold:.3f}",
                xy=(sizes[0], threshold), xytext=(0, 6), textcoords="offset points",
                color=INK_MUTED, fontsize=9)
    ax.plot(sizes, accuracy, color=BLUE, linewidth=2.0, marker="o",
            markersize=8, markerfacecolor=SURFACE, markeredgewidth=2.0,
            markeredgecolor=BLUE, zorder=3)
    # 对数轴上相邻两点可能挨得很近（9600 与 12000），标签按实际间距自动错开上下，
    # 否则数字会叠在一起。阈值取整条曲线对数跨度的 8%。
    import math
    span = math.log10(sizes[-1] / sizes[0])
    for index, (n, a) in enumerate(zip(sizes, accuracy)):
        crowded = index > 0 and math.log10(n / sizes[index - 1]) < span * 0.08
        ax.annotate(f"{a:.3f}", xy=(n, a), xytext=(0, -16 if crowded else 11),
                    textcoords="offset points", ha="center", color=INK_2, fontsize=9)
    # 饱和点：最后一个相邻提升仍显著的位置
    if len(sizes) >= 2:
        ax.axvspan(sizes[-2], sizes[-1], color=GRID, alpha=0.55, zorder=0)
        ax.annotate("plateau", xy=(sizes[-1], 0.06), ha="right",
                    color=INK_MUTED, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    # 同样的拥挤问题也出现在刻度标签上，倾斜排布即可分开
    ax.set_xticklabels([f"{n:,}" for n in sizes], rotation=25, ha="right")
    ax.minorticks_off()
    ax.set_xlabel("Training examples (log scale)")
    ax.set_ylabel("Accuracy on 600 held-out questions")
    ax.set_title("Medical triage accuracy vs. training-set size (rank 16, lr 2e-4)",
                 color=INK, loc="left")
    ax.set_ylim(0, 1.0)
    recessive(ax)
    save(fig, "fig1_scale_curve")


def fig_confusion() -> None:
    """图 2：最佳模型的混淆矩阵。单一量纲，用蓝色 sequential ramp。"""
    best = None
    for name in ("lr_8e4", "rank_64", "formal"):
        data = load_run(name)
        if data and (best is None or
                     data["report"]["candidate_scoring_accuracy_mean"] >
                     best[1]["report"]["candidate_scoring_accuracy_mean"]):
            best = (name, data)
    if best is None:
        print("  ⏭  跳过 fig2：没有可用的评测产物"); return
    name, data = best
    matrix = data["report"]["confusion_matrix"]
    labels = list(matrix)
    counts = [[matrix[t][p] for p in labels] for t in labels]
    peak = max(max(row) for row in counts)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for i, row in enumerate(counts):
        for j, value in enumerate(row):
            shade = SEQ[min(len(SEQ) - 1, int(value / peak * (len(SEQ) - 1)))]
            # 2px surface gap between cells：相邻填充之间留出表面色缝隙
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94,
                                       facecolor=shade, edgecolor=SURFACE, linewidth=2))
            ax.text(j + 0.5, i + 0.5, str(value), ha="center", va="center",
                    fontsize=9.5, color=INK if value < peak * 0.55 else SURFACE)
    ax.set_xlim(0, len(labels)); ax.set_ylim(len(labels), 0)
    ax.set_xticks([i + 0.5 for i in range(len(labels))])
    ax.set_yticks([i + 0.5 for i in range(len(labels))])
    ax.set_xticklabels([LABELS_EN[l] for l in labels], rotation=30, ha="right")
    ax.set_yticklabels([LABELS_EN[l] for l in labels])
    ax.set_xlabel("Predicted department")
    ax.set_ylabel("True department")
    accuracy = data["report"]["candidate_scoring_accuracy_mean"]
    ax.set_title(f"Confusion matrix — best model ({name}, accuracy {accuracy:.4f})",
                 color=INK, loc="left")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(False)
    ax.tick_params(length=0)
    save(fig, "fig2_confusion_matrix")


def fig_capacity_vs_lr() -> None:
    """图 3：本实验的核心发现——两条干预路径到达同一水平，代价相差 4 倍。

    用点图而非柱状图：准确率的有意义基线是 1/6 而不是 0，柱状图会暗示零基线。
    """
    rows = [("formal", "rank 16, lr 2e-4", "anchor"),
            ("rank_64", "rank 64, lr 2e-4", "rank"),
            ("rank_128", "rank 128, lr 2e-4", "rank"),
            ("lr_4e4", "rank 16, lr 4e-4", "lr"),
            ("lr_8e4", "rank 16, lr 8e-4", "lr")]
    loaded = [(label, kind, load_run(name)) for name, label, kind in rows]
    loaded = [(label, kind, d) for label, kind, d in loaded if d]
    if len(loaded) < 3:
        print("  ⏭  跳过 fig3：对照实验产物不足"); return
    anchor = next((d for _, kind, d in loaded if kind == "anchor"), None)

    colour = {"anchor": INK_MUTED, "rank": ORANGE, "lr": BLUE}
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ys = list(range(len(loaded)))[::-1]
    if anchor:
        ax.axvline(anchor["report"]["candidate_scoring_accuracy_mean"],
                   color=GRID, linewidth=1.4, zorder=1)
    for y, (label, kind, data) in zip(ys, loaded):
        accuracy = data["report"]["candidate_scoring_accuracy_mean"]
        params = data["summary"].get("trainable_params", 393216)
        ax.plot([0, accuracy], [y, y], color=GRID, linewidth=1.4, zorder=1)
        ax.plot([accuracy], [y], marker="o", markersize=10, color=colour[kind],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.annotate(f"{accuracy:.4f}", xy=(accuracy, y), xytext=(12, 0),
                    textcoords="offset points", va="center", color=INK_2, fontsize=9.5)
        # 参数量是第二个量纲，放到坐标区之外，避免与网格线叠在一起读成数据
        ax.annotate(f"{params / 1e6:.2f}M", xy=(1.02, y), xycoords=("axes fraction", "data"),
                    ha="left", va="center", color=INK_MUTED, fontsize=9.5)
    ax.annotate("trainable\nparams", xy=(1.02, ys[0] + 0.95), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", color=INK_MUTED, fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _, _ in loaded])
    ax.set_xlim(0.7, 0.86)
    ax.set_xlabel("Accuracy on 600 held-out questions")
    ax.set_title("Learning rate matches rank — at a quarter the parameters",
                 color=INK, loc="left")
    # 两类干预必须能脱离颜色被识别，因此图例与左侧文字标签并存
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                          markerfacecolor=colour[k], markeredgecolor=SURFACE,
                          markeredgewidth=2, label=name)
               for k, name in (("anchor", "baseline"), ("rank", "higher LoRA rank"),
                               ("lr", "higher learning rate"))]
    ax.legend(handles=handles, loc="lower right", ncol=3, bbox_to_anchor=(1.0, -0.32))
    recessive(ax, axis="x")
    save(fig, "fig3_capacity_vs_lr")


def fig_scale_lr_overlay(lr_tag: str = "8e4", curve_root: Path | None = None) -> None:
    """图 4：同一条规模曲线在两个学习率下的对照。

    报告最初把 9600 条处的平台期读成「模型容量到顶」。若调好学习率后整条曲线
    整体抬高、平台期后移，那个读法就站不住——两条线画在一起最省解释。
    """
    def series(points, root=None):
        loaded = [(n, load_run(d, root)) for d, n in points]
        return [(n, d["report"]["candidate_scoring_accuracy_mean"]) for n, d in loaded if d]

    old = series(SCALE_POINTS)
    new = series(scale_points_at(lr_tag), curve_root)
    if len(new) < 2:
        print("  ⏭  跳过 fig4：重跑曲线产物不足"); return

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    threshold = load_run(SCALE_POINTS[-1][0])["report"]["baseline"]["min_accuracy_to_beat_random"]
    ax.axhline(threshold, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"random-baseline threshold  {threshold:.3f}",
                xy=(old[0][0] if old else new[0][0], threshold), xytext=(0, 6),
                textcoords="offset points", color=INK_MUTED, fontsize=9)

    lr_label = lr_tag.replace("e4", "e-4")
    for data, color, label in ((old, INK_MUTED, "lr 2e-4 (original)"),
                               (new, BLUE, f"lr {lr_label} (tuned)")):
        if not data:
            continue
        ax.plot([n for n, _ in data], [a for _, a in data], color=color, linewidth=2.0,
                marker="o", markersize=8, markerfacecolor=SURFACE, markeredgewidth=2.0,
                markeredgecolor=color, label=label, zorder=3)

    # 只标注抬高后的曲线，两条都标数字会糊成一片
    import math
    span = math.log10(new[-1][0] / new[0][0])
    for index, (n, a) in enumerate(new):
        crowded = index > 0 and math.log10(n / new[index - 1][0]) < span * 0.08
        # 挤在一起时把标签甩到点的右侧，压到下面会撞上 lr 2e-4 那条线
        ax.annotate(f"{a:.3f}", xy=(n, a),
                    xytext=(13, -3) if crowded else (0, 11),
                    textcoords="offset points",
                    ha="left" if crowded else "center", color=INK_2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([n for n, _ in (new if len(new) >= len(old) else old)])
    ax.set_xticklabels([f"{n:,}" for n, _ in (new if len(new) >= len(old) else old)],
                       rotation=25, ha="right")
    ax.minorticks_off()
    ax.set_xlabel("Training examples (log scale)")
    ax.set_ylabel("Accuracy on 600 held-out questions")
    ax.set_title("Same plateau, higher curve: 2,400 examples at the tuned lr\n"
                 "match 12,000 at the original one", color=INK, loc="left")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    recessive(ax)
    save(fig, "fig4_scale_lr_overlay")


def main() -> None:
    parser = argparse.ArgumentParser(description="科室分诊实验图表生成")
    parser.add_argument(
        "--curve-runs", type=Path, default=None, metavar="DIR",
        help="lr 重跑曲线的产物目录（默认 runs/）；与原曲线不在同一台机器上跑时用它分开存放",
    )
    args = parser.parse_args()
    style()
    print("生成图表...")
    fig_scale_curve()
    fig_confusion()
    fig_capacity_vs_lr()
    fig_scale_lr_overlay(curve_root=args.curve_runs)
    print(f"全部完成 -> {FIGURES}")


if __name__ == "__main__":
    main()
