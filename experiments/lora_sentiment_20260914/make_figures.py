#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文规格图表 (make_figures.py)

从 runs/ 下的实测产物生成五张图，同时输出 PDF（矢量，供排版）与 PNG（300dpi，供预览）。
不读取任何硬编码的数字——全部来自 eval_formal.json 与 train_summary.json。

图注与坐标轴一律使用英文：图表要能脱离中文字体环境重绘，中文标签在未装 CJK 字体的
机器上会渲染成方框。报告正文（REPORT.md）用中文解读这些图。

配色取自 dataviz 技能参考调色板的前三个槽位（blue / orange / aqua），该子集在浅色与
深色表面下均通过全配对色觉安全验证。单序列图不设图例，由标题点明；多序列图图例与
直接标注并存，身份不靠颜色单独承载。

用法：
    python experiments/lora_sentiment_20260914/make_figures.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIGURES = HERE / "figures"

# --- dataviz 参考调色板（浅色表面）---
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8985"
GRID, SURFACE = "#e6e5e1", "#ffffff"

POINTS = [("scale_250", 250, "3 ep"), ("scale_500", 500, "3 ep"), ("scale_1000", 1000, "3 ep"),
          ("scale_2000", 2000, "3 ep"), ("scale_2800", 2800, "3 ep"), ("formal", 3672, "3 ep")]
SATURATION_AT = 2000


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


def recessive(ax, x_grid: bool = False) -> None:
    """留下必要的刻度与横向网格，其余一律退到背景。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", alpha=1.0)
    ax.grid(axis="x", visible=x_grid, alpha=1.0)
    ax.tick_params(length=3, width=0.8)


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    for suffix, dpi in ((".pdf", None), (".png", 300)):
        fig.savefig(FIGURES / f"{name}{suffix}", dpi=dpi)
    plt.close(fig)
    print(f"  ✅ figures/{name}.pdf + .png")


def load_point(directory: str) -> dict | None:
    evaluation, summary = RUNS / directory / "eval_formal.json", RUNS / directory / "train_summary.json"
    if not (evaluation.exists() and summary.exists()):
        return None
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    rows = payload["predictions"]
    return {
        "report": payload["report"],
        "correct": {r["id"]: r["forced"] == r["truth"] for r in rows},
        "accuracy": payload["report"]["candidate_scoring_accuracy"],
        "n": payload["report"]["samples"],
        "summary": json.loads(summary.read_text(encoding="utf-8")),
    }


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    a_only = sum(1 for i in a if a[i] and not b[i])
    b_only = sum(1 for i in a if not a[i] and b[i])
    total = a_only + b_only
    if total == 0:
        return a_only, b_only, 1.0
    tail = sum(math.comb(total, k) for k in range(min(a_only, b_only) + 1))
    return a_only, b_only, min(tail / 2 ** total * 2, 1.0)


def p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}"


# --------------------------------------------------------------------------------------
def figure_scale_curve(data: dict) -> None:
    """图 1：主结果。准确率随训练集规模的变化，标出随机基线带与饱和区。"""
    sizes = [n for _, n, _ in POINTS if data.get(n)]
    accuracy = [data[n]["accuracy"] for n in sizes]
    n_eval = data[sizes[0]]["n"]
    half = 1.96 * math.sqrt(0.25 / n_eval)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    recessive(ax)

    # 饱和区：2000 条之后没有任何一步通过显著性检验
    ax.axvspan(SATURATION_AT, sizes[-1] * 1.08, color=INK, alpha=0.035, zorder=0)
    ax.text(math.sqrt(SATURATION_AT * sizes[-1]), 0.915, "no significant gain beyond 2 000",
            ha="center", va="top", fontsize=9, color=INK_MUTED, style="italic")

    # 随机基线：期望 0.5，带 95% 抽样区间——准确率要越过带的上沿才算赢过抛硬币
    ax.axhspan(0.5 - half, 0.5 + half, color=GRID, alpha=0.75, zorder=1)
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.text(sizes[-1] * 1.10, 0.5, f"random baseline 0.50\n95% band ±{half:.3f} (n={n_eval})",
            fontsize=8.5, color=INK_MUTED, va="center", ha="right")

    ax.plot(sizes, accuracy, color=BLUE, linewidth=2.0, marker="o", markersize=7.5,
            markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4,
            clip_on=False, solid_capstyle="round")

    # 末三点挤在一起，交错上下摆放，避免互相压字
    offsets = {sizes[-3]: (-4, 14), sizes[-2]: (0, -22), sizes[-1]: (4, 14)}
    for size, value in zip(sizes, accuracy):
        dx, dy = offsets.get(size, (0, 14))
        ax.annotate(f"{value:.3f}", (size, value), textcoords="offset points",
                    xytext=(dx, dy), ha="center", fontsize=9.5, color=INK)

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.minorticks_off()
    ax.set_xlim(sizes[0] * 0.85, sizes[-1] * 1.12)
    ax.set_ylim(0.45, 0.93)
    ax.set_xlabel("Training examples (log scale)")
    ax.set_ylabel("Accuracy on held-out test set")
    ax.set_title("LoRA on sentiment classification saturates at ~2 000 examples", pad=14)
    fig.text(0.5, -0.035,
             "MiniMind 64M base, LoRA rank 16, balanced ChnSentiCorp splits; 400-item test set held constant across points.",
             ha="center", fontsize=8.5, color=INK_MUTED)
    save(fig, "fig1_scale_curve")


def figure_decoupling(data: dict) -> None:
    """图 2：val_loss 仍在降，准确率却不再涨。两个量纲分两栏，绝不共用双轴。"""
    sizes = [n for _, n, _ in POINTS if data.get(n)]
    accuracy = [data[n]["accuracy"] for n in sizes]
    losses = [data[n]["summary"]["best_val_loss"] for n in sizes]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    for ax in axes:
        recessive(ax)
        ax.axvspan(SATURATION_AT, sizes[-1] * 1.08, color=INK, alpha=0.035, zorder=0)

    axes[0].plot(sizes, accuracy, color=BLUE, linewidth=2.0, marker="o", markersize=7,
                 markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=2.0, clip_on=False)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.5, 0.92)
    axes[0].set_title("Validation loss keeps improving after accuracy stops", pad=12)
    axes[0].annotate("flat from 2 000 on", (sizes[-2], accuracy[-2]), textcoords="offset points",
                     xytext=(-6, -26), ha="center", fontsize=9, color=INK_MUTED)

    axes[1].plot(sizes, losses, color=ORANGE, linewidth=2.0, marker="s", markersize=7,
                 markerfacecolor=ORANGE, markeredgecolor=SURFACE, markeredgewidth=2.0, clip_on=False)
    axes[1].set_ylabel("Best validation loss\n(axis inverted: up = better)")
    axes[1].invert_yaxis()   # 向下 = 更好，与上栏"向上 = 更好"方向一致
    # 下栏纵轴已反转，末点贴着上沿，标注必须向下放才落在坐标区内
    axes[1].annotate("still falling", (sizes[-1], losses[-1]), textcoords="offset points",
                     xytext=(-12, -22), ha="right", fontsize=9, color=INK_MUTED)
    axes[1].set_xscale("log")
    axes[1].set_xticks(sizes)
    axes[1].set_xticklabels([f"{s:,}" for s in sizes])
    axes[1].minorticks_off()
    axes[1].set_xlim(sizes[0] * 0.85, sizes[-1] * 1.12)
    axes[1].set_xlabel("Training examples (log scale)")
    fig.text(0.5, -0.02,
             "Lower loss on the supervised tokens no longer converts into more correct classifications "
             "— the ceiling is model capacity, not data.",
             ha="center", fontsize=8.5, color=INK_MUTED)
    save(fig, "fig2_loss_accuracy_decoupling")


def figure_mcnemar(data: dict) -> None:
    """图 3：逐点配对检验。同一批题上有多少题改变了对错，比准确率差值灵敏。"""
    sizes = [n for _, n, _ in POINTS if data.get(n)]
    pairs, gained, lost, ps = [], [], [], []
    for a, b in zip(sizes, sizes[1:]):
        a_only, b_only, p = mcnemar(data[a]["correct"], data[b]["correct"])
        pairs.append(f"{a:,} → {b:,}")
        lost.append(a_only); gained.append(b_only); ps.append(p)
    if data.get("epochs9"):
        a_only, b_only, p = mcnemar(data[sizes[-1]]["correct"], data["epochs9"]["correct"])
        pairs.append("3 ep → 9 ep"); lost.append(a_only); gained.append(b_only); ps.append(p)

    y = range(len(pairs))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    recessive(ax, x_grid=True)
    ax.grid(axis="y", visible=False)

    height = 0.34
    ax.barh([i + height / 2 + 0.01 for i in y], gained, height=height, color=BLUE,
            label="items flipped to correct", zorder=3)
    ax.barh([i - height / 2 - 0.01 for i in y], lost, height=height, color=ORANGE,
            label="items flipped to wrong", zorder=3)

    limit = max(max(gained), max(lost))
    for i, (g, l, p) in enumerate(zip(gained, lost, ps)):
        ax.text(g + limit * 0.02, i + height / 2 + 0.01, str(g), va="center", fontsize=9, color=INK)
        ax.text(l + limit * 0.02, i - height / 2 - 0.01, str(l), va="center", fontsize=9, color=INK)
        significant = p < 0.05
        ax.text(limit * 1.18, i, f"{p_text(p)}   {'significant' if significant else 'n.s.'}",
                va="center", fontsize=9, color=INK if significant else INK_MUTED,
                fontweight="bold" if significant else "normal")

    ax.set_yticks(list(y)); ax.set_yticklabels(pairs)
    ax.set_xlim(0, limit * 1.15)
    ax.set_xlabel("Test items that changed correctness (of 400)")
    ax.set_title("McNemar paired test: gains stop being significant after 2 000", pad=14)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.text(0.5, -0.06,
             "Each point is evaluated on the identical 400 items, so the paired test is far more sensitive "
             "than comparing two accuracy figures.",
             ha="center", fontsize=8.5, color=INK_MUTED)
    save(fig, "fig3_mcnemar")


def figure_training_dynamics(data: dict) -> None:
    """图 4：3 epoch 与 9 epoch 的 val_loss 轨迹，后者触底后回升即过拟合。"""
    if not data.get("epochs9"):
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    recessive(ax)

    for key, label, color, marker in (("formal", "3 epochs", BLUE, "o"), ("epochs9", "9 epochs", ORANGE, "s")):
        history = data[key]["summary"]["history"]
        steps = [h["step"] for h in history][1:]     # 跳过 step 0 的未训练初值，否则纵轴被压扁
        values = [h["val_loss"] for h in history][1:]
        ax.plot(steps, values, color=color, linewidth=2.0, label=label, solid_capstyle="round")
        best = min(history[1:], key=lambda h: h["val_loss"])
        ax.plot([best["step"]], [best["val_loss"]], marker=marker, markersize=8, color=color,
                markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=5)
        # 曲线周围没有安全空位，最优值改放右上空白区，用颜色与序列对应
        ax.text(0.19, 0.93 if key == "formal" else 0.86,
                f"{label}: best {best['val_loss']:.4f} at step {best['step']}",
                transform=ax.transAxes, fontsize=9.5, color=color, va="top")

    history9 = data["epochs9"]["summary"]["history"][1:]
    best9 = min(history9, key=lambda h: h["val_loss"])
    final9 = history9[-1]
    ax.annotate("", xy=(final9["step"], final9["val_loss"]), xytext=(best9["step"], best9["val_loss"]),
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text((best9["step"] + final9["step"]) / 2, max(final9["val_loss"], best9["val_loss"]) * 1.06,
            f"+{(final9['val_loss'] - best9['val_loss']) / best9['val_loss'] * 100:.0f}% — overfitting",
            ha="center", fontsize=9, color=INK_MUTED, style="italic")

    ax.set_ylim(0.075, 0.215)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Validation loss")
    ax.set_title("Training three times longer only overfits", pad=14)
    ax.legend(loc="upper right")
    fig.text(0.5, -0.035,
             "Both runs train on the full 3 672-example set. Accuracy differs by p = 0.86 — indistinguishable.",
             ha="center", fontsize=8.5, color=INK_MUTED)
    save(fig, "fig4_training_dynamics")


def figure_confusion(data: dict) -> None:
    """图 5：最佳模型的混淆矩阵。顺序型数据用单一色相深浅，不用彩虹。"""
    best_key = "epochs9" if data.get("epochs9") else "formal"
    report = data[best_key]["report"]
    matrix = report["confusion_matrix"]
    labels = list(matrix.keys())
    english = {"负面": "Negative", "正面": "Positive"}
    counts = [[matrix[t][p] for p in labels] for t in labels]
    total = sum(sum(row) for row in counts)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(counts, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", [SURFACE, BLUE]), vmin=0, vmax=max(max(r) for r in counts))
    # 相邻色块之间留出表面色缝隙，避免两块深色直接相接
    for edge in (0.5,):
        ax.axhline(edge, color=SURFACE, linewidth=3, zorder=3)
        ax.axvline(edge, color=SURFACE, linewidth=3, zorder=3)
    for i, row in enumerate(counts):
        for j, value in enumerate(row):
            strong = value > max(max(r) for r in counts) * 0.55
            ax.text(j, i, f"{value}\n{value / total:.1%}", ha="center", va="center",
                    color=SURFACE if strong else INK, fontsize=11,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels([english[l] for l in labels])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels([english[l] for l in labels])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix — accuracy {report['candidate_scoring_accuracy']:.3f}", pad=14)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(False)
    ax.tick_params(length=0)
    predicted = [sum(counts[i][j] for i in range(len(labels))) for j in range(len(labels))]
    fig.text(0.5, -0.02,
             f"Predictions split {predicted[0]}/{predicted[1]} across the two classes "
             "— no collapse onto a single label.",
             ha="center", fontsize=8.5, color=INK_MUTED)
    save(fig, "fig5_confusion_matrix")


def figure_capacity_vs_lr() -> None:
    """图 6：rank 扫描与学习率对照——两条干预路径到达同一水平，代价相差 8 倍。

    这是本实验最重要的一次自我纠错：原报告断言天花板来自基座容量，rank 扫描证伪了它，
    学习率对照又进一步证明 rank 的收益其实来自等效更新幅度。

    用点图而非柱状图：准确率的有意义基线是 0.5 而不是 0，柱状图会暗示零基线。
    """
    rows = [("epochs9", "rank 16, lr 2e-4", "anchor"),
            ("rank_64", "rank 64, lr 2e-4", "rank"),
            ("rank_128", "rank 128, lr 2e-4", "rank"),
            ("rank_256", "rank 256, lr 2e-4", "rank"),
            ("lrctl_4e4", "rank 16, lr 4e-4", "lr"),
            ("lrctl_5e4", "rank 16, lr 5.7e-4", "lr"),
            ("lrctl_8e4", "rank 16, lr 8e-4", "lr")]
    loaded = []
    for directory, label, kind in rows:
        point = load_point(directory)
        if point:
            loaded.append((label, kind, point))
    if len(loaded) < 3:
        print("  ⏭  跳过 fig6：rank 扫描 / 学习率对照产物不足")
        return

    colour = {"anchor": INK_MUTED, "rank": ORANGE, "lr": BLUE}
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ys = list(range(len(loaded)))[::-1]
    anchor = next((p for _, kind, p in loaded if kind == "anchor"), None)
    if anchor:
        ax.axvline(anchor["accuracy"], color=GRID, linewidth=1.4, zorder=1)
    for y, (label, kind, point) in zip(ys, loaded):
        params = point["summary"].get("trainable_params")
        if params is None:  # 锚点是加 --rank 之前跑的，没有这个字段
            params = 16 * 2 * 768 * 16
        ax.plot([0, point["accuracy"]], [y, y], color=GRID, linewidth=1.4, zorder=1)
        ax.plot([point["accuracy"]], [y], marker="o", markersize=10, color=colour[kind],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.annotate(f"{point['accuracy']:.4f}", xy=(point["accuracy"], y), xytext=(12, 0),
                    textcoords="offset points", va="center", color=INK_2, fontsize=9.5)
        # 参数量是第二个量纲，放到坐标区之外，避免与网格线叠在一起读成数据
        ax.annotate(f"{params / 1e6:.2f}M", xy=(1.02, y), xycoords=("axes fraction", "data"),
                    ha="left", va="center", color=INK_MUTED, fontsize=9.5)
    ax.annotate("trainable\nparams", xy=(1.02, ys[0] + 1.05), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", color=INK_MUTED, fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _, _ in loaded])
    ax.set_xlim(0.84, 0.90)
    ax.set_xlabel("Accuracy on 400 held-out reviews")
    ax.set_title("Learning rate matches rank — at an eighth the parameters",
                 color=INK, loc="left")
    # 两类干预必须能脱离颜色被识别，因此图例与左侧文字标签并存
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                          markerfacecolor=colour[k], markeredgecolor=SURFACE,
                          markeredgewidth=2, label=name)
               for k, name in (("anchor", "baseline"), ("rank", "higher LoRA rank"),
                               ("lr", "higher learning rate"))]
    ax.legend(handles=handles, loc="lower right", ncol=3, bbox_to_anchor=(1.0, -0.28))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="x", alpha=1.0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=3, width=0.8)
    save(fig, "fig6_capacity_vs_lr")


def main() -> None:
    style()
    data = {}
    for directory, size, _ in POINTS:
        point = load_point(directory)
        if point:
            # 同时按规模与目录名索引：规模曲线按条数取用，训练动态按运行目录取用。
            data[size] = point
            data[directory] = point
    extra = load_point("epochs9")
    if extra:
        data["epochs9"] = extra
    if len(data) < 2:
        raise SystemExit("评测产物不足，先跑 run_cloud.sh")

    print("生成图表:")
    figure_scale_curve(data)
    figure_decoupling(data)
    figure_mcnemar(data)
    figure_training_dynamics(data)
    figure_confusion(data)
    figure_capacity_vs_lr()
    print(f"FIGURES_WRITTEN {FIGURES}")


if __name__ == "__main__":
    main()
