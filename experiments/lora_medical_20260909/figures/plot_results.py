"""Reproduce the validation-loss figure from the saved log; no model execution."""
from pathlib import Path
import csv
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
log_path = EXP / "logs/run_20260910_115857.log"
log = log_path.read_text()
points = [(0, float(re.search(r"Validation \[step 0\], val_loss: ([\d.]+)", log)[1]))]
points.extend((int(s), float(v)) for s, v in re.findall(
    r"Epoch:\[1/1\]\((\d+)/625\), val_loss: ([\d.]+)", log))
last = re.search(r"Validation \[final step (\d+)\], val_loss: ([\d.]+)", log)
points.append((int(last[1]), float(last[2])))
assert len(points) == 14 and points[-1][0] == 625

with (HERE / "validation_curve.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["optimizer_step", "validation_cross_entropy"])
    writer.writerows(points)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 11, "axes.linewidth": 0.8,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white",
})
blue, dark = "#0072B2", "#242424"
fig, ax = plt.subplots(figsize=(7.4, 4.8))
fig.subplots_adjust(left=0.13, right=0.965, top=0.77, bottom=0.235)
fig.text(0.13, 0.945, "MiniMind medical LoRA fine-tuning", fontsize=14, weight="bold", color=dark)
fig.text(0.13, 0.895, "64M base model  |  5,000 training examples  |  1 epoch  |  rank 16",
         fontsize=9, color="#4A4A4A")
fig.text(0.13, 0.855, "Batch size 8  |  Sequence length 512  |  100 held-out validation examples",
         fontsize=9, color="#4A4A4A")

for a in (ax,):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color="#E3E6E8", linewidth=0.65)
    a.set_axisbelow(True)
    a.tick_params(direction="out", length=3.5, color="#444444")

steps, losses = zip(*points)
ax.plot(steps, losses, color=blue, linewidth=1.8, marker="o", markersize=4,
        markeredgecolor="white", markeredgewidth=0.55)
ax.set_xlabel("Optimizer step")
ax.set_ylabel("Answer-token cross-entropy (nats)")
ax.set_xlim(-12, 650)
ax.set_ylim(1.445, 1.595)
ax.set_xticks([0, 125, 250, 375, 500, 625])
ax.yaxis.set_major_locator(MultipleLocator(0.03))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.annotate(f"{losses[0]:.4f}", xy=(0, losses[0]), xytext=(11, 7),
            textcoords="offset points", fontsize=9, color=blue)
ax.annotate(f"{losses[-1]:.4f}", xy=(625, losses[-1]), xytext=(-9, 13),
            textcoords="offset points", ha="right", fontsize=9, color=blue,
            arrowprops={"arrowstyle": "-", "color": blue, "lw": 0.8})
reduction = 100 * (1 - losses[-1] / losses[0])
ax.text(0.43, 0.83, f"Loss reduction: {reduction:.1f}%", transform=ax.transAxes,
        fontsize=10, color=blue)
ax.text(0.43, 0.73, "Validation loss during training", transform=ax.transAxes,
        fontsize=8.5, color="#555555")

fig.text(0.13, 0.115,
         "Single run (seed 42); 14 measured checkpoints connected by straight lines; no smoothing.",
         fontsize=8, color="#505050")
fig.text(0.13, 0.073,
         "The y-axis is zoomed. Loss reduction does not establish an improvement in medical accuracy.",
         fontsize=8, color="#505050")

fig.savefig(HERE / "minimind_lora_results.png", dpi=600)
fig.savefig(HERE / "minimind_lora_results.pdf")
fig.savefig(HERE / "preview.png", dpi=140)
plt.close(fig)

metadata = {
    "training_log": str(log_path.relative_to(EXP)),
    "validation_points": points,
    "curve_loss_relative_reduction_percent": reduction,
    "note": "Curve uses in-training validation. Single run; no smoothing or error bands. Multiple-choice evaluation is excluded from this figure.",
}
(HERE / "figure_data.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(HERE / "minimind_lora_results.png")
