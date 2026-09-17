#!/usr/bin/env bash
# ==============================================================================
# 情感分类 LoRA —— 容量 / 更新幅度分离对照
#
# rank 扫描发现 rank 128 显著优于 rank 16（0.8850 vs 0.8575, p=0.0074），
# 推翻了「天花板来自 64M 基座容量」。但 rank 同时改变了两件事：
#   (a) 可训练参数量；
#   (b) 等效更新幅度——本仓库的 LoRA 没有 alpha/rank 缩放，且 B 零初始化，
#       训练后 ||B·A||_F 大致随 sqrt(rank) 增长（AdamW 逐参数归一化梯度，
#       但参数个数随 rank 线性增加，总更新范数仍然随 sqrt(rank) 走）。
#   rank 16 → 128 是 8 倍 rank，约合 sqrt(8) ≈ 2.83 倍的更新范数。
#
# 因此本脚本跑两组对照，其余条件与 rank 扫描完全一致（3672 条 × 9 epoch）：
#   · rank 16 + 放大学习率（4e-4 / 5.7e-4 / 8e-4，5.7e-4 ≈ 2.83 × 2e-4）
#       —— 只放大 (b) 不动 (a)。若能达到 rank 128 的准确率，收益来自更新幅度；
#          若达不到，收益来自容量。
#   · rank 256 —— 看 rank 曲线还会不会继续往上走。
#
# 用法：
#   bash experiments/lora_sentiment_20260914/run_capacity_control.sh
#   bash experiments/lora_sentiment_20260914/run_capacity_control.sh --force
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

EXP_REL="experiments/lora_sentiment_20260914"
RUNS_DIR="${SCRIPT_DIR}/runs"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${RUNS_DIR}"

TS="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/capacity_control_${TS}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
else
    PYTHON="python3"
fi

EPOCHS=9
FORCE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=true; shift ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done

echo "================================================================================"
echo " 容量 / 更新幅度分离对照   $(date '+%Y-%m-%d %H:%M:%S')"
echo " 日志: ${LOG_FILE}"
echo "================================================================================"

if [ -f "${ROOT_DIR}/out/PLACEHOLDER_README.txt" ]; then
    echo "❌ 检测到占位权重，拒绝执行"; exit 1
fi

# "运行目录名|rank|学习率"
POINTS=(
    "lrctl_4e4|16|0.0004"
    "lrctl_5e4|16|0.00057"
    "lrctl_8e4|16|0.0008"
    "rank_256|256|0.0002"
)

for POINT in "${POINTS[@]}"; do
    IFS='|' read -r NAME RANK LR <<< "${POINT}"
    POINT_DIR="${RUNS_DIR}/${NAME}"
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "▶ ${NAME}  (rank=${RANK}, lr=${LR})"
    echo "--------------------------------------------------------------------------------"
    if [ "${FORCE}" != true ] && [ -f "${POINT_DIR}/eval_formal.json" ]; then
        echo "⏭  已有产物，跳过（--force 可重跑）"
        continue
    fi
    "${PYTHON}" "${EXP_REL}/train_sentiment_lora.py" --run-dir "${POINT_DIR}" --overwrite \
        --rank "${RANK}" --epochs "${EPOCHS}" --learning-rate "${LR}"
    "${PYTHON}" "${EXP_REL}/eval_sentiment.py" --split formal \
        --checkpoint "${POINT_DIR}/best_lora.pth"
done

echo ""
echo "================================================================================"
echo " 对照完成  $(date '+%Y-%m-%d %H:%M:%S')"
echo " 分析请在本地执行: python ${EXP_REL}/analyze_rank.py"
echo "================================================================================"
