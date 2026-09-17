#!/usr/bin/env bash
# ==============================================================================
# 科室分诊 —— 学习率对照
#
# 情感实验的结论已经改写：那里看似的「容量天花板」其实是 lr 2e-4 欠训练——
# rank 16 把学习率提到 8e-4 就达到了 rank 128 的水平（0.8875 vs 0.8850，
# 参数量只有 1/8）。本实验的规模曲线同样全程用 lr 2e-4，因此 9600 条处的
# 平台期有同样的嫌疑。
#
# 本脚本在 12000 条上只改学习率，其余与 runs/formal 完全一致。
#
# 用法：
#   bash experiments/lora_triage_20260917/run_lr_check.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

EXP_REL="experiments/lora_triage_20260917"
RUNS_DIR="${SCRIPT_DIR}/runs"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${RUNS_DIR}"

TS="$(date '+%Y%m%d_%H%M%S')"
exec > >(tee -a "${LOG_DIR}/lr_check_${TS}.log") 2>&1

PYTHON="${ROOT_DIR}/.venv/bin/python"
[ -x "${PYTHON}" ] || PYTHON=python3

LRS="0.0004 0.0008"
FORCE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lrs) LRS="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done

echo "================================================================================"
echo " 科室分诊 学习率对照   $(date '+%Y-%m-%d %H:%M:%S')"
echo " 锚点: runs/formal（rank 16, lr 2e-4, 12000 条 × 3 epoch）"
echo " 对照: lr ${LRS}"
echo "================================================================================"

for LR in ${LRS}; do
    # 0.0004 -> lr_4e4
    NAME="lr_$(printf '%s' "${LR}" | awk '{printf "%ge4", $1*10000}' | tr -d '.')"
    POINT_DIR="${RUNS_DIR}/${NAME}"
    echo ""
    echo "▶ ${NAME}  (lr=${LR})"
    if [ "${FORCE}" != true ] && [ -f "${POINT_DIR}/eval_formal.json" ]; then
        echo "⏭  已有产物，跳过"; continue
    fi
    "${PYTHON}" "${EXP_REL}/train_triage_lora.py" --run-dir "${POINT_DIR}" --overwrite --learning-rate "${LR}"
    "${PYTHON}" "${EXP_REL}/eval_triage.py" --split formal --checkpoint "${POINT_DIR}/best_lora.pth"
done

echo ""
echo " 学习率对照结束  $(date '+%Y-%m-%d %H:%M:%S')"
