#!/usr/bin/env bash
# ==============================================================================
# 科室分诊 —— 规模曲线在调好的学习率下重跑
#
# run_cloud.sh 的 Step 5 把整条规模曲线跑在 lr 2e-4 上，并据此得出「9600 条饱和」。
# 但 run_lr_check.sh 随后证明 2e-4 本身就偏低——同样的 12000 条，把 lr 提到 8e-4
# 准确率从 0.7900 升到 0.8333。于是那个饱和点只是「在一个欠训练的学习率下的饱和点」，
# 不是这个任务的固有性质。本脚本在指定学习率下重跑整条曲线，把两者分开。
#
# 12000 条那个点不在这里重跑——run_lr_check.sh 已经产出 runs/lr_<tag>，条件完全一致，
# 直接复用即可，重训一遍只会浪费时间并引入无谓的随机性。
#
# 用法：
#   bash experiments/lora_triage_20260917/run_scale_lr.sh --lr 0.0008
#   bash experiments/lora_triage_20260917/run_scale_lr.sh --lr 0.0016 --sizes "600 2400 9600"
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
exec > >(tee -a "${LOG_DIR}/scale_lr_${TS}.log") 2>&1

PYTHON="${ROOT_DIR}/.venv/bin/python"
[ -x "${PYTHON}" ] || PYTHON=python3

LR=""
SIZES="600 1200 2400 4800 9600"
FORCE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lr) LR="$2"; shift 2 ;;
        --sizes) SIZES="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done
[ -n "${LR}" ] || { echo "❌ 必须用 --lr 指定学习率，例如 --lr 0.0008"; exit 1; }

# 与 run_lr_check.sh 完全相同的命名规则，保证 12000 点能对上 runs/lr_<tag>
TAG="$(printf '%s' "${LR}" | awk '{printf "%ge4", $1*10000}' | tr -d '.')"
ANCHOR_12000="${RUNS_DIR}/lr_${TAG}"

echo "================================================================================"
echo " 科室分诊 规模曲线重跑   $(date '+%Y-%m-%d %H:%M:%S')"
echo " 学习率: ${LR}（tag ${TAG}）"
echo " 规模点: ${SIZES}"
echo " 12000 点: 复用 ${ANCHOR_12000}"
echo "================================================================================"

if [ ! -f "${ANCHOR_12000}/eval_formal.json" ]; then
    echo "⚠️  ${ANCHOR_12000}/eval_formal.json 不存在——曲线会缺 12000 这个点。"
    echo "    先跑: bash ${EXP_REL}/run_lr_check.sh --lrs ${LR}"
fi

for SIZE in ${SIZES}; do
    POINT_DIR="${RUNS_DIR}/scale_${SIZE}_lr${TAG}"
    echo ""
    echo "▶ scale_${SIZE}_lr${TAG}  (${SIZE} 条, lr=${LR})"
    if [ "${FORCE}" != true ] && [ -f "${POINT_DIR}/eval_formal.json" ]; then
        echo "⏭  已有产物，跳过"; continue
    fi
    "${PYTHON}" "${EXP_REL}/train_triage_lora.py" --run-dir "${POINT_DIR}" --overwrite \
        --learning-rate "${LR}" \
        --train-file "${EXP_REL}/data/scale/train_${SIZE}.jsonl"
    "${PYTHON}" "${EXP_REL}/eval_triage.py" --split formal \
        --checkpoint "${POINT_DIR}/best_lora.pth"
done

echo ""
echo " 规模曲线重跑结束  $(date '+%Y-%m-%d %H:%M:%S')"
echo " 分析: ${PYTHON} ${EXP_REL}/analyze_scale.py --curve-lr ${TAG}"
