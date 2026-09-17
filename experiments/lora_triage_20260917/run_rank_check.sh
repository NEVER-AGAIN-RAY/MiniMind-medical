#!/usr/bin/env bash
# ==============================================================================
# 科室分诊 —— 适配器容量检验
#
# 规模曲线在 9600 条处饱和，且语料还剩 98%，所以平台期不是数据造成的。但曲线各点
# 的 LoRA rank 全部是 16，而 lora_sentiment_20260914 的 rank 扫描已经证明：在那个
# 任务上，看似的「模型容量天花板」其实是适配器容量天花板（rank 16 → 128 让准确率
# 从 0.8575 涨到 0.8850，p=0.0074）。
#
# 本脚本在同样的 12000 条训练集上补两个高 rank 点，其余条件与 runs/formal 完全一致，
# 用来分开「基座容量」与「适配器容量」两种解释。
#
# 用法：
#   bash experiments/lora_triage_20260917/run_rank_check.sh
#   bash experiments/lora_triage_20260917/run_rank_check.sh --ranks "128"
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
LOG_FILE="${LOG_DIR}/rank_check_${TS}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
else
    PYTHON="python3"
fi

RANKS="64 128"
FORCE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ranks) RANKS="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "❌ 未知参数: $1"; exit 1 ;;
    esac
done

echo "================================================================================"
echo " 科室分诊 适配器容量检验   $(date '+%Y-%m-%d %H:%M:%S')"
echo " 锚点: runs/formal（rank 16，12000 条 × 3 epoch）"
echo " 扫描: rank ${RANKS}"
echo " 日志: ${LOG_FILE}"
echo "================================================================================"

if [ -f "${ROOT_DIR}/out/PLACEHOLDER_README.txt" ]; then
    echo "❌ 检测到占位权重，拒绝执行"; exit 1
fi
if [ ! -f "${RUNS_DIR}/formal/eval_formal.json" ]; then
    echo "❌ 缺少锚点 runs/formal/eval_formal.json，请先跑 run_cloud.sh"; exit 1
fi

for RANK in ${RANKS}; do
    POINT_DIR="${RUNS_DIR}/rank_${RANK}"
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "▶ rank ${RANK}"
    echo "--------------------------------------------------------------------------------"
    if [ "${FORCE}" != true ] && [ -f "${POINT_DIR}/eval_formal.json" ]; then
        echo "⏭  已有产物，跳过（--force 可重跑）"
        continue
    fi
    "${PYTHON}" "${EXP_REL}/train_triage_lora.py" --run-dir "${POINT_DIR}" --overwrite --rank "${RANK}"
    "${PYTHON}" "${EXP_REL}/eval_triage.py" --split formal --checkpoint "${POINT_DIR}/best_lora.pth"
done

echo ""
echo "--------------------------------------------------------------------------------"
"${PYTHON}" "${EXP_REL}/analyze_scale.py"
echo "================================================================================"
echo " 适配器容量检验结束  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
