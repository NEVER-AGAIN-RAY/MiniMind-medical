#!/usr/bin/env bash
# ==============================================================================
# 情感分类 LoRA 实验 —— rank 扫描（容量对照实验）
#
# 目的：REPORT.md 断言「0.86 的天花板来自 64M 基座容量」，但支撑它的是两条间接
# 证据（val_loss 与准确率脱钩、9 epoch 过拟合）。本扫描给出直接证据——
# 把 LoRA 容量从 rank 4 拉到 rank 128（可训练参数 0.10M → 3.15M，32 倍），
# 其余全部固定（3672 条 × 9 epoch，同一批 400 题测试集）：
#   · 若准确率不随 rank 上升 → 瓶颈不在 LoRA 容量，只能在基座，原断言成立；
#   · 若随 rank 上升        → 原断言被证伪，该调的是 rank 而不是换基座。
#
# 锚点 rank 16 复用已有的 runs/epochs9（同样 3672 条 × 9 epoch），不重复训练。
#
# 用法：
#   bash experiments/lora_sentiment_20260914/run_rank_sweep.sh            # 全部 rank
#   bash experiments/lora_sentiment_20260914/run_rank_sweep.sh --ranks "4 8"  # 指定
#   bash experiments/lora_sentiment_20260914/run_rank_sweep.sh --force    # 已有产物也重跑
#
# 已完成的 rank 点默认依据产物自动跳过。
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
LOG_FILE="${LOG_DIR}/rank_sweep_${TS}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "❌ 未找到可用的 Python 解释器"; exit 1
fi

# 锚点 rank 16 = runs/epochs9，与扫描点同为 3672 条 × 9 epoch。
RANKS="4 8 32 64 128"
EPOCHS=9
FORCE=false
ALLOW_PLACEHOLDER=false
RAW_ARGS="$*"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ranks) RANKS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        --allow-placeholder) ALLOW_PLACEHOLDER=true; shift ;;
        -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "❌ 未知参数: $1（--help 查看用法）"; exit 1 ;;
    esac
done

echo "================================================================================"
echo " MiniMind 情感分类 LoRA —— rank 扫描"
echo " 工作目录: ${ROOT_DIR}"
echo " Python  : ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo " 时间    : $(date '+%Y-%m-%d %H:%M:%S')"
echo " rank    : ${RANKS}（锚点 16 复用 runs/epochs9）"
echo " epoch   : ${EPOCHS}"
echo " 参数    : ${RAW_ARGS:-（无）}"
echo " 日志    : ${LOG_FILE}"
echo "================================================================================"

# 占位权重保护：随机初始化权重跑出来的准确率毫无意义，绝不能混进正式结果。
if [ -f "${ROOT_DIR}/out/PLACEHOLDER_README.txt" ] && [ "${ALLOW_PLACEHOLDER}" != true ]; then
    echo "❌ 检测到 out/PLACEHOLDER_README.txt —— 当前基座是随机初始化的占位权重。"
    echo "   请换上真实的 full_sft_768.pth，或在明知是流程干跑时显式追加 --allow-placeholder。"
    exit 1
fi

# 锚点必须已存在，否则扫描没有比较对象。
ANCHOR_DIR="${RUNS_DIR}/epochs9"
if [ ! -f "${ANCHOR_DIR}/eval_formal.json" ]; then
    echo "❌ 缺少锚点 ${ANCHOR_DIR}/eval_formal.json（rank 16，3672 条 × 9 epoch）。"
    echo "   先跑: ${PYTHON} ${EXP_REL}/train_sentiment_lora.py --run-dir ${ANCHOR_DIR} --epochs 9"
    exit 1
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
    "${PYTHON}" "${EXP_REL}/train_sentiment_lora.py" --run-dir "${POINT_DIR}" --overwrite \
        --rank "${RANK}" --epochs "${EPOCHS}"
    "${PYTHON}" "${EXP_REL}/eval_sentiment.py" --split formal \
        --checkpoint "${POINT_DIR}/best_lora.pth"
done

echo ""
echo "--------------------------------------------------------------------------------"
echo "▶ 配对分析"
echo "--------------------------------------------------------------------------------"
"${PYTHON}" "${EXP_REL}/analyze_rank.py"

echo ""
echo "================================================================================"
echo " rank 扫描结束  $(date '+%Y-%m-%d %H:%M:%S')"
echo " 日志: ${LOG_FILE}"
echo " 报告: ${SCRIPT_DIR}/rank_sweep.md"
echo "================================================================================"
