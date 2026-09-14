#!/usr/bin/env bash
# ==============================================================================
# MiniMind 医学单选题 LoRA 实验云端全流程调度脚本 (升级版)
# 路径: experiments/lora_medical_mcq_pilot/run_cloud.sh
# ==============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 日志目录与文件
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"
ln -sf "${LOG_FILE}" "${LOG_DIR}/latest.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================================================"
echo "MiniMind 医学单选题 LoRA 实验 (Pilot v2) - 云端执行调度"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "工作根目录: ${ROOT_DIR}"
echo "实验目录:   ${SCRIPT_DIR}"
echo "日志文件:   ${LOG_FILE}"
echo "======================================================================"

# 默认调度参数
STEP="all"
DEVICE="cuda:0"
DTYPE="bfloat16"
OVERWRITE_FLAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --step)
      STEP="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE_FLAG="--overwrite"
      shift
      ;;
    --help|-h)
      echo "用法: $0 [选项]"
      echo "  --step <0|1|2|3|4|5|6|all>  指定执行阶段 (默认: all)"
      echo "      0: 单元测试 (test_mcq_pipeline.py)"
      echo "      1: 数据下载、清洗与去重隔离 (prepare_data.py)"
      echo "      2: 极小规模冒烟验证 (少量train/val训练与验证，不碰正式测试集)"
      echo "      3: 正式基座模型测试集评估 (eval_mcq.py --mode base --split test)"
      echo "      4: 正式独立 LoRA 训练与验证选优 (train_mcq_lora.py)"
      echo "      5: 正式 LoRA 模型测试集评估 (eval_mcq.py --mode lora --split test)"
      echo "      6: 严格前后量化对比与报告输出 (eval_mcq.py --mode compare --split test)"
      echo "  --device <device>           计算设备 (默认: cuda:0)"
      echo "  --dtype <dtype>             数值精度 (默认: bfloat16)"
      echo "  --overwrite                 显式允许覆盖已有产物"
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 确定 Python 解释器
if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
  PY="${ROOT_DIR}/.venv/bin/python"
else
  PY="python3"
fi

echo "使用 Python: $("${PY}" -c 'import sys; print(sys.executable)')"

# 阶段 0: 单元测试 (Issue 7)
run_step_0() {
  echo ""
  echo ">>> [阶段 0/6] 运行单元测试集 (test_mcq_pipeline.py) <<<"
  "${PY}" -m unittest "${SCRIPT_DIR}/test_mcq_pipeline.py"
  echo "✅ 阶段 0 单元测试全部通过！"
}

# 阶段 1: 数据准备与去重划分 (Issue 1, 2, 4)
run_step_1() {
  echo ""
  echo ">>> [阶段 1/6] 准备正式与冒烟题集 (prepare_data.py) <<<"
  "${PY}" "${SCRIPT_DIR}/prepare_data.py" --download ${OVERWRITE_FLAG}
  echo "✅ 阶段 1 数据准备完成！"
}

# 阶段 2: 冒烟验证 (Issue 2: 仅用少量 train/val，绝不测正式测试集)
run_step_2() {
  echo ""
  echo ">>> [阶段 2/6] 少量样本冒烟验证 (训练与验证通路，不碰正式测试集) <<<"
  echo ">>> [2.1] 冒烟 LoRA 训练 (1 epoch, runs/smoke) ..."
  "${PY}" "${SCRIPT_DIR}/train_mcq_lora.py" \
    --smoke \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    ${OVERWRITE_FLAG}

  echo ">>> [2.2] 冒烟基座评测 (在验证集 val 上) ..."
  "${PY}" "${SCRIPT_DIR}/eval_mcq.py" \
    --smoke \
    --mode base \
    --split val \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    ${OVERWRITE_FLAG}

  echo ">>> [2.3] 冒烟 LoRA 评测 (在验证集 val 上) ..."
  "${PY}" "${SCRIPT_DIR}/eval_mcq.py" \
    --smoke \
    --mode lora \
    --split val \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    ${OVERWRITE_FLAG}

  echo ">>> [2.4] 冒烟对比核验 ..."
  "${PY}" "${SCRIPT_DIR}/eval_mcq.py" \
    --smoke \
    --mode compare \
    --split val \
    ${OVERWRITE_FLAG}
  echo "✅ 阶段 2 冒烟全流程通过！通路验证正常，进入正式全量阶段。"
}

# 阶段 3: 正式基座测试集评测
run_step_3() {
  echo ""
  echo ">>> [阶段 3/6] 正式基座模型在测试集 (200题) 评测 <<<"
  "${PY}" "${SCRIPT_DIR}/eval_mcq.py" \
    --mode base \
    --split test \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    ${OVERWRITE_FLAG}
  echo "✅ 阶段 3 正式基座评测完成！"
}

# 阶段 4: 正式全量训练 (1000题训练, 200题验证选优)
run_step_4() {
  echo ""
  echo ">>> [阶段 4/6] 正式独立 LoRA 训练与验证集选优 (runs/formal) <<<"
  "${PY}" "${SCRIPT_DIR}/train_mcq_lora.py" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    ${OVERWRITE_FLAG}
  echo "✅ 阶段 4 正式 LoRA 训练与选优完成！"
}

# 阶段 5: 正式 LoRA 测试集评测
run_step_5() {
  echo ""
  echo ">>> [阶段 5/6] 正式 LoRA 模型在测试集 (200题) 评测 <<<"
  "${PY}" "${SCRIPT_DIR}/eval_mcq.py" \
    --mode lora \
    --split test \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    ${OVERWRITE_FLAG}
  echo "✅ 阶段 5 正式 LoRA 评测完成！"
}

# 阶段 6: 严格前后量化对比与报告输出
run_step_6() {
  echo ""
  echo ">>> [阶段 6/6] Base vs LoRA 严格前后对比评估 (runs/formal) <<<"
  "${PY}" "${SCRIPT_DIR}/eval_mcq.py" \
    --mode compare \
    --split test \
    ${OVERWRITE_FLAG}
  echo "✅ 阶段 6 正式对比完成！"
}

# 调度分发
case "${STEP}" in
  0)
    run_step_0
    ;;
  1)
    run_step_1
    ;;
  2)
    run_step_2
    ;;
  3)
    run_step_3
    ;;
  4)
    run_step_4
    ;;
  5)
    run_step_5
    ;;
  6)
    run_step_6
    ;;
  all)
    run_step_0
    run_step_1
    run_step_2
    run_step_3
    run_step_4
    run_step_5
    run_step_6
    ;;
  *)
    echo "无效 step: ${STEP} (可选: 0, 1, 2, 3, 4, 5, 6, all)"
    exit 1
    ;;
esac

echo ""
echo "======================================================================"
echo "🎉 云端全流程成功执行完成！"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "对比产物: ${SCRIPT_DIR}/runs/formal/eval_results/comparison_test.json"
echo "======================================================================"
