#!/usr/bin/env bash
# ==============================================================================
# MiniMind LoRA 医学微调云端一次性全流程执行脚本
#
# 执行步骤：
#   Step 1: 预检环境 (GPU / 依赖 / 权重 / 数据 / 目录)
#   Step 2: 基座评估冒烟 (base, --limit 2, --val_loss)
#   Step 3: 训练验证冒烟 (4 optimizer step 极简训练 + LoRA 保存重载与 --limit 2 评估)
#   Step 4: 正式全量 Base 评估 (140题 + val_loss)
#   Step 5: 正式全量 LoRA 训练 (train_5000.jsonl, 1 epoch = 625 steps, 全新初始化)
#   Step 6: 正式全量 LoRA 评估 (140题 + val_loss)
#   Step 7: 生成前后对比分析报告 (Markdown)
#
# 用法：
#   bash experiments/lora_medical_20260909/run_cloud.sh                # 执行全部流程
#   bash experiments/lora_medical_20260909/run_cloud.sh --skip-smoke   # 跳过步骤 2-3 直接全量
#   bash experiments/lora_medical_20260909/run_cloud.sh --step 4       # 只执行步骤 4
#   bash experiments/lora_medical_20260909/run_cloud.sh --from-step 4  # 从步骤 4 开始执行
#   bash experiments/lora_medical_20260909/run_cloud.sh --force        # 不跳过已存在产物，强制重新跑
# ==============================================================================

set -euo pipefail

# 定位项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

# 日志落盘：本次运行的全部终端输出(含 loss/val_loss 与报错堆栈)同步写入文件，终端仍实时显示
TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="experiments/lora_medical_20260909/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_${TS}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
ln -sfn "run_${TS}.log" "${LOG_DIR}/latest.log"

# 寻找合适的 Python 解释器
if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "❌ 错误: 未找到可用的 Python 解释器！"
    exit 1
fi

echo "================================================================================"
echo " MiniMind 医学 LoRA 全流程云端运行系统"
echo " 工作目录: ${ROOT_DIR}"
echo " Python  : ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo " 时间    : $(date '+%Y-%m-%d %H:%M:%S')"
echo " 日志    : ${LOG_FILE}"
echo "================================================================================"

# 参数解析
TARGET_STEP=""
FROM_STEP=""
SKIP_SMOKE=false
FORCE=false
RAW_ARGS="$*"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)
            TARGET_STEP="$2"
            shift 2
            ;;
        --from-step)
            FROM_STEP="$2"
            shift 2
            ;;
        --skip-smoke)
            SKIP_SMOKE=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "用法: bash $0 [--step N] [--from-step N] [--skip-smoke] [--force]"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ==============================================================================
# 记录运行参数与实测环境版本（随终端输出一并写入日志文件）
# ==============================================================================
echo "--------------------------------------------------------------------------------"
echo " 运行命令: bash $0 ${RAW_ARGS}"
echo " 主机时间: $(hostname 2>/dev/null || echo unknown-host) | ${TS}"
if command -v git &>/dev/null && git -C "${ROOT_DIR}" rev-parse --short HEAD &>/dev/null; then
    echo " Git 提交: $(git -C "${ROOT_DIR}" rev-parse --short HEAD)"
fi
if command -v nvidia-smi &>/dev/null; then
    echo " GPU 信息: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
fi
${PYTHON} - <<'PY'
import sys
import importlib.metadata as md
print(f" Python  : {sys.version.split()[0]}")
for pkg in ["torch", "transformers", "datasets", "accelerate", "numpy", "tokenizers"]:
    try:
        print(f"   {pkg:<14s}== {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"   {pkg:<14s}: 未安装")
PY
${PYTHON} -m pip freeze > "${LOG_DIR}/pip_freeze_${TS}.txt" 2>/dev/null || true
echo " 依赖快照: ${LOG_DIR}/pip_freeze_${TS}.txt (完整 pip freeze)"
echo "--------------------------------------------------------------------------------"

should_run_step() {
    local step_num="$1"
    if [ -n "${TARGET_STEP}" ]; then
        [ "${TARGET_STEP}" -eq "${step_num}" ] && return 0 || return 1
    fi
    if [ -n "${FROM_STEP}" ]; then
        [ "${step_num}" -ge "${FROM_STEP}" ] && return 0 || return 1
    fi
    if [ "${SKIP_SMOKE}" = true ] && { [ "${step_num}" -eq 2 ] || [ "${step_num}" -eq 3 ]; }; then
        return 1
    fi
    return 0
}

# 路径常量定义
DATA_DIR="experiments/lora_medical_20260909/data/v2"
EVAL_DIR="experiments/lora_medical_20260909/eval/v1"
RES_DIR="experiments/lora_medical_20260909/eval_results"

BASE_WEIGHT="out/full_sft_768.pth"
SMOKE_LORA_WEIGHT="out/lora_medical_smoke_768.pth"
FORMAL_LORA_WEIGHT="out/lora_medical_formal_768.pth"
FORMAL_DONE="out/lora_medical_formal_768.complete.sha256"

TRAIN_5000="${DATA_DIR}/train_5000.jsonl"
VAL_JSONL="${DATA_DIR}/val.jsonl"
SMOKE_JSONL="${DATA_DIR}/smoke.jsonl"

mkdir -p out checkpoints "${RES_DIR}/smoke_base" "${RES_DIR}/smoke_lora" "${RES_DIR}/base" "${RES_DIR}/lora"

# Model steps must never silently fall back to the Mac/CPU.
DTYPE=""
for model_step in 1 2 3 4 5 6; do
    if should_run_step "${model_step}"; then
        DTYPE=$(${PYTHON} -c 'import torch; import sys; sys.exit("CUDA 不可用，请在云端 GPU 执行") if not torch.cuda.is_available() else None; print("bfloat16" if torch.cuda.is_bf16_supported() else "float16")')
        break
    fi
done

results_current() {
    local mode="$1"
    ${PYTHON} run_eval.py --check-results --mode "${mode}" \
        --out_dir "${RES_DIR}/${mode}" --eval_dir "${EVAL_DIR}" \
        --base_weight "${BASE_WEIGHT}" --lora_weight "${FORMAL_LORA_WEIGHT}" \
        --val_loss --val_path "${VAL_JSONL}" --device cuda:0 --dtype "${DTYPE}"
}

training_complete() {
    [ -f "${FORMAL_DONE}" ] && sha256sum --check --status "${FORMAL_DONE}"
}

# ==============================================================================
# Step 1: 环境与资产预检
# ==============================================================================
if should_run_step 1; then
    echo -e "\n--------------------------------------------------------------------------------"
    echo " [Step 1/7] 预检 GPU、环境依赖、模型权重与数据..."
    echo "--------------------------------------------------------------------------------"

    # 1. 检查 GPU
    if command -v nvidia-smi &>/dev/null; then
        echo "✅ 发现 NVIDIA GPU 设备:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        echo "⚠️ 警告: 未检测到 nvidia-smi 命令，尝试检测 PyTorch CUDA 状态..."
    fi

    ${PYTHON} -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
cuda_avail = torch.cuda.is_available()
print(f'CUDA 可用性: {cuda_avail}')
if cuda_avail:
    print(f'GPU 设备数量: {torch.cuda.device_count()}, 当前设备: {torch.cuda.get_device_name(0)}')
    print(f'bf16 硬件支持: {torch.cuda.is_bf16_supported()}')
else:
    print('⚠️ 警告: PyTorch 当前未检测到 CUDA，若在 CPU 上运行会极度缓慢！')
"

    # 2. 检查基座权重
    if [ ! -f "${BASE_WEIGHT}" ]; then
        echo "❌ 错误: 基座权重文件不存在: ${BASE_WEIGHT}"
        echo "请将 full_sft_768.pth 上传或下载至 out/full_sft_768.pth"
        exit 1
    else
        size_bytes=$(stat -c%s "${BASE_WEIGHT}" 2>/dev/null || stat -f%z "${BASE_WEIGHT}" 2>/dev/null || echo "0")
        echo "✅ 基座权重存在: ${BASE_WEIGHT} (文件大小: $(( size_bytes / 1024 / 1024 )) MB)"
    fi

    # 3. 检查数据与锁定题集
    for file in "${TRAIN_5000}" "${VAL_JSONL}" "${SMOKE_JSONL}" \
                "${EVAL_DIR}/manifest.json" "${EVAL_DIR}/questions_mcq.jsonl" \
                "${EVAL_DIR}/questions_medqa.jsonl" "${EVAL_DIR}/questions_general.jsonl"; do
        if [ ! -f "${file}" ]; then
            echo "❌ 错误: 缺少必要文件: ${file}"
            exit 1
        fi
    done
    echo "✅ 训练数据与锁定考卷完整检验通过。"
    echo "✅ 预检完成，环境满足要求！"
fi

# ==============================================================================
# Step 2: 基座评估冒烟 (base, --limit 2, --val_loss)
# ==============================================================================
if should_run_step 2; then
    echo -e "\n--------------------------------------------------------------------------------"
    echo " [Step 2/7] 基座模型评估冒烟测试 (--limit 2, 验证推理与 val_loss 计算)..."
    echo "--------------------------------------------------------------------------------"
    if [ "${FORCE}" = false ] && [ -f "${RES_DIR}/smoke_base/summary.json" ]; then
        echo "⏭️ 基座冒烟结果已存在 (${RES_DIR}/smoke_base/summary.json)，跳过执行。使用 --force 可重新运行。"
    else
        ${PYTHON} run_eval.py \
            --mode base --device cuda:0 --dtype "${DTYPE}" \
            --base_weight "${BASE_WEIGHT}" \
            --limit 2 \
            --val_loss \
            --val_path "${VAL_JSONL}" \
            --out_dir "${RES_DIR}/smoke_base"
        echo "✅ Step 2 基座评估冒烟通过！"
    fi
fi

# ==============================================================================
# Step 3: LoRA 训练冒烟 (4 optimizer steps) 与 LoRA 重载评估冒烟
# ==============================================================================
if should_run_step 3; then
    echo -e "\n--------------------------------------------------------------------------------"
    echo " [Step 3/7] LoRA 训练冒烟测试 (跑 4 个优化步，验证周期验证、权重落盘与加载评估)..."
    echo "--------------------------------------------------------------------------------"
    echo ">> 3.1 运行极简训练 (32条 smoke 数据, batch_size=8 -> 4 steps, 每2步验证一次)..."
    ${PYTHON} trainer/train_lora.py --device cuda:0 --dtype "${DTYPE}" \
        --data_path "${SMOKE_JSONL}" \
        --val_path "${VAL_JSONL}" \
        --val_interval 2 \
        --save_interval 4 \
        --max_seq_len 512 \
        --epochs 1 \
        --batch_size 8 \
        --accumulation_steps 1 \
        --lora_name lora_medical_smoke \
        --from_weight full_sft \
        --from_resume 0 \
        --save_dir out

    if [ ! -f "${SMOKE_LORA_WEIGHT}" ]; then
        echo "❌ 错误: 冒烟 LoRA 权重未生成: ${SMOKE_LORA_WEIGHT}"
        exit 1
    fi
    echo "✅ 冒烟 LoRA 权重已成功落盘: ${SMOKE_LORA_WEIGHT}"

    echo ">> 3.2 运行冒烟 LoRA 模型评估 (--limit 2)..."
    ${PYTHON} run_eval.py \
        --mode lora --device cuda:0 --dtype "${DTYPE}" \
        --base_weight "${BASE_WEIGHT}" \
        --lora_weight "${SMOKE_LORA_WEIGHT}" \
        --limit 2 \
        --val_loss \
        --val_path "${VAL_JSONL}" \
        --out_dir "${RES_DIR}/smoke_lora"
    echo "✅ Step 3 训练与评估冒烟全链路验证通过！"
fi

# ==============================================================================
# Step 4: 正式全量 Base 评估 (140题 + val_loss)
# ==============================================================================
if should_run_step 4; then
    echo -e "\n--------------------------------------------------------------------------------"
    echo " [Step 4/7] 正式全量 Base 评估 (100 MCQ + 30 MedQA + 10 General + Val Loss)..."
    echo "--------------------------------------------------------------------------------"
    if [ "${FORCE}" = false ] && results_current base; then
        echo "⏭️ 正式 Base 评估结果已存在且完整 (${RES_DIR}/base/summary.json)，跳过执行。使用 --force 可重新评估。"
    else
        ${PYTHON} run_eval.py \
            --mode base --device cuda:0 --dtype "${DTYPE}" \
            --base_weight "${BASE_WEIGHT}" \
            --val_loss \
            --val_path "${VAL_JSONL}" \
            --out_dir "${RES_DIR}/base"
        echo "✅ Step 4 正式全量 Base 评估完成！"
    fi
fi

# ==============================================================================
# Step 5: 正式全量 LoRA 训练 (5000条数据, 1 epoch, 全新初始化)
# ==============================================================================
if should_run_step 5; then
    echo -e "\n--------------------------------------------------------------------------------"
    echo " [Step 5/7] 正式全量 LoRA 训练 (5000 条医学问答, 1 epoch = 625 steps)..."
    echo "--------------------------------------------------------------------------------"
    if [ "${FORCE}" = false ] && training_complete; then
        echo "⏭️ 正式 LoRA 完成标记及文件校验通过，跳过训练。使用 --force 可强制重新训练。"
    else
        rm -f "${FORMAL_DONE}"
        echo "开始正式 LoRA 训练 (独立从 full_sft 与全新 LoRA 开始，不继承冒烟状态)..."
        ${PYTHON} trainer/train_lora.py --device cuda:0 --dtype "${DTYPE}" \
            --data_path "${TRAIN_5000}" \
            --val_path "${VAL_JSONL}" \
            --val_interval 50 \
            --save_interval 125 \
            --max_seq_len 512 \
            --epochs 1 \
            --batch_size 8 \
            --accumulation_steps 1 \
            --learning_rate 1e-4 \
            --lora_name lora_medical_formal \
            --from_weight full_sft \
            --from_resume 0 \
            --save_dir out

        if [ ! -f "${FORMAL_LORA_WEIGHT}" ]; then
            echo "❌ 错误: 正式 LoRA 权重文件未找到: ${FORMAL_LORA_WEIGHT}"
            exit 1
        fi
        # Write only after a successful full training process. Intermediate weights are not completion.
        sha256sum "${FORMAL_LORA_WEIGHT}" "${BASE_WEIGHT}" "${TRAIN_5000}" "${VAL_JSONL}" \
            trainer/train_lora.py trainer/trainer_utils.py model/model_lora.py model/model_minimind.py \
            "${SCRIPT_DIR}/run_cloud.sh" > "${FORMAL_DONE}.tmp"
        mv "${FORMAL_DONE}.tmp" "${FORMAL_DONE}"
        echo "✅ Step 5 正式全量 LoRA 训练顺利完成！权重已保存至: ${FORMAL_LORA_WEIGHT}"
    fi
fi

# ==============================================================================
# Step 6: 正式全量 LoRA 评估 (140题 + val_loss)
# ==============================================================================
if should_run_step 6; then
    training_complete || { echo "正式训练未完成或文件已变更，请先执行 Step 5"; exit 1; }
    echo -e "\n--------------------------------------------------------------------------------"
    echo " [Step 6/7] 正式全量 LoRA 评估 (100 MCQ + 30 MedQA + 10 General + Val Loss)..."
    echo "--------------------------------------------------------------------------------"
    if [ "${FORCE}" = false ] && results_current lora; then
        echo "⏭️ 正式 LoRA 评估结果已存在且完整 (${RES_DIR}/lora/summary.json)，跳过执行。使用 --force 可重新评估。"
    else
        ${PYTHON} run_eval.py \
            --mode lora --device cuda:0 --dtype "${DTYPE}" \
            --base_weight "${BASE_WEIGHT}" \
            --lora_weight "${FORMAL_LORA_WEIGHT}" \
            --val_loss \
            --val_path "${VAL_JSONL}" \
            --out_dir "${RES_DIR}/lora"
        echo "✅ Step 6 正式全量 LoRA 评估完成！"
    fi
fi

# ==============================================================================
# Step 7: 生成前后对比分析报告
# ==============================================================================
if should_run_step 7; then
    training_complete || { echo "正式训练完成标记缺失或已失效，请先执行 Step 5"; exit 1; }
    echo -e "\n--------------------------------------------------------------------------------"
    echo " [Step 7/7] 生成 Base 与 LoRA 前后对比分析报告 (Markdown)..."
    echo "--------------------------------------------------------------------------------"
    REPORT_FILE="${RES_DIR}/compare_report.md"
    ${PYTHON} run_eval.py \
        --compare \
        --base_dir "${RES_DIR}/base" \
        --lora_dir "${RES_DIR}/lora" \
        --save_report "${REPORT_FILE}"
    echo "✅ Step 7 前后对比报告生成完毕！路径: ${REPORT_FILE}"
fi

echo -e "\n================================================================================"
echo " 🎉 全部流程已顺利执行完成！"
echo " 产物清单："
echo "   - 基座评估结果: ${RES_DIR}/base/"
echo "   - 微调权重文件: ${FORMAL_LORA_WEIGHT}"
echo "   - 微调评估结果: ${RES_DIR}/lora/"
echo "   - 前后对比报告: ${RES_DIR}/compare_report.md"
echo "   - 运行日志与环境快照: ${LOG_DIR}/ (本次日志: ${LOG_FILE})"
echo ""
echo " 云端打包指令 (含日志与完成标记):"
echo "   tar -czvf results_bundle.tar.gz ${RES_DIR}/ ${LOG_DIR}/ ${FORMAL_LORA_WEIGHT} ${FORMAL_DONE}"
echo "   # 如需保留断点续训能力 (--from_resume 1)，打包时追加: checkpoints/"
echo " 本地下载指令: scp -P <端口> <用户名>@<云端IP>:<云端工作路径>/results_bundle.tar.gz ./"
echo "================================================================================"
