#!/usr/bin/env bash
# ==============================================================================
# 科室分诊 LoRA 实验 —— 云端一次性全流程执行脚本
#
# 执行步骤：
#   Step 1: 预检 (Python / GPU / 基座权重 / 切分数据 md5 / 目录)
#   Step 2: 冒烟 (60 条训练 + LoRA 重载与 30 条评测，仅验证流程)
#   Step 3: 正式训练 (formal/train.jsonl, 12000 条 × 3 epoch)
#   Step 4: 正式评测 (formal/test.jsonl, 600 题，六类平衡)
#   Step 5: 规模曲线 (600 / 1200 / 2400 / 4800 / 9600 各训练并评测；12000 即 Step 3/4)
#   Step 6: 饱和分析 (saturation.md)
#
# 用法：
#   bash experiments/lora_triage_20260917/run_cloud.sh                # 全流程
#   bash experiments/lora_triage_20260917/run_cloud.sh --skip-smoke   # 跳过 Step 2
#   bash experiments/lora_triage_20260917/run_cloud.sh --skip-scale   # 跳过 Step 5
#   bash experiments/lora_triage_20260917/run_cloud.sh --step 4       # 只跑 Step 4
#   bash experiments/lora_triage_20260917/run_cloud.sh --from-step 3  # 从 Step 3 开始
#   bash experiments/lora_triage_20260917/run_cloud.sh --force        # 已有产物也重跑
#
# 已完成的步骤默认自动跳过（依据产物是否存在），可用 --force 覆盖。
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

EXP_REL="experiments/lora_triage_20260917"
RUNS_DIR="${SCRIPT_DIR}/runs"
DATA_DIR="${SCRIPT_DIR}/data"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${RUNS_DIR}"

TS="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/run_${TS}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
ln -sfn "run_${TS}.log" "${LOG_DIR}/latest.log"

if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "❌ 未找到可用的 Python 解释器"; exit 1
fi

TARGET_STEP=""
FROM_STEP=""
SKIP_SMOKE=false
SKIP_SCALE=false
FORCE=false
ALLOW_PLACEHOLDER=false
RAW_ARGS="$*"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step) TARGET_STEP="$2"; shift 2 ;;
        --from-step) FROM_STEP="$2"; shift 2 ;;
        --skip-smoke) SKIP_SMOKE=true; shift ;;
        --skip-scale) SKIP_SCALE=true; shift ;;
        --force) FORCE=true; shift ;;
        --allow-placeholder) ALLOW_PLACEHOLDER=true; shift ;;
        -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "❌ 未知参数: $1（--help 查看用法）"; exit 1 ;;
    esac
done

echo "================================================================================"
echo " MiniMind 医疗科室分诊 LoRA 云端全流程"
echo " 工作目录: ${ROOT_DIR}"
echo " Python  : ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo " 时间    : $(date '+%Y-%m-%d %H:%M:%S')"
echo " 参数    : ${RAW_ARGS:-（无）}"
echo " 日志    : ${LOG_FILE}"
echo "================================================================================"

# 依赖快照：与上一份相同就不再落盘，避免堆积一堆逐字节相同的 pip_freeze 文件。
FREEZE_TMP="$(mktemp)"
"${PYTHON}" -m pip freeze > "${FREEZE_TMP}" 2>/dev/null || true
LATEST_FREEZE="$(ls -1t "${LOG_DIR}"/pip_freeze_*.txt 2>/dev/null | head -1 || true)"
if [ -n "${LATEST_FREEZE}" ] && cmp -s "${FREEZE_TMP}" "${LATEST_FREEZE}"; then
    echo "📦 依赖快照与 $(basename "${LATEST_FREEZE}") 一致，不重复落盘"
    rm -f "${FREEZE_TMP}"
else
    mv "${FREEZE_TMP}" "${LOG_DIR}/pip_freeze_${TS}.txt"
    echo "📦 依赖快照已保存: pip_freeze_${TS}.txt"
fi

should_run() {
    local step="$1"
    if [ -n "${TARGET_STEP}" ]; then [ "${step}" = "${TARGET_STEP}" ] && return 0 || return 1; fi
    if [ -n "${FROM_STEP}" ]; then [ "${step}" -ge "${FROM_STEP}" ] && return 0 || return 1; fi
    return 0
}

banner() {
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "▶ Step $1: $2"
    echo "--------------------------------------------------------------------------------"
}

# 产物已存在则跳过（除非 --force）
done_already() {
    local marker="$1"
    if [ "${FORCE}" = true ]; then return 1; fi
    [ -f "${marker}" ]
}

# ----------------------------------- Step 1 -----------------------------------
if should_run 1; then
    banner 1 "预检"
    "${PYTHON}" - <<'PYEOF'
import json, sys, hashlib
from pathlib import Path
ROOT = Path.cwd()
EXP = ROOT / "experiments" / "lora_triage_20260917"
ok = True

import torch, transformers
print(f"  torch {torch.__version__} | transformers {transformers.__version__}")
if torch.cuda.is_available():
    print(f"  ✅ GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
else:
    print("  ⚠️  未检测到 GPU，正式训练会极慢（本地干跑可忽略此项）")

config = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
weight = ROOT / config["base_model"]["weight_path"]
if weight.exists():
    print(f"  ✅ 基座权重: {weight} ({weight.stat().st_size / 1e6:.0f} MB)")
else:
    print(f"  ❌ 基座权重缺失: {weight}"); ok = False

manifest = json.loads((EXP / "data" / "manifest.json").read_text(encoding="utf-8"))
entries = list(manifest["splits"].items()) + [(f"scale_{k}", v) for k, v in manifest["scale_curve"].items()]
bad = []
for name, info in entries:
    path = ROOT / info["path"]
    if not path.exists():
        bad.append(f"{name} 缺失"); continue
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if digest != info["md5"]:
        bad.append(f"{name} md5 不符")
if bad:
    print("  ❌ 切分数据校验失败: " + ", ".join(bad)); ok = False
else:
    print(f"  ✅ 切分数据校验通过（{len(entries)} 个文件 md5 一致）")

for name, info in manifest["baselines"]["per_split"].items():
    print(f"  · {name}: n={info['n']}，判定优于随机需准确率 > {info['min_accuracy_to_beat_random']}")

sys.exit(0 if ok else 1)
PYEOF
    # 占位权重保护：绝不让随机初始化的权重混进正式结果
    if [ -f "${ROOT_DIR}/out/PLACEHOLDER_README.txt" ] && [ "${ALLOW_PLACEHOLDER}" != true ]; then
        echo ""
        echo "❌ 检测到 out/PLACEHOLDER_README.txt —— 当前基座是随机初始化的占位权重。"
        echo "   用它跑出来的任何准确率都毫无意义。请换上真实的 full_sft_768.pth，"
        echo "   或在明知是流程干跑时显式追加 --allow-placeholder。"
        exit 1
    fi
    echo "✅ Step 1 预检通过"
fi

# ----------------------------------- Step 2 -----------------------------------
if should_run 2 && [ "${SKIP_SMOKE}" != true ]; then
    banner 2 "冒烟（60 条训练 + 30 条评测，仅验证流程）"
    SMOKE_DIR="${RUNS_DIR}/smoke"
    if done_already "${SMOKE_DIR}/eval_smoke.json"; then
        echo "⏭  已有冒烟产物，跳过（--force 可重跑）"
    else
        "${PYTHON}" "${EXP_REL}/train_triage_lora.py" --smoke --overwrite --run-dir "${SMOKE_DIR}"
        "${PYTHON}" "${EXP_REL}/eval_triage.py" --split smoke \
            --checkpoint "${SMOKE_DIR}/final_lora.pth"
        echo "✅ Step 2 完成（冒烟准确率不可解读，只看流程是否跑通）"
    fi
fi

# ----------------------------------- Step 3 -----------------------------------
if should_run 3; then
    banner 3 "正式训练（12000 条 × 3 epoch）"
    FORMAL_DIR="${RUNS_DIR}/formal"
    if done_already "${FORMAL_DIR}/train_summary.json"; then
        echo "⏭  已有训练产物，跳过（--force 可重跑）"
    else
        EXTRA=""
        [ "${FORCE}" = true ] && EXTRA="--overwrite"
        "${PYTHON}" "${EXP_REL}/train_triage_lora.py" --run-dir "${FORMAL_DIR}" ${EXTRA}
        echo "✅ Step 3 完成"
    fi
fi

# ----------------------------------- Step 4 -----------------------------------
if should_run 4; then
    banner 4 "正式评测（test 600 题，六类平衡）"
    FORMAL_DIR="${RUNS_DIR}/formal"
    if done_already "${FORMAL_DIR}/eval_formal.json"; then
        echo "⏭  已有评测产物，跳过（--force 可重跑）"
    else
        [ -f "${FORMAL_DIR}/best_lora.pth" ] || { echo "❌ 缺少 ${FORMAL_DIR}/best_lora.pth，请先跑 Step 3"; exit 1; }
        "${PYTHON}" "${EXP_REL}/eval_triage.py" --split formal \
            --checkpoint "${FORMAL_DIR}/best_lora.pth"
        echo "✅ Step 4 完成"
    fi
fi

# ----------------------------------- Step 5 -----------------------------------
if should_run 5 && [ "${SKIP_SCALE}" != true ]; then
    banner 5 "规模曲线（600 / 1200 / 2400 / 4800 / 9600；12000 点即 Step 3/4 的正式训练）"
    for SIZE in 600 1200 2400 4800 9600; do
        POINT_DIR="${RUNS_DIR}/scale_${SIZE}"
        if done_already "${POINT_DIR}/eval_formal.json"; then
            echo "⏭  scale_${SIZE} 已有产物，跳过"
            continue
        fi
        echo ""
        echo "· 规模点 ${SIZE} 条"
        "${PYTHON}" "${EXP_REL}/train_triage_lora.py" --run-dir "${POINT_DIR}" --overwrite \
            --train-file "${EXP_REL}/data/scale/train_${SIZE}.jsonl"
        "${PYTHON}" "${EXP_REL}/eval_triage.py" --split formal \
            --checkpoint "${POINT_DIR}/best_lora.pth"
    done
    echo "✅ Step 5 完成"
fi

# ----------------------------------- Step 6 -----------------------------------
if should_run 6; then
    banner 6 "饱和分析"
    "${PYTHON}" "${EXP_REL}/analyze_scale.py"
    echo "✅ Step 6 完成"
fi

echo ""
echo "================================================================================"
echo " 全流程结束  $(date '+%Y-%m-%d %H:%M:%S')"
echo " 日志: ${LOG_FILE}"
echo " 报告: ${SCRIPT_DIR}/saturation.md"
echo "================================================================================"
