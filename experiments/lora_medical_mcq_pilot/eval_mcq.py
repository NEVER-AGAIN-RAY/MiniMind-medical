#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
医学单选题评测与对比脚本 (eval_mcq.py) - 升级版

修复项：
1. [Issue 1 修复]：正确传递并解析 max_new_tokens 与 device，消除参数为 None 导致的异常。
2. [Issue 5 修复]：完善正式对比评测的完整性核验。
   - 禁止在正式对比中使用截断题目 (--limit / is_limited)；
   - 必须全量覆盖锁定测试集 (test.jsonl) 的全部题目，逐题核对 ID、题干、选项及标准答案；
   - 严格核验题集文件 MD5 及 manifest.json 记录；
   - 严格核对 LoRA 权重来源与 train_summary.json 中的选定检查点 MD5，核对基座模型 MD5。
3. 严格禁止冒烟与正式结果混用；已有产物默认拒绝覆盖。
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 sys.path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from experiments.lora_medical_mcq_pilot.mcq_dataset import (
    ANSWER_RATIONALE_INSTRUCTION_PROMPT,
    ANSWER_TEXT_INSTRUCTION_PROMPT,
    format_mcq_prompt,
)
from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

CONFIG_FILE = HERE / "config.json"


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"配置文件缺失: {CONFIG_FILE}")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def md5_file(filepath: Path | str) -> str:
    p = Path(filepath)
    if not p.is_file():
        return ""
    hasher = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_model_and_tokenizer(effective_cfg, mode="base"):
    model_dir = Path(effective_cfg["base_model"]["model_dir"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    base_cfg = effective_cfg["base_model"]
    lm_config = MiniMindConfig(
        hidden_size=base_cfg["hidden_size"],
        num_hidden_layers=base_cfg["num_hidden_layers"],
        use_moe=bool(base_cfg["use_moe"]),
    )
    model = MiniMindForCausalLM(lm_config)

    base_path = Path(effective_cfg["base_model"]["weight_path"])
    if not base_path.exists():
        raise FileNotFoundError(f"基座权重不存在: {base_path}")

    base_md5 = md5_file(base_path)
    device = effective_cfg["runtime"]["device"]
    print(f"[{mode.upper()}] 加载基座权重: {base_path} (MD5: {base_md5[:10]}...)")
    base_state = torch.load(base_path, map_location=device)
    model.load_state_dict(base_state, strict=True)

    lora_path = None
    lora_md5 = None
    if mode == "lora":
        lora_path = Path(effective_cfg["lora_eval"]["lora_weight_path"])
        if not lora_path.exists():
            raise FileNotFoundError(f"待评测的 LoRA 权重不存在: {lora_path}")
        lora_md5 = md5_file(lora_path)
        print(f"[{mode.upper()}] 加载并挂载 LoRA 权重: {lora_path} (MD5: {lora_md5[:10]}...)")
        apply_lora(model, rank=effective_cfg["lora_config"]["rank"])
        load_lora(model, str(lora_path))

    dtype_str = effective_cfg["runtime"]["dtype"]
    if dtype_str == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        target_dtype = torch.bfloat16
    elif dtype_str == "float16":
        target_dtype = torch.float16
    else:
        target_dtype = torch.float32

    model = model.to(dtype=target_dtype, device=device)
    model.eval()

    return model, tokenizer, base_path, base_md5, lora_path, lora_md5


@torch.no_grad()
def evaluate_single_sample(
    model,
    tokenizer,
    item: dict,
    max_new_tokens: int,
    device: str,
    instruction: str,
    target_mode: str = "letter",
) -> dict:
    """单样本确定性评估，显式传递 max_new_tokens 与 device 避免 NoneType 异常"""
    assert isinstance(max_new_tokens, int) and max_new_tokens > 0, (
        f"max_new_tokens 必须为正整数，实际收到: {max_new_tokens}"
    )

    question = item["question"]
    options = item["options"]
    official_answer = item["answer"].strip().upper()
    valid_keys = set(options.keys())

    prompt_text = format_mcq_prompt(question, options, instruction)
    conversation = [{"role": "user", "content": prompt_text}]
    prompt_str = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )

    inputs = tokenizer(prompt_str, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device) if "attention_mask" in inputs else None

    prompt_len = input_ids.shape[1]

    # 上下文长度检查：绝不截断输入
    max_pos = getattr(model.config, "max_position_embeddings", 32768)
    if prompt_len + max_new_tokens > max_pos:
        raise ValueError(
            f"题目 ID: {item.get('id')} 输入长度 ({prompt_len}) 加最大输出长度 ({max_new_tokens}) "
            f"超出模型最大位置限制 ({max_pos})！系统禁止截断题干。"
        )

    generated_ids = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # 严格确定性贪心生成
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_token_ids = generated_ids[0][prompt_len:].tolist()
    num_new_tokens = len(new_token_ids)

    hit_eos = (tokenizer.eos_token_id in new_token_ids)
    if hit_eos or num_new_tokens < max_new_tokens:
        natural_termination = True
        truncated = False
    else:
        natural_termination = False
        truncated = True

    raw_output = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    stripped = raw_output.strip()

    expected_answer_text = options[official_answer]
    answer_text_present = False
    if target_mode in {"answer_text", "answer_rationale"}:
        match = re.fullmatch(
            r"正确答案\s*[：:]\s*(.+?)\s*[（(]([A-E])[）)]\s*[。.]*"
            r"(?:\s*解析\s*[：:]\s*.*)?",
            stripped,
            flags=re.S,
        )
        is_format_compliant = bool(match and match.group(2) in valid_keys)
        predicted_letter = match.group(2) if is_format_compliant else None
        predicted_answer_text = match.group(1).strip() if is_format_compliant else None
        normalize = lambda s: re.sub(r"[\s，。,.、；;：:]", "", s)
        answer_text_present = bool(
            predicted_answer_text
            and normalize(predicted_answer_text) == normalize(expected_answer_text)
        )
    elif stripped in valid_keys and len(stripped) == 1:
        is_format_compliant = True
        predicted_letter = stripped
        predicted_answer_text = None
    else:
        is_format_compliant = False
        predicted_letter = None
        predicted_answer_text = None

    is_correct = bool(is_format_compliant and predicted_letter == official_answer)

    return {
        "id": item["id"],
        "question": question,
        "options": options,
        "official_answer": official_answer,
        "raw_model_output": raw_output,
        "generated_tokens_count": num_new_tokens,
        "natural_termination": natural_termination,
        "truncated": truncated,
        "is_format_compliant": is_format_compliant,
        "predicted_letter": predicted_letter,
        "predicted_answer_text": predicted_answer_text,
        "expected_answer_text": expected_answer_text,
        "answer_text_present": answer_text_present,
        "knowledge_consistent": bool(is_correct and answer_text_present),
        "is_correct": is_correct,
    }


def run_evaluation(args, mode="base"):
    config = load_config()
    is_smoke = bool(args.smoke)

    # 1. 冒烟模式保护：冒烟严禁测试正式测试集
    if is_smoke and args.split == "test":
        raise ValueError(
            "【非法操作拦截】冒烟模式 (--smoke) 严禁在正式测试集 (--split test) 上运行！\n"
            "冒烟测试仅允许使用少量训练/验证样本 (例如 --split val) 排查环境与代码通路。"
        )

    # 2. 运行目录确定
    run_paths = config["run_paths"]
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
    else:
        run_dir_rel = run_paths["smoke_dir"] if is_smoke else run_paths["formal_dir"]
        run_dir = ROOT / run_dir_rel

    eval_out_dir = run_dir / "eval_results"
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    result_file = eval_out_dir / f"{mode}_{args.split}_results.json"
    if result_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"评测产物已存在: {result_file}。默认拒绝覆盖！"
            f"如需重新评估，请指定新 --run_dir 或追加 --overwrite 参数。"
        )

    # 3. 确定数据文件路径
    exp_root = ROOT / run_paths["experiments_root"]
    data_subdir = "smoke" if is_smoke else "formal"
    split_file = exp_root / "data" / data_subdir / f"{args.split}.jsonl"

    if not split_file.exists():
        raise FileNotFoundError(f"题集文件不存在: {split_file}")

    dataset_md5 = md5_file(split_file)

    # 4. 解析待测 LoRA 权重
    lora_weight_path = None
    if mode == "lora":
        if args.lora_weight:
            lora_weight_path = Path(args.lora_weight)
            if not lora_weight_path.is_absolute():
                lora_weight_path = ROOT / lora_weight_path
        else:
            summary_p = run_dir / "checkpoints" / "train_summary.json"
            if summary_p.exists():
                with open(summary_p, "r", encoding="utf-8") as f:
                    sm = json.load(f)
                lora_weight_path = Path(sm["selected_checkpoint"]["path"])
            else:
                lora_weight_path = run_dir / "checkpoints" / "best_lora.pth"

        if not lora_weight_path.exists():
            raise FileNotFoundError(f"未找到选定的 LoRA 检查点文件: {lora_weight_path}")

    # 构建有效配置
    base_cfg = config["base_model"]
    base_weight_p = Path(args.base_weight or (ROOT / base_cfg["weight_path"]))
    if not base_weight_p.exists():
        raise FileNotFoundError(f"基座权重不存在: {base_weight_p}")

    target_mode = getattr(args, "target_mode", None)
    if target_mode is None:
        summary_p = run_dir / "checkpoints" / "train_summary.json"
        if summary_p.exists():
            with open(summary_p, "r", encoding="utf-8") as f:
                target_mode = json.load(f).get("effective_config", {}).get("data", {}).get("target_mode", "letter")
        else:
            target_mode = "letter"
    if target_mode == "answer_rationale":
        instruction = ANSWER_RATIONALE_INSTRUCTION_PROMPT
    elif target_mode == "answer_text":
        instruction = ANSWER_TEXT_INSTRUCTION_PROMPT
    else:
        instruction = config["data_config"]["instruction_prompt"]
    # [Issue 1 修复] 确保 max_new_tokens 必定为正整数，并明确同步更新 args
    max_new_tokens = int(args.max_new_tokens or config["evaluation_config"]["max_new_tokens"])
    args.max_new_tokens = max_new_tokens

    effective_cfg = {
        "is_smoke": is_smoke,
        "split": args.split,
        "run_dir": str(run_dir),
        "base_model": {
            "weight_path": str(base_weight_p),
            "hidden_size": base_cfg["hidden_size"],
            "num_hidden_layers": base_cfg["num_hidden_layers"],
            "use_moe": base_cfg["use_moe"],
            "model_dir": str(ROOT / base_cfg["model_dir"]),
        },
        "lora_config": {
            "rank": config["lora_config"]["rank"],
        },
        "lora_eval": {
            "lora_weight_path": str(lora_weight_path) if lora_weight_path else None,
        },
        "generation": {
            "do_sample": False,
            "open_thinking": False,
            "max_new_tokens": max_new_tokens,
            "instruction": instruction,
            "target_mode": target_mode,
        },
        "runtime": {
            "device": args.device,
            "dtype": args.dtype or config["training_hyperparameters"]["dtype"],
        },
    }

    # 5. 加载模型与题集
    model, tokenizer, base_path, base_md5, lora_path, lora_md5 = load_model_and_tokenizer(
        effective_cfg, mode=mode
    )

    items = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    is_limited = bool(args.limit and args.limit > 0)
    if is_limited:
        items = items[:args.limit]
        print(f"⚠️ [截断限制] 仅评测前 {args.limit} 题 (is_limited=True)")

    print(f"\n开始在 [{args.split}] 集上执行 [{mode.upper()}] 评测 (共 {len(items)} 题, max_new_tokens={max_new_tokens}) ...")
    details = []
    letter_counts = Counter()

    for idx, item in enumerate(items, 1):
        res = evaluate_single_sample(
            model=model,
            tokenizer=tokenizer,
            item=item,
            max_new_tokens=max_new_tokens,
            device=args.device,
            instruction=instruction,
            target_mode=target_mode,
        )
        details.append(res)

        if res["is_format_compliant"]:
            letter_counts[res["predicted_letter"]] += 1
        else:
            letter_counts["NON_COMPLIANT"] += 1

        if idx % 25 == 0 or idx == len(items):
            corr = sum(1 for d in details if d["is_correct"])
            comp = sum(1 for d in details if d["is_format_compliant"])
            print(f"进度: {idx}/{len(items)} | 正确: {corr} | 合规: {comp}")

    total = len(details)
    compliant_count = sum(1 for d in details if d["is_format_compliant"])
    correct_count = sum(1 for d in details if d["is_correct"])
    nat_term_count = sum(1 for d in details if d["natural_termination"])
    truncated_count = sum(1 for d in details if d["truncated"])
    answer_text_count = sum(1 for d in details if d["answer_text_present"])
    knowledge_consistent_count = sum(1 for d in details if d["knowledge_consistent"])

    metrics = {
        "total_questions": total,
        "single_letter_format_compliance_rate": round(compliant_count / total, 6) if total else 0.0,
        "accuracy": round(correct_count / total, 6) if total else 0.0,
        "natural_termination_rate": round(nat_term_count / total, 6) if total else 0.0,
        "truncation_rate": round(truncated_count / total, 6) if total else 0.0,
        "letter_distribution": dict(letter_counts),
        "answer_text_match_rate": round(answer_text_count / total, 6) if total else 0.0,
        "knowledge_consistency_rate": round(knowledge_consistent_count / total, 6) if total else 0.0,
        "counts": {
            "format_compliant": compliant_count,
            "correct": correct_count,
            "natural_termination": nat_term_count,
            "truncated": truncated_count,
            "answer_text_match": answer_text_count,
            "knowledge_consistent": knowledge_consistent_count,
        },
    }

    result_payload = {
        "experiment": "lora_medical_mcq_pilot",
        "mode": mode,
        "is_smoke": is_smoke,
        "is_limited": is_limited,
        "limit_value": args.limit if is_limited else None,
        "split": args.split,
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "path": str(split_file),
            "md5": dataset_md5,
            "count": total,
        },
        "weights": {
            "base_weight_path": str(base_path),
            "base_weight_md5": base_md5,
            "lora_weight_path": str(lora_path) if lora_path else None,
            "lora_weight_md5": lora_md5,
        },
        "generation_config": effective_cfg["generation"],
        "runtime_env": effective_cfg["runtime"],
        "metrics": metrics,
        "details": details,
    }

    tmp_file = result_file.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)
    tmp_file.replace(result_file)

    print(f"\n✅ [{mode.upper()}] 评测完成！结果落盘: {result_file}")
    print(f"  - 合规率: {metrics['single_letter_format_compliance_rate']*100:.2f}% ({compliant_count}/{total})")
    print(f"  - 正确率: {metrics['accuracy']*100:.2f}% ({correct_count}/{total})")
    print(f"  - 自然结束率: {metrics['natural_termination_rate']*100:.2f}% ({nat_term_count}/{total})")
    print(f"  - 截断率: {metrics['truncation_rate']*100:.2f}% ({truncated_count}/{total})")
    print(f"  - 输出分布: {metrics['letter_distribution']}")

    return result_payload


def run_comparison(args):
    """
    [Issue 3 & Issue 5 升级核验]
    严格核验前后对比条件与正式评测完整性：
    1. 冒烟与正式禁止混用对比。
    2. 正式评测完整性检查：严禁在正式对比中使用截断题目 (is_limited=True)；
       必须 100% 覆盖锁定测试集 (test.jsonl) 的全部题目，核对 ID、题干、选项、标准答案。
    3. 校验题集文件 MD5 是否与 manifest.json 完全一致。
    4. 校验基座模型权重 MD5 是否与 config.json 预期基座完全一致。
    5. 校验 LoRA 权重来源：必须来自本次 formal 训练产出的 train_summary.json，核对 selected_checkpoint MD5。
    6. 校验生成配置与运行环境一致性。
    任何不符立即抛出异常并阻断！
    """
    config = load_config()
    is_smoke = bool(args.smoke)
    run_paths = config["run_paths"]

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
    else:
        run_dir_rel = run_paths["smoke_dir"] if is_smoke else run_paths["formal_dir"]
        run_dir = ROOT / run_dir_rel

    eval_out_dir = run_dir / "eval_results"
    base_file = eval_out_dir / f"base_{args.split}_results.json"
    lora_file = eval_out_dir / f"lora_{args.split}_results.json"

    if not base_file.exists():
        raise FileNotFoundError(f"基座评测结果缺失: {base_file}")
    if not lora_file.exists():
        raise FileNotFoundError(f"LoRA 评测结果缺失: {lora_file}")

    with open(base_file, "r", encoding="utf-8") as f:
        base_res = json.load(f)
    with open(lora_file, "r", encoding="utf-8") as f:
        lora_res = json.load(f)

    # 1. 冒烟与正式混用核验
    if base_res.get("is_smoke") != lora_res.get("is_smoke"):
        raise ValueError(
            f"【对比条件违规】检测到混用冒烟与正式结果！"
            f"Base is_smoke={base_res.get('is_smoke')} vs LoRA is_smoke={lora_res.get('is_smoke')}。"
            f"禁止将冒烟评测与正式评测进行对比！"
        )

    # 2. [Issue 5 修复] 正式对比完整性核验 (禁止使用 limit 结果充当正式对比)
    if not is_smoke:
        if base_res.get("is_limited") or lora_res.get("is_limited"):
            raise ValueError(
                "【正式评测完整性拦截】正式对比评测禁止使用截断/部分题目结果 (--limit)！\n"
                f"Base is_limited={base_res.get('is_limited')}, LoRA is_limited={lora_res.get('is_limited')}。\n"
                "必须完整评测全量锁定测试集方可生成正式量化对比报告。"
            )

    # 3. 数据切分与数据集哈希核验
    if base_res["split"] != lora_res["split"]:
        raise ValueError(f"评测切分不一致: Base({base_res['split']}) vs LoRA({lora_res['split']})")

    if base_res["dataset"]["md5"] != lora_res["dataset"]["md5"]:
        raise ValueError(
            f"【数据集不一致】Base 与 LoRA 评测的数据集哈希不符！"
            f"Base: {base_res['dataset']['md5']} vs LoRA: {lora_res['dataset']['md5']}"
        )

    exp_root = ROOT / run_paths["experiments_root"]
    data_subdir = "smoke" if is_smoke else "formal"
    locked_split_file = exp_root / "data" / data_subdir / f"{args.split}.jsonl"

    if not locked_split_file.exists():
        raise FileNotFoundError(f"锁定的数据集文件不存在: {locked_split_file}")

    expected_md5 = md5_file(locked_split_file)
    if base_res["dataset"]["md5"] != expected_md5:
        raise ValueError(
            f"【题集篡改拦截】评测记录中的题集 MD5 ({base_res['dataset']['md5']}) 与磁盘锁定文件 MD5 ({expected_md5}) 不一致！"
        )

    # 校验数据清单文件 (manifest.json)
    manifest_p = exp_root / "data" / data_subdir / "manifest.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"数据清单文件不存在: {manifest_p}")
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    expected_split_hash_key = f"{args.split}_jsonl_md5"
    manifest_split_md5 = (
        manifest_data.get("output_hashes", {}).get(data_subdir, {}).get(expected_split_hash_key)
    )
    if not manifest_split_md5:
        raise ValueError(f"【数据清单校验失败】清单中未找到切分 [{args.split}] 的哈希条目: {expected_split_hash_key}")
    if expected_md5 != manifest_split_md5:
        raise ValueError(
            f"【数据清单校验失败】锁定题集文件 MD5 ({expected_md5}) 与清单记录 MD5 ({manifest_split_md5}) 不一致！"
        )

    # 4. [Issue 5 修复] 正式测试集题目全量覆盖与逐题内容对齐核验
    locked_items = []
    with open(locked_split_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                locked_items.append(json.loads(line))

    b_details = base_res["details"]
    l_details = lora_res["details"]

    if not is_smoke and args.split == "test":
        target_test_count = config["data_config"]["target_counts"]["formal"]["test"]
        if len(locked_items) != target_test_count:
            raise ValueError(
                f"【锁定考卷题数异常】锁定测试集题数 ({len(locked_items)}) != 预期规格 ({target_test_count})！"
            )
        if len(b_details) != target_test_count:
            raise ValueError(
                f"【评测未全量覆盖】Base 评测题数 ({len(b_details)}) 未覆盖正式测试集全量题数 ({target_test_count})！"
            )
        if len(l_details) != target_test_count:
            raise ValueError(
                f"【评测未全量覆盖】LoRA 评测题数 ({len(l_details)}) 未覆盖正式测试集全量题数 ({target_test_count})！"
            )

    if len(b_details) != len(l_details):
        raise ValueError(f"【完成题数不符】Base 题数 ({len(b_details)}) != LoRA 题数 ({len(l_details)})！")

    b_ids = [d["id"] for d in b_details]
    l_ids = [d["id"] for d in l_details]

    if len(set(b_ids)) != len(b_ids):
        raise ValueError("Base 评测产物中存在重复题目 ID！")
    if len(set(l_ids)) != len(l_ids):
        raise ValueError("LoRA 评测产物中存在重复题目 ID！")
    if b_ids != l_ids:
        raise ValueError("Base 与 LoRA 评测的题目 ID 顺序不一致或存在缺失题目！")

    # 与锁定文件逐题逐项严格对齐校验 (Base 与 LoRA 双向均校验 ID、题干、选项字典与标准答案)
    for b_item, l_item, locked_item in zip(b_details, l_details, locked_items[:len(b_details)]):
        if b_item["id"] != locked_item["id"]:
            raise ValueError(f"题目 ID 错位: Base 评测 {b_item['id']} vs 锁定数据 {locked_item['id']}")
        if l_item["id"] != locked_item["id"]:
            raise ValueError(f"题目 ID 错位: LoRA 评测 {l_item['id']} vs 锁定数据 {locked_item['id']}")
        if b_item["question"] != locked_item["question"]:
            raise ValueError(f"Base 题目 [{b_item['id']}] 题干与锁定数据不符！")
        if l_item["question"] != locked_item["question"]:
            raise ValueError(f"LoRA 题目 [{l_item['id']}] 题干与锁定数据不符！")
        if b_item["options"] != locked_item["options"]:
            raise ValueError(f"Base 题目 [{b_item['id']}] 选项字典与锁定数据不符！")
        if l_item["options"] != locked_item["options"]:
            raise ValueError(f"LoRA 题目 [{l_item['id']}] 选项字典与锁定数据不符！")
        if b_item["official_answer"] != locked_item["answer"].strip().upper():
            raise ValueError(f"Base 题目 [{b_item['id']}] 标准答案与锁定数据不符！")
        if l_item["official_answer"] != locked_item["answer"].strip().upper():
            raise ValueError(f"LoRA 题目 [{l_item['id']}] 标准答案与锁定数据不符！")

    # 5. [Issue 5 修复 & Provenance 溯源核验] 基座模型与 LoRA 权重及训练来源校验
    b_base_md5 = base_res["weights"]["base_weight_md5"]
    l_base_md5 = lora_res["weights"]["base_weight_md5"]
    expected_base_p = ROOT / config["base_model"]["weight_path"]
    expected_base_md5 = md5_file(expected_base_p)

    if b_base_md5 != l_base_md5:
        raise ValueError(
            f"【基座权重不一致】Base 基座 MD5 ({b_base_md5}) 与 LoRA 所挂载基座 MD5 ({l_base_md5}) 不一致！"
        )
    if b_base_md5 != expected_base_md5:
        raise ValueError(
            f"【基座权重来源违规】评测所用基座 MD5 ({b_base_md5}) 与 config 指定基座 MD5 ({expected_base_md5}) 不符！"
        )

    if base_res["weights"]["lora_weight_path"] is not None:
        raise ValueError("基座 (Base) 评测产物中非法挂载了 LoRA 权重！")
    if lora_res["weights"]["lora_weight_path"] is None:
        raise ValueError("LoRA 评测产物中未检测到 LoRA 权重挂载！")

    # 核验 LoRA 训练摘要与全链路溯源 (Provenance)
    train_summary_p = run_dir / "checkpoints" / "train_summary.json"
    if not train_summary_p.exists():
        if not is_smoke:
            raise FileNotFoundError(f"【训练来源缺失】未在运行目录找到对应的训练摘要: {train_summary_p}")
    else:
        with open(train_summary_p, "r", encoding="utf-8") as f:
            train_summary = json.load(f)
        if not is_smoke and train_summary.get("is_smoke"):
            raise ValueError("【严重违规】正式对比评测所用的 LoRA 来自冒烟训练！禁止将冒烟权重用于正式评测！")

        expected_selected_md5 = train_summary["selected_checkpoint"]["md5"]
        actual_lora_md5 = lora_res["weights"]["lora_weight_md5"]
        if actual_lora_md5 != expected_selected_md5:
            raise ValueError(
                f"【LoRA权重篡改拦截】评测所载入 LoRA MD5 ({actual_lora_md5}) 与训练摘要选定检查点 MD5 ({expected_selected_md5}) 不符！"
            )

        # 训练溯源 (Provenance) 严格校验：确保 LoRA 是在相同的基座和已认证数据上训练
        provenance = train_summary.get("provenance", {})
        if not provenance:
            raise ValueError("【训练溯源缺失】训练摘要缺失 provenance 追溯信息！")

        train_base_md5 = provenance.get("base_weight_md5")
        if not train_base_md5:
            raise ValueError("【训练溯源缺失】训练摘要缺失 base_weight_md5！")
        if train_base_md5 != b_base_md5:
            raise ValueError(
                f"【训练基座溯源不一致】LoRA 训练所用基座 MD5 ({train_base_md5}) 与评测基座 MD5 ({b_base_md5}) 不一致！"
            )

        train_data_md5 = provenance.get("train_data_md5")
        if not train_data_md5:
            raise ValueError("【训练溯源缺失】训练摘要缺失 train_data_md5！")
        manifest_train_md5 = manifest_data.get("output_hashes", {}).get(data_subdir, {}).get("train_jsonl_md5")
        if manifest_train_md5 and train_data_md5 != manifest_train_md5:
            raise ValueError(
                f"【训练数据溯源不一致】LoRA 训练所用训练集 MD5 ({train_data_md5}) 与清单记录 ({manifest_train_md5}) 不一致！"
            )

        manifest_val_md5 = manifest_data.get("output_hashes", {}).get(data_subdir, {}).get("val_jsonl_md5")
        val_data_md5 = provenance.get("val_data_md5")
        if manifest_val_md5 and val_data_md5 and val_data_md5 != manifest_val_md5:
            raise ValueError(
                f"【验证数据溯源不一致】LoRA 训练所用验证集 MD5 ({val_data_md5}) 与清单记录 ({manifest_val_md5}) 不一致！"
            )

    # 6. 生成配置与运行环境核验
    b_gen = base_res["generation_config"]
    l_gen = lora_res["generation_config"]
    for k in ["do_sample", "open_thinking", "max_new_tokens", "instruction", "target_mode"]:
        if b_gen.get(k) != l_gen.get(k):
            raise ValueError(f"【生成配置不一致】参数 {k} 不符: Base={b_gen.get(k)} vs LoRA={l_gen.get(k)}")

    b_env = base_res["runtime_env"]
    l_env = lora_res["runtime_env"]
    for k in ["device", "dtype"]:
        if b_env.get(k) != l_env.get(k):
            raise ValueError(f"【运行环境不一致】参数 {k} 不符: Base={b_env.get(k)} vs LoRA={l_env.get(k)}")

    # 7. 异动统计与报告生成
    wrong_to_right = []
    right_to_wrong = []
    right_to_right = []
    wrong_to_wrong = []

    for b_item, l_item in zip(b_details, l_details):
        qid = b_item["id"]
        b_corr = b_item["is_correct"]
        l_corr = l_item["is_correct"]

        rec = {
            "id": qid,
            "question": b_item["question"],
            "official_answer": b_item["official_answer"],
            "base_raw": b_item["raw_model_output"],
            "base_letter": b_item["predicted_letter"],
            "base_compliant": b_item["is_format_compliant"],
            "lora_raw": l_item["raw_model_output"],
            "lora_letter": l_item["predicted_letter"],
            "lora_compliant": l_item["is_format_compliant"],
        }

        if not b_corr and l_corr:
            wrong_to_right.append(rec)
        elif b_corr and not l_corr:
            right_to_wrong.append(rec)
        elif b_corr and l_corr:
            right_to_right.append(rec)
        else:
            wrong_to_wrong.append(rec)

    b_metrics = base_res["metrics"]
    l_metrics = lora_res["metrics"]

    comparison = {
        "experiment": "lora_medical_mcq_pilot",
        "timestamp": datetime.now().isoformat(),
        "is_smoke": is_smoke,
        "split": args.split,
        "total_questions": b_metrics["total_questions"],
        "verification_status": "ALL_FORMAL_CONDITIONS_AND_COMPLETENESS_VERIFIED",
        "dataset": {
            "path": str(locked_split_file),
            "md5": expected_md5,
            "total_covered": len(b_details),
        },
        "base_model": {
            "path": base_res["weights"]["base_weight_path"],
            "md5": base_res["weights"]["base_weight_md5"],
        },
        "lora_model": {
            "path": lora_res["weights"]["lora_weight_path"],
            "md5": lora_res["weights"]["lora_weight_md5"],
        },
        "base_metrics": b_metrics,
        "lora_metrics": l_metrics,
        "deltas": {
            "format_compliance_delta": round(
                l_metrics["single_letter_format_compliance_rate"] - b_metrics["single_letter_format_compliance_rate"], 6
            ),
            "accuracy_delta": round(l_metrics["accuracy"] - b_metrics["accuracy"], 6),
            "natural_termination_delta": round(
                l_metrics["natural_termination_rate"] - b_metrics["natural_termination_rate"], 6
            ),
            "truncation_delta": round(l_metrics["truncation_rate"] - b_metrics["truncation_rate"], 6),
        },
        "transitions": {
            "wrong_to_right_count": len(wrong_to_right),
            "right_to_wrong_count": len(right_to_wrong),
            "right_to_right_count": len(right_to_right),
            "wrong_to_wrong_count": len(wrong_to_wrong),
        },
        "wrong_to_right_details": wrong_to_right,
        "right_to_wrong_details": right_to_wrong,
    }

    comp_file = eval_out_dir / f"comparison_{args.split}.json"
    if comp_file.exists() and not args.overwrite:
        raise FileExistsError(f"对比产物已存在: {comp_file}。默认拒绝覆盖！")

    tmp_comp = comp_file.with_suffix(".tmp")
    with open(tmp_comp, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    tmp_comp.replace(comp_file)

    print("\n" + "=" * 60)
    print(f"🎉 严格对比评估完成！产物落盘: {comp_file}")
    print(f"  - 校验状态: 锁定考卷题数全量覆盖、题目内容/选项/答案逐字对齐、训练权重来源一致性 100% 通过")
    print(f"  - 格式合规率: Base {b_metrics['single_letter_format_compliance_rate']*100:.2f}% -> LoRA {l_metrics['single_letter_format_compliance_rate']*100:.2f}% (变化: {comparison['deltas']['format_compliance_delta']*100:+.2f}%)")
    print(f"  - 全部正确率: Base {b_metrics['accuracy']*100:.2f}% -> LoRA {l_metrics['accuracy']*100:.2f}% (变化: {comparison['deltas']['accuracy_delta']*100:+.2f}%)")
    print(f"  - 自然结束率: Base {b_metrics['natural_termination_rate']*100:.2f}% -> LoRA {l_metrics['natural_termination_rate']*100:.2f}% (变化: {comparison['deltas']['natural_termination_delta']*100:+.2f}%)")
    print(f"  - 截断率变化: Base {b_metrics['truncation_rate']*100:.2f}% -> LoRA {l_metrics['truncation_rate']*100:.2f}% (变化: {comparison['deltas']['truncation_delta']*100:+.2f}%)")
    print(f"  - 改善题数 (错→对): {len(wrong_to_right)}")
    print(f"  - 退化题数 (对→错): {len(right_to_wrong)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="MiniMind 医学单选题评测与前后对比工具")
    parser.add_argument("--mode", type=str, choices=["base", "lora", "compare"], default="base")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test",
                        help="评测切分集：val 选拔/冒烟，test 仅用于正式终测")
    parser.add_argument("--smoke", action="store_true", help="以冒烟模式运行 (使用 smoke 运行目录与数据)")
    parser.add_argument("--run_dir", type=str, default=None, help="显式指定独立运行目录")
    parser.add_argument("--base_weight", type=str, default=None, help="基座模型权重路径")
    parser.add_argument("--lora_weight", type=str, default=None, help="指定 LoRA 检查点路径 (若不传自动使用选定检查点)")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="最大生成 Token 数")
    parser.add_argument("--target_mode", choices=["letter", "answer_text", "answer_rationale"], default=None,
                        help="评测输出格式；默认从训练摘要自动读取")
    parser.add_argument("--limit", type=int, default=None, help="测试题目截断数")
    parser.add_argument("--dtype", type=str, default=None, help="数值精度类型")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的评测产物")
    args = parser.parse_args()

    if args.mode == "compare":
        run_comparison(args)
    else:
        run_evaluation(args, mode=args.mode)


if __name__ == "__main__":
    main()
