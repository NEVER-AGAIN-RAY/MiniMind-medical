#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniMind LoRA 医学微调评估与对比脚本 (run_eval.py)

功能：
1. 评估模式 (--mode base / --mode lora):
   - 加载基座或带 LoRA 的模型
   - 在锁定题集上进行无抽样、确定性评估 (do_sample=False, open_thinking=False)
   - MCQ (64 tokens): 保守抽取选项、统计正确率与无效回答数
   - MedQA (512 tokens): 生成同分布医学问答回答（参考答案不输入模型）
   - General (256 tokens): 生成通用问题回答（灾难性遗忘检查）
   - 可选 --val-loss: 复用 SFTDataset(max_length=512, augment=False 固定模板) 按有效目标 token 严格加权计算交叉熵
   - 保存 mcq_answers.jsonl, medqa_answers.jsonl, general_answers.jsonl, summary.json
   - 支持 --limit N 用于云端冒烟测试（默认保存至独立 smoke 目录，防止覆盖正式结果）

2. 对比模式 (--compare):
   - 严格检查 base 与 lora 的生成配置、题集版本与题目 ID 是否一致
   - 对比 MCQ 正确率、无效回答数、val_loss
   - 统计选择题 错→对、对→错 异动
   - 逐题并排展示医学问答与通用问题
   - 预留人工临床评估栏（回答是否充分、事实错误、无依据诊疗建议、总体良好）
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.lm_dataset import SFTDataset
from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import evaluate_val_loss


def md5_file(filepath: Path | str) -> str:
    """计算文件 MD5 哈希"""
    p = Path(filepath)
    if not p.is_file():
        return ""
    hasher = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_mcq_choice(raw_text: str, valid_options) -> str | None:
    """Only accept an explicit leading choice; reject lists and contradictory answers."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    valid = {str(k).upper() for k in valid_options}
    text = raw_text.strip().strip("'\"`")
    prefix = r"(?:(?:正确)?答案|Answer|(?:(?:本题|故|所以|因此|我|建议|应|应当)?(?:选择|选项|选)))"
    decoration = r"[（(【\[\*\s]*"
    choice = r"([A-E])(?![A-Za-z])"
    match = re.match(
        rf"^(?:{prefix}(?:应该选|是|为|选|[:：\s])*)?{decoration}{choice}",
        text, re.IGNORECASE,
    )
    if not match or match.group(1).upper() not in valid:
        return None
    candidate = match.group(1).upper()
    tail = text[match.end():]
    # Require a separator after the leading choice, so '选项A不正确' is not an answer.
    if tail and not re.match(r"^[）)】\]\*\s.。、:：,，/;；-]", tail):
        return None
    # Multiple choices, including prefixed and decorated lists.
    if re.match(r"^[）)】\]\*\s]*(?:[、,，/]|或者|还是|或|和|与)+[（(【\[\*\s]*[A-E](?![A-Za-z])", tail, re.I):
        return None
    # A later explicit answer that differs from the first is ambiguous.
    for other in re.finditer(rf"{prefix}(?:应该选|是|为|选|[:：\s])*{decoration}{choice}", tail, re.I):
        if other.group(1).upper() != candidate:
            return None
    # Also reject a second standalone choice on a new sentence/line.
    if re.search(r"(?:[。；;\n])\s*[（(【\[\*]*[A-E](?![A-Za-z])", tail, re.I):
        return None
    return candidate


def table_cell(text: str) -> str:
    """Keep multiline model text inside one Markdown table cell."""
    import html
    text = html.escape(str(text), quote=False).replace("\r\n", "\n").replace("\r", "\n")
    for char in ("\\", "|", "`", "*", "_", "[", "]"):
        text = text.replace(char, f"&#{ord(char)};")
    return text.replace("\n", "<br>")


def format_mcq_prompt(item: dict) -> str:
    """构建 MCQ 的输入 prompt，提示仅输出选项字母"""
    question = item['question'].strip()
    options = item['options']
    lines = [question]
    for k in sorted(options.keys()):
        lines.append(f"{k}. {options[k]}")
    lines.append("请直接输出正确选项的字母（如 A、B、C、D、E），不要输出多余解释。")
    return "\n".join(lines)


def resolve_weight_path(weight_arg: str, hidden_size: int, default_dir: str = "out") -> str:
    """解析权重路径：支持全路径、相对路径或名称前缀"""
    p = Path(weight_arg)
    if p.is_file():
        return str(p.resolve())

    # 尝试在 default_dir 下查找
    cands = [
        Path(default_dir) / weight_arg,
        Path(default_dir) / f"{weight_arg}_{hidden_size}.pth",
        ROOT / default_dir / weight_arg,
        ROOT / default_dir / f"{weight_arg}_{hidden_size}.pth",
    ]
    for cand in cands:
        if cand.is_file():
            return str(cand.resolve())

    # 若未能定位现有文件，返回最佳猜测
    return str((ROOT / default_dir / f"{weight_arg}_{hidden_size}.pth").resolve())


def load_model_for_eval(args):
    """根据 args 初始化模型与 Tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindForCausalLM(lm_config)

    base_path = resolve_weight_path(args.base_weight, args.hidden_size)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"基座权重文件不存在: {base_path}")

    print(f"[{args.mode.upper()}] 加载基座权重: {base_path}")
    base_state = torch.load(base_path, map_location=args.device)
    model.load_state_dict(base_state, strict=True)

    if args.mode == "lora":
        lora_path = resolve_weight_path(args.lora_weight, args.hidden_size)
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA 权重文件不存在: {lora_path}")
        print(f"[{args.mode.upper()}] 应用 LoRA 并加载权重: {lora_path}")
        apply_lora(model)
        load_lora(model, lora_path)

    # 设置推理精度与 eval 状态
    target_dtype = torch.bfloat16 if args.dtype == "bfloat16" else (
        torch.float16 if args.dtype == "float16" else torch.float32
    )
    model = model.to(dtype=target_dtype, device=args.device)
    model.eval()

    return model, tokenizer, base_path, (lora_path if args.mode == "lora" else None)


@torch.no_grad()
def generate_single_response(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    device: str,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    """确定性生成单轮回答 (do_sample=False, open_thinking=False)"""
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

    # 上下文长度检查：防止输入加输出超出模型最大位置编码
    max_pos = getattr(model.config, "max_position_embeddings", 32768)
    if input_ids.shape[1] + max_new_tokens > max_pos:
        keep_len = max_pos - max_new_tokens
        input_ids = input_ids[:, -keep_len:]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -keep_len:]

    prompt_len = input_ids.shape[1]
    generated_ids = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    new_token_ids = generated_ids[0][prompt_len:]
    response = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return response.strip()


def run_evaluation(args):
    """执行评估流程"""
    # 确定输出目录
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        prefix = "smoke_" if args.limit else ""
        out_dir = ROOT / "experiments" / "lora_medical_20260909" / "eval_results" / f"{prefix}{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = Path(args.eval_dir)
    mcq_file = eval_dir / "questions_mcq.jsonl"
    medqa_file = eval_dir / "questions_medqa.jsonl"
    gen_file = eval_dir / "questions_general.jsonl"
    manifest_file = eval_dir / "manifest.json"

    for f in [mcq_file, medqa_file, gen_file]:
        if not f.exists():
            raise FileNotFoundError(f"题集文件不存在: {f}")

    # 读取全部题集
    with open(mcq_file, "r", encoding="utf-8") as f:
        mcq_items = [json.loads(line) for line in f if line.strip()]
    with open(medqa_file, "r", encoding="utf-8") as f:
        medqa_items = [json.loads(line) for line in f if line.strip()]
    with open(gen_file, "r", encoding="utf-8") as f:
        gen_items = [json.loads(line) for line in f if line.strip()]

    # 若指定 --limit 则截断
    if args.limit is not None and args.limit > 0:
        mcq_items = mcq_items[:args.limit]
        medqa_items = medqa_items[:args.limit]
        gen_items = gen_items[:args.limit]
        print(f"⚠️ [冒烟模式] 每类考卷仅评估前 {args.limit} 道题，结果将写入: {out_dir}")

    # Validate the locked files before spending GPU time.
    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)
    for source in (mcq_file, medqa_file, gen_file):
        if md5_file(source) != manifest.get("files", {}).get(source.name):
            raise ValueError(f"锁定题集哈希不匹配: {source}")

    # 加载模型
    model, tokenizer, base_path, lora_path = load_model_for_eval(args)

    # 1. 评估 MCQ
    print(f"\n--- [1/3] 评估医学选择题 (MCQ, 共 {len(mcq_items)} 题, max_new_tokens=64) ---")
    mcq_results = []
    mcq_correct = 0
    mcq_invalid = 0

    for idx, item in enumerate(mcq_items, 1):
        prompt = format_mcq_prompt(item)
        raw_ans = generate_single_response(
            model, tokenizer, prompt, max_new_tokens=64, device=args.device,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        extracted = extract_mcq_choice(raw_ans, item["options"].keys())
        official = item.get("answer", "").strip().upper()

        if extracted is None:
            is_correct = False
            parse_status = "invalid"
            mcq_invalid += 1
        else:
            is_correct = (extracted == official)
            parse_status = "parsed"
            if is_correct:
                mcq_correct += 1

        res_entry = {
            "id": item["id"],
            "q_md5": item.get("q_md5", ""),
            "question": item["question"],
            "options": item["options"],
            "official_answer": official,
            "explanation": item.get("explanation", ""),
            "model_raw_answer": raw_ans,
            "extracted_answer": extracted,
            "is_correct": is_correct,
            "parse_status": parse_status,
        }
        mcq_results.append(res_entry)
        if idx % 20 == 0 or idx == len(mcq_items):
            print(f"MCQ 进度: {idx}/{len(mcq_items)} | 正确: {mcq_correct} | 无效: {mcq_invalid}")

    mcq_acc = (mcq_correct / len(mcq_items)) if mcq_items else 0.0
    print(f"MCQ 评估完成: 正确率 {mcq_acc * 100:.2f}% ({mcq_correct}/{len(mcq_items)}), 无效回答数: {mcq_invalid}")

    # 2. 评估医学问答 (MedQA)
    print(f"\n--- [2/3] 评估医学问答 (MedQA, 共 {len(medqa_items)} 题, max_new_tokens=512) ---")
    medqa_results = []
    for idx, item in enumerate(medqa_items, 1):
        prompt = item["question"].strip()
        ans = generate_single_response(
            model, tokenizer, prompt, max_new_tokens=512, device=args.device,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        res_entry = {
            "id": item["id"],
            "q_md5": item.get("q_md5", ""),
            "question": item["question"],
            "reference_answer": item.get("reference_answer", ""),
            "model_answer": ans,
        }
        medqa_results.append(res_entry)
        if idx % 10 == 0 or idx == len(medqa_items):
            print(f"MedQA 进度: {idx}/{len(medqa_items)}")

    # 3. 评估通用问题 (General QA)
    print(f"\n--- [3/3] 评估通用问题 (General QA, 共 {len(gen_items)} 题, max_new_tokens=256) ---")
    gen_results = []
    for idx, item in enumerate(gen_items, 1):
        prompt = item["question"].strip()
        ans = generate_single_response(
            model, tokenizer, prompt, max_new_tokens=256, device=args.device,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        res_entry = {
            "id": item["id"],
            "question": item["question"],
            "model_answer": ans,
        }
        gen_results.append(res_entry)

    # 4. 可选: 计算验证集 Loss (val_loss)
    val_loss_val = None
    if args.val_loss:
        val_path = Path(args.val_path)
        if not val_path.exists():
            print(f"⚠️ 警告: 验证集文件不存在: {val_path}，跳过 val_loss 计算")
        else:
            print(f"\n--- 计算验证集 Loss (SFTDataset, max_length=512, 路径: {val_path}) ---")
            # 评估必须用固定模板/固定标签(augment=False)，否则 base 与 lora 两次运行
            # 的验证输入随机不同，val_loss 前后对比失去意义
            val_ds = SFTDataset(str(val_path), tokenizer, max_length=512, augment=False)
            if args.limit is not None and args.limit > 0:
                # 冒烟模式下只取少量样本
                from torch.utils.data import Subset
                val_ds = Subset(val_ds, list(range(min(len(val_ds), args.limit * 4))))
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            device_type = "cuda" if "cuda" in args.device else "cpu"
            target_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
            autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=target_dtype)
            val_loss_val = evaluate_val_loss(model, val_loader, args.device, autocast_ctx)
            print(f"验证集加权平均交叉熵 Loss: {val_loss_val:.4f}")

    # 5. 落盘全部结果
    def write_jsonl_atomic(path: Path, items: list[dict]):
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        tmp.replace(path)

    write_jsonl_atomic(out_dir / "mcq_answers.jsonl", mcq_results)
    write_jsonl_atomic(out_dir / "medqa_answers.jsonl", medqa_results)
    write_jsonl_atomic(out_dir / "general_answers.jsonl", gen_results)

    summary = {
        "mode": args.mode,
        "is_smoke": bool(args.limit is not None and args.limit > 0),
        "limit": args.limit,
        "timestamp": datetime.now().isoformat(),
        "completed_questions": {
            "mcq": len(mcq_results),
            "medqa": len(medqa_results),
            "general": len(gen_results),
            "total": len(mcq_results) + len(medqa_results) + len(gen_results),
        },
        "mcq_metrics": {
            "total": len(mcq_results),
            "correct": mcq_correct,
            "accuracy": round(mcq_acc, 6),
            "invalid_count": mcq_invalid,
        },
        "val_loss": round(val_loss_val, 6) if val_loss_val is not None else None,
        "generation_config": {
            "do_sample": False,
            "open_thinking": False,
            "add_generation_prompt": True,
            "mcq_max_new_tokens": 64,
            "medqa_max_new_tokens": 512,
            "general_max_new_tokens": 256,
            "mcq_parser_version": 2,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        },
        "runtime_env": {
            "device": args.device,
            "dtype": args.dtype,
        },
        "identity": {
            "val_path_md5": md5_file(args.val_path) if args.val_loss else None,
            "base_weight_path": base_path,
            "base_weight_md5": md5_file(base_path),
            "lora_weight_path": lora_path,
            "lora_weight_md5": md5_file(lora_path) if lora_path else None,
            "eval_set_manifest": str(manifest_file),
            "eval_set_manifest_md5": md5_file(manifest_file),
        },
    }

    summary_tmp = out_dir / "summary.json.tmp"
    with open(summary_tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary_tmp.replace(out_dir / "summary.json")

    print(f"\n✅ 评估成功完成，全部结果已写入: {out_dir}")
    print(f"summary: {json.dumps(summary['mcq_metrics'], ensure_ascii=False)}, val_loss: {summary['val_loss']}")


def validate_results(result_dir, eval_dir, expected_mode=None, allow_smoke=False):
    """Reject stale, truncated or edited result bundles, including identical omissions."""
    result_dir, eval_dir = Path(result_dir), Path(eval_dir)
    summary = json.loads((result_dir / "summary.json").read_text())
    manifest = json.loads((eval_dir / "manifest.json").read_text())
    if expected_mode and summary.get("mode") != expected_mode:
        raise ValueError("评估模式不匹配")
    if summary.get("is_smoke") and not allow_smoke:
        raise ValueError("冒烟结果不能用作正式结果")
    if summary.get("identity", {}).get("eval_set_manifest_md5") != md5_file(eval_dir / "manifest.json"):
        raise ValueError("题集 manifest 不匹配")
    counts = {}
    for kind in ("mcq", "medqa", "general"):
        source = eval_dir / f"questions_{kind}.jsonl"
        if md5_file(source) != manifest.get("files", {}).get(source.name):
            raise ValueError(f"锁定题集内容变更: {source.name}")
        expected = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if summary.get("is_smoke") and allow_smoke:
            expected = expected[:summary["limit"]]
        rows = [json.loads(line) for line in (result_dir / f"{kind}_answers.jsonl").read_text().splitlines() if line.strip()]
        ids = [row["id"] for row in rows]
        if len(set(ids)) != len(ids) or ids != [row["id"] for row in expected]:
            raise ValueError(f"{kind} 缺题、重复或 ID 不匹配")
        for row, original in zip(rows, expected):
            for key in ("question", "options", "q_md5", "reference_answer"):
                if key in original and row.get(key) != original[key]:
                    raise ValueError(f"{kind} 题目内容不匹配: {row['id']}")
            if kind == "mcq":
                if row.get("official_answer") != original["answer"]:
                    raise ValueError("MCQ 标准答案不匹配")
                extracted = extract_mcq_choice(row.get("model_raw_answer", ""), original["options"])
                correct = extracted == original["answer"] if extracted else False
                if (row.get("extracted_answer") != extracted or row.get("is_correct") != correct
                        or row.get("parse_status") != ("parsed" if extracted else "invalid")):
                    raise ValueError("MCQ 评分与原始回答不匹配，请重新评估")
        counts[kind] = len(rows)
        if kind == "mcq":
            correct = sum(bool(row["is_correct"]) for row in rows)
            metrics = {"total": len(rows), "correct": correct,
                       "accuracy": round(correct / len(rows), 6) if rows else 0.0,
                       "invalid_count": sum(row["parse_status"] == "invalid" for row in rows)}
            if summary.get("mcq_metrics") != metrics:
                raise ValueError("MCQ 汇总与逐题结果不一致")
    counts["total"] = sum(counts.values())
    if summary.get("completed_questions") != counts:
        raise ValueError("完成题数与文件实际内容不一致")
    return summary


def run_comparison(args):
    """执行对比流程，生成 Markdown 比较报告"""
    base_dir = Path(args.base_dir)
    lora_dir = Path(args.lora_dir)
    save_path = Path(args.save_report)

    if not base_dir.is_dir():
        raise FileNotFoundError(f"基座评估目录不存在: {base_dir}")
    if not lora_dir.is_dir():
        raise FileNotFoundError(f"LoRA 评估目录不存在: {lora_dir}")

    base_summary_p = base_dir / "summary.json"
    lora_summary_p = lora_dir / "summary.json"

    if not base_summary_p.exists() or not lora_summary_p.exists():
        raise FileNotFoundError("两个评估目录都必须包含 summary.json")

    with open(base_summary_p, "r", encoding="utf-8") as f:
        base_summary = json.load(f)
    with open(lora_summary_p, "r", encoding="utf-8") as f:
        lora_summary = json.load(f)

    eval_dir = args.eval_dir
    validate_results(base_dir, eval_dir, "base", allow_smoke=args.force)
    validate_results(lora_dir, eval_dir, "lora", allow_smoke=args.force)
    for field in ("base_weight_md5", "val_path_md5"):
        b = base_summary.get("identity", {}).get(field)
        l = lora_summary.get("identity", {}).get(field)
        if (field == "base_weight_md5" and not b) or b != l:
            raise ValueError(f"前后 {field} 不一致或缺失")
    for field in ("device", "dtype"):
        b = base_summary.get("runtime_env", {}).get(field)
        l = lora_summary.get("runtime_env", {}).get(field)
        if not b or b != l:
            raise ValueError(f"前后 {field} 不一致或缺失")

    # 1. 严格配置与题集校验
    print("正在检查 Base 与 LoRA 评估产物一致性...")

    # 检查冒烟标志
    if not args.force:
        if base_summary.get("is_smoke") or lora_summary.get("is_smoke"):
            raise ValueError(
                "检测到评估产物为冒烟测试数据 (--limit)，不能直接生成正式比较报告！\n"
                "如确需对比冒烟数据，请显式追加 --force 参数。"
            )

    # 检查生成配置
    b_gen = base_summary.get("generation_config", {})
    l_gen = lora_summary.get("generation_config", {})
    if b_gen != l_gen:
        msg = f"生成配置不一致！Base: {b_gen} vs LoRA: {l_gen}"
        raise ValueError(msg)

    # 检查题集 manifest md5
    b_man_md5 = base_summary.get("identity", {}).get("eval_set_manifest_md5")
    l_man_md5 = lora_summary.get("identity", {}).get("eval_set_manifest_md5")
    if b_man_md5 and l_man_md5 and b_man_md5 != l_man_md5:
        msg = f"题集版本不一致！Base manifest md5: {b_man_md5} vs LoRA: {l_man_md5}"
        raise ValueError(msg)

    # 2. 读取题目与回答并校验 ID 对应
    def load_jsonl(p: Path) -> list[dict]:
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    base_mcq = load_jsonl(base_dir / "mcq_answers.jsonl")
    lora_mcq = load_jsonl(lora_dir / "mcq_answers.jsonl")
    base_medqa = load_jsonl(base_dir / "medqa_answers.jsonl")
    lora_medqa = load_jsonl(lora_dir / "medqa_answers.jsonl")
    base_gen = load_jsonl(base_dir / "general_answers.jsonl")
    lora_gen = load_jsonl(lora_dir / "general_answers.jsonl")

    if len(base_mcq) != len(lora_mcq):
        raise ValueError(f"MCQ 题数不匹配: Base {len(base_mcq)} 题 vs LoRA {len(lora_mcq)} 题")
    if len(base_medqa) != len(lora_medqa):
        raise ValueError(f"MedQA 题数不匹配: Base {len(base_medqa)} 题 vs LoRA {len(lora_medqa)} 题")
    if len(base_gen) != len(lora_gen):
        raise ValueError(f"General 题数不匹配: Base {len(base_gen)} 题 vs LoRA {len(lora_gen)} 题")

    # 校验 ID
    for b_item, l_item in zip(base_mcq, lora_mcq):
        if b_item["id"] != l_item["id"]:
            raise ValueError(f"MCQ 题目 ID 错位: {b_item['id']} vs {l_item['id']}")
    for b_item, l_item in zip(base_medqa, lora_medqa):
        if b_item["id"] != l_item["id"] or b_item.get("q_md5") != l_item.get("q_md5"):
            raise ValueError(f"MedQA 题目 ID / q_md5 错位: {b_item['id']} vs {l_item['id']}")
    for b_item, l_item in zip(base_gen, lora_gen):
        if b_item["id"] != l_item["id"]:
            raise ValueError(f"General 题目 ID 错位: {b_item['id']} vs {l_item['id']}")

    print("✅ 一致性校验通过，开始统计异动指标...")

    # 3. 计算 MCQ 统计与异动转移
    mcq_total = len(base_mcq)
    wrong_to_right = []  # 错 -> 对 (改进)
    right_to_wrong = []  # 对 -> 错 (退化)
    right_to_right = []  # 对 -> 对
    wrong_to_wrong = []  # 错 -> 错

    for b, l in zip(base_mcq, lora_mcq):
        qid = b["id"]
        b_corr = b["is_correct"]
        l_corr = l["is_correct"]
        if not b_corr and l_corr:
            wrong_to_right.append((qid, b, l))
        elif b_corr and not l_corr:
            right_to_wrong.append((qid, b, l))
        elif b_corr and l_corr:
            right_to_right.append((qid, b, l))
        else:
            wrong_to_wrong.append((qid, b, l))

    b_acc = base_summary["mcq_metrics"]["accuracy"]
    l_acc = lora_summary["mcq_metrics"]["accuracy"]
    b_inv = base_summary["mcq_metrics"]["invalid_count"]
    l_inv = lora_summary["mcq_metrics"]["invalid_count"]
    b_loss = base_summary.get("val_loss")
    l_loss = lora_summary.get("val_loss")

    # 4. 组装 Markdown 比较报告
    md = []
    md.append("# MiniMind LoRA 医学微调前后对比评估报告")
    md.append(f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"> 基座产物: `{base_dir}` | LoRA 产物: `{lora_dir}`\n")

    # 基础信息表
    md.append("## 1. 实验运行元信息\n")
    md.append("| 属性 | Base 基座 | LoRA 微调 |")
    md.append("| :--- | :--- | :--- |")
    md.append(f"| **模型权重** | `{Path(base_summary['identity']['base_weight_path']).name}` | `{Path(lora_summary['identity']['lora_weight_path']).name if lora_summary['identity']['lora_weight_path'] else 'N/A'}` |")
    md.append(f"| **权重 SHA256/MD5** | `{base_summary['identity']['base_weight_md5'][:10]}...` | `{lora_summary['identity']['lora_weight_md5'][:10] if lora_summary['identity']['lora_weight_md5'] else 'N/A'}...` |")
    md.append(f"| **评估设备 & Dtype** | `{base_summary['runtime_env']['device']}` (`{base_summary['runtime_env']['dtype']}`) | `{lora_summary['runtime_env']['device']}` (`{lora_summary['runtime_env']['dtype']}`) |")
    md.append(f"| **题集 Manifest** | `{Path(base_summary['identity']['eval_set_manifest']).name}` (`{base_summary['identity']['eval_set_manifest_md5'][:10]}...`) | `{Path(lora_summary['identity']['eval_set_manifest']).name}` (`{lora_summary['identity']['eval_set_manifest_md5'][:10]}...`) |")
    md.append(f"| **生成配置** | `do_sample=False, open_thinking=False` | `do_sample=False, open_thinking=False` |")
    md.append(f"| **最大生成 Token** | MCQ: 64, MedQA: 512, General: 256 | MCQ: 64, MedQA: 512, General: 256 |\n")

    # 核心指标对比表
    md.append("## 2. 核心量化指标对比\n")
    loss_delta_str = f"{l_loss - b_loss:+.4f}" if (b_loss is not None and l_loss is not None) else "N/A"
    b_loss_str = f"{b_loss:.4f}" if b_loss is not None else "N/A"
    l_loss_str = f"{l_loss:.4f}" if l_loss is not None else "N/A"

    acc_delta = (l_acc - b_acc) * 100
    inv_delta = l_inv - b_inv

    md.append("| 评估维度 | Base 基座 | LoRA 微调 | 差异变化 (Delta) | 评估说明 |")
    md.append("| :--- | :---: | :---: | :---: | :--- |")
    md.append(f"| **MCQ 正确率** | **{b_acc * 100:.2f}%** ({base_summary['mcq_metrics']['correct']}/{mcq_total}) | **{l_acc * 100:.2f}%** ({lora_summary['mcq_metrics']['correct']}/{mcq_total}) | **{acc_delta:+.2f}%** | {mcq_total}道CMExam医学选择题 |")
    md.append(f"| **MCQ 无效回答数** | {b_inv} | {l_inv} | {inv_delta:+d} | 无法提取确定选项或冲突格式 |")
    md.append(f"| **验证集 Loss** | {b_loss_str} | {l_loss_str} | **{loss_delta_str}** | 100条val.jsonl加权平均交叉熵 |")
    md.append(f"| **错→对 (翻盘修复)** | — | **{len(wrong_to_right)} 题** | +{len(wrong_to_right)} | 基座答错但LoRA答对 |")
    md.append(f"| **对→错 (退化倒退)** | — | **{len(right_to_wrong)} 题** | -{len(right_to_wrong)} | 基座答对但LoRA答错 |")
    md.append(f"| **对→对 (保持正确)** | — | {len(right_to_right)} 题 | — | 始终答对 |")
    md.append(f"| **错→错 (保持错误)** | — | {len(wrong_to_wrong)} 题 | — | 始终答错 |\n")

    # 选择题异动明细
    md.append("## 3. 选择题异动明细 (错→对 与 对→错)\n")
    if wrong_to_right:
        md.append(f"### 3.1 错→对 (微调改进题目, 共 {len(wrong_to_right)} 题)\n")
        md.append("| 题目 ID | 题干摘要 | 官方答案 | Base 输出 (抽取) | LoRA 输出 (抽取) |")
        md.append("| :--- | :--- | :---: | :---: | :---: |")
        for qid, b, l in wrong_to_right:
            q_brief = b['question'][:30] + ("..." if len(b['question']) > 30 else "")
            b_str = f"`{b['extracted_answer'] or '无效'}` ({b['parse_status']})"
            l_str = f"`{l['extracted_answer'] or '无效'}` ({l['parse_status']})"
            md.append(f"| **{qid}** | {table_cell(q_brief)} | `{b['official_answer']}` | {b_str} | **{l_str}** |")
        md.append("")
    else:
        md.append("### 3.1 错→对: 无题目\n")

    if right_to_wrong:
        md.append(f"### 3.2 对→错 (微调退化题目, 共 {len(right_to_wrong)} 题)\n")
        md.append("| 题目 ID | 题干摘要 | 官方答案 | Base 输出 (抽取) | LoRA 输出 (抽取) |")
        md.append("| :--- | :--- | :---: | :---: | :---: |")
        for qid, b, l in right_to_wrong:
            q_brief = b['question'][:30] + ("..." if len(b['question']) > 30 else "")
            b_str = f"`{b['extracted_answer'] or '无效'}` ({b['parse_status']})"
            l_str = f"`{l['extracted_answer'] or '无效'}` ({l['parse_status']})"
            md.append(f"| **{qid}** | {table_cell(q_brief)} | `{b['official_answer']}` | {b_str} | **{l_str}** |")
        md.append("")
    else:
        md.append("### 3.2 对→错: 无退化题目\n")

    # 医学问答逐题并排展示与人工评分栏
    md.append(f"## 4. 医学问答逐题对比与人工评估 ({len(base_medqa)} 题)\n")
    md.append("> 声明：医学问答评分栏预留给临床人工专家评审，未自动编造任何评分结论。\n")

    for idx, (b, l) in enumerate(zip(base_medqa, lora_medqa), 1):
        qid = b["id"]
        q_text = b["question"]
        ref_ans = b.get("reference_answer", "").strip()
        b_ans = b.get("model_answer", "").strip()
        l_ans = l.get("model_answer", "").strip()

        md.append(f"### [{qid}] 问题 {idx}: {q_text}\n")
        md.append(f"**参考答案**:\n```\n{ref_ans}\n```\n")
        md.append("| 模型 | 生成回答 |")
        md.append("| :--- | :--- |")
        md.append(f"| **Base 基座** | {table_cell(b_ans)} |")
        md.append(f"| **LoRA 微调** | {table_cell(l_ans)} |\n")
        md.append("**人工评分栏**:\n")
        md.append("| 评估项 | Base 基座 | LoRA 微调 | 评分准则 |")
        md.append("| :--- | :---: | :---: | :--- |")
        md.append("| 回答是否充分 (on_topic) | [ ] 是 / [ ] 否 | [ ] 是 / [ ] 否 | 是否正面且充分回应了问题 |")
        md.append("| 存在事实错误 (factual_error) | [ ] 是 / [ ] 否 | [ ] 是 / [ ] 否 | 是否存在违背医学常识的错误陈述 |")
        md.append("| 无依据诊疗建议 (unsourced_advice) | [ ] 是 / [ ] 否 | [ ] 是 / [ ] 否 | 是否未经提示就医直接开具处方/剂量 |")
        md.append("| 总体良好 (overall_good) | [ ] 是 / [ ] 否 | [ ] 是 / [ ] 否 | 综合质量评价 |")
        md.append("| **评审员备注** | ________________ | ________________ | ________________ |\n")
        md.append("---\n")

    # 通用问题逐题并排展示 (灾难性遗忘检查)
    md.append("## 5. 通用问题逐题对比 (灾难性遗忘检查, 10 题)\n")
    for idx, (b, l) in enumerate(zip(base_gen, lora_gen), 1):
        qid = b["id"]
        q_text = b["question"]
        b_ans = b.get("model_answer", "").strip()
        l_ans = l.get("model_answer", "").strip()

        md.append(f"### [{qid}] 通用问题 {idx}: {q_text}\n")
        md.append("| 模型 | 生成回答 |")
        md.append("| :--- | :--- |")
        md.append(f"| **Base 基座** | {table_cell(b_ans)} |")
        md.append(f"| **LoRA 微调** | {table_cell(l_ans)} |\n")
        md.append("**人工检查项**:\n")
        md.append(f"- Base: [ ] 通顺自然 / [ ] 异常或遗忘 | 备注: ________________")
        md.append(f"- LoRA: [ ] 通顺自然 / [ ] 异常或遗忘 | 备注: ________________\n")

    # 写入报告
    save_path.parent.mkdir(parents=True, exist_ok=True)
    report_content = "\n".join(md)
    tmp_report = save_path.with_suffix(".tmp")
    with open(tmp_report, "w", encoding="utf-8") as f:
        f.write(report_content)
    tmp_report.replace(save_path)

    print(f"\n🎉 比较报告已成功生成: {save_path}")
    print(f"MCQ: Base {b_acc*100:.2f}% -> LoRA {l_acc*100:.2f}% (错→对: {len(wrong_to_right)}, 对→错: {len(right_to_wrong)})")


def main():
    parser = argparse.ArgumentParser(description="MiniMind LoRA 评估与对比工具")
    parser.add_argument("--check-results", action="store_true", help="只核验已存在的正式产物，不加载模型")
    # 模式选择
    parser.add_argument("--mode", type=str, choices=["base", "lora", "compare"], default="base",
                        help="运行模式: base=评估基座, lora=评估LoRA, compare=对比已生成的评估产物")
    parser.add_argument("--compare", action="store_true",
                        help="对比模式快捷开关 (等同于 --mode compare)")

    # 模型权重与架构参数
    parser.add_argument("--base_weight", type=str, default="out/full_sft_768.pth",
                        help="基座模型权重路径或名称前缀")
    parser.add_argument("--lora_weight", type=str, default="out/lora_medical_768.pth",
                        help="LoRA 权重路径或名称前缀 (仅在 mode=lora 时使用)")
    parser.add_argument("--model_dir", type=str, default="model",
                        help="Tokenizer 及模型架构目录")
    parser.add_argument("--hidden_size", type=int, default=768, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="隐藏层数量")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="是否使用 MoE 架构")

    # 题集与数据路径
    parser.add_argument("--eval_dir", type=str, default="experiments/lora_medical_20260909/eval/v1",
                        help="锁定题集目录")
    parser.add_argument("--val_path", type=str, default="experiments/lora_medical_20260909/data/v2/val.jsonl",
                        help="验证集路径 (用于计算 val_loss)")
    parser.add_argument("--val_loss", "--val-loss", action="store_true",
                        help="是否在评估时同步计算验证集加权平均交叉熵 Loss")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="验证集 Loss 计算批次大小")

    # 运行与设备参数
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    parser.add_argument("--dtype", type=str,
                        default="bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16",
                        help="推理混合精度类型 (bfloat16 / float16 / float32)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="重复惩罚；1.0 表示关闭，医学 LoRA 可先尝试 1.1")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0,
                        help="禁止重复的 token n-gram 长度；0 表示关闭，建议先测试 3 或 4")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制每类题目数量 N (用于云端快速冒烟)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="评估结果输出目录 (若未指定且指定了 limit，默认写到 smoke 独立目录)")

    # 对比参数
    parser.add_argument("--base_dir", type=str,
                        default="experiments/lora_medical_20260909/eval_results/base",
                        help="对比时 Base 产物目录")
    parser.add_argument("--lora_dir", type=str,
                        default="experiments/lora_medical_20260909/eval_results/lora",
                        help="对比时 LoRA 产物目录")
    parser.add_argument("--save_report", type=str,
                        default="experiments/lora_medical_20260909/eval_results/compare_report.md",
                        help="Markdown 对比报告保存路径")
    parser.add_argument("--force", action="store_true",
                        help="允许比较冒烟结果；不绕过题集、权重和运行配置校验")

    args = parser.parse_args()

    if args.check_results:
        summary = validate_results(args.out_dir, args.eval_dir, args.mode)
        identity = summary["identity"]
        if identity.get("base_weight_md5") != md5_file(args.base_weight):
            raise ValueError("基座权重已变更")
        if args.mode == "lora" and identity.get("lora_weight_md5") != md5_file(args.lora_weight):
            raise ValueError("LoRA 权重已变更")
        if args.val_loss and (summary.get("val_loss") is None or identity.get("val_path_md5") != md5_file(args.val_path)):
            raise ValueError("验证集或 val_loss 不匹配")
        if summary.get("runtime_env") != {"device": args.device, "dtype": args.dtype}:
            raise ValueError("运行设备或 dtype 已变更")
        if summary.get("generation_config", {}).get("mcq_parser_version") != 2:
            raise ValueError("评分版本已变更")
        print("完整结果核验通过")
        return

    # 处理快捷标志
    if args.compare:
        args.mode = "compare"

    if args.mode == "compare":
        run_comparison(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
