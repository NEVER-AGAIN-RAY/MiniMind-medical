#!/usr/bin/env python
"""评测精简知识 LoRA：候选内容评分 + 确定性自由生成。"""

import argparse
import json
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from experiments.lora_medical_mcq_pilot.diagnose_concise_overfit import (
    candidate_accuracy,
    generation_check,
    load_base,
    md5_file,
    read_jsonl,
)
from model.model_lora import apply_lora, load_lora


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base_weight", type=Path, required=True)
    parser.add_argument("--lora_weight", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--model_dir", type=Path, default=Path("model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖已有评测结果: {args.output}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    samples = read_jsonl(args.data)
    model = load_base(args.base_weight, args.device)
    apply_lora(model, rank=args.rank)
    load_lora(model, str(args.lora_weight))
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if "cuda" in args.device else nullcontext()
    accuracy, details = candidate_accuracy(model, tokenizer, samples, args.device, amp_ctx)
    generation = generation_check(model, tokenizer, samples, args.device)
    result = {
        "created_at": datetime.now().isoformat(),
        "data": str(args.data), "data_md5": md5_file(args.data),
        "base_weight": str(args.base_weight), "base_weight_md5": md5_file(args.base_weight),
        "lora_weight": str(args.lora_weight), "lora_weight_md5": md5_file(args.lora_weight),
        "rank": args.rank, "sample_count": len(samples),
        "candidate_accuracy": accuracy,
        "answer_text_prefix_accuracy": generation["answer_text_prefix_accuracy"],
        "natural_termination_rate": generation["natural_termination_rate"],
        "candidate_details": details,
        "generation": generation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in (
        "sample_count", "candidate_accuracy", "answer_text_prefix_accuracy", "natural_termination_rate"
    )}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
