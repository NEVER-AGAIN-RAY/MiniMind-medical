#!/usr/bin/env python
"""用同一批 50 题比较精简知识目标下的 LoRA rank16/rank32/全参数记忆能力。"""

import argparse
import hashlib
import json
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiments.lora_medical_mcq_pilot.mcq_dataset import (
    CONCISE_KNOWLEDGE_INSTRUCTION_PROMPT,
    MedicalMCQDataset,
    format_mcq_prompt,
)
from model.model_lora import apply_lora, save_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_prompt(tokenizer, sample: dict) -> tuple[str, list[int]]:
    prompt_text = format_mcq_prompt(
        sample["question"], sample["options"], CONCISE_KNOWLEDGE_INSTRUCTION_PROMPT
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    return prompt, tokenizer(prompt, add_special_tokens=False).input_ids


@torch.inference_mode()
def candidate_accuracy(model, tokenizer, samples: list[dict], device: str, amp_ctx) -> tuple[float, list[dict]]:
    """比较模型从回答起点生成各选项文字的平均 token NLL。"""
    was_training = model.training
    model.eval()
    details = []
    correct = 0
    pad_id = tokenizer.pad_token_id or 0

    for sample in samples:
        _, prompt_ids = make_prompt(tokenizer, sample)
        rows, label_rows, letters = [], [], []
        for letter, text in sorted(sample["options"].items()):
            target_ids = tokenizer(text.strip(), add_special_tokens=False).input_ids
            rows.append(prompt_ids + target_ids)
            label_rows.append([-100] * len(prompt_ids) + target_ids)
            letters.append(letter)
        max_len = max(map(len, rows))
        attention = []
        for i in range(len(rows)):
            pad = max_len - len(rows[i])
            attention.append([1] * len(rows[i]) + [0] * pad)
            rows[i] += [pad_id] * pad
            label_rows[i] += [-100] * pad

        input_ids = torch.tensor(rows, dtype=torch.long, device=device)
        labels = torch.tensor(label_rows, dtype=torch.long, device=device)
        mask = torch.tensor(attention, dtype=torch.long, device=device)
        with amp_ctx:
            logits = model(input_ids, attention_mask=mask).logits[:, :-1, :]
        shifted = labels[:, 1:]
        token_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            shifted.reshape(-1), ignore_index=-100, reduction="none",
        ).view(len(rows), -1)
        valid = shifted.ne(-100)
        scores = (token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
        predicted = letters[int(scores.argmin().item())]
        correct += int(predicted == sample["answer"])
        details.append({
            "id": sample["id"],
            "gold": sample["answer"],
            "predicted": predicted,
            "scores": {k: round(float(v), 6) for k, v in zip(letters, scores.cpu())},
        })

    if was_training:
        model.train()
    return correct / len(samples), details


@torch.inference_mode()
def generation_check(model, tokenizer, samples: list[dict], device: str) -> dict:
    model.eval()
    rows = []
    prefix_correct = 0
    natural = 0
    for sample in samples:
        _, prompt_ids = make_prompt(tokenizer, sample)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output = model.generate(
            inputs=inputs, max_new_tokens=64, do_sample=False,
            temperature=1.0, top_p=1.0, top_k=0,
            repetition_penalty=1.0, eos_token_id=tokenizer.eos_token_id,
        )[0, len(prompt_ids):].tolist()
        ended = tokenizer.eos_token_id in output
        if ended:
            output = output[:output.index(tokenizer.eos_token_id)]
            natural += 1
        text = tokenizer.decode(output, skip_special_tokens=True).strip()
        answer_text = sample["options"][sample["answer"]].strip()
        ok = text.startswith(answer_text)
        prefix_correct += int(ok)
        rows.append({
            "id": sample["id"], "gold": sample["answer"],
            "answer_text": answer_text, "output": text,
            "answer_text_prefix_correct": ok,
            "termination": "natural_eos" if ended else "max_tokens_truncated",
        })
    return {
        "answer_text_prefix_accuracy": prefix_correct / len(samples),
        "natural_termination_rate": natural / len(samples),
        "items": rows,
    }


def load_base(base_weight: Path, device: str) -> MiniMindForCausalLM:
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=768, num_hidden_layers=8, use_moe=False,
    ))
    state = torch.load(base_weight, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device)


def run_one(args, scope: str, rank: int | None, tokenizer, samples, loader, amp_ctx, output_dir: Path) -> dict:
    seed_everything(args.seed)
    model = load_base(args.base_weight, args.device)
    if scope == "lora":
        apply_lora(model, rank=rank)
        params = []
        for name, param in model.named_parameters():
            param.requires_grad = "lora" in name
            if param.requires_grad:
                params.append(param)
        lr = args.lora_lr
        name = f"lora_rank{rank}"
    else:
        for param in model.parameters():
            param.requires_grad = True
        params = list(model.parameters())
        lr = args.full_lr
        name = "full_parameter"

    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    history = []
    checkpoints = {0, 10, 25, args.epochs}
    initial_acc, _ = candidate_accuracy(model, tokenizer, samples, args.device, amp_ctx)
    history.append({"epoch": 0, "candidate_accuracy": initial_acc})
    print(f"\n[{name}] trainable={trainable:,}/{total:,}, initial candidate accuracy={initial_acc:.1%}", flush=True)

    started = time.time()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batches = 0
        for input_ids, labels in loader:
            global_step += 1
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            progress = global_step / total_steps
            current_lr = lr * (0.1 + 0.45 * (1 + np.cos(np.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = float(current_lr)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                loss = model(input_ids, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1

        if epoch in checkpoints:
            acc, _ = candidate_accuracy(model, tokenizer, samples, args.device, amp_ctx)
            row = {"epoch": epoch, "train_loss": loss_sum / batches, "candidate_accuracy": acc}
            history.append(row)
            print(f"[{name}] epoch={epoch}, loss={row['train_loss']:.4f}, candidate accuracy={acc:.1%}", flush=True)

    final_acc, candidate_details = candidate_accuracy(model, tokenizer, samples, args.device, amp_ctx)
    generated = generation_check(model, tokenizer, samples, args.device)
    if scope == "lora":
        save_lora(model, str(output_dir / f"{name}_final.pth"))

    result = {
        "name": name, "scope": scope, "rank": rank,
        "epochs": args.epochs, "learning_rate": lr,
        "trainable_parameters": trainable, "total_parameters": total,
        "elapsed_seconds": round(time.time() - started, 2),
        "history": history,
        "final_candidate_accuracy": final_acc,
        "candidate_details": candidate_details,
        "generation": generated,
    }
    with (output_dir / f"{name}_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base_weight", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, default=ROOT / "model")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_lr", type=float, default=2e-4)
    parser.add_argument("--full_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    samples = read_jsonl(args.data)
    if len(samples) != 50:
        raise ValueError(f"诊断实验固定要求 50 题，实际 {len(samples)}")
    dataset = MedicalMCQDataset(
        args.data, tokenizer, max_length=args.max_length,
        instruction=CONCISE_KNOWLEDGE_INSTRUCTION_PROMPT,
        target_mode="concise_knowledge",
    )
    # 先逐条触发长度与知识句校验，再开始昂贵训练。
    lengths = []
    for i in range(len(dataset)):
        _, labels = dataset[i]
        lengths.append(int(labels.ne(-100).sum()))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if "cuda" in args.device else nullcontext()

    results = []
    for scope, rank in [("lora", 16), ("lora", 32), ("full", None)]:
        results.append(run_one(args, scope, rank, tokenizer, samples, loader, amp_ctx, args.output_dir))

    summary = {
        "experiment": "concise_knowledge_overfit50_diagnosis",
        "created_at": datetime.now().isoformat(),
        "data": str(args.data), "data_md5": md5_file(args.data),
        "base_weight": str(args.base_weight), "base_weight_md5": md5_file(args.base_weight),
        "sample_count": len(samples), "max_length": args.max_length,
        "target_supervised_tokens": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
        "target_definition": "正确选项文字 + 原解析中第一句包含该答案文字的医学陈述；不含答案字母，不监督完整错误选项解析",
        "results": [{
            "name": r["name"], "trainable_parameters": r["trainable_parameters"],
            "final_candidate_accuracy": r["final_candidate_accuracy"],
            "answer_text_prefix_accuracy": r["generation"]["answer_text_prefix_accuracy"],
            "natural_termination_rate": r["generation"]["natural_termination_rate"],
            "elapsed_seconds": r["elapsed_seconds"],
        } for r in results],
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nFINAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
