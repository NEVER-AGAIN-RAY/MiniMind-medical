#!/usr/bin/env python3
"""Evaluate a triage LoRA with generation and length-normalised forced-choice scoring."""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.lora_triage_20260917.triage_dataset import (
    LABELS, TriageDataset, build_prompt_str, build_target_str,
)
from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="评测中文医疗科室分诊 LoRA")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--rank", type=int, default=None,
        help="覆盖 LoRA rank；默认从 checkpoint 的 A 矩阵形状推断，避免与训练时的 rank 不一致",
    )
    return parser.parse_args()


def infer_rank(checkpoint: Path) -> int:
    """从 checkpoint 自身推断 rank：A 矩阵形状为 (rank, in_features)。"""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    ranks = {tuple(v.shape)[0] for k, v in state.items() if k.endswith("lora.A.weight")}
    if len(ranks) != 1:
        raise ValueError(f"无法从 {checkpoint} 推断唯一的 rank: {sorted(ranks)}")
    return ranks.pop()


def candidate_score(model, prompt_ids, target_ids, device) -> tuple[float, float]:
    """返回 (log-prob 之和, 每 token 平均 log-prob)。

    六个科室名的 token 数不等（内科 4 / 妇产科 6 / 肿瘤科 7），求和口径会系统性
    压低长标签——每多一个 token 就多乘一个小于 1 的概率。因此这里一并返回平均口径，
    由调用方以平均口径为主指标，求和口径仅作对照。
    """
    ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=device)
    logits = model(ids).logits[:, :-1, :]
    labels = ids[:, 1:]
    start = len(prompt_ids) - 1
    relevant_logits = logits[:, start:, :]
    relevant_labels = labels[:, start:]
    token_logp = F.log_softmax(relevant_logits.float(), dim=-1).gather(
        -1, relevant_labels.unsqueeze(-1)
    ).squeeze(-1)
    total = float(token_logp.sum().item())
    return total, total / len(target_ids)


def parse_generated(text: str):
    """从生成文本中抽取科室名。

    保守口径（沿用 run_eval.py 的 extract_mcq_choice）：模型同时吐出两个不同科室时
    属于犹豫而非作答，判为无效，不能任选其一记成答对。
    六个标签互不为子串，因此可以直接用整词交替匹配。
    """
    stripped_text = text.strip()
    pattern = "|".join(LABELS)
    if len(set(re.findall(pattern, stripped_text))) > 1:
        return None
    match = re.search(rf"(?:^|[\s：:「『])({pattern})(?:$|[\s。！!，,、」』])", stripped_text)
    if match:
        return match.group(1)
    return stripped_text if stripped_text in LABELS else None


def main() -> None:
    args = parse_args()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    data_path = HERE / "data" / ("smoke" if args.split == "smoke" else "formal") / (
        "val.jsonl" if args.split == "smoke" else "test.jsonl"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / config["base_model"]["model_dir"]))
    dataset = TriageDataset(data_path, tokenizer, config["sequence_config"]["max_seq_len"])
    base = config["base_model"]
    model_config = MiniMindConfig(
        hidden_size=base["hidden_size"], num_hidden_layers=base["num_hidden_layers"],
        use_moe=bool(base["use_moe"]),
    )
    model = MiniMindForCausalLM(model_config)
    weights = torch.load(ROOT / base["weight_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    lora_rank = args.rank if args.rank is not None else infer_rank(checkpoint)
    apply_lora(model, rank=lora_rank)
    load_lora(model, str(checkpoint))
    model.to(args.device).eval()

    # 六个候选标签的 token 序列每题都一样，预先编码一次。
    candidate_ids = {
        label: tokenizer(build_target_str(label, tokenizer), add_special_tokens=False).input_ids
        for label in LABELS
    }

    total = min(len(dataset), args.limit) if args.limit else len(dataset)
    correct = {"mean": 0, "sum": 0, "generation": 0}
    compliant = 0
    agreement = 0
    confusion = {truth: {pred: 0 for pred in LABELS} for truth in LABELS}
    generation_predictions = Counter()
    mean_predictions = Counter()
    rows = []
    with torch.inference_mode():
        for index in range(total):
            sample = dataset.samples[index]
            truth = sample["label_text"]
            prompt = build_prompt_str(sample["ask"], tokenizer, dataset.instruction)
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            sum_scores, mean_scores = {}, {}
            for label, target_ids in candidate_ids.items():
                sum_scores[label], mean_scores[label] = candidate_score(
                    model, prompt_ids, target_ids, args.device
                )
            forced_mean = max(mean_scores, key=mean_scores.get)
            forced_sum = max(sum_scores, key=sum_scores.get)
            correct["mean"] += int(forced_mean == truth)
            correct["sum"] += int(forced_sum == truth)
            agreement += int(forced_mean == forced_sum)
            # 混淆矩阵以主指标（平均口径）为准。
            confusion[truth][forced_mean] += 1
            mean_predictions[forced_mean] += 1

            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
            generated_ids = model.generate(
                input_tensor, max_new_tokens=config["evaluation_config"]["max_new_tokens"],
                do_sample=False, eos_token_id=tokenizer.eos_token_id,
            )[0, len(prompt_ids):]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generated = parse_generated(generated_text)
            compliant += int(generated is not None)
            correct["generation"] += int(generated == truth)
            generation_predictions[generated or "INVALID"] += 1
            rows.append({
                "id": sample.get("id", index), "truth": truth,
                "forced": forced_mean, "forced_sum": forced_sum,
                "mean_scores": mean_scores, "sum_scores": sum_scores,
                "generated": generated, "generated_text": generated_text,
            })
            if (index + 1) % 50 == 0 or index + 1 == total:
                print(f"evaluated={index + 1}/{total}", flush=True)

    per_class = {}
    f1_values = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[t][label] for t in LABELS if t != label)
        fn = sum(confusion[label][p] for p in LABELS if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)

    # 基线阈值随实际评测条数计算：--limit 会改变 n，进而改变判定门槛。
    chance = 1.0 / len(LABELS)
    half_width = 1.96 * math.sqrt(chance * (1 - chance) / total)
    threshold = chance + half_width
    primary_accuracy = correct["mean"] / total
    report = {
        "checkpoint": str(checkpoint), "data": str(data_path), "samples": total,
        "lora_rank": lora_rank,
        "num_classes": len(LABELS),
        "candidate_scoring_accuracy_mean": primary_accuracy,
        "candidate_scoring_accuracy_sum": correct["sum"] / total,
        "generation_accuracy": correct["generation"] / total,
        "scoring_agreement": agreement / total,
        "scoring_note": "主指标是 candidate_scoring_accuracy_mean（每 token 平均 log-prob）。"
                        "六个科室名 token 数不等，求和口径会系统性压低长标签，仅作对照。",
        "baseline": {
            "majority_class_accuracy": round(chance, 4),
            "random_expected_accuracy": round(chance, 4),
            "random_95ci": [round(chance - half_width, 4), round(chance + half_width, 4)],
            "min_accuracy_to_beat_random": round(threshold, 4),
            "beats_random": primary_accuracy > threshold,
            "note": "candidate_scoring_accuracy_mean 需高于 min_accuracy_to_beat_random 才算显著优于随机；"
                    "不要直接与 1/6 比较。",
        },
        "format_compliance_rate": compliant / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "confusion_matrix": confusion,
        "predicted_label_distribution": dict(mean_predictions),
        "generation_predictions": dict(generation_predictions),
        "per_class": per_class,
    }
    output = args.output or checkpoint.parent / f"eval_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"report": report, "predictions": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"EVALUATION_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
