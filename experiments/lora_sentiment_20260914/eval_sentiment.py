#!/usr/bin/env python3
"""Evaluate a sentiment LoRA with generation and forced-choice scoring."""

import argparse
import json
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

from experiments.lora_sentiment_20260914.sentiment_dataset import (
    LABELS, SentimentDataset, build_prompt_str, build_target_str,
)
from model.model_lora import apply_lora, load_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="评测中文情感 LoRA")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def candidate_score(model, prompt_ids, target_ids, device) -> float:
    ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=device)
    logits = model(ids).logits[:, :-1, :]
    labels = ids[:, 1:]
    start = len(prompt_ids) - 1
    relevant_logits = logits[:, start:, :]
    relevant_labels = labels[:, start:]
    token_logp = F.log_softmax(relevant_logits.float(), dim=-1).gather(
        -1, relevant_labels.unsqueeze(-1)
    ).squeeze(-1)
    return float(token_logp.sum().item())


def parse_generated(text: str):
    match = re.search(r"(?:^|[\s：:「『])(正面|负面)(?:$|[\s。！!，,」』])", text.strip())
    if match:
        return match.group(1)
    stripped = text.strip()
    return stripped if stripped in LABELS else None


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
    dataset = SentimentDataset(data_path, tokenizer, config["sequence_config"]["max_seq_len"])
    base = config["base_model"]
    model_config = MiniMindConfig(
        hidden_size=base["hidden_size"], num_hidden_layers=base["num_hidden_layers"],
        use_moe=bool(base["use_moe"]),
    )
    model = MiniMindForCausalLM(model_config)
    weights = torch.load(ROOT / base["weight_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    apply_lora(model, rank=config["lora_config"]["rank"])
    load_lora(model, str(checkpoint))
    model.to(args.device).eval()

    total = min(len(dataset), args.limit) if args.limit else len(dataset)
    forced_correct = 0
    generated_correct = 0
    compliant = 0
    forced_confusion = {truth: {pred: 0 for pred in LABELS} for truth in LABELS}
    generation_predictions = Counter()
    rows = []
    with torch.inference_mode():
        for index in range(total):
            sample = dataset.samples[index]
            truth = sample["label_text"]
            prompt = build_prompt_str(sample["review"], tokenizer, dataset.instruction)
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            scores = {}
            for label in LABELS:
                target_ids = tokenizer(build_target_str(label, tokenizer), add_special_tokens=False).input_ids
                scores[label] = candidate_score(model, prompt_ids, target_ids, args.device)
            forced = max(scores, key=scores.get)
            forced_correct += int(forced == truth)
            forced_confusion[truth][forced] += 1

            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
            generated_ids = model.generate(
                input_tensor, max_new_tokens=config["evaluation_config"]["max_new_tokens"],
                do_sample=False, eos_token_id=tokenizer.eos_token_id,
            )[0, len(prompt_ids):]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generated = parse_generated(generated_text)
            compliant += int(generated is not None)
            generated_correct += int(generated == truth)
            generation_predictions[generated or "INVALID"] += 1
            rows.append({
                "id": sample.get("id", index), "truth": truth, "forced": forced,
                "forced_scores": scores, "generated": generated,
                "generated_text": generated_text,
            })
            if (index + 1) % 50 == 0 or index + 1 == total:
                print(f"evaluated={index + 1}/{total}", flush=True)

    per_class = {}
    f1_values = []
    for label in LABELS:
        tp = forced_confusion[label][label]
        fp = sum(forced_confusion[t][label] for t in LABELS if t != label)
        fn = sum(forced_confusion[label][p] for p in LABELS if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    report = {
        "checkpoint": str(checkpoint), "data": str(data_path), "samples": total,
        "candidate_scoring_accuracy": forced_correct / total,
        "generation_accuracy": generated_correct / total,
        "format_compliance_rate": compliant / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "confusion_matrix": forced_confusion,
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
