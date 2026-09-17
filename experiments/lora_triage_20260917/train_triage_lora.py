#!/usr/bin/env python3
"""Train a MiniMind LoRA adapter for balanced Chinese medical department triage."""

import argparse
import hashlib
import json
import math
import os
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

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiments.lora_triage_20260917.triage_dataset import TriageDataset
from model.model_lora import apply_lora, save_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import evaluate_val_loss, get_lr, setup_seed


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_lora_atomic(model, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_lora(model, str(temporary))
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind 中文医疗科室分诊 LoRA 训练")
    parser.add_argument("--smoke", action="store_true", help="使用 60/30 冒烟数据与冒烟超参")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--rank", type=int, default=None,
        help="覆盖 LoRA rank，用于 rank 扫描；默认取 config.json 的 lora_config.rank",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--train-file", type=Path, default=None,
        help="覆盖训练集路径，用于 data/scale/ 下的规模曲线子集；验证集始终使用 formal/val.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    mode = "smoke" if args.smoke else "formal"
    hp = dict(config["smoke_hyperparameters" if args.smoke else "training_hyperparameters"])
    if args.epochs is not None:
        hp["epochs"] = args.epochs
    if args.batch_size is not None:
        hp["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        hp["learning_rate"] = args.learning_rate

    run_dir = args.run_dir or ROOT / config["run_paths"][f"{mode}_dir"]
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    artifacts = [run_dir / name for name in ("best_lora.pth", "final_lora.pth", "train_summary.json")]
    if any(path.exists() for path in artifacts) and not args.overwrite:
        raise FileExistsError(f"运行目录已有训练产物: {run_dir}；请换目录或显式使用 --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = config["data_config"]["seed"]
    setup_seed(seed)
    random.seed(seed)
    max_length = config["sequence_config"]["max_seq_len"]
    if args.train_file is not None:
        train_path = args.train_file if args.train_file.is_absolute() else ROOT / args.train_file
        # 规模曲线的各个点必须共用同一个验证集，否则 val_loss 在点之间不可比。
        val_path = HERE / "data" / "formal" / "val.jsonl"
    else:
        train_path = HERE / "data" / mode / "train.jsonl"
        val_path = HERE / "data" / mode / "val.jsonl"
    base_weight = ROOT / config["base_model"]["weight_path"]
    for required in (train_path, val_path, base_weight):
        if not required.exists():
            raise FileNotFoundError(required)

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / config["base_model"]["model_dir"]))
    train_ds = TriageDataset(train_path, tokenizer, max_length=max_length)
    val_ds = TriageDataset(val_path, tokenizer, max_length=max_length)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=hp["batch_size"], shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=hp["batch_size"], shuffle=False,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
    )

    base = config["base_model"]
    model_config = MiniMindConfig(
        hidden_size=base["hidden_size"], num_hidden_layers=base["num_hidden_layers"],
        use_moe=bool(base["use_moe"]),
    )
    model = MiniMindForCausalLM(model_config)
    state = torch.load(base_weight, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"基础权重不兼容: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    lora_rank = args.rank if args.rank is not None else config["lora_config"]["rank"]
    apply_lora(model, rank=lora_rank)
    lora_params = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora" in name
        if parameter.requires_grad:
            lora_params.append(parameter)
    model.to(args.device)

    dtype_name = hp["dtype"]
    amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype)
        if args.device.startswith("cuda") else nullcontext()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda") and dtype_name == "float16")
    optimizer = AdamW(lora_params, lr=hp["learning_rate"], weight_decay=hp.get("weight_decay", 0.01))
    accumulation = hp["accumulation_steps"]
    total_batches = hp["epochs"] * len(train_loader)
    total_updates = hp["epochs"] * math.ceil(len(train_loader) / accumulation)

    # 六类平衡是评测口径（随机基线严格等于 1/6）的前提，训练前当场校验，
    # 不要等准确率异常了再回头查切分。
    for name, dataset in (("train", train_ds), ("val", val_ds)):
        distribution = dataset.label_distribution()
        if len(set(distribution.values())) != 1:
            raise ValueError(f"{name} 集类别不平衡: {distribution}")

    initial_val_loss = evaluate_val_loss(model, val_loader, args.device, autocast_ctx)
    best_val_loss = initial_val_loss
    best_step = 0
    save_lora_atomic(model, run_dir / "best_lora.pth")
    started = time.time()
    global_batch = 0
    update_step = 0
    history = [{"step": 0, "val_loss": initial_val_loss}]
    print(
        f"MODE={mode} rank={lora_rank} trainable={sum(p.numel() for p in lora_params)} "
        f"train={len(train_ds)} val={len(val_ds)} initial_val_loss={initial_val_loss:.6f}",
        flush=True,
    )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(hp["epochs"]):
        model.train()
        for batch_index, (input_ids, labels) in enumerate(train_loader, start=1):
            global_batch += 1
            input_ids = input_ids.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            with autocast_ctx:
                output = model(input_ids, labels=labels)
                loss = (output.loss + output.aux_loss) / accumulation
            scaler.scale(loss).backward()
            is_boundary = batch_index % accumulation == 0 or batch_index == len(train_loader)
            if is_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, hp["grad_clip"])
                update_step += 1
                lr = get_lr(update_step, total_updates, hp["learning_rate"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if global_batch % hp["log_interval"] == 0 or global_batch == total_batches:
                print(
                    f"epoch={epoch + 1}/{hp['epochs']} batch={global_batch}/{total_batches} "
                    f"loss={float(loss.detach()) * accumulation:.6f} lr={optimizer.param_groups[0]['lr']:.8f}",
                    flush=True,
                )
            if is_boundary and (update_step % hp["val_interval"] == 0 or global_batch == total_batches):
                val_loss = evaluate_val_loss(model, val_loader, args.device, autocast_ctx)
                history.append({"step": update_step, "val_loss": val_loss})
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss, best_step = val_loss, update_step
                    save_lora_atomic(model, run_dir / "best_lora.pth")
                print(f"validation step={update_step} val_loss={val_loss:.6f} improved={improved}", flush=True)
                model.train()

    save_lora_atomic(model, run_dir / "final_lora.pth")
    summary = {
        "status": "complete",
        "mode": mode,
        "created_at": datetime.now().isoformat(),
        "elapsed_seconds": round(time.time() - started, 2),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_label_distribution": train_ds.label_distribution(),
        "val_label_distribution": val_ds.label_distribution(),
        "hyperparameters": hp,
        "lora_rank": lora_rank,
        "trainable_params": sum(p.numel() for p in lora_params),
        "initial_val_loss": initial_val_loss,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "history": history,
        "base_weight": {"path": str(base_weight), "md5": md5_file(base_weight)},
        "train_data_path": str(train_path),
        "train_data_md5": md5_file(train_path),
        "val_data_md5": md5_file(val_path),
        "best_lora_md5": md5_file(run_dir / "best_lora.pth"),
        "final_lora_md5": md5_file(run_dir / "final_lora.pth"),
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TRAINING_COMPLETE best_step={best_step} best_val_loss={best_val_loss:.6f} run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
