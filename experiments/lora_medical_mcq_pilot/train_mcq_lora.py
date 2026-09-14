#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
医学单选题 LoRA 训练脚本 (train_mcq_lora.py) - 严格溯源与精确更新版

修复项：
1. [Issue 2 修复]：执行“更新 → 验证 → 保存最佳”顺序。
   - 每轮尾批（未凑满 accumulation_steps）在步循环末尾立即完成梯度补更新并补偿梯度尺度；
   - 保证最新的梯度更新在周期性验证（尤其是 global_step == total_steps 终步）前已施加到模型；
   - 最终轮次更新后的最新权重能即刻被验证，并有机会被选拔为 best_lora.pth。
2. [Issue 3 修复]：完整记录基座与训练/验证数据的 MD5 指纹，实现可审计溯源。
   - effective_config 与 train_summary 中完整记录 base_weight_md5、train_data_md5、val_data_md5 及 manifest_md5。
3. 全面防覆盖拦截与 Step 0 检查点初值恒存保证。
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 sys.path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiments.lora_medical_mcq_pilot.mcq_dataset import (
    ANSWER_RATIONALE_INSTRUCTION_PROMPT,
    ANSWER_TEXT_INSTRUCTION_PROMPT,
    CONCISE_KNOWLEDGE_INSTRUCTION_PROMPT,
    MedicalMCQDataset,
)
from model.model_lora import apply_lora, save_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import evaluate_val_loss, get_lr, setup_seed

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


class CheckpointTracker:
    """
    负责 LoRA 检查点的初始化、选优追踪及最终归档。
    保障：
    1. Step 0 初始权重必定落盘保存为 step_0_lora.pth 与 best_lora.pth 保底候选。
    2. 严格按“验证 Loss 相比历史最优降低”更新 best_lora.pth 与 improved 标志。
    3. 若整个训练未发生改善，如实标记 improved=False，保留 Step 0 作为最佳检查点，不伪造改善。
    4. 训练结束时归档 final_lora.pth 并导出规范化的检查点元数据。
    """
    def __init__(self, ckpt_dir: Path, model, init_val_loss: float):
        self.ckpt_dir = Path(ckpt_dir)
        self.model = model
        self.init_val_loss = float(init_val_loss)
        self.best_val_loss = float(init_val_loss)
        self.best_step = 0
        self.improved = False

        self.step_0_ckpt = self.ckpt_dir / "step_0_lora.pth"
        self.best_ckpt = self.ckpt_dir / "best_lora.pth"
        self.final_ckpt = self.ckpt_dir / "final_lora.pth"

        save_lora(self.model, str(self.step_0_ckpt))
        save_lora(self.model, str(self.best_ckpt))
        print(f"✅ 初始 LoRA 检查点已落盘: {self.step_0_ckpt} 并初始化为候选: {self.best_ckpt}")

    def step(self, global_step: int, val_loss: float) -> bool:
        val_loss = float(val_loss)
        if val_loss < self.best_val_loss:
            self.improved = True
            self.best_val_loss = val_loss
            self.best_step = global_step
            save_lora(self.model, str(self.best_ckpt))
            return True
        return False

    def finalize(self, model=None) -> dict:
        m = model or self.model
        save_lora(m, str(self.final_ckpt))
        print(f"✅ 末步权重已落盘: {self.final_ckpt}")

        selected_ckpt = self.best_ckpt
        assert selected_ckpt.exists(), f"关键选定检查点缺失: {selected_ckpt}"
        selected_ckpt_md5 = md5_file(selected_ckpt)

        status_message = (
            f"训练在 Step {self.best_step} 取得更优 Loss ({self.best_val_loss:.4f} < 初始 {self.init_val_loss:.4f})，选定为最佳检查点。"
            if self.improved else
            f"训练期间验证 Loss 未低于初始值 ({self.best_val_loss:.4f} >= 初始 {self.init_val_loss:.4f})，如实保留 Step 0 初始 LoRA 为选定检查点。"
        )

        return {
            "improved": self.improved,
            "best_step": self.best_step,
            "initial_val_loss": round(self.init_val_loss, 4),
            "best_val_loss": round(self.best_val_loss, 4),
            "selected_checkpoint": {
                "path": str(selected_ckpt),
                "filename": selected_ckpt.name,
                "md5": selected_ckpt_md5,
                "source_step": self.best_step,
            },
            "all_checkpoints": {
                "step_0_lora": {"path": str(self.step_0_ckpt), "md5": md5_file(self.step_0_ckpt)},
                "best_lora": {"path": str(self.best_ckpt), "md5": selected_ckpt_md5},
                "final_lora": {"path": str(self.final_ckpt), "md5": md5_file(self.final_ckpt)},
            },
            "message": status_message,
        }


def train(args):
    config = load_config()

    # 1. 确定运行模式与独立运行目录
    is_smoke = bool(args.smoke)
    run_paths = config["run_paths"]

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
    else:
        run_dir_rel = run_paths["smoke_dir"] if is_smoke else run_paths["formal_dir"]
        run_dir = ROOT / run_dir_rel

    ckpt_dir = run_dir / "checkpoints"
    summary_file = ckpt_dir / "train_summary.json"

    # 防覆盖与中断重跑检查：检查所有可能存在的中间/结束产物
    artifact_candidates = [
        summary_file,
        ckpt_dir / "step_0_lora.pth",
        ckpt_dir / "best_lora.pth",
        ckpt_dir / "final_lora.pth",
        run_dir / "config_effective.json",
    ]
    found_artifacts = [p for p in artifact_candidates if p.exists()]
    if found_artifacts and not args.overwrite:
        raise FileExistsError(
            f"运行目录已存在先前执行留存的产物 (检测到: {found_artifacts[0]})！\n"
            f"为避免中断重跑时意外覆盖中途检查点或历史记录，默认拒绝执行。\n"
            f"请指定全新的 --run_dir，或显式追加 --overwrite 确认覆盖现有产物。"
        )

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 2. 合并生效配置参数
    hparams_key = "smoke_hyperparameters" if is_smoke else "training_hyperparameters"
    base_hparams = config[hparams_key]
    data_cfg = config["data_config"]
    base_cfg = config["base_model"]
    lora_cfg = config["lora_config"]
    seq_cfg = config["sequence_config"]

    epochs = args.epochs or base_hparams["epochs"]
    batch_size = args.batch_size or base_hparams["batch_size"]
    learning_rate = args.learning_rate or base_hparams["learning_rate"]
    weight_decay = args.weight_decay or base_hparams.get("weight_decay", 0.01)
    accumulation_steps = args.accumulation_steps or base_hparams["accumulation_steps"]
    grad_clip = args.grad_clip or base_hparams["grad_clip"]
    log_interval = args.log_interval or base_hparams["log_interval"]
    val_interval = args.val_interval or base_hparams["val_interval"]
    dtype_str = args.dtype or base_hparams["dtype"]
    max_seq_len = args.max_seq_len or seq_cfg["max_seq_len"]
    seed = args.seed or data_cfg.get("seed", 42)
    lora_rank = args.lora_rank or lora_cfg["rank"]
    target_mode = args.target_mode

    # 确定数据路径 (冒烟与正式严格隔离)
    exp_root = ROOT / run_paths["experiments_root"]
    default_data_subdir = "smoke" if is_smoke else "formal"

    train_path = Path(args.train_path) if args.train_path else (exp_root / "data" / default_data_subdir / "train.jsonl")
    val_path = Path(args.val_path) if args.val_path else (exp_root / "data" / default_data_subdir / "val.jsonl")

    if not train_path.exists():
        raise FileNotFoundError(f"训练数据不存在: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"验证数据不存在: {val_path}")

    # [Issue 3 修复] 提取并校验数据指纹
    train_data_md5 = md5_file(train_path)
    val_data_md5 = md5_file(val_path)
    manifest_p = exp_root / "data" / default_data_subdir / "manifest.json"
    manifest_md5 = md5_file(manifest_p) if manifest_p.exists() else None

    base_weight_path = Path(args.base_weight or (ROOT / base_cfg["weight_path"]))
    if not base_weight_path.exists():
        raise FileNotFoundError(f"基座权重不存在: {base_weight_path}")
    base_weight_md5 = md5_file(base_weight_path)

    # 3. 记录最终生效配置 (含完整指纹)
    effective_config = {
        "experiment_name": config["experiment_name"],
        "is_smoke": is_smoke,
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(),
        "base_model": {
            "weight_path": str(base_weight_path),
            "weight_md5": base_weight_md5,
            "hidden_size": base_cfg["hidden_size"],
            "num_hidden_layers": base_cfg["num_hidden_layers"],
            "use_moe": base_cfg["use_moe"],
            "model_dir": str(ROOT / base_cfg["model_dir"]),
        },
        "lora_config": {
            "rank": lora_rank,
            "target_modules": lora_cfg["target_modules"],
            "save_name": lora_cfg["save_name"],
        },
        "data": {
            "train_path": str(train_path),
            "train_md5": train_data_md5,
            "val_path": str(val_path),
            "val_md5": val_data_md5,
            "manifest_path": str(manifest_p) if manifest_p.exists() else None,
            "manifest_md5": manifest_md5,
            "instruction_prompt": data_cfg["instruction_prompt"],
            "target_mode": target_mode,
            "max_seq_len": max_seq_len,
            "seed": seed,
        },
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "accumulation_steps": accumulation_steps,
            "grad_clip": grad_clip,
            "log_interval": log_interval,
            "val_interval": val_interval,
            "dtype": dtype_str,
        },
        "device": args.device,
    }

    effective_config_file = run_dir / "config_effective.json"
    with open(effective_config_file, "w", encoding="utf-8") as f:
        json.dump(effective_config, f, ensure_ascii=False, indent=2)

    setup_seed(seed)

    # 4. 设备与精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    if dtype_str == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif dtype_str == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    print("=" * 60)
    print(f"【MiniMind 医学单选题 LoRA 训练 | 模式: {'冒烟 SMOKE' if is_smoke else '正式 FORMAL'}】")
    print(f"运行目录: {run_dir}")
    print(f"生效配置: {effective_config_file}")
    print(f"基座指纹: {base_weight_md5[:10]}... | 训练集指纹: {train_data_md5[:10]}...")
    print(f"设备: {args.device}, 精度: {dtype}")
    print("=" * 60)

    # 5. 加载 Tokenizer 与基座模型
    model_dir = Path(effective_config["base_model"]["model_dir"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    lm_config = MiniMindConfig(
        hidden_size=base_cfg["hidden_size"],
        num_hidden_layers=base_cfg["num_hidden_layers"],
        use_moe=bool(base_cfg["use_moe"]),
    )
    model = MiniMindForCausalLM(lm_config)

    print(f"加载基座模型权重: {base_weight_path}")
    base_state = torch.load(base_weight_path, map_location=args.device)
    model.load_state_dict(base_state, strict=True)

    # 6. 初始化独立 LoRA 并冻结非 LoRA 参数
    print(f"重新初始化独立 LoRA 模块 (rank={lora_rank}) ...")
    apply_lora(model, rank=lora_rank)

    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in lora_params)
    print(f"模型总参数量: {total_params / 1e6:.3f} M | LoRA 可训参数: {trainable_params / 1e6:.3f} M")

    model = model.to(device=args.device, dtype=dtype)

    # 7. 构建 Dataset 与 DataLoader
    if target_mode == "concise_knowledge":
        instruction = CONCISE_KNOWLEDGE_INSTRUCTION_PROMPT
    elif target_mode == "answer_rationale":
        instruction = ANSWER_RATIONALE_INSTRUCTION_PROMPT
    elif target_mode == "answer_text":
        instruction = ANSWER_TEXT_INSTRUCTION_PROMPT
    else:
        instruction = data_cfg["instruction_prompt"]
    effective_config["data"]["instruction_prompt"] = instruction
    with open(effective_config_file, "w", encoding="utf-8") as f:
        json.dump(effective_config, f, ensure_ascii=False, indent=2)
    train_ds = MedicalMCQDataset(
        train_path, tokenizer, max_length=max_seq_len,
        instruction=instruction, target_mode=target_mode,
    )
    val_ds = MedicalMCQDataset(
        val_path, tokenizer, max_length=max_seq_len,
        instruction=instruction, target_mode=target_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
    )

    optimizer = optim.AdamW(lora_params, lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    total_steps = epochs * len(train_loader)
    print(f"训练样本数: {len(train_ds)}, 验证样本数: {len(val_ds)}, 总步数: {total_steps}")

    # 7. 计算 Step 0 初始验证 Loss 并初始化 CheckpointTracker
    print("\n评估初始模型在验证集上的 Step 0 Loss ...")
    init_val_loss = evaluate_val_loss(model, val_loader, args.device, autocast_ctx)
    print(f"Step 0 验证 Loss: {init_val_loss:.4f}")

    tracker = CheckpointTracker(ckpt_dir=ckpt_dir, model=model, init_val_loss=init_val_loss)

    global_step = 0
    history = []
    start_time = time.time()

    # 8. [Issue 2 修复] 训练循环：严格保证“完成更新 → 验证 → 保存最佳”
    model.train()
    for epoch in range(epochs):
        epoch_start = time.time()
        accumulated_batches = 0

        for step, (input_ids, labels) in enumerate(train_loader, 1):
            global_step += 1
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)

            lr = get_lr(global_step, total_steps, learning_rate)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            with autocast_ctx:
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()
            accumulated_batches += 1

            # (1) 满 accumulation_steps 时执行常规更新
            if accumulated_batches == accumulation_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0

            # (2) [Issue 2 修复关键点] 若此步为本轮最后一步且有残留尾批，必须在此步内立即执行尾批更新！
            if step == len(train_loader) and accumulated_batches > 0:
                scaler.unscale_(optimizer)
                scale_compensation = accumulation_steps / accumulated_batches
                for p in lora_params:
                    if p.grad is not None:
                        p.grad.data.mul_(scale_compensation)
                torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0

            cur_loss = loss.item() * accumulation_steps

            if global_step % log_interval == 0 or global_step == total_steps:
                elapsed = time.time() - start_time
                print(
                    f"Epoch:[{epoch + 1}/{epochs}] Step:[{step}/{len(train_loader)}] "
                    f"GlobalStep:{global_step}/{total_steps} | Loss:{cur_loss:.4f} | "
                    f"LR:{lr:.2e} | Time:{elapsed:.1f}s"
                )

            # (3) [Issue 2 修复关键点] 验证与选优在更新完成后执行！
            # 无论是常规满批更新还是轮末尾批更新，最新的梯度均已生效至权重，可接受验证与选优
            if global_step % val_interval == 0 or global_step == total_steps:
                val_loss = evaluate_val_loss(model, val_loader, args.device, autocast_ctx)
                is_better = tracker.step(global_step=global_step, val_loss=val_loss)
                tag = "🔥 [更新最佳检查点]" if is_better else ""

                print(f"==> Step {global_step} 验证 Loss: {val_loss:.4f} (历史最优: {tracker.best_val_loss:.4f}, Step: {tracker.best_step}) {tag}")

                history.append({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": round(cur_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "lr": lr,
                    "is_best": is_better,
                })

        print(f"Epoch {epoch + 1} 结束，耗时: {time.time() - epoch_start:.1f}s")

    # 9. 归档最终检查点与写入审计记录
    ckpt_summary = tracker.finalize(model)

    summary = {
        "experiment": "lora_medical_mcq_pilot",
        "is_smoke": is_smoke,
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": round(time.time() - start_time, 2),
        "status": "COMPLETED",
        "improved": ckpt_summary["improved"],
        "best_step": ckpt_summary["best_step"],
        "initial_val_loss": ckpt_summary["initial_val_loss"],
        "best_val_loss": ckpt_summary["best_val_loss"],
        "provenance": {
            "base_weight_path": str(base_weight_path),
            "base_weight_md5": base_weight_md5,
            "train_data_path": str(train_path),
            "train_data_md5": train_data_md5,
            "val_data_path": str(val_path),
            "val_data_md5": val_data_md5,
            "data_manifest_md5": manifest_md5,
        },
        "selected_checkpoint": ckpt_summary["selected_checkpoint"],
        "all_checkpoints": ckpt_summary["all_checkpoints"],
        "effective_config": effective_config,
        "message": ckpt_summary["message"],
        "history": history,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    selected_ckpt = Path(ckpt_summary["selected_checkpoint"]["path"])
    selected_ckpt_md5 = ckpt_summary["selected_checkpoint"]["md5"]

    print("\n" + "=" * 60)
    print("🎉 训练流程顺利完成！")
    print(f"  - 运行目录:       {run_dir}")
    print(f"  - 选定检查点:     {selected_ckpt} (MD5: {selected_ckpt_md5[:10]}...)")
    print(f"  - 改善状态:       {'已改善' if ckpt_summary['improved'] else '未改善 (如实记录，保留初始候选)'}")
    print(f"  - 完整训练摘要:   {summary_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="MiniMind 医学单选题 LoRA 训练工具")
    parser.add_argument("--smoke", action="store_true", help="以冒烟模式运行 (使用 smoke 配置与数据)")
    parser.add_argument("--run_dir", type=str, default=None, help="显式指定独立运行目录")
    parser.add_argument("--train_path", type=str, default=None, help="显式指定训练集路径")
    parser.add_argument("--val_path", type=str, default=None, help="显式指定验证集路径")
    parser.add_argument("--base_weight", type=str, default=None, help="基座模型权重路径")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数 (覆盖 config)")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小 (覆盖 config)")
    parser.add_argument("--learning_rate", type=float, default=None, help="初始学习率 (覆盖 config)")
    parser.add_argument("--weight_decay", type=float, default=None, help="权重衰减 (覆盖 config)")
    parser.add_argument("--accumulation_steps", type=int, default=None, help="梯度累积步数 (覆盖 config)")
    parser.add_argument("--grad_clip", type=float, default=None, help="梯度裁剪阈值 (覆盖 config)")
    parser.add_argument("--max_seq_len", type=int, default=None, help="最大序列长度 (覆盖 config)")
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank (覆盖 config)")
    parser.add_argument("--target_mode", choices=["letter", "answer_text", "answer_rationale", "concise_knowledge"], default="letter",
                        help="训练目标：仅字母，或正确答案文字加字母")
    parser.add_argument("--log_interval", type=int, default=None, help="日志打印间隔 (覆盖 config)")
    parser.add_argument("--val_interval", type=int, default=None, help="验证间隔 (覆盖 config)")
    parser.add_argument("--dtype", type=str, default=None, help="混合精度类型")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (覆盖 config)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有产物")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
