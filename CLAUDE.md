# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of the upstream [MiniMind](https://github.com/jingyaogong/minimind) project (a from-scratch, tiny LLM training pipeline: Pretrain → SFT → LoRA → RLHF/RLAIF → distillation). This fork's own contribution is a set of medical-domain LoRA fine-tuning experiments layered on top of the upstream code, plus repo-hygiene conventions for keeping experiment artifacts separate from reusable code (see `docs/project-layout.md`).

All core training/model code is implemented from scratch in native PyTorch — it deliberately avoids high-level abstractions from `transformers`/`trl`/`peft` for the algorithmic core (dataset classes, model, LoRA, trainers), even though those libraries are used for tokenization, generation utilities, and I/O.

## Commands

Install dependencies: `pip install -r requirements.txt` (torch/torchvision and peft/matplotlib are commented out in `requirements.txt` — install the correct torch build for your hardware separately).

Training scripts assume you `cd trainer` first (they use relative paths like `../out`, `../model`):

```bash
cd trainer
python train_pretrain.py          # or: torchrun --nproc_per_node N train_pretrain.py
python train_full_sft.py
python train_lora.py              # LoRA fine-tuning; runs fine on CPU
python train_distillation.py
python train_dpo.py
python train_ppo.py / train_grpo.py
python train_agent.py             # Agentic RL (tool-use, GRPO/CISPO)
```

Evaluation / inference from the repo root:

```bash
python eval_llm.py --weight pretrain            # quick interactive check after pretrain/SFT
python eval_llm.py --weight full_sft
python eval_llm.py --weight full_sft --lora_weight lora_medical   # base + LoRA overlay
python run_eval.py --mode base|lora [--val-loss] [--limit N]      # medical LoRA eval on a locked question set
python run_eval.py --compare                                       # base vs lora comparison report
```

Tests (pure-logic, no real model weights needed — uses tiny in-memory tensors):

```bash
python -m pytest tests/                    # or: python -m unittest tests/test_eval_logic.py
python -m unittest tests.test_eval_logic.TestMCQExtraction   # single test class
```

`run_eval.py` stays at the repo root (not under `trainer/`) because `tests/test_eval_logic.py` imports it as a module, and medical experiment dirs symlink to it rather than duplicating it.

## Architecture

- `model/model_minimind.py` — the MiniMind model itself (`MiniMindConfig`, attention/MoE blocks), aligned with the Qwen3/Qwen3-MoE architecture family. Config knobs: `hidden_size`, `num_hidden_layers`, `use_moe`, GQA (`num_attention_heads` vs `num_key_value_heads`), etc.
- `model/model_lora.py` — hand-rolled LoRA: `apply_lora` monkey-patches `nn.Linear` layers (square weight matrices only) by attaching a `LoRA` submodule and wrapping `.forward`; `save_lora`/`load_lora` (de)serialize only the LoRA deltas by name; `merge_lora` bakes LoRA deltas into the base weights for export. No PEFT library involved.
- `dataset/lm_dataset.py` — one `Dataset` subclass per training stage: `PretrainDataset`, `SFTDataset` (chat-template based, produces loss masks over assistant turns only), `DPODataset`, `RLAIFDataset`, `AgentRLDataset`.
- `trainer/trainer_utils.py` — shared helpers used by every `train_*.py` script: `init_model` (loads base weights + tokenizer), `lm_checkpoint` (save/resume with epoch/step/optimizer state), `evaluate_val_loss`, distributed-training setup (`init_distributed_mode`, DDP/DeepSpeed), `get_lr` (schedule), seeding.
- `trainer/train_*.py` — one script per training stage; each is a standalone entry point (argparse + a training loop), not a shared abstraction — they intentionally duplicate loop structure rather than share a `Trainer` class, matching the project's "read every line" teaching philosophy.
- `run_eval.py` — the medical-domain evaluation harness: MCQ answer extraction (`extract_mcq_choice`, robust to `A`/`(A)`/`【A】`/`答案：A` etc.), deterministic generation (`generate_single_response`, `do_sample=False`), a full base-vs-LoRA comparison report (`run_comparison`) that checks generation config / question-set version / question IDs match before comparing, and reserves columns for manual clinical review. This is the piece with real test coverage (`tests/test_eval_logic.py`).
- `scripts/` — inference-adjacent utilities independent of training: `serve_openai_api.py` (OpenAI-compatible server), `web_demo.py` (Streamlit UI), `chat_api.py`, `convert_model.py` (merges LoRA into a full checkpoint), `eval_toolcall.py`.
- `minimind-3/` — a downloaded Transformers-format model package (tokenizer, config, chat template); weights are gitignored but metadata/config files are tracked.

### Experiments convention

Project-specific work (e.g. medical LoRA fine-tuning) lives under `experiments/<topic>_YYYYMMDD/`, each a self-contained bundle of code, frozen data, eval sets, results, figures, logs, and audit manifests — see `experiments/README.md` for the exact layout and `docs/project-layout.md` for what's tracked in Git vs. kept local-only (checkpoints in `out/`, model weights, transfer archives). Notable non-obvious detail: experiment `data/` dirs hold *frozen* inputs, and where a frozen copy turned out byte-identical to a canonical file it was replaced by a `MANIFEST.json` recording its `sha256` plus a restore command (e.g. `experiments/lora_intro_20260909/data/`). Restore from the manifest before re-running such an experiment; don't silently repoint the script at a newer canonical dataset.

When adding a new experiment, follow the existing directory shape (`data/`, `eval/`, `eval_results/`, `figures/`, `logs/`, `audit/`) and resolve paths via `Path(__file__)` rather than assuming a working directory.
