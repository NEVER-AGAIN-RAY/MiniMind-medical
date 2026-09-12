# Project layout

This repository keeps the upstream MiniMind layout intact and separates reusable
code from local experiments and generated artifacts.

## Core project

| Path | Purpose | Tracked in Git |
| --- | --- | --- |
| `model/` | MiniMind model and LoRA implementation | Yes |
| `trainer/` | Training entry points and shared training utilities | Yes |
| `dataset/` | Dataset loader plus canonical project datasets | Code and selected data |
| `scripts/` | Conversion, serving, demo, and tool-call utilities | Yes |
| `tests/` | Fast regression tests | Yes |
| `images/` | Assets referenced by the root README files | Yes |
| `eval_llm.py` | General interactive model evaluation | Yes |
| `run_eval.py` | Medical LoRA evaluation and comparison entry point | Yes |

`run_eval.py` stays at the repository root because the test suite imports it as a
module and the medical experiment's `run_eval.py` is a symlink to it.

## Experiments

All project-specific investigations live under `experiments/`. Each experiment
should keep its code, data, results, figures, logs, and audit material together.
See [`experiments/README.md`](../experiments/README.md) for the convention.

Some apparent duplicates are intentional:

- `minimind-3/images/` belongs to the standalone Hugging Face model package and
  supports the README files shipped with that package.
- `experiments/lora_intro_20260909/data/lora_medical.jsonl` is a frozen input to
  a historical experiment, while `dataset/lora_medical.jsonl` is the canonical
  project dataset.
- Timestamped `pip_freeze` files and run logs are retained as audit evidence for
  completed runs.

## Local and generated artifacts

| Path | Purpose | Policy |
| --- | --- | --- |
| `.venv/` | Local Python environment | Keep locally; never commit |
| `out/` | PyTorch checkpoints and completion markers | Keep locally; never commit weights |
| `minimind-3/` | Downloaded Transformers model package | Track metadata; ignore model weights |
| `artifacts/archives/` | Transfer bundles and cloud-run archives | Keep locally; never commit archives |

The names `out/` and `minimind-3/` deliberately match upstream scripts. Moving
them would make the repository look marginally cleaner but would require many
fragile path overrides.

## Cleanup rules

- Safe to remove: `.DS_Store`, `__pycache__/`, `.pytest_cache/`, `*.pyc`,
  `*.profraw`, and experiment `tmp/` or `cache/` directories.
- Review before removing: checkpoints, model packages, datasets, logs, reports,
  archives, and generated figures.
- Before a larger cleanup, run `git status --short --ignored` and preserve any
  pre-existing source changes.
