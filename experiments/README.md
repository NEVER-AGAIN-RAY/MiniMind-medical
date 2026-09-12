# Experiments

Use one dated directory per experiment:

```text
experiments/<topic>_YYYYMMDD/
├── README.md or handoff.md   # goal, commands, and conclusions
├── *.py / *.sh              # experiment entry points
├── data/                    # frozen inputs and generated splits
├── eval/                    # fixed evaluation sets
├── eval_results/            # metrics and model responses
├── figures/                 # plots and their source material
├── logs/                    # retained run evidence
└── audit/                   # manifests, checksums, and reviews
```

Guidelines:

- Resolve paths from the repository root or from `Path(__file__)`; do not depend
  on an undocumented working directory.
- Keep small manifests, metrics, reports, and reproduction scripts in Git.
- Keep large weights, caches, temporary files, and transfer archives out of Git.
- Treat copied input data as a frozen experiment dependency and record its hash
  rather than silently replacing it with a newer canonical dataset.
- Put raw or alternate figure exports in `figures/source/`; keep the selected
  report figure directly in `figures/`.
