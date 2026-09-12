# Local artifacts

This directory is for generated or transferable files that are not source code.

- `archives/`: cloud-run bundles and other large transfer archives.

Archive files remain ignored by Git. Model checkpoints continue to use `out/`,
and downloaded Transformers files continue to use `minimind-3/`, because those
locations are part of the existing MiniMind command-line defaults.
