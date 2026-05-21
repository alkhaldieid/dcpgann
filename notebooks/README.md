# Notebook Guide

These notebooks explain the research path around the reproducible CLI. They are
not the source of truth for experiments; the source of truth is the config file,
the saved split file, and the generated `experiment.json`.

Run them with:

```bash
uv sync --extra dev --extra notebooks
uv run jupyter lab notebooks
```

Recommended order:

1. `01_dataset_audit.ipynb` - inspect dataset structure, class balance, and splits.
2. `02_baseline_training.ipynb` - run or inspect a lightweight baseline.
3. `03_backbone_comparison.ipynb` - compare individual CNN backbones.
4. `04_ensemble_optimization.ipynb` - compare equal-weight, weighted, and Cartesian-program ensembles.
5. `05_paper_results_reproduction.ipynb` - assemble the paper-style reproduction ledger.

Every notebook is safe to open before the full experiment has been run. Cells
that require local data or artifacts print clear next-step instructions instead
of failing obscurely.
