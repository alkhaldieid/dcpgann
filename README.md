# DCPGANN: Reproducible IDC Ensemble Experiments

This repository contains a research-grade PyTorch implementation for IDC-positive
versus IDC-negative histopathology patch classification, inspired by:

> Eid Alkhaldi and Ehsan Salari, "Ensemble Optimization for Invasive Ductal
> Carcinoma (IDC) Classification Using Differential Cartesian Genetic
> Programming," IEEE Access, 2022. DOI: `10.1109/ACCESS.2022.3228176`.

The code is intentionally structured around reproducibility: deterministic
splits, saved experiment configuration, per-backbone checkpoints, individual
model metrics, equal-weight ensemble metrics, and optimized ensemble metrics.

## What This Code Does

- Trains CNN backbones for binary IDC classification.
- Supports the paper-style ensemble backbones:
  - `resnet50_end_to_end`
  - `resnet50_partial`
  - `vgg19_finetune`
  - `densenet121_partial`
- Provides a transparent weighted-logit ensemble baseline using differential
  evolution.
- Provides a compact Cartesian-program ensemble optimizer over model
  probabilities, with the evolved graph saved in `experiment.json`.
- Reports accuracy, balanced accuracy, precision, sensitivity/recall,
  specificity, F1, ROC-AUC, and confusion-matrix counts.

## Dataset

Use the public IDC benchmark commonly distributed as:

- Dataset: **Breast Histopathology Images**
- Kaggle: `paultimothymooney/breast-histopathology-images`
- Original archive name: `IDC_regular_ps50_idx5.zip`
- Task: binary patch classification, `0` for non-IDC and `1` for IDC-positive

The current loader expects a class-folder layout:

```text
/path/to/idc/
  0/
    image_class0.png
  1/
    image_class1.png
```

If you download the original nested patient-folder layout, create or symlink a
flat class-folder copy before running experiments.

## Installation

Recommended `uv` workflow:

```bash
uv sync --extra dev
```

Then run commands through `uv run`:

```bash
uv run pytest
uv run dcpgann-train /path/to/idc --config configs/paper_2022_idc.json --epochs 1 --backbones simple_cnn
```

Classic `venv` workflow:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

## Smoke Test

Run unit tests first:

```bash
uv run pytest
```

For a tiny local sanity run, edit `configs/smoke_test.json` so `data.data_dir`
points to a small two-class folder, then run:

```bash
uv run dcpgann-train --config configs/smoke_test.json
```

This uses `simple_cnn` for one epoch and is meant to validate the local
environment, not reproduce the paper.

## Research Notebooks

The `notebooks/` directory contains explanation notebooks that document how the
dataset, baselines, backbones, ensemble optimizer, and paper-style reproduction
fit together. They are narrative companions to the CLI, not the source of truth.

```bash
uv sync --extra dev --extra notebooks
uv run jupyter lab notebooks
```

Read them in order:

1. `01_dataset_audit.ipynb`
2. `02_baseline_training.ipynb`
3. `03_backbone_comparison.ipynb`
4. `04_ensemble_optimization.ipynb`
5. `05_paper_results_reproduction.ipynb`

## Paper-Style Experiment

Edit `configs/paper_2022_idc.json` and set:

```json
"data_dir": "/absolute/path/to/idc"
```

Then run:

```bash
uv run dcpgann-train --config configs/paper_2022_idc.json
```

Useful overrides:

```bash
uv run dcpgann-train /path/to/idc --config configs/paper_2022_idc.json --epochs 10
uv run dcpgann-train /path/to/idc --config configs/paper_2022_idc.json --ensemble_method scipy
uv run dcpgann-train /path/to/idc --config configs/paper_2022_idc.json --output artifacts/local_mps
```

On Apple Silicon, the trainer automatically uses `mps` when PyTorch reports it
as available; otherwise it falls back to CPU. For 24 GB unified memory, start
with `batch_size` 16 or 32 for the ImageNet backbones if 64 is too large.

Outputs are written under `artifacts/paper_2022_idc/` by default:

- `splits.json`: exact train/validation/test indices
- `checkpoints/*.pt`: best state per backbone
- `classification_report.txt`: sklearn test report
- `experiment.json`: full reproducibility record and metrics

## Interpreting Results

The repository reports both:

- **Equal-weight ensemble**: a sanity baseline.
- **Optimized ensemble**: selected by validation performance and evaluated once
  on the held-out test split.

Do not compare a new run to the paper unless the dataset copy, preprocessing,
split protocol, seeds, backbone settings, and training budget are documented.
The code is designed to make those assumptions visible rather than bury them.

## Development Workflow

```bash
uv run pytest
uv run python -m compileall -q src tests
git status --short
git add README.md pyproject.toml uv.lock requirements.txt configs notebooks src tests
git commit -m "Professionalize reproducible IDC ensemble pipeline"
git push origin main
```

## Research Integrity Notes

This implementation is written to be explicit and auditable. If your goal is to
claim exact paper reproduction, keep the paper configuration immutable, preserve
the generated `splits.json`, and report all model, ensemble, and baseline
metrics. If your goal is to improve on the paper, create a new config and report
it as a modernized experiment rather than overwriting the historical protocol.
