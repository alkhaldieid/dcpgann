# Differential Cartesian Genetic Programming for IDC Classification

This repository provides a research-inspired implementation of **Ensemble Optimization for Invasive Ductal Carcinoma (IDC) Classification Using Differential Cartesian Genetic Programming**. It trains four CNN backbones on the IDC histopathology tiles dataset, then optimizes ensemble weights via differential cartesian genetic programming (DCGP) or differential evolution to maximize balanced accuracy or F1 on a validation split.

## Features
- Torch-based training loop with gradient clipping and cosine learning-rate schedule.
- Ready-to-use backbones tailored to the paper: `resnet50_end_to_end`, ImageNet-initialized `resnet50_partial` with the last 69 layers unfrozen, fully finetuned ImageNet `vgg19_finetune`, and partially unfrozen ImageNet `densenet121_partial` (429 trainable layers).
- Automated validation/test evaluation and checkpointing per backbone.
- Ensemble optimizer with a DCGP searcher (via the `dcgp` library) and a SciPy differential-evolution fallback.

## Project structure
- `src/dcpgann/` – library code (data module, models, training utilities, ensemble optimizer).
- `src/train.py` – CLI entrypoint to train backbones and compute the optimized ensemble.
- `requirements.txt` – Python dependencies.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset
Place the IDC dataset in a directory with class-labeled subfolders:
```
/path/to/idc/
    0/  # negative tiles
    1/  # IDC-positive tiles
```
You can obtain the dataset from Kaggle's "Breast Histopathology Images" release or another equivalent IDC tile dataset.

## Usage
Train the four backbone variants and optimize ensemble weights:
```bash
python -m src.train /path/to/idc --output artifacts --epochs 10 --metric balanced_accuracy
```
Key arguments:
- `--backbones`: comma-separated backbones (defaults to the four predefined architectures above; must contain exactly four entries for the ensemble).
- `--metric`: `balanced_accuracy` or `f1` for ensemble objective.
- `--ensemble_method`: `dcgp` (default) or `scipy` for weight search.
- `--image_size`: resize for tiles (default 224).
- `--val_split` / `--test_split`: fractions for validation/test splits.

Outputs are written to `--output` (checkpoints per backbone and `experiment.json` containing histories and ensemble weights).

## Notes
- Training defaults are conservative (5 epochs). Increase for thorough experiments.
- The ensemble optimizer uses DCGP by default; you can adjust population size, generations, and DCGP hyperparameters in `dcpgann/ensemble.py`.
- To continue from saved checkpoints, load the `.pt` files and rerun ensemble optimization on fresh logits.

## License
This codebase is provided for research and educational purposes. Ensure dataset licenses permit your use case.
