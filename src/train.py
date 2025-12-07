"""Train IDC classifiers and optimize an ensemble using differential evolution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import classification_report

from dcpgann.data import DataConfig, IDCDataModule
from dcpgann.ensemble import EnsembleConfig, EnsembleOptimizer
from dcpgann.models import list_backbones
from dcpgann.training import TrainConfig, predict_logits, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Path to IDC root directory (folders 0/ and 1/)")
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for each backbone")
    parser.add_argument(
        "--backbones",
        type=str,
        default="resnet50_end_to_end,resnet50_partial,vgg19_finetune,densenet121_partial",
        help="Comma-separated list of backbones to train (defaults to the four-model IDC ensemble)",
    )
    parser.add_argument("--metric", type=str, default="balanced_accuracy", choices=["balanced_accuracy", "f1"], help="Ensemble objective")
    parser.add_argument("--ensemble_method", type=str, default="dcgp", choices=["dcgp", "scipy"], help="Optimizer for ensemble weights")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    backbones_registry = list_backbones()
    backbone_names = [name.strip() for name in args.backbones.split(",") if name.strip()]
    for name in backbone_names:
        if name not in backbones_registry:
            raise ValueError(f"Unknown backbone '{name}'. Available: {list(backbones_registry)}")
    if len(backbone_names) != 4:
        raise ValueError(
            "The ensemble must contain exactly four models: resnet50_end_to_end, resnet50_partial, vgg19_finetune, densenet121_partial"
        )

    data_config = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        image_size=args.image_size,
    )
    datamodule = IDCDataModule(data_config)
    loaders = datamodule.setup()

    train_cfg = TrainConfig(epochs=args.epochs)
    val_logits: List[torch.Tensor] = []
    test_logits: List[torch.Tensor] = []
    histories: Dict[str, List[Dict[str, float]]] = {}

    for name in backbone_names:
        model_builder = backbones_registry[name]
        model = model_builder()
        print(f"Training backbone: {name}")
        result = train_model(model, (loaders.train, loaders.val), train_cfg)
        histories[name] = result.history

        # Load best weights and compute logits for val/test
        model.load_state_dict(result.best_state_dict)
        val_logits.append(predict_logits(model, loaders.val, train_cfg.device))
        test_logits.append(predict_logits(model, loaders.test, train_cfg.device))

        # Persist checkpoint
        ckpt_path = args.output / f"{name}.pt"
        torch.save(result.best_state_dict, ckpt_path)
        print(f"Saved {name} checkpoint to {ckpt_path}")

    # Optimize ensemble weights on validation set
    labels_val = torch.cat([labels for _, labels in loaders.val.dataset])
    ensemble_config = EnsembleConfig(metric=args.metric, method=args.ensemble_method)
    optimizer = EnsembleOptimizer(ensemble_config)
    weights = optimizer.optimize(val_logits, labels_val)
    print("Optimized ensemble weights:", weights)

    # Evaluate on test set
    labels_test = torch.cat([labels for _, labels in loaders.test.dataset])
    ensemble_logits = optimizer.ensemble_logits(test_logits, weights)
    y_pred = torch.argmax(ensemble_logits, dim=1).numpy()
    report = classification_report(labels_test.numpy(), y_pred, output_dict=False)
    print(report)

    # Persist metadata
    meta = {
        "backbones": backbone_names,
        "weights": weights.tolist(),
        "train_histories": histories,
        "metric": args.metric,
    }
    meta_path = args.output / "experiment.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved experiment metadata to {meta_path}")


if __name__ == "__main__":
    main()
