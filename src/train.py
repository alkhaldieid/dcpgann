"""Run a reproducible IDC ensemble experiment."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import classification_report

from dcpgann.data import DataConfig, IDCDataModule, class_counts
from dcpgann.ensemble import CartesianProgramOptimizer, EnsembleConfig, EnsembleOptimizer
from dcpgann.models import count_parameters, list_backbones
from dcpgann.training import TrainConfig, classification_metrics, predict_logits, seed_everything, train_model


DEFAULT_BACKBONES = [
    "resnet50_end_to_end",
    "resnet50_partial",
    "vgg19_finetune",
    "densenet121_partial",
]


def _load_json(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_cli(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = dict(config)
    merged.setdefault("data", {})
    merged.setdefault("train", {})
    merged.setdefault("ensemble", {})
    if args.data_dir is not None:
        merged["data"]["data_dir"] = str(args.data_dir)
    if args.output is not None:
        merged["output_dir"] = str(args.output)
    if args.epochs is not None:
        merged["train"]["epochs"] = args.epochs
    if args.backbones is not None:
        merged["backbones"] = [name.strip() for name in args.backbones.split(",") if name.strip()]
    if args.metric is not None:
        merged["ensemble"]["metric"] = args.metric
    if args.ensemble_method is not None:
        merged["ensemble"]["method"] = args.ensemble_method
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", help="Path to IDC root directory with folders 0/ and 1/")
    parser.add_argument("--config", type=Path, help="JSON experiment config")
    parser.add_argument("--output", type=Path, help="Directory for checkpoints and reports")
    parser.add_argument("--epochs", type=int, help="Training epochs for each backbone")
    parser.add_argument("--backbones", type=str, help="Comma-separated backbone list")
    parser.add_argument("--metric", choices=["balanced_accuracy", "f1", "roc_auc"], help="Ensemble objective")
    parser.add_argument("--ensemble_method", choices=["cgp", "dcgp", "scipy", "weighted_logits"], help="Ensemble optimizer")
    return parser.parse_args()


def _make_data_config(payload: Dict[str, Any], output_dir: Path, seed: int) -> DataConfig:
    data = dict(payload.get("data", {}))
    if "data_dir" not in data:
        raise ValueError("data.data_dir is required. Pass it positionally or set it in the config.")
    split_file = data.get("split_file")
    if split_file is None:
        split_file = output_dir / "splits.json"
    else:
        split_file = Path(split_file)
    return DataConfig(
        data_dir=Path(data["data_dir"]),
        batch_size=int(data.get("batch_size", 64)),
        num_workers=int(data.get("num_workers", 4)),
        val_split=float(data.get("val_split", 0.15)),
        test_split=float(data.get("test_split", 0.15)),
        image_size=int(data.get("image_size", 224)),
        seed=int(data.get("seed", seed)),
        split_file=split_file,
        patient_level_split=bool(data.get("patient_level_split", False)),
    )


def _make_train_config(payload: Dict[str, Any], seed: int) -> TrainConfig:
    train = dict(payload.get("train", {}))
    return TrainConfig(
        epochs=int(train.get("epochs", 5)),
        learning_rate=float(train.get("learning_rate", 3e-4)),
        weight_decay=float(train.get("weight_decay", 1e-4)),
        max_grad_norm=float(train.get("max_grad_norm", 5.0)),
        seed=int(train.get("seed", seed)),
        device=str(
            train.get(
                "device",
                "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
            )
        ),
        use_amp=bool(train.get("use_amp", True)),
        class_weighted_loss=bool(train.get("class_weighted_loss", True)),
        early_stopping_patience=train.get("early_stopping_patience"),
    )


def _make_ensemble_config(payload: Dict[str, Any], seed: int) -> EnsembleConfig:
    ensemble = dict(payload.get("ensemble", {}))
    return EnsembleConfig(
        metric=str(ensemble.get("metric", "balanced_accuracy")),
        method=str(ensemble.get("method", "cgp")),
        seed=int(ensemble.get("seed", seed)),
        scipy_pop_size=int(ensemble.get("scipy_pop_size", 15)),
        scipy_max_iter=int(ensemble.get("scipy_max_iter", 50)),
        cgp_nodes=int(ensemble.get("cgp_nodes", 12)),
        cgp_population=int(ensemble.get("cgp_population", 24)),
        cgp_generations=int(ensemble.get("cgp_generations", 120)),
        cgp_elite=int(ensemble.get("cgp_elite", 4)),
        cgp_mutation_rate=float(ensemble.get("cgp_mutation_rate", 0.12)),
        cgp_constant_scale=float(ensemble.get("cgp_constant_scale", 1.0)),
    )


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(config.get("seed", 42))
    output_dir = Path(config.get("output_dir", "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    data_config = _make_data_config(config, output_dir, seed)
    train_config = _make_train_config(config, seed)
    ensemble_config = _make_ensemble_config(config, seed)
    backbone_names = list(config.get("backbones", DEFAULT_BACKBONES))

    registry = list_backbones()
    unknown = [name for name in backbone_names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown backbones {unknown}. Available: {sorted(registry)}")
    if len(backbone_names) < 1:
        raise ValueError("At least one backbone is required.")

    datamodule = IDCDataModule(data_config)
    loaders = datamodule.setup()

    val_logits: List[torch.Tensor] = []
    test_logits: List[torch.Tensor] = []
    histories: Dict[str, List[Dict[str, float]]] = {}
    model_summaries: Dict[str, Dict[str, int]] = {}
    individual_test_metrics: Dict[str, Dict[str, float]] = {}
    labels_val: torch.Tensor | None = None
    labels_test: torch.Tensor | None = None

    for name in backbone_names:
        model = registry[name]()
        model_summaries[name] = count_parameters(model)
        print(f"Training backbone: {name} ({model_summaries[name]['trainable']:,} trainable parameters)")
        result = train_model(model, (loaders.train, loaders.val), train_config)
        histories[name] = result.history
        model.load_state_dict(result.best_state_dict)

        val_output = predict_logits(model, loaders.val, train_config.device)
        test_output = predict_logits(model, loaders.test, train_config.device)
        val_logits.append(val_output.logits)
        test_logits.append(test_output.logits)
        labels_val = val_output.labels
        labels_test = test_output.labels
        individual_test_metrics[name] = test_output.metrics

        ckpt_path = output_dir / "checkpoints" / f"{name}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": name,
                "state_dict": result.best_state_dict,
                "train_config": asdict(train_config),
                "best_val_loss": result.best_val_loss,
                "best_val_balanced_accuracy": result.best_val_balanced_accuracy,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

    assert labels_val is not None
    assert labels_test is not None

    equal_weights = np.ones(len(test_logits), dtype=np.float64) / len(test_logits)
    ensemble = EnsembleOptimizer(ensemble_config)
    weighted_baseline = ensemble.weighted
    equal_test_logits = weighted_baseline.ensemble_logits(test_logits, equal_weights)
    equal_test_metrics = classification_metrics(labels_test, equal_test_logits)

    ensemble_result = ensemble.fit(val_logits, labels_val)
    if ensemble_result.method == "cartesian_program":
        cgp = CartesianProgramOptimizer(ensemble_config)
        test_ensemble_logits = cgp.predict_logits(test_logits, ensemble_result.metadata["genome"])
    else:
        test_ensemble_logits = weighted_baseline.ensemble_logits(test_logits, ensemble_result.weights)

    test_metrics = classification_metrics(labels_test, test_ensemble_logits)
    y_pred = torch.argmax(test_ensemble_logits, dim=1).numpy()
    report_text = classification_report(labels_test.numpy(), y_pred, digits=4)
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    metadata: Dict[str, Any] = {
        "seed": seed,
        "paper": {
            "title": "Ensemble Optimization for Invasive Ductal Carcinoma (IDC) Classification Using Differential Cartesian Genetic Programming",
            "doi": "10.1109/ACCESS.2022.3228176",
            "dataset": "Breast Histopathology Images / IDC_regular_ps50_idx5",
        },
        "data_config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(data_config).items()},
        "train_config": asdict(train_config),
        "ensemble_config": asdict(ensemble_config),
        "backbones": backbone_names,
        "class_to_idx": loaders.class_to_idx,
        "split_sizes": {
            "train": len(loaders.splits.train),
            "val": len(loaders.splits.val),
            "test": len(loaders.splits.test),
        },
        "split_class_counts": {
            "train": class_counts([loaders.train.dataset.dataset.samples[i][1] for i in loaders.splits.train]),
            "val": class_counts([loaders.val.dataset.dataset.samples[i][1] for i in loaders.splits.val]),
            "test": class_counts([loaders.test.dataset.dataset.samples[i][1] for i in loaders.splits.test]),
        },
        "model_summaries": model_summaries,
        "train_histories": histories,
        "individual_test_metrics": individual_test_metrics,
        "equal_weight_ensemble_test_metrics": equal_test_metrics,
        "optimized_ensemble_validation": {
            "method": ensemble_result.method,
            "score": ensemble_result.score,
            "metrics": ensemble_result.metrics,
            "weights": ensemble_result.weights,
            "metadata": ensemble_result.metadata,
        },
        "optimized_ensemble_test_metrics": test_metrics,
    }
    _save_json(output_dir / "experiment.json", metadata)

    print("\nEqual-weight ensemble test metrics")
    print(json.dumps(equal_test_metrics, indent=2))
    print("\nOptimized ensemble validation result")
    print(json.dumps({"method": ensemble_result.method, "score": ensemble_result.score}, indent=2))
    print("\nOptimized ensemble test metrics")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved experiment report to {output_dir / 'experiment.json'}")
    return metadata


def main() -> None:
    args = parse_args()
    config = _merge_cli(_load_json(args.config), args)
    run_experiment(config)


if __name__ == "__main__":
    main()
