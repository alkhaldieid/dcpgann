"""Training, evaluation, and reproducibility utilities."""
from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0
    seed: int = 42
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp: bool = True
    class_weighted_loss: bool = True
    early_stopping_patience: Optional[int] = None


@dataclass(frozen=True)
class EvalOutput:
    loss: float
    logits: torch.Tensor
    labels: torch.Tensor
    metrics: Dict[str, float]


@dataclass(frozen=True)
class TrainResult:
    best_state_dict: Dict[str, torch.Tensor]
    history: List[Dict[str, float]]
    best_val_loss: float
    best_val_balanced_accuracy: float


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def compute_class_weights(loader: torch.utils.data.DataLoader, num_classes: int = 2) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels in loader:
        counts += torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


def make_grad_scaler(enabled: bool) -> torch.cuda.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@contextmanager
def autocast_cuda(enabled: bool) -> Iterator[None]:
    try:
        with torch.amp.autocast("cuda", enabled=enabled):
            yield
    except (AttributeError, TypeError):
        with torch.cuda.amp.autocast(enabled=enabled):
            yield


def classification_metrics(labels: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = torch.argmax(logits, dim=1).numpy()
    y_true = labels.numpy()

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall_sensitivity": recall_score(y_true, preds, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, preds, zero_division=0),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, probs)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return {key: float(value) for key, value in metrics.items()}


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    amp_enabled = use_amp and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)

    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_cuda(amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvalOutput:
    model.eval()
    total_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)
    return EvalOutput(
        loss=total_loss / len(loader.dataset),
        logits=logits_tensor,
        labels=labels_tensor,
        metrics=classification_metrics(labels_tensor, logits_tensor),
    )


def train_model(
    model: nn.Module,
    loaders: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
    config: TrainConfig,
) -> TrainResult:
    train_loader, val_loader = loaders
    seed_everything(config.seed)
    device = torch.device(config.device)
    model.to(device)

    class_weights = None
    if config.class_weighted_loss:
        class_weights = compute_class_weights(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))

    best_val_loss = float("inf")
    best_val_balanced_accuracy = 0.0
    best_state: Dict[str, torch.Tensor] = clone_state_dict(model)
    history: List[Dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config.max_grad_norm,
            config.use_amp,
        )
        val_output = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_output.loss),
            **{f"val_{key}": value for key, value in val_output.metrics.items()},
        }
        history.append(row)

        current_score = val_output.metrics["balanced_accuracy"]
        improved = (current_score > best_val_balanced_accuracy) or (
            current_score == best_val_balanced_accuracy and val_output.loss < best_val_loss
        )
        if improved:
            best_val_loss = val_output.loss
            best_val_balanced_accuracy = current_score
            best_state = clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if config.early_stopping_patience is not None and epochs_without_improvement >= config.early_stopping_patience:
            break

    return TrainResult(
        best_state_dict=copy.deepcopy(best_state),
        history=history,
        best_val_loss=best_val_loss,
        best_val_balanced_accuracy=best_val_balanced_accuracy,
    )


def predict_logits(model: nn.Module, loader: torch.utils.data.DataLoader, device: str) -> EvalOutput:
    dev = torch.device(device)
    model.to(dev)
    criterion = nn.CrossEntropyLoss()
    return eval_epoch(model, loader, criterion, dev)
