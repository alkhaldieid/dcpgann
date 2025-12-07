"""Training utilities for IDC classification models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm


@dataclass
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainResult:
    best_state_dict: Dict[str, torch.Tensor]
    history: List[Dict[str, float]]
    best_val_loss: float


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
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
    return total_loss / len(loader.dataset), logits_tensor, labels_tensor


def train_model(
    model: nn.Module,
    loaders: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
    config: TrainConfig,
) -> TrainResult:
    train_loader, val_loader = loaders
    device = torch.device(config.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_loss = float("inf")
    best_state: Dict[str, torch.Tensor] = {}
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config.max_grad_norm)
        val_loss, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return TrainResult(best_state_dict=best_state, history=history, best_val_loss=best_val_loss)


def predict_logits(model: nn.Module, loader: torch.utils.data.DataLoader, device: str) -> torch.Tensor:
    model.eval()
    outputs: List[torch.Tensor] = []
    dev = torch.device(device)
    model.to(dev)
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="predict", leave=False):
            images = images.to(dev)
            logits = model(images)
            outputs.append(logits.cpu())
    return torch.cat(outputs)
