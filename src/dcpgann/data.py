"""Data utilities for IDC classification.

The IDC dataset is typically provided as a directory containing two folders:
```
root/
    0/  # negative tiles
    1/  # IDC-positive tiles
```
Images are loaded using :class:`torchvision.datasets.ImageFolder` with training,
validation, and test splits derived from a single root directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


@dataclass
class DataConfig:
    data_dir: Path
    batch_size: int = 64
    num_workers: int = 4
    val_split: float = 0.15
    test_split: float = 0.15
    image_size: int = 224


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


class IDCDataModule:
    """Create train/val/test dataloaders with common augmentations."""

    def __init__(self, config: DataConfig):
        self.config = config

    def _build_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        train_tf = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
            ]
        )
        eval_tf = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
            ]
        )
        return train_tf, eval_tf

    def setup(self) -> DataLoaders:
        train_tf, eval_tf = self._build_transforms()
        full_dataset = datasets.ImageFolder(self.config.data_dir, transform=train_tf)

        val_size = int(len(full_dataset) * self.config.val_split)
        test_size = int(len(full_dataset) * self.config.test_split)
        train_size = len(full_dataset) - val_size - test_size
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Evaluation transforms are applied at loader level to avoid re-creating datasets
        val_dataset.dataset.transform = eval_tf
        test_dataset.dataset.transform = eval_tf

        return DataLoaders(
            train=DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
            ),
            val=DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
            ),
            test=DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
            ),
        )
