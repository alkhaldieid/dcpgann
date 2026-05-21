"""Dataset and split utilities for IDC classification.

The public IDC benchmark is commonly distributed as 50x50 histopathology
patches with binary labels: ``0`` for non-IDC tissue and ``1`` for IDC-positive
tissue.  This module deliberately keeps the data contract explicit and records
the split indices used by an experiment, because reproducibility in this task is
more important than a clever loader.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path
    batch_size: int = 64
    num_workers: int = 4
    val_split: float = 0.15
    test_split: float = 0.15
    image_size: int = 224
    seed: int = 42
    split_file: Optional[Path] = None
    patient_level_split: bool = False


@dataclass(frozen=True)
class SplitIndices:
    train: List[int]
    val: List[int]
    test: List[int]

    def to_json(self, path: Path, config: DataConfig, class_to_idx: dict[str, int]) -> None:
        payload = {
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
            "class_to_idx": class_to_idx,
            "splits": asdict(self),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "SplitIndices":
        payload = json.loads(path.read_text(encoding="utf-8"))
        splits = payload["splits"]
        return cls(train=list(splits["train"]), val=list(splits["val"]), test=list(splits["test"]))


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    splits: SplitIndices
    class_to_idx: dict[str, int]


class TransformSubset(Dataset):
    """Subset wrapper that applies a transform without mutating shared datasets."""

    def __init__(
        self,
        dataset: datasets.ImageFolder,
        indices: Sequence[int],
        transform: transforms.Compose,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        sample_index = self.indices[item]
        path, target = self.dataset.samples[sample_index]
        image = self.dataset.loader(path)
        return self.transform(image), target


def labels_from_dataset(dataset: Dataset) -> torch.Tensor:
    """Collect integer labels from any dataset/subset used by this project."""

    labels: List[int] = []
    for _, target in dataset:
        labels.append(int(target))
    return torch.tensor(labels, dtype=torch.long)


def _validate_split_fractions(val_split: float, test_split: float) -> None:
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")
    if not 0.0 < test_split < 1.0:
        raise ValueError("test_split must be between 0 and 1.")
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be less than 1.")


def _extract_patient_id(path: str) -> str:
    """Infer patient id from common IDC filenames/paths.

    Kaggle/Academic Torrents IDC files commonly look like
    ``8863_idx5_x51_y1251_class0.png`` under patient-specific directories.
    The fallback uses the parent directory, which still prevents leakage for
    original nested layouts.
    """

    name = Path(path).name
    match = re.match(r"(?P<patient>[A-Za-z0-9]+)_idx", name)
    if match:
        return match.group("patient")
    return Path(path).parent.name


def _stratified_indices(targets: Sequence[int], config: DataConfig) -> SplitIndices:
    indices = list(range(len(targets)))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_split,
        stratify=targets,
        random_state=config.seed,
    )
    train_val_targets = [targets[i] for i in train_val_idx]
    relative_val = config.val_split / (1.0 - config.test_split)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        stratify=train_val_targets,
        random_state=config.seed,
    )
    return SplitIndices(train=sorted(train_idx), val=sorted(val_idx), test=sorted(test_idx))


def _group_indices(samples: Sequence[Tuple[str, int]], config: DataConfig) -> SplitIndices:
    indices = list(range(len(samples)))
    targets = [target for _, target in samples]
    groups = [_extract_patient_id(path) for path, _ in samples]

    first = GroupShuffleSplit(n_splits=1, test_size=config.test_split, random_state=config.seed)
    train_val_pos, test_pos = next(first.split(indices, targets, groups))
    train_val_idx = [indices[i] for i in train_val_pos]
    test_idx = [indices[i] for i in test_pos]

    train_val_groups = [groups[i] for i in train_val_idx]
    train_val_targets = [targets[i] for i in train_val_idx]
    relative_val = config.val_split / (1.0 - config.test_split)
    second = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=config.seed)
    train_pos, val_pos = next(second.split(train_val_idx, train_val_targets, train_val_groups))

    train_idx = [train_val_idx[i] for i in train_pos]
    val_idx = [train_val_idx[i] for i in val_pos]
    return SplitIndices(train=sorted(train_idx), val=sorted(val_idx), test=sorted(test_idx))


class IDCDataModule:
    """Create reproducible train/validation/test loaders for IDC experiments."""

    def __init__(self, config: DataConfig):
        _validate_split_fractions(config.val_split, config.test_split)
        self.config = config

    def _build_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        train_tf = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
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

    def _make_splits(self, base_dataset: datasets.ImageFolder) -> SplitIndices:
        if self.config.split_file and self.config.split_file.exists():
            return SplitIndices.from_json(self.config.split_file)
        if self.config.patient_level_split:
            splits = _group_indices(base_dataset.samples, self.config)
        else:
            splits = _stratified_indices(base_dataset.targets, self.config)
        if self.config.split_file:
            splits.to_json(self.config.split_file, self.config, base_dataset.class_to_idx)
        return splits

    def setup(self) -> DataLoaders:
        train_tf, eval_tf = self._build_transforms()
        base_dataset = datasets.ImageFolder(self.config.data_dir)
        if set(base_dataset.class_to_idx) != {"0", "1"}:
            raise ValueError(
                "Expected class-labeled folders named '0' and '1'. "
                f"Found: {sorted(base_dataset.class_to_idx)}"
            )

        splits = self._make_splits(base_dataset)
        train_dataset = TransformSubset(base_dataset, splits.train, train_tf)
        val_dataset = TransformSubset(base_dataset, splits.val, eval_tf)
        test_dataset = TransformSubset(base_dataset, splits.test, eval_tf)

        generator = torch.Generator().manual_seed(self.config.seed)
        pin_memory = torch.cuda.is_available()
        return DataLoaders(
            train=DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=pin_memory,
                generator=generator,
            ),
            val=DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=pin_memory,
            ),
            test=DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=pin_memory,
            ),
            splits=splits,
            class_to_idx=base_dataset.class_to_idx,
        )


def class_counts(labels: Iterable[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for label in labels:
        counts[int(label)] = counts.get(int(label), 0) + 1
    return dict(sorted(counts.items()))
