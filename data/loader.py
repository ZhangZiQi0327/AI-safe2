from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def read_labels(label_file: str | Path) -> list[tuple[str, int]]:
    label_path = Path(label_file)
    samples: list[tuple[str, int]] = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid label line {line_no}: {line!r}")
        name, label_text = parts
        label = int(label_text)
        if label < 0 or label >= len(CIFAR10_CLASSES):
            raise ValueError(f"Invalid class id {label} in line {line_no}")
        samples.append((name, label))
    return samples


class CifarAttackDataset(Dataset):
    """Dataset for the competition folder: images/ plus label.txt."""

    def __init__(self, root: str | Path = "dataset", transform=None) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.label_file = self.root / "label.txt"
        self.transform = transform
        self.samples = read_labels(self.label_file)

        missing = [name for name, _ in self.samples if not (self.image_dir / name).exists()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"Missing {len(missing)} images under {self.image_dir}: {preview}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        name, label = self.samples[index]
        image = Image.open(self.image_dir / name).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, name

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]


def build_transforms(train: bool, color_jitter: bool = True):
    if train:
        augments: list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if color_jitter:
            augments.append(
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.03)
            )
        augments.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        return transforms.Compose(augments)

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _stratified_split(labels: Iterable[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[int(label)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in by_label.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        if len(shuffled) <= 1 or val_ratio <= 0:
            val_count = 0
        else:
            val_count = max(1, round(len(shuffled) * val_ratio))
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def make_dataloaders(
    root: str | Path = "dataset",
    batch_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    color_jitter: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = CifarAttackDataset(root, transform=build_transforms(train=True, color_jitter=color_jitter))
    val_dataset = CifarAttackDataset(root, transform=build_transforms(train=False))

    train_indices, val_indices = _stratified_split(train_dataset.labels, val_ratio, seed)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader
