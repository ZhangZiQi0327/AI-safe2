from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from data.loader import build_transforms, make_dataloaders
from models import available_models, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 proxy models for adversarial transfer.")
    parser.add_argument("--dataset", default="dataset", help="Path containing images/ and label.txt.")
    parser.add_argument("--model", choices=available_models(), required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--no-color-jitter", action="store_true")
    parser.add_argument(
        "--include-public-cifar10",
        action="store_true",
        help="Also train on torchvision CIFAR-10 train split for stronger proxy models.",
    )
    parser.add_argument("--cifar10-root", default="data_cache", help="Root directory for torchvision CIFAR-10.")
    parser.add_argument("--download-cifar10", action="store_true", help="Download public CIFAR-10 if missing.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


class ImageLabelOnly(Dataset):
    """Drop optional filename fields so mixed datasets collate consistently."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        return sample[0], sample[1]


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    progress = tqdm(loader, leave=False)
    for batch in progress:
        images, labels = batch[0], batch[1]
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size
        progress.set_description(("train" if train else "eval") + f" loss={loss.item():.4f}")

    return total_loss / total_count, total_correct / total_count


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_dataloaders(
        root=args.dataset,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        color_jitter=not args.no_color_jitter,
    )
    if args.include_public_cifar10:
        public_train = CIFAR10(
            root=args.cifar10_root,
            train=True,
            transform=build_transforms(train=True, color_jitter=not args.no_color_jitter),
            download=args.download_cifar10,
        )
        train_loader = DataLoader(
            ConcatDataset([public_train, ImageLabelOnly(train_loader.dataset)]),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"{args.model}_best.pt"
    latest_path = output_dir / f"{args.model}_latest.pt"
    log_path = output_dir / f"{args.model}_train_log.jsonl"

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        record = {
            "epoch": epoch,
            "model": args.model,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        checkpoint = {
            "model_name": args.model,
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "args": vars(args),
        }
        torch.save(checkpoint, latest_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, best_path)
            print(f"saved best checkpoint to {best_path} (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
