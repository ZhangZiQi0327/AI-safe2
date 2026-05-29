from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from attacks import ensemble_attack, ensemble_logits, fgsm_attack, mi_fgsm_attack, pgd_attack
from attacks.common import denormalize
from data.loader import CifarAttackDataset, build_transforms
from models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate transferability of adversarial attacks.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--source-checkpoints", nargs="+", required=True)
    parser.add_argument("--target-checkpoints", nargs="+", required=True)
    parser.add_argument("--attack", choices=("fgsm", "pgd", "mi_fgsm", "ensemble"), required=True)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--momentum", type=float, default=1.0)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--diversity-prob", type=float, default=0.0)
    parser.add_argument("--resize-rate", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="transfer_results.csv")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: str | Path, device: torch.device, cache: dict[str, tuple[str, torch.nn.Module, dict]]):
    key = str(Path(path))
    if key in cache:
        return cache[key]

    checkpoint = torch.load(key, map_location=device)
    model_name = checkpoint["model_name"]
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    cache[key] = (model_name, model, checkpoint)
    return cache[key]


def global_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.flatten(start_dim=1)
    y = y.flatten(start_dim=1)
    c1 = 0.01**2
    c2 = 0.03**2
    mux = x.mean(dim=1)
    muy = y.mean(dim=1)
    varx = x.var(dim=1, unbiased=True)
    vary = y.var(dim=1, unbiased=True)
    cov = ((x - mux[:, None]) * (y - muy[:, None])).sum(dim=1) / (x.size(1) - 1)
    numerator = (2 * mux * muy + c1) * (2 * cov + c2)
    denominator = (mux.square() + muy.square() + c1) * (varx + vary + c2)
    return numerator / denominator


@torch.no_grad()
def predict(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    return model(images).argmax(dim=1)


@torch.no_grad()
def predict_source(models: list[torch.nn.Module], images: torch.Tensor, use_ensemble: bool) -> torch.Tensor:
    if use_ensemble:
        return ensemble_logits(models, images).argmax(dim=1)
    return models[0](images).argmax(dim=1)


def generate_attack(
    attack_name: str,
    source_models: list[torch.nn.Module],
    images: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if attack_name == "fgsm":
        return fgsm_attack(source_models[0], images, labels, epsilon=args.epsilon)
    if attack_name == "pgd":
        return pgd_attack(
            source_models[0],
            images,
            labels,
            epsilon=args.epsilon,
            alpha=args.alpha,
            steps=args.steps,
            random_start=args.random_start,
        )
    if attack_name == "mi_fgsm":
        return mi_fgsm_attack(
            source_models[0],
            images,
            labels,
            epsilon=args.epsilon,
            alpha=args.alpha,
            steps=args.steps,
            decay=args.momentum,
            random_start=args.random_start,
            diversity_prob=args.diversity_prob,
            resize_rate=args.resize_rate,
        )
    return ensemble_attack(
        source_models,
        images,
        labels,
        epsilon=args.epsilon,
        alpha=args.alpha,
        steps=args.steps,
        decay=args.momentum,
        random_start=args.random_start,
        diversity_prob=args.diversity_prob,
        resize_rate=args.resize_rate,
    )


def empty_stats() -> dict[str, float]:
    return {
        "total": 0,
        "source_clean_correct": 0,
        "target_clean_correct": 0,
        "joint_clean_correct": 0,
        "adv_wrong_all": 0,
        "transfer_success_target_clean": 0,
        "transfer_success_joint_clean": 0,
        "ssim_sum_target_clean": 0.0,
        "ssim_sum_joint_clean": 0.0,
        "ssim_count_target_clean": 0,
        "ssim_count_joint_clean": 0,
    }


def evaluate_transfer(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.attack != "ensemble" and len(args.source_checkpoints) != 1:
        raise ValueError("Single-model attacks require exactly one source checkpoint.")
    if args.attack == "ensemble" and len(args.source_checkpoints) < 2:
        raise ValueError("Ensemble attack requires at least two source checkpoints.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache: dict[str, tuple[str, torch.nn.Module, dict]] = {}

    source_entries = [load_checkpoint(path, device, cache) for path in args.source_checkpoints]
    source_names = [entry[0] for entry in source_entries]
    source_models = [entry[1] for entry in source_entries]
    target_entries = [load_checkpoint(path, device, cache) for path in args.target_checkpoints]

    dataset = CifarAttackDataset(args.dataset, transform=build_transforms(train=False))
    if args.max_samples > 0:
        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    results = {
        str(Path(path)): {
            "target_name": name,
            "checkpoint": checkpoint,
            "model": model,
            "stats": empty_stats(),
        }
        for path, (name, model, checkpoint) in zip(args.target_checkpoints, target_entries)
    }
    use_ensemble_source = args.attack == "ensemble"

    for images, labels, _ in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        source_clean_pred = predict_source(source_models, images, use_ensemble=use_ensemble_source)
        source_clean_ok = source_clean_pred == labels
        adversarial = generate_attack(args.attack, source_models, images, labels, args)
        raw_ssim = global_ssim(denormalize(images), denormalize(adversarial))

        for target_path, info in results.items():
            target_model = info["model"]
            stats = info["stats"]

            target_clean_pred = predict(target_model, images)
            adv_pred = predict(target_model, adversarial)

            target_clean_ok = target_clean_pred == labels
            adv_wrong = adv_pred != labels
            joint_clean_ok = source_clean_ok & target_clean_ok
            target_success = target_clean_ok & adv_wrong
            joint_success = joint_clean_ok & adv_wrong

            stats["total"] += labels.numel()
            stats["source_clean_correct"] += source_clean_ok.sum().item()
            stats["target_clean_correct"] += target_clean_ok.sum().item()
            stats["joint_clean_correct"] += joint_clean_ok.sum().item()
            stats["adv_wrong_all"] += adv_wrong.sum().item()
            stats["transfer_success_target_clean"] += target_success.sum().item()
            stats["transfer_success_joint_clean"] += joint_success.sum().item()
            stats["ssim_sum_target_clean"] += raw_ssim[target_success].sum().item()
            stats["ssim_sum_joint_clean"] += raw_ssim[joint_success].sum().item()
            stats["ssim_count_target_clean"] += target_success.sum().item()
            stats["ssim_count_joint_clean"] += joint_success.sum().item()

    source_label = "+".join(source_names)
    rows: list[dict[str, object]] = []
    for target_path, info in results.items():
        stats = info["stats"]
        total = max(int(stats["total"]), 1)
        source_clean_acc = stats["source_clean_correct"] / total
        target_clean_acc = stats["target_clean_correct"] / total
        joint_clean_acc = stats["joint_clean_correct"] / total
        adv_error_rate_all = stats["adv_wrong_all"] / total
        transfer_asr_target_clean = stats["transfer_success_target_clean"] / max(stats["target_clean_correct"], 1)
        transfer_asr_joint_clean = stats["transfer_success_joint_clean"] / max(stats["joint_clean_correct"], 1)
        mean_ssim_target_clean = stats["ssim_sum_target_clean"] / max(stats["ssim_count_target_clean"], 1)
        mean_ssim_joint_clean = stats["ssim_sum_joint_clean"] / max(stats["ssim_count_joint_clean"], 1)

        row = {
            "source_models": source_label,
            "source_checkpoints": "|".join(str(Path(path)) for path in args.source_checkpoints),
            "target_model": info["target_name"],
            "target_checkpoint": target_path,
            "attack": args.attack,
            "epsilon": args.epsilon,
            "alpha": args.alpha if args.attack != "fgsm" else "",
            "steps": args.steps if args.attack != "fgsm" else "",
            "momentum": args.momentum if args.attack in {"mi_fgsm", "ensemble"} else "",
            "random_start": args.random_start if args.attack in {"pgd", "mi_fgsm", "ensemble"} else "",
            "diversity_prob": args.diversity_prob if args.attack in {"mi_fgsm", "ensemble"} else "",
            "resize_rate": args.resize_rate if args.attack in {"mi_fgsm", "ensemble"} else "",
            "samples": total,
            "source_clean_acc": source_clean_acc,
            "target_clean_acc": target_clean_acc,
            "joint_clean_acc": joint_clean_acc,
            "adv_error_rate_all": adv_error_rate_all,
            "transfer_asr_target_clean": transfer_asr_target_clean,
            "transfer_asr_joint_clean": transfer_asr_joint_clean,
            "mean_ssim_target_clean": mean_ssim_target_clean,
            "mean_ssim_joint_clean": mean_ssim_joint_clean,
            "transfer_score_target_clean": 100 * transfer_asr_target_clean * mean_ssim_target_clean,
            "transfer_score_joint_clean": 100 * transfer_asr_joint_clean * mean_ssim_joint_clean,
        }
        print(row)
        rows.append(row)

    return rows


def write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = evaluate_transfer(args)
    output_path = Path(args.output)
    write_rows(output_path, rows)


if __name__ == "__main__":
    main()
