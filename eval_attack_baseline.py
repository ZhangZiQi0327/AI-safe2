from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks import fgsm_attack, pgd_attack
from attacks.common import denormalize
from data.loader import CifarAttackDataset, build_transforms
from models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FGSM/PGD baseline on a trained proxy model.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--attack", choices=("fgsm", "pgd"), required=True)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default="baseline_results.csv")
    return parser.parse_args()


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model_name = checkpoint["model_name"]
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model_name, model, checkpoint


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
def predict(model, images):
    return model(images).argmax(dim=1)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name, model, checkpoint = load_checkpoint(args.checkpoint, device)

    dataset = CifarAttackDataset(args.dataset, transform=build_transforms(train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    total = 0
    clean_correct = 0
    attacked_wrong = 0
    clean_correct_attacked_wrong = 0
    ssim_sum = 0.0
    ssim_count = 0

    for images, labels, _ in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        clean_pred = predict(model, images)
        clean_ok = clean_pred == labels

        if args.attack == "fgsm":
            adversarial = fgsm_attack(model, images, labels, epsilon=args.epsilon)
        else:
            adversarial = pgd_attack(
                model,
                images,
                labels,
                epsilon=args.epsilon,
                alpha=args.alpha,
                steps=args.steps,
                random_start=True,
            )

        adv_pred = predict(model, adversarial)
        adv_wrong = adv_pred != labels
        raw_ssim = global_ssim(denormalize(images), denormalize(adversarial))

        total += labels.numel()
        clean_correct += clean_ok.sum().item()
        attacked_wrong += adv_wrong.sum().item()
        clean_correct_attacked_wrong += (clean_ok & adv_wrong).sum().item()
        ssim_sum += raw_ssim[adv_wrong].sum().item()
        ssim_count += adv_wrong.sum().item()

    clean_acc = clean_correct / total
    adv_error_rate = attacked_wrong / total
    asr_on_clean = clean_correct_attacked_wrong / max(clean_correct, 1)
    mean_ssim_success = ssim_sum / max(ssim_count, 1)
    proxy_score_m = 100 * asr_on_clean * mean_ssim_success

    row = {
        "model": model_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch", ""),
        "attack": args.attack,
        "epsilon": args.epsilon,
        "alpha": args.alpha if args.attack == "pgd" else "",
        "steps": args.steps if args.attack == "pgd" else "",
        "clean_acc": clean_acc,
        "adv_error_rate_all": adv_error_rate,
        "asr_on_clean": asr_on_clean,
        "mean_ssim_success": mean_ssim_success,
        "proxy_score_m": proxy_score_m,
    }
    print(row)

    output_path = Path(args.output)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
