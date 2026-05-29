"""参数网格搜索

系统搜索 epsilon、alpha、steps、momentum 等参数的最优组合。
输出到 CSV，按 proxy_score_m 排序。

用法:
    python pipeline/grid_search.py --checkpoint models_public/resnet18_best.pt
    python pipeline/grid_search.py --checkpoint models_public/resnet18_best.pt --attack pgd --fast
"""
from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attacks import fgsm_attack, pgd_attack
from attacks.common import denormalize
from data.loader import CifarAttackDataset, build_transforms
from models import build_model


def global_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.flatten(start_dim=1)
    y = y.flatten(start_dim=1)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mux = x.mean(dim=1)
    muy = y.mean(dim=1)
    varx = x.var(dim=1, unbiased=True)
    vary = y.var(dim=1, unbiased=True)
    cov = ((x - mux[:, None]) * (y - muy[:, None])).sum(dim=1) / (x.size(1) - 1)
    numerator = (2 * mux * muy + c1) * (2 * cov + c2)
    denominator = (mux.square() + muy.square() + c1) * (varx + vary + c2)
    return numerator / denominator


def load_model(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return ckpt["model_name"], model


@torch.no_grad()
def evaluate_attack(model, dataset, attack_fn, batch_size=64, device="cpu"):
    """对整个数据集运行攻击，返回 (asr, mean_ssim, proxy_score_m, per_image_results)"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    total = 0
    clean_correct = 0
    attack_success = 0
    ssim_sum = 0.0
    ssim_count = 0
    per_image = []

    for images, labels, filenames in loader:
        images = images.to(device)
        labels = labels.to(device)

        clean_pred = model(images).argmax(dim=1)
        clean_ok = (clean_pred == labels)

        adversarial = attack_fn(model, images, labels)

        adv_pred = model(adversarial).argmax(dim=1)
        adv_wrong = (adv_pred != labels)

        ssim = global_ssim(denormalize(images), denormalize(adversarial))

        for i in range(images.size(0)):
            total += 1
            is_clean = clean_ok[i].item()
            is_attacked = adv_wrong[i].item()
            if is_clean:
                clean_correct += 1
            if is_attacked:
                attack_success += 1
                ssim_sum += ssim[i].item()
                ssim_count += 1

    asr = attack_success / max(total, 1)
    mean_ssim = ssim_sum / max(ssim_count, 1)
    proxy_score = 100 * asr * mean_ssim
    return asr, mean_ssim, proxy_score


def fgsm_grid():
    """FGSM 参数网格"""
    epsilons = [4/255, 6/255, 8/255, 10/255, 12/255, 16/255]
    return [{"epsilon": e} for e in epsilons]


def pgd_grid(fast=False):
    """PGD 参数网格"""
    if fast:
        # 快速模式：少量组合
        epsilons = [6/255, 8/255, 10/255]
        alphas = [1/255, 2/255]
        steps_list = [5, 10, 20]
    else:
        # 完整模式
        epsilons = [4/255, 6/255, 8/255, 10/255, 12/255, 16/255]
        alphas = [1/255, 2/255, 4/255]
        steps_list = [5, 10, 15, 20, 30]

    combos = list(itertools.product(epsilons, alphas, steps_list))
    # 过滤掉 alpha > epsilon 的无效组合
    valid = [(e, a, s) for e, a, s in combos if a <= e]
    return [{"epsilon": e, "alpha": a, "steps": s} for e, a, s in valid]


def run_grid_search(
    checkpoint_path: str | Path,
    dataset_root: str = "dataset",
    attack: str = "pgd",
    fast: bool = False,
    output_csv: str = "grid_search_results.csv",
    batch_size: int = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name, model = load_model(checkpoint_path, device)
    dataset = CifarAttackDataset(dataset_root, transform=build_transforms(train=False))

    # 构建参数网格
    if attack == "fgsm":
        param_grid = fgsm_grid()
    else:
        param_grid = pgd_grid(fast=fast)

    print(f"Grid search: {model_name} | {attack.upper()} | {len(param_grid)} combinations")
    print(f"Device: {device}")

    results = []
    t0 = time.time()

    for i, params in enumerate(param_grid):
        eps = params["epsilon"]
        alpha = params.get("alpha", 2/255)
        steps = params.get("steps", 10)

        if attack == "fgsm":
            desc = f"eps={eps:.4f}"
            attack_fn = lambda m, x, y, _eps=eps: fgsm_attack(m, x, y, epsilon=_eps)
        else:
            desc = f"eps={eps:.4f} a={alpha:.4f} s={steps}"
            attack_fn = lambda m, x, y, _e=eps, _a=alpha, _s=steps: pgd_attack(
                m, x, y, epsilon=_e, alpha=_a, steps=_s, random_start=True
            )

        print(f"[{i+1}/{len(param_grid)}] {desc}...", end=" ", flush=True)
        asr, mean_ssim, proxy_score = evaluate_attack(
            model, dataset, attack_fn, batch_size=batch_size, device=device
        )
        print(f"ASR={asr:.4f} SSIM={mean_ssim:.6f} M={proxy_score:.2f}")

        row = {
            "model": model_name,
            "attack": attack,
            **params,
            "asr": asr,
            "mean_ssim": mean_ssim,
            "proxy_score_m": proxy_score,
        }
        results.append(row)

    elapsed = time.time() - t0

    # 按 proxy_score_m 降序排列
    results.sort(key=lambda r: r["proxy_score_m"], reverse=True)

    # 写入CSV
    output_path = Path(output_csv)
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 打印 Top 10
    print(f"\n{'='*70}")
    print(f"Top 10 results (sorted by proxy_score_m):")
    print(f"{'='*70}")
    for i, r in enumerate(results[:10]):
        print(f"  #{i+1}: eps={r.get('epsilon',0):.4f} "
              f"alpha={r.get('alpha',0):.4f} steps={r.get('steps','-')} "
              f"| ASR={r['asr']:.4f} SSIM={r['mean_ssim']:.6f} M={r['proxy_score_m']:.2f}")

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to: {output_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="参数网格搜索")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--attack", choices=["fgsm", "pgd"], default="pgd")
    parser.add_argument("--fast", action="store_true", help="快速模式，减少参数组合")
    parser.add_argument("--output", default="grid_search_results.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_grid_search(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset,
        attack=args.attack,
        fast=args.fast,
        output_csv=args.output,
        batch_size=args.batch_size,
    )
