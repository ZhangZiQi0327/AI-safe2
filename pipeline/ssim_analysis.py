"""SSIM 质量分析

分析对抗样本在各类别、各图片上的 SSIM 表现，找出规律和薄弱环节。

用法:
    python pipeline/ssim_analysis.py --checkpoint models_public/resnet18_best.pt
    python pipeline/ssim_analysis.py --checkpoint models_public/resnet18_best.pt --attack pgd --epsilon 8/255 --steps 20
    python pipeline/ssim_analysis.py --per-image-results output/per_image_results.json  # 从已有结果分析
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attacks import fgsm_attack, pgd_attack
from attacks.common import denormalize
from data.loader import CifarAttackDataset, build_transforms, CIFAR10_CLASSES
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


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return ckpt["model_name"], model


def collect_per_image_results(model, dataset, attack_fn, device, batch_size=64):
    """收集每张图的 SSIM 和攻击结果"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    results = []

    for images, labels, filenames in tqdm(loader, desc="Analyzing"):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_pred = model(images).argmax(dim=1)
        clean_ok = (clean_pred == labels)

        adversarial = attack_fn(model, images, labels)

        with torch.no_grad():
            adv_pred = model(adversarial).argmax(dim=1)
        adv_wrong = (adv_pred != labels)

        ssim = global_ssim(denormalize(images), denormalize(adversarial))

        for i in range(images.size(0)):
            results.append({
                "filename": filenames[i],
                "true_label": labels[i].item(),
                "true_label_name": CIFAR10_CLASSES[labels[i].item()],
                "clean_correct": clean_ok[i].item(),
                "attack_success": adv_wrong[i].item(),
                "ssim": ssim[i].item(),
            })

    return results


def analyze_ssim(results: list[dict], output_dir: str = "ssim_analysis"):
    """分析 SSIM 分布和规律"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 按类别分析 ---
    by_class = defaultdict(list)
    for r in results:
        by_class[r["true_label_name"]].append(r)

    print(f"\n{'='*60}")
    print(f"{'Category':<15} {'Count':>6} {'ASR':>8} {'Mean SSIM':>10} {'Min SSIM':>10}")
    print(f"{'='*60}")

    class_stats = []
    for cls_name in CIFAR10_CLASSES:
        items = by_class[cls_name]
        total = len(items)
        attacked = sum(1 for r in items if r["attack_success"])
        asr = attacked / max(total, 1)

        ssim_success = [r["ssim"] for r in items if r["attack_success"]]
        mean_ssim = sum(ssim_success) / max(len(ssim_success), 1)
        min_ssim = min(ssim_success) if ssim_success else 0

        print(f"{cls_name:<15} {total:>6} {asr:>8.4f} {mean_ssim:>10.6f} {min_ssim:>10.6f}")
        class_stats.append({
            "class": cls_name,
            "count": total,
            "asr": asr,
            "mean_ssim": mean_ssim,
            "min_ssim": min_ssim,
        })

    # --- 2. SSIM 分布统计 ---
    all_ssim = [r["ssim"] for r in results]
    success_ssim = [r["ssim"] for r in results if r["attack_success"]]
    fail_ssim = [r["ssim"] for r in results if not r["attack_success"]]

    print(f"\n{'='*60}")
    print(f"SSIM Distribution (all {len(all_ssim)} images):")
    print(f"{'='*60}")
    print(f"  Mean:    {sum(all_ssim)/len(all_ssim):.6f}")
    print(f"  Median:  {sorted(all_ssim)[len(all_ssim)//2]:.6f}")
    print(f"  Min:     {min(all_ssim):.6f}")
    print(f"  Max:     {max(all_ssim):.6f}")

    # SSIM 分段统计
    thresholds = [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
    print(f"\nSSIM Threshold Distribution:")
    for t in thresholds:
        above = sum(1 for s in all_ssim if s >= t)
        print(f"  SSIM >= {t}: {above}/{len(all_ssim)} ({100*above/len(all_ssim):.1f}%)")

    # --- 3. 找出 SSIM 最低的图片 ---
    sorted_by_ssim = sorted(results, key=lambda r: r["ssim"])
    print(f"\n{'='*60}")
    print(f"Bottom 10 images by SSIM:")
    print(f"{'='*60}")
    for r in sorted_by_ssim[:10]:
        status = "SUCCESS" if r["attack_success"] else "FAILED"
        print(f"  {r['filename']:<15} {r['true_label_name']:<12} "
              f"SSIM={r['ssim']:.6f} [{status}]")

    # --- 4. 找出 SSIM 最高的攻击失败图片 ---
    attack_failed = [r for r in results if not r["attack_success"]]
    if attack_failed:
        attack_failed_sorted = sorted(attack_failed, key=lambda r: r["ssim"], reverse=True)
        print(f"\n{'='*60}")
        print(f"Attack FAILED but high SSIM (potential easy targets):")
        print(f"{'='*60}")
        for r in attack_failed_sorted[:10]:
            print(f"  {r['filename']:<15} {r['true_label_name']:<12} SSIM={r['ssim']:.6f}")

    # --- 5. 找出 SSIM 极低的攻击成功图片 (需要修复) ---
    low_ssim_success = [r for r in results if r["attack_success"] and r["ssim"] < 0.98]
    if low_ssim_success:
        print(f"\n{'='*60}")
        print(f"Low SSIM + Attack SUCCESS (quality issues, {len(low_ssim_success)} images):")
        print(f"{'='*60}")
        for r in sorted(low_ssim_success, key=lambda r: r["ssim"])[:20]:
            print(f"  {r['filename']:<15} {r['true_label_name']:<12} SSIM={r['ssim']:.6f}")

    # --- 6. 保存分析结果 ---
    analysis = {
        "total_images": len(results),
        "overall_ssim": {
            "mean": sum(all_ssim) / len(all_ssim),
            "min": min(all_ssim),
            "max": max(all_ssim),
        },
        "class_stats": class_stats,
        "bottom_10_ssim": sorted_by_ssim[:10],
        "low_ssim_success_count": len(low_ssim_success),
    }

    analysis_path = output_dir / "ssim_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis saved to: {analysis_path}")

    # --- 7. 生成可视化 (可选) ---
    try:
        generate_plots(results, class_stats, output_dir)
    except ImportError:
        print("matplotlib not available, skipping plots")

    return analysis


def generate_plots(results, class_stats, output_dir):
    """生成 SSIM 分析图表"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as plt_np

    # 1. 各类别 SSIM 箱线图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    by_class = defaultdict(list)
    for r in results:
        by_class[r["true_label_name"]].append(r["ssim"])

    class_names = list(CIFAR10_CLASSES)
    class_ssim = [by_class[c] for c in class_names]

    axes[0].boxplot(class_ssim, labels=class_names, showmeans=True)
    axes[0].set_ylabel("SSIM")
    axes[0].set_title("SSIM Distribution by Class")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.3)

    # 2. 各类别 ASR vs Mean SSIM
    classes = [s["class"] for s in class_stats]
    asrs = [s["asr"] for s in class_stats]
    ssims = [s["mean_ssim"] for s in class_stats]

    axes[1].scatter(asrs, ssims, s=100, c=range(len(classes)), cmap="tab10")
    for i, c in enumerate(classes):
        axes[1].annotate(c, (asrs[i], ssims[i]), fontsize=8, ha="center", va="bottom")
    axes[1].set_xlabel("ASR")
    axes[1].set_ylabel("Mean SSIM")
    axes[1].set_title("ASR vs SSIM by Class")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ssim_by_class.png", dpi=150)
    plt.close()

    # 3. SSIM 直方图
    fig, ax = plt.subplots(figsize=(8, 4))
    all_ssim = [r["ssim"] for r in results]
    ax.hist(all_ssim, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(sum(all_ssim)/len(all_ssim), color="red", linestyle="--", label=f"Mean: {sum(all_ssim)/len(all_ssim):.4f}")
    ax.set_xlabel("SSIM")
    ax.set_ylabel("Count")
    ax.set_title("SSIM Distribution (All Images)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ssim_histogram.png", dpi=150)
    plt.close()

    print(f"Plots saved to: {output_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description="SSIM 质量分析")
    parser.add_argument("--checkpoint", help="模型权重路径（用于在线分析）")
    parser.add_argument("--per-image-results", help="已有 per_image_results.json（用于离线分析）")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output", default="ssim_analysis", help="分析结果输出目录")
    parser.add_argument("--attack", choices=["fgsm", "pgd"], default="pgd")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.per_image_results:
        # 从已有结果文件分析
        with open(args.per_image_results, encoding="utf-8") as f:
            results = json.load(f)
        analyze_ssim(results, output_dir=args.output)
    elif args.checkpoint:
        # 在线分析：运行攻击 + 分析
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name, model = load_model(args.checkpoint, device)
        dataset = CifarAttackDataset(args.dataset, transform=build_transforms(train=False))

        if args.attack == "fgsm":
            attack_fn = lambda m, x, y: fgsm_attack(m, x, y, epsilon=args.epsilon)
        else:
            attack_fn = lambda m, x, y: pgd_attack(
                m, x, y, epsilon=args.epsilon, alpha=args.alpha,
                steps=args.steps, random_start=True
            )

        print(f"Analyzing {model_name} with {args.attack.upper()}...")
        results = collect_per_image_results(model, dataset, attack_fn, device, args.batch_size)
        analyze_ssim(results, output_dir=args.output)
    else:
        print("Error: provide --checkpoint or --per-image-results")
