"""批量对抗样本生成管线

用法:
    # 单模型攻击
    python pipeline/generate.py --checkpoint models_public/resnet18_best.pt --attack pgd
    python pipeline/generate.py --checkpoint models_public/resnet18_best.pt --attack pgd --epsilon 8/255 --steps 20

    # 集成攻击（多模型）
    python pipeline/generate.py --attack ensemble \
        --checkpoints models_public/resnet34_best.pt models_public/vgg16_best.pt models_public/densenet121_best.pt \
        --epsilon 0.0392 --alpha 0.0078 --steps 12 --decay 1.0 \
        --diversity-prob 0.3 --resize-rate 0.85
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attacks import fgsm_attack, pgd_attack, mi_fgsm_attack, ensemble_attack
from attacks.common import denormalize
from data.loader import CifarAttackDataset, build_transforms, CIFAR10_CLASSES
from models import build_model


# --- SSIM 计算 ---

def global_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """逐样本计算全局 SSIM，x/y shape: (N, C, H, W) in [0,1]."""
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


# --- 模型加载 ---

def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model_name, model


# --- 模型加载 ---

def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model_name, model


def load_checkpoints(paths: list[str | Path], device: torch.device) -> list:
    """加载多个模型用于集成攻击"""
    models = []
    names = []
    for p in paths:
        name, model = load_checkpoint(p, device)
        models.append(model)
        names.append(name)
    return models, names


# --- 攻击函数路由 ---

def apply_attack(model, images, labels, attack: str, **kwargs) -> torch.Tensor:
    """单模型攻击（fgsm/pgd/mi_fgsm）"""
    if attack == "fgsm":
        return fgsm_attack(model, images, labels, epsilon=kwargs.get("epsilon", 8 / 255))
    elif attack == "pgd":
        return pgd_attack(
            model, images, labels,
            epsilon=kwargs.get("epsilon", 8 / 255),
            alpha=kwargs.get("alpha", 2 / 255),
            steps=kwargs.get("steps", 10),
            random_start=kwargs.get("random_start", True),
        )
    elif attack == "mi_fgsm":
        return mi_fgsm_attack(
            model, images, labels,
            epsilon=kwargs.get("epsilon", 8 / 255),
            alpha=kwargs.get("alpha", 2 / 255),
            steps=kwargs.get("steps", 10),
            decay=kwargs.get("decay", 1.0),
            diversity_prob=kwargs.get("diversity_prob", 0.0),
            resize_rate=kwargs.get("resize_rate", 0.9),
        )
    else:
        raise ValueError(f"Unknown single-model attack: {attack}")


def apply_ensemble_attack(models, images, labels, **kwargs) -> torch.Tensor:
    """集成攻击（多模型）"""
    return ensemble_attack(
        models, images, labels,
        epsilon=kwargs.get("epsilon", 8 / 255),
        alpha=kwargs.get("alpha", 2 / 255),
        steps=kwargs.get("steps", 10),
        decay=kwargs.get("decay", 1.0),
        random_start=kwargs.get("random_start", False),
        diversity_prob=kwargs.get("diversity_prob", 0.0),
        resize_rate=kwargs.get("resize_rate", 0.9),
    )


# --- 主流程 ---

def generate_adversarial_images(
    dataset_root: str | Path = "dataset",
    output_dir: str | Path = "output",
    attack: str = "pgd",
    # 单模型参数
    checkpoint_path: str | Path | None = None,
    # 集成攻击参数
    checkpoints: list[str | Path] | None = None,
    # 共享攻击参数
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    random_start: bool = True,
    decay: float = 1.0,
    diversity_prob: float = 0.0,
    resize_rate: float = 0.9,
    batch_size: int = 64,
    num_workers: int = 0,
    save_images: bool = True,
) -> dict:
    """
    完整的对抗样本生成管线:
    1. 加载模型和数据
    2. 对500张图执行攻击
    3. 计算ASR和SSIM
    4. 保存对抗样本为PNG
    5. 打包zip
    6. 返回评估结果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    # 自动追加时间戳避免覆盖，如 output_ensemble → output_ensemble_20260529_143000
    if output_dir.exists() and any(output_dir.iterdir()):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir.parent / f"{output_dir.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    is_ensemble = attack == "ensemble"
    if is_ensemble:
        if not checkpoints:
            raise ValueError("Ensemble attack requires --checkpoints")
        models, model_names = load_checkpoints(checkpoints, device)
        model_display = "+".join(model_names)
        print(f"Loaded ensemble: {model_display}")
    else:
        if not checkpoint_path:
            raise ValueError("Single-model attack requires --checkpoint")
        model_name, model = load_checkpoint(checkpoint_path, device)
        model_display = model_name
        print(f"Loaded model: {model_name}")

    # 加载数据
    dataset = CifarAttackDataset(dataset_root, transform=build_transforms(train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 攻击参数
    attack_kwargs = {
        "epsilon": epsilon,
        "alpha": alpha,
        "steps": steps,
        "random_start": random_start,
        "decay": decay,
        "diversity_prob": diversity_prob,
        "resize_rate": resize_rate,
    }

    # 逐样本统计
    results = {
        "model": model_display,
        "attack": attack,
        "params": attack_kwargs,
        "per_image": [],
        "total": 0,
        "clean_correct": 0,
        "attack_success": 0,
        "ssim_sum": 0.0,
        "ssim_count": 0,
    }

    print(f"Running {attack.upper()} on {len(dataset)} images...")
    t0 = time.time()

    for images, labels, filenames in tqdm(loader, desc="Generating"):
        images = images.to(device)
        labels = labels.to(device)

        # 白盒推理（用第一个模型或集成判断 clean accuracy）
        with torch.no_grad():
            if is_ensemble:
                from attacks.ensemble import ensemble_logits
                clean_logits = ensemble_logits(models, images)
            else:
                clean_logits = model(images)
        clean_pred = clean_logits.argmax(dim=1)
        clean_ok = (clean_pred == labels)

        # 攻击
        if is_ensemble:
            adversarial = apply_ensemble_attack(models, images, labels, **attack_kwargs)
        else:
            adversarial = apply_attack(model, images, labels, attack, **attack_kwargs)

        # 攻击后推理
        with torch.no_grad():
            if is_ensemble:
                adv_logits = ensemble_logits(models, images)
            else:
                adv_logits = model(adversarial)
        adv_pred = adv_logits.argmax(dim=1)
        adv_wrong = (adv_pred != labels)

        # SSIM (在像素空间计算)
        raw_images = denormalize(images)
        raw_adversarial = denormalize(adversarial)
        ssim_per_image = global_ssim(raw_images, raw_adversarial)

        # 记录逐样本结果
        for i in range(images.size(0)):
            fname = filenames[i]
            label = labels[i].item()
            clean_pred_i = clean_pred[i].item()
            adv_pred_i = adv_pred[i].item()
            ssim_i = ssim_per_image[i].item()

            results["per_image"].append({
                "filename": fname,
                "true_label": label,
                "true_label_name": CIFAR10_CLASSES[label],
                "clean_pred": clean_pred_i,
                "clean_pred_name": CIFAR10_CLASSES[clean_pred_i],
                "clean_correct": clean_ok[i].item(),
                "adv_pred": adv_pred_i,
                "adv_pred_name": CIFAR10_CLASSES[adv_pred_i],
                "attack_success": adv_wrong[i].item(),
                "ssim": ssim_i,
            })

            results["total"] += 1
            if clean_ok[i].item():
                results["clean_correct"] += 1
            if adv_wrong[i].item():
                results["attack_success"] += 1
                results["ssim_sum"] += ssim_i
                results["ssim_count"] += 1

        # 保存对抗样本图片
        if save_images:
            adv_pixel = raw_adversarial.cpu()
            for i in range(images.size(0)):
                fname = filenames[i]
                img_array = (adv_pixel[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
                Image.fromarray(img_array).save(output_dir / fname)

    elapsed = time.time() - t0

    # 汇总指标
    results["clean_acc"] = results["clean_correct"] / max(results["total"], 1)
    results["asr"] = results["attack_success"] / max(results["total"], 1)
    results["mean_ssim"] = results["ssim_sum"] / max(results["ssim_count"], 1)
    results["proxy_score_m"] = 100 * results["asr"] * results["mean_ssim"]
    results["elapsed_sec"] = elapsed

    # 打印结果
    print(f"\n{'='*50}")
    print(f"Model:      {model_display}")
    print(f"Attack:     {attack.upper()}")
    print(f"Clean Acc:  {results['clean_acc']:.4f}")
    print(f"ASR:        {results['asr']:.4f}")
    print(f"Mean SSIM:  {results['mean_ssim']:.6f}")
    print(f"Score M:    {results['proxy_score_m']:.2f}")
    print(f"Time:       {elapsed:.1f}s")
    print(f"{'='*50}")

    # 保存详细结果 JSON
    summary = {k: v for k, v in results.items() if k != "per_image"}
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    per_image_path = output_dir / "per_image_results.json"
    with open(per_image_path, "w", encoding="utf-8") as f:
        json.dump(results["per_image"], f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_dir}/")

    # 打包zip（images/ 子目录 + label.txt）
    if save_images:
        zip_path = output_dir / "adversarial_images.zip"
        label_src = Path(dataset_root) / "label.txt"
        print(f"Creating zip: {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for img_file in sorted(output_dir.glob("*.png")):
                zf.write(img_file, f"images/{img_file.name}")
            if label_src.exists():
                zf.write(label_src, "label.txt")
        print(f"Zip created: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")

    return results


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(description="批量生成对抗样本")
    parser.add_argument("--checkpoint", help="单模型权重路径")
    parser.add_argument("--checkpoints", nargs="+", help="集成攻击的多个模型权重路径")
    parser.add_argument("--dataset", default="dataset", help="数据集目录")
    parser.add_argument("--output", default="output", help="输出目录")
    parser.add_argument("--attack", choices=["fgsm", "pgd", "mi_fgsm", "ensemble"], default="pgd")
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--decay", type=float, default=1.0, help="MI-FGSM/Ensemble momentum decay")
    parser.add_argument("--diversity-prob", type=float, default=0.0, help="Input diversity probability")
    parser.add_argument("--resize-rate", type=float, default=0.9, help="Input diversity resize rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-save", action="store_true", help="不保存图片，只计算指标")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_adversarial_images(
        dataset_root=args.dataset,
        output_dir=args.output,
        attack=args.attack,
        checkpoint_path=args.checkpoint,
        checkpoints=args.checkpoints,
        epsilon=args.epsilon,
        alpha=args.alpha,
        steps=args.steps,
        random_start=not args.no_random_start,
        decay=args.decay,
        diversity_prob=args.diversity_prob,
        resize_rate=args.resize_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_images=not args.no_save,
    )
