from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from data.loader import CIFAR10_MEAN, CIFAR10_STD


def channel_tensor(values, device) -> torch.Tensor:
    return torch.tensor(values, device=device).view(1, 3, 1, 1)


def normalized_bounds(device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = channel_tensor(CIFAR10_MEAN, device)
    std = channel_tensor(CIFAR10_STD, device)
    return (0.0 - mean) / std, (1.0 - mean) / std


def raw_eps_to_normalized(epsilon: float, device) -> torch.Tensor:
    std = channel_tensor(CIFAR10_STD, device)
    return torch.tensor(float(epsilon), device=device) / std


def clamp_normalized(images: torch.Tensor) -> torch.Tensor:
    lower, upper = normalized_bounds(images.device)
    return torch.max(torch.min(images, upper), lower)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = channel_tensor(CIFAR10_MEAN, images.device)
    std = channel_tensor(CIFAR10_STD, images.device)
    return torch.clamp(images * std + mean, 0.0, 1.0)


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = channel_tensor(CIFAR10_MEAN, images.device)
    std = channel_tensor(CIFAR10_STD, images.device)
    return (images - mean) / std


def project_linf(adversarial: torch.Tensor, original: torch.Tensor, epsilon: float) -> torch.Tensor:
    eps = raw_eps_to_normalized(epsilon, adversarial.device)
    return torch.max(torch.min(adversarial, original + eps), original - eps)


def input_diversity(
    images: torch.Tensor,
    diversity_prob: float = 0.0,
    resize_rate: float = 0.9,
) -> torch.Tensor:
    """Apply differentiable random resize-and-pad on raw images, then renormalize."""

    if diversity_prob <= 0.0 or random.random() > diversity_prob:
        return images

    raw = denormalize(images)
    _, _, height, width = raw.shape
    min_height = max(1, min(height, int(round(height * resize_rate))))
    min_width = max(1, min(width, int(round(width * resize_rate))))
    if min_height >= height and min_width >= width:
        return images

    resize_height = random.randint(min_height, height)
    resize_width = random.randint(min_width, width)
    if resize_height == height and resize_width == width:
        return images

    resized = F.interpolate(raw, size=(resize_height, resize_width), mode="bilinear", align_corners=False)
    pad_top = random.randint(0, height - resize_height)
    pad_left = random.randint(0, width - resize_width)
    pad_bottom = height - resize_height - pad_top
    pad_right = width - resize_width - pad_left
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
    return normalize(padded)
