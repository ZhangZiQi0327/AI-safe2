from __future__ import annotations

import torch

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
