from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import clamp_normalized, raw_eps_to_normalized


def fgsm_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
) -> torch.Tensor:
    """FGSM on normalized tensors. epsilon is measured in raw pixel scale [0, 1]."""

    model.eval()
    x = images.detach().clone().requires_grad_(True)
    labels = labels.to(images.device)

    logits = model(x)
    loss = F.cross_entropy(logits, labels)
    model.zero_grad(set_to_none=True)
    loss.backward()

    step = raw_eps_to_normalized(epsilon, images.device)
    adversarial = x + step * x.grad.detach().sign()
    return clamp_normalized(adversarial).detach()
