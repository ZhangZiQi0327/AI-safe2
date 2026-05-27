from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import clamp_normalized, raw_eps_to_normalized


def pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    random_start: bool = True,
) -> torch.Tensor:
    """Untargeted PGD on normalized tensors. epsilon and alpha use raw pixel scale [0, 1]."""

    model.eval()
    labels = labels.to(images.device)
    eps = raw_eps_to_normalized(epsilon, images.device)
    step_size = raw_eps_to_normalized(alpha, images.device)

    original = images.detach()
    if random_start:
        noise = torch.empty_like(original).uniform_(-1.0, 1.0) * eps
        adversarial = clamp_normalized(original + noise)
    else:
        adversarial = original.clone()

    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        logits = model(adversarial)
        loss = F.cross_entropy(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()

        adversarial = adversarial + step_size * adversarial.grad.detach().sign()
        adversarial = torch.max(torch.min(adversarial, original + eps), original - eps)
        adversarial = clamp_normalized(adversarial)

    return adversarial.detach()
