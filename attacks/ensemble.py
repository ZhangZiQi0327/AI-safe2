from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .common import clamp_normalized, input_diversity, project_linf, raw_eps_to_normalized


def ensemble_logits(models: Sequence[torch.nn.Module], images: torch.Tensor) -> torch.Tensor:
    if not models:
        raise ValueError("Expected at least one model for ensemble inference.")

    logits_sum = None
    for model in models:
        model.eval()
        logits = model(images)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / len(models)


def ensemble_attack(
    models: Sequence[torch.nn.Module],
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    decay: float = 1.0,
    random_start: bool = False,
    diversity_prob: float = 0.0,
    resize_rate: float = 0.9,
) -> torch.Tensor:
    """Iterative ensemble attack with averaged source-model gradients."""

    if not models:
        raise ValueError("Expected at least one model for ensemble attack.")

    labels = labels.to(images.device)
    original = images.detach()
    step_size = raw_eps_to_normalized(alpha, images.device)

    if random_start:
        noise = torch.empty_like(original).uniform_(-1.0, 1.0) * raw_eps_to_normalized(epsilon, images.device)
        adversarial = clamp_normalized(original + noise)
    else:
        adversarial = original.clone()

    momentum = torch.zeros_like(original)
    for _ in range(max(steps, 0)):
        adversarial = adversarial.detach().requires_grad_(True)
        transformed = input_diversity(adversarial, diversity_prob=diversity_prob, resize_rate=resize_rate)

        loss = 0.0
        for model in models:
            model.eval()
            loss = loss + F.cross_entropy(model(transformed), labels)
        loss = loss / len(models)

        for model in models:
            model.zero_grad(set_to_none=True)
        loss.backward()

        grad = adversarial.grad.detach()
        grad = grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        momentum = decay * momentum + grad

        adversarial = adversarial + step_size * momentum.sign()
        adversarial = project_linf(adversarial, original, epsilon)
        adversarial = clamp_normalized(adversarial)

    return adversarial.detach()
