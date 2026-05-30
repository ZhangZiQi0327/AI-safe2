from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .common import clamp_normalized, input_diversity, project_linf, raw_eps_to_normalized, smooth_gradients


def _normalized_weights(
    models: Sequence[torch.nn.Module],
    model_weights: Sequence[float] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if model_weights is None:
        return torch.full((len(models),), 1.0 / len(models), device=device, dtype=dtype)

    if len(model_weights) != len(models):
        raise ValueError("model_weights length must match number of models.")

    weights = torch.tensor(model_weights, device=device, dtype=dtype)
    weights = weights / weights.sum().clamp_min(1e-12)
    return weights


def ensemble_logits(
    models: Sequence[torch.nn.Module],
    images: torch.Tensor,
    model_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    if not models:
        raise ValueError("Expected at least one model for ensemble inference.")

    weights = _normalized_weights(models, model_weights, device=images.device, dtype=images.dtype)
    logits_sum = None
    for index, model in enumerate(models):
        model.eval()
        logits = model(images) * weights[index]
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum


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
    ti_kernel_size: int = 0,
    ti_sigma: float = 1.0,
    model_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Iterative ensemble attack with averaged source-model gradients."""

    if not models:
        raise ValueError("Expected at least one model for ensemble attack.")

    labels = labels.to(images.device)
    original = images.detach()
    step_size = raw_eps_to_normalized(alpha, images.device)
    weights = _normalized_weights(models, model_weights, device=images.device, dtype=images.dtype)

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
        for index, model in enumerate(models):
            model.eval()
            loss = loss + weights[index] * F.cross_entropy(model(transformed), labels)

        for model in models:
            model.zero_grad(set_to_none=True)
        loss.backward()

        grad = adversarial.grad.detach()
        grad = smooth_gradients(grad, kernel_size=ti_kernel_size, sigma=ti_sigma)
        grad = grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        momentum = decay * momentum + grad

        adversarial = adversarial + step_size * momentum.sign()
        adversarial = project_linf(adversarial, original, epsilon)
        adversarial = clamp_normalized(adversarial)

    return adversarial.detach()
