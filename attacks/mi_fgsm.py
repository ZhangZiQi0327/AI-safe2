from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import clamp_normalized, input_diversity, project_linf, raw_eps_to_normalized, smooth_gradients


def mi_fgsm_attack(
    model: torch.nn.Module,
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
) -> torch.Tensor:
    """Momentum iterative FGSM on normalized tensors."""

    model.eval()
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
        logits = model(transformed)
        loss = F.cross_entropy(logits, labels)
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
