from __future__ import annotations

import torch.nn as nn
from torchvision import models as tv_models


def _cifar_resnet(name: str, num_classes: int) -> nn.Module:
    constructor = {
        "resnet18": tv_models.resnet18,
        "resnet34": tv_models.resnet34,
    }[name]
    model = constructor(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def _make_vgg_layers(cfg: list[int | str], batch_norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        out_channels = int(value)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    return nn.Sequential(*layers)


class CifarVGG16(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        cfg: list[int | str] = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ]
        self.features = _make_vgg_layers(cfg, batch_norm=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def _cifar_densenet121(num_classes: int) -> nn.Module:
    model = tv_models.densenet121(weights=None, num_classes=num_classes)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.features.pool0 = nn.Identity()
    return model


def available_models() -> tuple[str, ...]:
    return ("resnet18", "resnet34", "vgg16", "densenet121")


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    name = name.lower()
    if name in {"resnet18", "resnet34"}:
        return _cifar_resnet(name, num_classes)
    if name == "vgg16":
        return CifarVGG16(num_classes=num_classes)
    if name == "densenet121":
        return _cifar_densenet121(num_classes=num_classes)
    choices = ", ".join(available_models())
    raise ValueError(f"Unknown model {name!r}. Available: {choices}")
