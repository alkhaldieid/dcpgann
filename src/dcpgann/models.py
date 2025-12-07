"""Model definitions for IDC classification."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Lightweight CNN suitable for small IDC tiles."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_resnet50_end_to_end(num_classes: int = 2) -> nn.Module:
    """ResNet50 trained end-to-end (no pretrained weights)."""

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet50_partial(num_classes: int = 2, trainable_layers: int = 69) -> nn.Module:
    """Imagenet-pretrained ResNet50 with only the last layers unfrozen."""

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    params = list(model.parameters())
    for param in params:
        param.requires_grad = False
    for param in params[-trainable_layers:]:
        param.requires_grad = True
    return model


def build_vgg19_finetune(num_classes: int = 2) -> nn.Module:
    """Imagenet-pretrained VGG19 fully finetuned for IDC."""

    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    for param in model.parameters():
        param.requires_grad = True
    return model


def build_densenet121_partial(num_classes: int = 2, trainable_layers: int = 429) -> nn.Module:
    """Imagenet-pretrained DenseNet121 with partial finetuning."""

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    params = list(model.parameters())
    for param in params:
        param.requires_grad = False
    for param in params[-trainable_layers:]:
        param.requires_grad = True
    return model


def list_backbones() -> Dict[str, nn.Module]:
    return {
        "resnet50_end_to_end": build_resnet50_end_to_end,
        "resnet50_partial": build_resnet50_partial,
        "vgg19_finetune": build_vgg19_finetune,
        "densenet121_partial": build_densenet121_partial,
    }
