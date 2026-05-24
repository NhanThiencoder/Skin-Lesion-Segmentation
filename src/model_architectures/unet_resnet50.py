from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _require_smp() -> Any:
    try:
        import segmentation_models_pytorch as smp  # type: ignore

        return smp
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency 'segmentation-models-pytorch'. "
            "Install with: pip install segmentation-models-pytorch timm"
        ) from exc


class UNetResNet50(nn.Module):
    """U-Net decoder with ResNet50 encoder (ImageNet-pretrained optional).

    Notes:
    - Outputs raw logits by default (no sigmoid). Apply sigmoid in loss/metrics.
    - Uses `segmentation-models-pytorch` implementation for stability.
    """

    def __init__(
        self,
        in_channels: int = 3,
        classes: int = 1,
        encoder_weights: str | None = "imagenet",
        decoder_channels: tuple[int, int, int, int, int] = (256, 128, 64, 32, 16),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        smp = _require_smp()
        self.net = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            decoder_channels=decoder_channels,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
