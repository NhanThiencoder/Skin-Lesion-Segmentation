from _future_ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_timm():
    try:
        import timm  # type: ignore

        return timm
    except Exception as exc:  # pragma: no cover
        raise ImportError("Missing dependency 'timm'. Install with: pip install timm") from exc


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = _ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SwinUNet(nn.Module):
    """Swin-UNet style model: Swin Transformer encoder + U-Net decoder.

    Encoder is a hierarchical Swin backbone from timm in features_only mode.

    Notes:
    - Uses raw logits by default (no sigmoid).
    - For best results, use input size divisible by 32 (256x256 works well).
    """

    def __init__(
        self,
        in_channels: int = 3,
        classes: int = 1,
        backbone: str = "swin_base_patch4_window7_224",
        img_size: int = 256,
        pretrained: bool = False,
        decoder_channels: tuple[int, int, int, int] = (256, 128, 64, 32),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError(
                "Swin backbones in timm are typically pretrained for 3-channel RGB; "
                "set in_channels=3 and adapt inputs upstream."
            )

        timm = _require_timm()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,
            **kwargs,
        )
        enc_channels = list(self.encoder.feature_info.channels())
        if len(enc_channels) != 4:
            raise RuntimeError(
                f"Expected 4 feature levels from encoder, got {len(enc_channels)}: {enc_channels}"
            )

        c0, c1, c2, c3 = enc_channels

        d3, d2, d1, d0 = decoder_channels
        self.up3 = _UpBlock(in_channels=c3, skip_channels=c2, out_channels=d3)
        self.up2 = _UpBlock(in_channels=d3, skip_channels=c1, out_channels=d2)
        self.up1 = _UpBlock(in_channels=d2, skip_channels=c0, out_channels=d1)
        self.refine = _ConvBlock(in_channels=d1, out_channels=d0)

        self.head = nn.Conv2d(d0, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = self.encoder(x)
        # timm Swin feature extractor returns NHWC tensors by default.
        f0, f1, f2, f3 = [f.permute(0, 3, 1, 2).contiguous() for f in feats]

        y = self.up3(f3, f2)
        y = self.up2(y, f1)
        y = self.up1(y, f0)
        y = self.refine(y)
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(y)