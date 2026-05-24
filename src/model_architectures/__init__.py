from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from .swin_unet import SwinUNet
from .unet_resnet50 import UNetResNet50


_Builder = Callable[..., nn.Module]


MODEL_REGISTRY: dict[str, _Builder] = {
	"unet_resnet50": UNetResNet50,
	"swin_unet": SwinUNet,
}


def list_models() -> list[str]:
	return sorted(MODEL_REGISTRY.keys())


def build_model(name: str, **kwargs: Any) -> nn.Module:
	"""Build a segmentation model by name.

	Supported names: see `list_models()`.
	"""

	key = name.strip().lower()
	if key not in MODEL_REGISTRY:
		raise ValueError(
			f"Unknown model '{name}'. Available: {', '.join(list_models())}"
		)
	return MODEL_REGISTRY[key](**kwargs)
