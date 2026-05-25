from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from .swin_unet import SwinUNet
from .unet_resnet50 import UNetResNet50


_Builder = Callable[..., nn.Module]


MODEL_REGISTRY: dict[str, _Builder] = {
	"unet_resnet50": UNetResNet50,
	"swin_unet": SwinUNet,
}

# Default pretrained kwargs per model
_PRETRAINED_DEFAULTS: dict[str, dict[str, Any]] = {
	"unet_resnet50": {"encoder_weights": "imagenet"},
	"swin_unet": {"pretrained": True},
}


def list_models() -> list[str]:
	"""Return sorted list of available model names."""
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


def build_model_pretrained(name: str, **kwargs: Any) -> nn.Module:
	"""Build a model with ImageNet-pretrained encoder weights.

	Convenience wrapper that merges default pretrained kwargs
	with any user overrides.
	"""
	key = name.strip().lower()
	merged = {**_PRETRAINED_DEFAULTS.get(key, {}), **kwargs}
	return build_model(key, **merged)


def get_device() -> torch.device:
	"""Detect the best available device.

	Priority: CUDA > DirectML > CPU.
	"""
	if torch.cuda.is_available():
		return torch.device("cuda")
	try:
		import torch_directml  # type: ignore
		return torch_directml.device()
	except Exception:
		pass
	return torch.device("cpu")
