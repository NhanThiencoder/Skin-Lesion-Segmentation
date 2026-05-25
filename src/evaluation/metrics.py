"""Segmentation quality metrics for skin-lesion binary segmentation.

All standalone metric functions follow the same contract:
- ``pred`` contains **raw logits** (pre-sigmoid), shape ``(B, 1, H, W)``.
- ``target`` is a binary mask, shape ``(B, 1, H, W)`` with values in {0, 1}.
- A sigmoid is applied internally to ``pred`` and then binarised at
  ``threshold``.
- The returned value is a **scalar** tensor (mean over the batch dimension).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binarize(pred: Tensor, threshold: float = 0.5) -> Tensor:
    """Apply sigmoid and threshold to raw logits.

    Parameters
    ----------
    pred : Tensor
        Raw logits of shape ``(B, 1, H, W)``.
    threshold : float
        Decision boundary applied after sigmoid.

    Returns
    -------
    Tensor
        Binary predictions of same shape, dtype ``float32``.
    """
    return (torch.sigmoid(pred) >= threshold).float()


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def dice_coefficient(
    pred: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> Tensor:
    """Compute the Sørensen–Dice coefficient.

    .. math::
        \\text{Dice} = \\frac{2 |A \\cap B|}{|A| + |B|}

    Parameters
    ----------
    pred : Tensor
        Raw logits, shape ``(B, 1, H, W)``.
    target : Tensor
        Binary ground truth, shape ``(B, 1, H, W)``.
    threshold : float
        Binarisation threshold applied after sigmoid.
    smooth : float
        Laplace smoothing term to avoid division by zero.

    Returns
    -------
    Tensor
        Scalar mean Dice over the batch.
    """
    pred_bin = _binarize(pred, threshold)
    # Flatten spatial dims per sample: (B, N)
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice.mean()


def iou_score(
    pred: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> Tensor:
    """Compute the Intersection-over-Union (Jaccard Index).

    .. math::
        \\text{IoU} = \\frac{|A \\cap B|}{|A \\cup B|}

    Parameters
    ----------
    pred : Tensor
        Raw logits, shape ``(B, 1, H, W)``.
    target : Tensor
        Binary ground truth, shape ``(B, 1, H, W)``.
    threshold : float
        Binarisation threshold applied after sigmoid.
    smooth : float
        Laplace smoothing term to avoid division by zero.

    Returns
    -------
    Tensor
        Scalar mean IoU over the batch.
    """
    pred_bin = _binarize(pred, threshold)
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def pixel_accuracy(
    pred: Tensor,
    target: Tensor,
    threshold: float = 0.5,
) -> Tensor:
    """Compute pixel-wise accuracy.

    .. math::
        \\text{Accuracy} = \\frac{\\text{correct pixels}}{\\text{total pixels}}

    Parameters
    ----------
    pred : Tensor
        Raw logits, shape ``(B, 1, H, W)``.
    target : Tensor
        Binary ground truth, shape ``(B, 1, H, W)``.
    threshold : float
        Binarisation threshold applied after sigmoid.

    Returns
    -------
    Tensor
        Scalar mean pixel accuracy over the batch.
    """
    pred_bin = _binarize(pred, threshold)
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    correct = (pred_flat == target_flat).float().sum(dim=1)
    total = target_flat.size(1)

    accuracy = correct / total
    return accuracy.mean()


def sensitivity(
    pred: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-7,
) -> Tensor:
    """Compute sensitivity (True Positive Rate / Recall).

    .. math::
        \\text{Sensitivity} = \\frac{TP}{TP + FN}

    Parameters
    ----------
    pred : Tensor
        Raw logits, shape ``(B, 1, H, W)``.
    target : Tensor
        Binary ground truth, shape ``(B, 1, H, W)``.
    threshold : float
        Binarisation threshold applied after sigmoid.
    smooth : float
        Small constant to avoid division by zero.

    Returns
    -------
    Tensor
        Scalar mean sensitivity over the batch.
    """
    pred_bin = _binarize(pred, threshold)
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fn = (target_flat * (1.0 - pred_flat)).sum(dim=1)

    sens = (tp + smooth) / (tp + fn + smooth)
    return sens.mean()


def specificity(
    pred: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-7,
) -> Tensor:
    """Compute specificity (True Negative Rate).

    .. math::
        \\text{Specificity} = \\frac{TN}{TN + FP}

    Parameters
    ----------
    pred : Tensor
        Raw logits, shape ``(B, 1, H, W)``.
    target : Tensor
        Binary ground truth, shape ``(B, 1, H, W)``.
    threshold : float
        Binarisation threshold applied after sigmoid.
    smooth : float
        Small constant to avoid division by zero.

    Returns
    -------
    Tensor
        Scalar mean specificity over the batch.
    """
    pred_bin = _binarize(pred, threshold)
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    neg_pred = 1.0 - pred_flat
    neg_target = 1.0 - target_flat

    tn = (neg_pred * neg_target).sum(dim=1)
    fp = (pred_flat * neg_target).sum(dim=1)

    spec = (tn + smooth) / (tn + fp + smooth)
    return spec.mean()


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------

# Map from human-readable name → callable
_METRIC_FN_REGISTRY: dict[str, type] = {
    "dice": dice_coefficient,
    "iou": iou_score,
    "pixel_accuracy": pixel_accuracy,
    "accuracy": pixel_accuracy,  # alias
    "sensitivity": sensitivity,
    "specificity": specificity,
}


class MetricsTracker:
    """Accumulate metrics over batches and compute epoch averages.

    Parameters
    ----------
    metric_names : list[str]
        Names of metrics to track. Supported:
        ``"dice"``, ``"iou"``, ``"pixel_accuracy"``,
        ``"sensitivity"``, ``"specificity"``.

    Example
    -------
    >>> tracker = MetricsTracker(["dice", "iou"])
    >>> for images, masks in dataloader:
    ...     logits = model(images)
    ...     tracker.update(logits, masks, batch_size=images.size(0))
    >>> epoch_metrics = tracker.compute()
    >>> print(epoch_metrics)
    {'dice': 0.87, 'iou': 0.78}
    """

    def __init__(self, metric_names: list[str]) -> None:
        unknown = set(metric_names) - set(_METRIC_FN_REGISTRY)
        if unknown:
            raise ValueError(
                f"Unknown metric(s): {unknown}. "
                f"Available: {sorted(_METRIC_FN_REGISTRY)}"
            )
        self._names: list[str] = list(metric_names)
        self._running_sum: dict[str, float] = {}
        self._total_samples: int = 0
        self.reset()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def update(self, pred: Tensor, target: Tensor, batch_size: int) -> None:
        """Compute all tracked metrics for one batch and accumulate.

        Parameters
        ----------
        pred : Tensor
            Raw logits, shape ``(B, 1, H, W)``.
        target : Tensor
            Binary ground truth, shape ``(B, 1, H, W)``.
        batch_size : int
            Number of samples in this batch (used for weighted averaging).
        """
        for name in self._names:
            value = _METRIC_FN_REGISTRY[name](pred, target)
            self._running_sum[name] += value.item() * batch_size
        self._total_samples += batch_size

    def compute(self) -> dict[str, float]:
        """Return epoch-averaged metrics.

        Returns
        -------
        dict[str, float]
            Mapping from metric name to its epoch average.

        Raises
        ------
        RuntimeError
            If called before any ``update()`` call.
        """
        if self._total_samples == 0:
            raise RuntimeError(
                "MetricsTracker.compute() called before any update(). "
                "Call update() at least once first."
            )
        return {
            name: self._running_sum[name] / self._total_samples
            for name in self._names
        }

    def reset(self) -> None:
        """Reset all accumulators for a new epoch."""
        self._running_sum = {name: 0.0 for name in self._names}
        self._total_samples = 0

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"MetricsTracker(metrics={self._names}, "
            f"samples={self._total_samples})"
        )
