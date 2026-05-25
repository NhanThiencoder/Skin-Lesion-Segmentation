"""Loss functions for binary skin-lesion segmentation.

All losses accept **raw logits** (i.e. before sigmoid) and apply the
activation internally.  This is both numerically more stable and
consistent with PyTorch's ``BCEWithLogitsLoss``.

Typical usage
-------------
>>> from src.training.losses import get_loss_function
>>> criterion = get_loss_function("bce_dice")
>>> loss = criterion(logits, targets)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    .. math::
        \\text{Dice} = 1 - \\frac{2 |P \\cap G| + \\epsilon}
                                  {|P| + |G| + \\epsilon}

    where *P* = predicted probabilities and *G* = ground-truth labels.

    Parameters
    ----------
    smooth:
        Laplace / additive smoothing constant to avoid division by zero
        and to stabilise gradients when both prediction and target are
        empty.

    Notes
    -----
    * Input  ``logits``: ``(B, 1, H, W)`` – **raw logits** (before sigmoid).
    * Target ``targets``: ``(B, 1, H, W)`` – binary in ``{0, 1}``.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the soft Dice loss.

        Parameters
        ----------
        logits:
            Raw model output of shape ``(B, 1, H, W)``.
        targets:
            Binary ground-truth mask of shape ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Scalar loss value in ``[0, 1]``.
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dims → (B, -1) so the sum is per-sample.
        probs_flat = probs.reshape(probs.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )
        return 1.0 - dice_score.mean()


# ---------------------------------------------------------------------------
# BCE + Dice Loss
# ---------------------------------------------------------------------------


class BCEDiceLoss(nn.Module):
    """Weighted combination of Binary Cross-Entropy and Dice loss.

    This is the standard composite loss for medical image segmentation:
    BCE provides stable pixel-level gradients while Dice directly optimises
    the overlap metric.

    Parameters
    ----------
    bce_weight:
        Weighting factor for the BCE component.
    dice_weight:
        Weighting factor for the Dice component.
    smooth:
        Smoothing constant forwarded to :class:`DiceLoss`.
    pos_weight:
        Optional positive-class weight for ``BCEWithLogitsLoss``.  Useful
        when foreground (lesion) pixels are under-represented.

    Notes
    -----
    * Input  ``logits``: ``(B, 1, H, W)`` – **raw logits** (before sigmoid).
    * Target ``targets``: ``(B, 1, H, W)`` – binary in ``{0, 1}``.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined BCE + Dice loss.

        Parameters
        ----------
        logits:
            Raw model output of shape ``(B, 1, H, W)``.
        targets:
            Binary ground-truth mask of shape ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Scalar weighted sum ``bce_weight * BCE + dice_weight * Dice``.
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in binary segmentation.

    Focal loss down-weights the contribution of *easy* examples so that the
    model concentrates on *hard* (mis-classified) pixels – especially useful
    when the lesion occupies a small fraction of the image.

    .. math::
        \\text{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\log(p_t)

    Parameters
    ----------
    alpha:
        Balancing factor for the positive class.  ``alpha=0.25`` is the
        default from the original paper (Lin et al., 2017).
    gamma:
        Focusing parameter.  ``gamma=0`` recovers standard CE;
        ``gamma=2`` is the recommended default.

    Notes
    -----
    * Input  ``logits``: ``(B, 1, H, W)`` – **raw logits** (before sigmoid).
    * Target ``targets``: ``(B, 1, H, W)`` – binary in ``{0, 1}``.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the pixel-wise focal loss.

        Parameters
        ----------
        logits:
            Raw model output of shape ``(B, 1, H, W)``.
        targets:
            Binary ground-truth mask of shape ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Scalar mean focal loss.
        """
        # Numerically stable BCE per pixel (no reduction).
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        # p_t = probability of the *correct* class.
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        # Per-pixel alpha weighting.
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma

        return (focal_weight * bce_loss).mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "dice": DiceLoss,
    "bce_dice": BCEDiceLoss,
    "bce": nn.BCEWithLogitsLoss,
    "focal": FocalLoss,
}


def get_loss_function(name: str, **kwargs: object) -> nn.Module:
    """Factory that returns a loss module by name.

    Parameters
    ----------
    name:
        One of ``"dice"``, ``"bce_dice"``, ``"bce"``, ``"focal"``.
    **kwargs:
        Extra keyword arguments forwarded to the loss constructor
        (e.g. ``smooth=1.0`` for ``DiceLoss``).

    Returns
    -------
    nn.Module
        Instantiated loss function.

    Raises
    ------
    ValueError
        If *name* is not in the registry.

    Examples
    --------
    >>> criterion = get_loss_function("bce_dice", bce_weight=0.4, dice_weight=0.6)
    >>> loss = criterion(logits, targets)
    """
    name_lower = name.lower().strip()
    if name_lower not in _LOSS_REGISTRY:
        available = ", ".join(sorted(_LOSS_REGISTRY))
        raise ValueError(
            f"Unknown loss {name!r}. Available losses: {available}"
        )
    cls = _LOSS_REGISTRY[name_lower]
    return cls(**kwargs)  # type: ignore[arg-type]
