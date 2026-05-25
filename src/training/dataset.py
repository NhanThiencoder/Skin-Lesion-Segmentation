"""PyTorch Dataset for preprocessed ISIC 2018 skin-lesion segmentation data.

Loads image/mask pairs described by manifest JSON files produced by the
preprocessing pipeline.  Supports merging multiple manifests (e.g. original
processed data + offline-augmented data) and optional online spatial
augmentation via *albumentations*.

Typical usage
-------------
>>> from src.training.dataset import create_dataloaders
>>> loaders = create_dataloaders(
...     data_dir=Path("data/processed/ISIC2018_256_imagenet"),
...     project_root=Path("."),
...     batch_size=16,
... )
>>> train_loader = loaders["train"]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

ManifestItem = dict[str, str]  # {"id": ..., "image": ..., "mask": ...}


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read and return a manifest JSON file.

    Parameters
    ----------
    manifest_path:
        Absolute or resolvable path to a ``*.json`` manifest.

    Returns
    -------
    dict
        Parsed manifest with keys ``split``, ``size``, ``normalize_imagenet``,
        ``channels_first``, and ``items``.

    Raises
    ------
    FileNotFoundError
        If *manifest_path* does not exist.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    logger.info(
        "Loaded manifest %s  (split=%s, items=%d)",
        manifest_path.name,
        data.get("split", "?"),
        len(data.get("items", [])),
    )
    return data


def _merge_items(*manifests: dict[str, Any]) -> list[ManifestItem]:
    """Merge ``items`` lists from one or more manifests.

    Duplicate sample IDs are kept (the caller may intentionally duplicate
    samples via augmentation).
    """
    merged: list[ManifestItem] = []
    for m in manifests:
        merged.extend(m.get("items", []))
    return merged


# ---------------------------------------------------------------------------
# Default augmentation
# ---------------------------------------------------------------------------


def get_default_train_augmentation(size: int = 256) -> A.Compose:
    """Return a safe *spatial-only* augmentation pipeline.

    Because the images are **already ImageNet-normalised**, colour-space
    transforms (brightness, contrast, hue/saturation jitter, …) would
    corrupt the statistics.  We therefore restrict ourselves to spatial
    transforms that can be applied identically to both image and mask.

    Parameters
    ----------
    size:
        Expected spatial extent (H = W).  Not used for resizing (data is
        already at the correct resolution) but reserved for future
        ``RandomCrop`` / ``PadIfNeeded`` additions.

    Returns
    -------
    A.Compose
        Albumentations pipeline expecting ``image`` (H, W, 3) and
        ``mask`` (H, W) keyword arguments.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                limit=30,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.95, 1.05),
                rotate=0,  # rotation already handled above
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3,
            ),
            A.ElasticTransform(
                alpha=30,
                sigma=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.2,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ISICSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch :class:`~torch.utils.data.Dataset` for ISIC 2018 skin lesion
    segmentation.

    Each sample is a ``(image, mask)`` pair where:

    * **image** – ``FloatTensor`` of shape ``(3, H, W)``, already
      ImageNet-normalised.
    * **mask**  – ``FloatTensor`` of shape ``(1, H, W)`` with values in
      ``{0.0, 1.0}``.

    Parameters
    ----------
    manifest_paths:
        One or more paths to manifest JSON files.  Items from all manifests
        are concatenated to form the full sample list.
    project_root:
        Project root directory.  Relative paths stored in the manifest are
        resolved against this directory.
    transform:
        Optional *albumentations* ``Compose`` pipeline.  Must accept
        ``image`` (H, W, 3 float32) and ``mask`` (H, W uint8) keyword
        arguments.
    """

    def __init__(
        self,
        manifest_paths: Path | str | Sequence[Path | str],
        project_root: Path | str = Path("."),
        transform: A.Compose | None = None,
    ) -> None:
        super().__init__()

        # Normalise to a list of Paths.
        if isinstance(manifest_paths, (str, Path)):
            manifest_paths = [manifest_paths]
        manifest_paths = [Path(p) for p in manifest_paths]

        self.project_root = Path(project_root).resolve()
        self.transform = transform

        # Load & merge manifests.
        manifests = [_load_manifest(p) for p in manifest_paths]
        self.items: list[ManifestItem] = _merge_items(*manifests)

        if len(self.items) == 0:
            logger.warning("Dataset is empty – no items found in manifests.")

        # Store metadata from the *first* manifest for reference.
        first = manifests[0] if manifests else {}
        self.split: str = first.get("split", "unknown")
        self.image_size: int = first.get("size", 256)
        self.channels_first: bool = first.get("channels_first", False)

        logger.info(
            "ISICSegmentationDataset created: split=%s, samples=%d, "
            "augmentation=%s",
            self.split,
            len(self.items),
            "yes" if self.transform else "no",
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(image, mask)`` tensors for sample at *index*.

        Returns
        -------
        image:
            ``FloatTensor`` of shape ``(3, H, W)``.
        mask:
            ``FloatTensor`` of shape ``(1, H, W)`` with binary values.
        """
        item = self.items[index]

        # --- Load image (.npy, float32, HWC or CHW) ---
        image_path = self.project_root / item["image"]
        image: np.ndarray = np.load(str(image_path))  # (H, W, 3) float32
        if self.channels_first:
            # Convert CHW → HWC so albumentations can work on it.
            image = np.transpose(image, (1, 2, 0))
        # Ensure float32 (should already be, but be safe).
        image = image.astype(np.float32)

        # --- Load mask (.png, grayscale, 0/255) ---
        mask_path = self.project_root / item["mask"]
        mask: np.ndarray = cv2.imread(
            str(mask_path), cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            raise FileNotFoundError(
                f"Could not read mask at {mask_path} "
                f"(sample id={item.get('id', '?')})"
            )
        # Binarise: 0 → 0, 255 → 1.  Anything in between is thresholded.
        mask = (mask >= 128).astype(np.uint8)  # (H, W) uint8 {0, 1}

        # --- Optional augmentation ---
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # --- Convert to tensors ---
        # Image: HWC → CHW
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )  # (3, H, W) float32

        # Mask: (H, W) uint8 → (1, H, W) float32
        mask_tensor = (
            torch.from_numpy(np.ascontiguousarray(mask))
            .unsqueeze(0)
            .float()
        )  # (1, H, W)

        return image_tensor, mask_tensor

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self.split!r}, "
            f"samples={len(self)}, "
            f"size={self.image_size}, "
            f"augmentation={'yes' if self.transform else 'no'})"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def create_dataloaders(
    data_dir: Path | str,
    project_root: Path | str = Path("."),
    batch_size: int = 16,
    num_workers: int = 4,
    augmented_dir: Path | str | None = None,
    train_transform: A.Compose | None = None,
    pin_memory: bool | None = None,
) -> dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """Build train / val / test :class:`~torch.utils.data.DataLoader` objects.

    Parameters
    ----------
    data_dir:
        Directory containing a ``manifests/`` subfolder with
        ``train.json``, ``val.json``, and ``test.json``.
    project_root:
        Root directory of the project, used to resolve the relative paths
        stored inside manifest files.
    batch_size:
        Mini-batch size for all splits.
    num_workers:
        Number of background data-loading workers.
    augmented_dir:
        Optional directory with an augmented manifest at
        ``<augmented_dir>/manifests/train.json``.  If provided, those
        samples are **merged** with the primary training set.
    train_transform:
        Optional custom augmentation pipeline for the training split.
        If ``None`` the default from :func:`get_default_train_augmentation`
        is used.
    pin_memory:
        Whether to use pinned (page-locked) memory for faster host→device
        transfers.

    Returns
    -------
    dict[str, DataLoader]
        Mapping ``{"train": …, "val": …, "test": …}``.
    """
    data_dir = Path(data_dir)
    project_root = Path(project_root)
    manifest_dir = data_dir / "manifests"

    # Locate manifests.
    train_manifest = manifest_dir / "train.json"
    val_manifest = manifest_dir / "val.json"
    test_manifest = manifest_dir / "test.json"

    for mf in (train_manifest, val_manifest, test_manifest):
        if not mf.exists():
            raise FileNotFoundError(
                f"Expected manifest not found: {mf}. "
                "Run the preprocessing pipeline first."
            )

    # --- Training set (possibly merged with augmented data) ---
    train_manifests: list[Path] = [train_manifest]
    if augmented_dir is not None:
        aug_manifest = Path(augmented_dir) / "manifests" / "train.json"
        if aug_manifest.exists():
            train_manifests.append(aug_manifest)
            logger.info("Merging augmented manifest: %s", aug_manifest)
        else:
            logger.warning(
                "augmented_dir provided but manifest not found: %s",
                aug_manifest,
            )

    if train_transform is None:
        # Read image size from the first manifest for the augmentation.
        first_manifest = _load_manifest(train_manifest)
        img_size: int = first_manifest.get("size", 256)
        train_transform = get_default_train_augmentation(size=img_size)

    train_ds = ISICSegmentationDataset(
        manifest_paths=train_manifests,
        project_root=project_root,
        transform=train_transform,
    )

    # --- Validation & test sets (no augmentation) ---
    val_ds = ISICSegmentationDataset(
        manifest_paths=val_manifest,
        project_root=project_root,
        transform=None,
    )
    test_ds = ISICSegmentationDataset(
        manifest_paths=test_manifest,
        project_root=project_root,
        transform=None,
    )

    # --- DataLoaders ---
    # Auto-detect pin_memory: only useful with GPU accelerator
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    common_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    loaders: dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]] = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **common_kwargs,
        ),
        "val": DataLoader(
            train_ds if len(val_ds) == 0 else val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        ),
        "test": DataLoader(
            train_ds if len(test_ds) == 0 else test_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        ),
    }

    for name, loader in loaders.items():
        logger.info(
            "DataLoader[%s]: %d samples, %d batches",
            name,
            len(loader.dataset),  # type: ignore[arg-type]
            len(loader),
        )

    return loaders
