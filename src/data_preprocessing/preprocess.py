import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    images_dir: Path
    masks_dir: Path


def _iter_images(images_dir: Path) -> list[Path]:
    patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(images_dir.glob(pattern))
    return sorted(set(image_paths))


def _mask_id(mask_path: Path) -> str:
    stem = mask_path.stem
    suffix = "_segmentation"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _build_mask_index(masks_dir: Path) -> dict[str, Path]:
    patterns = ("*.png", "*.PNG", "*.jpg", "*.JPG")
    masks: list[Path] = []
    for pattern in patterns:
        masks.extend(masks_dir.glob(pattern))
    masks = sorted(set(masks))
    index: dict[str, Path] = {}
    for mask_path in masks:
        index[_mask_id(mask_path)] = mask_path
    return index


def _read_and_preprocess_image(
    image_path: Path,
    size: int,
    normalize_imagenet: bool,
    channels_first: bool,
) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)

    x = rgb.astype(np.float32) / 255.0
    if normalize_imagenet:
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

    if channels_first:
        x = np.transpose(x, (2, 0, 1))
    return x


def _read_and_preprocess_mask(mask_path: Path, size: int) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    # Ensure binary mask.
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


def _read_raw_image(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_raw_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    return mask


def _prepare_image(
    rgb: np.ndarray,
    size: int,
    normalize_imagenet: bool,
    channels_first: bool,
) -> np.ndarray:
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    if normalize_imagenet:
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    if channels_first:
        x = np.transpose(x, (2, 0, 1))
    return x


def _prepare_mask(mask: np.ndarray, size: int) -> np.ndarray:
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def _build_augmentations(size: int):
    try:
        import albumentations as A
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Albumentations is required for --augment. Install via pip install albumentations"
        ) from exc

    return A.Compose(
        [
            A.Resize(height=size, width=size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.25,
                hue=0.1,
                p=0.7,
            ),
            A.ElasticTransform(
                alpha=40.0,
                sigma=6.0,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
        ]
    )


def _maybe_tqdm(items: Iterable[Path], desc: str):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(list(items), desc=desc)
    except Exception:
        return list(items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess ISIC 2018 Task 1 dataset: resize + optional ImageNet normalization; "
            "save processed files for reuse."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "raw",
        help="Dataset root (default: data/raw)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "processed",
        help="Output root (default: data/processed)",
    )
    parser.add_argument(
        "--augment-out-dir",
        type=Path,
        default=Path("data") / "augmented",
        help="Augmented output root (default: data/augmented)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Target square size (default: 256)",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help=(
            "If > 0, only process up to N (image,mask) pairs per split. "
            "Useful to create a small sharable processed subset."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --limit-per-split is set (default: 42)",
    )
    parser.add_argument(
        "--normalize-imagenet",
        action="store_true",
        default=True,
        help="Normalize RGB by ImageNet mean/std (default: enabled)",
    )
    parser.add_argument(
        "--no-normalize-imagenet",
        action="store_false",
        dest="normalize_imagenet",
        help="Disable ImageNet normalization.",
    )
    parser.add_argument(
        "--channels-first",
        action="store_true",
        help="Save images as CHW instead of HWC (useful for PyTorch).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed files.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Enable data augmentation (flip, rotation, color jitter, elastic transform). "
            "Augmented samples are saved alongside originals."
        ),
    )
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=2,
        help="Number of augmented samples to generate per image (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate input folders and print counts; do not write output.",
    )
    args = parser.parse_args()

    dataset_dir = args.data_dir
    if not dataset_dir.exists():
        fallback_dir = Path("data") / "raw"
        if args.data_dir == Path("data") / "ISIC2018" and fallback_dir.exists():
            dataset_dir = fallback_dir
            print("[preprocess] data-dir not found, using:", dataset_dir)
        else:
            raise SystemExit(f"data-dir does not exist: {dataset_dir}")

    train = SplitSpec(
        name="train",
        images_dir=dataset_dir / "ISIC2018_Task1-2_Training_Input",
        masks_dir=dataset_dir / "ISIC2018_Task1_Training_GroundTruth",
    )
    val = SplitSpec(
        name="val",
        images_dir=dataset_dir / "ISIC2018_Task1-2_Validation_Input",
        masks_dir=dataset_dir / "ISIC2018_Task1_Validation_GroundTruth",
    )
    test = SplitSpec(
        name="test",
        images_dir=dataset_dir / "ISIC2018_Task1-2_Test_Input",
        masks_dir=dataset_dir / "ISIC2018_Task1_Test_GroundTruth",
    )
    splits = [train, val, test]

    for sp in splits:
        if not sp.images_dir.exists():
            raise SystemExit(f"Missing images dir: {sp.images_dir}")
        if not sp.masks_dir.exists():
            raise SystemExit(f"Missing masks dir: {sp.masks_dir}")

    norm_tag = "imagenet" if args.normalize_imagenet else "nonorm"
    layout_tag = "chw" if args.channels_first else "hwc"
    run_tag = f"ISIC2018_{args.size}_{norm_tag}_{layout_tag}"
    if args.limit_per_split and args.limit_per_split > 0:
        run_tag += f"_limit{args.limit_per_split}"
    if args.augment:
        run_tag += f"_aug{args.augment_copies}"
    out_root = args.out_dir / run_tag
    images_out_root = out_root / "images"
    masks_out_root = out_root / "masks"
    aug_out_root = args.augment_out_dir / run_tag
    aug_images_out_root = aug_out_root / "images"
    aug_masks_out_root = aug_out_root / "masks"
    out_root.mkdir(parents=True, exist_ok=True)

    print("[preprocess] input :", dataset_dir.resolve())
    print("[preprocess] output:", out_root.resolve())
    print("[preprocess] size  :", args.size)
    print("[preprocess] norm  :", "ImageNet" if args.normalize_imagenet else "disabled")
    print("[preprocess] layout:", "CHW" if args.channels_first else "HWC")
    if args.limit_per_split and args.limit_per_split > 0:
        print("[preprocess] limit :", args.limit_per_split, "(seed=", args.seed, ")")
    if args.augment:
        print("[preprocess] augment:", args.augment_copies, "copies per image")
        print("[preprocess] augment_out:", aug_out_root.resolve())

    summary: dict[str, dict[str, int]] = {}
    augmenter = _build_augmentations(args.size) if args.augment else None

    for sp in splits:
        image_paths = _iter_images(sp.images_dir)
        mask_index = _build_mask_index(sp.masks_dir)

        matched: list[tuple[Path, Path]] = []
        for img_path in image_paths:
            img_id = img_path.stem
            mask_path = mask_index.get(img_id)
            if mask_path is None:
                continue
            matched.append((img_path, mask_path))

        summary[sp.name] = {
            "images_found": len(image_paths),
            "masks_found": len(mask_index),
            "pairs_matched": len(matched),
        }

        if args.limit_per_split and args.limit_per_split > 0:
            random.Random(args.seed).shuffle(matched)
            matched = matched[: min(args.limit_per_split, len(matched))]

        print(
            f"[{sp.name}] images={len(image_paths)} masks={len(mask_index)} matched={len(matched)}"
        )

        if args.dry_run:
            continue

        images_out_dir = images_out_root / sp.name
        masks_out_dir = masks_out_root / sp.name
        images_out_dir.mkdir(parents=True, exist_ok=True)
        masks_out_dir.mkdir(parents=True, exist_ok=True)

        if args.augment:
            aug_images_out_dir = aug_images_out_root / sp.name
            aug_masks_out_dir = aug_masks_out_root / sp.name
            aug_images_out_dir.mkdir(parents=True, exist_ok=True)
            aug_masks_out_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows: list[dict[str, str]] = []

        for img_path, mask_path in _maybe_tqdm(matched, desc=f"Processing {sp.name}"):
            img_id = img_path.stem
            out_img = images_out_dir / f"{sp.name}_{img_id}.npy"
            out_mask = masks_out_dir / f"{sp.name}_{img_id}.png"

            if not args.overwrite and out_img.exists() and out_mask.exists():
                manifest_rows.append(
                    {
                        "id": img_id,
                        "image": str(out_img.as_posix()),
                        "mask": str(out_mask.as_posix()),
                    }
                )
                continue

            rgb = _read_raw_image(img_path)
            mask_raw = _read_raw_mask(mask_path)
            x = _prepare_image(
                rgb,
                size=args.size,
                normalize_imagenet=args.normalize_imagenet,
                channels_first=args.channels_first,
            )
            m = _prepare_mask(mask_raw, size=args.size)

            np.save(str(out_img), x)
            cv2.imwrite(str(out_mask), m)

            manifest_rows.append(
                {
                    "id": img_id,
                    "image": str(out_img.as_posix()),
                    "mask": str(out_mask.as_posix()),
                    "augmented": False,
                }
            )

            if args.augment and augmenter is not None and args.augment_copies > 0:
                for idx in range(args.augment_copies):
                    aug = augmenter(image=rgb, mask=mask_raw)
                    aug_img = _prepare_image(
                        aug["image"],
                        size=args.size,
                        normalize_imagenet=args.normalize_imagenet,
                        channels_first=args.channels_first,
                    )
                    aug_mask = _prepare_mask(aug["mask"], size=args.size)

                    aug_id = f"{img_id}_aug{idx + 1}"
                    aug_img_path = aug_images_out_dir / f"{sp.name}_{aug_id}.npy"
                    aug_mask_path = aug_masks_out_dir / f"{sp.name}_{aug_id}.png"

                    if not args.overwrite and aug_img_path.exists() and aug_mask_path.exists():
                        manifest_rows.append(
                            {
                                "id": aug_id,
                                "image": str(aug_img_path.as_posix()),
                                "mask": str(aug_mask_path.as_posix()),
                                "augmented": True,
                            }
                        )
                        continue

                    np.save(str(aug_img_path), aug_img)
                    cv2.imwrite(str(aug_mask_path), aug_mask)

                    manifest_rows.append(
                        {
                            "id": aug_id,
                            "image": str(aug_img_path.as_posix()),
                            "mask": str(aug_mask_path.as_posix()),
                            "augmented": True,
                        }
                    )

        (out_root / "manifests").mkdir(parents=True, exist_ok=True)
        manifest_path = out_root / "manifests" / f"{sp.name}.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": sp.name,
                    "size": args.size,
                    "normalize_imagenet": args.normalize_imagenet,
                    "channels_first": args.channels_first,
                    "items": manifest_rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    meta_path = out_root / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
                {
                    "input_data_dir": str(dataset_dir.as_posix()),
                    "size": args.size,
                    "normalize_imagenet": args.normalize_imagenet,
                    "imagenet_mean": IMAGENET_MEAN.tolist(),
                    "imagenet_std": IMAGENET_STD.tolist(),
                    "channels_first": args.channels_first,
                    "augmentation": {
                        "enabled": args.augment,
                        "copies_per_image": args.augment_copies if args.augment else 0,
                        "output_dir": str(aug_out_root.as_posix()) if args.augment else "",
                        "transforms": [
                            "horizontal_flip",
                            "vertical_flip",
                            "rotate",
                            "color_jitter",
                            "elastic_transform",
                        ]
                        if args.augment
                        else [],
                    },
                    "splits": summary,
                },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.dry_run:
        print("[preprocess] Dry-run complete (no files written).")
    else:
        print("[preprocess] Done. Processed data saved to:", out_root.resolve())


if __name__ == "__main__":
    main()
