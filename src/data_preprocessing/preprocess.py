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
    return sorted(images_dir.glob("*.jpg"))


def _mask_id(mask_path: Path) -> str:
    stem = mask_path.stem
    suffix = "_segmentation"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _build_mask_index(masks_dir: Path) -> dict[str, Path]:
    masks = sorted(masks_dir.glob("*.png"))
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
        default=Path("data") / "ISIC2018",
        help="Dataset root (default: data/ISIC2018)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("processed"),
        help="Output root (default: processed/)",
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
        "--dry-run",
        action="store_true",
        help="Only validate input folders and print counts; do not write output.",
    )
    args = parser.parse_args()

    dataset_dir = args.data_dir
    if not dataset_dir.exists():
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
    out_root = args.out_dir / run_tag
    images_out_root = out_root / "images"
    masks_out_root = out_root / "masks"
    out_root.mkdir(parents=True, exist_ok=True)

    print("[preprocess] input :", dataset_dir.resolve())
    print("[preprocess] output:", out_root.resolve())
    print("[preprocess] size  :", args.size)
    print("[preprocess] norm  :", "ImageNet" if args.normalize_imagenet else "disabled")
    print("[preprocess] layout:", "CHW" if args.channels_first else "HWC")
    if args.limit_per_split and args.limit_per_split > 0:
        print("[preprocess] limit :", args.limit_per_split, "(seed=", args.seed, ")")

    summary: dict[str, dict[str, int]] = {}

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

            x = _read_and_preprocess_image(
                img_path,
                size=args.size,
                normalize_imagenet=args.normalize_imagenet,
                channels_first=args.channels_first,
            )
            m = _read_and_preprocess_mask(mask_path, size=args.size)

            np.save(str(out_img), x)
            cv2.imwrite(str(out_mask), m)

            manifest_rows.append(
                {
                    "id": img_id,
                    "image": str(out_img.as_posix()),
                    "mask": str(out_mask.as_posix()),
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
