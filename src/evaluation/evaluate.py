from __future__ import annotations

"""Full evaluation pipeline for skin lesion segmentation models.

Features:
- Load trained checkpoint and evaluate on test set
- Compute comprehensive metrics: Dice, IoU, Accuracy, Sensitivity, Specificity
- Generate visual overlays: original image + mask + prediction
- Export results as JSON and CSV reports
- Support comparing multiple models

Usage:
    python src/evaluation/evaluate.py --model unet_resnet50
    python src/evaluation/evaluate.py --model swin_unet --visualize --num-vis 20
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model_architectures import build_model, get_device, list_models  # noqa: E402
from src.training.dataset import ISICSegmentationDataset, create_dataloaders  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    MetricsTracker,
    dice_coefficient,
    iou_score,
    pixel_accuracy,
    sensitivity,
    specificity,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════


def _denormalize_image(img: np.ndarray) -> np.ndarray:
    """Undo ImageNet normalization and convert to uint8 [0, 255].

    Args:
        img: (H, W, 3) float32 array, ImageNet-normalized.

    Returns:
        (H, W, 3) uint8 array.
    """
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _create_overlay(
    image_hwc: np.ndarray,
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Create a side-by-side overlay visualization.

    Layout: [Original] [GT overlay] [Pred overlay] [GT vs Pred]

    Args:
        image_hwc: (H, W, 3) uint8 RGB image.
        mask_gt: (H, W) binary mask (0 or 255).
        mask_pred: (H, W) binary mask (0 or 255).
        alpha: Overlay transparency.

    Returns:
        (H, W*4, 3) uint8 image.
    """
    h, w = image_hwc.shape[:2]

    # Panel 1: Original image
    panel1 = image_hwc.copy()

    # Panel 2: Ground truth overlay (green)
    panel2 = image_hwc.copy()
    gt_mask_bool = mask_gt > 127
    green_overlay = np.zeros_like(image_hwc)
    green_overlay[gt_mask_bool] = [0, 255, 0]
    panel2 = cv2.addWeighted(panel2, 1.0, green_overlay, alpha, 0)

    # Panel 3: Prediction overlay (blue)
    panel3 = image_hwc.copy()
    pred_mask_bool = mask_pred > 127
    blue_overlay = np.zeros_like(image_hwc)
    blue_overlay[pred_mask_bool] = [0, 100, 255]
    panel3 = cv2.addWeighted(panel3, 1.0, blue_overlay, alpha, 0)

    # Panel 4: Comparison — TP (green), FP (red), FN (yellow)
    panel4 = image_hwc.copy()
    tp = gt_mask_bool & pred_mask_bool
    fp = ~gt_mask_bool & pred_mask_bool
    fn = gt_mask_bool & ~pred_mask_bool
    comparison_overlay = np.zeros_like(image_hwc)
    comparison_overlay[tp] = [0, 255, 0]    # True Positive: green
    comparison_overlay[fp] = [255, 0, 0]    # False Positive: red
    comparison_overlay[fn] = [255, 255, 0]  # False Negative: yellow
    panel4 = cv2.addWeighted(panel4, 0.6, comparison_overlay, 0.6, 0)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_args = dict(fontFace=font, fontScale=0.5, thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(panel1, "Original", (5, 20), color=(255, 255, 255), **label_args)
    cv2.putText(panel2, "Ground Truth", (5, 20), color=(0, 255, 0), **label_args)
    cv2.putText(panel3, "Prediction", (5, 20), color=(0, 100, 255), **label_args)
    cv2.putText(panel4, "TP/FP/FN", (5, 20), color=(255, 255, 255), **label_args)

    return np.concatenate([panel1, panel2, panel3, panel4], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: int | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """Evaluate model on a dataset split.

    Returns:
        (aggregated_metrics, per_sample_results)
    """
    model.eval()
    metric_names = ["dice", "iou", "accuracy", "sensitivity", "specificity"]
    tracker = MetricsTracker(metric_names)

    per_sample_results: list[dict] = []
    n_batches = 0

    for images, masks in loader:
        if limit_batches is not None and n_batches >= limit_batches:
            break
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)  # raw logits

        # Batch-level accumulation
        tracker.update(outputs, masks, batch_size=images.size(0))

        # Per-sample metrics
        probs = torch.sigmoid(outputs)
        preds_bin = (probs > 0.5).float()
        batch_size = images.size(0)

        for i in range(batch_size):
            pred_i = outputs[i : i + 1]
            mask_i = masks[i : i + 1]
            per_sample_results.append({
                "dice": dice_coefficient(pred_i, mask_i).item(),
                "iou": iou_score(pred_i, mask_i).item(),
                "accuracy": pixel_accuracy(pred_i, mask_i).item(),
                "sensitivity": sensitivity(pred_i, mask_i).item(),
                "specificity": specificity(pred_i, mask_i).item(),
            })
        n_batches += 1

    aggregated = tracker.compute()
    return aggregated, per_sample_results


@torch.no_grad()
def generate_visualizations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 10,
    is_imagenet_normalized: bool = True,
) -> list[Path]:
    """Generate and save overlay visualizations."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    count = 0

    for images, masks in loader:
        if count >= num_samples:
            break

        images_dev = images.to(device)
        outputs = model(images_dev)
        probs = torch.sigmoid(outputs).cpu().numpy()

        for i in range(images.size(0)):
            if count >= num_samples:
                break

            # Get image in HWC format
            img_chw = images[i].numpy()  # (3, H, W)
            img_hwc = np.transpose(img_chw, (1, 2, 0))  # (H, W, 3)

            if is_imagenet_normalized:
                img_hwc = _denormalize_image(img_hwc)
            else:
                img_hwc = np.clip(img_hwc * 255, 0, 255).astype(np.uint8)

            # Masks
            mask_gt = masks[i, 0].numpy()  # (H, W) float [0, 1]
            mask_gt_uint8 = (mask_gt * 255).astype(np.uint8)

            mask_pred = probs[i, 0]  # (H, W) float [0, 1]
            mask_pred_uint8 = ((mask_pred > 0.5).astype(np.uint8)) * 255

            # Create overlay
            overlay = _create_overlay(img_hwc, mask_gt_uint8, mask_pred_uint8)

            # Save as BGR for OpenCV
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            save_path = output_dir / f"vis_{count:04d}.png"
            cv2.imwrite(str(save_path), overlay_bgr)
            saved_paths.append(save_path)
            count += 1

    return saved_paths


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="unet_resnet50",
        choices=list_models(),
        help="Model architecture to evaluate",
    )
    parser.add_argument("--img-size", type=int, default=256, help="Input image size")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint .pth file. If not given, auto-detects from models/ dir.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "processed",
        help="Root of processed data",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="ISIC2018_256_imagenet",
        help="Data subdirectory name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")

    # Visualization
    parser.add_argument("--visualize", action="store_true", help="Generate visual overlays")
    parser.add_argument("--num-vis", type=int, default=10, help="Number of visualization samples")

    # Output
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Number of batches to evaluate (for quick dry-runs)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained weights",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    device = get_device()
    print("=" * 70)
    print("  Skin Lesion Segmentation - Model Evaluation")
    print("=" * 70)
    print(f"  Model      : {args.model}")
    print(f"  Device     : {device}")
    print(f"  Split      : {args.split}")
    print(f"  Batch size : {args.batch_size}")

    # -----------------------------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------------------------
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = args.models_dir / f"best_{args.model}.pth"

    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train a model first with: python src/training/train.py --model {args.model}"
        )

    print(f"  Checkpoint : {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Build model (no pretrained needed, we load from checkpoint)
    model = build_model(args.model, img_size=args.img_size, pretrained=False)

    # Handle both old and new checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        ckpt_info = {
            "epoch": checkpoint.get("epoch", "?"),
            "val_dice": checkpoint.get("val_dice", "?"),
        }
        print(f"  Ckpt epoch : {ckpt_info['epoch']}")
        print(f"  Ckpt Dice  : {ckpt_info['val_dice']}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    data_root = args.data_dir / args.run_tag
    if not data_root.exists():
        raise SystemExit(f"Data directory does not exist: {data_root}")

    loaders = create_dataloaders(
        data_dir=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        project_root=_PROJECT_ROOT,
    )

    # Select the right loader
    eval_loader = loaders[args.split]

    if eval_loader is None or len(eval_loader.dataset) == 0:  # type: ignore[arg-type]
        raise SystemExit(f"No data found for split '{args.split}'")

    print(f"  Samples    : {len(eval_loader.dataset)}")  # type: ignore[arg-type]
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    print(f"\n[eval] Running evaluation on '{args.split}' set...")
    aggregated, per_sample = evaluate_model(model, eval_loader, device, limit_batches=args.limit_batches)

    print(f"\n{'-'*50}")
    print(f"  Results on {args.split} set ({len(per_sample)} samples)")
    print(f"{'-'*50}")
    for metric_name, value in aggregated.items():
        print(f"  {metric_name:<15s}: {value:.4f}")
    print(f"{'-'*50}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    results_json_path = args.outputs_dir / f"eval_{args.model}_{args.split}.json"
    results_csv_path = args.outputs_dir / f"eval_{args.model}_{args.split}_per_sample.csv"

    # JSON report
    results_data = {
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_samples": len(per_sample),
        "aggregated_metrics": aggregated,
        "per_sample_summary": {
            metric: {
                "mean": float(np.mean([s[metric] for s in per_sample])),
                "std": float(np.std([s[metric] for s in per_sample])),
                "min": float(np.min([s[metric] for s in per_sample])),
                "max": float(np.max([s[metric] for s in per_sample])),
            }
            for metric in ["dice", "iou", "accuracy", "sensitivity", "specificity"]
        },
    }
    with results_json_path.open("w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] Results saved to: {results_json_path}")

    # CSV per-sample
    if per_sample:
        with results_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_idx"] + list(per_sample[0].keys()))
            writer.writeheader()
            for idx, row in enumerate(per_sample):
                writer.writerow({"sample_idx": idx, **row})
        print(f"[eval] Per-sample CSV : {results_csv_path}")

    # -----------------------------------------------------------------------
    # Visualizations
    # -----------------------------------------------------------------------
    if args.visualize:
        vis_dir = args.outputs_dir / f"vis_{args.model}_{args.split}"
        print(f"\n[eval] Generating {args.num_vis} visualizations...")
        saved = generate_visualizations(
            model, eval_loader, device, vis_dir,
            num_samples=args.num_vis,
            is_imagenet_normalized=True,
        )
        print(f"[eval] Saved {len(saved)} visualizations to: {vis_dir}")

    print(f"\n{'='*70}")
    print("  Evaluation Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
