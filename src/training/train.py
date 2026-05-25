from __future__ import annotations

"""Full training pipeline with Transfer Learning for skin lesion segmentation.

Features:
- Pretrained encoder weights (ImageNet) via segmentation-models-pytorch / timm
- Discriminative learning rates (encoder LR << decoder LR)
- Gradual unfreezing: freeze encoder for N initial epochs
- CosineAnnealingWarmRestarts LR scheduler with warm-up
- Early stopping on validation Dice
- Mixed precision (CUDA only, skipped on DirectML/CPU)
- Best model checkpoint auto-save
- Training history logged to JSON

Usage:
    python src/training/train.py --model unet_resnet50 --epochs 50
    python src/training/train.py --model swin_unet --epochs 50 --batch-size 4
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Resolve project root so that relative imports work when running as script
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model_architectures import build_model_pretrained, get_device, list_models  # noqa: E402
from src.training.dataset import create_dataloaders  # noqa: E402
from src.training.losses import get_loss_function  # noqa: E402
from src.evaluation.metrics import MetricsTracker  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _separate_param_groups(
    model: nn.Module,
    model_name: str,
    encoder_lr: float,
    decoder_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Split model params into encoder (low LR) and decoder (high LR) groups.

    This is the core of discriminative fine-tuning: the pretrained encoder
    layers need a smaller LR to preserve learned features, while the randomly
    initialized decoder layers need a larger LR to learn quickly.
    """
    encoder_params: list[nn.Parameter] = []
    decoder_params: list[nn.Parameter] = []

    if model_name == "unet_resnet50":
        # smp.Unet stores encoder under model.net.encoder
        for name, param in model.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
    elif model_name == "swin_unet":
        # SwinUNet stores encoder under model.encoder
        for name, param in model.named_parameters():
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                decoder_params.append(param)
    else:
        # Fallback: everything as decoder LR
        decoder_params = list(model.parameters())

    groups = []
    if encoder_params:
        groups.append({
            "params": encoder_params,
            "lr": encoder_lr,
            "weight_decay": weight_decay,
            "name": "encoder",
        })
    if decoder_params:
        groups.append({
            "params": decoder_params,
            "lr": decoder_lr,
            "weight_decay": weight_decay,
            "name": "decoder",
        })

    return groups


def _freeze_encoder(model: nn.Module, model_name: str) -> int:
    """Freeze all encoder parameters. Returns count of frozen params."""
    count = 0
    if model_name == "unet_resnet50":
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False
                count += 1
    elif model_name == "swin_unet":
        for name, param in model.named_parameters():
            if name.startswith("encoder."):
                param.requires_grad = False
                count += 1
    return count


def _unfreeze_encoder(model: nn.Module, model_name: str) -> int:
    """Unfreeze all encoder parameters. Returns count of unfrozen params."""
    count = 0
    if model_name == "unet_resnet50":
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = True
                count += 1
    elif model_name == "swin_unet":
        for name, param in model.named_parameters():
            if name.startswith("encoder."):
                param.requires_grad = True
                count += 1
    return count


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ═══════════════════════════════════════════════════════════════════════════
# Training & Validation Steps
# ═══════════════════════════════════════════════════════════════════════════


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    limit_batches: int | None = None,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc="  Train",
        leave=False,
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    )
    for images, masks in pbar:
        if limit_batches is not None and n_batches >= limit_batches:
            break
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{running_loss / n_batches:.4f}"})

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics_tracker: MetricsTracker,
    limit_batches: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Validate and compute metrics. Returns (avg_loss, metrics_dict)."""
    model.eval()
    running_loss = 0.0
    n_batches = 0
    metrics_tracker.reset()

    pbar = tqdm(
        loader,
        desc="  Val",
        leave=False,
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    )
    for images, masks in pbar:
        if limit_batches is not None and n_batches >= limit_batches:
            break
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item()
        n_batches += 1

        metrics_tracker.update(outputs, masks, batch_size=images.size(0))
        pbar.set_postfix({"loss": f"{running_loss / n_batches:.4f}"})

    avg_loss = running_loss / max(n_batches, 1)
    metrics = metrics_tracker.compute()
    return avg_loss, metrics


# ═══════════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train segmentation model with Transfer Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model ---
    parser.add_argument(
        "--model",
        type=str,
        default="unet_resnet50",
        choices=list_models(),
        help="Model architecture",
    )
    parser.add_argument("--img-size", type=int, default=256, help="Input image size")

    # --- Data ---
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "processed",
        help="Root of processed data (contains run tag subdirectories)",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="ISIC2018_256_imagenet",
        help="Subdirectory name inside data-dir",
    )
    parser.add_argument(
        "--augmented-data-dir",
        type=Path,
        default=None,
        help="Root of augmented data (optional, same structure as data-dir)",
    )

    # --- Training ---
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Decoder learning rate")
    parser.add_argument(
        "--encoder-lr", type=float, default=1e-5, help="Encoder learning rate (lower for transfer learning)"
    )
    parser.add_argument(
        "--freeze-encoder",
        type=int,
        default=5,
        help="Freeze encoder for this many initial epochs (0 = no freezing)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--loss",
        type=str,
        default="bce_dice",
        choices=["dice", "bce_dice", "bce", "focal"],
        help="Loss function",
    )
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (0 = disabled)")

    # --- Output ---
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained weights",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save training history/logs",
    )

    # --- Misc ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 = main process)")
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Number of batches to run per epoch (for quick validation/dry-runs)",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    torch.manual_seed(args.seed)
    device = get_device()
    device_type = device.type  # "cuda", "privateuseone" (directml), or "cpu"

    # AMP only works reliably on CUDA
    use_amp = args.amp and device_type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print("=" * 70)
    print("  Skin Lesion Segmentation — Transfer Learning Training")
    print("=" * 70)
    print(f"  Model       : {args.model}")
    print(f"  Device      : {device} (type={device_type})")
    print(f"  Mixed Prec. : {'ON' if use_amp else 'OFF'}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch Size  : {args.batch_size}")
    print(f"  Decoder LR  : {args.lr}")
    print(f"  Encoder LR  : {args.encoder_lr}")
    print(f"  Freeze enc. : {args.freeze_encoder} epochs")
    print(f"  Loss        : {args.loss}")
    print(f"  Patience    : {args.patience}")
    print(f"  Seed        : {args.seed}")
    print("=" * 70)

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
        augmented_dir=args.augmented_data_dir,
        project_root=_PROJECT_ROOT,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(f"  Train set   : {len(train_loader.dataset)} samples")  # type: ignore[arg-type]
    print(f"  Val set     : {len(val_loader.dataset)} samples")  # type: ignore[arg-type]

    # -----------------------------------------------------------------------
    # Model (Transfer Learning — pretrained encoder)
    # -----------------------------------------------------------------------
    print(f"\n[model] Building {args.model} with pretrained encoder weights...")
    model = build_model_pretrained(args.model, img_size=args.img_size)
    model = model.to(device)

    total_params, trainable_params = _count_parameters(model)
    print(f"[model] Total params    : {total_params:,}")
    print(f"[model] Trainable params: {trainable_params:,}")

    # -----------------------------------------------------------------------
    # Gradual Unfreezing
    # -----------------------------------------------------------------------
    encoder_frozen = False
    if args.freeze_encoder > 0:
        n_frozen = _freeze_encoder(model, args.model)
        encoder_frozen = True
        _, trainable_after_freeze = _count_parameters(model)
        print(
            f"[transfer] Encoder FROZEN for {args.freeze_encoder} epochs "
            f"({n_frozen} param tensors frozen, {trainable_after_freeze:,} trainable)"
        )

    # -----------------------------------------------------------------------
    # Optimizer with Discriminative LR
    # -----------------------------------------------------------------------
    param_groups = _separate_param_groups(
        model,
        args.model,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    # Log param groups
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg["params"])
        print(f"[optimizer] {pg['name']:>8s}: lr={pg['lr']:.1e}, params={n_params:,}")

    # -----------------------------------------------------------------------
    # LR Scheduler — Cosine Annealing with Warm Restarts
    # -----------------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    # -----------------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------------
    criterion = get_loss_function(args.loss).to(device)
    print(f"[loss] {criterion.__class__.__name__}")

    # -----------------------------------------------------------------------
    # Metrics Tracker
    # -----------------------------------------------------------------------
    metric_names = ["dice", "iou", "accuracy", "sensitivity", "specificity"]
    val_tracker = MetricsTracker(metric_names)

    # -----------------------------------------------------------------------
    # Output dirs
    # -----------------------------------------------------------------------
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{timestamp}"
    checkpoint_path = args.models_dir / f"best_{args.model}.pth"
    history_path = args.outputs_dir / f"history_{run_name}.json"

    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------
    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        **{f"val_{m}": [] for m in metric_names},
        "lr_encoder": [],
        "lr_decoder": [],
        "epoch_time": [],
    }

    best_dice = 0.0
    patience_counter = 0

    print(f"\n{'='*70}")
    print("  Starting Training")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # --- Gradual Unfreezing ---
        if encoder_frozen and epoch > args.freeze_encoder:
            n_unfrozen = _unfreeze_encoder(model, args.model)
            encoder_frozen = False
            # Re-create optimizer to include now-unfrozen params
            param_groups = _separate_param_groups(
                model,
                args.model,
                encoder_lr=args.encoder_lr,
                decoder_lr=args.lr,
                weight_decay=args.weight_decay,
            )
            optimizer = torch.optim.AdamW(param_groups)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-7
            )
            print(
                f"\n[transfer] Epoch {epoch}: Encoder UNFROZEN "
                f"({n_unfrozen} param tensors). Discriminative LR active.\n"
            )

        # --- Train ---
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp,
            scaler,
            limit_batches=args.limit_batches,
        )

        # --- Validate ---
        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            val_tracker,
            limit_batches=args.limit_batches,
        )

        # --- LR Scheduler ---
        scheduler.step()

        # --- Logging ---
        epoch_time = time.time() - epoch_start
        current_lrs = {pg.get("name", f"group{i}"): pg["lr"] for i, pg in enumerate(optimizer.param_groups)}

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for m in metric_names:
            history[f"val_{m}"].append(val_metrics.get(m, 0.0))
        history["lr_encoder"].append(current_lrs.get("encoder", current_lrs.get("group0", 0.0)))
        history["lr_decoder"].append(current_lrs.get("decoder", current_lrs.get("group1", args.lr)))
        history["epoch_time"].append(epoch_time)

        val_dice = val_metrics.get("dice", 0.0)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_metrics.get('iou', 0.0):.4f} | "
            f"Time: {epoch_time:.1f}s"
            + (" [FROZEN]" if encoder_frozen else ""),
        )

        # --- Best model checkpoint ---
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": best_dice,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  * New best model saved (Dice={best_dice:.4f}) -> {checkpoint_path}")
        else:
            patience_counter += 1

        # --- Early Stopping ---
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n[early stop] No improvement for {args.patience} epochs. Stopping.")
            break

    # -----------------------------------------------------------------------
    # Save History
    # -----------------------------------------------------------------------
    # Convert any non-serializable types
    serializable_args = {}
    for k, v in vars(args).items():
        serializable_args[k] = str(v) if isinstance(v, Path) else v

    history_data = {
        "model": args.model,
        "args": serializable_args,
        "device": str(device),
        "best_dice": best_dice,
        "best_checkpoint": str(checkpoint_path),
        "epochs_completed": len(history["train_loss"]),
        "history": history,
    }
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("  Training Complete!")
    print(f"{'='*70}")
    print(f"  Best Val Dice : {best_dice:.4f}")
    print(f"  Checkpoint    : {checkpoint_path}")
    print(f"  History       : {history_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
