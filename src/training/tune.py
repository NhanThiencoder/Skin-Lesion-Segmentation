from __future__ import annotations

"""Hyperparameter tuning pipeline using Optuna.

Supports:
- TPE (Tree-structured Parzen Estimator) search for learning rates, weight decay, freeze epochs, and loss functions.
- Automatic trial pruning (early stopping of sub-optimal trials).
- CUDA device check and AMP (Automatic Mixed Precision) activation.
- Disk space cleanup: deletes intermediate weights, keeping only the best overall model weights.
- Clean folder organization: outputs everything to outputs/tuning/{model_name}/.

Usage:
    python src/training/tune.py --model unet_resnet50 --num-trials 10 --epochs 15
    python src/training/tune.py --model swin_unet --num-trials 10 --epochs 15 --device cpu
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import optuna
except ImportError:
    print("Optuna is not installed. Please install it using: pip install optuna")
    sys.exit(1)

from src.model_architectures import build_model_pretrained, get_device, list_models
from src.training.dataset import create_dataloaders
from src.training.losses import get_loss_function
from src.evaluation.metrics import MetricsTracker
from src.training.train import (
    train_one_epoch,
    validate,
    _separate_param_groups,
    _freeze_encoder,
    _unfreeze_encoder,
    _count_parameters,
)

# ---------------------------------------------------------------------------
# Globals for tracking the absolute best model across all trials
# ---------------------------------------------------------------------------
BEST_GLOBAL_DICE = 0.0


def save_trial_summary(
    output_dir: Path,
    trial_id: int,
    params: dict,
    metrics: dict | None,
    status: str,
    duration: float,
) -> None:
    """Append or update this trial's stats in outputs/tuning/{model_name}/summary.json."""
    summary_path = output_dir / "summary.json"
    
    summary_data = {"trials": []}
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    trial_entry = {
        "trial_id": trial_id,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "duration_sec": duration,
        "params": params,
        "metrics": metrics or {},
    }

    # If trial already exists in summary (e.g. updated status), replace it, otherwise append
    exists = False
    for idx, entry in enumerate(summary_data["trials"]):
        if entry["trial_id"] == trial_id:
            summary_data["trials"][idx] = trial_entry
            exists = True
            break
    if not exists:
        summary_data["trials"].append(trial_entry)

    # Sort trials by ID
    summary_data["trials"].sort(key=lambda x: x["trial_id"])

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)


def make_objective(args, device, train_loader, val_loader, tuning_dir: Path):
    """Factory to create the objective function for Optuna study."""
    
    def objective(trial: optuna.Trial) -> float:
        global BEST_GLOBAL_DICE
        trial_id = trial.number
        trial_dir = tuning_dir / "trials" / f"trial_{trial_id:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # 1. Suggest Hyperparameters
        lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
        encoder_lr = trial.suggest_float("encoder_lr", 5e-6, 5e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        freeze_encoder = trial.suggest_int("freeze_encoder", 0, 10)
        loss_name = trial.suggest_categorical("loss", ["bce_dice", "dice", "focal"])

        params = {
            "lr": lr,
            "encoder_lr": encoder_lr,
            "weight_decay": weight_decay,
            "freeze_encoder": freeze_encoder,
            "loss": loss_name,
        }

        print(f"\n[tuning] Starting Trial {trial_id} with parameters:")
        print(json.dumps(params, indent=2))

        # 2. Build Model
        model = build_model_pretrained(args.model, img_size=args.img_size)
        model = model.to(device)

        # 3. Handle Initial Encoder Freezing
        encoder_frozen = False
        if freeze_encoder > 0:
            _freeze_encoder(model, args.model)
            encoder_frozen = True

        # 4. Set up Optimizer and Scheduler
        param_groups = _separate_param_groups(
            model,
            args.model,
            encoder_lr=encoder_lr,
            decoder_lr=lr,
            weight_decay=weight_decay,
        )
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )

        criterion = get_loss_function(loss_name).to(device)
        val_tracker = MetricsTracker(["dice", "iou", "accuracy", "sensitivity", "specificity"])

        # AMP setup (CUDA only)
        use_amp = args.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # Trial progress history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
        }

        trial_best_dice = 0.0
        trial_best_ckpt_path = trial_dir / f"best_trial_{trial_id}.pth"
        start_time = time.time()

        try:
            for epoch in range(1, args.epochs + 1):
                # Unfreeze encoder if epoch threshold reached
                if encoder_frozen and epoch > freeze_encoder:
                    _unfreeze_encoder(model, args.model)
                    encoder_frozen = False
                    param_groups = _separate_param_groups(
                        model,
                        args.model,
                        encoder_lr=encoder_lr,
                        decoder_lr=lr,
                        weight_decay=weight_decay,
                    )
                    optimizer = torch.optim.AdamW(param_groups)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=10, T_mult=2, eta_min=1e-7
                    )

                # Train
                train_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, device,
                    use_amp=use_amp, scaler=scaler, limit_batches=args.limit_batches
                )

                # Validate
                val_loss, val_metrics = validate(
                    model, val_loader, criterion, device, val_tracker,
                    limit_batches=args.limit_batches
                )

                # Scheduler step
                scheduler.step()

                val_dice = val_metrics.get("dice", 0.0)
                val_iou = val_metrics.get("iou", 0.0)

                # Log epoch history
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_dice"].append(val_dice)
                history["val_iou"].append(val_iou)

                print(
                    f"  Trial {trial_id} | Epoch {epoch:2d}/{args.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
                )

                # Check if this is the best epoch in this specific trial
                if val_dice > trial_best_dice:
                    trial_best_dice = val_dice
                    # Save checkpoint for this trial
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_dice": val_dice,
                        "val_metrics": val_metrics,
                        "params": params,
                    }, trial_best_ckpt_path)

                # Optuna report & prune
                trial.report(val_dice, epoch)
                if trial.should_prune():
                    print(f"[tuning] Trial {trial_id} pruned at epoch {epoch}")
                    # Save status to summary and clean up weights
                    duration = time.time() - start_time
                    save_trial_summary(tuning_dir, trial_id, params, None, "PRUNED", duration)
                    if trial_best_ckpt_path.exists():
                        os.remove(trial_best_ckpt_path)
                    raise optuna.exceptions.TrialPruned()

            # --- End of Training Trial ---
            duration = time.time() - start_time
            
            # Load trial's best weights to save final metrics correctly
            checkpoint = torch.load(trial_best_ckpt_path, map_location="cpu")
            final_metrics = checkpoint["val_metrics"]
            final_dice = final_metrics["dice"]

            # Save trial history JSON
            with (trial_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            # Check if this trial is the absolute best across all trials so far
            if final_dice > BEST_GLOBAL_DICE:
                BEST_GLOBAL_DICE = final_dice
                global_best_path = tuning_dir / "best_tuned_model.pth"
                shutil.copy(trial_best_ckpt_path, global_best_path)
                print(f"  *** NEW BEST GLOBAL MODEL: Dice = {final_dice:.4f} -> Saved to {global_best_path} ***")

            # Clean up the trial's local weight file to save disk space
            if trial_best_ckpt_path.exists():
                os.remove(trial_best_ckpt_path)

            save_trial_summary(tuning_dir, trial_id, params, final_metrics, "COMPLETE", duration)
            return final_dice

        except Exception as e:
            # Handle unexpected failures cleanly
            duration = time.time() - start_time
            print(f"[tuning] Trial {trial_id} failed with exception: {e}")
            save_trial_summary(tuning_dir, trial_id, params, None, "FAIL", duration)
            if trial_best_ckpt_path.exists():
                os.remove(trial_best_ckpt_path)
            raise e

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for skin lesion segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Core args ---
    parser.add_argument(
        "--model",
        type=str,
        default="unet_resnet50",
        choices=list_models(),
        help="Model architecture to tune",
    )
    parser.add_argument("--img-size", type=int, default=256, help="Input image size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs per trial")
    parser.add_argument("--num-trials", type=int, default=10, help="Number of tuning trials")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g. 'cpu', 'cuda'). If not specified, auto-detected.",
    )
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only)")
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Number of batches to run per epoch (for quick dry-runs)",
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
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory for outputs",
    )

    args = parser.parse_args()

    # --- Setup Device ---
    device = torch.device(args.device) if args.device is not None else get_device()
    
    # Clean output dirs
    tuning_dir = args.outputs_dir / "tuning" / args.model
    tuning_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Skin Lesion Segmentation — Hyperparameter Tuning")
    print("=" * 70)
    print(f"  Model       : {args.model}")
    print(f"  Device      : {device}")
    print(f"  Trials      : {args.num_trials}")
    print(f"  Epochs/Trial: {args.epochs}")
    print(f"  Outputs     : {tuning_dir}")
    print("=" * 70)

    # --- Load Data ---
    data_root = args.data_dir / args.run_tag
    if not data_root.exists():
        raise SystemExit(f"Data directory does not exist: {data_root}")

    loaders = create_dataloaders(
        data_dir=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        project_root=_PROJECT_ROOT,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # --- Optuna Study ---
    study_name = f"tuning_{args.model}"
    
    # We want to maximize validation Dice score
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
    )

    objective_fn = make_objective(args, device, train_loader, val_loader, tuning_dir)
    
    study.optimize(objective_fn, n_trials=args.num_trials)

    print("\n" + "=" * 70)
    print("  Hyperparameter Tuning Phase Complete!")
    print("=" * 70)
    print(f"  Best Trial: #{study.best_trial.number}")
    print(f"  Best Val Dice: {study.best_value:.4f}")
    print("  Best Parameters:")
    print(json.dumps(study.best_params, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
