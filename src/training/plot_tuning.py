from __future__ import annotations

"""Tuning analyzer and visualization generator.

Reads outputs/tuning/{model_name}/summary.json and generates plots:
- Optimization history (Dice score vs. trial index)
- Hyperparameter scatter plots (Learning rate vs. Dice)
- Categorical comparisons (Loss function impact on Dice)

Usage:
    python src/training/plot_tuning.py --model unet_resnet50
    python src/training/plot_tuning.py --model swin_unet
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Style configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Inter", "DejaVu Sans"],
    "figure.titlesize": 16,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# Curated palette
PRIMARY_COLOR = "#1976D2"  # Blue
SECONDARY_COLOR = "#388E3C"  # Green
ACCENT_COLOR = "#D32F2F"  # Red
DARK_NEUTRAL = "#37474F"  # Slate gray


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize hyperparameter tuning results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet_resnet50",
        help="Model directory to read from outputs/tuning/",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory for outputs",
    )

    args = parser.parse_args()

    tuning_dir = args.outputs_dir / "tuning" / args.model
    summary_path = tuning_dir / "summary.json"

    if not summary_path.exists():
        print(f"Error: summary file not found: {summary_path}")
        print("Please run tuning first.")
        sys.exit(1)

    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    trials = data.get("trials", [])
    if not trials:
        print("No trials found in summary.")
        sys.exit(0)

    # Flatten trial information into a list of dicts for pandas
    rows = []
    for t in trials:
        if t["status"] != "COMPLETE":
            # Skip failed or pruned trials from metric plotting
            continue
        
        row = {
            "trial_id": t["trial_id"],
            "duration_sec": t["duration_sec"],
            "status": t["status"],
            "lr": t["params"]["lr"],
            "encoder_lr": t["params"]["encoder_lr"],
            "weight_decay": t["params"]["weight_decay"],
            "freeze_encoder": t["params"]["freeze_encoder"],
            "loss": t["params"]["loss"],
            "dice": t["metrics"].get("dice", 0.0),
            "iou": t["metrics"].get("iou", 0.0),
            "accuracy": t["metrics"].get("accuracy", 0.0),
            "sensitivity": t["metrics"].get("sensitivity", 0.0),
            "specificity": t["metrics"].get("specificity", 0.0),
        }
        rows.append(row)

    if not rows:
        print("No completed trials found to visualize.")
        sys.exit(0)

    df = pd.DataFrame(rows)

    plots_dir = tuning_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyzer] Analyzing {len(df)} completed trials...")
    print(f"[analyzer] Best configuration found:")
    best_row = df.loc[df["dice"].idxmax()]
    print(f"  Trial ID     : #{best_row['trial_id']}")
    print(f"  Val Dice     : {best_row['dice']:.4f}")
    print(f"  Val IoU      : {best_row['iou']:.4f}")
    print(f"  Decoder LR   : {best_row['lr']:.2e}")
    print(f"  Encoder LR   : {best_row['encoder_lr']:.2e}")
    print(f"  Loss Function: {best_row['loss']}")
    print(f"  Freeze Enc.  : {best_row['freeze_encoder']} epochs")

    # ═══════════════════════════════════════════════════════════════════════
    # Plot 1: Optimization History
    # ═══════════════════════════════════════════════════════════════════════
    plt.figure(figsize=(10, 5))
    running_max = np.maximum.accumulate(df["dice"].values)
    
    plt.plot(df["trial_id"], df["dice"], "o-", color=PRIMARY_COLOR, label="Trial Val Dice", alpha=0.6)
    plt.plot(df["trial_id"], running_max, "s--", color=SECONDARY_COLOR, label="Best So Far", linewidth=2)
    
    # Highlight overall best
    best_idx = df["dice"].idxmax()
    plt.scatter(
        df.loc[best_idx, "trial_id"], df.loc[best_idx, "dice"],
        color=ACCENT_COLOR, s=150, zorder=5, edgecolor="black", linewidth=1.5,
        label=f"Best (Trial #{df.loc[best_idx, 'trial_id']}: {df.loc[best_idx, 'dice']:.4f})"
    )

    plt.title(f"Optimization History — {args.model}")
    plt.xlabel("Trial ID")
    plt.ylabel("Validation Dice Score")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plot1_path = plots_dir / "01_optimization_history.png"
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"[analyzer] Saved plot: {plot1_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # Plot 2: Loss Function Performance
    # ═══════════════════════════════════════════════════════════════════════
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="loss", y="dice", data=df, palette="crest", hue="loss", legend=False)
    sns.stripplot(x="loss", y="dice", data=df, color=DARK_NEUTRAL, size=6, jitter=0.1, alpha=0.8)
    
    plt.title(f"Validation Dice by Loss Function — {args.model}")
    plt.xlabel("Loss Function")
    plt.ylabel("Validation Dice")
    plt.tight_layout()
    plot2_path = plots_dir / "02_loss_comparison.png"
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"[analyzer] Saved plot: {plot2_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # Plot 3: Learning Rate vs Dice (Scatter)
    # ═══════════════════════════════════════════════════════════════════════
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        df["lr"], df["encoder_lr"],
        c=df["dice"], cmap="viridis",
        s=df["duration_sec"] * 0.1 + 50, # size depends on duration
        edgecolors="black", alpha=0.8
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Validation Dice", rotation=270, labelpad=15)
    
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"Learning Rates Search Space — {args.model}")
    plt.xlabel("Decoder Learning Rate (log scale)")
    plt.ylabel("Encoder Learning Rate (log scale)")
    
    # Highlight best
    plt.scatter(
        best_row["lr"], best_row["encoder_lr"],
        facecolors="none", edgecolors=ACCENT_COLOR, s=200, linewidths=2,
        label="Best Config"
    )
    plt.legend()
    plt.tight_layout()
    plot3_path = plots_dir / "03_learning_rates_search.png"
    plt.savefig(plot3_path, dpi=150)
    plt.close()
    print(f"[analyzer] Saved plot: {plot3_path}")
    print("[analyzer] Done!")


if __name__ == "__main__":
    main()
