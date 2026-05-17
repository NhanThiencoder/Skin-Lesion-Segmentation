import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train segmentation model (placeholder)")
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "ISIC2018",
        help="Path to dataset root (default: data/ISIC2018)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained weights (default: models/)",
    )
    args = parser.parse_args()

    print("[train] model     :", args.model)
    print("[train] epochs    :", args.epochs)
    print("[train] data_dir  :", args.data_dir.resolve())
    print("[train] models_dir:", args.models_dir.resolve())
    print("TODO: implement dataset loader + training loop.")


if __name__ == "__main__":
    main()
