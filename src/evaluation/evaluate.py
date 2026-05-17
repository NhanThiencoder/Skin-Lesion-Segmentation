import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation model (placeholder)")
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
        help="Directory containing trained weights (default: models/)",
    )
    args = parser.parse_args()

    print("[evaluate] data_dir  :", args.data_dir.resolve())
    print("[evaluate] models_dir:", args.models_dir.resolve())
    print("TODO: implement Dice/IoU/Accuracy metrics + visualization.")


if __name__ == "__main__":
    main()
