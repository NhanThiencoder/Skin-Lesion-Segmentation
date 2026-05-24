# Agent notes for this repo

## Setup
- Install deps with `pip install -r requirements.txt`.

## Data and artifacts
- Large data and trained models are intentionally not in git: `data/`, `models/`, and most of `processed/` are ignored.
- Only small `processed/**/metadata.json` and `processed/**/manifests/*.json` are versioned; other `processed/` files come from local/Drive.

## Common commands
- Preprocess data: `python src/data_preprocessing/preprocess.py`.
- Train a model (example): `python src/training/train.py --model unet --epochs 50`.

## Structure hints
- Core code lives under `src/` with subfolders for preprocessing, models, training, evaluation, and deployment.
