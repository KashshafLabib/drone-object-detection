"""
Upload existing training results to Weights & Biases (no retraining required).

Usage:
    pip install wandb pandas
    python scripts/upload_to_wandb.py

This script reads the two training_log.csv files already in results/ and
logs them as proper W&B runs, then attaches the model weights as artifacts.
"""

from pathlib import Path

import pandas as pd
import wandb

# ── Configuration ────────────────────────────────────────────
PROJECT_NAME = "drone-object-detection"  # Your W&B project name
ENTITY = None  # Your W&B username (None = auto-detect)

ROOT = Path(__file__).parent.parent

RUNS = [
    {
        "name": "baseline-640px",
        "csv": ROOT / "results/baseline_640/training_log.csv",
        "weights": ROOT / "weights/baseline_640/best.pt",
        "config": {
            "model": "YOLOv11m",
            "dataset": "VisDrone 2019 DET (2-class filtered)",
            "imgsz": 640,
            "epochs": 50,
            "batch_size": 16,
            "optimizer": "AdamW",
            "lr": 0.001,
            "classes": ["Human", "Car"],
            "pretrained": True,
        },
    },
    {
        "name": "optimized-1280px",
        "csv": ROOT / "results/optimized_1280/training_log.csv",
        "weights": ROOT / "weights/optimized_1280/best.pt",
        "config": {
            "model": "YOLOv11m",
            "dataset": "VisDrone 2019 DET (2-class filtered)",
            "imgsz": 1280,
            "epochs": 100,
            "early_stop": 86,
            "batch_size": 4,
            "optimizer": "AdamW",
            "lr": 0.0005,
            "patience": 20,
            "classes": ["Human", "Car"],
            "pretrained": True,
        },
    },
]

# Column mapping: CSV column name → clean W&B metric name
COLUMN_MAP = {
    "train/box_loss": "train/box_loss",
    "train/cls_loss": "train/cls_loss",
    "train/dfl_loss": "train/dfl_loss",
    "metrics/precision(B)": "metrics/precision",
    "metrics/recall(B)": "metrics/recall",
    "metrics/mAP50(B)": "metrics/mAP@0.5",
    "metrics/mAP50-95(B)": "metrics/mAP@0.5:0.95",
    "val/box_loss": "val/box_loss",
    "val/cls_loss": "val/cls_loss",
    "val/dfl_loss": "val/dfl_loss",
    "lr/pg0": "lr/pg0",
}


def upload_run(run_cfg: dict):
    print(f"\n── Uploading: {run_cfg['name']} ──")

    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY,
        name=run_cfg["name"],
        config=run_cfg["config"],
        tags=["VisDrone", "YOLOv11", "aerial-detection"],
        notes="Uploaded from local training_log.csv — no retraining required.",
    )

    # Log metrics epoch-by-epoch
    df = pd.read_csv(run_cfg["csv"])
    for _, row in df.iterrows():
        log_dict = {"epoch": int(row["epoch"])}
        for csv_col, wb_name in COLUMN_MAP.items():
            if csv_col in row:
                log_dict[wb_name] = float(row[csv_col])
        wandb.log(log_dict)

    # Log final best metrics as summary
    best_epoch = df["metrics/mAP50(B)"].idxmax()
    best_row = df.loc[best_epoch]
    run.summary["best_mAP@0.5"] = best_row["metrics/mAP50(B)"]
    run.summary["best_mAP@0.5:0.95"] = best_row["metrics/mAP50-95(B)"]
    run.summary["best_precision"] = best_row["metrics/precision(B)"]
    run.summary["best_recall"] = best_row["metrics/recall(B)"]
    run.summary["best_epoch"] = int(best_row["epoch"])
    run.summary["total_epochs"] = len(df)

    # Log model weights as a W&B Artifact
    weights_path = run_cfg["weights"]
    if weights_path.exists():
        artifact = wandb.Artifact(
            name=f"model-{run_cfg['name']}",
            type="model",
            description=f"YOLOv11m best.pt — {run_cfg['name']}",
            metadata=run_cfg["config"],
        )
        artifact.add_file(str(weights_path))
        run.log_artifact(artifact)
        print(f"  ✅ Weights logged: {weights_path.name}")
    else:
        print(f"  ⚠️  Weights not found at {weights_path} — skipping artifact.")

    run.finish()
    print(f"  ✅ Run complete: {run.url}")


if __name__ == "__main__":
    wandb.login()  # Will prompt for API key on first run
    for run_cfg in RUNS:
        upload_run(run_cfg)

    print("\n✅ All runs uploaded. View your dashboard at:")
    print(f"   https://wandb.ai/{ENTITY or '<your-username>'}/{PROJECT_NAME}")
