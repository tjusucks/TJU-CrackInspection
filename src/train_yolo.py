import time
from pathlib import Path

import torch
from ultralytics import settings
from ultralytics.models.yolo import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "datasets" / "crack_yolo" / "data.yaml"
PROJECT_DIR = REPO_ROOT / "runs" / "yolo_detect"
MODELS_DIR = REPO_ROOT / "models"

# Choose a lightweight detection model first because the current project goal is
# robust real-time crack detection, not segmentation quality.
MODEL_NAME = "yolo11n.pt"
MODEL_PATH = MODELS_DIR / MODEL_NAME

# Training settings.
EPOCHS = 100
IMG_SIZE = 960
BATCH_SIZE = 8
PATIENCE = 20
WORKERS = 4
DEVICE = "0" if torch.cuda.is_available() else "cpu"
RUN_NAME = f"{MODEL_PATH.stem}_crack_detect"


def configure_ultralytics_dirs() -> None:
    """Store pretrained weights and run outputs under project directories."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    settings.update(
        {
            "weights_dir": str(MODELS_DIR),
            "runs_dir": str(REPO_ROOT / "runs"),
            "datasets_dir": str(REPO_ROOT / "datasets"),
        }
    )


def validate_paths() -> None:
    """Fail early if the converted detection dataset is missing."""
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_YAML}\n"
            "Run src/mass_production_converter.py first, or update the dataset path."
        )


def print_run_config() -> None:
    """Print the training configuration for the current run."""
    print("\n" + "=" * 72)
    print(f"Model:      {MODEL_PATH}")
    print(f"Dataset:    {DATA_YAML}")
    print(f"Project:    {PROJECT_DIR}")
    print(f"Run name:   {RUN_NAME}")
    print(f"Epochs:     {EPOCHS}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device:     {DEVICE}")
    print("=" * 72)


def train() -> None:
    """Train a YOLO detection model on the converted crack dataset."""
    configure_ultralytics_dirs()
    validate_paths()
    print_run_config()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = YOLO(str(MODEL_PATH if MODEL_PATH.exists() else MODEL_NAME))

    start_time = time.time()
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        device=DEVICE,
        workers=WORKERS,
        patience=PATIENCE,
        pretrained=True,
        close_mosaic=10,
        cache=False,
        degrees=0.0,
        translate=0.05,
        scale=0.20,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.30,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
    )
    train_time_minutes = (time.time() - start_time) / 60

    metrics = results.box

    print("\n" + "=" * 72)
    print("Training completed.")
    print(f"Training time (minutes): {train_time_minutes:.2f}")
    print(f"Box mAP@0.5:            {metrics.map50:.4f}")
    print(f"Box mAP@0.5:0.95:       {metrics.map:.4f}")
    print(f"Precision:              {metrics.mp:.4f}")
    print(f"Recall:                 {metrics.mr:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    train()
