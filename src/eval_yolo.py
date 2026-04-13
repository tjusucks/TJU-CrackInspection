from pathlib import Path

from ultralytics.models.yolo import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "datasets" / "crack_yolo" / "data.yaml"
RUN_DIR = REPO_ROOT / "runs" / "yolo_detect" / "yolo11n_crack_detect"
WEIGHTS_PATH = RUN_DIR / "weights" / "best.pt"
EVAL_PROJECT_DIR = REPO_ROOT / "runs" / "yolo_eval"
PREDICT_PROJECT_DIR = REPO_ROOT / "runs" / "yolo_predict"
TEST_SOURCE = REPO_ROOT / "datasets" / "crack_yolo" / "images" / "test"
IMG_SIZE = 960
CONF_THRESHOLD = 0.25
DEVICE = "0"


def validate_paths() -> None:
    """Fail early if the checkpoint or dataset config is missing."""
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {WEIGHTS_PATH}\n"
            "Train the model first or update WEIGHTS_PATH."
        )
    if not TEST_SOURCE.exists():
        raise FileNotFoundError(f"Test image directory not found: {TEST_SOURCE}")


def run_validation(model: YOLO) -> None:
    """Run Ultralytics validation and print key detection metrics."""
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMG_SIZE,
        batch=8,
        device=DEVICE,
        project=str(EVAL_PROJECT_DIR),
        name=WEIGHTS_PATH.stem,
        plots=True,
        save_json=False,
    )

    print("\n" + "=" * 72)
    print("Evaluation completed.")
    print(f"Checkpoint:             {WEIGHTS_PATH}")
    print(f"Box mAP@0.5:            {metrics.box.map50:.4f}")
    print(f"Box mAP@0.5:0.95:       {metrics.box.map:.4f}")
    print(f"Precision:              {metrics.box.mp:.4f}")
    print(f"Recall:                 {metrics.box.mr:.4f}")
    print("=" * 72)


def run_prediction(model: YOLO) -> None:
    """Generate visualization images for the test split."""
    model.predict(
        source=str(TEST_SOURCE),
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        save=True,
        project=str(PREDICT_PROJECT_DIR),
        name=WEIGHTS_PATH.stem,
        line_width=2,
        show_labels=True,
        show_conf=True,
    )

    print(
        f"Prediction visualizations saved under: {PREDICT_PROJECT_DIR / WEIGHTS_PATH.stem}"
    )


def main() -> None:
    """Evaluate a trained YOLO detection checkpoint and save predictions."""
    validate_paths()

    print("\n" + "=" * 72)
    print(f"Dataset:    {DATA_YAML}")
    print(f"Checkpoint: {WEIGHTS_PATH}")
    print(f"Test set:   {TEST_SOURCE}")
    print("=" * 72)

    model = YOLO(str(WEIGHTS_PATH))
    run_validation(model)
    run_prediction(model)


if __name__ == "__main__":
    main()
