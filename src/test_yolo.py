from pathlib import Path

from ultralytics.models.yolo import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SOURCE = REPO_ROOT / "datasets" / "crack_tests"
RUN_DIR = REPO_ROOT / "runs" / "yolo_detect" / "yolo11n_crack_detect"
WEIGHTS_PATH = RUN_DIR / "weights" / "best.pt"
OUTPUT_DIR = REPO_ROOT / "runs" / "yolo_test"
IMG_SIZE = 960
CONF_THRESHOLD = 0.25
DEVICE = "0"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_paths() -> None:
    """Fail early if the checkpoint or test source is missing."""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {WEIGHTS_PATH}\n"
            "Train the model first or update WEIGHTS_PATH."
        )
    if not TEST_SOURCE.exists():
        raise FileNotFoundError(f"Test source not found: {TEST_SOURCE}")


def collect_test_images(test_source: Path) -> list[Path]:
    """Collect supported test images from datasets/crack_tests."""
    if test_source.is_file():
        return [test_source] if test_source.suffix.lower() in IMAGE_EXTENSIONS else []

    return sorted(
        path
        for path in test_source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_test_inference(model: YOLO, image_paths: list[Path]) -> None:
    """Run inference on one or more test images and save visualized results."""
    model.predict(
        source=[str(path) for path in image_paths],
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        save=True,
        project=str(OUTPUT_DIR),
        name=WEIGHTS_PATH.stem,
        line_width=2,
        show_labels=True,
        show_conf=True,
    )

    print("\n" + "=" * 72)
    print(f"Checkpoint: {WEIGHTS_PATH}")
    print(f"Input dir:  {TEST_SOURCE}")
    print(f"Images:     {len(image_paths)}")
    print(f"Output dir: {OUTPUT_DIR / WEIGHTS_PATH.stem}")
    print("=" * 72)


def main() -> None:
    """Test a trained YOLO checkpoint on images in datasets/crack_tests."""
    validate_paths()
    image_paths = collect_test_images(TEST_SOURCE)

    if not image_paths:
        raise RuntimeError(f"No supported test images found in {TEST_SOURCE}")

    model = YOLO(str(WEIGHTS_PATH))
    run_test_inference(model, image_paths)


if __name__ == "__main__":
    main()
