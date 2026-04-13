import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
SOURCE_DIR = DATASETS_DIR / "crack_segmentation"
OUTPUT_DIR = DATASETS_DIR / "crack_yolo"

IMAGE_DIR = SOURCE_DIR / "images"
MASK_DIR = SOURCE_DIR / "masks"
CLASS_NAME = "crack"
CLASS_ID = 0
RANDOM_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)
MIN_COMPONENT_AREA = 16
MIN_BOX_WIDTH = 4
MIN_BOX_HEIGHT = 4
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def collect_images(image_dir: Path) -> list[Path]:
    """Collect all supported image files from the source image directory."""
    images: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        images.extend(image_dir.glob(pattern))
    return sorted(images)


def find_valid_pairs(
    image_dir: Path, mask_dir: Path
) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Match each source image with a mask that uses the same file name."""
    pairs = []
    missing_masks = []

    for image_path in collect_images(image_dir):
        mask_path = mask_dir / image_path.name
        if mask_path.exists():
            pairs.append((image_path, mask_path))
        else:
            missing_masks.append(image_path.name)

    return pairs, missing_masks


def extract_boxes_from_mask(
    mask_path: Path,
) -> list[tuple[float, float, float, float]] | None:
    """Convert a binary crack mask into YOLO detection boxes.

    Each connected component becomes one bounding box. This is a better fit than
    polygon segmentation when the first project goal is real-time detection.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    height, width = mask.shape
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []

    for label_index in range(1, num_labels):
        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        w = int(stats[label_index, cv2.CC_STAT_WIDTH])
        h = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_index, cv2.CC_STAT_AREA])

        # Keep small but still meaningful thin cracks, while filtering obvious noise.
        if area < MIN_COMPONENT_AREA:
            continue
        if w < MIN_BOX_WIDTH and h < MIN_BOX_HEIGHT:
            continue

        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        box_width = w / width
        box_height = h / height
        boxes.append((x_center, y_center, box_width, box_height))

    return boxes


def write_yolo_detection_label(
    label_path: Path, boxes: list[tuple[float, float, float, float]]
) -> None:
    """Write one YOLO detection label file for a single image."""
    lines = [
        f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        for x_center, y_center, box_width, box_height in boxes
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def split_pairs(pairs: list[tuple[Path, Path]]) -> dict[str, list[tuple[Path, Path]]]:
    """Split image-mask pairs into reproducible train/val/test sets."""
    rng = random.Random(RANDOM_SEED)
    shuffled_pairs = pairs[:]
    rng.shuffle(shuffled_pairs)

    total = len(shuffled_pairs)
    train_end = int(total * SPLIT_RATIOS[0])
    val_end = train_end + int(total * SPLIT_RATIOS[1])

    return {
        "train": shuffled_pairs[:train_end],
        "val": shuffled_pairs[train_end:val_end],
        "test": shuffled_pairs[val_end:],
    }


def ensure_output_dirs(output_dir: Path) -> None:
    """Create the standard YOLO detection directory structure."""
    for split_name in ("train", "val", "test"):
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path) -> None:
    """Write the YOLO dataset configuration file."""
    yaml_content = (
        f"path: {output_dir.resolve()}\n\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 1\n\n"
        f"names:\n  {CLASS_ID}: {CLASS_NAME}\n"
    )
    (output_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")


def convert_dataset() -> None:
    """Convert crack_segmentation into a YOLO detection dataset."""
    if not IMAGE_DIR.exists() or not MASK_DIR.exists():
        raise FileNotFoundError(
            f"Expected source dataset under {SOURCE_DIR},"
            "but images/ or masks/ is missing."
        )

    pairs, missing_masks = find_valid_pairs(IMAGE_DIR, MASK_DIR)
    if not pairs:
        raise RuntimeError(f"No valid image/mask pairs found in {SOURCE_DIR}.")

    ensure_output_dirs(OUTPUT_DIR)
    write_data_yaml(OUTPUT_DIR)

    split_pairs_map = split_pairs(pairs)
    summary: Counter[str] = Counter()

    print(f"Source dataset: {SOURCE_DIR}")
    print(f"Output dataset: {OUTPUT_DIR}")
    print(f"Valid pairs: {len(pairs)}")
    print(f"Missing masks: {len(missing_masks)}")

    for split_name, split_items in split_pairs_map.items():
        print(f"\nProcessing {split_name} split with {len(split_items)} images...")
        split_object_count = 0
        split_empty_labels = 0

        for image_path, mask_path in tqdm(split_items, desc=split_name):
            boxes = extract_boxes_from_mask(mask_path)
            if boxes is None:
                summary["failed_masks"] += 1
                continue

            dst_image_path = OUTPUT_DIR / "images" / split_name / image_path.name
            dst_label_path = (
                OUTPUT_DIR / "labels" / split_name / f"{image_path.stem}.txt"
            )

            shutil.copy2(image_path, dst_image_path)
            write_yolo_detection_label(dst_label_path, boxes)

            summary[f"{split_name}_images"] += 1
            split_object_count += len(boxes)
            if not boxes:
                split_empty_labels += 1

        summary[f"{split_name}_objects"] = split_object_count
        summary[f"{split_name}_empty_labels"] = split_empty_labels

    print("\nConversion finished.")
    print(
        "Train/Val/Test images: "
        f"{summary['train_images']}/{summary['val_images']}/{summary['test_images']}"
    )
    print(
        "Train/Val/Test objects: "
        f"{summary['train_objects']}/{summary['val_objects']}/{summary['test_objects']}"
    )
    print(
        "Train/Val/Test empty labels: "
        f"{summary['train_empty_labels']}/{summary['val_empty_labels']}/{summary['test_empty_labels']}"
    )
    print(f"Unreadable masks: {summary['failed_masks']}")

    if missing_masks:
        sample_missing = ", ".join(missing_masks[:10])
        print(f"Sample images without masks: {sample_missing}")


if __name__ == "__main__":
    convert_dataset()
