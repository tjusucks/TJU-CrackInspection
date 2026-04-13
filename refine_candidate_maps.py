from pathlib import Path

import cv2
import numpy as np


IMAGE_DIR = Path("~/crack_datasets/images_denoise").expanduser()
CANDIDATE_DIR = Path("~/crack_datasets/kmeans_candidate_maps").expanduser()
OUTPUT_DIR = Path("~/crack_datasets/refined_candidate_maps").expanduser()

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

MIN_COMPONENT_AREA = 20
MIN_ASPECT_RATIO = 2.0
MIN_EXTENT = 0.08

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

BLACKHAT_KERNEL_SIZE = 9
OPEN_KERNEL_SIZE = 3


def find_image_path(relative_stem: Path):
    for ext in IMAGE_EXTS:
        candidate = IMAGE_DIR / relative_stem.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def load_candidate_map(label_up_path: Path):
    label_map = np.load(label_up_path)
    if label_map.ndim != 2:
        raise ValueError(f"Unexpected label map shape: {label_map.shape}")
    return label_map.astype(np.uint8)


def select_crack_cluster(gray, label_map):
    unique_labels = np.unique(label_map)
    best_label = None
    best_score = None

    for label in unique_labels:
        mask = (label_map == label)
        if mask.sum() == 0:
            continue

        mean_intensity = gray[mask].mean()
        area_ratio = mask.mean()

        score = mean_intensity + 40.0 * area_ratio

        if best_score is None or score < best_score:
            best_score = score
            best_label = label

    return best_label


def refine_inside_candidate(image_bgr, candidate_mask):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    gray_enhanced = clahe.apply(gray)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BLACKHAT_KERNEL_SIZE, BLACKHAT_KERNEL_SIZE)
    )
    blackhat = cv2.morphologyEx(gray_enhanced, cv2.MORPH_BLACKHAT, kernel)

    blackhat_candidate = cv2.bitwise_and(blackhat, blackhat, mask=candidate_mask)

    _, binary = cv2.threshold(
        blackhat_candidate,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    binary = cv2.bitwise_and(binary, binary, mask=candidate_mask)

    return binary, gray_enhanced, blackhat


def filter_connected_components(binary_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    filtered = np.zeros_like(binary_mask)

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect_ratio = long_side / short_side

        bbox_area = max(1, w * h)
        extent = area / bbox_area

        if aspect_ratio < MIN_ASPECT_RATIO:
            continue

        if extent < MIN_EXTENT:
            continue

        filtered[labels == label] = 255

    return filtered


def make_overlay(image_bgr, mask, color=(0, 0, 255), alpha=0.35):
    color_mask = np.zeros_like(image_bgr)
    color_mask[mask > 0] = color
    overlay = cv2.addWeighted(image_bgr, 1.0, color_mask, alpha, 0)
    return overlay


def process_one(label_up_path: Path):
    relative_path = label_up_path.relative_to(CANDIDATE_DIR)

    name = relative_path.name
    if not name.endswith("_labels_up.npy"):
        return False

    relative_stem = relative_path.with_name(name.replace("_labels_up.npy", ""))

    image_path = find_image_path(relative_stem)
    if image_path is None:
        print(f"Missing image for: {label_up_path}")
        return False

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"Failed to read image: {image_path}")
        return False

    label_map = load_candidate_map(label_up_path)

    image_h, image_w = image_bgr.shape[:2]
    if label_map.shape != (image_h, image_w):
        label_map = cv2.resize(label_map, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    crack_label = select_crack_cluster(gray, label_map)
    candidate_mask = np.zeros_like(label_map, dtype=np.uint8)
    candidate_mask[label_map == crack_label] = 255

    refined_binary, gray_enhanced, blackhat = refine_inside_candidate(image_bgr, candidate_mask)
    filtered_mask = filter_connected_components(refined_binary)

    candidate_overlay = make_overlay(image_bgr, candidate_mask, color=(255, 0, 0), alpha=0.25)
    refined_overlay = make_overlay(image_bgr, filtered_mask, color=(0, 0, 255), alpha=0.35)

    save_base = OUTPUT_DIR / relative_stem
    save_base.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(save_base) + "_candidate_mask.png", candidate_mask)
    cv2.imwrite(str(save_base) + "_gray_enhanced.png", gray_enhanced)
    cv2.imwrite(str(save_base) + "_blackhat.png", blackhat)
    cv2.imwrite(str(save_base) + "_refined_binary.png", refined_binary)
    cv2.imwrite(str(save_base) + "_filtered_mask.png", filtered_mask)
    cv2.imwrite(str(save_base) + "_candidate_overlay.png", candidate_overlay)
    cv2.imwrite(str(save_base) + "_refined_overlay.png", refined_overlay)

    np.save(str(save_base) + "_filtered_mask.npy", filtered_mask)

    return True


def main():
    if not CANDIDATE_DIR.exists():
        print(f"Candidate directory does not exist: {CANDIDATE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_files = sorted(CANDIDATE_DIR.rglob("*_labels_up.npy"))

    total = 0
    success = 0
    failed = 0

    for label_up_path in label_files:
        total += 1
        try:
            ok = process_one(label_up_path)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Processing error: {label_up_path} | Error: {e}")
            failed += 1

    print("Done")
    print(f"Total files: {total}")
    print(f"Successfully processed: {success}")
    print(f"Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()