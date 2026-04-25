from pathlib import Path

import cv2
import numpy as np
from minisom import MiniSom
from skimage.filters import frangi
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
)
from skimage.measure import label, regionprops


INPUT_DIR = Path("~/tests/tests").expanduser()
OUTPUT_DIR = Path("~/tests/results").expanduser()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------- experiment mode ----------
# Paper setting: concrete -> 1x3, brick/shadow -> 1x5
SURFACE_TYPE = "brick"  # options: concrete, brick, shadow
CRACK_LABEL_MODE = "auto"  # options: auto, manual
MANUAL_CRACK_LABEL = 0

# ---------- image preprocessing ----------
CLAHE_CLIP = 1.3
CLAHE_GRID = (8, 8)
LOCAL_AVG_SMALL = 15
LOCAL_AVG_LARGE = 25
EDGE_SOBEL_KSIZE = 3
FRANGI_SIGMAS = range(2, 6)

# ---------- SOM ----------
SOM_H = 1
SOM_WIDTH_BY_SURFACE = {
    "concrete": 3,
    "brick": 5,
    "shadow": 5,
}
SOM_SIGMA = 1.0
SOM_LR = 0.5
SOM_EPOCHS = 200
SOM_SAMPLE_SIZE = 50000
RANDOM_SEED = 42

# ---------- crack-cluster auto identification (paper Sec. 2.4) ----------
CRACK_DARKNESS_W = 0.85
CRACK_THINNESS_W = 0.12
CRACK_CONTRAST_W = 0.10
MIN_CONNECTEDNESS = 0.25
MIN_CLUSTER_DARKNESS_DELTA = 0.06

# ---------- darkness gating ----------
PIXEL_DARKNESS_PERCENTILE = 62
MIN_PIXEL_DARKNESS_DELTA = 0.04
MIN_DARK_MASK_KEEP_RATIO = 0.20

# ---------- postprocessing ----------
CLOSE_RADIUS = 2
OPEN_RADIUS = 1
# Paper core rule: remove connected components smaller than 1% of largest crack.
MIN_COMPONENT_RATIO = 0.01
# Keep top-K largest connected components (by area).
KEEP_LARGEST_COMPONENTS = 6

# Remove block-like false positives: compact blobs usually have low elongation.
ENABLE_BLOB_SHAPE_FILTER = True
MIN_COMPONENT_ECCENTRICITY = 0.88
MIN_MAJOR_MINOR_RATIO = 2.40


def normalize01(x):
    x = x.astype(np.float32)
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def read_image(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def build_features(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    gray_eq = clahe.apply(gray)

    gray_f = gray_eq.astype(np.float32) / 255.0
    hue_f = hsv[:, :, 0].astype(np.float32) / 179.0

    local_avg_small = cv2.blur(gray_f, (LOCAL_AVG_SMALL, LOCAL_AVG_SMALL))
    local_avg_large = cv2.blur(gray_f, (LOCAL_AVG_LARGE, LOCAL_AVG_LARGE))

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=EDGE_SOBEL_KSIZE)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=EDGE_SOBEL_KSIZE)
    grad_mag = cv2.magnitude(gx, gy)

    # Thin-crack channel follows the paper's structural cue.
    thinness = frangi(gray_f, sigmas=FRANGI_SIGMAS, black_ridges=True)

    rel_gray_small = local_avg_small - gray_f
    rel_gray_large = local_avg_large - gray_f
    contrast = np.abs(gray_f - local_avg_small)

    gray_n = normalize01(gray_f)
    local_contrast_n = normalize01(contrast)
    edge_sharpness_n = normalize01(grad_mag)
    thinness_n = normalize01(thinness)
    local_avg_small_n = normalize01(local_avg_small)
    local_avg_large_n = normalize01(local_avg_large)
    rel_gray_small_n = normalize01(rel_gray_small)
    rel_gray_large_n = normalize01(rel_gray_large)
    hue_n = normalize01(hue_f)

    # 9D feature vector following the paper's design:
    # grayscale, local averages (9/25), relative grayscale (9/25),
    # edge strength, thinness, contrast, hue.
    feats = np.stack(
        [
            gray_n,
            local_avg_small_n,
            local_avg_large_n,
            rel_gray_small_n,
            rel_gray_large_n,
            edge_sharpness_n,
            thinness_n,
            local_contrast_n,
            hue_n,
        ],
        axis=-1,
    )

    aux = {
        "gray": gray,
        "gray_intensity": gray_n,
        "thinness": thinness_n,
        "contrast": local_contrast_n,
    }

    return feats.astype(np.float32), aux


def flatten_features(feats):
    h, w, c = feats.shape
    x = feats.reshape(-1, c).astype(np.float32)
    return np.clip(x, 0.0, 1.0), h, w


def get_som_width():
    return SOM_WIDTH_BY_SURFACE.get(SURFACE_TYPE, 3)


def train_som(x, som_w):
    n, c = x.shape

    if n > SOM_SAMPLE_SIZE:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(n, size=SOM_SAMPLE_SIZE, replace=False)
        x_train = x[idx]
    else:
        x_train = x

    som = MiniSom(
        SOM_H,
        som_w,
        c,
        sigma=SOM_SIGMA,
        learning_rate=SOM_LR,
        random_seed=RANDOM_SEED,
    )
    som.random_weights_init(x_train)
    som.train_batch(x_train, SOM_EPOCHS, verbose=False)

    return som


def som_predict_labels(som, x, som_w):
    labels = np.zeros((x.shape[0],), dtype=np.int32)
    for i, v in enumerate(x):
        r, c = som.winner(v)
        labels[i] = r * som_w + c
    return labels


def largest_component_ratio(mask):
    area = int(mask.sum())
    if area == 0:
        return 0.0

    labeled = label(mask)
    if labeled.max() == 0:
        return 0.0

    counts = np.bincount(labeled.ravel())[1:]
    largest = int(counts.max())
    return largest / float(area)


def select_crack_cluster(label_map, aux):
    if CRACK_LABEL_MODE == "manual":
        labels = np.unique(label_map)
        if int(MANUAL_CRACK_LABEL) in labels:
            return int(MANUAL_CRACK_LABEL)

    # Paper optional auto rule:
    # crack cluster = high thinness + low grayscale + connected structure.
    darkness = 1.0 - aux["gray_intensity"]
    thinness = aux["thinness"]
    contrast = aux["contrast"]
    global_darkness = float(darkness.mean())

    best_label = None
    best_score = -1e9
    fallback_label = None
    fallback_score = -1e9

    for label_id in np.unique(label_map):
        mask = label_map == label_id
        if mask.sum() == 0:
            continue

        dark_score = float(darkness[mask].mean())
        thin_score = float(thinness[mask].mean())
        contrast_score = float(contrast[mask].mean())
        connectedness = largest_component_ratio(mask)

        score = (
            + CRACK_DARKNESS_W * dark_score
            + CRACK_THINNESS_W * thin_score
            + CRACK_CONTRAST_W * contrast_score
        )

        if score > fallback_score:
            fallback_score = score
            fallback_label = int(label_id)

        is_dark_enough = dark_score >= (global_darkness + MIN_CLUSTER_DARKNESS_DELTA)
        if connectedness >= MIN_CONNECTEDNESS and is_dark_enough and score > best_score:
            best_score = score
            best_label = int(label_id)

    if best_label is not None:
        return best_label
    return fallback_label


def build_candidate_mask(label_map, crack_label):
    if crack_label is None:
        return np.zeros_like(label_map, dtype=bool)
    return label_map == crack_label


def apply_darkness_gate(candidate_mask, gray_intensity):
    if candidate_mask.sum() == 0:
        return candidate_mask

    darkness = 1.0 - gray_intensity
    global_darkness = float(darkness.mean())
    cluster_dark = darkness[candidate_mask]

    p_thr = float(np.percentile(cluster_dark, PIXEL_DARKNESS_PERCENTILE))
    abs_thr = global_darkness + MIN_PIXEL_DARKNESS_DELTA
    dark_thr = max(p_thr, abs_thr)

    refined = np.logical_and(candidate_mask, darkness >= dark_thr)

    min_keep = max(1, int(candidate_mask.sum() * MIN_DARK_MASK_KEEP_RATIO))
    if refined.sum() < min_keep:
        relaxed_percentile = max(50, PIXEL_DARKNESS_PERCENTILE - 10)
        relaxed_p_thr = float(np.percentile(cluster_dark, relaxed_percentile))
        relaxed_thr = max(relaxed_p_thr, global_darkness + 0.5 * MIN_PIXEL_DARKNESS_DELTA)
        refined = np.logical_and(candidate_mask, darkness >= relaxed_thr)

    return refined


def postprocess_candidate(mask):
    cleaned = binary_closing(mask, disk(CLOSE_RADIUS))
    cleaned = binary_opening(cleaned, disk(OPEN_RADIUS))
    return cleaned


def filter_components(mask_bool):
    # Paper postprocess rule: remove components < 1% of the largest crack component.
    labeled = label(mask_bool)
    if labeled.max() == 0:
        return mask_bool

    areas = np.bincount(labeled.ravel())[1:]
    largest_area = int(areas.max())
    min_keep_area = max(1, int(np.ceil(MIN_COMPONENT_RATIO * largest_area)))

    valid_regions = []
    for region in regionprops(labeled):
        area = int(region.area)
        if area < min_keep_area:
            continue

        if ENABLE_BLOB_SHAPE_FILTER:
            major = float(region.major_axis_length)
            minor = float(region.minor_axis_length)
            ecc = float(region.eccentricity)
            if major <= 1e-6:
                continue
            major_minor_ratio = major / max(minor, 1e-6)
            if major_minor_ratio < MIN_MAJOR_MINOR_RATIO and ecc < MIN_COMPONENT_ECCENTRICITY:
                continue

        valid_regions.append((int(region.label), area))

    if not valid_regions:
        return np.zeros_like(mask_bool, dtype=bool)

    valid_regions.sort(key=lambda x: x[1], reverse=True)
    keep_k = max(1, int(KEEP_LARGEST_COMPONENTS))
    selected_ids = {rid for rid, _ in valid_regions[:keep_k]}

    kept = np.zeros_like(mask_bool, dtype=bool)
    for rid in selected_ids:
        kept[labeled == rid] = True

    return kept


def make_overlay(image_bgr, mask_bool):
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[mask_bool] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 1.0, color, 0.45, 0)
    return overlay


def segment_image_bgr(image_bgr):
    feats, aux = build_features(image_bgr)
    x, h, w = flatten_features(feats)
    som_w = get_som_width()
    som = train_som(x, som_w)
    labels = som_predict_labels(som, x, som_w).reshape(h, w)

    crack_label = select_crack_cluster(labels, aux)
    candidate_mask = build_candidate_mask(labels, crack_label)
    candidate_mask = apply_darkness_gate(candidate_mask, aux["gray_intensity"])

    candidate_bool = postprocess_candidate(candidate_mask)
    final_bool = filter_components(candidate_bool)
    final_mask_u8 = (final_bool.astype(np.uint8) * 255)

    overlay = make_overlay(image_bgr, final_bool)
    return final_bool, final_mask_u8, overlay


def process_one(image_path: Path):
    image_bgr = read_image(image_path)
    if image_bgr is None:
        print(f"Failed to read: {image_path}")
        return False

    final_bool, final_mask_u8, overlay = segment_image_bgr(image_bgr)

    # Keep legacy output path as overlay for compatibility with existing workflow.
    legacy_overlay_path = (OUTPUT_DIR / image_path.stem).with_suffix(".png")
    cv2.imwrite(str(legacy_overlay_path), overlay)

    # Also save explicit mask/overlay files for direct evaluation.
    mask_path = OUTPUT_DIR / f"{image_path.stem}_mask.png"
    overlay_path = OUTPUT_DIR / f"{image_path.stem}_overlay.png"
    cv2.imwrite(str(mask_path), final_mask_u8)
    cv2.imwrite(str(overlay_path), overlay)

    return True


def main():
    if not INPUT_DIR.exists():
        print(f"Input directory does not exist: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in INPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    total = 0
    success = 0
    failed = 0

    for image_path in image_files:
        total += 1
        try:
            ok = process_one(image_path)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Processing error: {image_path} | Error: {e}")
            failed += 1

    print("Done")
    print(f"Total images: {total}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
