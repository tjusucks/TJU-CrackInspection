from pathlib import Path

import cv2
import numpy as np
from minisom import MiniSom
from skimage.filters import frangi
from skimage.morphology import (
    binary_closing,
    binary_opening,
    binary_dilation,
    binary_erosion,
    disk,
)
from skimage.measure import label, regionprops


INPUT_DIR = Path("~/tests").expanduser()
OUTPUT_DIR = Path("~/tests/results_v2").expanduser()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------- experiment mode ----------
SURFACE_TYPE = "brick"  # options: concrete, brick, shadow

# ---------- image preprocessing ----------
CLAHE_CLIP = 1.6
CLAHE_GRID = (8, 8)

LOCAL_AVG_SMALL = 15
LOCAL_AVG_MEDIUM = 31
LOCAL_AVG_LARGE = 51

EDGE_SOBEL_KSIZE = 3
FRANGI_SIGMAS = range(1, 4)

BLACKHAT_KERNEL_SMALL = 9
BLACKHAT_KERNEL_MEDIUM = 17
BLACKHAT_KERNEL_LARGE = 29

# ---------- SOM ----------
SOM_H = 1
SOM_WIDTH_BY_SURFACE = {
    "concrete": 4,
    "brick": 6,
    "shadow": 6,
}
SOM_SIGMA = 1.15
SOM_LR = 0.35
SOM_EPOCHS = 320
SOM_SAMPLE_SIZE = 60000
RANDOM_SEED = 42

# ---------- feature weighting ----------
# 重点提高 dark_prior，让颜色/明暗主导更强
FEATURE_WEIGHTS = np.array([
    1.10,  # gray
    0.35,  # local_avg_small
    0.35,  # local_avg_medium
    0.35,  # local_avg_large
    1.00,  # rel_gray_small
    1.05,  # rel_gray_medium
    1.15,  # rel_gray_large
    0.55,  # edge_sharpness
    0.25,  # thinness
    0.85,  # local contrast
    0.55,  # blackhat_small
    0.75,  # blackhat_medium
    0.95,  # blackhat_large
    1.05,  # valley
    1.80,  # dark_prior
    0.08,  # hue
    0.08,  # x coord
    0.08,  # y coord
], dtype=np.float32)

# ---------- crack-cluster scoring ----------
CRACK_DARKNESS_W = 1.65
CRACK_DARK_PRIOR_W = 1.80
CRACK_THINNESS_W = 0.20
CRACK_BLACKHAT_W = 0.80
CRACK_VALLEY_W = 1.20
CRACK_CONTRAST_W = 1.00
CRACK_EDGE_W = 0.35
CRACK_CONNECTEDNESS_W = 1.20
CRACK_LARGEST_AREA_W = 1.35
CRACK_ELONGATION_W = 0.12
CRACK_WIDTH_HINT_W = 0.60

MIN_CLUSTER_DARKNESS_DELTA = 0.025
MIN_CLUSTER_SCORE = 1.45
MAX_SELECTED_CLUSTERS = 2

# ---------- darkness gating ----------
PIXEL_DARKNESS_PERCENTILE = 56
MIN_PIXEL_DARKNESS_DELTA = 0.03
MIN_DARK_MASK_KEEP_RATIO = 0.42

# ---------- candidate enhancement ----------
ENABLE_LINE_ENHANCE_GATE = True
LINE_ENHANCE_PERCENTILE = 32

ENABLE_SPATIAL_VOTE = True
SPATIAL_VOTE_KERNEL = 9
SPATIAL_VOTE_RATIO = 0.20

# ---------- postprocessing ----------
CLOSE_RADIUS_1 = 2
CLOSE_RADIUS_2 = 4
OPEN_RADIUS = 1
BRIDGE_DILATE_RADIUS = 2

MIN_COMPONENT_RATIO = 0.008
KEEP_LARGEST_COMPONENTS = 8

ENABLE_BLOB_SHAPE_FILTER = True
MIN_COMPONENT_ECCENTRICITY = 0.76
MIN_MAJOR_MINOR_RATIO = 1.45
MIN_COMPONENT_AREA = 18


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


def get_som_width():
    return SOM_WIDTH_BY_SURFACE.get(SURFACE_TYPE, 5)


def multi_scale_blackhat(gray_f):
    gray_u8 = (gray_f * 255).astype(np.uint8)

    k1 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (BLACKHAT_KERNEL_SMALL, BLACKHAT_KERNEL_SMALL)
    )
    k2 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (BLACKHAT_KERNEL_MEDIUM, BLACKHAT_KERNEL_MEDIUM)
    )
    k3 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (BLACKHAT_KERNEL_LARGE, BLACKHAT_KERNEL_LARGE)
    )

    bh1 = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, k1).astype(np.float32) / 255.0
    bh2 = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, k2).astype(np.float32) / 255.0
    bh3 = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, k3).astype(np.float32) / 255.0

    return bh1, bh2, bh3


def build_features(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    gray_eq = clahe.apply(gray)

    gray_f = gray_eq.astype(np.float32) / 255.0
    hue_f = hsv[:, :, 0].astype(np.float32) / 179.0

    local_avg_small = cv2.blur(gray_f, (LOCAL_AVG_SMALL, LOCAL_AVG_SMALL))
    local_avg_medium = cv2.blur(gray_f, (LOCAL_AVG_MEDIUM, LOCAL_AVG_MEDIUM))
    local_avg_large = cv2.blur(gray_f, (LOCAL_AVG_LARGE, LOCAL_AVG_LARGE))

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=EDGE_SOBEL_KSIZE)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=EDGE_SOBEL_KSIZE)
    grad_mag = cv2.magnitude(gx, gy)

    thinness = frangi(gray_f, sigmas=FRANGI_SIGMAS, black_ridges=True)

    blackhat_small, blackhat_medium, blackhat_large = multi_scale_blackhat(gray_f)

    rel_gray_small = local_avg_small - gray_f
    rel_gray_medium = local_avg_medium - gray_f
    rel_gray_large = local_avg_large - gray_f

    contrast = np.abs(gray_f - local_avg_small)

    valley = np.maximum(local_avg_large - gray_f, 0.0)

    # ---------- dark prior ----------
    darkness = 1.0 - gray_f
    darkness_local = np.maximum(local_avg_large - gray_f, 0.0)
    dark_prior = 0.65 * normalize01(darkness) + 0.35 * normalize01(darkness_local)
    dark_prior = normalize01(dark_prior)

    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x_coord = xx.astype(np.float32) / max(w - 1, 1)
    y_coord = yy.astype(np.float32) / max(h - 1, 1)

    gray_n = normalize01(gray_f)
    local_avg_small_n = normalize01(local_avg_small)
    local_avg_medium_n = normalize01(local_avg_medium)
    local_avg_large_n = normalize01(local_avg_large)
    rel_gray_small_n = normalize01(rel_gray_small)
    rel_gray_medium_n = normalize01(rel_gray_medium)
    rel_gray_large_n = normalize01(rel_gray_large)
    edge_sharpness_n = normalize01(grad_mag)
    thinness_n = normalize01(thinness)
    local_contrast_n = normalize01(contrast)
    blackhat_small_n = normalize01(blackhat_small)
    blackhat_medium_n = normalize01(blackhat_medium)
    blackhat_large_n = normalize01(blackhat_large)
    valley_n = normalize01(valley)
    hue_n = normalize01(hue_f)

    feats = np.stack(
        [
            gray_n,
            local_avg_small_n,
            local_avg_medium_n,
            local_avg_large_n,
            rel_gray_small_n,
            rel_gray_medium_n,
            rel_gray_large_n,
            edge_sharpness_n,
            thinness_n,
            local_contrast_n,
            blackhat_small_n,
            blackhat_medium_n,
            blackhat_large_n,
            valley_n,
            dark_prior,
            hue_n,
            x_coord,
            y_coord,
        ],
        axis=-1,
    ).astype(np.float32)

    feats *= FEATURE_WEIGHTS.reshape(1, 1, -1)

    aux = {
        "gray": gray,
        "gray_intensity": gray_n,
        "thinness": thinness_n,
        "contrast": local_contrast_n,
        "edge": edge_sharpness_n,
        "blackhat": normalize01(
            0.20 * blackhat_small_n + 0.35 * blackhat_medium_n + 0.45 * blackhat_large_n
        ),
        "blackhat_large": blackhat_large_n,
        "valley": valley_n,
        "dark_prior": dark_prior,
    }
    return feats, aux


def flatten_features(feats):
    h, w, c = feats.shape
    x = feats.reshape(-1, c).astype(np.float32)
    return x, h, w


def train_som(x, som_w, sample_weight=None):
    n, c = x.shape

    if n > SOM_SAMPLE_SIZE:
        rng = np.random.default_rng(RANDOM_SEED)

        if sample_weight is not None:
            p = sample_weight.astype(np.float64).ravel()
            p = np.maximum(p, 1e-8)
            p = p / p.sum()
            idx = rng.choice(n, size=SOM_SAMPLE_SIZE, replace=False, p=p)
        else:
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


def largest_component_area_ratio(mask):
    if mask.sum() == 0:
        return 0.0
    labeled = label(mask)
    if labeled.max() == 0:
        return 0.0
    counts = np.bincount(labeled.ravel())[1:]
    largest = int(counts.max())
    h, w = mask.shape
    return largest / float(max(h * w, 1))


def estimate_cluster_elongation(mask):
    labeled = label(mask)
    if labeled.max() == 0:
        return 0.0

    best = 0.0
    for region in regionprops(labeled):
        major = float(region.major_axis_length)
        minor = float(region.minor_axis_length)
        if major < 1e-6:
            continue
        ratio = major / max(minor, 1e-6)
        score = min(ratio / 10.0, 1.0)
        best = max(best, score)
    return best


def estimate_cluster_width_hint(mask):
    labeled = label(mask)
    if labeled.max() == 0:
        return 0.0

    best_minor = 0.0
    for region in regionprops(labeled):
        best_minor = max(best_minor, float(region.minor_axis_length))
    return min(best_minor / 8.0, 1.0)


def select_crack_clusters(label_map, aux):
    darkness = 1.0 - aux["gray_intensity"]
    dark_prior = aux["dark_prior"]
    thinness = aux["thinness"]
    contrast = aux["contrast"]
    edge = aux["edge"]
    blackhat = aux["blackhat"]
    valley = aux["valley"]

    global_darkness = float(darkness.mean())
    scored = []

    for label_id in np.unique(label_map):
        mask = (label_map == label_id)
        area = int(mask.sum())
        if area == 0:
            continue

        dark_score = float(darkness[mask].mean())
        dark_prior_score = float(dark_prior[mask].mean())
        thin_score = float(thinness[mask].mean())
        contrast_score = float(contrast[mask].mean())
        edge_score = float(edge[mask].mean())
        blackhat_score = float(blackhat[mask].mean())
        valley_score = float(valley[mask].mean())
        connectedness = largest_component_ratio(mask)
        largest_area_ratio = largest_component_area_ratio(mask)
        elongation = estimate_cluster_elongation(mask)
        width_hint = estimate_cluster_width_hint(mask)

        score = (
            CRACK_DARKNESS_W * dark_score
            + CRACK_DARK_PRIOR_W * dark_prior_score
            + CRACK_THINNESS_W * thin_score
            + CRACK_BLACKHAT_W * blackhat_score
            + CRACK_VALLEY_W * valley_score
            + CRACK_CONTRAST_W * contrast_score
            + CRACK_EDGE_W * edge_score
            + CRACK_CONNECTEDNESS_W * connectedness
            + CRACK_LARGEST_AREA_W * largest_area_ratio
            + CRACK_ELONGATION_W * elongation
            + CRACK_WIDTH_HINT_W * width_hint
        )

        is_dark_enough = dark_score >= (global_darkness + MIN_CLUSTER_DARKNESS_DELTA)

        scored.append({
            "label_id": int(label_id),
            "score": float(score),
            "is_dark_enough": is_dark_enough,
        })

    if not scored:
        return []

    scored.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    for item in scored:
        if len(selected) >= MAX_SELECTED_CLUSTERS:
            break
        if item["score"] < MIN_CLUSTER_SCORE:
            continue
        if not item["is_dark_enough"]:
            continue
        selected.append(item["label_id"])

    if not selected:
        for item in scored:
            if item["is_dark_enough"]:
                selected.append(item["label_id"])
            if len(selected) >= 1:
                break

    if not selected:
        selected = [scored[0]["label_id"]]

    return selected


def build_candidate_mask(label_map, crack_labels):
    if crack_labels is None or len(crack_labels) == 0:
        return np.zeros_like(label_map, dtype=bool)

    mask = np.zeros_like(label_map, dtype=bool)
    for lb in crack_labels:
        mask |= (label_map == lb)
    return mask


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
        relaxed_percentile = max(40, PIXEL_DARKNESS_PERCENTILE - 15)
        relaxed_p_thr = float(np.percentile(cluster_dark, relaxed_percentile))
        relaxed_thr = max(relaxed_p_thr, global_darkness + 0.4 * MIN_PIXEL_DARKNESS_DELTA)
        refined = np.logical_and(candidate_mask, darkness >= relaxed_thr)

    return refined


def apply_line_enhance_gate(mask, aux):
    if not ENABLE_LINE_ENHANCE_GATE:
        return mask
    if mask.sum() == 0:
        return mask

    enhance_map = (
        0.08 * aux["thinness"]
        + 0.22 * aux["blackhat"]
        + 0.25 * aux["blackhat_large"]
        + 0.20 * aux["valley"]
        + 0.25 * aux["dark_prior"]
    )
    vals = enhance_map[mask]
    thr = float(np.percentile(vals, LINE_ENHANCE_PERCENTILE))
    refined = np.logical_and(mask, enhance_map >= thr)

    if refined.sum() < max(1, int(mask.sum() * 0.30)):
        return mask
    return refined


def apply_spatial_vote(mask):
    if not ENABLE_SPATIAL_VOTE:
        return mask
    if mask.sum() == 0:
        return mask

    k = SPATIAL_VOTE_KERNEL
    kernel = np.ones((k, k), np.float32)
    density = cv2.filter2D(mask.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT)
    density /= float(k * k)

    refined = density >= SPATIAL_VOTE_RATIO
    return refined


def bridge_crack_segments(mask):
    bridged = binary_dilation(mask, disk(BRIDGE_DILATE_RADIUS))
    bridged = binary_closing(bridged, disk(CLOSE_RADIUS_2))
    bridged = binary_erosion(bridged, disk(1))
    return bridged


def postprocess_candidate(mask):
    cleaned = binary_closing(mask, disk(CLOSE_RADIUS_1))
    cleaned = bridge_crack_segments(cleaned)
    cleaned = binary_closing(cleaned, disk(CLOSE_RADIUS_2))
    cleaned = binary_opening(cleaned, disk(OPEN_RADIUS))
    return cleaned


def filter_components(mask_bool, aux):
    labeled = label(mask_bool)
    if labeled.max() == 0:
        return mask_bool

    areas = np.bincount(labeled.ravel())[1:]
    largest_area = int(areas.max())
    min_keep_area = max(MIN_COMPONENT_AREA, int(np.ceil(MIN_COMPONENT_RATIO * largest_area)))

    kept = np.zeros_like(mask_bool, dtype=bool)
    region_list = []

    darkness_map = 1.0 - aux["gray_intensity"]
    support_map = (
        0.08 * aux["thinness"]
        + 0.18 * aux["blackhat"]
        + 0.22 * aux["blackhat_large"]
        + 0.22 * aux["valley"]
        + 0.30 * aux["dark_prior"]
    )

    for region in regionprops(labeled):
        area = int(region.area)
        if area < min_keep_area:
            continue

        rr, cc = region.coords[:, 0], region.coords[:, 1]

        major = float(region.major_axis_length)
        minor = float(region.minor_axis_length)
        ecc = float(region.eccentricity)

        if major <= 1e-6:
            continue

        major_minor_ratio = major / max(minor, 1e-6)
        width_score = min(minor / 8.0, 1.0)
        area_score = min(area / max(largest_area, 1), 1.0)
        elong_score = min(major_minor_ratio / 10.0, 1.0)

        if ENABLE_BLOB_SHAPE_FILTER:
            if major_minor_ratio < MIN_MAJOR_MINOR_RATIO and ecc < MIN_COMPONENT_ECCENTRICITY and area_score < 0.20:
                continue

        dark_mean = float(darkness_map[rr, cc].mean())
        support_mean = float(support_map[rr, cc].mean())

        score = (
            1.40 * dark_mean
            + 0.90 * support_mean
            + 0.70 * area_score
            + 0.48 * width_score
            + 0.18 * elong_score
        )

        region_list.append((int(region.label), area, score))

    if not region_list:
        return np.zeros_like(mask_bool, dtype=bool)

    region_list.sort(key=lambda x: (x[2], x[1]), reverse=True)
    keep_k = max(1, int(KEEP_LARGEST_COMPONENTS))
    selected_ids = {rid for rid, _, _ in region_list[:keep_k]}

    for rid in selected_ids:
        kept[labeled == rid] = True

    return kept


def make_overlay(image_bgr, mask_bool):
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[mask_bool] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 1.0, color, 0.45, 0)
    return overlay


def process_one(image_path: Path):
    image_bgr = read_image(image_path)
    if image_bgr is None:
        print(f"Failed to read: {image_path}")
        return False

    feats, aux = build_features(image_bgr)
    x, h, w = flatten_features(feats)

    som_w = get_som_width()

    # 暗区优先采样，减少背景大面积主导
    sample_weight = aux["dark_prior"].reshape(-1)
    sample_weight = 0.25 + 0.75 * sample_weight

    som = train_som(x, som_w, sample_weight=sample_weight)
    labels = som_predict_labels(som, x, som_w).reshape(h, w)

    crack_labels = select_crack_clusters(labels, aux)
    candidate_mask = build_candidate_mask(labels, crack_labels)

    candidate_mask = apply_darkness_gate(candidate_mask, aux["gray_intensity"])
    candidate_mask = apply_line_enhance_gate(candidate_mask, aux)
    candidate_mask = apply_spatial_vote(candidate_mask)

    candidate_bool = postprocess_candidate(candidate_mask)
    final_bool = filter_components(candidate_bool, aux)
    final_mask_u8 = (final_bool.astype(np.uint8) * 255)

    overlay = make_overlay(image_bgr, final_bool)

    legacy_overlay_path = (OUTPUT_DIR / image_path.stem).with_suffix(".png")
    cv2.imwrite(str(legacy_overlay_path), overlay)

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