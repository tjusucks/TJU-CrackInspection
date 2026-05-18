from pathlib import Path

import cv2
import numpy as np
from minisom import MiniSom
from skimage.filters import frangi
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    disk,
    skeletonize,
)
from skimage.measure import label, regionprops


INPUT_DIR = Path("~/tests/tests").expanduser()
OUTPUT_DIR = Path("~/tests/results").expanduser()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# SOM is expensive on multi-megapixel images. Running the unsupervised model on
# a moderately resized copy is usually enough for crack localization, then the
# binary mask is mapped back to the original resolution.
MAX_PROCESS_SIDE = 1600

# ---------- experiment mode ----------
# Paper setting: concrete -> 1x3, brick/shadow -> 1x5
SURFACE_TYPE = "brick"  # options: concrete, brick, shadow
CRACK_LABEL_MODE = "auto"  # options: auto, manual
MANUAL_CRACK_LABEL = 0

# ---------- image preprocessing ----------
CLAHE_CLIP = 1.6
CLAHE_GRID = (8, 8)
LOCAL_AVG_SMALL = 15
LOCAL_AVG_MEDIUM = 31
LOCAL_AVG_LARGE = 51
EDGE_SOBEL_KSIZE = 3
FRANGI_SIGMAS = range(1, 5)

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
FEATURE_WEIGHTS = np.array([
    1.10,  # gray
    0.35,  # local_avg_small
    0.35,  # local_avg_medium
    0.35,  # local_avg_large
    1.00,  # rel_gray_small
    1.05,  # rel_gray_medium
    1.15,  # rel_gray_large
    0.55,  # edge_sharpness
    0.30,  # thinness
    0.85,  # local contrast
    0.55,  # blackhat_small
    0.75,  # blackhat_medium
    0.95,  # blackhat_large
    1.05,  # valley
    1.75,  # dark_prior
    0.08,  # hue
    0.08,  # x coord
    0.08,  # y coord
], dtype=np.float32)

# ---------- crack-cluster auto identification ----------
CRACK_DARKNESS_W = 1.55
CRACK_DARK_PRIOR_W = 1.75
CRACK_THINNESS_W = 0.25
CRACK_BLACKHAT_W = 0.85
CRACK_VALLEY_W = 1.15
CRACK_CONTRAST_W = 0.95
CRACK_EDGE_W = 0.35
CRACK_CONNECTEDNESS_W = 1.05
CRACK_LARGEST_AREA_W = 0.85
CRACK_ELONGATION_W = 0.25

MIN_CLUSTER_DARKNESS_DELTA = 0.025
MIN_CLUSTER_SCORE = 1.35
MAX_SELECTED_CLUSTERS = 2

# ---------- darkness gating ----------
PIXEL_DARKNESS_PERCENTILE = 58
MIN_PIXEL_DARKNESS_DELTA = 0.03
MIN_DARK_MASK_KEEP_RATIO = 0.35

# ---------- candidate enhancement ----------
LINE_ENHANCE_PERCENTILE = 35
SPATIAL_VOTE_KERNEL = 7
SPATIAL_VOTE_RATIO = 0.18

# ---------- postprocessing ----------
CLOSE_RADIUS_1 = 2
CLOSE_RADIUS_2 = 3
OPEN_RADIUS = 1
BRIDGE_DILATE_RADIUS = 1

MIN_COMPONENT_RATIO = 0.008
KEEP_LARGEST_COMPONENTS = 8
MIN_COMPONENT_AREA = 18
MIN_SKELETON_LENGTH = 18
MAX_MEAN_WIDTH = 32.0
WIDTH_RECOVERY_RADIUS = 2
WIDTH_RECOVERY_STEPS = 4
WIDTH_RECOVERY_SUPPORT_PERCENTILE = 82

# Remove block-like false positives: compact blobs usually have low elongation.
ENABLE_BLOB_SHAPE_FILTER = True
MIN_COMPONENT_ECCENTRICITY = 0.74
MIN_MAJOR_MINOR_RATIO = 1.45

# Suppress very regular mortar/tile seams. These are usually long, nearly
# horizontal/vertical, straight, and span much of the image.
ENABLE_SEAM_FILTER = True
SEAM_SPAN_RATIO = 0.62
SEAM_STRAIGHTNESS_RATIO = 0.82


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


def multi_scale_blackhat(gray_f):
    gray_u8 = (gray_f * 255).astype(np.uint8)

    kernels = [
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_KERNEL_SMALL, BLACKHAT_KERNEL_SMALL)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_KERNEL_MEDIUM, BLACKHAT_KERNEL_MEDIUM)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_KERNEL_LARGE, BLACKHAT_KERNEL_LARGE)),
    ]

    return [
        cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32) / 255.0
        for kernel in kernels
    ]


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

    # Thin-crack channel follows the paper's structural cue.
    thinness = frangi(gray_f, sigmas=FRANGI_SIGMAS, black_ridges=True)
    blackhat_small, blackhat_medium, blackhat_large = multi_scale_blackhat(gray_f)

    rel_gray_small = local_avg_small - gray_f
    rel_gray_medium = local_avg_medium - gray_f
    rel_gray_large = local_avg_large - gray_f
    contrast = np.abs(gray_f - local_avg_small)
    valley = np.maximum(local_avg_large - gray_f, 0.0)

    darkness = 1.0 - gray_f
    darkness_local = np.maximum(local_avg_large - gray_f, 0.0)
    dark_prior = 0.65 * normalize01(darkness) + 0.35 * normalize01(darkness_local)
    dark_prior = normalize01(dark_prior)

    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x_coord = xx.astype(np.float32) / max(w - 1, 1)
    y_coord = yy.astype(np.float32) / max(h - 1, 1)

    gray_n = normalize01(gray_f)
    local_contrast_n = normalize01(contrast)
    edge_sharpness_n = normalize01(grad_mag)
    thinness_n = normalize01(thinness)
    local_avg_small_n = normalize01(local_avg_small)
    local_avg_medium_n = normalize01(local_avg_medium)
    local_avg_large_n = normalize01(local_avg_large)
    rel_gray_small_n = normalize01(rel_gray_small)
    rel_gray_medium_n = normalize01(rel_gray_medium)
    rel_gray_large_n = normalize01(rel_gray_large)
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


def get_som_width():
    return SOM_WIDTH_BY_SURFACE.get(SURFACE_TYPE, 5)


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
    weights = som.get_weights().reshape(-1, x.shape[1]).astype(np.float32)
    labels = np.empty((x.shape[0],), dtype=np.int32)
    chunk_size = 250000

    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        chunk = x[start:end]
        # Distance to each 1xN SOM codebook vector. This replaces millions of
        # Python-level som.winner calls with a small vectorized operation.
        dist = ((chunk[:, None, :] - weights[None, :, :]) ** 2).sum(axis=2)
        labels[start:end] = np.argmin(dist, axis=1)

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
        if major <= 1e-6:
            continue
        best = max(best, min((major / max(minor, 1e-6)) / 10.0, 1.0))
    return best


def select_crack_clusters(label_map, aux):
    if CRACK_LABEL_MODE == "manual":
        labels = np.unique(label_map)
        if int(MANUAL_CRACK_LABEL) in labels:
            return [int(MANUAL_CRACK_LABEL)]

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
        mask = label_map == label_id
        if mask.sum() == 0:
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
        )

        is_dark_enough = dark_score >= (global_darkness + MIN_CLUSTER_DARKNESS_DELTA)
        scored.append({
            "label_id": int(label_id),
            "score": float(score),
            "is_dark_enough": bool(is_dark_enough),
        })

    if not scored:
        return []

    scored.sort(key=lambda x: x["score"], reverse=True)
    selected = [
        item["label_id"]
        for item in scored
        if item["is_dark_enough"] and item["score"] >= MIN_CLUSTER_SCORE
    ][:MAX_SELECTED_CLUSTERS]

    if selected:
        return selected

    for item in scored:
        if item["is_dark_enough"]:
            return [item["label_id"]]
    return [scored[0]["label_id"]]


def build_candidate_mask(label_map, crack_labels):
    if crack_labels is None or len(crack_labels) == 0:
        return np.zeros_like(label_map, dtype=bool)
    mask = np.zeros_like(label_map, dtype=bool)
    for crack_label in crack_labels:
        mask |= label_map == crack_label
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
        relaxed_percentile = max(50, PIXEL_DARKNESS_PERCENTILE - 10)
        relaxed_p_thr = float(np.percentile(cluster_dark, relaxed_percentile))
        relaxed_thr = max(relaxed_p_thr, global_darkness + 0.5 * MIN_PIXEL_DARKNESS_DELTA)
        refined = np.logical_and(candidate_mask, darkness >= relaxed_thr)

    return refined


def build_support_map(aux):
    return normalize01(
        0.10 * aux["thinness"]
        + 0.20 * aux["blackhat"]
        + 0.20 * aux["blackhat_large"]
        + 0.20 * aux["valley"]
        + 0.30 * aux["dark_prior"]
    )


def apply_line_enhance_gate(mask, aux):
    if mask.sum() == 0:
        return mask

    support_map = build_support_map(aux)
    vals = support_map[mask]
    thr = float(np.percentile(vals, LINE_ENHANCE_PERCENTILE))
    refined = np.logical_and(mask, support_map >= thr)

    if refined.sum() < max(1, int(mask.sum() * 0.30)):
        return mask
    return refined


def apply_spatial_vote(mask):
    if mask.sum() == 0:
        return mask

    kernel = np.ones((SPATIAL_VOTE_KERNEL, SPATIAL_VOTE_KERNEL), np.float32)
    density = cv2.filter2D(mask.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT)
    density /= float(SPATIAL_VOTE_KERNEL * SPATIAL_VOTE_KERNEL)
    return density >= SPATIAL_VOTE_RATIO


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


def is_regular_seam(region, image_shape):
    if not ENABLE_SEAM_FILTER:
        return False

    h, w = image_shape
    minr, minc, maxr, maxc = region.bbox
    bbox_h = maxr - minr
    bbox_w = maxc - minc
    span_x = bbox_w / float(max(w, 1))
    span_y = bbox_h / float(max(h, 1))

    major = float(region.major_axis_length)
    if major <= 1e-6:
        return False

    coords = region.coords.astype(np.float32)
    yy = coords[:, 0]
    xx = coords[:, 1]
    horizontal_thin = bbox_h <= max(8, int(0.025 * h))
    vertical_thin = bbox_w <= max(8, int(0.025 * w))

    if span_x >= SEAM_SPAN_RATIO and horizontal_thin:
        straightness = bbox_w / max(major, 1e-6)
        y_std = float(np.std(yy)) / max(bbox_h, 1)
        return straightness >= SEAM_STRAIGHTNESS_RATIO and y_std < 0.32

    if span_y >= SEAM_SPAN_RATIO and vertical_thin:
        straightness = bbox_h / max(major, 1e-6)
        x_std = float(np.std(xx)) / max(bbox_w, 1)
        return straightness >= SEAM_STRAIGHTNESS_RATIO and x_std < 0.32

    return False


def filter_components(mask_bool, aux):
    labeled = label(mask_bool)
    if labeled.max() == 0:
        return mask_bool

    areas = np.bincount(labeled.ravel())[1:]
    largest_area = int(areas.max())
    min_keep_area = max(1, int(np.ceil(MIN_COMPONENT_RATIO * largest_area)))

    valid_regions = []
    support_map = build_support_map(aux)
    darkness_map = 1.0 - aux["gray_intensity"]

    for region in regionprops(labeled):
        area = int(region.area)
        if area < min_keep_area:
            continue
        if area < MIN_COMPONENT_AREA:
            continue
        if is_regular_seam(region, mask_bool.shape):
            continue

        major = float(region.major_axis_length)
        minor = float(region.minor_axis_length)
        ecc = float(region.eccentricity)
        if major <= 1e-6:
            continue

        major_minor_ratio = major / max(minor, 1e-6)
        if ENABLE_BLOB_SHAPE_FILTER:
            if major_minor_ratio < MIN_MAJOR_MINOR_RATIO and ecc < MIN_COMPONENT_ECCENTRICITY:
                continue

        region_mask = labeled == region.label
        region_skel = skeletonize(region_mask)
        skeleton_len = int(region_skel.sum())
        if skeleton_len < MIN_SKELETON_LENGTH:
            continue

        mean_width = area / float(max(skeleton_len, 1))
        if mean_width > MAX_MEAN_WIDTH:
            continue

        rr, cc = region.coords[:, 0], region.coords[:, 1]
        support_mean = float(support_map[rr, cc].mean())
        dark_mean = float(darkness_map[rr, cc].mean())
        area_score = min(area / float(max(largest_area, 1)), 1.0)
        elong_score = min(major_minor_ratio / 10.0, 1.0)
        width_score = max(0.0, 1.0 - mean_width / MAX_MEAN_WIDTH)
        score = (
            1.15 * support_mean
            + 0.85 * dark_mean
            + 0.42 * area_score
            + 0.30 * elong_score
            + 0.35 * width_score
        )

        valid_regions.append((int(region.label), score, area))

    if not valid_regions:
        return np.zeros_like(mask_bool, dtype=bool)

    valid_regions.sort(key=lambda x: (x[1], x[2]), reverse=True)
    keep_k = max(1, int(KEEP_LARGEST_COMPONENTS))
    selected_ids = {rid for rid, _, _ in valid_regions[:keep_k]}

    kept = np.zeros_like(mask_bool, dtype=bool)
    for rid in selected_ids:
        kept[labeled == rid] = True

    return kept


def recover_crack_width(mask_bool, aux):
    if mask_bool.sum() == 0:
        return mask_bool

    support_map = build_support_map(aux)
    darkness_map = 1.0 - aux["gray_intensity"]

    support_thr = float(np.percentile(support_map, WIDTH_RECOVERY_SUPPORT_PERCENTILE))
    dark_thr = float(darkness_map.mean() + MIN_PIXEL_DARKNESS_DELTA)
    grow_candidate = np.logical_or(
        support_map >= support_thr,
        np.logical_and(darkness_map >= dark_thr, aux["valley"] >= np.percentile(aux["valley"], 65)),
    )

    current = mask_bool.copy()
    kernel = disk(WIDTH_RECOVERY_RADIUS)
    for _ in range(WIDTH_RECOVERY_STEPS):
        expanded = binary_dilation(current, kernel)
        expanded = np.logical_and(expanded, grow_candidate)
        expanded = np.logical_or(expanded, mask_bool)
        if np.array_equal(expanded, current):
            break
        current = expanded

    return filter_components(current, aux)


def make_overlay(image_bgr, mask_bool):
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[mask_bool] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 1.0, color, 0.45, 0)
    return overlay


def segment_image_bgr_core(image_bgr):
    feats, aux = build_features(image_bgr)
    x, h, w = flatten_features(feats)
    som_w = get_som_width()

    # Crack pixels are usually a minority. Sampling darker, valley-like pixels
    # a little more often prevents large clean background regions from
    # dominating the SOM codebook.
    sample_weight = 0.25 + 0.75 * aux["dark_prior"].reshape(-1)
    som = train_som(x, som_w, sample_weight=sample_weight)
    labels = som_predict_labels(som, x, som_w).reshape(h, w)

    crack_labels = select_crack_clusters(labels, aux)
    candidate_mask = build_candidate_mask(labels, crack_labels)
    candidate_mask = apply_darkness_gate(candidate_mask, aux["gray_intensity"])
    candidate_mask = apply_line_enhance_gate(candidate_mask, aux)
    candidate_mask = apply_spatial_vote(candidate_mask)

    candidate_bool = postprocess_candidate(candidate_mask)
    final_bool = filter_components(candidate_bool, aux)
    final_bool = recover_crack_width(final_bool, aux)
    final_mask_u8 = (final_bool.astype(np.uint8) * 255)

    overlay = make_overlay(image_bgr, final_bool)
    return final_bool, final_mask_u8, overlay


def segment_image_bgr(image_bgr):
    original_h, original_w = image_bgr.shape[:2]
    max_side = max(original_h, original_w)

    if max_side <= MAX_PROCESS_SIDE:
        return segment_image_bgr_core(image_bgr)

    scale = MAX_PROCESS_SIDE / float(max_side)
    work_w = max(1, int(round(original_w * scale)))
    work_h = max(1, int(round(original_h * scale)))
    work_image = cv2.resize(image_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)

    work_bool, _, _ = segment_image_bgr_core(work_image)
    mask_u8 = (work_bool.astype(np.uint8) * 255)
    mask_u8 = cv2.resize(mask_u8, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    final_bool = mask_u8 > 0
    overlay = make_overlay(image_bgr, final_bool)
    return final_bool, mask_u8, overlay


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
