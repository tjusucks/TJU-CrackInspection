from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi
from skimage.morphology import (
    skeletonize,
    remove_small_objects,
    binary_closing,
    binary_dilation,
    disk,
)
from skimage.measure import label, regionprops


INPUT_DIR = Path("~/crack_datasets/images_denoise").expanduser()
MASK_DIR = Path("~/crack_datasets/unsup_crack_masks_v2").expanduser()
OVERLAY_DIR = Path("~/crack_datasets/unsup_crack_overlays_v2").expanduser()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------- preprocessing ----------
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
BG_BLUR_SIZE = 41

# ---------- response fusion ----------
BLACKHAT_KERNEL = 15
FRANGI_SIGMAS = range(1, 5)

# ---------- dual-threshold seed / candidate ----------
STRONG_SEED_PERCENTILE = 96
WEAK_CANDIDATE_PERCENTILE = 82
MIN_STRONG_SEED_OBJECT = 12
MIN_WEAK_CANDIDATE_OBJECT = 20

# ---------- coarse completion ----------
COARSE_CLOSE_RADIUS = 2

# ---------- final geometry filtering ----------
MIN_AREA = 35
MIN_SKELETON_LENGTH = 25
MIN_ASPECT_RATIO = 1.8
MAX_MEAN_WIDTH = 35.0

# ---------- width recovery ----------
REGION_GROW_INTENSITY_MARGIN = 24


def read_image(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def normalize_gray(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    gray_eq = clahe.apply(gray)

    bg = cv2.GaussianBlur(gray_eq, (BG_BLUR_SIZE, BG_BLUR_SIZE), 0)
    corrected = cv2.normalize(
        cv2.subtract(bg, gray_eq),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return gray, gray_eq, corrected


def build_crack_response(gray_eq, corrected):
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BLACKHAT_KERNEL, BLACKHAT_KERNEL)
    )
    blackhat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, blackhat_kernel)

    corrected_f = corrected.astype(np.float32) / 255.0
    frangi_resp = frangi(corrected_f, sigmas=FRANGI_SIGMAS, black_ridges=True)
    frangi_resp = cv2.normalize(frangi_resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blackhat_f = blackhat.astype(np.float32) / 255.0
    frangi_f = frangi_resp.astype(np.float32) / 255.0
    corrected_f2 = corrected.astype(np.float32) / 255.0

    response = 0.35 * blackhat_f + 0.45 * frangi_f + 0.20 * corrected_f2
    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return response, blackhat, frangi_resp


def extract_seed_and_candidate(response):
    strong_thr = np.percentile(response, STRONG_SEED_PERCENTILE)
    weak_thr = np.percentile(response, WEAK_CANDIDATE_PERCENTILE)

    strong_seed = response >= strong_thr
    weak_candidate = response >= weak_thr

    strong_seed = remove_small_objects(strong_seed, min_size=MIN_STRONG_SEED_OBJECT)
    weak_candidate = remove_small_objects(weak_candidate, min_size=MIN_WEAK_CANDIDATE_OBJECT)

    return strong_seed, weak_candidate


def filter_seed_components(seed):
    labeled = label(seed)
    kept = np.zeros_like(seed, dtype=bool)

    for region in regionprops(labeled):
        area = region.area
        minr, minc, maxr, maxc = region.bbox
        h = maxr - minr
        w = maxc - minc

        long_side = max(h, w)
        short_side = max(1, min(h, w))
        aspect_ratio = long_side / short_side

        if area < 15:
            continue
        if aspect_ratio < 1.3:
            continue

        kept[labeled == region.label] = True

    return kept


def geodesic_expand(seed, candidate, max_iter=300):
    current = seed.copy()
    kernel = disk(1)

    for _ in range(max_iter):
        expanded = binary_dilation(current, kernel)
        expanded = np.logical_and(expanded, candidate)

        if np.array_equal(expanded, current):
            break

        current = expanded

    return current


def extract_main_skeleton(mask):
    return skeletonize(mask)


def filter_by_skeleton(mask, skeleton):
    labeled = label(mask)
    kept = np.zeros_like(mask, dtype=bool)

    for region in regionprops(labeled):
        region_mask = labeled == region.label
        region_skel = np.logical_and(skeleton, region_mask)

        skel_len = int(region_skel.sum())
        area = int(region_mask.sum())

        minr, minc, maxr, maxc = region.bbox
        h = maxr - minr
        w = maxc - minc
        long_side = max(h, w)
        short_side = max(1, min(h, w))
        aspect_ratio = long_side / short_side
        mean_width = area / max(1, skel_len)

        if area < MIN_AREA:
            continue
        if skel_len < MIN_SKELETON_LENGTH:
            continue
        if aspect_ratio < MIN_ASPECT_RATIO:
            continue
        if mean_width > MAX_MEAN_WIDTH:
            continue

        kept[region_mask] = True

    return kept


def recover_width_from_skeleton(gray, mask, skeleton):
    distance = ndi.distance_transform_edt(mask)
    skeleton_dist = distance[skeleton]

    if skeleton_dist.size == 0:
        return mask

    median_half_width = np.median(skeleton_dist)
    radius = max(1, int(round(median_half_width)))

    grown = binary_dilation(skeleton, disk(radius))

    local_values = gray[mask]
    if local_values.size == 0:
        return mask

    local_dark_threshold = np.median(local_values) + REGION_GROW_INTENSITY_MARGIN
    local_dark = gray < local_dark_threshold

    refined = np.logical_and(grown, local_dark)
    refined = binary_closing(refined, disk(1))
    refined = remove_small_objects(refined, min_size=MIN_AREA)

    return refined


def make_overlay(image_bgr, mask):
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[mask] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 1.0, color, 0.45, 0)
    return overlay


def process_one(image_path: Path):
    image_bgr = read_image(image_path)
    if image_bgr is None:
        print(f"Failed to read: {image_path}")
        return False

    gray, gray_eq, corrected = normalize_gray(image_bgr)
    response, blackhat, frangi_resp = build_crack_response(gray_eq, corrected)

    strong_seed, weak_candidate = extract_seed_and_candidate(response)
    strong_seed = filter_seed_components(strong_seed)

    coarse = geodesic_expand(strong_seed, weak_candidate)
    coarse = binary_closing(coarse, disk(COARSE_CLOSE_RADIUS))
    coarse = remove_small_objects(coarse, min_size=MIN_AREA)

    skeleton_1 = extract_main_skeleton(coarse)
    filtered = filter_by_skeleton(coarse, skeleton_1)

    skeleton_2 = extract_main_skeleton(filtered)
    final_mask = recover_width_from_skeleton(gray, filtered, skeleton_2)
    final_mask = binary_closing(final_mask, disk(1))
    final_mask = remove_small_objects(final_mask, min_size=MIN_AREA)

    mask_uint8 = (final_mask.astype(np.uint8) * 255)
    overlay = make_overlay(image_bgr, final_mask)

    relative_name = image_path.stem + ".png"
    cv2.imwrite(str(MASK_DIR / relative_name), mask_uint8)
    cv2.imwrite(str(OVERLAY_DIR / relative_name), overlay)

    return True


def main():
    if not INPUT_DIR.exists():
        print(f"Input directory does not exist: {INPUT_DIR}")
        return

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

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
    print(f"Mask output: {MASK_DIR}")
    print(f"Overlay output: {OVERLAY_DIR}")


if __name__ == "__main__":
    main()