from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans


FEATURE_DIR = Path("~/crack_datasets/dinov2_patch_features").expanduser()
IMAGE_DIR = Path("~/crack_datasets/images_denoise").expanduser()
OUTPUT_DIR = Path("~/crack_datasets/kmeans_candidate_maps").expanduser()

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
N_CLUSTERS = 5
RANDOM_STATE = 42


def find_image_path(relative_stem: Path):
    for ext in IMAGE_EXTS:
        candidate = IMAGE_DIR / relative_stem.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def make_cluster_visual(label_map):
    h, w = label_map.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    colors = [
        (0, 0, 0),
        (255, 255, 255),
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
    ]

    unique_labels = np.unique(label_map)
    for label in unique_labels:
        color = colors[int(label) % len(colors)]
        vis[label_map == label] = color

    return vis


def process_one_feature(feature_path: Path):
    relative_path = feature_path.relative_to(FEATURE_DIR)
    relative_stem = relative_path.with_suffix("")

    image_path = find_image_path(relative_stem)
    if image_path is None:
        print(f"Missing image for feature: {feature_path}")
        return False

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return False

    feature = np.load(feature_path)
    if feature.ndim != 3:
        print(f"Unexpected feature shape: {feature.shape} | {feature_path}")
        return False

    ph, pw, c = feature.shape
    x = feature.reshape(-1, c)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(x)
    label_map = labels.reshape(ph, pw).astype(np.uint8)

    image_h, image_w = image.shape[:2]
    label_map_up = cv2.resize(
        label_map,
        (image_w, image_h),
        interpolation=cv2.INTER_NEAREST
    )

    cluster_vis_small = make_cluster_visual(label_map)
    cluster_vis_up = make_cluster_visual(label_map_up)

    overlay = cv2.addWeighted(image, 0.7, cluster_vis_up, 0.3, 0)

    save_base = OUTPUT_DIR / relative_stem
    save_base.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(save_base) + "_labels.npy", label_map)
    np.save(str(save_base) + "_labels_up.npy", label_map_up)

    cv2.imwrite(str(save_base) + "_cluster_small.png", cluster_vis_small)
    cv2.imwrite(str(save_base) + "_cluster_up.png", cluster_vis_up)
    cv2.imwrite(str(save_base) + "_overlay.png", overlay)

    return True


def main():
    if not FEATURE_DIR.exists():
        print(f"Feature directory does not exist: {FEATURE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_files = sorted(FEATURE_DIR.rglob("*.npy"))

    total = 0
    success = 0
    failed = 0

    for feature_path in feature_files:
        total += 1
        try:
            ok = process_one_feature(feature_path)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Processing error: {feature_path} | Error: {e}")
            failed += 1

    print("Done")
    print(f"Total feature files: {total}")
    print(f"Successfully processed: {success}")
    print(f"Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()