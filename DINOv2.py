from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, Dinov2Model


SRC_DIR = Path("~/crack_datasets/images_denoise").expanduser()
DST_DIR = Path("~/crack_datasets/dinov2_patch_features").expanduser()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

MODEL_NAME = "facebook/dinov2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image_rgb(image_path: Path):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_patch_features(processor, model, image_rgb: np.ndarray):
    inputs = processor(images=image_rgb, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state.squeeze(0)

    # remove CLS token
    patch_tokens = hidden[1:, :]

    num_patches = patch_tokens.shape[0]
    feat_dim = patch_tokens.shape[1]

    grid_size = int(num_patches ** 0.5)
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Unexpected patch count: {num_patches}")

    patch_grid = patch_tokens.reshape(grid_size, grid_size, feat_dim)
    return patch_grid.cpu().numpy().astype(np.float32)


def main():
    if not SRC_DIR.exists():
        print(f"Source directory does not exist: {SRC_DIR}")
        return

    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = Dinov2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    total = 0
    success = 0
    failed = 0

    for image_path in SRC_DIR.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        total += 1

        relative_path = image_path.relative_to(SRC_DIR)
        save_path = (DST_DIR / relative_path).with_suffix(".npy")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        image_rgb = load_image_rgb(image_path)
        if image_rgb is None:
            print(f"Failed to read: {image_path}")
            failed += 1
            continue

        try:
            patch_grid = extract_patch_features(processor, model, image_rgb)
            np.save(save_path, patch_grid)
            success += 1
        except Exception as e:
            print(f"Processing error: {image_path} | Error: {e}")
            failed += 1

    print("Done")
    print(f"Device: {DEVICE}")
    print(f"Total images: {total}")
    print(f"Successfully processed: {success}")
    print(f"Failed: {failed}")
    print(f"Output directory: {DST_DIR}")


if __name__ == "__main__":
    main()