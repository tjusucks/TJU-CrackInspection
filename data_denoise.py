import cv2
from pathlib import Path

SRC_DIR = Path("~/crack_datasets/images").expanduser()
DST_DIR = Path("~/crack_datasets/images_denoise").expanduser()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

D = 5
SIGMA_COLOR = 30
SIGMA_SPACE = 30


def denoise_image(img):
    return cv2.bilateralFilter(img, d=D, sigmaColor=SIGMA_COLOR, sigmaSpace=SIGMA_SPACE)


def main():
    if not SRC_DIR.exists():
        print(f"Source directory does not exist: {SRC_DIR}")
        return

    DST_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    success = 0
    failed = 0

    for file_path in SRC_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in IMAGE_EXTS:
            continue

        total += 1

        relative_path = file_path.relative_to(SRC_DIR)
        save_path = DST_DIR / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(file_path))
        if img is None:
            print(f"Failed to read: {file_path}")
            failed += 1
            continue

        try:
            denoised = denoise_image(img)
            ok = cv2.imwrite(str(save_path), denoised)
            if ok:
                success += 1
            else:
                print(f"Failed to save: {save_path}")
                failed += 1
        except Exception as e:
            print(f"Processing error: {file_path} | Error: {e}")
            failed += 1

    print("Done")
    print(f"Total images: {total}")
    print(f"Successfully processed: {success}")
    print(f"Failed: {failed}")
    print(f"Output directory: {DST_DIR}")


if __name__ == "__main__":
    main()