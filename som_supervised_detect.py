from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from crack_run_report import summarize_supervised_run, write_run_report
from som_component_classifier import filter_mask_with_classifier, load_classifier_bundle
from som_crack_seg import IMAGE_EXTS, make_overlay, read_image, segment_image_bgr


DEFAULT_MODEL = Path("~/som_component_annotation/som_crack_classifier.joblib").expanduser()
DEFAULT_OUTPUT_DIR = Path("~/tests/som_supervised_results").expanduser()


def iter_images(input_path: Path):
    input_path = input_path.expanduser()
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTS:
            yield input_path
        return

    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def process_one(image_path: Path, output_dir: Path, classifier_bundle, threshold):
    image_bgr = read_image(image_path)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    raw_mask_bool, _, _ = segment_image_bgr(image_bgr)
    filtered_mask_bool, component_predictions = filter_mask_with_classifier(
        image_bgr,
        raw_mask_bool,
        classifier_bundle,
        threshold=threshold,
    )

    stem = image_path.stem
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    raw_mask_dir = output_dir / "raw_som_masks"
    raw_overlay_dir = output_dir / "raw_som_overlays"
    for directory in (mask_dir, overlay_dir, raw_mask_dir, raw_overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw_mask_u8 = raw_mask_bool.astype("uint8") * 255
    mask_u8 = filtered_mask_bool.astype("uint8") * 255
    raw_overlay = make_overlay(image_bgr, raw_mask_bool)
    overlay = make_overlay(image_bgr, filtered_mask_bool)

    raw_mask_path = raw_mask_dir / f"{stem}_raw_som_mask.png"
    raw_overlay_path = raw_overlay_dir / f"{stem}_raw_som_overlay.png"
    mask_path = mask_dir / f"{stem}_mask.png"
    overlay_path = overlay_dir / f"{stem}_overlay.png"

    cv2.imwrite(str(raw_mask_path), raw_mask_u8)
    cv2.imwrite(str(raw_overlay_path), raw_overlay)
    cv2.imwrite(str(mask_path), mask_u8)
    cv2.imwrite(str(overlay_path), overlay)

    h, w = filtered_mask_bool.shape
    crack_pixels = int(filtered_mask_bool.sum())
    raw_crack_pixels = int(raw_mask_bool.sum())
    kept_components = sum(1 for item in component_predictions if item["kept"])

    return {
        "image_path": str(image_path),
        "raw_mask_path": str(raw_mask_path),
        "raw_overlay_path": str(raw_overlay_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "width": int(w),
        "height": int(h),
        "raw_crack_pixels": raw_crack_pixels,
        "crack_pixels": crack_pixels,
        "removed_pixels": raw_crack_pixels - crack_pixels,
        "crack_area_ratio": crack_pixels / float(max(h * w, 1)),
        "component_count": len(component_predictions),
        "kept_component_count": kept_components,
        "component_predictions": component_predictions,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run SOM crack segmentation with a supervised component filter.")
    parser.add_argument("--input", required=True, help="Input image or image directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--classifier-model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--classifier-threshold",
        type=float,
        default=None,
        help="Override crack probability threshold. Defaults to the threshold saved in the model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier_bundle = load_classifier_bundle(args.classifier_model)
    threshold = (
        float(args.classifier_threshold)
        if args.classifier_threshold is not None
        else float(classifier_bundle.get("threshold", 0.5))
    )

    image_paths = list(iter_images(input_path))
    if not image_paths:
        print(f"No input images found: {input_path}")
        return

    summaries = []
    failures = []
    for image_path in image_paths:
        try:
            summary = process_one(image_path, output_dir, classifier_bundle, threshold)
            summaries.append(summary)
            print(
                f"OK {image_path} | components={summary['component_count']} "
                f"kept={summary['kept_component_count']} "
                f"pixels={summary['raw_crack_pixels']}->{summary['crack_pixels']}"
            )
        except Exception as exc:
            failures.append({"image_path": str(image_path), "error": str(exc)})
            print(f"FAILED {image_path} | {exc}")

    index = {
        "classifier_model": str(Path(args.classifier_model).expanduser()),
        "classifier_threshold": threshold,
        "total": len(image_paths),
        "success": len(summaries),
        "failed": len(failures),
        "summaries": summaries,
        "failures": failures,
    }
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    report = summarize_supervised_run(
        summaries=summaries,
        failures=failures,
        total=len(image_paths),
        classifier_model=Path(args.classifier_model).expanduser(),
        classifier_threshold=threshold,
    )
    report_json_path, report_txt_path = write_run_report(output_dir, report)

    print("Done")
    print(f"Index: {index_path}")
    print(f"Report JSON: {report_json_path}")
    print(f"Report TXT: {report_txt_path}")


if __name__ == "__main__":
    main()
