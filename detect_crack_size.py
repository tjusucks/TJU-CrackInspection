from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from crack_size_measure import (
    draw_measurement_overlay,
    measure_crack_components,
    summarize_measurements,
)
from som_component_classifier import filter_mask_with_classifier, load_classifier_bundle
from som_crack_seg import read_image, segment_image_bgr


DEFAULT_MODEL = Path("~/som_component_annotation/som_crack_classifier.joblib").expanduser()
DEFAULT_OUTPUT_DIR = Path("~/tests/crack_size_results").expanduser()


def write_measurement_csv(csv_path, measurements):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "crack_id",
        "area_px",
        "length_px",
        "avg_width_px",
        "max_width_px",
        "bbox_width_px",
        "bbox_height_px",
        "length_mm",
        "avg_width_mm",
        "max_width_mm",
        "bbox_width_mm",
        "bbox_height_mm",
        "area_mm2",
        "bbox_min_row",
        "bbox_min_col",
        "bbox_max_row",
        "bbox_max_col",
        "orientation_deg",
        "eccentricity",
        "endpoint_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in measurements:
            writer.writerow({name: item.get(name) for name in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect cracks in one image and output crack size measurements."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--classifier-model", default=str(DEFAULT_MODEL))
    parser.add_argument("--classifier-threshold", type=float, default=None)
    parser.add_argument(
        "--mm-per-pixel",
        type=float,
        default=None,
        help="Physical scale. If omitted, size is reported only in pixels.",
    )
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument(
        "--no-classifier",
        action="store_true",
        help="Use raw SOM mask without the supervised component classifier.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = read_image(input_path)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {input_path}")

    raw_mask_bool, _, _ = segment_image_bgr(image_bgr)
    component_predictions = []
    classifier_model = None
    threshold = None
    if args.no_classifier:
        final_mask_bool = raw_mask_bool
    else:
        classifier_model = Path(args.classifier_model).expanduser()
        classifier_bundle = load_classifier_bundle(classifier_model)
        threshold = (
            float(args.classifier_threshold)
            if args.classifier_threshold is not None
            else float(classifier_bundle.get("threshold", 0.5))
        )
        final_mask_bool, component_predictions = filter_mask_with_classifier(
            image_bgr,
            raw_mask_bool,
            classifier_bundle,
            threshold=threshold,
        )

    measurements = measure_crack_components(
        final_mask_bool,
        mm_per_pixel=args.mm_per_pixel,
        min_area=args.min_area,
    )
    summary = summarize_measurements(
        measurements,
        image_bgr.shape,
        mm_per_pixel=args.mm_per_pixel,
    )

    stem = input_path.stem
    mask_path = output_dir / f"{stem}_mask.png"
    raw_mask_path = output_dir / f"{stem}_raw_som_mask.png"
    overlay_path = output_dir / f"{stem}_size_overlay.png"
    json_path = output_dir / f"{stem}_crack_sizes.json"
    csv_path = output_dir / f"{stem}_crack_sizes.csv"

    cv2.imwrite(str(raw_mask_path), raw_mask_bool.astype("uint8") * 255)
    cv2.imwrite(str(mask_path), final_mask_bool.astype("uint8") * 255)
    overlay = draw_measurement_overlay(
        image_bgr,
        final_mask_bool,
        measurements,
        mm_per_pixel=args.mm_per_pixel,
    )
    cv2.imwrite(str(overlay_path), overlay)

    result = {
        "input_image": str(input_path),
        "classifier_model": None if classifier_model is None else str(classifier_model),
        "classifier_threshold": threshold,
        "raw_crack_pixels": int(raw_mask_bool.sum()),
        "kept_crack_pixels": int(final_mask_bool.sum()),
        "mask_path": str(mask_path),
        "raw_mask_path": str(raw_mask_path),
        "overlay_path": str(overlay_path),
        "csv_path": str(csv_path),
        "summary": summary,
        "measurements": measurements,
        "component_predictions": component_predictions,
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_measurement_csv(csv_path, measurements)

    print("Done")
    print(f"Input: {input_path}")
    print(f"Has crack: {summary['has_crack']}")
    print(f"Crack count: {summary['crack_count']}")
    print(f"Total length px: {summary['total_length_px']:.2f}")
    print(f"Average width px: {summary['average_width_px']:.2f}")
    print(f"Max width px: {summary['max_width_px']:.2f}")
    if args.mm_per_pixel is not None:
        print(f"Total length mm: {summary['total_length_mm']:.2f}")
        print(f"Average width mm: {summary['average_width_mm']:.2f}")
        print(f"Max width mm: {summary['max_width_mm']:.2f}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()
