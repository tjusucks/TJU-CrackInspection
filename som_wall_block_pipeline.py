from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from crack_run_report import summarize_wall_block_run, write_run_report
from som_component_classifier import filter_mask_with_classifier, load_classifier_bundle
from som_crack_seg import IMAGE_EXTS, make_overlay, read_image, segment_image_bgr


DEFAULT_BACKEND_DIR = Path("~/CrackDetection_Backend").expanduser()


def import_backend_preprocessors(backend_dir: Path):
    backend_dir = backend_dir.expanduser().resolve()
    if not backend_dir.exists():
        raise FileNotFoundError(f"Backend directory does not exist: {backend_dir}")

    preprocess_dir = backend_dir / "preprocessImages"
    sys.path.insert(0, str(backend_dir))
    sys.path.insert(0, str(preprocess_dir))
    from preprocessImages.utils.detectBoxAndSegBlock import (  # noqa: PLC0415
        detect_boxes_local,
        segment_blocks_local,
    )

    return detect_boxes_local, segment_blocks_local


def iter_images(input_path: Path):
    input_path = input_path.expanduser()
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTS:
            yield input_path
        return

    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def ensure_model(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} model not found: {path}. "
            "Use --yolo-model/--sam-model or run the backend once to download models."
        )


def run_wall_block_cutting(
    image_path: Path,
    output_dir: Path,
    yolo_model: Path,
    sam_model: Path,
    confidence: float,
    detect_boxes_local,
    segment_blocks_local,
):
    stem = image_path.stem
    box_dir = output_dir / "boxes" / stem
    block_dir = output_dir / "blocks" / stem
    box_dir.mkdir(parents=True, exist_ok=True)
    block_dir.mkdir(parents=True, exist_ok=True)

    detect_result = detect_boxes_local(
        str(yolo_model),
        str(image_path),
        str(box_dir),
        confidence=confidence,
    )
    if detect_result.get("boxes_object_path") == "-1":
        return {
            "block_paths": [],
            "segment_overview": None,
            "boxes_path": None,
            "boxes_image": None,
        }

    segment_result = segment_blocks_local(
        str(sam_model),
        str(image_path),
        detect_result["boxes_object_path"],
        str(block_dir),
    )

    return {
        "block_paths": [Path(p) for p in segment_result["wall_block_images"]],
        "segment_overview": segment_result["segment_whole_image_path"],
        "boxes_path": detect_result["boxes_object_path"],
        "boxes_image": detect_result["image_obb_boxes_path"],
    }


def run_som_on_image(
    image_path: Path,
    save_stem: str,
    mask_dir: Path,
    overlay_dir: Path,
    min_crack_pixels: int,
    min_crack_area_ratio: float,
    classifier_bundle=None,
    classifier_threshold: float | None = None,
):
    image_bgr = read_image(image_path)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    mask_bool, _, _ = segment_image_bgr(image_bgr)
    raw_crack_pixels = int(mask_bool.sum())
    component_predictions = []
    if classifier_bundle is not None:
        mask_bool, component_predictions = filter_mask_with_classifier(
            image_bgr,
            mask_bool,
            classifier_bundle,
            threshold=classifier_threshold,
        )

    mask_u8 = (mask_bool.astype("uint8") * 255)
    overlay = make_overlay(image_bgr, mask_bool)
    h, w = mask_bool.shape
    crack_pixels = int(mask_bool.sum())
    crack_area_ratio = crack_pixels / float(max(h * w, 1))
    have_crack = (
        crack_pixels >= min_crack_pixels
        and crack_area_ratio >= min_crack_area_ratio
    )

    mask_path = mask_dir / f"{save_stem}_mask.png"
    overlay_path = overlay_dir / f"{save_stem}_overlay.png"
    cv2.imwrite(str(mask_path), mask_u8)
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "width": int(w),
        "height": int(h),
        "raw_crack_pixels": raw_crack_pixels,
        "crack_pixels": crack_pixels,
        "crack_area_ratio": crack_area_ratio,
        "have_crack": bool(have_crack),
        "classifier_used": classifier_bundle is not None,
        "component_predictions": component_predictions,
    }


def process_one(
    image_path: Path,
    output_dir: Path,
    yolo_model: Path,
    sam_model: Path,
    confidence: float,
    min_crack_pixels: int,
    min_crack_area_ratio: float,
    fallback_original: bool,
    detect_boxes_local,
    segment_blocks_local,
    classifier_bundle=None,
    classifier_threshold: float | None = None,
):
    stem = image_path.stem
    mask_dir = output_dir / "som_masks" / stem
    overlay_dir = output_dir / "som_overlays" / stem
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    cutting = run_wall_block_cutting(
        image_path,
        output_dir,
        yolo_model,
        sam_model,
        confidence,
        detect_boxes_local,
        segment_blocks_local,
    )

    block_paths = cutting["block_paths"]
    used_fallback = False
    if not block_paths and fallback_original:
        block_paths = [image_path]
        used_fallback = True

    block_results = []
    for index, block_path in enumerate(block_paths, start=1):
        save_stem = f"{stem}_block_{index:03d}"
        if used_fallback:
            save_stem = f"{stem}_original"
        result = run_som_on_image(
            block_path,
            save_stem,
            mask_dir,
            overlay_dir,
            min_crack_pixels,
            min_crack_area_ratio,
            classifier_bundle=classifier_bundle,
            classifier_threshold=classifier_threshold,
        )
        result["block_index"] = index
        block_results.append(result)

    have_crack = any(item["have_crack"] for item in block_results)
    crack_block_count = sum(1 for item in block_results if item["have_crack"])

    summary = {
        "source_image": str(image_path),
        "used_original_fallback": used_fallback,
        "have_crack": have_crack,
        "block_count": len(block_results),
        "crack_block_count": crack_block_count,
        "segment_overview": cutting["segment_overview"],
        "boxes_image": cutting["boxes_image"],
        "blocks": block_results,
    }

    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut curtain-wall blocks with YOLO-OBB + MobileSAM, then run SOM crack segmentation per block."
    )
    parser.add_argument("--input", required=True, help="Input image or image directory.")
    parser.add_argument(
        "--output-dir",
        default="~/tests/som_wall_block_results",
        help="Directory for cut blocks, SOM masks, overlays, and summaries.",
    )
    parser.add_argument(
        "--backend-dir",
        default=str(DEFAULT_BACKEND_DIR),
        help="CrackDetection_Backend directory.",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="YOLOv8-OBB wall block model path.",
    )
    parser.add_argument(
        "--sam-model",
        default=None,
        help="MobileSAM model path.",
    )
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--min-crack-pixels", type=int, default=25)
    parser.add_argument("--min-crack-area-ratio", type=float, default=0.0002)
    parser.add_argument(
        "--classifier-model",
        default=None,
        help="Optional supervised crack/non-crack component classifier .joblib.",
    )
    parser.add_argument(
        "--classifier-threshold",
        type=float,
        default=None,
        help="Override classifier crack probability threshold. Defaults to the value saved in the model.",
    )
    parser.add_argument("--no-fallback-original", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    backend_dir = Path(args.backend_dir).expanduser()
    yolo_model = (
        Path(args.yolo_model).expanduser()
        if args.yolo_model
        else backend_dir / "model" / "stonewall-YOLOv8-obb.pt"
    )
    sam_model = (
        Path(args.sam_model).expanduser()
        if args.sam_model
        else backend_dir / "model" / "mobile-sam.pt"
    )

    ensure_model(yolo_model, "YOLO-OBB")
    ensure_model(sam_model, "MobileSAM")
    detect_boxes_local, segment_blocks_local = import_backend_preprocessors(backend_dir)
    classifier_bundle = (
        load_classifier_bundle(args.classifier_model)
        if args.classifier_model
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(iter_images(input_path))
    if not image_paths:
        print(f"No input images found: {input_path}")
        return

    summaries = []
    failed = []
    for image_path in image_paths:
        try:
            summary = process_one(
                image_path=image_path,
                output_dir=output_dir,
                yolo_model=yolo_model,
                sam_model=sam_model,
                confidence=args.confidence,
                min_crack_pixels=args.min_crack_pixels,
                min_crack_area_ratio=args.min_crack_area_ratio,
                fallback_original=not args.no_fallback_original,
                detect_boxes_local=detect_boxes_local,
                segment_blocks_local=segment_blocks_local,
                classifier_bundle=classifier_bundle,
                classifier_threshold=args.classifier_threshold,
            )
            summaries.append(summary)
            print(
                f"OK {image_path} | blocks={summary['block_count']} "
                f"crack_blocks={summary['crack_block_count']}"
            )
        except Exception as exc:
            failed.append({"image_path": str(image_path), "error": str(exc)})
            print(f"FAILED {image_path} | {exc}")

    index = {
        "total": len(image_paths),
        "success": len(summaries),
        "failed": len(failed),
        "summaries": summaries,
        "failures": failed,
    }
    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = summarize_wall_block_run(
        summaries=summaries,
        failures=failed,
        total=len(image_paths),
        classifier_model=Path(args.classifier_model).expanduser()
        if args.classifier_model
        else None,
        classifier_threshold=args.classifier_threshold
        if args.classifier_threshold is not None
        else (
            None
            if classifier_bundle is None
            else float(classifier_bundle.get("threshold", 0.5))
        ),
    )
    report_json_path, report_txt_path = write_run_report(output_dir, report)

    print("Done")
    print(f"Index: {index_path}")
    print(f"Report JSON: {report_json_path}")
    print(f"Report TXT: {report_txt_path}")


if __name__ == "__main__":
    main()
