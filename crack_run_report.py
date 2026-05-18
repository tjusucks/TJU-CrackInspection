from __future__ import annotations

import json
from pathlib import Path


def _safe_ratio(num, den):
    return float(num) / float(den) if den else 0.0


def _pct(value):
    return f"{value * 100:.2f}%"


def summarize_supervised_run(
    summaries,
    failures,
    total,
    classifier_model=None,
    classifier_threshold=None,
):
    detected = [item for item in summaries if item.get("crack_pixels", 0) > 0]
    raw_pixels = sum(int(item.get("raw_crack_pixels", 0)) for item in summaries)
    kept_pixels = sum(int(item.get("crack_pixels", 0)) for item in summaries)
    raw_components = sum(int(item.get("component_count", 0)) for item in summaries)
    kept_components = sum(int(item.get("kept_component_count", 0)) for item in summaries)

    return {
        "total_images": int(total),
        "processed_images": int(len(summaries)),
        "failed_images": int(len(failures)),
        "processing_success_rate": _safe_ratio(len(summaries), total),
        "images_with_crack": int(len(detected)),
        "crack_detection_rate": _safe_ratio(len(detected), len(summaries)),
        "raw_crack_pixels": int(raw_pixels),
        "kept_crack_pixels": int(kept_pixels),
        "removed_crack_pixels": int(raw_pixels - kept_pixels),
        "removed_pixel_rate": _safe_ratio(raw_pixels - kept_pixels, raw_pixels),
        "raw_component_count": int(raw_components),
        "kept_component_count": int(kept_components),
        "kept_component_rate": _safe_ratio(kept_components, raw_components),
        "classifier_model": None if classifier_model is None else str(classifier_model),
        "classifier_threshold": classifier_threshold,
    }


def summarize_wall_block_run(
    summaries,
    failures,
    total,
    classifier_model=None,
    classifier_threshold=None,
):
    images_with_crack = [item for item in summaries if item.get("have_crack")]
    fallback_count = sum(1 for item in summaries if item.get("used_original_fallback"))
    total_blocks = sum(int(item.get("block_count", 0)) for item in summaries)
    crack_blocks = sum(int(item.get("crack_block_count", 0)) for item in summaries)

    raw_pixels = 0
    kept_pixels = 0
    raw_components = 0
    kept_components = 0
    for summary in summaries:
        for block in summary.get("blocks", []):
            raw_pixels += int(block.get("raw_crack_pixels", 0))
            kept_pixels += int(block.get("crack_pixels", 0))
            predictions = block.get("component_predictions", [])
            raw_components += len(predictions)
            kept_components += sum(1 for item in predictions if item.get("kept"))

    return {
        "total_images": int(total),
        "processed_images": int(len(summaries)),
        "failed_images": int(len(failures)),
        "processing_success_rate": _safe_ratio(len(summaries), total),
        "images_with_crack": int(len(images_with_crack)),
        "crack_detection_rate": _safe_ratio(len(images_with_crack), len(summaries)),
        "fallback_original_images": int(fallback_count),
        "fallback_original_rate": _safe_ratio(fallback_count, len(summaries)),
        "total_blocks": int(total_blocks),
        "crack_blocks": int(crack_blocks),
        "crack_block_rate": _safe_ratio(crack_blocks, total_blocks),
        "raw_crack_pixels": int(raw_pixels),
        "kept_crack_pixels": int(kept_pixels),
        "removed_crack_pixels": int(raw_pixels - kept_pixels),
        "removed_pixel_rate": _safe_ratio(raw_pixels - kept_pixels, raw_pixels),
        "raw_component_count": int(raw_components),
        "kept_component_count": int(kept_components),
        "kept_component_rate": _safe_ratio(kept_components, raw_components),
        "classifier_model": None if classifier_model is None else str(classifier_model),
        "classifier_threshold": classifier_threshold,
    }


def write_run_report(output_dir: str | Path, report):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "run_report.json"
    txt_path = output_dir / "run_report.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Crack Detection Run Report",
        "",
        f"Total images: {report['total_images']}",
        f"Processed images: {report['processed_images']}",
        f"Failed images: {report['failed_images']}",
        f"Processing success rate: {_pct(report['processing_success_rate'])}",
        f"Images with crack: {report['images_with_crack']}",
        f"Crack detection rate: {_pct(report['crack_detection_rate'])}",
    ]

    if "fallback_original_images" in report:
        lines.extend(
            [
                f"Fallback original images: {report['fallback_original_images']}",
                f"Fallback original rate: {_pct(report['fallback_original_rate'])}",
                f"Total blocks: {report['total_blocks']}",
                f"Crack blocks: {report['crack_blocks']}",
                f"Crack block rate: {_pct(report['crack_block_rate'])}",
            ]
        )

    lines.extend(
        [
            f"Raw crack pixels: {report['raw_crack_pixels']}",
            f"Kept crack pixels: {report['kept_crack_pixels']}",
            f"Removed crack pixels: {report['removed_crack_pixels']}",
            f"Removed pixel rate: {_pct(report['removed_pixel_rate'])}",
            f"Raw component count: {report['raw_component_count']}",
            f"Kept component count: {report['kept_component_count']}",
            f"Kept component rate: {_pct(report['kept_component_rate'])}",
            f"Classifier model: {report.get('classifier_model')}",
            f"Classifier threshold: {report.get('classifier_threshold')}",
        ]
    )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, txt_path
