from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk, skeletonize


FEATURE_NAMES = [
    "area",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "area_ratio_image",
    "bbox_width_ratio",
    "bbox_height_ratio",
    "fill_ratio",
    "aspect_ratio",
    "major_axis_length",
    "minor_axis_length",
    "major_minor_ratio",
    "eccentricity",
    "solidity",
    "extent",
    "orientation",
    "abs_orientation",
    "touches_top",
    "touches_bottom",
    "touches_left",
    "touches_right",
    "border_min_dist_ratio",
    "skeleton_pixels",
    "skeleton_area_ratio",
    "skeleton_endpoint_count",
    "gray_mean",
    "gray_std",
    "gray_min",
    "gray_p05",
    "gray_p50",
    "gray_p95",
    "ring_gray_mean",
    "ring_gray_std",
    "gray_contrast",
    "gray_contrast_ratio",
    "blackhat_mean",
    "blackhat_p95",
    "ring_blackhat_mean",
    "blackhat_contrast",
    "edge_density",
    "ring_edge_density",
    "lap_abs_mean",
    "ring_lap_abs_mean",
    "sat_mean",
    "sat_std",
    "value_mean",
    "value_contrast",
]


def load_classifier_bundle(model_path: str | Path):
    return joblib.load(str(Path(model_path).expanduser()))


def save_classifier_bundle(bundle, model_path: str | Path):
    model_path = Path(model_path).expanduser()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, str(model_path))
    return model_path


def _safe_float(value, default=0.0):
    if value is None:
        return default
    value = float(value)
    if not np.isfinite(value):
        return default
    return value


def _safe_ratio(num, den):
    return float(num) / float(den) if den else 0.0


def _values_stats(values):
    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def build_feature_context(image_bgr):
    gray_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0

    edge = cv2.Canny(gray_u8, 50, 150) > 0
    lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    blackhat_maps = []
    for size in (9, 17, 31):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        blackhat_maps.append(cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel))
    blackhat = np.maximum.reduce(blackhat_maps).astype(np.float32) / 255.0

    return {
        "gray": gray,
        "sat": sat,
        "value": value,
        "edge": edge,
        "lap_abs": lap_abs,
        "blackhat": blackhat,
    }


def _component_region(component_mask, region=None):
    if region is not None:
        return region
    labeled = label(component_mask)
    regions = regionprops(labeled)
    if not regions:
        return None
    return max(regions, key=lambda r: r.area)


def _endpoint_count(skeleton_crop):
    if skeleton_crop.size == 0 or skeleton_crop.sum() == 0:
        return 0
    neighbors = ndi.convolve(
        skeleton_crop.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    return int(np.logical_and(skeleton_crop, neighbors == 2).sum())


def extract_component_features(image_bgr, component_mask, region=None, context=None):
    component_mask = component_mask.astype(bool)
    h, w = image_bgr.shape[:2]
    region = _component_region(component_mask, region)
    if region is None:
        return {name: 0.0 for name in FEATURE_NAMES}

    if context is None:
        context = build_feature_context(image_bgr)

    minr, minc, maxr, maxc = region.bbox
    bbox_h = int(maxr - minr)
    bbox_w = int(maxc - minc)
    bbox_area = int(max(bbox_h * bbox_w, 1))
    image_area = int(max(h * w, 1))
    area = int(region.area)
    major = _safe_float(region.major_axis_length)
    minor = _safe_float(region.minor_axis_length)

    component_crop = component_mask[minr:maxr, minc:maxc]
    skeleton_crop = skeletonize(component_crop)
    skeleton_pixels = int(skeleton_crop.sum())

    ring_mask = binary_dilation(component_mask, disk(6))
    ring_mask = np.logical_and(ring_mask, ~component_mask)
    if ring_mask.sum() == 0:
        ring_mask = ~component_mask

    gray_values = context["gray"][component_mask]
    ring_gray_values = context["gray"][ring_mask]
    blackhat_values = context["blackhat"][component_mask]
    ring_blackhat_values = context["blackhat"][ring_mask]
    value_values = context["value"][component_mask]
    ring_value_values = context["value"][ring_mask]
    sat_values = context["sat"][component_mask]

    gray_stats = _values_stats(gray_values)
    ring_gray_stats = _values_stats(ring_gray_values)
    blackhat_stats = _values_stats(blackhat_values)

    gray_contrast = ring_gray_stats["mean"] - gray_stats["mean"]
    value_contrast = float(np.mean(ring_value_values) - np.mean(value_values)) if value_values.size else 0.0
    blackhat_contrast = (
        float(np.mean(blackhat_values) - np.mean(ring_blackhat_values))
        if blackhat_values.size and ring_blackhat_values.size
        else 0.0
    )

    features = {
        "area": area,
        "bbox_width": bbox_w,
        "bbox_height": bbox_h,
        "bbox_area": bbox_area,
        "area_ratio_image": _safe_ratio(area, image_area),
        "bbox_width_ratio": _safe_ratio(bbox_w, w),
        "bbox_height_ratio": _safe_ratio(bbox_h, h),
        "fill_ratio": _safe_ratio(area, bbox_area),
        "aspect_ratio": max(_safe_ratio(bbox_w, bbox_h), _safe_ratio(bbox_h, bbox_w)),
        "major_axis_length": major,
        "minor_axis_length": minor,
        "major_minor_ratio": _safe_ratio(major, max(minor, 1e-6)),
        "eccentricity": _safe_float(region.eccentricity),
        "solidity": _safe_float(region.solidity),
        "extent": _safe_float(region.extent),
        "orientation": _safe_float(region.orientation),
        "abs_orientation": abs(_safe_float(region.orientation)),
        "touches_top": float(minr <= 0),
        "touches_bottom": float(maxr >= h),
        "touches_left": float(minc <= 0),
        "touches_right": float(maxc >= w),
        "border_min_dist_ratio": _safe_ratio(min(minr, minc, h - maxr, w - maxc), max(min(h, w), 1)),
        "skeleton_pixels": skeleton_pixels,
        "skeleton_area_ratio": _safe_ratio(skeleton_pixels, max(area, 1)),
        "skeleton_endpoint_count": _endpoint_count(skeleton_crop),
        "gray_mean": gray_stats["mean"],
        "gray_std": gray_stats["std"],
        "gray_min": gray_stats["min"],
        "gray_p05": gray_stats["p05"],
        "gray_p50": gray_stats["p50"],
        "gray_p95": gray_stats["p95"],
        "ring_gray_mean": ring_gray_stats["mean"],
        "ring_gray_std": ring_gray_stats["std"],
        "gray_contrast": gray_contrast,
        "gray_contrast_ratio": _safe_ratio(gray_contrast, max(ring_gray_stats["mean"], 1e-6)),
        "blackhat_mean": blackhat_stats["mean"],
        "blackhat_p95": blackhat_stats["p95"],
        "ring_blackhat_mean": float(np.mean(ring_blackhat_values)) if ring_blackhat_values.size else 0.0,
        "blackhat_contrast": blackhat_contrast,
        "edge_density": float(np.mean(context["edge"][component_mask])) if area else 0.0,
        "ring_edge_density": float(np.mean(context["edge"][ring_mask])) if ring_mask.sum() else 0.0,
        "lap_abs_mean": float(np.mean(context["lap_abs"][component_mask])) if area else 0.0,
        "ring_lap_abs_mean": float(np.mean(context["lap_abs"][ring_mask])) if ring_mask.sum() else 0.0,
        "sat_mean": float(np.mean(sat_values)) if sat_values.size else 0.0,
        "sat_std": float(np.std(sat_values)) if sat_values.size else 0.0,
        "value_mean": float(np.mean(value_values)) if value_values.size else 0.0,
        "value_contrast": value_contrast,
    }

    return {name: _safe_float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def features_to_array(feature_dicts, feature_names=FEATURE_NAMES):
    return np.asarray(
        [[_safe_float(item.get(name, 0.0)) for name in feature_names] for item in feature_dicts],
        dtype=np.float32,
    )


def filter_mask_with_classifier(image_bgr, mask_bool, bundle, threshold=None):
    mask_bool = mask_bool.astype(bool)
    if mask_bool.sum() == 0:
        return mask_bool, []

    model = bundle["model"]
    feature_names = bundle.get("feature_names", FEATURE_NAMES)
    threshold = float(bundle.get("threshold", 0.5) if threshold is None else threshold)

    labeled = label(mask_bool)
    regions = regionprops(labeled)
    context = build_feature_context(image_bgr)

    kept_mask = np.zeros_like(mask_bool, dtype=bool)
    component_results = []

    for region in regions:
        component_mask = labeled == region.label
        features = extract_component_features(image_bgr, component_mask, region, context)
        x = features_to_array([features], feature_names)
        crack_probability = float(model.predict_proba(x)[0, 1])
        keep = crack_probability >= threshold
        if keep:
            kept_mask |= component_mask

        minr, minc, maxr, maxc = region.bbox
        component_results.append(
            {
                "component_label": int(region.label),
                "probability": crack_probability,
                "kept": bool(keep),
                "area": int(region.area),
                "bbox": [int(minr), int(minc), int(maxr), int(maxc)],
            }
        )

    return kept_mask, component_results
